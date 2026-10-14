# 项目交接文档（供 Codex 二次开发快速上手）

> 本文档记录「让训练好的鸭子实现自我意识」项目从零到现在的**全部架构、已完成工作、关键修复、踩坑、当前状态**。读完可快速继续开发。

---

## 1. 项目目标

把 Pollen Robotics 的 **Microduck**（~25cm、800g 双足机器人，14 个 XL330 舵机）变成一个
**有自我意识的宠物**，三个核心行为：

1. **自己溜达**（没事时自主探索）
2. **累了休息**（疲劳时回窝）
3. **知道你在喊它**（听到主人呼唤 → 走向主人）

**核心理念**：自我意识 ≠ 大模型对话。它是**分层决策 + 多层自模型**——用主动推断（Active
Inference）框架让鸭子**自主决定干什么**，不靠手写 `if` 硬规则。

---

## 2. 目录结构（三个子项目）

```
/home/zkxt/dev/Duck/
├── LV4/                 # ★ 用户自己写的 aif 框架（主动推断 + 三层自我）
│   ├── src/aif/         #   框架核心 + examples/
│   ├── tests/           #   30 个测试
│   └── docs/            #   pymdp 踩坑笔记 + 迁移指南
├── microduck-main/      # Pollen 的机器人固件（Rust daemons，非 git 仓库）
│   ├── robotd/          #   控制环 daemon（50Hz + RL 策略 + 摔倒检测）
│   ├── duck-ipc-proto/  #   IPC 协议（JSON-RPC）
│   ├── robotctl/        #   客户端
│   └── docs/            #   simulation.md / architecture.md 等
└── microduck_rl/        # Pollen 的 RL 训练仓库（MuJoCo + PPO → ONNX）
    ├── src/mjlab_microduck/  # 桥、客户端、身体 server
    ├── scripts/              # 闭环脚本、infer_policy.py
    └── .venv/                # mujoco + bam + onnxruntime（无 jax）
```

**两个 venv 的依赖割裂（重要）**：
- `LV4/.venv`：jax + pymdp + numpy（**无** mujoco）
- `microduck_rl/.venv`：mujoco + bam + onnxruntime（**无** jax）

认知层脚本用 **LV4 venv** 跑，通过 `sys.path` 把 `microduck_rl/src` 加进去取桥/客户端。

---

## 3. 总体架构（分层控制）

```
┌─ 认知层（aif，LV4，离散决策 ~1Hz）────────────┐
│  SelfAwareAgent + 三层自我                    │
│  决定「干什么」：去 HOME / 去探索点 / 去主人     │
└──────────────┬───────────────────────────────┘
               │ 整数动作
┌──────────────▼───────────────────────────────┐
│  duck_bridge（microduck_rl）                  │
│  动作 → (策略槽, twist 命令) + 朝向换算 + 弧线导航│
└──────────────┬───────────────────────────────┘
               │ twist (vx,vy,vyaw)
┌──────────────▼───────────────────────────────┐
│  robotd_client（IPC JSON-RPC）                │
│  robot.move / robot.stop / robot.do / state   │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│  robotd（Rust，50Hz 控制环）                   │
│  RL 策略（ONNX）+ 摔倒检测 + 策略热插拔          │
└──────────────┬───────────────────────────────┘
               │
┌──────────────▼───────────────────────────────┐
│  duck-body（MuJoCo，3D viewer）               │
│  真实物理身体（BAM 执行器模型）                 │
└───────────────────────────────────────────────┘
```

**一次 aif 步 = 一次「走到目标格」**（turn-then-go 或弧线导航，到达目标格即停）。

---

## 4. 核心概念

### 4.1 aif 框架（LV4，用户自己的代码）

基于 [pymdp](https://github.com/infer-actively/pymdp) 1.0.4 的主动推断框架，**三层「自我」**：

| 层 | 模块 | 做什么 |
|---|---|---|
| M2 内感受 | `self/interoception.py` | 内部状态（能量/精力/兴致）偏离稳态 → allostatic 精度 γ |
| M3 元认知 | `self/metacognition.py` | 逐模态传感器可靠性 + 策略精度 β |
| M4 结构学习 | `self/structure.py` | 候选模型（B 矩阵）的贝叶斯比较 |

**接入一个新场景只需**：`AgentSpec`（维度 + 哪些因子/模态是「自我」）+ `GenerativeModel`
（A/B/C/D）+ `Environment`（reset/step 返回观测字典）。见 `src/aif/examples/` 的 4 个例子。

**关键 API**：
```python
agent = SelfAwareAgent(spec, model)
result = agent.step(obs_list, rng_key)  # → result.action / self_state / precision
```

### 4.2 robotd 接口（IPC，newline-delimited JSON-RPC）

- 帧：换行分隔 JSON。请求带 `id`，连续意图（`robot.move`）用通知（无 id）。
- `robot.move {vx, vy, vyaw}`：速度 twist（前进/左移 m/s，左转 rad/s）。
- `robot.stop {}`、`robot.enable {on}`、`robot.do {skill}`（skill 如 `sit_toggle`）。
- `robot.subscribe {hz}` → 逐 tick 推送 `robot.state`（含 `safety.fallen`、`odom.position`、
  `imu.quat`）。
- **Deadman**：twist 意图超过 500ms 未刷新就被清零（`safety.rs`）。

### 4.3 训练好的策略（ONNX，61D obs → 14D action）

- 61D obs = 48 本体 + 13 命令 `[twist(3), head_pose(4), body_pose(6)]`。
- 策略槽：`walk` / `stand` / `sitstand` / `ground_pick` / `kick_left/right` / `roulade`。
- 官方策略集（`pollen-robotics/microduck-policies` v5）已在
  `~/.cache/duck-sim/policies/current/`（10 个 ONNX，含 `velstand`、`alpha_walking`、
  `alpha_stand`、`alpha_sitstand` 等）。

---

## 5. 已完成的工作（按时间顺序）

| 阶段 | 文件 | 成果 |
|---|---|---|
| Phase 0 | `LV4/src/aif/examples/duck_life.py` + 测试 | 离散「鸭子生活」世界，三层自我，觅食循环 |
| Phase 1 | `microduck_rl/src/mjlab_microduck/duck_bridge.py` + 测试 | 命令桥 + 朝向换算 + 弧线导航 |
| Phase 2 | `microduck_rl/scripts/duck_self_drive.py` | mock 闭环 + `--viz`/`--live` 可视化 |
| 集成 | `microduck_rl/src/mjlab_microduck/robotd_client.py` + 测试 | robotd IPC 客户端 |
| 集成 | 构建 6 个 daemon + 拉取策略集 + 起 duck-body + robotd | 完整 robotd 栈跑通 |
| 宠物 v1 | `LV4/src/aif/examples/duck_pet.py` + 测试 | 精力/兴致双稳态 →「溜达+休息」 |
| 宠物 v1 | `microduck_rl/scripts/duck_pet_self_drive.py` | mock「溜达+休息」+ 实时 2D |
| 宠物 v1 | `microduck_rl/scripts/duck_pet_robotd.py` | **真机 3D「溜达+休息」跑通** |
| 宠物 v2 | `LV4/src/aif/examples/duck_pet_call.py` | 「知道你在喊它」认知模型 |

---

## 6. 关键文件清单

### LV4（认知层）
- `src/aif/agent.py` — `SelfAwareAgent`（统一闭环）
- `src/aif/interfaces.py` — `AgentSpec` / `Environment` / `Observation`
- `src/aif/model/generative.py` — `GenerativeModel`（A/B/C/D）
- `src/aif/self/{interoception,metacognition,structure}.py` — 三层自我
- `src/aif/examples/duck_life.py` — 电量/玩心双稳态觅食（最老）
- `src/aif/examples/duck_pet.py` — **精力/兴致双稳态（溜达+休息）**
- `src/aif/examples/duck_pet_call.py` — **+ 知道你在喊它**
- `docs/pymdp_api_notes.md` — pymdp 1.0.4 的坑（必读）

### microduck_rl（桥 + 身体）
- `src/mjlab_microduck/duck_bridge.py` — `world_move_twist`（转向-再走）+ `world_move_arc`（弧线）+ 命令组装
- `src/mjlab_microduck/robotd_client.py` — `RobotdClient`（IPC）
- `scripts/duck_pet_robotd.py` — **真机 3D 宠物闭环（最完整）**
- `scripts/duck_pet_self_drive.py` — mock 宠物闭环
- `scripts/infer_policy.py` — `PolicyInference`（直接驱动 ONNX，参考）
- `src/mjlab_microduck/sim/body_server.py` — `duck-body`（MuJoCo 身体 server）

### microduck-main（固件）
- `robotd/` `robotctl/` `duck-ipc-proto/` — daemon 与协议
- `.cargo/config.toml` — rsproxy 镜像（已加，勿删）
- `docs/robot/simulation.md` — 仿真用法

---

## 7. ★ 关键修复与踩坑（最重要，别重蹈覆辙）

### 7.1 aif 认知层

1. **露营 bug**：只放一个内部稳态（如只有电量）→ 鸭子永远赖在补给点不动。**修**：放两个
   互斥稳态（能量在 A 回满、兴致在 B 回满），逼它往返。
2. **信念腐蚀 bug**：主模型 B 若硬编码 0/1，转移先验与精确观测硬冲突会腐蚀位置信念。
   **修**：B 的概率严格在 (0,1)（软先验）。
3. **D 先验不匹配**：agent 的初始先验 D 必须匹配鸭子**实际起点**，否则硬先验与实际观测
   冲突 → 走错方向。**修**：循环里 `agent._pymdp.D[0] = onehot(实际起点格)`。
4. **探索点太远**：补给点离 HOME 超过 policy_len 能看到的距离 → 露营。**修**：补给点距
   HOME ≤ 2 格（policy_len=3）。

### 7.2 真机导航

5. **velstand 转弯锁死**：velstand（稳定性优先、防摔）大角度转弯慢到 0.125°/s（小角度
   15°/s，差 100 倍），导航不可用。**修**：换 `alpha_walking`（18–27°/s，机动性好但无
   防摔）。摔倒恢复靠 `sit_toggle`（sitstand 策略）。
6. **纯原地转锁死**：vx=0 的纯转向（`world_move_arc` 背对目标时）会让 velstand/alpha_walking
   都锁死。**修**：弧线永远带前进速度 `vx = max(speed*cos(rel), 0.15)`。
7. **回充半径不匹配**：到达半径(0.2m) > 回充半径(0.15m) → 鸭子「到了」格却没近到触发
   回充。**修**：回充半径 ≥ 到达半径（用 0.35m）。
8. **身体读取要用行缓冲**：单次 `recv` 可能拿到部分 JSON。**修**：读到 `\n` 为止。
9. **状态流必须 drain**：订阅 `robot.subscribe` 后不读 → robotd IPC 线程写满缓冲阻塞 →
   deadman 清零命令。**修**：循环里每次 `poll_state` drain。

### 7.3 环境/构建

10. **本机 IPv6 不通**：直连 IPv6 超时、IPv4 正常。uv/pip/cargo 下载慢或卡死时：
    - uv/pip：`ALL_PROXY=socks5://127.0.0.1:7890`（Clash SOCKS5 端口 7890，HTTP 模式不可用）。
    - cargo：已配 `microduck-main/.cargo/config.toml` 用 rsproxy.cn 镜像（别用 socks5，会卡）。
11. **microduck-main 不是 git 仓库**：`scripts/duck-sim` 用 `git rev-parse` 会挂。**修**：
    走 by-hand（直接起 duck-body + robotd），不用 duck-sim 脚本。
12. **seed-policies.sh 超时**：HF CDN 慢（~68KB/s），脚本 `--max-time 8` 太短。**修**：手动
    curl 下载（策略已在 `~/.cache/duck-sim/policies/current/`）。
13. **真机「冻结」**：反复 kill/restart 后鸭子会冻结（heading 不动）。**修**：干净重启
    duck-body + robotd（`setsid` 启动），鸭子站起后立即测。

---

## 8. 当前状态

### ✅ 已完成并验证
- aif 框架 + 三层自我 + 4 个例子 + 30 测试全过
- 命令桥、robotd IPC 客户端
- 完整 robotd 栈（策略拉取 + daemon 构建 + duck-body + robotd）
- **真机 3D「自己溜达 + 累了休息」跑通**（alpha_walking 策略）
- **「知道你在喊它」认知模型**（duck_pet_call.py）

### ⚠️ 未完成 / 未验证
- **摔倒恢复**：`recover()` 已写（sit_toggle 两次），但**未实测**（alpha_walking 机动性好，
  没摔过）。脸朝下摔倒能否靠 alpha_sitstand 站起未知。
- **「知道你在喊它」接真机**：认知层好了，真机接线未做。
- **无限循环**：`--steps 0` 已实现，但后台跑会被系统内存保护杀掉（前台跑没问题）。

### ❌ 阻塞项
- **真机大角度转弯**（已由 alpha_walking 解决）
- **硬件**：RK3588 + 麦克风阵列（唤醒词+声纹）——「知道你在喊它」的真机感知需要它

---

## 9. 运行命令

### 认知层（LV4 venv，含 jax）
```bash
cd /home/zkxt/dev/Duck/LV4
.venv/bin/python -m aif.examples.duck_pet --steps 200 --viz     # 溜达+休息（GIF）
.venv/bin/python -m aif.examples.duck_pet --steps 200 --live    # 实时 2D 窗口
.venv/bin/python -m aif.examples.duck_pet_call --steps 120      # + 知道你在喊它
.venv/bin/python -m pytest                                       # 30 测试
```

### 真机 3D 闭环（先起运行时，再用 LV4 venv 跑认知层）
```bash
# 1. 起 duck-body（MuJoCo 3D viewer，microduck_rl venv）
cd /home/zkxt/dev/Duck/microduck_rl
setsid .venv/bin/python -m mjlab_microduck.sim.body_server --port 7801 --keyframe SIT &

# 2. 起 robotd（microduck-main）
cd /home/zkxt/dev/Duck/microduck-main
setsid env DUCK_RUNTIME_DIR=/home/zkxt/.cache/duck-sim \
  ORT_DYLIB_PATH=/home/zkxt/dev/Duck/microduck_rl/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.24.4 \
  target/debug/robotd --sim 127.0.0.1:7801 \
  --params /home/zkxt/.cache/duck-sim/robotd.toml \
  --socket /home/zkxt/.cache/duck-sim/duck-a.sock &

# 3. 等鸭子站起（~15s），跑认知层闭环（LV4 venv）
cd /home/zkxt/dev/Duck
LV4/.venv/bin/python microduck_rl/scripts/duck_pet_robotd.py \
  --socket ~/.cache/duck-sim/duck-a.sock --body 127.0.0.1:7801 --steps 0   # 0=无限
```

### 构建 daemon（首次）
```bash
cd /home/zkxt/dev/Duck/microduck-main
cargo build -p robotd -p robotctl -p tof -p sounds -p configd -p updater
```

---

## 10. 关键决策与取舍

| 决策 | 理由 |
|---|---|
| 认知层用 aif 而非大模型 | 自我意识=自主决策，不是对话；aif 轻量可跑 A76 |
| 认知层 ↔ 身体用 IPC 而非同进程 | 两个 venv 依赖割裂；robotd 自带摔倒/策略热插拔 |
| walk 用 alpha_walking 不用 velstand | 机动性（18°/s vs 0.125°/s）换防摔（靠 sitstand 兜底） |
| 真机定位用 ground-truth trunk 不用接触里程计 | 里程计漂移导致到不了精确格 |
| 导航用弧线不用转向-再走 | 慢转/纯转锁死，弧线（前进+转向）才有效 |

---

## 11. 下一步建议（供 Codex 参考）

1. **「知道你在喊它」接真机**：把 `duck_pet_call.py` 的「呼唤→走向 OWNER」接到
   `duck_pet_robotd.py`（真机上一只鸭子 + 一个 OWNER 点，脚本模拟呼唤）。
2. **实测摔倒恢复**：故意把鸭子推倒，看 `recover()`（sit_toggle）能否站起；不行就换
   velstand 或配专门的 StandUp 策略。
3. **RK3588 麦克风阵列**：唤醒词（KWS）+ 声纹（说话人识别）+ 声源定位（DOA），把模拟
   呼唤换成真实声源。核心隔离：robotd 钉 A55，aif+感知跑 A76+NPU。
4. **增强溜达自由度**：现在是两点往返，加 epistemic 驱动（探索未知区域）让它逛不同地方。

---

## 12. 记忆索引（已存到 auto-memory，可查）

- `local-training-environment.md`：RTX 4050 6GB → 训练 3072 envs + WANDB_MODE=offline
- `network-ipv6-proxy.md`：IPv6 不通 → socks5 代理 / rsproxy 镜像
- `duck-self-project.md`：本项目架构与进度的精炼版

---

*最后更新：2026-10-14*
