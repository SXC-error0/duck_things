# 本机 Microduck RL 训练说明

这份文档是在**本机**（RTX 4050 Laptop，6GB 显存）训练、续训、可视化、导出、发布 Microduck
双足机器人策略的操作手册。上游通用文档见仓库根目录的
[README.md](../README.md) 和 [AGENTS.md](../AGENTS.md)；这里只记录**本机特有的差异**和这次踩过的坑。

## 0. 本机环境速览（先读这个）

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop，**6GB 显存** |
| 训练规模 | **3072 envs**（README 默认 4096 会爆显存，见 §8） |
| wandb | **无 API key**，必须 `WANDB_MODE=offline`（见 §8） |
| 显示 | Wayland；MuJoCo 原生 viewer 可能黑屏，备选 viser（见 §6） |
| 包管理 | uv + `.venv`（已同步完成） |

两条铁律，记不住别的也要记住这两条：

1. 训练命令永远带 `WANDB_MODE=offline`，否则 wandb 登录直接报错退出。
2. 训练命令永远用 `--env.scene.num-envs 3072`，4096 在这张 6GB 卡上会 OOM。

## 1. 环境准备

```bash
cd ~/dev/Duck/microduck_rl
uv sync                          # 首次或依赖变更后；本机 x86_64 无需特殊处理
uv run list-envs                 # 验证环境正常，列出全部任务
```

> ARM 盒子（DGX Spark / Jetson）首次 `uv sync` 会拉 ~2GB CUDA wheel，需先
> `export UV_HTTP_TIMEOUT=600`。本机 x86_64 不涉及。

## 2. 训练

**先跑 smoke test**（64 envs、5 轮，几秒钟，能拦住 ~95% 的配置错误）：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

**正式训练**（后台运行，日志落盘）：

```bash
cd ~/dev/Duck/microduck_rl
WANDB_MODE=offline nohup uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 3072 \
  > train.log 2>&1 &
```

主要任务（`uv run list-envs` 可看全表）：

| 任务 | 说明 |
|---|---|
| `Mjlab-Velocity-Flat-MicroDuck` | **主任务**：速度指令 + 头部姿态控制行走 |
| `Mjlab-VelStand-Flat-MicroDuck` | 行走 + 摔倒恢复 |
| `Mjlab-StandUp-Flat-MicroDuck` | 从趴/仰/坐起身并保持站立 |
| `Mjlab-SitStand-Flat-MicroDuck` | 指令式坐 ↔ 站 |
| `Mjlab-Roulade-Flat-MicroDuck` | 前滚翻 |

## 3. 断点续训

checkpoint 位置约定：

```
logs/rsl_rl/<experiment_name>/<run_dir>/model_<N>.pt
```

其中 `experiment_name` 对 Velocity 是 `velocity`，`run_dir` 形如
`2026-09-18_15-29-05_resume-750`（时间戳 + run_name）。每 `save_interval=250` iter 存一份。

续训命令（`load-run` 换成**新 run 目录**、`load-checkpoint` 换成具体文件）：

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 3072 \
  --agent.resume True \
  --agent.load-run "2026-09-18_15-29-05_resume-750" \
  --agent.load-checkpoint "model_7500.pt" \
  --agent.run-name resume-7500
```

注意点：

- `--agent.resume True` 会**恢复迭代计数**（继续往后跑）。这是同任务续训，正是想要的。
- 跨任务「热启动」不同：需设 `MICRODUCK_WARM_START=1` 并折叠源任务课程到最终阶段，详见
  AGENTS.md 的「Warm start ≠ resume」一节，这里不展开。
- 不传 `load-run` / `load-checkpoint` 时，默认按字母序取最新 run / 最新 checkpoint。

## 4. 看训练进度

实时日志：

```bash
tail -f train.log
```

每个 iteration 会打印一屏指标。判断训练是否健康，盯这几个（`Episode_Reward/<term>` 记录的是
**加权后**的值，term 权重为 0 时恒为 0，要结合权重课程解读）：

| 指标 | 健康信号 |
|---|---|
| Mean reward | 单调上升 |
| Mean episode length | 接近满长度（~1000），不倒 |
| `Episode_Termination/fell_over` | 下降 |
| `Episode_Reward/air_time` | 正且增长 → 真的在迈步，不是站着混奖励 |
| `Episode_Reward/upright` | 接近 2 |
| `Metrics/twist/error_vel_xy` / `error_vel_yaw` | 课程爬坡期会先涨后回落，属正常 |
| 每个 `Episode_Reward/<penalty>` | **必须 ≤ 0**（否则是奖励符号写反了，策略会反向利用） |

预算参考：**可用走行步态约 4000–6000 iter** 出现（3072 envs 下约 3–5 小时）；`max_iterations=50000`
只是上限不是目标，看到步态像样就可以停。

## 5. 可视化（本地 checkpoint，不需要 wandb）

`play` 支持直接用本地 checkpoint（`--checkpoint-file`），`--device cpu` 避免和训练抢 GPU：

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/2026-09-18_15-29-05_resume-750/model_7500.pt \
  --device cpu \
  --viewer native
```

- **native**：桌面原生窗口。Wayland 下可能黑屏（`OpenGL error 0x502`，见 §8）。
- **viser**：网页 viewer，渲染更稳。黑屏时改用 `--viewer viser`，按终端打印的地址在浏览器打开。

原生 viewer 不支持 checkpoint 热切换，看新权重就重新开一次。

## 6. 导出 ONNX

**必须走 `scripts/export.py`**：它会把 observation normalizer bake 进 ONNX 图
（`actor(normalizer(obs))`）。手转 checkpoint 会让机器人运行时看到未归一化的观测，不要这么干。

本地 checkpoint 导出（`--checkpoint-file`）：

```bash
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/2026-09-18_15-29-05_resume-750/model_7500.pt \
  --onnx-file output.onnx
```

导出结果应满足 `[1,61] -> [1,14]`（61 维观测 → 14 维动作），51 维的旧策略会被拒绝。

## 7. 发布到 Hugging Face Hub

前置：HF Hub 登录（`hf auth login` 或设置 `HF_TOKEN` 环境变量）。

**从 checkpoint 发布**（内部会先走一次 §6 的安全导出再上传）：

```bash
uv run publish --task Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/2026-09-18_15-29-05_resume-750/model_7500.pt \
  --repo <你的用户名>/microduck-my-walk \
  --kind perpetual --slot walk
```

**从已导出的 ONNX 发布**（直接上传，不重新导出）：

```bash
uv run publish --onnx output.onnx \
  --repo <你的用户名>/microduck-my-walk \
  --kind perpetual --slot walk
```

`--kind` 与配套参数：

- **episodic**：跑 `--duration-s` 秒后自行回到站立（踢球、前滚翻等）。`--chain` 表示按住按钮可连做。
- **perpetual**：持续到被叫停。两种形态：
  - **步态**（走行/站立）：`--slot walk`（或 `stand`），不加 `unwind-s`。
  - **保持姿态**（如单脚 flamingo）：给 `--unwind-s`，指回退前由 daemon 驱动 `idle` 的秒数。

发布前会自动校验 ONNX 形状、跑通 plausible 输入、拒绝 NaN/常量输出、拒绝覆盖已有
`policy.onnx`（除非 `--force`）。建议先 `--dry-run` 看要上传什么。

安装到真机（在机器人上）：

```bash
sudo robotctl policy add my-walk <用户名>/microduck-my-walk
sudo robotctl policy load walk <用户名>/microduck-my-walk
```

## 8. 本机踩过的坑（快速排障）

| 现象 | 原因 | 解决 |
|---|---|---|
| 训练启动即报 `wandb.errors.UsageError: No API key configured` | 本机没配 wandb key | 命令加 `WANDB_MODE=offline` |
| 4096 envs 训练 OOM / 跑不动 | 6GB 显存不够 | 用 `--env.scene.num-envs 3072`（实测 ~4.7GB） |
| `play --viewer native` 窗口黑屏 | Wayland + GLFW OpenGL 0x502 | 改用 `--viewer viser`（网页） |
| `uv run play` 不带 `--checkpoint-file` 报要 `wandb_run_path` | 默认从 wandb 拉权重 | 用 `--checkpoint-file <本地 .pt>` |
| 训练时 `play` 抢 GPU | play 默认 `cuda:0` | 加 `--device cpu` |
| 指标里某个 penalty 是正数 | 奖励符号反了（双重取负变奖励） | 检查该项权重符号，见 AGENTS.md 奖励设计 |

## 附：本次训练的完整命令链（2026-09-18，Velocity）

```bash
# 续训（从 model_750 起步）
WANDB_MODE=offline nohup uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 3072 --agent.resume True \
  --agent.load-run "2026-09-17_22-54-03_velocity" \
  --agent.load-checkpoint "model_750.pt" \
  --agent.run-name resume-750 > train_resume.log 2>&1 &

# 观察（CPU 渲染，不抢 GPU）
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/velocity/2026-09-18_15-29-05_resume-750/model_7500.pt \
  --device cpu --viewer native

# 停训练
kill <train_pid>
```
