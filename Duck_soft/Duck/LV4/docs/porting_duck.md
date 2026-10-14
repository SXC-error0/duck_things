# 迁移对照：从 `duck_life_sim.py` 到 `aif` 框架

`duck_life_sim.py`（根工程 `LV4 - 副本/`，1363 行）是一份跑得很漂亮的主动推断鸭子仿真，
但它把「域相关物理 / 传感器 / 电池」和「AIF 决策核心」焊死，靠大量手工阈值 hack 才让
行为涌现。本文把鸭子代码里那些**具体魔法**逐一对应到 `aif` 框架的**通用机制**，作为
「任何场景怎么接入」的教科书式例子。

> 原则：**提炼机制，不照搬阈值**。鸭子里的每个 `if soc < THIRST` 都对应框架里一个
> 概率性的、可微调的、可测试的构件。

## 一张总表

| 鸭子代码里的东西 | 它其实在做什么 | `aif` 里的对应物 |
|------------------|----------------|------------------|
| `THIRST` 门控（`soc < THIRST → 激活回家`） | 内稳态阈值触发的目标切换 | `Interoception` 的 allostatic 精度 `gamma`（偏离 setpoint → 策略变确定，无 if） |
| `homeo` 稳态项（`((soc−μ)/σ)²`、非对称 ×1.5、死亡罚） | 把「生理偏离」编码进偏好 | 内部因子的 log-偏好 `C[m]` + 转移 `B[f]`（耗散/补给） |
| 自适应 `_we`（`comfort + 5·hunger`） | 认知项权重随生理状态软调制 | 策略精度 `β`（M3）× allostatic `γ`（M2），`gamma_eff = γ·β` |
| `epistemic_gain` 软遮挡射线（`(1−m)` 穿透不确定墙） | 对不确定障碍保持好奇、主动验证 | 认知项 + **结构学习**：不确定障碍 = 候选结构（静态 vs 可动） |
| `HabitLayer`（习惯 = 精度加权的策略先验） | 已兑现的动作获得先验偏置 | M3 元认知层的策略先验精度（可扩展的 habit 槽位） |
| `StructLearner`（LLR/sigmoid slot） | 静态 vs 可动的贝叶斯模型比较 | `StructureLearning`（log-evidence 累积 + softmax 后验） |
| `beacon_conf`（信标置信度 + 遗忘衰减） | 一个传感器时好时坏 | `Metacognition.modality_reliability`（逐模态 EMA） |

## 1. THIRST 门控 → 内感受 allostatic 精度

鸭子（硬规则）：

```python
if soc < c["THIRST"]:
    self._tgt = self.dock_pos          # 低于阈值才激活回家目标
```

框架（概率涌现）：把「电量」声明为**内部因子**，把「电量观测」声明为**内感受模态**，
其 log-偏好 `C` 编码稳态 setpoint。电量的转移 `B` 写「每步 −1，站充电坞 → 充满」。
`Interoception` 每步计算后验与 setpoint 的偏离度，映射成 allostatic 精度：

```python
spec = AgentSpec(
    num_obs=[N_POS, N_ENERGY], num_states=[N_POS, N_ENERGY], num_controls=[4, 1],
    internal_factors=[1], intero_modalities=[1],
)
```

越饿，`gamma` 越高，策略越确定地奔向充电坞——**没有一行 `if`**。「饿」是后验偏离
setpoint 的连续量，不是阈值。

## 2. homeo 稳态项 → 偏好 C + 转移 B

鸭子把生理项写成数值惩罚（非对称、死亡罚）。框架把它拆成**两件正交的事**：

- **转移 `B[f]`**：能量如何耗散与补给（站 HOME 格充满，否则 −1）。这是世界的物理，
  不是偏好。
- **偏好 `C[m]`**：`[-6, -3, -1.5, -0.5, 0]` 这种单调 log-偏好，softmax 后就是 setpoint。
  偏离 setpoint 越远，EFE 的 pragmatic 项越负——「饿」自动变成负效用。

死亡罚（`homeo += 1e3*(0.10−soc2)`）对应「能量归零」状态在 `C` 里给一个极负偏好，
以及 `B` 里「能量 0 是吸收态」。

## 3. 自适应 `_we` → 元认知策略精度 β

鸭子的 `_we = W_EPI·(comfort + 5·hunger)` 是「认知项权重」随生理与探索状态软调制。
框架把它拆成两个**可分别测试**的信号：

- **策略精度 `β`（M3）**：EFE 地形平坦（`softmax(−EFE)` 的最大值低）→ 决策置信度低 →
  `β` 压低 → 更愿意探索。对应鸭子「comfort 高时探索欲望强」。
- **allostatic `γ`（M2）**：饿 → `γ` 高 → 策略确定（回家）。对应鸭子「hunger 驱动觅食」。

两者相乘 `gamma_eff = γ·β`，就是鸭子 `_we` 的通用版——但这里每个分量都可独立
unit-test，且不再有 `5.0`、`0.25` 这样的魔数散落。

## 4. 软遮挡射线 → 认知项 + 结构学习

鸭子 `epistemic_gain` 的精髓：对「0.6~0.9 不确定的墙」，想象射线以 `(1−m)` 概率穿透，
于是**模糊边界保持好奇、主动去验证**；`>0.9` 确认墙才截断。

在框架里这对应两层：

1. **认知项**（pymdp 的 `use_states_info_gain`）给出「去观察能降多少不确定」；
2. **结构学习**：一个「可能是墙、可能是可动箱子」的东西，本质是**两个候选模型**——
   `B_static`（撞不动）vs `B_movable`（推得动）。`StructureLearning` 对它们累积
   log-evidence，维持后验。不确定 → 后验接近 0.5 → 后验熵高 → 认知项自动奖励
   「去推一下验证」，这正是鸭子 `StructLearner` 用 `−W_EXP·H(q)` 驱动主动实验的通用版。

## 5. HabitLayer → M3 策略先验精度（扩展槽位）

鸭子的习惯层：缓存动作按精度拿 EFE bonus，兑现加固、失信重罚，静止动作不 habitualize。

框架把它归入**精度估计**的范畴：习惯本质是「策略先验 + 精度」，与 M3 的策略精度 `β`
同族。当前版本未内置 habit 槽位，但扩展点清晰——在 `Metacognition` 上维护一个
「状态 → 策略先验精度」的表，用同样的 EMA 兑现/失信更新即可。这也是 `structure.py`
留「模型平均 / 结构切换」扩展点的原因：两者都是「用经验更新二阶估计」。

## 6. StructLearner → StructureLearning

鸭子（LLR/sigmoid slot）：

```python
slot["llr"] += cap * tanh((disp - 0.03) / 0.02)   # 推动了 → 可动；纹丝不动 → 静态
slot["q"] = sigmoid(slot["llr"])
```

框架（统一成 log-evidence）：

```python
from aif.self.structure import StructureLearning

sl = StructureLearning(spec, A=..., A_dependencies=..., B_dependencies=...)
sl.add_model("static", B_static)     # 转移：撞上原地
sl.add_model("movable", B_movable)   # 转移：被推一格
# 每步：用上一步后验/动作经各 B 转移成先验，算 VFE 累积 evidence
sl.update(qs, obs, prev_qs, u_t)
sl.posterior   # {"static": 0.97, "movable": 0.03}
```

鸭子用**有界 tanh** 防单个证据冲爆；框架用**数值稳定的 softmax**（先减 max 再 exp），
同一目的。鸭子的「碰撞簇 → 开槽」聚类逻辑属于**槽位管理**（哪个物体验哪个），框架
`add_model` 是显式的——接入方决定何时注册候选结构。

## 7. beacon_conf → 逐模态可靠性

鸭子 `beacon_conf` 随遗忘衰减、随观测更新，`>0.5` 才采信信标三角化。框架的
`Metacognition.modality_reliability[m]` 是它的通用版：对每个模态做
`exp(EMA[E_q ln A_m])`，持续报错（低 accuracy）的模态自动降权。二者同一机制，只是
鸭子只对「信标」这一个传感器做了，框架对**每个模态**都做。

## 接入一个新场景的 checklist

1. **声明维度**：`AgentSpec(num_obs, num_states, num_controls, internal_factors,
   intero_modalities, policy_len)`。哪些状态因子 / 观测模态是「自我」？
2. **写生成模型**：`GenerativeModel(A, B, C, D, A_dependencies, B_dependencies)`。
   注意 C 一维、D 无 batch、A 全依赖、B 依赖含自身（见 `pymdp_api_notes.md`）。
3. **实现环境**：继承 `Environment`，`reset(key)` / `step(key, action)` 返回
   `{模态名: 整数下标数组}` 字典。
4. **（可选）稳态**：给内部因子传 `setpoints={f: setpoint}`，或让它从 `C` 自动推导。
5. **（可选）结构**：`agent.structure.add_model(name, B)` 注册候选结构。
6. **跑**：`agent.step(obs_list, rng_key)`，读 `result.self_state` / `result.precision` /
   `agent.structure.posterior` 诊断三层自我的活动。
