# aif — 域无关的主动推断与决策框架（带三层「自我」）

把「主动推断」（Active Inference，自由能原理）提炼成一个**可复用、可插拔、可测试**的
决策框架，并在其上叠加三层**自我意识**能力。它不绑定任何具体场景——机器人、交易、
通用离散 POMDP 都能接入。

基于 [pymdp](https://github.com/infer-actively/pymdp) 1.0.4 扩展：感知、策略推断、
动作采样用 pymdp 的 JAX 实现；pymdp 缺失的「元认知 / 结构学习」作为扩展层在本框架实现。

> 关于「自我意识」：这里不做现象意识，而是把它落成三个**可测试**的能力层——
> ① 对自身内部状态做显式推断（内感受 + 自模型）；② 对「自身信念的置信度 / 精度」
> 做二阶推断（元认知）；③ 对自身 / 世界的模型结构做贝叶斯比较（结构学习）。

## 快速开始

```bash
# 1. 建虚拟环境并安装（含 pymdp + jax）
uv sync

# 2. 跑觅食网格演示（内感受 + allostatic 精度驱动）
.venv/Scripts/python -m aif.examples.gridworld --steps 80

# 3. 跑三层「自我」组合演示（M2 内感受 + M3 元认知 + M4 结构学习）
.venv/Scripts/python -m aif.examples.self_demo --steps 120

# 4. 跑「有自我意识的交易者」场景演示（资金稳态 + 噪声分析师信号 + 市场结构学习）
.venv/Scripts/python -m aif.examples.trader_demo --steps 120

# 5. 跑「动态闸门迷宫导航」场景演示（动态障碍 + 闸门追踪 + 可视化 + 评价指标）
.venv/Scripts/python -m aif.examples.navigation --steps 200

# 6. 跑测试 / lint
.venv/Scripts/python -m pytest
.venv/Scripts/python -m ruff check src/ tests/
```

`gridworld` 演示里，智能体**只靠偏好 C**（没有任何 `if energy < X: 回家` 的硬规则）在
两个补给格之间穿梭，维持能量 / 饱足两个内部稳态阈值不被耗尽；`allo_γ` 列显示 allostatic
精度随内部状态偏离稳态而升高（越饿策略越确定）。`self_demo` 则在此基础上叠加一个
**故意错报的坏传感器**（元认知降权）与两个候选**动作模型**（结构学习后验收敛），一次跑
通三层自我。`trader_demo` 把三层自我翻译成交易的三件真事：资金水位是「自我」稳态
（回撤越大减仓越果断）、一个随机喊涨跌的「分析师信号」被 M3 逐渐降权、趋势 vs 均值回归
的两种市场结构被 M4 后验收敛到真实的那一个。`navigation` 把三层自我放进一个**动态闸门
迷宫**：中央闸门（门开近路直通 / 门关须绕行）是动态障碍（每步按概率翻转，M4 在
「静态 vs 动态」两个候选转移模型间收敛到真实的那一个、并测每次翻转后的恢复时间）；
电量是 M2 稳态、随机信标被 M3 降权，epistemic 价值驱动持续追踪闸门状态，另带成功率/
路径长度/信念熵/轨迹平滑度/恢复时间/epistemic-pragmatic 贡献比等评价指标，并输出动画
GIF 与汇总图。

## 核心概念

一个主动推断智能体维护一个离散 POMDP 生成模型：

| 张量 | 含义 |
|------|------|
| `A[m]` | 似然 P(o_m \| s) |
| `B[f]` | 转移 P(s'_f \| s, u) |
| `C[m]` | 对观测的 log-偏好（稳态 / 目标） |
| `D[f]` | 初始状态先验 |

一步闭环（`SelfAwareAgent.step`）：

```
感知（加权）→ 推断状态（含内部）→ 三层自我 → 推断策略 → 采样动作
```

**「内部 vs 外部」只是一个标签，不是一套机制**。某个状态因子描述的是自身（能量 /
电池 / 隐藏资源），某个观测模态恰是内感受——把这两个下标标注到 `AgentSpec` 上，
一个通用 POMDP 智能体就变成「有自我」的智能体。

## 三层「自我」

| 层 | 模块 | 做什么 | 对应二阶推断 |
|----|------|--------|-------------|
| M2 内感受 | `aif.self.interoception` | 显式推断自身内部状态，把「偏离稳态」映射成 allostatic 精度 `gamma` | 对自身状态的自模型 |
| M3 元认知 | `aif.self.metacognition` | 估计逐模态可靠性 `γ_m` + 策略精度 `β`，回填感知与决策 | 对自身信念精度的估计 |
| M4 结构学习 | `aif.self.structure` | 在一组候选转移模型上累积 log-evidence，维持模型后验 | 对自身 / 世界模型结构的比较 |

三者都是**逐步在线**递归（EMA / evidence 累积），用上一步的估计驱动本步的感知 / 决策，
再用本步观测更新估计，避免「鸡生蛋」。

- **内感受**把饥饿 / 渴 / 疲劳从 `if` 阈值换成**概率涌现**：越偏离稳态，策略反温度越高
  （越直奔恢复稳态的行动）；越满足，策略越平坦（越愿意探索）。
- **元认知**让智能体「知道自己现在拿不准」：持续报错的传感器被降权；决策置信度低时
  自动加大探索。
- **结构学习**让智能体「知道自己哪个动作 / 模型还灵不灵」：动作损坏 vs 可用、障碍
  静态 vs 可动，都是不同 `B` 的贝叶斯模型比较。

## 目录

```
aif/
  src/aif/
    interfaces.py        # AgentSpec / Environment / Observation 域无关契约
    agent.py             # SelfAwareAgent：统一闭环，包装 pymdp.Agent
    model/
      generative.py      # GenerativeModel（A/B/C/D + 依赖）
      factors.py         # FactorSpec / ModalitySpec 声明 + summarize()
    self/
      interoception.py   # 内感受 + 自模型 + allostatic 精度（M2）
      metacognition.py   # 逐模态 γ + 策略 β 元认知（M3）
      structure.py       # 贝叶斯模型比较 / 结构学习（M4）
  examples/
    gridworld.py         # 觅食网格：三因子（位置/能量/饱足）端到端演示（M2）
    self_demo.py         # 三层自我组合演示：内感受 + 元认知 + 结构学习（M2+M3+M4）
    trader_demo.py       # 交易场景：资金稳态 + 噪声信号 + 市场结构学习（M2+M3+M4）
    navigation.py        # 导航场景：动态闸门 + 结构学习 + 可视化 + 指标
  tests/                 # 每层单元测试 + 端到端
  docs/
    pymdp_api_notes.md   # pymdp 1.0.4 的坑与约定（踩坑笔记）
    porting_duck.md      # 从鸭子仿真迁移的对照指南
```

## 最小用法

```python
import jax
import jax.numpy as jnp

from aif import AgentSpec, GenerativeModel, SelfAwareAgent

spec = AgentSpec(
    num_obs=[3], num_states=[3], num_controls=[4],
    internal_factors=[], intero_modalities=[],
)
model = GenerativeModel(A=..., B=..., C=..., D=..., A_dependencies=..., B_dependencies=...)
agent = SelfAwareAgent(spec, model)

key = jax.random.PRNGKey(0)
key, k = jax.random.split(key)
result = agent.step([obs_array], k)          # obs 按模态顺序
result.action                                 # 每个状态因子一个下标（非控制因子恒 0）
result.self_state["gamma"]                    # allostatic 精度（M2）
result.precision["modality_reliability"]      # 逐模态可靠性（M3）
```

接入新场景只需三步：① 用 `AgentSpec` 声明维度并标注内部因子 / 内感受模态；② 提供
`GenerativeModel`（A/B/C/D）；③ 实现 `Environment.reset` / `step` 返回观测字典。
详见 [docs/porting_duck.md](docs/porting_duck.md) 的迁移对照。

## 关键约定（必读）

- `C` 是一维 `(num_obs,)` log-偏好（**不能**带 `(num_obs,1)` 尾单例维）。
- `D` 不带 batch 维；`A` 需给出完整依赖（单因子模态的 marginal 似然在 pymdp 1.0.4 有 bug）。
- `B[f]` 依赖须显式给出 `B_dependencies[f]`，且必须包含 `f` 自身。
- 随机数显式传 `jax.random.PRNGKey`（禁用 `np.random`）。
- 更多踩坑见 [docs/pymdp_api_notes.md](docs/pymdp_api_notes.md)。

## 路线图

- [x] M0 环境就绪 + pymdp API 锁定
- [x] M1 骨架 + 最小闭环
- [x] M2 内感受 + 自模型 + allostatic 精度驱动
- [x] M3 元认知（逐模态可靠性 + 策略精度）
- [x] M4 结构学习（贝叶斯模型比较）
- [x] M5 文档 + 测试 + 组合演示
