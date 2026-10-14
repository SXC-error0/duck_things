"""内感受 + 自模型 + allostatic 精度驱动（M2）。

智能体对**自身**内部状态（能量 / 电池 / 隐藏资源）做显式推断，并把「内部状态偏离
稳态 setpoint」转成一个**精度权重**压到策略选择上——鸭子代码里 THIRST 门控的通用版：

- 不是 ``if energy < X: 去找吃的``；
- 而是把「偏离偏好」作为 allostatic 精度（``gamma`` 的反温度倍率）作用到策略后验，
  让「饥饿」从概率上涌现：越偏离稳态，策略越确定（越直奔恢复稳态的行动）；
  越接近稳态，策略越平坦（越愿意探索 / 顾及其他目标）。

内感受不是一种特殊机制，只是某个状态因子 / 观测模态恰好描述**自身**而已（见
``interfaces.py``）。本层做的事：

1. 取出 ``spec.internal_factors`` 这些因子的后验（这就是「自模型」——对自身内部状态
   的信念）；
2. 与稳态 setpoint（目标分布，见下）比较，得「满足度 / 偏离度」；
3. 把偏离度映射成 allostatic 精度 ``gamma = base * (1 + k_allo * 偏离)``，回填进策略
   推断（``SelfAwareAgent`` 用 ``control.update_posterior_policies(gamma=…)`` 消费）。

**setpoint 约定**：它是内部状态上的目标分布（概率）。由 ``C`` 偏好推导时，``C`` 是
log-偏好，故做 ``softmax(C)``；显式传入时按概率向量线性归一化。
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from ..interfaces import AgentSpec

_EPS = 1e-12


def normalize_prob(p: jnp.ndarray) -> jnp.ndarray:
    """把（可能带单例维的）概率/权重向量线性归一化到 [0,1] 分布。"""
    p = jnp.asarray(p, dtype=jnp.float32).reshape(-1)
    return p / (p.sum() + _EPS)


def softmax_log_pref(c: jnp.ndarray) -> jnp.ndarray:
    """把 log-偏好 ``C`` 转成内部状态上的目标分布（softmax）。"""
    c = jnp.asarray(c, dtype=jnp.float32).reshape(-1)
    return jnp.exp(c - c.max()) / (jnp.exp(c - c.max()).sum() + _EPS)


class Interoception:
    """内感受层：推断自身内部状态 + 计算 allostatic 精度驱动。

    Parameters
    ----------
    spec:
        生成模型声明（哪些因子 / 模态是「自我」）。
    setpoints:
        稳态 setpoint，``{内部因子下标: 该因子各状态的稳态概率}``。缺省时由
        ``SelfAwareAgent`` 从 ``C``（log-偏好）softmax 得到。
    base_gamma:
        基础策略精度（pymdp ``gamma`` 的底数，对应 ``Agent`` 默认 1.0）。
    k_allo:
        allostatic 增益——偏离度 1.0（完全偏离稳态）时 gamma 的倍率增量。
    gamma_max:
        allostatic 精度的上限，防止极端偏离时软最大化溢出。
    """

    def __init__(
        self,
        spec: AgentSpec,
        setpoints: Mapping[int, jnp.ndarray] | None = None,
        base_gamma: float = 1.0,
        k_allo: float = 4.0,
        gamma_max: float = 16.0,
    ):
        self.spec = spec
        self.base_gamma = base_gamma
        self.k_allo = k_allo
        self.gamma_max = gamma_max
        # 稳态 setpoint：内部因子下标 -> 该因子各状态的稳态概率（归一化目标分布）。
        self.setpoints: dict[int, jnp.ndarray] = {
            f: normalize_prob(sp) for f, sp in (setpoints or {}).items()
        }
        self.state: dict = {}

    @property
    def allostatic_gamma(self) -> float:
        """最近一次 ``update`` 得到的 allostatic 策略精度。"""
        return float(self.state.get("gamma", self.base_gamma))

    def update(self, qs, info=None) -> float:
        """感知后更新自模型与 allostatic 精度，返回 ``gamma``。

        Parameters
        ----------
        qs:
            ``infer_states`` 的状态后验（每因子 ``(batch, time, num_states)``）。
        info:
            保留位（M3 元认知层用 VFE 分量；内感受层暂不需要）。

        Returns
        -------
        float
            allostatic 策略精度 ``gamma``（已按偏离度缩放、并夹到上限）。
        """
        posterior: dict[int, jnp.ndarray] = {}
        satisfaction: dict[int, float] = {}
        error: dict[int, float] = {}

        for f in self.spec.internal_factors:
            q_f = normalize_prob(jnp.asarray(qs[f])[:, -1])  # 当前时间步的边际后验
            setpoint = self.setpoints.get(f)
            if setpoint is None:
                setpoint = jnp.ones_like(q_f) / q_f.shape[0]  # 无 setpoint → 均匀
            # 满足度 = 当前后验与稳态目标分布的重叠，min-max 归一化到 [0,1]。
            sat_raw = float(jnp.dot(setpoint, q_f))
            sat = (sat_raw - float(setpoint.min())) / (
                float(setpoint.max()) - float(setpoint.min()) + _EPS
            )
            posterior[f] = q_f
            satisfaction[f] = sat
            error[f] = 1.0 - sat

        mean_error = (
            float(jnp.mean(jnp.asarray([error[f] for f in self.spec.internal_factors])))
            if self.spec.internal_factors
            else 0.0
        )
        allostatic_gain = 1.0 + self.k_allo * mean_error
        gamma = float(min(self.base_gamma * allostatic_gain, self.gamma_max))

        self.state = {
            "posterior": posterior,
            "satisfaction": satisfaction,
            "error": error,
            "allostatic_gain": allostatic_gain,
            "gamma": gamma,
        }
        return gamma
