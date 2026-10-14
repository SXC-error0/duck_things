"""元认知 / 精度估计（M3）。

对「自身信念的置信度」做二阶推断，输出两个动态精度，闭环回填进感知与策略选择：

- **逐模态可靠性 ``modality_reliability[m]``**：追踪各模态对观测的期望对数似然
  （accuracy = E_{q(s)}[ln A_m[o_m | s]]），持续高惊奇（低 accuracy）的模态被降权——
  学会「这个传感器现在不可靠」（泛化鸭子代码里的自适应 ``_we`` 与 beacon_conf）。
  该权重作为**精度**乘到该模态的对数似然上（``w_m * ln A_m``），等价于对模态加精；
  ``SelfAwareAgent`` 用它在 ``infer_states`` 里做加权感知。
- **策略精度 ``policy_precision``（β）**：EFE 地形平坦（策略后验接近均匀、决策置信度
  低）时把策略反温度压得更低——「我不确定该干嘛 → 多探索一点」的元认知信号。它作为
  乘子作用到 allostatic 精度（M2）上：``gamma_eff = gamma_allo * β``。

两者都是**逐步 EMA**（在线递归，跟经验先验同一模式）：用上一步估计出的精度做本步
感知/决策，再用本步观测更新精度供下一步使用，避免鸡生蛋。

pymdp 的 ``gamma``（策略精度）构造期静态；要逐步动态，须直接调
``control.update_posterior_policies(gamma=…)``（见 ``docs/pymdp_api_notes.md`` §6），
本框架在 ``SelfAwareAgent._infer_policies`` 里已按此复刻。``update`` 在每步感知后
调用，维护并返回 ``self.precision``。
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
from pymdp.maths import compute_log_likelihood_single_modality, factor_dot

from ..interfaces import AgentSpec

_EPS = 1e-12


class Metacognition:
    """二阶精度估计：逐模态可靠性 + 策略精度。

    Parameters
    ----------
    spec:
        生成模型声明（``num_obs`` 决定模态数）。
    lr:
        accuracy / 置信度 EMA 的学习率（0~1，越大越灵敏）。
    reliability_min:
        模态可靠性的下界，避免某个模态被彻底置零后无法恢复。
    beta_floor:
        策略精度 β 的下界（低置信度时 gamma 的倍率底数，<1 即放大探索）。
    """

    def __init__(
        self,
        spec: AgentSpec,
        lr: float = 0.5,
        reliability_min: float = 1e-2,
        beta_floor: float = 0.5,
    ):
        self.spec = spec
        self.lr = lr
        self.reliability_min = reliability_min
        self.beta_floor = beta_floor
        self.modality_reliability = [1.0] * spec.num_modalities
        self.policy_precision = 1.0
        self._accuracy_ema = [0.0] * spec.num_modalities
        self._confidence_ema = 1.0
        self.precision: dict = {}

    def update(
        self,
        qs: Sequence[jnp.ndarray],
        obs: Sequence[jnp.ndarray],
        A: Sequence[jnp.ndarray],
        A_dependencies: Sequence[Sequence[int]],
        neg_efe: jnp.ndarray | None = None,
    ) -> None:
        """感知后更新逐模态可靠性 + 策略精度。

        Parameters
        ----------
        qs:
            当前状态边际后验，每因子 ``(num_states_f,)``（已剥离 batch/time 维）。
        obs:
            本步观测（离散下标，按模态顺序）。用 ``spec.num_obs`` 转 one-hot。
        A / A_dependencies:
            似然张量与其稀疏依赖，用于计算逐模态 accuracy。
        neg_efe:
            ``-期望自由能``（策略推断的中间量），用于估计决策置信度；可为 None
            （此时只更新可靠性）。
        """
        # —— 逐模态可靠性：accuracy = E_q[ln A_m[o_m | s]] ——
        for m in range(self.spec.num_modalities):
            o_m = one_hot(obs[m], self.spec.num_obs[m]).astype(jnp.float32)
            ll_m = compute_log_likelihood_single_modality(o_m, A[m], distr_obs=True)
            deps = A_dependencies[m]
            acc_m = float(factor_dot(ll_m, [qs[f] for f in deps]))
            self._accuracy_ema[m] = (1.0 - self.lr) * self._accuracy_ema[m] + self.lr * acc_m
            # accuracy ≤ 0（ln A ≤ 0），exp(accuracy) ∈ (0,1]：好 ≈ 1，坏 → 0。
            self.modality_reliability[m] = float(
                min(1.0, max(self.reliability_min, np.exp(self._accuracy_ema[m])))
            )

        # —— 策略精度：EFE 地形越平坦，β 越低（越愿意探索） ——
        if neg_efe is not None:
            neg_efe = jnp.asarray(neg_efe)
            q_pi = jnp.exp(neg_efe - neg_efe.max()) / (
                jnp.exp(neg_efe - neg_efe.max()).sum() + _EPS
            )
            confidence = float(q_pi.max())
            self._confidence_ema = (1.0 - self.lr) * self._confidence_ema + self.lr * confidence
            n_pol = neg_efe.shape[0]
            conf_norm = (self._confidence_ema - 1.0 / n_pol) / (1.0 - 1.0 / n_pol + _EPS)
            conf_norm = float(min(1.0, max(0.0, conf_norm)))
            self.policy_precision = self.beta_floor + (1.0 - self.beta_floor) * conf_norm

        self.precision = {
            "modality_reliability": dict(enumerate(self.modality_reliability)),
            "policy_precision": self.policy_precision,
        }
