"""结构学习 / 贝叶斯模型比较（M4）。

对「世界 / 自身模型的**结构**」做二阶推断：在一组候选生成模型上累积 log-evidence，
维持模型后验。候选模型共享似然 ``A``，只在转移结构 ``B`` 上不同——例如：

- **世界结构**：某区域 / 物体是「静态」还是「可动」（不同的 ``B``）；
- **自身结构**：某个动作当前是否可用 / 有效（对自身 action-affordance 的不同 ``B``，
  动作可用 → ``B`` 真的移动，动作损坏 → ``B`` 恒等/不动）。

evidence 用**变分自由能**（``maths.calc_vfe``）累积：每一步把上一步后验经该模型的
``B`` 转移成先验，再算当前后验在该先验下的 VFE。转移先验越贴合实际后验（即该模型的
动态越能解释观测到的转移），VFE 越低、evidence 越高。累积后 softmax 得模型后验——
泛化鸭子代码里 ``StructLearner`` 的 LLR/sigmoid slot。

用法：

    structure = StructureLearning(spec, A=A, A_dependencies=A_deps, B_dependencies=B_deps)
    structure.add_model("static", B_static)
    structure.add_model("movable", B_movable)
    structure.update(qs, obs, prev_qs, u_t)   # 每步累积 evidence
    structure.posterior  # {"static": p, "movable": 1-p}

本层是**离线模型比较**：它估计「哪个模型更可信」，但不直接改主循环的 B（模型平均 /
结构切换是后续扩展）。``SelfAwareAgent`` 在模型已注册时自动累积 evidence。
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jax.nn import one_hot
from pymdp import control, maths

from ..interfaces import AgentSpec

_EPS = 1e-12


class StructureLearning:
    """在一组候选转移模型上做贝叶斯模型比较。

    Parameters
    ----------
    spec:
        生成模型声明。
    A / A_dependencies:
        所有候选模型共享的似然张量与依赖（只比较转移结构）。
    B_dependencies:
        转移依赖（候选模型共享结构，仅 B 数值不同）。
    """

    def __init__(
        self,
        spec: AgentSpec,
        A: Sequence[jnp.ndarray],
        A_dependencies: Sequence[Sequence[int]],
        B_dependencies: Sequence[Sequence[int]],
    ):
        self.spec = spec
        self.A = list(A)
        self.A_dependencies = [list(d) for d in A_dependencies]
        self.B_dependencies = [list(d) for d in B_dependencies]
        self.models: dict[str, list[jnp.ndarray]] = {}
        self.log_evidence: dict[str, float] = {}
        self.posterior: dict[str, float] = {}

    def add_model(self, name: str, B: Sequence[jnp.ndarray]) -> None:
        """注册一个候选模型：以转移张量 ``B`` 区分。"""
        self.models[name] = list(B)
        self.log_evidence[name] = 0.0
        self._normalize_posterior()  # 注册后即可读取均匀后验

    def update(
        self,
        qs: Sequence[jnp.ndarray],
        obs: Sequence[jnp.ndarray],
        prev_qs: Sequence[jnp.ndarray],
        u_t: Sequence[int],
    ) -> dict[str, float]:
        """累积各候选模型在本步的 log-evidence，返回更新后的模型后验。

        Parameters
        ----------
        qs:
            当前状态边际后验，每因子 ``(num_states_f,)``（批量自由）。
        obs:
            本步观测（离散下标，批量自由）。
        prev_qs:
            上一步后验（用于经 ``B`` 转移成先验）。
        u_t:
            全因子动作下标（长度 = 因子数，非控制因子填 0）。
        """
        obs_vec = [
            one_hot(o, n).astype(jnp.float32)
            for o, n in zip(obs, self.spec.num_obs, strict=True)
        ]
        for name, B in self.models.items():
            prior_k = control.compute_expected_state(
                list(prev_qs), B, list(u_t), self.B_dependencies
            )
            _, vfe = maths.calc_vfe(
                list(qs),
                prior_k,
                obs=obs_vec,
                A=self.A,
                A_dependencies=self.A_dependencies,
            )
            self.log_evidence[name] += float(-vfe)  # 越高越好

        self._normalize_posterior()
        return dict(self.posterior)

    def _normalize_posterior(self) -> None:
        if not self.models:
            self.posterior = {}
            return
        ev = np.asarray([self.log_evidence[name] for name in self.models], dtype=np.float32)
        ev = ev - ev.max()  # 数值稳定
        w = np.exp(ev)
        w = w / (w.sum() + _EPS)
        self.posterior = {name: float(w[i]) for i, name in enumerate(self.models)}
