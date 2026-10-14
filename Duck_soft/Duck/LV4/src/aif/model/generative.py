"""生成模型的容器：A（似然）/ B（转移）/ C（偏好）/ D（初始先验）+ 依赖。

pymdp 的生成模型由四件套构成：

- ``A[m]``：似然 P(o_m | s)，shape ``(num_obs_m, *num_states[A_deps[m]])``；
- ``B[f]``：转移 P(s'_f | s_f, u)，shape
  ``(num_states_f, *num_states[B_deps[f]], num_controls_f)``；
- ``C[m]``：对观测的 log-偏好（正值=偏好，负值=惩罚），shape ``(num_obs_m,)``
  （一维，**不能**带 ``(num_obs_m,1)`` 尾单例维，见 §8e）；
- ``D[f]``：初始状态先验，shape ``(num_states_f,)``（**不带** batch 维，见 §8d）。

依赖（``A_dependencies`` / ``B_dependencies``）在稀疏模型下必须显式给出，否则
pymdp 默认「所有模态依赖所有因子」（见 ``docs/pymdp_api_notes.md`` §8b）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GenerativeModel:
    """一个完整的离散 POMDP 生成模型（按 pymdp 约定布局）。"""

    A: Sequence[jnp.ndarray]
    B: Sequence[jnp.ndarray]
    C: Sequence[jnp.ndarray]
    D: Sequence[jnp.ndarray]
    A_dependencies: Sequence[Sequence[int]]
    B_dependencies: Sequence[Sequence[int]]
