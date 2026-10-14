"""因子与模态的声明式描述。

核心思想：「内部（自我）」vs「外部（世界）」只是一个**标签**，不是一套机制。内部
因子与外部因子用完全相同的方式被推断；唯一区别在于：

1. 内部因子的偏好（写入 C）编码稳态 setpoint；
2. 内部因子的精度由 allostatic 驱动（见 ``self.interoception``）。

因此这里只提供轻量的声明数据结构，把「哪个因子是自我」「哪个模态是内感受」作为
显式元数据挂在模型上。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    """一个状态因子的声明。

    ``name`` 人类可读名；``num_states`` 基数；``internal`` 是否为「自我」因子；
    ``num_controls`` 作用于该因子的控制数（未控因子 = 1）。
    """

    name: str
    num_states: int
    num_controls: int = 1
    internal: bool = False


@dataclass(frozen=True)
class ModalitySpec:
    """一个观测模态的声明。

    ``name`` 人类可读名；``num_obs`` 基数；``factors`` 该模态依赖的状态因子下标；
    ``interoceptive`` 是否为内感受模态。
    """

    name: str
    num_obs: int
    factors: Sequence[int]
    interoceptive: bool = False


def summarize(factors: Sequence[FactorSpec], modalities: Sequence[ModalitySpec]):
    """由声明推导出 ``AgentSpec`` 所需的基数与下标。"""
    num_states = [f.num_states for f in factors]
    num_controls = [f.num_controls for f in factors]
    num_obs = [m.num_obs for m in modalities]
    internal_factors = [i for i, f in enumerate(factors) if f.internal]
    intero_modalities = [i for i, m in enumerate(modalities) if m.interoceptive]
    a_dependencies = [list(m.factors) for m in modalities]
    b_dependencies = [[i] for i in range(len(factors))]
    return dict(
        num_obs=num_obs,
        num_states=num_states,
        num_controls=num_controls,
        internal_factors=internal_factors,
        intero_modalities=intero_modalities,
        a_dependencies=a_dependencies,
        b_dependencies=b_dependencies,
    )
