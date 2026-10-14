"""域无关契约（Domain-agnostic contracts）。

这里把「主动推断智能体」与其嵌入的环境解耦。一个环境只暴露观测（按模态名分组的
字典）与动作推进；「内感受」不是特殊机制，只是某个模态的取值恰好关乎智能体自身
的内部状态（电池 / 能量 / 隐藏资源）而已。正是这一个标注，把一个通用 POMDP 智能体
变成「有自我」的智能体。

所有数组约定为 JAX 数组（随 pymdp 1.0.4）；与 numpy 用 ``np.asarray`` 互通。
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

# 观测 = 模态名 -> 数组。外感受 / 内感受只是不同模态名，机制相同。
#
# 取值格式（pymdp 约定）：默认每模态是**整数下标**，且带前导 batch 维（batch=1 时为
# shape ``(1,)``）；也可用 ``categorical_obs=True`` 传概率向量。见
# ``docs/pymdp_api_notes.md`` 与 pymdp ``Agent.infer_states`` 的 docstring。
Observation = Mapping[str, jnp.ndarray]


@dataclass(frozen=True)
class AgentSpec:
    """生成模型的静态描述（维度 + 哪些因子/模态是「自我」）。

    ``num_obs`` / ``num_states`` / ``num_controls`` 是 pymdp 的基数，分别按
    模态 / 状态因子 / 控制因子列出。``internal_factors`` 与 ``intero_modalities``
    标记哪些状态因子、哪些观测模态描述智能体**自身**（身体/能量/隐藏状态），而非
    外部世界。除此之外，框架不区分内外。
    """

    num_obs: Sequence[int]
    num_states: Sequence[int]
    num_controls: Sequence[int]
    internal_factors: Sequence[int] = ()
    intero_modalities: Sequence[int] = ()
    policy_len: int = 1

    @property
    def num_modalities(self) -> int:
        return len(self.num_obs)

    @property
    def num_factors(self) -> int:
        return len(self.num_states)


class Environment(abc.ABC):
    """智能体嵌入的任务 / 世界。

    契约刻意做到最小：``reset`` 返回初始观测，``step`` 用动作推进世界并返回下一个
    观测与 ``done`` 标志。观测是 ``{模态名: 数组}``，动作是每个**状态因子**一个整数
    下标构成的数组（pymdp ``sample_action`` 的返回；非控制因子恒为 0）。
    """

    # 观测模态名（按 pymdp 模态顺序排列），供观测字典 -> 数组列表转换。
    modality_names: Sequence[str]

    @abc.abstractmethod
    def reset(self, key: jax.Array) -> Observation:
        """返回初始观测（同时内部状态归位）。"""

    @abc.abstractmethod
    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[Observation, bool]:
        """推进世界一步，返回 ``(观测, done)``。"""


def obs_to_list(obs: Observation, names: Sequence[str]) -> list[jnp.ndarray]:
    """把 ``{模态名: 数组}`` 转成 pymdp 期望的有序数组列表。"""
    return [jnp.asarray(obs[n]) for n in names]
