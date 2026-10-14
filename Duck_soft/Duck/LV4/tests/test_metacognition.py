"""M3 元认知 / 精度估计的单元测试。

验证两件事：

1. **逐模态可靠性**：一个观测持续与似然模型不一致（惊奇）的模态被降权——「这个传感器
   现在不可靠」；一致（accuracy 高）的模态保持被信任。
2. **策略精度 β**：EFE 地形平坦（决策置信度低）时 β 压低（放大探索），地形尖锐时 β 逼近 1。
"""

from __future__ import annotations

import jax.numpy as jnp

from aif import AgentSpec
from aif.self.metacognition import Metacognition


def _one_factor_two_modalities() -> tuple[AgentSpec, list[jnp.ndarray], list[list[int]]]:
    spec = AgentSpec(num_obs=[2, 2], num_states=[2], num_controls=[1])
    # 两个模态都用「精确」似然模型；区别在于喂进去的观测是否匹配真状态。
    A = [jnp.array([[0.9, 0.1], [0.1, 0.9]]), jnp.array([[0.9, 0.1], [0.1, 0.9]])]
    A_deps = [[0], [0]]
    return spec, A, A_deps


def test_modality_reliability_downweights_surprising_modality():
    spec, A, A_deps = _one_factor_two_modalities()
    meta = Metacognition(spec, lr=0.5)
    qs = [jnp.array([1.0, 0.0])]  # 真状态 0
    # 模态 0 观测一致（o=0），模态 1 持续错报（o=1 → A[1,0]=0.1，高惊奇）。
    for _ in range(10):
        meta.update(qs, [jnp.array(0), jnp.array(1)], A, A_deps)

    assert meta.modality_reliability[0] > 0.5   # 一致模态保持信任
    assert meta.modality_reliability[1] < 0.3   # 惊奇模态被降权
    assert meta.modality_reliability[1] < meta.modality_reliability[0]


def test_modality_reliability_recovers_when_observation_consistent():
    spec, A, A_deps = _one_factor_two_modalities()
    meta = Metacognition(spec, lr=0.5)
    qs = [jnp.array([1.0, 0.0])]
    for _ in range(10):
        meta.update(qs, [jnp.array(0), jnp.array(1)], A, A_deps)
    low = meta.modality_reliability[1]
    # 之后模态 1 开始一致观测，可靠性应回升。
    for _ in range(10):
        meta.update(qs, [jnp.array(0), jnp.array(0)], A, A_deps)
    assert meta.modality_reliability[1] > low


def test_policy_precision_lower_when_efe_flat():
    spec, A, A_deps = _one_factor_two_modalities()
    meta = Metacognition(spec, lr=1.0, beta_floor=0.5)
    qs = [jnp.array([1.0, 0.0])]
    obs = [jnp.array(0), jnp.array(0)]

    for _ in range(3):
        meta.update(qs, obs, A, A_deps, neg_efe=jnp.array([0.0, 0.0, 0.0]))
    assert meta.policy_precision < 0.6  # 平坦 → 逼近 beta_floor

    for _ in range(3):
        meta.update(qs, obs, A, A_deps, neg_efe=jnp.array([0.0, 0.0, 5.0]))
    assert meta.policy_precision > 0.9  # 尖锐 → 逼近 1.0
