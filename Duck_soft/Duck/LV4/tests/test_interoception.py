"""M2 内感受 + 自模型 + allostatic 精度驱动的单元测试。

验证三件事：

1. ``softmax_log_pref`` / ``normalize_prob`` 把 log-偏好 / 权重正确归一化为 setpoint；
2. ``Interoception.update`` 的 allostatic ``gamma`` 随内部状态偏离 setpoint 单调上升
   （越饿策略越确定），回到稳态时退回 ``base_gamma``；
3. 端到端：只靠偏好（无任何 if 规则）两个内部因子（能量 + 饱足）在各自补给格间穿梭，
   维持阈值不被耗尽，且 allostatic 驱动确实被激活。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aif import AgentSpec, SelfAwareAgent, obs_to_list
from aif.examples import gridworld as gw
from aif.self.interoception import Interoception, normalize_prob, softmax_log_pref


def test_softmax_log_pref_peaks_at_most_preferred():
    c = jnp.array([-4.0, -1.0, 0.0])
    setpoint = softmax_log_pref(c)
    assert setpoint.shape == (3,)
    assert np.argmax(setpoint) == 2  # 0 是最高偏好
    assert float(setpoint.sum()) == pytest.approx(1.0, abs=1e-4)
    # 单调：偏好越高，概率越大
    assert float(setpoint[0]) < float(setpoint[1]) < float(setpoint[2])


def test_normalize_prob_handles_batch_and_singleton_dims():
    p = jnp.array([[1.0, 3.0], [1.0, 1.0]])  # 每行和为 4 / 2
    n = normalize_prob(p)
    assert n.shape == (4,)
    assert float(n.sum()) == pytest.approx(1.0, abs=1e-4)
    # 相对比例保持（扁平化后 1,3,1,1 → 归一化）
    assert float(n[0]) == pytest.approx(1 / 6, abs=1e-4)
    assert float(n[1]) == pytest.approx(3 / 6, abs=1e-4)


def _one_factor_spec() -> AgentSpec:
    return AgentSpec(
        num_obs=[3],
        num_states=[3],
        num_controls=[1],
        internal_factors=[0],
        intero_modalities=[0],
    )


def _posterior(one_hot: list[float]) -> list[jnp.ndarray]:
    # Interoception 期望每因子后验形如 (batch, time, num_states)。
    return [jnp.asarray([[one_hot]], dtype=jnp.float32)]


def test_allostatic_gamma_rises_with_deficit():
    spec = _one_factor_spec()
    setpoint = softmax_log_pref(jnp.array([-4.0, -1.0, 0.0]))  # 峰值在状态 2
    intero = Interoception(spec, setpoints={0: setpoint}, base_gamma=1.0, k_allo=4.0)

    g_full = intero.update(_posterior([0.0, 0.0, 1.0]))  # 满足（状态 2）
    g_mid = intero.update(_posterior([0.0, 1.0, 0.0]))   # 中度偏离
    g_dep = intero.update(_posterior([1.0, 0.0, 0.0]))   # 完全偏离

    assert g_full == pytest.approx(1.0, abs=1e-4)  # 稳态 → base_gamma
    assert g_full < g_mid < g_dep
    assert g_dep > g_full  # allostatic 驱动确实放大精度


def test_allostatic_gamma_clipped_at_max():
    spec = _one_factor_spec()
    setpoint = softmax_log_pref(jnp.array([-4.0, -1.0, 0.0]))
    intero = Interoception(
        spec, setpoints={0: setpoint}, base_gamma=1.0, k_allo=100.0, gamma_max=16.0
    )
    g = intero.update(_posterior([1.0, 0.0, 0.0]))
    assert g == pytest.approx(16.0, abs=1e-4)


def test_end_to_end_foraging_maintains_homeostasis():
    """端到端：两个内部因子只靠偏好维持，能量/饱足都不被耗尽，且穿梭于两补给格。"""
    key = jax.random.PRNGKey(0)
    spec = gw.make_spec()
    model = gw.build_model()
    agent = SelfAwareAgent(spec, model)
    env = gw.GridWorld(seed=0)

    key, k = jax.random.split(key)
    obs = env.reset(k)

    min_energy = 999
    min_satiety = 999
    visited_home = False
    visited_goal = False
    gamma_activated = False

    for _ in range(60):
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)
        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)

        min_energy = min(min_energy, env.energy)
        min_satiety = min(min_satiety, env.satiety)
        visited_home = visited_home or env.pos[0] == gw.HOME
        visited_goal = visited_goal or env.pos[0] == gw.GOAL
        gamma_activated = gamma_activated or result.self_state["gamma"] > 1.5

    # 两个内部因子都曾被补给（穿梭），阈值未被耗尽（最低不低于 1，即没到 0）。
    assert visited_home and visited_goal
    assert min_energy >= 1
    assert min_satiety >= 1
    # allostatic 精度驱动确实被激活（越偏离稳态 gamma 越高）。
    assert gamma_activated
