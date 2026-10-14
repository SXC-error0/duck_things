"""宠物鸭子场景（``aif.examples.duck_pet``）的端到端测试。

验证：生成模型/环境构造正确、精力/兴致双稳态动力学正确、一次短运行产出「溜达-休息」
循环（鸭子往返 HOME 与探索点）、M3 降权幻影传感器、M4 收敛到「探索点好玩」。
只断言结构性不变量，避免示例参数微调时测试脆断。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.nn import one_hot

from aif import SelfAwareAgent
from aif.examples import duck_pet as dp
from aif.self.structure import StructureLearning


def test_model_and_spec_shapes():
    spec = dp._make_spec()
    model = dp._build_model()

    n, ne, nf = dp.N_CELLS, dp.N_ENERGY, dp.N_FUN
    assert list(spec.num_obs) == [n, ne, nf, nf]
    assert list(spec.num_states) == [n, ne, nf]
    assert list(spec.num_controls) == [4, 1, 1]
    assert spec.internal_factors == [dp.ENERGY_FACTOR, dp.FUN_FACTOR]
    assert spec.intero_modalities == [1, 2]
    assert all(d == [0, 1, 2] for d in model.A_dependencies)
    assert all(f in d for f, d in enumerate(model.B_dependencies))
    assert model.C[0].ndim == 1
    assert model.D[0].ndim == 1


def test_energy_recharges_at_home():
    env = dp.PetArena(seed=0, boring=False)
    env.r, env.c = 1, 0
    env.energy = 0
    env.step(jax.random.PRNGKey(0), jnp.asarray([dp.UP, 0, 0]))  # (1,0)→(0,0)=HOME
    assert env.energy == dp.N_ENERGY - 1


def test_fun_recharges_at_explore():
    # 探索点 cell2 = (0,2)。从 (1,2) 上移进入。
    env = dp.PetArena(seed=0, boring=False)
    env.r, env.c = 1, 2
    env.fun = 0
    env.step(jax.random.PRNGKey(0), jnp.asarray([dp.UP, 0, 0]))
    assert env.fun == dp.N_FUN - 1


def test_fun_B_candidates_differ():
    model = dp._build_model()
    B_fun, B_boring = dp._candidate_models(model)
    f = np.asarray(B_fun)
    b = np.asarray(B_boring)
    assert f.shape == (dp.N_FUN, dp.N_CELLS, dp.N_FUN, 1)
    # fun 候选：探索点以高概率回满。
    assert f[dp.N_FUN - 1, dp.EXPLORE[0], 0, 0] == pytest.approx(dp.P_FUN_REAL)
    # boring 候选：探索点以高概率照常 -1。
    assert b[0, dp.EXPLORE[0], 0, 0] == pytest.approx(1.0 - dp.P_FUN_BORING)
    assert 0.0 < dp.P_FUN_BORING < dp.P_FUN_MAIN < dp.P_FUN_REAL < 1.0


def test_run_episode_wanders_and_rests():
    spec = dp._make_spec()
    model = dp._build_model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dp._candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    env = dp.PetArena(seed=0, boring=False)
    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    obs = env.reset(k)

    visited_home, visited_explore = False, False
    gammas, rel_flaky, p_fun = [], [], []
    for _ in range(80):
        key, k = jax.random.split(key)
        result = agent.step(dp.obs_to_list(obs, env.modality_names), k)
        pos = dp._cell(env.r, env.c)
        if pos == dp.HOME:
            visited_home = True
        if pos in dp.EXPLORE:
            visited_explore = True
        gammas.append(float(result.self_state["gamma"]))
        rel_flaky.append(float(result.precision["modality_reliability"][3]))
        p_fun.append(float(agent.structure.posterior.get("fun", 0.0)))
        key, k = jax.random.split(key)
        obs, _ = env.step(k, result.action)

    # 核心：鸭子确实既回家休息过、又去探索过（溜达-休息循环）。
    assert visited_home and visited_explore
    # M3：幻影传感器被降权。
    assert np.mean(rel_flaky) < 0.2
    # M4：后验偏向「探索点好玩」。
    assert p_fun[-1] > 0.9


def test_m4_converges_to_boring():
    model = dp._build_model()
    spec = dp._make_spec()
    B_fun, B_boring = dp._candidate_models(model)

    sl = StructureLearning(
        spec,
        A=list(model.A),
        A_dependencies=[list(d) for d in model.A_dependencies],
        B_dependencies=[list(d) for d in model.B_dependencies],
    )
    sl.add_model("fun", [model.B[0], model.B[1], B_fun])
    sl.add_model("boring", [model.B[0], model.B[1], B_boring])

    n = dp.N_CELLS
    # 真动力学 = boring：探索点兴致仍每步 -1。
    for _ in range(6):
        prev_qs = [one_hot(dp.EXPLORE[0], n), one_hot(dp.N_ENERGY - 1, dp.N_ENERGY), one_hot(2, dp.N_FUN)]
        qs = [one_hot(dp.EXPLORE[0], n), one_hot(dp.N_ENERGY - 2, dp.N_ENERGY), one_hot(1, dp.N_FUN)]
        obs = [jnp.array(dp.EXPLORE[0]), jnp.array(dp.N_ENERGY - 2), jnp.array(1), jnp.array(1)]
        sl.update(qs, obs, prev_qs, [dp.DOWN, 0, 0])

    assert sl.posterior["boring"] > 0.99
    assert sl.posterior["fun"] < 0.01
