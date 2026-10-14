"""鸭子生活场景（``aif.examples.duck_life``）的端到端测试。

验证：生成模型/环境构造正确、球候选结构（fun vs boring）有意义的差异、环境动力学
（移动/耗电/耗玩心/双稳态回充）正确、一次短运行产出完整的诊断量、M4 结构学习收敛到
真实球结构。只断言结构性不变量，避免示例参数微调时测试脆断。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.nn import one_hot

from aif import SelfAwareAgent
from aif.examples import duck_life as dl
from aif.self.structure import StructureLearning


def _model():
    return dl._build_model()


def _spec():
    return dl._make_spec()


def test_model_and_spec_shapes():
    spec = _spec()
    model = _model()

    n, nb, np_ = dl.N_CELLS, dl.N_BATT, dl.N_PLAY
    assert list(spec.num_obs) == [n, nb, np_, n]
    assert list(spec.num_states) == [n, nb, np_]
    assert list(spec.num_controls) == [4, 1, 1]
    assert spec.internal_factors == [dl.BATT_FACTOR, dl.PLAY_FACTOR]
    assert spec.intero_modalities == [1, 2]

    # A：4 模态，全依赖 [0,1,2]。
    assert len(model.A) == 4
    assert all(d == [0, 1, 2] for d in model.A_dependencies)
    # B：3 因子，依赖含自身。
    assert len(model.B) == 3
    assert all(f in d for f, d in enumerate(model.B_dependencies))
    # C 一维 (num_obs,)、D 不带 batch 维。
    assert model.C[0].ndim == 1
    assert model.D[0].ndim == 1


def test_play_B_candidates_differ():
    model = _model()
    B_fun, B_boring = dl._candidate_models(model)

    f = np.asarray(B_fun)
    b = np.asarray(B_boring)
    assert f.shape == (dl.N_PLAY, dl.N_CELLS, dl.N_PLAY, 1)
    # fun：球格以高概率回满。
    assert f[dl.N_PLAY - 1, dl.BALL, 0, 0] == pytest.approx(dl.P_RECHARGE_FUN)
    # boring：球格以高概率照常 -1（0 档仍为 0）。
    assert b[0, dl.BALL, 0, 0] == pytest.approx(1.0 - dl.P_RECHARGE_BORING)
    # 概率都严格在 (0,1)，且 fun 的回满概率远高于 boring。
    assert 0.0 < dl.P_RECHARGE_BORING < dl.P_RECHARGE_MAIN < dl.P_RECHARGE_FUN < 1.0
    # 转移矩阵列和为 1。
    assert np.allclose(f[:, dl.BALL, 0, 0].sum(), 1.0)
    assert np.allclose(b[:, dl.BALL, 0, 0].sum(), 1.0)


def test_env_step_moves_and_drains():
    env = dl.DuckArena(seed=0, boring=False)
    obs = env.reset(jax.random.PRNGKey(0))
    assert set(obs) == {"position", "battery", "play", "flaky"}

    start = (env.r, env.c)
    # 向右一步（起点 (0,1)，非边界）应移动；电量与玩心各 -1。
    obs, done = env.step(jax.random.PRNGKey(1), jnp.asarray([dl.RIGHT, 0, 0]))
    assert (env.r, env.c) == (start[0], start[1] + 1)
    assert env.battery == dl.N_BATT - 2
    assert env.play == dl.N_PLAY - 2
    assert not done


def test_env_recharge_at_dock_and_ball():
    # 充电坞回满电量：从 (1,0) 上移进入 DOCK(0,0)。
    env = dl.DuckArena(seed=0, boring=False)
    env.r, env.c = 1, 0
    env.battery = 0
    env.step(jax.random.PRNGKey(0), jnp.asarray([dl.UP, 0, 0]))
    assert env.battery == dl.N_BATT - 1

    # 球格回满玩心（fun 世界）：从 (1,0) 右移进入 BALL(1,1)。
    env = dl.DuckArena(seed=0, boring=False)
    env.r, env.c = 1, 0
    env.play = 0
    env.step(jax.random.PRNGKey(0), jnp.asarray([dl.RIGHT, 0, 0]))
    assert env.play == dl.N_PLAY - 1

    # boring 世界：球格不回满。
    env = dl.DuckArena(seed=0, boring=True)
    env.r, env.c = 1, 0
    env.play = 2
    env.step(jax.random.PRNGKey(0), jnp.asarray([dl.RIGHT, 0, 0]))
    assert env.play == 1


def test_run_episode_produces_diagnostics():
    spec = _spec()
    model = _model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dl._candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    env = dl.DuckArena(seed=0, boring=False)
    key = jax.random.PRNGKey(0)
    key, k = jax.random.split(key)
    obs = env.reset(k)

    gammas, rel_flaky, rel_pos, p_fun = [], [], [], []
    for _ in range(80):
        key, k = jax.random.split(key)
        result = agent.step(dl.obs_to_list(obs, env.modality_names), k)
        gammas.append(float(result.self_state["gamma"]))
        rel_flaky.append(float(result.precision["modality_reliability"][3]))
        rel_pos.append(float(result.precision["modality_reliability"][0]))
        p_fun.append(float(agent.structure.posterior.get("fun", 0.0)))
        key, k = jax.random.split(key)
        obs, _ = env.step(k, result.action)

    assert len(gammas) == 80
    assert all(0 <= p <= 1 for p in p_fun)
    # M3：幻影定位源（flaky）比真实里程计（position）更不被信任。
    assert np.mean(rel_flaky) < np.mean(rel_pos)


def test_m4_converges_to_fun():
    model = _model()
    spec = _spec()
    B_fun, B_boring = dl._candidate_models(model)

    sl = StructureLearning(
        spec,
        A=list(model.A),
        A_dependencies=[list(d) for d in model.A_dependencies],
        B_dependencies=[list(d) for d in model.B_dependencies],
    )
    sl.add_model("fun", [model.B[0], model.B[1], B_fun])
    sl.add_model("boring", [model.B[0], model.B[1], B_boring])
    assert sl.posterior["fun"] == pytest.approx(0.5, abs=1e-4)

    n = dl.N_CELLS
    # 真动力学 = fun：反复观察「站球格 + 玩心 0 → 回满(7)」的转移。
    for _ in range(6):
        prev_qs = [one_hot(dl.BALL, n), one_hot(dl.N_BATT - 1, dl.N_BATT), one_hot(0, dl.N_PLAY)]
        qs = [one_hot(dl.BALL, n), one_hot(dl.N_BATT - 2, dl.N_BATT), one_hot(dl.N_PLAY - 1, dl.N_PLAY)]
        obs = [jnp.array(dl.BALL), jnp.array(dl.N_BATT - 2), jnp.array(dl.N_PLAY - 1), jnp.array(dl.BALL)]
        sl.update(qs, obs, prev_qs, [dl.DOWN, 0, 0])

    assert sl.posterior["fun"] > 0.99
    assert sl.posterior["boring"] < 0.01


def test_m4_converges_to_boring():
    model = _model()
    spec = _spec()
    B_fun, B_boring = dl._candidate_models(model)

    sl = StructureLearning(
        spec,
        A=list(model.A),
        A_dependencies=[list(d) for d in model.A_dependencies],
        B_dependencies=[list(d) for d in model.B_dependencies],
    )
    sl.add_model("fun", [model.B[0], model.B[1], B_fun])
    sl.add_model("boring", [model.B[0], model.B[1], B_boring])

    n = dl.N_CELLS
    # 真动力学 = boring：站球格，玩心仍每步 -1。
    for _ in range(6):
        prev_qs = [one_hot(dl.BALL, n), one_hot(dl.N_BATT - 1, dl.N_BATT), one_hot(2, dl.N_PLAY)]
        qs = [one_hot(dl.BALL, n), one_hot(dl.N_BATT - 2, dl.N_BATT), one_hot(1, dl.N_PLAY)]
        obs = [jnp.array(dl.BALL), jnp.array(dl.N_BATT - 2), jnp.array(1), jnp.array(dl.BALL)]
        sl.update(qs, obs, prev_qs, [dl.DOWN, 0, 0])

    assert sl.posterior["boring"] > 0.99
    assert sl.posterior["fun"] < 0.01
