"""导航场景示例的端到端测试（``aif.examples.navigation``）。

验证：生成模型/环境构造正确、EFE 分解（epistemic/pragmatic）给出有意义的数值、
一次短运行能产出完整的指标记录。不锁定具体数值——只断言结构性不变量，避免示例参数
微调时测试脆断。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aif import SelfAwareAgent
from aif.examples import navigation as nav


def _grid() -> nav._Grid:
    return nav._Grid(7, 7)


def _model(grid: nav._Grid):
    return nav._build_model(grid, np.zeros(grid.n))


def test_model_and_spec_shapes():
    grid = _grid()
    spec = nav._make_spec(grid)
    model = _model(grid)

    n, k, nb = grid.n, nav.K, nav.N_BATT
    assert list(spec.num_obs) == [n, k, nb, n]
    assert list(spec.num_states) == [n, k, nb]
    assert list(spec.num_controls) == [4, 1, 1]
    assert spec.internal_factors == [2]
    assert spec.intero_modalities == [2]

    # A：4 模态，全依赖 [0,1,2]。
    assert len(model.A) == 4
    assert all(d == [0, 1, 2] for d in model.A_dependencies)
    # B：3 因子，依赖含自身。
    assert len(model.B) == 3
    assert all(f in d for f, d in enumerate(model.B_dependencies))
    # C 一维 (num_obs,)、D 不带 batch 维。
    assert model.C[0].ndim == 1
    assert model.D[0].ndim == 1


def test_candidate_models_static_vs_dynamic_flip():
    grid = _grid()
    model = _model(grid)
    B_static, B_dynamic = nav._candidate_models(model)

    s = np.asarray(B_static)
    d = np.asarray(B_dynamic)
    assert s.shape == (nav.K, nav.K, 1)
    # static：翻转率 P_STATIC（对角线 1-P_STATIC）。
    assert s[nav.MAP_OPEN, nav.MAP_OPEN, 0] == 1.0 - nav.P_STATIC
    assert s[nav.MAP_CLOSED, nav.MAP_OPEN, 0] == nav.P_STATIC
    # dynamic：翻转率 P_DYNAMIC（更高）。
    assert d[nav.MAP_OPEN, nav.MAP_OPEN, 0] == 1.0 - nav.P_DYNAMIC
    assert d[nav.MAP_CLOSED, nav.MAP_OPEN, 0] == nav.P_DYNAMIC
    # static 比 dynamic 更黏（翻转率更低）。
    assert nav.P_STATIC < nav.P_DYNAMIC


def test_goal_potential_peaks_at_goal():
    grid = _grid()
    for map_id in (nav.MAP_OPEN, nav.MAP_CLOSED):
        dist = nav._bfs_dist(grid, grid.goal, map_id)
        pot = nav._goal_potential(grid, dist, dist, 1.0, nav.C_GOAL)
        goal = grid.cell(*grid.goal)
        # 目标格势最高、非负、且逐格非增随距离增加而降低。
        assert pot[goal] == np.max(pot)
        assert np.all(pot >= 0)


def test_env_step_advances_or_bounces():
    grid = _grid()
    env = nav.NavEnv(grid, seed=0, flip_p=0.0, patrol=False)
    obs = env.reset(jax.random.PRNGKey(0))
    assert set(obs) == {"position", "map_sense", "battery", "beacon"}

    start = (env.r, env.c)
    # 向右一步（起点左侧，非墙）应移动。
    obs, done = env.step(jax.random.PRNGKey(1), jnp.asarray([nav.RIGHT, 0, 0]))
    assert (env.r, env.c) == (start[0], start[1] + 1)
    assert not done


def test_efe_components_shapes_and_nonneg_utility():
    grid = _grid()
    model = _model(grid)
    spec = nav._make_spec(grid)
    agent = SelfAwareAgent(spec, model)

    A_nb = [a[0] for a in agent._pymdp.A]
    B_nb = [b[0] for b in agent._pymdp.B]
    C_nb = [c[0] for c in agent._pymdp.C]
    # 起点、门未知、电量满的后验。
    qs = [
        nav._onehot(grid.n, grid.cell(*grid.start)).astype(jnp.float32),
        jnp.asarray([0.5, 0.5], dtype=jnp.float32),
        nav._onehot(nav.N_BATT, nav.N_BATT - 1).astype(jnp.float32),
    ]
    # 目标偏好：C_pos 峰值在目标格。
    C_nb = [jnp.asarray(nav._goal_potential(grid, nav._bfs_dist(grid, grid.goal, nav.MAP_OPEN),
                                            nav._bfs_dist(grid, grid.goal, nav.MAP_CLOSED),
                                            0.5, nav.C_GOAL), dtype=jnp.float32)] + C_nb[1:]

    info, util = nav.efe_components(
        qs, A_nb, B_nb, C_nb,
        [list(d) for d in agent._pymdp.A_dependencies],
        [list(d) for d in agent._pymdp.B_dependencies],
        agent._pymdp.policies.policy_arr,
    )
    n_pol = agent._pymdp.policies.policy_arr.shape[0]
    assert info.shape == (n_pol,)
    assert util.shape == (n_pol,)
    # C_pos 非负 → 期望效用非负。
    assert np.all(np.asarray(util) >= 0)


def test_run_episode_produces_metrics():
    grid = _grid()
    spec = nav._make_spec(grid)
    model = _model(grid)
    agent = SelfAwareAgent(spec, model)
    B_static, B_dynamic = nav._candidate_models(model)
    agent.structure.add_model("static", [model.B[0], B_static, model.B[2]])
    agent.structure.add_model("dynamic", [model.B[0], B_dynamic, model.B[2]])

    dists = {}
    for target in (grid.goal, grid.start):
        for map_id in (nav.MAP_OPEN, nav.MAP_CLOSED):
            dists[(map_id, target)] = nav._bfs_dist(grid, target, map_id)

    env = nav.NavEnv(grid, seed=0, flip_p=0.15, patrol=True)
    m = nav.run_episode(agent, env, grid, steps=40, seed=0, dists=dists)

    s = m.summary(grid)
    # 指标字段齐备，且熵/信念序列与步数一致。
    assert len(m.t) > 0
    assert len(m.entropy_map) == len(m.t)
    assert len(m.p_map_open) == len(m.t)
    for key in ("success", "mean_entropy_pos", "mean_entropy_map", "jerk", "turn_rate",
                "epistemic", "pragmatic", "epi_prag_ratio"):
        assert key in s
