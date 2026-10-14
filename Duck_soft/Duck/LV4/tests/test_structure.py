"""M4 结构学习 / 贝叶斯模型比较的单元测试。

验证：在一组**只在转移结构 ``B`` 上不同**的候选模型上累积 log-evidence，
模型后验收敛到「真正解释观测转移」的那个模型。两个靶场景：

1. **自身 action-affordance**：动作 0「右移」是好的还是坏的（``B`` 真移动 vs 恒等）。
2. 世界结构（静态 vs 可动障碍）是同一机制的另一个实例，见 ``structure.py`` 顶部说明。
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.nn import one_hot

from aif import AgentSpec
from aif.self.structure import StructureLearning


def _affordance_setup():
    """单位置因子(3 格) + 单动作因子(0=右移, 1=停)。两候选模型只在动作 0 是否生效上不同。"""
    spec = AgentSpec(num_obs=[3], num_states=[3], num_controls=[2])
    A = [jnp.eye(3)]          # 精确位置观测
    A_deps = [[0]]
    B_deps = [[0]]

    # "working"：动作 0 真右移（越界截断），动作 1 停。
    B_working = np.zeros((3, 3, 2))
    for s in range(3):
        B_working[min(s + 1, 2), s, 0] = 1.0
        B_working[s, s, 1] = 1.0

    # "broken"：动作 0 损坏（恒等），动作 1 停。
    B_broken = np.zeros((3, 3, 2))
    for s in range(3):
        B_broken[s, s, 0] = 1.0
        B_broken[s, s, 1] = 1.0

    return spec, A, A_deps, B_deps, jnp.asarray(B_working), jnp.asarray(B_broken)


def test_posterior_starts_uniform():
    spec, A, A_deps, B_deps, B_working, B_broken = _affordance_setup()
    sl = StructureLearning(spec, A, A_deps, B_deps)
    sl.add_model("working", [B_working])
    sl.add_model("broken", [B_broken])
    assert sl.posterior["working"] == pytest.approx(0.5, abs=1e-4)
    assert sl.posterior["broken"] == pytest.approx(0.5, abs=1e-4)


def test_posterior_converges_to_working_action_model():
    spec, A, A_deps, B_deps, B_working, B_broken = _affordance_setup()
    sl = StructureLearning(spec, A, A_deps, B_deps)
    sl.add_model("working", [B_working])
    sl.add_model("broken", [B_broken])

    # 真动力学 = working：从 0 出发，每步动作 0（右移）。0→1→2→2…
    # 仅在「动作让状态改变」的转移上，"working" 的转移先验贴合实际后验、evidence 更高。
    state = 0
    for _ in range(4):
        prev = state
        state = min(state + 1, 2)
        prev_qs = [one_hot(prev, 3).astype(jnp.float32)]
        qs = [one_hot(state, 3).astype(jnp.float32)]
        obs = [jnp.array(state)]
        sl.update(qs, obs, prev_qs, [0])

    assert sl.posterior["working"] > 0.99
    assert sl.posterior["broken"] < 0.01


def test_posterior_converges_to_broken_when_action_does_nothing():
    spec, A, A_deps, B_deps, B_working, B_broken = _affordance_setup()
    sl = StructureLearning(spec, A, A_deps, B_deps)
    sl.add_model("working", [B_working])
    sl.add_model("broken", [B_broken])

    # 真动力学 = broken：动作 0 不动，状态始终停在 0。
    for _ in range(4):
        prev_qs = [one_hot(0, 3).astype(jnp.float32)]
        qs = [one_hot(0, 3).astype(jnp.float32)]
        obs = [jnp.array(0)]
        sl.update(qs, obs, prev_qs, [0])

    assert sl.posterior["broken"] > 0.99
    assert sl.posterior["working"] < 0.01
