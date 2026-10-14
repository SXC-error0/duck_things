"""宠物鸭子 v2：自己溜达 + 累了休息 + **知道你在喊它**。

在 ``duck_pet`` 的双稳态（精力/兴致 → 溜达/休息）之上，加第三个行为：主人呼唤时，
鸭子放下手头的事，**走向主人**。

机制（复用 navigation.py 的「动态目标势」模式）：

- 主人固定在一个位置 ``OWNER``；
- 「呼唤」是一个**外部事件**（真实场景里是唤醒词+声纹，这里按 ``CALL_INTERVAL`` 周期
  模拟，每次持续 ``CALL_DURATION`` 步）；
- 呼唤期间，动态把位置偏好 ``C_pos`` 设成一个**在 OWNER 处取峰值的 BFS 距离势**——鸭子
  的 EFE 务实项被拉向主人，自动「转身走向主人」；
- 呼唤结束，``C_pos`` 归零，鸭子恢复「溜达/休息」的自主循环。

注意：呼唤只是**加一个位置偏好**，精力/兴致的稳态仍在。所以一只**精疲力尽的鸭子**可能
先回家休息、再响应呼唤——这正是「有自我」的分寸：不是无脑服从，而是在自身状态与主人
召唤之间权衡。C_GOAL 调强，呼唤的优先级越高。

运行：``python -m aif.examples.duck_pet_call --steps 120``
"""

from __future__ import annotations

import argparse
import collections
import sys

import jax
import jax.numpy as jnp
import numpy as np

from aif import SelfAwareAgent, obs_to_list
from aif.examples import duck_pet as dp

OWNER = 24               # 主人位置 (4,4)（右下角，离 HOME/探索点都远）
CALL_INTERVAL = 15       # 每多少步呼唤一次
CALL_DURATION = 3        # 每次呼唤持续多少步
C_GOAL = 5.0             # 呼唤时位置偏好的强度（越大越优先响应）


def _bfs_dist(target: int) -> np.ndarray:
    """到 ``target`` 的 BFS 距离，逐格返回。"""
    dist = np.full(dp.N_CELLS, -1.0)
    dist[target] = 0.0
    q = collections.deque([target])
    while q:
        cur = q.popleft()
        r, c = dp._rc(cur)
        for a in range(4):
            nr, nc = dp._move(r, c, a)
            nxt = dp._cell(nr, nc)
            if dist[nxt] < 0:
                dist[nxt] = dist[cur] + 1.0
                q.append(nxt)
    return dist


def _goal_potential(target: int, c_goal: float) -> np.ndarray:
    """``C_pos = c_goal * (max_dist - BFS_dist)``：目标处最高、随距离线性下降。"""
    d = _bfs_dist(target)
    max_d = float(d.max())
    return c_goal * (max_d - d)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="aif 宠物鸭子：溜达 + 休息 + 知道你在喊它")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    spec = dp._make_spec()
    model = dp._build_model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dp._candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    env = dp.PetArena(seed=args.seed)
    goal_pot = _goal_potential(OWNER, C_GOAL)  # 预计算：呼唤时的位置势

    key, k = jax.random.split(key)
    obs = env.reset(k)
    print(f"{'t':>3s} {'call':>4s} {'act':>3s} {'pos':>4s} {'e':>1s} {'f':>1s} {'γ':>6s}")
    for t in range(args.steps):
        # 外部事件：主人呼唤（周期 + 时长）。
        called = (t % CALL_INTERVAL) < CALL_DURATION
        # 动态位置偏好：呼唤时拉向主人，否则归零（恢复溜达/休息）。
        agent._pymdp.C[0] = jnp.asarray(
            goal_pot if called else np.zeros(dp.N_CELLS), dtype=jnp.float32
        )[None, :]

        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)
        a = int(np.asarray(result.action).reshape(-1)[0])

        pos = dp._cell(env.r, env.c)
        marker = "OWNER" if pos == OWNER else ("H" if pos == dp.HOME else ("E" if pos in dp.EXPLORE else str(pos)))
        gamma = result.self_state["gamma"]
        print(f"{t:3d} {'🔔' if called else '·':>4s} {dp.ACTION_NAMES[a]:>3s} {marker:>5s} "
              f"{env.energy:1d} {env.fun:1d} {gamma:6.2f}")

        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)


if __name__ == "__main__":
    main()
