"""宠物鸭子闭环（mock 身体）：自己溜达 + 累了休息，带实时 2D 可视化。

认知层是 ``aif.examples.duck_pet``（精力/兴致双稳态），身体层是 MockBody（点质量 + 朝向，
纯 numpy，无 GPU）。这是「溜达-休息」决策逻辑的完整闭环演示——先把决策跑对，真机
（velstand 策略转弯太慢的问题）等换策略/调 BAM 后再接。

闭环每步：

    认知层 aif.step(离散观测) → 整数动作（世界方向）
      → 身体「走到相邻目标格为止」（world_move_twist 转向-再走，点质量转身瞬时）
      → 精力/兴致耗散与回充（HOME 回精力、探索点回兴致）
      → 再离散化 → 回认知层

运行（需 LV4 venv 有 jax + aif；本脚本把 microduck_rl/src 加进路径取桥）：

    cd /home/zkxt/dev/Duck
    LV4/.venv/bin/python microduck_rl/scripts/duck_pet_self_drive.py --steps 100 --live
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_MICRODUCK_RL_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_MICRODUCK_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_MICRODUCK_RL_SRC))

# ruff: noqa: I001  —— sys.path 注入后导入跨项目依赖
import jax
import jax.numpy as jnp

from aif import SelfAwareAgent, obs_to_list
from aif.examples import duck_pet as dp
from mjlab_microduck.duck_bridge import world_move_twist, wrap_to_pi

CELL = 0.5  # 每格边长（米）


class Arena:
    """把 duck_pet 的 5×5 格映射到连续地面坐标（x=c·CELL, y=-r·CELL）。"""

    def __init__(self, grid: int = dp.GRID, cell: float = CELL):
        self.grid = grid
        self.cell = cell

    def cell_center(self, i: int) -> tuple[float, float]:
        r, c = divmod(i, self.grid)
        return (c * self.cell, -r * self.cell)

    def cell_of(self, x: float, y: float) -> int:
        c = round(x / self.cell)
        r = round(-y / self.cell)
        c = max(0, min(self.grid - 1, c))
        r = max(0, min(self.grid - 1, r))
        return r * self.grid + c


class MockBody:
    """点质量身体：给定 twist 连续移动 + 朝向积分 + 精力/兴致耗散与回充。"""

    modality_names = dp.PetArena.modality_names

    def __init__(self, arena: Arena, seed: int = 0):
        self.arena = arena
        self.x, self.y = arena.cell_center(dp.START)
        self.heading = 0.0
        self.energy = dp.N_ENERGY - 1
        self.fun = dp.N_FUN - 1
        self._rng = np.random.default_rng(seed)

    def act(self, twist: tuple[float, float, float], dt: float) -> None:
        vx, vy, vtheta = twist
        self.heading = wrap_to_pi(self.heading + vtheta * dt)
        wx = vx * math.cos(self.heading) - vy * math.sin(self.heading)
        wy = vx * math.sin(self.heading) + vy * math.cos(self.heading)
        self.x += wx * dt
        self.y += wy * dt

    def move_to_cell(self, target_cell: int, speed: float, yaw_rate: float, max_t: float = 8.0) -> None:
        tx, ty = self.arena.cell_center(target_cell)
        direction_angle = math.atan2(ty - self.y, tx - self.x)
        t = 0.0
        while t < max_t:
            twist = world_move_twist(self.heading, direction_angle, speed, yaw_rate)
            self.act(twist, 0.02)
            t += 0.02
            if self.arena.cell_of(self.x, self.y) == target_cell:
                break

    def drain_and_recharge(self) -> None:
        pos = self.arena.cell_of(self.x, self.y)
        self.energy = dp.N_ENERGY - 1 if pos == dp.HOME else max(self.energy - 1, 0)
        if pos in dp.EXPLORE:
            self.fun = dp.N_FUN - 1
        else:
            self.fun = max(self.fun - 1, 0)

    def observe(self) -> dict[str, jnp.ndarray]:
        pos = self.arena.cell_of(self.x, self.y)
        if self._rng.random() < dp.POS_ACC:
            position = pos
        else:
            nbrs = dp._neighbors(pos)
            position = int(nbrs[self._rng.integers(len(nbrs))])
        flaky = int(self._rng.integers(dp.N_FUN))
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "energy": jnp.asarray([self.energy], dtype=jnp.int32),
            "fun": jnp.asarray([self.fun], dtype=jnp.int32),
            "flaky": jnp.asarray([flaky], dtype=jnp.int32),
        }


def _marker(cell: int) -> str:
    return "H" if cell == dp.HOME else ("E" if cell in dp.EXPLORE else str(cell))


def _target_cell(cur_cell: int, action: int) -> int:
    r, c = dp._rc(cur_cell)
    nr, nc = dp._move(r, c, action)
    return dp._cell(nr, nc)


def run_loop(
    steps: int = 80,
    speed: float = 0.3,
    yaw_rate: float = 0.8,
    seed: int = 0,
    viz: bool = False,
    outdir: str = ".",
    live: bool = False,
    live_delay: float = 0.3,
) -> list[dict]:
    spec = dp._make_spec()
    model = dp._build_model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dp._candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    body = MockBody(Arena(), seed=seed)
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs = body.observe()

    fig = ax = None
    if live:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=80)

    rec: list[dict] = []
    print(f"{'t':>3s} {'act':>3s} {'cell':>4s} {'e':>1s} {'f':>1s} "
          f"{'γ(M2)':>6s} {'rel_flaky(M3)':>13s} {'P(fun)(M4)':>11s}")
    for t in range(steps):
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, body.modality_names), k)
        action = int(np.asarray(result.action).reshape(-1)[0])

        cur_cell = body.arena.cell_of(body.x, body.y)
        body.move_to_cell(_target_cell(cur_cell, action), speed, yaw_rate)
        body.drain_and_recharge()
        obs = body.observe()

        cell = body.arena.cell_of(body.x, body.y)
        gamma = result.self_state["gamma"]
        rel_flaky = result.precision["modality_reliability"][3]
        p_fun = agent.structure.posterior.get("fun", 0.0)
        rec.append({"cell": cell, "energy": body.energy, "fun": body.fun, "gamma": gamma})
        print(f"{t:3d} {dp.ACTION_NAMES[action]:>3s} {_marker(cell):>4s} "
              f"{body.energy:1d} {body.fun:1d} {gamma:6.2f} {rel_flaky:13.3f} {p_fun:11.3f}")

        if live:
            _render_frame(rec, t, ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            import matplotlib.pyplot as plt

            plt.pause(live_delay)

    if viz:
        import os

        gif_path = os.path.join(outdir, "duck_pet_self.gif")
        png_path = os.path.join(outdir, "duck_pet_self_summary.png")
        save_gif(rec, gif_path)
        save_summary(rec, png_path)
        print(f"  已生成: {gif_path}\n         {png_path}")
    return rec


# --------------------------------------------------------------------------- #
# 可视化（matplotlib，仅 --viz / --live 时 import）
# --------------------------------------------------------------------------- #
def _cell_xy(cell: int) -> tuple[int, int]:
    return (cell % dp.GRID, -(cell // dp.GRID))


def _render_frame(rec: list[dict], idx: int, ax) -> None:
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_xlim(-0.5, dp.GRID - 0.5)
    ax.set_ylim(-(dp.GRID - 0.5), 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(dp.GRID))
    ax.set_yticks(range(-(dp.GRID - 1), 1))
    ax.set_yticklabels([str(-y) for y in range(-(dp.GRID - 1), 1)])

    for r in range(dp.GRID):
        for c in range(dp.GRID):
            cell = r * dp.GRID + c
            color = "#f5f5f5"
            if cell == dp.HOME:
                color = "#f1c40f"
            elif cell in dp.EXPLORE:
                color = "#2ecc71"
            ax.add_patch(Rectangle((c - 0.5, -r - 0.5), 1, 1, facecolor=color, edgecolor="#999999", lw=0.5))

    tr = rec[: idx + 1]
    if len(tr) > 1:
        xs = [_cell_xy(d["cell"])[0] for d in tr]
        ys = [_cell_xy(d["cell"])[1] for d in tr]
        ax.plot(xs, ys, color="#e74c3c", lw=1.5, alpha=0.7)
    cx, cy = _cell_xy(rec[idx]["cell"])
    ax.plot(cx, cy, "o", color="#e74c3c", ms=9)
    d = rec[idx]
    ax.set_title(f"t={idx}  energy={d['energy']}  fun={d['fun']}  gamma={d['gamma']:.2f}", fontsize=8)


def save_gif(rec: list[dict], path: str, stride: int = 1) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=70)
    frames = []
    for idx in range(0, len(rec), max(stride, 1)):
        _render_frame(rec, idx, ax)
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()), "RGBA").convert("RGB")
        frames.append(img)
    plt.close(fig)
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    return path


def save_summary(rec: list[dict], path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _render_frame(rec, len(rec) - 1, axes[0])
    axes[0].set_title("final trajectory")

    t = range(len(rec))
    axes[1].plot(t, [d["energy"] for d in rec], label="energy")
    axes[1].plot(t, [d["fun"] for d in rec], label="fun")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("homeostatic level")
    axes[1].legend()
    axes[1].set_title("wander-rest cycle")

    axes[2].plot(t, [d["gamma"] for d in rec], label="gamma (M2)")
    axes[2].set_xlabel("step")
    axes[2].legend()
    axes[2].set_title("allostatic precision")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="aif 宠物鸭子 ↔ 命令桥 ↔ mock 身体闭环")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--yaw-rate", type=float, default=0.8)
    ap.add_argument("--viz", action="store_true", help="生成 GIF 动画 + 汇总图")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--live", action="store_true", help="实时窗口：每步刷新网格 + 轨迹 + 精力/兴致/γ")
    ap.add_argument("--live-delay", type=float, default=0.3, help="实时窗口每步间隔（秒）")
    args = ap.parse_args()
    run_loop(steps=args.steps, speed=args.speed, yaw_rate=args.yaw_rate,
             seed=args.seed, viz=args.viz, outdir=args.outdir,
             live=args.live, live_delay=args.live_delay)


if __name__ == "__main__":
    main()
