"""闭环演示：aif 认知层 → 命令桥 → 身体 → 再离散化 → 回认知层。

这是「训练好的鸭子自己决定干什么」的**决策闭环**（Phase 2）。认知层是 LV4 的
``aif.examples.duck_life``（离散鸭子生活模型，M2/M3/M4 三层自我）；身体层目前是
``MockBody``（点质量 + 朝向，纯 numpy，可无 GPU 复现），真实 MuJoCo 身体（``PolicyInference``）
是下一步换入的同一接口实现。

闭环每步：

    认知层 aif.step(离散观测) → 整数动作（世界方向）
      → 身体「走到相邻目标格为止」：world_move_twist 每控制节拍重算（先转身再前进），
        进入目标格即停（对应 duck_life 的 B_pos「一步 = 走一格」语义）
      → 身体 drain/recharge（电量/玩心耗散与回充）
      → 再离散化 (x,y) → 格 → 回认知层

关键点：身体用「走到目标格为止」的原语，而非固定时长的开环区间——否则转身耗时会让
鸭子没走完就进入下一步，位置信念与真实位置分叉（初版实测踩中，电量掉到 0 也回不了坞）。

运行（需 LV4 的 venv，含 jax + pymdp + aif；本脚本再把 microduck_rl/src 加入路径取桥）：

    cd /home/zkxt/dev/Duck
    LV4/.venv/bin/python microduck_rl/scripts/duck_self_drive.py --steps 80
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# —— 跨项目路径：桥在 microduck_rl/src，aif 在 LV4/src（aif 已装进 LV4 venv）——
_MICRODUCK_RL_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_MICRODUCK_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_MICRODUCK_RL_SRC))

# ruff: noqa: I001  —— sys.path 注入后导入跨项目依赖，无法放到文件顶部
import jax
import jax.numpy as jnp

from aif import SelfAwareAgent, obs_to_list
from aif.examples import duck_life as dl
from mjlab_microduck.duck_bridge import world_move_twist, wrap_to_pi

CELL = 0.5  # 每格边长（米）
CONTROL_DT = 0.02  # 身体控制节拍（秒，50 Hz）


class Arena:
    """把 duck_life 的 5×5 格映射到连续地面坐标（x=c·CELL, y=-r·CELL）。"""

    def __init__(self, grid: int = dl.GRID, cell: float = CELL):
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
    """点质量身体：给定 twist 连续移动 + 朝向积分 + 电量/玩心耗散与回充。"""

    modality_names = dl.DuckArena.modality_names

    def __init__(self, arena: Arena, seed: int = 0):
        self.arena = arena
        self.x, self.y = arena.cell_center(dl.START)
        self.heading = 0.0  # 初始朝东
        self.battery = dl.N_BATT - 1
        self.play = dl.N_PLAY - 1
        self._rng = np.random.default_rng(seed)

    def act(self, twist: tuple[float, float, float], dt: float) -> None:
        vx, vy, vtheta = twist
        self.heading = wrap_to_pi(self.heading + vtheta * dt)
        # 身体系速度 → 世界系
        wx = vx * math.cos(self.heading) - vy * math.sin(self.heading)
        wy = vx * math.sin(self.heading) + vy * math.cos(self.heading)
        self.x += wx * dt
        self.y += wy * dt

    def move_to_cell(self, target_cell: int, speed: float, yaw_rate: float, max_t: float = 8.0) -> None:
        """转身 + 前进，直到进入 ``target_cell``（或超时）。用朝向控制器逐节拍重算。"""
        tx, ty = self.arena.cell_center(target_cell)
        direction_angle = math.atan2(ty - self.y, tx - self.x)
        t = 0.0
        while t < max_t:
            twist = world_move_twist(self.heading, direction_angle, speed, yaw_rate)
            self.act(twist, CONTROL_DT)
            t += CONTROL_DT
            if self.arena.cell_of(self.x, self.y) == target_cell:
                break

    def drain_and_recharge(self) -> None:
        pos = self.arena.cell_of(self.x, self.y)
        self.battery = dl.N_BATT - 1 if pos == dl.DOCK else max(self.battery - 1, 0)
        self.play = dl.N_PLAY - 1 if pos == dl.BALL else max(self.play - 1, 0)

    def observe(self) -> dict[str, jnp.ndarray]:
        pos = self.arena.cell_of(self.x, self.y)
        # 里程计噪声（与 duck_life.DuckArena._observe 一致）：95% 精确，否则邻接格。
        if self._rng.random() < dl.POS_ACC:
            position = pos
        else:
            nbrs = dl._neighbors(pos)
            position = int(nbrs[self._rng.integers(len(nbrs))])
        flaky = int(self._rng.integers(dl.N_CELLS))  # 幻影定位源（随机）
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "battery": jnp.asarray([self.battery], dtype=jnp.int32),
            "play": jnp.asarray([self.play], dtype=jnp.int32),
            "flaky": jnp.asarray([flaky], dtype=jnp.int32),
        }


def _pos_marker(cell: int) -> str:
    return "D" if cell == dl.DOCK else ("B" if cell == dl.BALL else str(cell))


def _target_cell(cur_cell: int, action: int) -> int:
    """duck_life 动作 → 相邻目标格（与 B_pos 撞墙回弹一致）。"""
    r, c = dl._rc(cur_cell)
    nr, nc = dl._move(r, c, action)
    return dl._cell(nr, nc)


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
    spec = dl._make_spec()
    model = dl._build_model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dl._candidate_models(model)
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
    print(f"{'t':>3s} {'act':>3s} {'cell':>4s} {'b':>1s} {'p':>1s} "
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
        rec.append({"cell": cell, "battery": body.battery, "play": body.play,
                    "gamma": gamma, "rel_flaky": rel_flaky, "p_fun": p_fun})
        print(f"{t:3d} {dl.ACTION_NAMES[action]:>3s} {_pos_marker(cell):>4s} "
              f"{body.battery:1d} {body.play:1d} {gamma:6.2f} {rel_flaky:13.3f} {p_fun:11.3f}")

        if live:
            _render_frame(rec, t, ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            import matplotlib.pyplot as plt

            plt.pause(live_delay)

    if viz:
        import os

        gif_path = os.path.join(outdir, "duck_self.gif")
        png_path = os.path.join(outdir, "duck_self_summary.png")
        save_gif(rec, gif_path)
        save_summary(rec, png_path)
        print(f"  已生成: {gif_path}\n         {png_path}")
    return rec


# --------------------------------------------------------------------------- #
# 可视化（matplotlib，仅 --viz 时 import）
# --------------------------------------------------------------------------- #
def _cell_xy(cell: int) -> tuple[int, int]:
    return (cell % dl.GRID, -(cell // dl.GRID))


def _render_frame(rec: list[dict], idx: int, ax) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_xlim(-0.5, dl.GRID - 0.5)
    ax.set_ylim(-(dl.GRID - 0.5), 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(dl.GRID))
    ax.set_yticks(range(-(dl.GRID - 1), 1))
    ax.set_yticklabels([str(-y) for y in range(-(dl.GRID - 1), 1)])

    for r in range(dl.GRID):
        for c in range(dl.GRID):
            cell = r * dl.GRID + c
            color = "#f5f5f5"
            if cell == dl.DOCK:
                color = "#f1c40f"   # 充电坞（黄）
            elif cell == dl.BALL:
                color = "#2ecc71"   # 球（绿）
            ax.add_patch(Rectangle((c - 0.5, -r - 0.5), 1, 1, facecolor=color, edgecolor="#999999", lw=0.5))

    tr = rec[: idx + 1]
    if len(tr) > 1:
        xs = [_cell_xy(d["cell"])[0] for d in tr]
        ys = [_cell_xy(d["cell"])[1] for d in tr]
        ax.plot(xs, ys, color="#e74c3c", lw=1.5, alpha=0.7)
    cx, cy = _cell_xy(rec[idx]["cell"])
    ax.plot(cx, cy, "o", color="#e74c3c", ms=9)
    d = rec[idx]
    ax.set_title(f"t={idx}  batt={d['battery']}  play={d['play']}  gamma={d['gamma']:.2f}  P(fun)={d['p_fun']:.2f}",
                 fontsize=8)


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
    axes[1].plot(t, [d["battery"] for d in rec], label="battery(M2)")
    axes[1].plot(t, [d["play"] for d in rec], label="play(M2)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("homeostatic level")
    axes[1].legend()
    axes[1].set_title("homeostatic drives")

    axes[2].plot(t, [d["gamma"] for d in rec], label="gamma (M2)")
    axes[2].plot(t, [d["rel_flaky"] for d in rec], label="rel_flaky (M3)")
    axes[2].plot(t, [d["p_fun"] for d in rec], label="P(fun) (M4)")
    axes[2].set_xlabel("step")
    axes[2].legend()
    axes[2].set_title("three self layers")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="aif 认知层 ↔ 命令桥 ↔ 身体的决策闭环")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--yaw-rate", type=float, default=0.8)
    ap.add_argument("--viz", action="store_true", help="生成 GIF 动画 + 汇总图")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--live", action="store_true", help="实时窗口：每步刷新网格 + 三层自我指标")
    ap.add_argument("--live-delay", type=float, default=0.3, help="实时窗口每步间隔（秒）")
    args = ap.parse_args()
    run_loop(steps=args.steps, speed=args.speed, yaw_rate=args.yaw_rate,
             seed=args.seed, viz=args.viz, outdir=args.outdir,
             live=args.live, live_delay=args.live_delay)


if __name__ == "__main__":
    main()
