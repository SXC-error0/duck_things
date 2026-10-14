"""宠物鸭子：自己溜达 + 累了休息（三层「自我」驱动，M2 + M3 + M4）。

这是「有自我意识的宠物」的第一步——在离散抽象世界里把两个核心行为验证出来：

- **没事自己溜达**：兴致（curiosity）稳态——在家待着会「无聊」，兴致被探索满足；
- **累了回家休息**：精力（energy/fatigue）稳态——四处溜达会「累」，回休息点恢复。

两个**互斥的内部稳态**（这是自主循环涌现的关键，见 gridworld/duck_life 同款机制）：

- 精力 ``energy``：每步 -1，只在 **HOME（休息点）** 回满 → 累了回窝；
- 兴致 ``fun``：每步 -1，只在 **探索点** 回满 → 无聊出去逛。

于是「精力满、兴致空 → 出去逛 → 兴致满、精力空 → 回家休息 → …」的**溜达-休息循环**
从概率上涌现，没有任何 ``if 累 < X: 回家`` 硬规则。多个探索点让「溜达」是逛到不同地方，
而不是两点来回。

三层「自我」对应：

- **M2 内感受**：精力 + 兴致都是「自我」因子，偏好 C 编码稳态 setpoint。偏离稳态时
  allostatic 精度 γ 升高——**哪个稳态被打破，就越果断奔向哪个补给点**；都满足时 γ 低，
  越愿意探索。
- **M3 元认知**：``flaky`` 是一个随机噪声的「幻影有趣检测器」（模型起初以为可靠）。
  逐模态可靠性下降、被自动降权——**学会别信一个失效的传感器**。
- **M4 结构学习**：探索点是「好玩（fun，去了兴致回满）」还是「无聊（boring，去了也没用）」？
  两个候选 ``B_fun`` 的贝叶斯模型比较收敛到真实的那一个——**学会这个去处到底值不值得逛**。

运行：``python -m aif.examples.duck_pet --steps 200``（加 ``--boring`` 让探索点变无聊）。
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from aif import AgentSpec, Environment, GenerativeModel, SelfAwareAgent, obs_to_list

# --------------------------------------------------------------------------- #
# 离散取值与常量
# --------------------------------------------------------------------------- #
GRID = 5
N_CELLS = GRID * GRID
N_ENERGY = 8               # 精力档位：0 累趴 … 7 精神饱满
N_FUN = 8                  # 兴致档位：0 无聊 … 7 开心
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

HOME = 0                   # 休息点 (0,0)
EXPLORE = [2, 10]          # 探索点 (0,2) (2,0) —— 距 HOME 各 2 步，policy_len=3 能看到回家长远收益
START = 12                 # 起步 (2,2)

POS_FACTOR = 0             # 可控因子（position）
ENERGY_FACTOR = 1          # 精力（自我）因子
FUN_FACTOR = 2             # 兴致（自我）因子

POS_ACC = 0.95             # 里程计准确率（噪声落在邻接格）
FUN_ACC = 0.95             # 有趣传感器准确率。噪声率 1-FUN_ACC 与「boring」候选的 p_fun 对齐。
P_FUN_MAIN = 0.90          # 主模型 B_fun：探索点回满兴致的概率（软，防信念腐蚀）
P_FUN_REAL = 0.95          # M4「fun」候选
P_FUN_BORING = 0.05        # M4「boring」候选（= 1 - FUN_ACC）

C_ENERGY = [float(b) for b in range(N_ENERGY)]  # 精力 log-偏好：越高越受偏爱（M2 稳态）
C_FUN = [float(b) for b in range(N_FUN)]        # 兴致 log-偏好


def _cell(r: int, c: int) -> int:
    return r * GRID + c


def _rc(i: int) -> tuple[int, int]:
    return divmod(i, GRID)


def _move(r: int, c: int, a: int) -> tuple[int, int]:
    if a == UP:
        return max(0, r - 1), c
    if a == DOWN:
        return min(GRID - 1, r + 1), c
    if a == LEFT:
        return r, max(0, c - 1)
    return r, min(GRID - 1, c + 1)  # RIGHT


def _neighbors(i: int) -> list[int]:
    r, c = _rc(i)
    out = []
    for a in range(4):
        nr, nc = _move(r, c, a)
        if (nr, nc) != (r, c):
            out.append(_cell(nr, nc))
    return out


# --------------------------------------------------------------------------- #
# 生成模型
# --------------------------------------------------------------------------- #
def _energy_B() -> np.ndarray:
    """精力转移 (N_ENERGY, N_CELLS, N_ENERGY, 1)：依赖 [position, energy]。HOME 回满，否则 -1。"""
    B = np.zeros((N_ENERGY, N_CELLS, N_ENERGY, 1))
    for pos in range(N_CELLS):
        for cur in range(N_ENERGY):
            nxt = N_ENERGY - 1 if pos == HOME else max(cur - 1, 0)
            B[nxt, pos, cur, 0] = 1.0
    return B


def _fun_B(p_fun: float) -> np.ndarray:
    """兴致转移 (N_FUN, N_CELLS, N_FUN, 1)：依赖 [position, fun]。

    探索点以概率 ``p_fun`` 回满、否则 -1；非探索点必然 -1。概率须严格在 (0,1)。
    """
    B = np.zeros((N_FUN, N_CELLS, N_FUN, 1))
    for pos in range(N_CELLS):
        for cur in range(N_FUN):
            if pos in EXPLORE:
                nxt_full = N_FUN - 1
                nxt_drain = max(cur - 1, 0)
                B[nxt_full, pos, cur, 0] = p_fun
                B[nxt_drain, pos, cur, 0] = 1.0 - p_fun
            else:
                B[max(cur - 1, 0), pos, cur, 0] = 1.0
    return B


def _build_model() -> GenerativeModel:
    n = N_CELLS
    ne, nf = N_ENERGY, N_FUN

    # A[0] 位置：可靠里程计，噪声落在邻接格。
    a_pos_2d = np.zeros((n, n))
    for pos in range(n):
        a_pos_2d[pos, pos] = POS_ACC
        nbrs = _neighbors(pos)
        w = (1.0 - POS_ACC) / max(len(nbrs), 1)
        for nxt in nbrs:
            a_pos_2d[nxt, pos] += w
    A_pos = np.broadcast_to(a_pos_2d[:, :, None, None], (n, n, ne, nf))

    # A[1] 精力 / A[2] 兴致：精确内感受。
    A_energy = np.broadcast_to(np.eye(ne)[:, None, :, None], (ne, n, ne, nf))
    A_fun = np.broadcast_to(np.eye(nf)[:, None, None, :], (nf, n, ne, nf))

    # A[3] 幻影有趣检测器：模型「以为」和兴致内感受一样可靠（环境实为随机 → M3 降权）。
    A_flaky = np.broadcast_to(np.eye(nf)[:, None, None, :], (nf, n, ne, nf))

    # B[0] 位置转移 (n, n, 4)：撞墙回弹。
    B_pos = np.zeros((n, n, 4))
    for pos in range(n):
        r, c = _rc(pos)
        for a in range(4):
            nr, nc = _move(r, c, a)
            B_pos[_cell(nr, nc), pos, a] = 1.0

    B_energy = _energy_B()
    B_fun = _fun_B(P_FUN_MAIN)

    return GenerativeModel(
        A=[jnp.asarray(A_pos), jnp.asarray(A_energy), jnp.asarray(A_fun), jnp.asarray(A_flaky)],
        B=[jnp.asarray(B_pos), jnp.asarray(B_energy), jnp.asarray(B_fun)],
        C=[
            jnp.zeros(n),
            jnp.asarray(C_ENERGY, dtype=jnp.float32),
            jnp.asarray(C_FUN, dtype=jnp.float32),
            jnp.zeros(nf),
        ],
        D=[
            jnp.asarray(_onehot(n, START), dtype=jnp.float32),
            jnp.asarray(_onehot(ne, ne - 1), dtype=jnp.float32),
            jnp.asarray(_onehot(nf, nf - 1), dtype=jnp.float32),
        ],
        A_dependencies=[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        B_dependencies=[[0], [0, 1], [0, 2]],
    )


def _onehot(n: int, i: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def _make_spec() -> AgentSpec:
    return AgentSpec(
        num_obs=[N_CELLS, N_ENERGY, N_FUN, N_FUN],
        num_states=[N_CELLS, N_ENERGY, N_FUN],
        num_controls=[4, 1, 1],
        internal_factors=[ENERGY_FACTOR, FUN_FACTOR],  # 精力 + 兴致都是「自我」因子
        intero_modalities=[1, 2],                      # 对应的内感受模态
        policy_len=3,
    )


def _candidate_models(model: GenerativeModel):
    """结构学习两候选：只换兴致转移 B[2]（fun 回满 vs boring 不回满）。"""
    B_fun = jnp.asarray(_fun_B(P_FUN_REAL))
    B_boring = jnp.asarray(_fun_B(P_FUN_BORING))
    return B_fun, B_boring


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
class PetArena(Environment):
    """5×5 竞技场：HOME 休息点回满精力 + 多个探索点回满兴致 + 幻影有趣检测器。"""

    modality_names = ["position", "energy", "fun", "flaky"]

    def __init__(self, seed: int = 0, boring: bool = False):
        streams = np.random.SeedSequence(seed).spawn(2)
        self._rng_pos, self._rng_flaky = (np.random.default_rng(s) for s in streams)
        self.boring = boring
        self.r, self.c = _rc(START)
        self.energy = N_ENERGY - 1
        self.fun = N_FUN - 1

    def reset(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        self.r, self.c = _rc(START)
        self.energy = N_ENERGY - 1
        self.fun = N_FUN - 1
        return self._observe()

    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], bool]:
        a = int(np.asarray(action).reshape(-1)[POS_FACTOR])
        self.r, self.c = _move(self.r, self.c, a)

        pos = _cell(self.r, self.c)
        self.energy = N_ENERGY - 1 if pos == HOME else max(self.energy - 1, 0)
        # 兴致：探索点回满（除非 boring 世界），否则 -1。
        if pos in EXPLORE and not self.boring:
            self.fun = N_FUN - 1
        else:
            self.fun = max(self.fun - 1, 0)

        return self._observe(), False

    def _observe(self) -> dict[str, jnp.ndarray]:
        pos = _cell(self.r, self.c)
        if self._rng_pos.random() < POS_ACC:
            position = pos
        else:
            nbrs = _neighbors(pos)
            position = int(nbrs[self._rng_pos.integers(len(nbrs))])
        flaky = int(self._rng_flaky.integers(N_FUN))  # 幻影有趣检测器（随机兴致值）
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "energy": jnp.asarray([self.energy], dtype=jnp.int32),
            "fun": jnp.asarray([self.fun], dtype=jnp.int32),
            "flaky": jnp.asarray([flaky], dtype=jnp.int32),
        }


ACTION_NAMES = ["↑", "↓", "←", "→"]


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="aif 宠物鸭子场景（自己溜达 + 累了休息）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boring", action="store_true", help="探索点变无聊，测 M4 收敛到 boring")
    ap.add_argument("--viz", action="store_true", help="生成 GIF 动画 + 汇总图")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--live", action="store_true", help="实时窗口：每步刷新网格 + 轨迹 + 精力/兴致/γ")
    ap.add_argument("--live-delay", type=float, default=0.3, help="实时窗口每步间隔（秒）")
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    spec = _make_spec()
    model = _build_model()
    agent = SelfAwareAgent(spec, model)
    env = PetArena(seed=args.seed, boring=args.boring)

    B_fun, B_boring = _candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    fig = ax = None
    if args.live:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=80)

    rec: list[dict] = []
    key, k = jax.random.split(key)
    obs = env.reset(k)
    print(f"{'t':>3s} {'act':>3s} {'pos':>3s} {'e':>1s} {'f':>1s} "
          f"{'γ(M2)':>6s} {'rel_flaky(M3)':>13s} {'P(fun)(M4)':>11s}")
    for t in range(args.steps):
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)
        a = int(np.asarray(result.action).reshape(-1)[0])

        pos = _cell(env.r, env.c)
        pos_marker = "H" if pos == HOME else ("E" if pos in EXPLORE else str(pos))
        gamma = result.self_state["gamma"]
        rel_flaky = result.precision["modality_reliability"][3]
        p_fun = agent.structure.posterior.get("fun", 0.0)
        rec.append({"cell": pos, "energy": env.energy, "fun": env.fun, "gamma": gamma})
        print(f"{t:3d} {ACTION_NAMES[a]:>3s} {pos_marker:>3s} {env.energy:1d} {env.fun:1d} "
              f"{gamma:6.2f} {rel_flaky:13.3f} {p_fun:11.3f}")

        if args.live:
            _render_frame(rec, t, ax)
            fig.canvas.draw()
            fig.canvas.flush_events()
            import matplotlib.pyplot as plt

            plt.pause(args.live_delay)

        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)

    if args.viz:
        import os

        gif_path = os.path.join(args.outdir, "duck_pet.gif")
        png_path = os.path.join(args.outdir, "duck_pet_summary.png")
        save_gif(rec, gif_path)
        save_summary(rec, png_path)
        print(f"  已生成: {gif_path}\n         {png_path}")


# --------------------------------------------------------------------------- #
# 可视化（matplotlib，仅 --viz 时 import）
# --------------------------------------------------------------------------- #
def _cell_xy(cell: int) -> tuple[int, int]:
    return (cell % GRID, -(cell // GRID))


def _render_frame(rec: list[dict], idx: int, ax) -> None:
    from matplotlib.patches import Rectangle

    ax.clear()
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(-(GRID - 0.5), 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(-(GRID - 1), 1))
    ax.set_yticklabels([str(-y) for y in range(-(GRID - 1), 1)])

    for r in range(GRID):
        for c in range(GRID):
            cell = r * GRID + c
            color = "#f5f5f5"
            if cell == HOME:
                color = "#f1c40f"   # 休息点（黄）
            elif cell in EXPLORE:
                color = "#2ecc71"   # 探索点（绿）
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


if __name__ == "__main__":
    main()
