"""鸭子生活场景：三层「自我」驱动的自主决策（M2 + M3 + M4）。

这是「训练好的鸭子」在**离散抽象世界**里的认知层干跑——还没有接真实 RL 身体，先把
「鸭子该干什么」的决策机制独立验证出来（对应 microduck_rl 侧桥接前的 Phase 0）。

一个 5×5 竞技场，两处地标：

- **DOCK（充电坞）**：站上去电量回满；
- **BALL（球）**：站上去「玩」的满足感回满。

**两个互斥的内部稳态**（这是自主循环涌现的关键，见 gridworld 同款机制）：

- 电量 ``battery``：每步 -1，只在 DOCK 回满；
- 玩心 ``play``：每步 -1，只在 BALL 回满。

于是「电量满、玩心空 → 去 BALL 玩 → 玩心满、电量空 → 回 DOCK 充电 → …」的**觅食循环**
从概率上涌现——鸭子自己决定**什么时候去玩、什么时候回家充电**，没有任何 ``if`` 硬规则。
（只放一个稳态的初版实测会「露营」在充电坞——电量满就没有离开的理由，见本文件历史。）

三层「自我」各自对应一个真实问题：

- **M2 内感受**：电量 + 玩心都是「自我」因子，偏好 ``C`` 编码稳态 setpoint。偏离稳态时
  allostatic 精度 ``γ`` 升高——**哪个稳态被打破，就越果断奔向哪个补给格**；都满足时 γ 低，
  越愿意探索。
- **M3 元认知**：``flaky`` 是一个**随机噪声**的「幻影定位源」（模型起初以为它和里程计一样
  可靠）。逐模态可靠性持续下降、被自动降权——**学会别信一个失效的定位源**，转而依赖真实
  里程计。
- **M4 结构学习**：球是「好玩（fun，站上去玩心回满）」还是「无聊（boring，站上去也没用）」？
  注册两个候选 ``B_play``，贝叶斯模型比较的后验收敛到真实的那一个——**学会这个物体到底
  能不能满足我**。

运行：``python -m aif.examples.duck_life --steps 200``（加 ``--boring`` 让球变得无聊，
测 M4 收敛到 boring）。
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
N_BATT = 8                 # 电量档位：0 空 … 7 满
N_PLAY = 8                 # 玩心档位：0 空 … 7 满
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3

DOCK = 0                    # 充电坞 (0,0)
BALL = 6                    # 球 (1,1) = 1*5+1
START = 1                   # 起步 (0,1)，到 DOCK / BALL 各 1 步

POS_FACTOR = 0              # 可控因子（position）的状态因子下标
BATT_FACTOR = 1             # 电量（自我）因子
PLAY_FACTOR = 2             # 玩心（自我）因子

POS_ACC = 0.95              # 里程计准确率（噪声落在邻接格）

# 球「好玩」概率（在 BALL 格玩心回满的概率）。主模型软（非确定性，避免硬 one-hot 先验与
# 精确观测硬冲突腐蚀信念，见 navigation.py 的 sticky 主模型同理）；M4 候选分别取近确定的两端。
P_RECHARGE_MAIN = 0.90      # 主模型：默认偏「好玩」但留不确定
P_RECHARGE_FUN = 0.95       # M4「fun」候选
P_RECHARGE_BORING = 0.05    # M4「boring」候选

C_BATT = [float(b) for b in range(N_BATT)]  # 电量 log-偏好：线性，越高越受偏爱（M2 稳态）
C_PLAY = [float(b) for b in range(N_PLAY)]  # 玩心 log-偏好（M2 稳态）


def _cell(r: int, c: int) -> int:
    return r * GRID + c


def _rc(i: int) -> tuple[int, int]:
    return divmod(i, GRID)


def _move(r: int, c: int, a: int) -> tuple[int, int]:
    """撞墙回弹（bounce）。"""
    if a == UP:
        return max(0, r - 1), c
    if a == DOWN:
        return min(GRID - 1, r + 1), c
    if a == LEFT:
        return r, max(0, c - 1)
    return r, min(GRID - 1, c + 1)  # RIGHT


def _neighbors(i: int) -> list[int]:
    """一个格子的 4 邻接（界内），用于里程计噪声分布。"""
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
def _battery_B() -> np.ndarray:
    """电量转移 (N_BATT, N_CELLS, N_BATT, 1)：依赖 [position, battery]。DOCK 回满，否则 -1。"""
    B = np.zeros((N_BATT, N_CELLS, N_BATT, 1))
    for pos in range(N_CELLS):
        for cur in range(N_BATT):
            nxt = N_BATT - 1 if pos == DOCK else max(cur - 1, 0)
            B[nxt, pos, cur, 0] = 1.0
    return B


def _play_B(p_recharge: float) -> np.ndarray:
    """玩心转移 (N_PLAY, N_CELLS, N_PLAY, 1)：依赖 [position, play]。

    BALL 格以概率 ``p_recharge`` 回满、以 ``1 - p_recharge`` 照常 -1；非 BALL 格必然 -1。
    概率必须严格在 (0,1)——硬 0/1 会让转移先验与精确观测硬冲突、把信念腐蚀掉（实测踩中）。
    """
    B = np.zeros((N_PLAY, N_CELLS, N_PLAY, 1))
    for pos in range(N_CELLS):
        for cur in range(N_PLAY):
            if pos == BALL:
                nxt_full = N_PLAY - 1
                nxt_drain = max(cur - 1, 0)
                B[nxt_full, pos, cur, 0] = p_recharge
                B[nxt_drain, pos, cur, 0] = 1.0 - p_recharge
            else:
                B[max(cur - 1, 0), pos, cur, 0] = 1.0
    return B


def _build_model() -> GenerativeModel:
    n = N_CELLS
    nb, np_ = N_BATT, N_PLAY

    # A[0] 位置：可靠里程计，噪声落在邻接格（避免均匀噪声的 ln 大负项拖垮 M3 可靠性）。
    a_pos_2d = np.zeros((n, n))
    for pos in range(n):
        a_pos_2d[pos, pos] = POS_ACC
        nbrs = _neighbors(pos)
        w = (1.0 - POS_ACC) / max(len(nbrs), 1)
        for nxt in nbrs:
            a_pos_2d[nxt, pos] += w
    A_pos = np.broadcast_to(a_pos_2d[:, :, None, None], (n, n, nb, np_))

    # A[1] 电量 / A[2] 玩心：精确内感受。
    A_batt = np.broadcast_to(np.eye(nb)[:, None, :, None], (nb, n, nb, np_))
    A_play = np.broadcast_to(np.eye(np_)[:, None, None, :], (np_, n, nb, np_))

    # A[3] 幻影定位源：模型「以为」和里程计一样可靠（环境实为随机）。
    A_flaky = np.array(A_pos)

    # B[0] 位置转移 (n, n, 4)：撞墙回弹。
    B_pos = np.zeros((n, n, 4))
    for pos in range(n):
        r, c = _rc(pos)
        for a in range(4):
            nr, nc = _move(r, c, a)
            B_pos[_cell(nr, nc), pos, a] = 1.0

    B_batt = _battery_B()
    B_play = _play_B(P_RECHARGE_MAIN)  # 主模型：软（默认偏好玩但留不确定）

    return GenerativeModel(
        A=[jnp.asarray(A_pos), jnp.asarray(A_batt), jnp.asarray(A_play), jnp.asarray(A_flaky)],
        B=[jnp.asarray(B_pos), jnp.asarray(B_batt), jnp.asarray(B_play)],
        C=[
            jnp.zeros(n),
            jnp.asarray(C_BATT, dtype=jnp.float32),
            jnp.asarray(C_PLAY, dtype=jnp.float32),
            jnp.zeros(n),
        ],
        D=[
            jnp.asarray(_onehot(n, START), dtype=jnp.float32),
            jnp.asarray(_onehot(nb, nb - 1), dtype=jnp.float32),
            jnp.asarray(_onehot(np_, np_ - 1), dtype=jnp.float32),
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
        num_obs=[N_CELLS, N_BATT, N_PLAY, N_CELLS],
        num_states=[N_CELLS, N_BATT, N_PLAY],
        num_controls=[4, 1, 1],
        internal_factors=[BATT_FACTOR, PLAY_FACTOR],   # 电量 + 玩心都是「自我」因子
        intero_modalities=[1, 2],                      # 对应的内感受模态
        policy_len=3,
    )


def _candidate_models(model: GenerativeModel):
    """结构学习两候选：只换玩心转移 B[2]（fun 近确定回满 vs boring 近确定不回满），其余共享。"""
    B_fun = jnp.asarray(_play_B(P_RECHARGE_FUN))
    B_boring = jnp.asarray(_play_B(P_RECHARGE_BORING))
    return B_fun, B_boring


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
class DuckArena(Environment):
    """5×5 竞技场 + 电量/玩心双稳态 + 一个随机噪声「幻影定位源」。"""

    modality_names = ["position", "battery", "play", "flaky"]

    def __init__(self, seed: int = 0, boring: bool = False):
        # 每个随机量各用一条独立 RNG 流（同一 seed 派生），保证可复现（见 navigation.py）。
        streams = np.random.SeedSequence(seed).spawn(2)
        self._rng_pos, self._rng_flaky = (np.random.default_rng(s) for s in streams)
        self.boring = boring
        self.r, self.c = _rc(START)
        self.battery = N_BATT - 1
        self.play = N_PLAY - 1

    def reset(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        self.r, self.c = _rc(START)
        self.battery = N_BATT - 1
        self.play = N_PLAY - 1
        return self._observe()

    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], bool]:
        a = int(np.asarray(action).reshape(-1)[POS_FACTOR])
        self.r, self.c = _move(self.r, self.c, a)

        pos = _cell(self.r, self.c)
        self.battery = N_BATT - 1 if pos == DOCK else max(self.battery - 1, 0)
        # 玩心：BALL 回满（除非 boring 世界），否则 -1。
        if pos == BALL and not self.boring:
            self.play = N_PLAY - 1
        else:
            self.play = max(self.play - 1, 0)

        return self._observe(), False

    def _observe(self) -> dict[str, jnp.ndarray]:
        pos = _cell(self.r, self.c)
        # 里程计：POS_ACC 精确，否则落在邻接格（与 A[0] 模型一致）。
        if self._rng_pos.random() < POS_ACC:
            position = pos
        else:
            nbrs = _neighbors(pos)
            position = int(nbrs[self._rng_pos.integers(len(nbrs))])
        # 幻影定位源：纯随机噪声（模型却以为它可靠 → M3 降权）。
        flaky = int(self._rng_flaky.integers(N_CELLS))
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "battery": jnp.asarray([self.battery], dtype=jnp.int32),
            "play": jnp.asarray([self.play], dtype=jnp.int32),
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

    ap = argparse.ArgumentParser(description="aif 鸭子生活场景（M2+M3+M4 自主决策）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boring", action="store_true", help="球变得无聊（不回满玩心），测 M4 收敛到 boring")
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    spec = _make_spec()
    model = _build_model()
    agent = SelfAwareAgent(spec, model)
    env = DuckArena(seed=args.seed, boring=args.boring)

    # M4：注册候选球结构（fun 回满 vs boring 不回满）。
    B_fun, B_boring = _candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    key, k = jax.random.split(key)
    obs = env.reset(k)
    print(f"{'t':>3s} {'act':>3s} {'pos':>3s} {'b':>1s} {'p':>1s} "
          f"{'γ(M2)':>6s} {'rel_flaky(M3)':>13s} {'β(M3)':>6s} {'P(fun)(M4)':>11s}")
    for t in range(args.steps):
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)
        a = int(np.asarray(result.action).reshape(-1)[0])

        pos = _cell(env.r, env.c)
        pos_marker = "D" if pos == DOCK else ("B" if pos == BALL else str(pos))
        gamma = result.self_state["gamma"]
        rel_flaky = result.precision["modality_reliability"][3]  # 幻影定位源可靠性
        beta = result.precision["policy_precision"]
        p_fun = agent.structure.posterior.get("fun", 0.0)
        print(f"{t:3d} {ACTION_NAMES[a]:>3s} {pos_marker:>3s} {env.battery:1d} {env.play:1d} "
              f"{gamma:6.2f} {rel_flaky:13.3f} {beta:6.2f} {p_fun:11.3f}")

        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)


if __name__ == "__main__":
    main()
