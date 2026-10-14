"""场景演示：动态闸门迷宫中的目标导航（M2 + M3 + M4，带可视化与评价指标）。

一个离散 POMDP 室内导航场景，把三层自我对应到导航的三个真实问题：

- **M2 内感受 / 自模型**：电量 ``battery`` 是「自我」因子，偏好 ``C`` 编码「想保持高电量」
  的稳态；每步耗电、在充电格回充。电量偏低时 allostatic 精度 ``γ`` 升高——**越没电，
  越果断地（在需要时）去找充电格**。
- **M3 元认知 / 精度估计**：一个「信标」位置模态其实是**随机噪声**（模型起初以为它可靠）。
  逐模态可靠性持续下降、该信号被降权——**学会别信一个失效的定位源**，转而依赖真实里程计。
- **M4 结构学习**：中央闸门是「静态」还是「动态」？注册两个候选 ``B_map``（翻转率 0.05
  vs 0.15），贝叶斯模型比较的后验收敛到真实的那一个——**学会这堵墙会不会变**。这需要
  ``map_sense`` **每步观测**：位置门控会在两次去闸门之间冻结 map 信念，使「静态/动态」的
  转移先验无法与观测噪声区分（已用扫描验证）。

**动态障碍 + 探索**：中央一列竖墙带一个「闸门」缺口。布局 A（门开）近路直通目标；
布局 B（门关）须绕顶部缺口绕行。闸门**每步以 ``FLIP_P`` 概率翻转**（memoryless），智能体
须持续追踪闸门状态才能高效导航（每次翻转后 map 信念重新收敛 = 恢复时间）。探索落在
**主动追踪动态闸门**：闸门状态不断变化，epistemic 价值驱使智能体持续解析闸门不确定性，
在信息寻求（epistemic）与目标达成（pragmatic）之间权衡。

**目标如何给出**：``C_pos`` 是「到当前目标点的期望 BFS 距离」的负值（在智能体自身的
转移模型 ``B_pos`` 与当前 map 信念下求期望）。这比裸曼哈顿距离更适合有墙环境——它让
「绕行」在偏好上就体现为进步，从而让 myopic 的 ``policy_len`` 也能导航到远目标。

状态因子：position（位置，可控）、map（布局 A/B，隐藏）、battery（电量，自我）。
动作：↑↓←→（撞墙回弹）。观测：position（带噪声里程计）、map_sense（每步闸门传感器）、
battery（内感受）、beacon（随机噪声定位）。

运行：``python -m aif.examples.navigation --steps 200``（生成 GIF + 汇总图 + 指标表）。
"""

from __future__ import annotations

import argparse
import collections
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
from pymdp import control

from aif import AgentSpec, Environment, GenerativeModel, SelfAwareAgent, obs_to_list

# --------------------------------------------------------------------------- #
# 离散取值与常量
# --------------------------------------------------------------------------- #
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MAP_OPEN, MAP_CLOSED = 0, 1
K = 2          # 布局因子基数（门开 / 门关）
N_BATT = 8     # 电量档位（0 空 … 7 满）
POS_FACTOR = 0  # 可控因子（position）的状态因子下标：动作向量里取这一位

C_GOAL = 1.0   # 目标偏好的量级（C_pos 每步梯度），越大越「直奔目标」、越小越「爱探索」
C_BATT = [float(b) for b in range(N_BATT)]  # 电量 log-偏好：线性，越高越受偏爱（M2 稳态）

POS_ACC = 0.95   # 里程计准确率（噪声落在邻接格，见 _build_model 的 A[0]）
MAP_ACC = 0.95   # 闸门传感器准确率（每步观测，供 M4 判别静态/动态）。噪声率 1-MAP_ACC
                 # = 5% 须与 M4「静态」候选翻转率 P_STATIC 对齐，否则传感器噪声会被
                 # M4 误读为「闸门在动」，使静态环境被误判为动态。
FLIP_P = 0.15    # 闸门 memoryless 翻转概率（每步）
B_MAP_STAY = 0.85  # 主模型 B_map 的 stay（先验：认为闸门较黏滞）
P_STATIC = 0.05    # M4「静态」候选的翻转率
P_DYNAMIC = 0.15   # M4「动态」候选的翻转率（= FLIP_P）


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #
class _Grid:
    """H×W 网格：中央竖墙 + 底部闸门（布局相关）+ 顶部常开缺口。"""

    def __init__(self, h: int, w: int):
        self.h, self.w = h, w
        self.n = h * w
        self.c_gate = w // 2
        self.start = (h - 1, 0)      # 左下
        self.goal = (h - 1, w - 1)   # 右下
        self.charger = (0, w - 1)    # 右上（绕行路线天然经过，供回充）
        self.door = (h - 1, w // 2)  # 底部闸门

    def cell(self, r: int, c: int) -> int:
        return r * self.w + c

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.w)

    def move(self, r: int, c: int, a: int) -> tuple[int, int]:
        if a == UP:
            return r - 1, c
        if a == DOWN:
            return r + 1, c
        if a == LEFT:
            return r, c - 1
        return r, c + 1  # RIGHT

    def is_wall(self, r: int, c: int, m: int) -> bool:
        if r < 0 or r >= self.h or c < 0 or c >= self.w:
            return True
        if c == self.c_gate:
            if r == 0:
                return False                       # 顶部缺口：常开
            if r == self.h - 1:
                return m == MAP_CLOSED             # 底部闸门：门关则墙
            return True                            # 墙身
        return False


def _bfs_dist(grid: _Grid, target: tuple[int, int], map_id: int) -> np.ndarray:
    """到 ``target`` 的 BFS 距离（布局 ``map_id`` 下），逐格返回；-1 = 不可达。"""
    dist = np.full(grid.n, -1.0)
    tgt = grid.cell(*target)
    dist[tgt] = 0.0
    q = collections.deque([tgt])
    while q:
        cur = q.popleft()
        r, c = grid.rc(cur)
        for a in range(4):
            nr, nc = grid.move(r, c, a)
            if grid.is_wall(nr, nc, map_id):
                continue
            nxt = grid.cell(nr, nc)
            if dist[nxt] < 0:
                dist[nxt] = dist[cur] + 1.0
                q.append(nxt)
    return dist


# --------------------------------------------------------------------------- #
# 生成模型
# --------------------------------------------------------------------------- #
def _build_model(grid: _Grid, C_pos: np.ndarray) -> GenerativeModel:
    n, k, nb = grid.n, K, N_BATT

    # A[0] 位置：可靠里程计。噪声落在**邻接格**而非均匀全图——既贴合里程计漂移，也避免
    # 均匀噪声里 ln((1-acc)/48) 的大负项把 M3 可靠性拖垮（见 docs/pymdp_api_notes.md）。
    a_pos_2d = np.zeros((n, n))
    for pos in range(n):
        r, c = grid.rc(pos)
        a_pos_2d[pos, pos] = POS_ACC
        nbrs = [
            grid.cell(nr, nc)
            for a in range(4)
            for nr, nc in [grid.move(r, c, a)]
            if 0 <= nr < grid.h and 0 <= nc < grid.w
        ]
        w = (1.0 - POS_ACC) / max(len(nbrs), 1)
        for nxt in nbrs:
            a_pos_2d[nxt, pos] += w
    A_pos = np.broadcast_to(a_pos_2d[:, :, None, None], (n, n, k, nb))

    # A[1] 闸门传感器：每步观测（无位置门控）。准确率 MAP_ACC（对角 = MAP_ACC，与
    # 环境 `map_sense` 的「P(正确)=MAP_ACC」一致；也与位置里程计的「对角=POS_ACC」
    # 约定对齐）。位置门控会在两次去闸门之间冻结 map 信念，使 M4 的「静态/动态」转移
    # 先验无法与观测噪声区分（已用扫描验证）。
    off = (1.0 - MAP_ACC) / (k - 1)  # 噪声均分给 k-1 个错误状态
    sharp = MAP_ACC * np.eye(k) + off * (np.ones((k, k)) - np.eye(k))
    A_map = np.zeros((k, n, k, nb))
    for pos in range(n):
        A_map[:, pos, :, :] = sharp[:, :, None]

    # A[2] 电量：精确内感受。
    A_batt = np.broadcast_to(np.eye(nb)[:, None, None, :], (nb, n, k, nb))

    # A[3] 信标：模型「以为」是可靠位置传感器（环境实为随机）。
    A_beacon = np.array(A_pos)

    # B[0] 位置转移（n,n,k,4）：撞墙回弹，门开/关决定底部闸门格是否可通行。
    B_pos = np.zeros((n, n, k, 4))
    for pos in range(n):
        r, c = grid.rc(pos)
        for a in range(4):
            nr, nc = grid.move(r, c, a)
            for m in range(k):
                nxt = pos if grid.is_wall(nr, nc, m) else grid.cell(nr, nc)
                B_pos[nxt, pos, m, a] = 1.0

    # B[1] 布局转移（k,k,1）：主模型 = 较黏滞（小概率翻转）。必须非恒等——若恒等，
    # 一旦 P(map) 收敛到 1，log(0) 的转移先验会把 map 信念永久冻结，动态障碍无法恢复。
    # stay 取 0.85：既让主模型足够「黏」以压低观测噪声造成的伪翻转（否则 M4 会把
    # 10% 观测噪声误判为「动态」），又保留对真实翻转的响应。M4 在此先验之上用候选
    # （静态 P_STATIC vs 动态 P_DYNAMIC）判别闸门到底是静态还是动态。
    stay = B_MAP_STAY
    B_map = np.zeros((k, k, 1))
    B_map[MAP_OPEN, MAP_OPEN, 0] = stay
    B_map[MAP_CLOSED, MAP_OPEN, 0] = 1.0 - stay
    B_map[MAP_CLOSED, MAP_CLOSED, 0] = stay
    B_map[MAP_OPEN, MAP_CLOSED, 0] = 1.0 - stay

    # B[2] 电量转移（nb,n,nb,1）：充电格回满，否则每步 −1。
    B_batt = np.zeros((nb, n, nb, 1))
    charger = grid.cell(*grid.charger)
    for pos in range(n):
        for cur in range(nb):
            nxt = nb - 1 if pos == charger else max(cur - 1, 0)
            B_batt[nxt, pos, cur, 0] = 1.0

    return GenerativeModel(
        A=[jnp.asarray(A_pos), jnp.asarray(A_map), jnp.asarray(A_batt), jnp.asarray(A_beacon)],
        B=[jnp.asarray(B_pos), jnp.asarray(B_map), jnp.asarray(B_batt)],
        C=[
            jnp.asarray(C_pos, dtype=jnp.float32),
            jnp.zeros(k),
            jnp.asarray(C_BATT, dtype=jnp.float32),
            jnp.zeros(n),
        ],
        D=[
            jnp.asarray(_onehot(n, grid.cell(*grid.start)), dtype=jnp.float32),
            jnp.asarray([0.5, 0.5], dtype=jnp.float32),
            jnp.asarray(_onehot(nb, nb - 1), dtype=jnp.float32),
        ],
        A_dependencies=[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        B_dependencies=[[0, 1], [1], [0, 2]],
    )


def _onehot(n: int, i: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def _make_spec(grid: _Grid) -> AgentSpec:
    return AgentSpec(
        num_obs=[grid.n, K, N_BATT, grid.n],
        num_states=[grid.n, K, N_BATT],
        num_controls=[4, 1, 1],
        internal_factors=[2],    # battery 是「自我」因子
        intero_modalities=[2],   # battery 观测是内感受模态
        policy_len=3,
    )


def _candidate_models(model: GenerativeModel):
    """结构学习两候选：只换布局转移 B[1]（静态 vs 动态），其余共享。

    「静态」翻转率 P_STATIC（≈0）、「动态」翻转率 P_DYNAMIC（= FLIP_P）。主模型 stay
    取 0.85（介于两者之间），M4 凭每步观测到的 map 信念转移频率判别闸门到底是静态还是
    动态——这正是结构学习在纠正主模型先验。
    """

    def mat(p: float) -> np.ndarray:
        B = np.zeros_like(np.asarray(model.B[1]))
        B[MAP_OPEN, MAP_OPEN, 0] = 1.0 - p
        B[MAP_CLOSED, MAP_OPEN, 0] = p
        B[MAP_OPEN, MAP_CLOSED, 0] = p
        B[MAP_CLOSED, MAP_CLOSED, 0] = 1.0 - p
        return B

    return jnp.asarray(mat(P_STATIC)), jnp.asarray(mat(P_DYNAMIC))


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
class NavEnv(Environment):
    """中央闸门迷宫 + 每步闸门传感器 + 随机信标 + 电量 + memoryless 动态翻转。"""

    modality_names = ["position", "map_sense", "battery", "beacon"]

    def __init__(self, grid: _Grid, seed: int = 0, flip_p: float = 0.0, patrol: bool = True):
        self.grid = grid
        self.flip_p = flip_p
        self.patrol = patrol
        # 每个随机量各用一条独立 RNG 流（同一 seed 派生）。共享单一 RNG 会让「改某个
        # 模态的观测模型」改变该模态每步消耗的随机数个数，进而移位下游流，使其它模态的
        # 随机序列在同一 seed 下完全不同——不可复现（M4 调试时踩中：POS_ACC 0.95→0.85
        # 使闸门翻转/观测序列都变了，被误读为「位置噪声泄漏进 map 因子」）。
        streams = np.random.SeedSequence(seed).spawn(4)
        (self._rng_pos, self._rng_flip, self._rng_map, self._rng_beacon) = (
            np.random.default_rng(s) for s in streams
        )
        self.r, self.c = grid.start
        self.map = MAP_OPEN
        self.battery = N_BATT - 1
        self.waypoint = grid.goal   # 当前目标点（巡逻时在 goal/start 间切换）
        self.step_count = 0
        self.arrivals = 0

    def reset(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        self.r, self.c = self.grid.start
        self.map = MAP_OPEN
        self.battery = N_BATT - 1
        self.waypoint = self.grid.goal
        self.step_count = 0
        self.arrivals = 0
        return self._observe()

    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], bool]:
        a = int(np.asarray(action).reshape(-1)[POS_FACTOR])
        nr, nc = self.grid.move(self.r, self.c, a)
        if not self.grid.is_wall(nr, nc, self.map):
            self.r, self.c = nr, nc

        # 电量：充电格回满，否则 −1。
        if (self.r, self.c) == self.grid.charger:
            self.battery = N_BATT - 1
        else:
            self.battery = max(self.battery - 1, 0)

        # 动态障碍：每步以 flip_p 概率翻转闸门（memoryless）。
        self.step_count += 1
        if self._rng_flip.random() < self.flip_p:
            self.map = MAP_CLOSED if self.map == MAP_OPEN else MAP_OPEN

        # 巡逻：到达当前目标点就切换目标。
        if self.patrol and (self.r, self.c) == self.waypoint:
            self.waypoint = self.grid.start if self.waypoint == self.grid.goal else self.grid.goal
            self.arrivals += 1

        done = (not self.patrol) and (self.r, self.c) == self.grid.goal
        return self._observe(), done

    def _observe(self) -> dict[str, jnp.ndarray]:
        pos = self.grid.cell(self.r, self.c)
        # 里程计：POS_ACC 精确，否则落在邻接格（与 A[0] 模型一致）。
        if self._rng_pos.random() < POS_ACC:
            position = pos
        else:
            r, c = self.grid.rc(pos)
            nbrs = [
                self.grid.cell(nr, nc)
                for a in range(4)
                for nr, nc in [self.grid.move(r, c, a)]
                if 0 <= nr < self.grid.h and 0 <= nc < self.grid.w
            ]
            position = int(nbrs[self._rng_pos.integers(len(nbrs))])
        # 闸门传感器：每步给出真实布局（MAP_ACC 准）。
        map_sense = self.map if self._rng_map.random() < MAP_ACC else (1 - self.map)
        beacon = int(self._rng_beacon.integers(self.grid.n))  # 纯随机噪声
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "map_sense": jnp.asarray([map_sense], dtype=jnp.int32),
            "battery": jnp.asarray([self.battery], dtype=jnp.int32),
            "beacon": jnp.asarray([beacon], dtype=jnp.int32),
        }


# --------------------------------------------------------------------------- #
# 目标偏好（自适应 BFS 距离势）
# --------------------------------------------------------------------------- #
def _goal_potential(
    grid: _Grid, dist_open: np.ndarray, dist_closed: np.ndarray, p_open: float, c_goal: float
) -> np.ndarray:
    """``C_pos`` = ``c_goal * (max_dist - 期望 BFS 距离)``（在 map 信念 ``p_open`` 下求期望）。

    线性势：每逼近目标一步，偏好 +``c_goal``；``max_dist`` 取两布局可达格的最大 BFS
    距离做参考，使势非负、目标格最高、不可达格最低（不会被电量偏好淹没）。这比
    ``c_goal * (1 - exp/n)`` 那种压缩势强得多——后者在 49 格上每步只有 1/49 的梯度，
    会被 epistemic 项与观测噪声淹没。
    """
    exp = p_open * dist_open + (1.0 - p_open) * dist_closed
    ok = exp >= 0
    max_dist = float(np.max(exp[ok])) if ok.any() else 0.0
    exp = np.where(exp >= 0, exp, max_dist)   # 不可达 → 参考距离（偏好 0）
    return c_goal * (max_dist - np.minimum(exp, max_dist))


# --------------------------------------------------------------------------- #
# EFE 分解（epistemic / pragmatic 各自累加）
# --------------------------------------------------------------------------- #
def efe_components(
    qs: list[jnp.ndarray],
    A: list[jnp.ndarray],
    B: list[jnp.ndarray],
    C: list[jnp.ndarray],
    A_deps: list[list[int]],
    B_deps: list[list[int]],
    policies: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """复刻 ``compute_neg_efe_policy`` 的 scan，但把 info_gain 与 utility 分开累加。

    返回 ``(info, util)``，各 ``(num_policies,)``：epistemic 与 pragmatic 分量。
    """

    def one(policy: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def scan(carry, t):
            qs, info, util = carry
            qs_next = control.compute_expected_state(qs, B, policy[t], B_deps)
            qo = control.compute_expected_obs(qs_next, A, A_deps)
            info = info + control.compute_info_gain(qs_next, qo, A, A_deps)
            util = util + control.compute_expected_utility(qo, C, t)
            return (qs_next, info, util), None

        (_, info, util), _ = jax.lax.scan(
            scan, (qs, 0.0, 0.0), jnp.arange(policies.shape[1])
        )
        return info, util

    return jax.vmap(one)(policies)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _smoothness(positions: list[tuple[int, int]]) -> tuple[float, float]:
    """轨迹平滑度：朝向序列的二阶差分均值（jerk）与转向率。"""
    ang = []
    for i in range(1, len(positions)):
        dr = positions[i][0] - positions[i - 1][0]
        dc = positions[i][1] - positions[i - 1][1]
        if dr == 0 and dc == 0:
            continue  # 撞墙回弹（原地）不计
        ang.append(math.atan2(dc, dr))
    if len(ang) < 3:
        return 0.0, 0.0
    jerk = sum(abs(ang[i] - 2 * ang[i - 1] + ang[i - 2]) for i in range(2, len(ang)))
    jerk /= len(ang) - 2
    turns = sum(1 for i in range(1, len(ang)) if abs(ang[i] - ang[i - 1]) > 1e-6)
    turn_rate = turns / max(len(ang) - 1, 1)
    return float(jerk), float(turn_rate)


def _recovery_times(flips: list[tuple[int, int]], p_open_hist: list[float]) -> list[int | None]:
    """每次布局翻转后，P(map=true_new) 重新 > 0.8 所需步数（恢复时间）。"""
    out: list[int | None] = []
    for t_flip, new_true in flips:
        rt = None
        for dt in range(t_flip, len(p_open_hist)):
            p_open = p_open_hist[dt]
            if new_true == MAP_OPEN and p_open > 0.8:
                rt = dt - t_flip
                break
            if new_true == MAP_CLOSED and p_open < 0.2:
                rt = dt - t_flip
                break
        out.append(rt)
    return out


class Metrics:
    """逐步记录的诊断量 + 事后汇总。"""

    def __init__(self) -> None:
        self.t: list[int] = []
        self.positions: list[tuple[int, int]] = []
        self.map_true: list[int] = []
        self.battery: list[int] = []
        self.p_map_open: list[float] = []
        self.p_dynamic: list[float] = []
        self.entropy_pos: list[float] = []
        self.entropy_map: list[float] = []
        self.gamma: list[float] = []
        self.rel_beacon: list[float] = []
        self.beta: list[float] = []
        self.epi: list[float] = []
        self.prag: list[float] = []
        self.flips: list[tuple[int, int]] = []  # (step, new_true_map)
        self.success = False
        self.steps_to_goal: int | None = None
        self.arrivals = 0

    def summary(self, grid: _Grid) -> dict:
        jerk, turn_rate = _smoothness(self.positions)
        rec = _recovery_times(self.flips, self.p_map_open)
        epi_tot = sum(self.epi)
        prag_tot = sum(self.prag)
        ratio = epi_tot / (prag_tot + 1e-12)
        return {
            "success": self.success,
            "steps_to_goal": self.steps_to_goal,
            "arrivals": self.arrivals,
            "path_length": len(self.positions) - 1,
            "mean_entropy_pos": float(np.mean(self.entropy_pos)),
            "mean_entropy_map": float(np.mean(self.entropy_map)),
            "jerk": jerk,
            "turn_rate": turn_rate,
            "epistemic": epi_tot,
            "pragmatic": prag_tot,
            "epi_prag_ratio": ratio,
            "recovery_times": rec,
            "n_flips": len(self.flips),
            "final_p_dynamic": float(self.p_dynamic[-1]) if self.p_dynamic else 0.0,
        }


# --------------------------------------------------------------------------- #
# 一次运行
# --------------------------------------------------------------------------- #
def run_episode(
    agent: SelfAwareAgent,
    env: NavEnv,
    grid: _Grid,
    steps: int,
    seed: int,
    dists: dict[tuple[int, int], np.ndarray],
) -> Metrics:
    m = Metrics()
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs = env.reset(k)
    p_open = 0.5  # 先验：map 未知
    prev_map = env.map

    for _ in range(steps):
        # 自适应目标偏好：按当前 map 信念与当前目标点更新 C_pos（场景侧 hook）。
        wp = env.waypoint
        C_pos = _goal_potential(
            grid, dists[(MAP_OPEN, wp)], dists[(MAP_CLOSED, wp)], p_open, C_GOAL
        )
        agent._pymdp.C[0] = jnp.asarray(C_pos, dtype=jnp.float32)[None, :]

        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)

        # 记录。
        p_open = float(np.asarray(result.qs[1][0, -1])[MAP_OPEN])
        qs_pos = np.asarray(result.qs[0][0, -1])
        qs_map = np.asarray(result.qs[1][0, -1])
        p_dynamic = agent.structure.posterior.get("dynamic", 0.0) if agent.structure.models else 0.0

        # EFE 分解（与智能体本步策略推断同信念、同 C）。
        A_nb = [a[0] for a in agent._pymdp.A]
        B_nb = [b[0] for b in agent._pymdp.B]
        C_nb = [c[0] for c in agent._pymdp.C]
        qs_nb = [jnp.asarray(q)[0, -1] for q in result.qs]
        info, util = efe_components(
            qs_nb, A_nb, B_nb, C_nb,
            [list(d) for d in agent._pymdp.A_dependencies],
            [list(d) for d in agent._pymdp.B_dependencies],
            agent._pymdp.policies.policy_arr,
        )
        q_pi = np.asarray(result.q_pi[0])
        epi = float((q_pi * np.asarray(info)).sum())
        prag = float((q_pi * np.asarray(util)).sum())

        m.t.append(env.step_count)
        m.positions.append((env.r, env.c))
        m.map_true.append(env.map)
        m.battery.append(env.battery)
        m.p_map_open.append(p_open)
        m.p_dynamic.append(p_dynamic)
        m.entropy_pos.append(_entropy(qs_pos))
        m.entropy_map.append(_entropy(qs_map))
        m.gamma.append(float(result.self_state["gamma"]))
        m.rel_beacon.append(float(result.precision["modality_reliability"][3]))
        m.beta.append(float(result.precision["policy_precision"]))
        m.epi.append(epi)
        m.prag.append(prag)

        # 布局翻转事件（用于恢复时间）。
        if env.map != prev_map:
            m.flips.append((env.step_count, env.map))
            prev_map = env.map

        key, k = jax.random.split(key)
        obs, done = env.step(k, result.action)

        # 成功 / 到达：动作落地后才可能到达目标，故在此（env.step 之后）判定。
        if (env.r, env.c) == grid.goal:
            m.success = True
            if m.steps_to_goal is None:
                m.steps_to_goal = env.step_count
        m.arrivals = env.arrivals
        if done:
            break

    return m


# --------------------------------------------------------------------------- #
# 可视化
# --------------------------------------------------------------------------- #
def _render_frame(grid: _Grid, m: Metrics, idx: int, ax) -> None:
    """渲染一帧：墙/闸门/目标/充电/轨迹 + 闸门信念着色。"""
    ax.clear()
    h, w = grid.h, grid.w
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks(range(w))
    ax.set_yticks(range(h))
    ax.set_aspect("equal")

    for r in range(h):
        for c in range(w):
            if grid.is_wall(r, c, MAP_CLOSED):  # 墙身（两布局共享的部分）
                ax.add_patch(_rect(c, r, color="#222222"))
    # 闸门格：按真实布局画墙 + 按信念叠加颜色。
    dr, dc = grid.door
    if grid.is_wall(dr, dc, m.map_true[idx]):
        ax.add_patch(_rect(dc, dr, color="#222222"))
    else:
        ax.add_patch(_rect(dc, dr, color="#f5f5f5", ec="#999999"))
    # 顶部缺口常开。
    ax.add_patch(_rect(grid.c_gate, 0, color="#f5f5f5", ec="#999999"))

    # 目标 / 充电 / 起点。
    gr, gc = grid.goal
    ax.add_patch(_rect(gc, gr, color="#2ecc71"))
    chr_, chc = grid.charger
    ax.add_patch(_rect(chc, chr_, color="#f1c40f"))
    sr, sc = grid.start
    ax.add_patch(_rect(sc, sr, color="#3498db"))

    # 轨迹 + 当前位置。
    tr = m.positions[: idx + 1]
    if len(tr) > 1:
        xs = [p[1] for p in tr]
        ys = [p[0] for p in tr]
        ax.plot(xs, ys, color="#e74c3c", lw=1.5, alpha=0.7)
    rr, cc = m.positions[idx]
    ax.plot(cc, rr, "o", color="#e74c3c", ms=9)

    p_open = m.p_map_open[idx]
    ax.set_title(
        f"t={m.t[idx]}  pos=({rr},{cc})  P(map=open)={p_open:.2f}  "
        f"γ={m.gamma[idx]:.2f}  epi/prag={m.epi[idx] / (m.prag[idx] + 1e-9):.2f}",
        fontsize=8,
    )
    # 闸门信念着色：绿=偏「门开」，红=偏「门关」。
    col = (p_open, 1.0 - p_open, 0.4)
    ax.plot(dc, dr, "s", color=col, ms=10, mec="black", mew=1.0)


def _rect(c: float, r: float, color: str, ec: str | None = None) -> object:
    from matplotlib.patches import Rectangle

    return Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor=ec or color, lw=0.5)


def save_gif(grid: _Grid, m: Metrics, path: str, stride: int = 1) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=70)
    frames = []
    for idx in range(0, len(m.t), max(stride, 1)):
        _render_frame(grid, m, idx, ax)
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()), "RGBA").convert("RGB")
        frames.append(img)
    plt.close(fig)
    if frames:
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=120, loop=0
        )
    return path


def save_summary(grid: _Grid, m: Metrics, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = m.summary(grid)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    t = m.t

    # (a) 网格 + 最终轨迹。
    ax = axes[0, 0]
    _render_frame(grid, m, len(m.t) - 1, ax)
    ax.set_title(f"final trajectory  (success={s['success']})")

    # (b) 信念熵随时间。
    ax = axes[0, 1]
    ax.plot(t, m.entropy_pos, label="H(position)")
    ax.plot(t, m.entropy_map, label="H(map)")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy (nats)")
    ax.legend()
    ax.set_title("belief entropy")

    # (c) epistemic vs pragmatic 价值。
    ax = axes[0, 2]
    ax.plot(t, m.epi, label="epistemic")
    ax.plot(t, m.prag, label="pragmatic")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title("EFE contribution")

    # (d) P(map=open) + 翻转标记。
    ax = axes[1, 0]
    ax.plot(t, m.p_map_open, label="P(map=open)")
    ax.axhline(0.8, color="gray", ls="--", lw=0.6)
    ax.axhline(0.2, color="gray", ls="--", lw=0.6)
    for t_flip, _ in m.flips:
        ax.axvline(t_flip, color="orange", ls=":", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("map belief + flips")

    # (e) γ(M2) 与 beacon 可靠性(M3)。
    ax = axes[1, 1]
    ax.plot(t, m.gamma, label="γ (M2)")
    ax.plot(t, m.rel_beacon, label="beacon rel (M3)")
    ax.plot(t, m.beta, label="β (M3)")
    ax.set_xlabel("step")
    ax.legend()
    ax.set_title("self layers")

    # (f) P(dynamic) (M4)。
    ax = axes[1, 2]
    ax.plot(t, m.p_dynamic, label="P(dynamic)")
    ax.set_xlabel("step")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("M4 structure")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def _print_metrics(header: str, s: dict) -> None:
    print(header)
    print(f"  成功率/到达目标        : {'是' if s['success'] else '否'}"
          + (f"  (首达 @ {s['steps_to_goal']} 步)" if s['steps_to_goal'] else ""))
    print(f"  巡逻到达次数           : {s['arrivals']}")
    print(f"  路径长度               : {s['path_length']}")
    print(f"  平均信念熵  H(pos)     : {s['mean_entropy_pos']:.3f}"
          f"  H(map)={s['mean_entropy_map']:.3f}")
    print(f"  轨迹平滑度 jerk        : {s['jerk']:.3f}  (转向率 {s['turn_rate']:.3f})")
    print(f"  Epistemic/Pragmatic    : {s['epistemic']:.2f} / {s['pragmatic']:.2f}"
          f"  (比值 {s['epi_prag_ratio']:.3f})")
    if s["n_flips"]:
        rt = " ".join(f"{r}" if r is not None else "—" for r in s["recovery_times"])
        print(f"  布局翻转次数           : {s['n_flips']}  恢复时间(步): {rt}")
    else:
        print("  布局翻转次数           : 0（动态障碍关闭）")
    print(f"  M4 P(dynamic)          : {s['final_p_dynamic']:.3f}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="aif 导航场景演示（探索 + 建图 + 动态障碍）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h", type=int, default=7)
    ap.add_argument("--w", type=int, default=7)
    ap.add_argument("--flip-p", type=float, default=0.15, help="闸门每步翻转概率（0=静态）")
    ap.add_argument("--patrol", action="store_true", default=True,
                    help="巡逻模式：到达目标后折返起点（默认开）")
    ap.add_argument("--no-patrol", dest="patrol", action="store_false")
    ap.add_argument("--episodes", type=int, default=0, help="聚合多次单程导航的成功率/步数")
    ap.add_argument("--no-viz", action="store_true", help="不生成 GIF/PNG")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--gif-stride", type=int, default=1)
    args = ap.parse_args()

    grid = _Grid(args.h, args.w)
    spec = _make_spec(grid)

    # 预计算 BFS 距离（goal/start × open/closed），供自适应目标偏好。
    dists: dict[tuple[int, int], np.ndarray] = {}
    for target in (grid.goal, grid.start):
        for map_id in (MAP_OPEN, MAP_CLOSED):
            dists[(map_id, target)] = _bfs_dist(grid, target, map_id)

    # 聚合模式：多次单程导航（无翻转、无巡逻）。
    if args.episodes > 0:
        successes = 0
        steps_list: list[int] = []
        for seed in range(args.episodes):
            env = NavEnv(grid, seed=seed, flip_p=0.0, patrol=False)
            agent = SelfAwareAgent(spec, _build_model(grid, np.zeros(grid.n)))
            m = run_episode(agent, env, grid, args.steps, seed, dists)
            if m.success:
                successes += 1
                steps_list.append(m.steps_to_goal or 0)
        print(f"聚合 {args.episodes} 次单程导航：成功率 {successes}/{args.episodes} = "
              f"{successes / args.episodes:.2f}")
        if steps_list:
            print(f"  平均步数(成功者) = {np.mean(steps_list):.1f}  (min={min(steps_list)}, "
                  f"max={max(steps_list)})")
        return

    # 单次运行（默认带动态障碍 + 巡逻 + 可视化）。
    import os

    env = NavEnv(grid, seed=args.seed, flip_p=args.flip_p, patrol=args.patrol)
    main_model = _build_model(grid, np.zeros(grid.n))
    agent = SelfAwareAgent(spec, main_model)
    # M4：结构学习候选只换 B[1]（布局转移），B[0]/B[2] 与主模型一致。
    B_static, B_dynamic = _candidate_models(main_model)
    agent.structure.add_model("static", [main_model.B[0], B_static, main_model.B[2]])
    agent.structure.add_model("dynamic", [main_model.B[0], B_dynamic, main_model.B[2]])

    m = run_episode(agent, env, grid, args.steps, args.seed, dists)
    s = m.summary(grid)
    _print_metrics(f"导航场景（steps={args.steps}, seed={args.seed}, flip_p={args.flip_p}, "
                   f"patrol={args.patrol}）:", s)

    if not args.no_viz:
        gif_path = os.path.join(args.outdir, "navigation.gif")
        png_path = os.path.join(args.outdir, "navigation_summary.png")
        save_gif(grid, m, gif_path, stride=args.gif_stride)
        save_summary(grid, m, png_path)
        print(f"  已生成: {gif_path}\n         {png_path}")


if __name__ == "__main__":
    main()
