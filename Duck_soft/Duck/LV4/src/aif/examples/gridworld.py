"""觅食网格：内感受 + allostatic 精度驱动的端到端演示（M2）。

3×3 网格 + 三个状态因子：

- **位置**（9 格，外部因子，4 个动作：上/下/左/右，撞墙原地）；
- **能量**（5 档，内部因子，不可控，转移依赖「位置 + 能量」）：站 HOME 格 → 充满
  （能量=4），否则每步 -1（下限 0）；
- **饱足**（5 档，内部因子，不可控，转移依赖「位置 + 饱足」）：站 GOAL 格 → 充满
  （饱足=4），否则每步 -1（下限 0）。

HOME 是「充电格」，GOAL 是「觅食格」。智能体**只靠偏好 C 维持两个内部稳态阈值**——
没有任何 ``if energy < X: 回家`` / ``if satiety < Y: 去找吃的`` 的硬规则：

- ``C_eng`` / ``C_sat`` 是各自内部因子的稳态 setpoint（log-偏好，越低越惩罚）；
- 两个内部因子都在持续耗散（各自只在专属格被补充），于是「能量见底 → 回 HOME 充电 →
  饱足见底 → 去 GOAL 觅食」的**觅食循环从概率上涌现**——这正是鸭子代码里 THIRST /
  HUNGER 多稳态驱动的通用版。

内感受层（``Interoception``）在此之上把「内部状态偏离 setpoint」映射成 allostatic 精度
``gamma``：任一稳态被打破，策略就越确定（直奔对应补给格）；两者都满足，策略越平坦
（越愿意探索）——每一步都打印这个 gamma 与推断出的两个内部后验。

运行：``python -m aif.examples.gridworld --steps 80``
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from aif import AgentSpec, Environment, GenerativeModel, SelfAwareAgent, obs_to_list

GRID = 3
N_CELLS = GRID * GRID
N_ENERGY = 5
N_SATIETY = 5
N_ACTIONS = 4  # 0=上 1=下 2=左 3=右
HOME = 0       # 充电格（左上角）：能量 → 充满
GOAL = 2       # 觅食格（右上角）：饱足 → 充满（距离 HOME 2 步）

# 两个内部因子的稳态偏好（log-偏好，越高越受偏爱；0 = 充满）。
C_ENERGY = [-6.0, -3.0, -1.5, -0.5, 0.0]
C_SATIETY = [-6.0, -3.0, -1.5, -0.5, 0.0]


def _cell(r: int, c: int) -> int:
    return r * GRID + c


def _move(i: int, a: int) -> int:
    """在 3×3 网格上执行动作 a，撞墙原地。"""
    r, c = divmod(i, GRID)
    if a == 0:
        r = max(0, r - 1)
    elif a == 1:
        r = min(GRID - 1, r + 1)
    elif a == 2:
        c = max(0, c - 1)
    elif a == 3:
        c = min(GRID - 1, c + 1)
    return _cell(r, c)


# ---------------- 生成模型（A/B/C/D） ----------------
def build_model() -> GenerativeModel:
    # B[0]：位置转移 (N_CELLS, N_CELLS, N_ACTIONS)
    B_pos = np.zeros((N_CELLS, N_CELLS, N_ACTIONS))
    for old in range(N_CELLS):
        for a in range(N_ACTIONS):
            B_pos[_move(old, a), old, a] = 1.0

    # B[1]：能量转移，依赖 [位置, 能量]（B_dependencies[1]=[0,1]）。
    # shape (N_ENERGY, N_CELLS, N_ENERGY, 1)：[下一能量, 当前位置, 当前能量, 控制(1)]。
    #   HOME 格 → 充满；其余格 → 每步 -1（下限 0）。
    B_eng = np.zeros((N_ENERGY, N_CELLS, N_ENERGY, 1))
    for pos in range(N_CELLS):
        for cur in range(N_ENERGY):
            nxt = N_ENERGY - 1 if pos == HOME else max(cur - 1, 0)
            B_eng[nxt, pos, cur, 0] = 1.0

    # B[2]：饱足转移，依赖 [位置, 饱足]（B_dependencies[2]=[0,2]）。
    # shape (N_SATIETY, N_CELLS, N_SATIETY, 1)：GOAL 格 → 充满；其余格 → 每步 -1。
    B_sat = np.zeros((N_SATIETY, N_CELLS, N_SATIETY, 1))
    for pos in range(N_CELLS):
        for cur in range(N_SATIETY):
            nxt = N_SATIETY - 1 if pos == GOAL else max(cur - 1, 0)
            B_sat[nxt, pos, cur, 0] = 1.0

    # A：全依赖张量（pymdp 1.0.4 对单因子观测模态的 marginal 似然有 bug，故让每个模态
    # 都依赖全部因子，无关因子在 A 里复制摊平，见 docs/pymdp_api_notes.md §8d）。
    #   A[0][o, pos, eng, sat] = P(o|pos)（位置观测，10% 噪声）
    #   A[1][o, pos, eng, sat] = P(o|eng)（能量观测，精确独热 = 内感受）
    #   A[2][o, pos, eng, sat] = P(o|sat)（饱足观测，精确独热 = 内感受）
    A_pos_2d = 0.9 * np.eye(N_CELLS) + 0.1 / N_CELLS * np.ones((N_CELLS, N_CELLS))
    A_eng_2d = np.eye(N_ENERGY)
    A_sat_2d = np.eye(N_SATIETY)
    A_pos = np.broadcast_to(A_pos_2d[:, :, None, None], (N_CELLS, N_CELLS, N_ENERGY, N_SATIETY))
    A_eng = np.broadcast_to(A_eng_2d[:, None, :, None], (N_ENERGY, N_CELLS, N_ENERGY, N_SATIETY))
    A_sat = np.broadcast_to(A_sat_2d[:, None, None, :], (N_SATIETY, N_CELLS, N_ENERGY, N_SATIETY))

    # C：log-偏好。位置无偏好；能量/饱足偏好高值（稳态 setpoint）。
    # 注意：C 必须是一维 (num_obs,)，不能带 (num_obs,1) 尾单例维——否则 pymdp 的
    # compute_expected_utility 里 (o_m * C_m[t]) 会广播成 (num_obs,num_obs)，把效用
    # 变成与观测无关的常数（见 docs/pymdp_api_notes.md §8e）。
    C_pos = np.zeros(N_CELLS)
    C_eng = np.array(C_ENERGY)
    C_sat = np.array(C_SATIETY)

    # D：位置均匀（需自行定位），能量/饱足起始充满。D 不带 batch 维（见 §8d）。
    D_pos = np.full(N_CELLS, 1.0 / N_CELLS)
    D_eng = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    D_sat = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    return GenerativeModel(
        A=[jnp.asarray(A_pos), jnp.asarray(A_eng), jnp.asarray(A_sat)],
        B=[jnp.asarray(B_pos), jnp.asarray(B_eng), jnp.asarray(B_sat)],
        C=[jnp.asarray(C_pos), jnp.asarray(C_eng), jnp.asarray(C_sat)],
        D=[jnp.asarray(D_pos), jnp.asarray(D_eng), jnp.asarray(D_sat)],
        A_dependencies=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        B_dependencies=[[0], [0, 1], [0, 2]],
    )


def make_spec() -> AgentSpec:
    return AgentSpec(
        num_obs=[N_CELLS, N_ENERGY, N_SATIETY],
        num_states=[N_CELLS, N_ENERGY, N_SATIETY],
        num_controls=[N_ACTIONS, 1, 1],
        internal_factors=[1, 2],   # 能量 + 饱足都是「自我」因子
        intero_modalities=[1, 2],  # 对应的内感受模态
        policy_len=3,              # 足够看到「离补给格 2 步」的多步收益
    )


class GridWorld(Environment):
    """3×3 网格：HOME 充电 / GOAL 觅食。位置观测含轻微噪声。"""

    modality_names = ["position", "energy", "satiety"]

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.pos = np.array([_cell(1, 1)])  # 中心起步
        self.energy = N_ENERGY - 1
        self.satiety = N_SATIETY - 1

    def reset(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        self.pos = np.array([_cell(1, 1)])
        self.energy = N_ENERGY - 1
        self.satiety = N_SATIETY - 1
        return self._observe()

    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], bool]:
        move = int(np.asarray(action).reshape(-1)[0])
        self.pos = np.array([_move(int(self.pos[0]), move)])
        # 能量/饱足转移与 B[1]/B[2] 完全一致：HOME 充满能量，GOAL 充满饱足，否则 -1。
        self.energy = N_ENERGY - 1 if self.pos[0] == HOME else max(self.energy - 1, 0)
        self.satiety = N_SATIETY - 1 if self.pos[0] == GOAL else max(self.satiety - 1, 0)
        return self._observe(), False

    def _observe(self) -> dict[str, jnp.ndarray]:
        p = self.pos[0]
        if self._rng.random() < 0.1:  # 与 A[0] 一致的 10% 位置噪声
            p = int(self._rng.integers(N_CELLS))
        return {
            "position": jnp.asarray([p], dtype=jnp.int32),
            "energy": jnp.asarray([self.energy], dtype=jnp.int32),
            "satiety": jnp.asarray([self.satiety], dtype=jnp.int32),
        }


ACTION_NAMES = ["↑", "↓", "←", "→"]


def main() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="aif 框架觅食网格（内感受 allostatic 驱动）")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    spec = make_spec()
    model = build_model()
    agent = SelfAwareAgent(spec, model)
    env = GridWorld(seed=args.seed)

    key, k = jax.random.split(key)
    obs = env.reset(k)
    print(f"{'t':>3s} {'act':>4s} {'pos':>3s} {'e':>2s} {'s':>2s} "
          f"{'allo_γ':>7s} {'e_sat':>5s} {'s_sat':>5s}")
    for t in range(args.steps):
        # 感知（此刻 env 状态就是 obs 的来源）→ 推断 → 选动作。
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)
        move = int(np.asarray(result.action).reshape(-1)[0])

        # 打印与「感知时刻」对齐的状态 + 后验 + 所选动作。
        ss = result.self_state
        e_post = np.asarray(ss["posterior"][1]).reshape(-1)
        s_post = np.asarray(ss["posterior"][2]).reshape(-1)
        gamma = ss["gamma"]
        pos_marker = "H" if env.pos[0] == HOME else ("G" if env.pos[0] == GOAL else str(env.pos[0]))
        print(f"{t:3d} {ACTION_NAMES[move]:>4s} {pos_marker:>3s} {env.energy:2d} {env.satiety:2d} "
              f"{gamma:7.2f} {ss['satisfaction'][1]:5.2f} {ss['satisfaction'][2]:5.2f}  "
              f"P(e)={np.round(e_post, 1)}  P(s)={np.round(s_post, 1)}")

        # 推进环境。
        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)


if __name__ == "__main__":
    main()
