"""三层「自我」组合演示：一次跑通 M2 内感受 + M3 元认知 + M4 结构学习。

在一条 1D 走廊上，智能体同时：

- **M2 内感受**：能量 / 饱足两个「自我」因子，分别在 HOME（pos=0）与 GOAL（pos=2）
  补给、别处每步 −1；偏好 ``C`` 编码稳态 setpoint。两个内部稳态互斥（在 HOME 补能量
  时耗饱足，反之亦然），于是「能量见底 → 回 HOME → 饱足见底 → 去 GOAL」的觅食循环从
  概率上涌现（没有任何 ``if`` 硬规则）。偏离 setpoint 时 allostatic 精度 ``γ`` 升高。
- **M3 元认知**：多一个**故意错报**的「坏传感器」（50% 概率报随机位置）。逐模态可靠性
  持续下降，该模态被自动降权——「我知道这个传感器现在不可靠」。
- **M4 结构学习**：注册两个候选转移模型 ``working``（左右移真生效）vs ``broken``（左右
  移是恒等/失灵），贝叶斯模型比较的后验收敛到 ``working``——「我知道我的腿还灵」。

运行：``python -m aif.examples.self_demo --steps 120``
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from aif import AgentSpec, Environment, GenerativeModel, SelfAwareAgent, obs_to_list

N_POS = 4        # 0=HOME(充能量) 1 2=GOAL(充饱足) 3=死胡同
N_DRIVE = 5      # 能量 / 饱足的档位：0 空 … 4 满
N_ACTIONS = 3    # 0=左 1=右 2=停
HOME = 0
GOAL = 2

# 两个内部稳态的 log-偏好（越高越受偏爱；0 = 充满）。
C_DRIVE = [-6.0, -3.0, -1.5, -0.5, 0.0]


def _build_model() -> GenerativeModel:
    # B[0]：位置转移（4,4,3），动作 0=左 1=右 2=停，撞墙原地。
    B_pos = np.zeros((N_POS, N_POS, N_ACTIONS))
    for old in range(N_POS):
        for a in range(N_ACTIONS):
            nxt = old
            if a == 0:
                nxt = max(old - 1, 0)
            elif a == 1:
                nxt = min(old + 1, N_POS - 1)
            B_pos[nxt, old, a] = 1.0

    # B[1]/B[2]：能量 / 饱足转移（N,4,N,1），依赖 [位置, 自身]。对应格充满，否则 −1。
    def _drive_B(recharge: int) -> np.ndarray:
        B = np.zeros((N_DRIVE, N_POS, N_DRIVE, 1))
        for pos in range(N_POS):
            for cur in range(N_DRIVE):
                nxt = N_DRIVE - 1 if pos == recharge else max(cur - 1, 0)
                B[nxt, pos, cur, 0] = 1.0
        return B

    B_eng = _drive_B(HOME)
    B_food = _drive_B(GOAL)

    # A：全依赖（避免 pymdp 单因子模态 marginal 似然 bug，见 docs/pymdp_api_notes.md §8d）。
    # m0 位置（可靠）、m1 能量、m2 饱足（内感受）、m3「坏传感器」（模型以为可靠，环境错报）。
    A_pos_2d = 0.9 * np.eye(N_POS) + 0.1 / N_POS * np.ones((N_POS, N_POS))
    A_drv_2d = np.eye(N_DRIVE)
    A_pos = np.broadcast_to(
        A_pos_2d[:, :, None, None], (N_POS, N_POS, N_DRIVE, N_DRIVE)
    )
    A_eng = np.broadcast_to(
        A_drv_2d[:, None, :, None], (N_DRIVE, N_POS, N_DRIVE, N_DRIVE)
    )
    A_food = np.broadcast_to(
        A_drv_2d[:, None, None, :], (N_DRIVE, N_POS, N_DRIVE, N_DRIVE)
    )
    A_flaky = np.array(A_pos)  # 与 m0 同构：模型「以为」它也可靠

    return GenerativeModel(
        A=[jnp.asarray(A_pos), jnp.asarray(A_eng), jnp.asarray(A_food), jnp.asarray(A_flaky)],
        B=[jnp.asarray(B_pos), jnp.asarray(B_eng), jnp.asarray(B_food)],
        C=[
            jnp.zeros(N_POS),
            jnp.asarray(C_DRIVE),
            jnp.asarray(C_DRIVE),
            jnp.zeros(N_POS),
        ],
        D=[
            jnp.asarray([1.0, 0.0, 0.0, 0.0]),
            jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
            jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
        A_dependencies=[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        B_dependencies=[[0], [0, 1], [0, 2]],
    )


def _make_spec() -> AgentSpec:
    return AgentSpec(
        num_obs=[N_POS, N_DRIVE, N_DRIVE, N_POS],
        num_states=[N_POS, N_DRIVE, N_DRIVE],
        num_controls=[N_ACTIONS, 1, 1],
        internal_factors=[1, 2],   # 能量 + 饱足都是「自我」因子
        intero_modalities=[1, 2],  # 对应的内感受模态
        policy_len=3,
    )


def _candidate_models(model: GenerativeModel):
    """构造结构学习的两候选：只换 B[0]（动作 affordance），B[1]/B[2] 共享。"""
    B_working = np.array(model.B[0])
    B_broken = np.zeros_like(B_working)
    for old in range(N_POS):
        B_broken[old, old, :] = 1.0  # 左右移都失灵：恒等
    return jnp.asarray(B_working), jnp.asarray(B_broken)


class Corridor(Environment):
    """1D 走廊 + 能量/饱足双稳态 + 一个错报的坏传感器。"""

    modality_names = ["position", "energy", "food", "flaky"]

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.pos = 0
        self.energy = N_DRIVE - 1
        self.food = N_DRIVE - 1

    def reset(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        self.pos = 0
        self.energy = N_DRIVE - 1
        self.food = N_DRIVE - 1
        return self._observe()

    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], bool]:
        a = int(np.asarray(action).reshape(-1)[0])
        if a == 0:
            self.pos = max(self.pos - 1, 0)
        elif a == 1:
            self.pos = min(self.pos + 1, N_POS - 1)
        self.energy = N_DRIVE - 1 if self.pos == HOME else max(self.energy - 1, 0)
        self.food = N_DRIVE - 1 if self.pos == GOAL else max(self.food - 1, 0)
        return self._observe(), False

    def _observe(self) -> dict[str, jnp.ndarray]:
        # 坏传感器：50% 概率报一个随机位置。
        flaky = self.pos
        if self._rng.random() < 0.5:
            flaky = int(self._rng.integers(N_POS))
        return {
            "position": jnp.asarray([self.pos], dtype=jnp.int32),
            "energy": jnp.asarray([self.energy], dtype=jnp.int32),
            "food": jnp.asarray([self.food], dtype=jnp.int32),
            "flaky": jnp.asarray([flaky], dtype=jnp.int32),
        }


ACTION_NAMES = ["←", "→", "·"]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="aif 三层自我组合演示（M2+M3+M4）")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    spec = _make_spec()
    model = _build_model()
    agent = SelfAwareAgent(spec, model)
    env = Corridor(seed=args.seed)

    # M4：注册候选结构模型（动作 affordance）。
    B_working, B_broken = _candidate_models(model)
    agent.structure.add_model("working", [B_working, model.B[1], model.B[2]])
    agent.structure.add_model("broken", [B_broken, model.B[1], model.B[2]])

    key, k = jax.random.split(key)
    obs = env.reset(k)
    print(f"{'t':>3s} {'act':>3s} {'pos':>3s} {'e':>1s} {'f':>1s} {'γ(M2)':>6s} "
          f"{'rel_bad(M3)':>11s} {'β(M3)':>6s} {'P(work)(M4)':>11s}")
    for t in range(args.steps):
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)
        a = int(np.asarray(result.action).reshape(-1)[0])

        gamma = result.self_state["gamma"]
        rel_bad = result.precision["modality_reliability"][3]  # 坏传感器的可靠性
        beta = result.precision["policy_precision"]
        p_work = agent.structure.posterior.get("working", 0.0)
        print(f"{t:3d} {ACTION_NAMES[a]:>3s} {env.pos:3d} {env.energy:1d} {env.food:1d} "
              f"{gamma:6.2f} {rel_bad:11.3f} {beta:6.2f} {p_work:11.3f}")

        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)


if __name__ == "__main__":
    main()
