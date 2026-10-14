"""场景演示：有「自我意识」的交易者（M2 + M3 + M4）。

一个离散 POMDP 交易决策场景，把三层自我对应到交易的三个真实问题：

- **M2 内感受 / 自模型**：资金水位 ``capital`` 是「自我」因子（0 低 / 1 中 / 2 高），偏好
  ``C`` 编码「想维持高水位」的稳态。持仓在熊市亏钱、牛市赚钱。资金跌破稳态时 allostatic
  精度 ``γ`` 升高——**回撤越大，去杠杆/减仓的决策越果断**（恐惧/贪婪的生理调节，无 if 规则）。
- **M3 元认知 / 精度估计**：一个「分析师信号」模态（喊涨/喊跌）**其实是随机的**（噪声）。
  模型起初以为它可靠，逐模态可靠性持续下降、该信号被降权——**学会别信一个失效的信息源**，
  转而依赖真实价格动作。
- **M4 结构学习**：市场结构是「趋势（regime 粘滞）」还是「均值回归（regime 频繁翻转）」？
  注册两个候选 regime 转移模型，贝叶斯模型比较的后验收敛到真实的那一个——**学会市场的脾气**。

状态因子：regime（牛/熊，隐藏）、position（现金/持仓，可控）、capital（资金，自我）。
动作：HOLD 保持 / TOGGLE 切换仓位。观测：price（价格动作，与 regime 相关）、capital（内感受）、
signal（分析师信号，实际为随机噪声）。

运行：``python -m aif.examples.trader_demo --steps 120``
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from aif import AgentSpec, Environment, GenerativeModel, SelfAwareAgent, obs_to_list

# 离散取值
BEAR, BULL = 0, 1      # regime：熊 / 牛（隐藏）
CASH, ASSET = 0, 1     # position：现金 / 持仓（可控）
LOW, MID, HIGH = 0, 1, 2  # capital：资金水位（自我）
HOLD, TOGGLE = 0, 1    # 动作：保持 / 切换仓位
DOWN, UP = 0, 1        # price / signal：跌 / 涨

N_REGIME = 2
N_POS = 2
N_CAP = 3

POS_FACTOR = 1  # 可控因子（position）的状态因子下标：动作向量里取这一位

C_CAPITAL = [-6.0, -2.0, 0.0]   # log-偏好：资金越高越受偏爱（稳态 setpoint）


def _build_model() -> GenerativeModel:
    # B[0] regime：隐藏市场状态转移（2,2,1），粘滞（趋势市）。
    B_regime = np.zeros((N_REGIME, N_REGIME, 1))
    B_regime[BEAR, BEAR, 0] = 0.85
    B_regime[BULL, BEAR, 0] = 0.15
    B_regime[BEAR, BULL, 0] = 0.15
    B_regime[BULL, BULL, 0] = 0.85

    # B[1] position：仓位转移（2,2,2）。HOLD 保持 / TOGGLE 切换。
    B_pos = np.zeros((N_POS, N_POS, 2))
    B_pos[CASH, CASH, HOLD] = 1.0
    B_pos[ASSET, ASSET, HOLD] = 1.0
    B_pos[ASSET, CASH, TOGGLE] = 1.0
    B_pos[CASH, ASSET, TOGGLE] = 1.0

    # B[2] capital：资金转移（3,2,2,3,1），依赖 [regime, position, capital]。
    #   现金 → 不变；持仓 + 牛 → +1；持仓 + 熊 → −1（都截断）。
    B_cap = np.zeros((N_CAP, N_REGIME, N_POS, N_CAP, 1))
    for regime in range(N_REGIME):
        for pos in range(N_POS):
            for cur in range(N_CAP):
                if pos == CASH:
                    nxt = cur
                elif regime == BULL:
                    nxt = min(cur + 1, HIGH)
                else:
                    nxt = max(cur - 1, LOW)
                B_cap[nxt, regime, pos, cur, 0] = 1.0

    # A：全依赖（见 docs/pymdp_api_notes.md §8d）。
    # m0 price：P(price|regime)，85% 与 regime 一致。m1 capital：精确（内感受）。
    # m2 signal：模型「以为」与 regime 相关（同 price），但环境其实给随机噪声。
    A_price_2d = np.array([[0.85, 0.15], [0.15, 0.85]])  # [o, regime]
    A_cap_2d = np.eye(N_CAP)
    A_price = np.broadcast_to(A_price_2d[:, :, None, None], (2, N_REGIME, N_POS, N_CAP))
    A_cap = np.broadcast_to(A_cap_2d[:, None, None, :], (N_CAP, N_REGIME, N_POS, N_CAP))
    A_sig = np.array(A_price)  # 模型以为 signal 也可靠

    return GenerativeModel(
        A=[jnp.asarray(A_price), jnp.asarray(A_cap), jnp.asarray(A_sig)],
        B=[jnp.asarray(B_regime), jnp.asarray(B_pos), jnp.asarray(B_cap)],
        C=[jnp.zeros(2), jnp.asarray(C_CAPITAL), jnp.zeros(2)],
        D=[
            jnp.asarray([0.5, 0.5]),            # regime 未知
            jnp.asarray([1.0, 0.0]),            # 起始现金
            jnp.asarray([0.0, 1.0, 0.0]),       # 起始中资金（有上行空间，才愿意冒险）
        ],
        A_dependencies=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        B_dependencies=[[0], [1], [0, 1, 2]],
    )


def _make_spec() -> AgentSpec:
    return AgentSpec(
        num_obs=[2, N_CAP, 2],
        num_states=[N_REGIME, N_POS, N_CAP],
        num_controls=[1, 2, 1],
        internal_factors=[2],    # capital 是「自我」因子
        intero_modalities=[1],   # capital 观测是内感受模态
        policy_len=3,
    )


def _candidate_models(model: GenerativeModel):
    """结构学习两候选：只换 regime 的转移 B[0]（趋势 vs 均值回归），其余共享。"""
    B_trending = np.array(model.B[0])  # 粘滞 0.85（真实）

    B_meanrev = np.zeros_like(B_trending)
    B_meanrev[BEAR, BEAR, 0] = 0.30
    B_meanrev[BULL, BEAR, 0] = 0.70
    B_meanrev[BEAR, BULL, 0] = 0.70
    B_meanrev[BULL, BULL, 0] = 0.30

    return jnp.asarray(B_trending), jnp.asarray(B_meanrev)


class Market(Environment):
    """隐藏 regime（趋势）+ 仓位 + 资金 + 一个随机噪声「分析师信号」。"""

    modality_names = ["price", "capital", "signal"]

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.regime = BULL
        self.position = CASH
        self.capital = MID

    def reset(self, key: jax.Array) -> dict[str, jnp.ndarray]:
        self.regime = BULL
        self.position = CASH
        self.capital = MID
        return self._observe()

    def step(self, key: jax.Array, action: jnp.ndarray) -> tuple[dict[str, jnp.ndarray], bool]:
        a = int(np.asarray(action).reshape(-1)[POS_FACTOR])
        # 仓位：HOLD 保持 / TOGGLE 切换。
        if a == TOGGLE:
            self.position = ASSET if self.position == CASH else CASH
        # 资金：现金不变；持仓按 regime 盈亏。
        if self.position == CASH:
            self.capital = self.capital
        elif self.regime == BULL:
            self.capital = min(self.capital + 1, HIGH)
        else:
            self.capital = max(self.capital - 1, LOW)
        # regime：粘滞转移（趋势市）。
        flip = self._rng.random() < 0.15
        if flip:
            self.regime = BULL if self.regime == BEAR else BEAR
        return self._observe(), False

    def _observe(self) -> dict[str, jnp.ndarray]:
        # price：85% 与 regime 一致。
        correct = self._rng.random() < 0.85
        price = self.regime if correct else (1 - self.regime)
        # signal：纯随机噪声（模型却以为它可靠 → M3 降权）。
        signal = int(self._rng.integers(2))
        return {
            "price": jnp.asarray([price], dtype=jnp.int32),
            "capital": jnp.asarray([self.capital], dtype=jnp.int32),
            "signal": jnp.asarray([signal], dtype=jnp.int32),
        }


POS_NAMES = ["cash", "asset"]
REGIME_NAMES = ["bear", "bull"]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="aif 交易场景演示（M2+M3+M4）")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    spec = _make_spec()
    model = _build_model()
    agent = SelfAwareAgent(spec, model)
    env = Market(seed=args.seed)

    # M4：注册候选市场结构（趋势 vs 均值回归）。
    B_trending, B_meanrev = _candidate_models(model)
    agent.structure.add_model("trending", [B_trending, model.B[1], model.B[2]])
    agent.structure.add_model("mean_revert", [B_meanrev, model.B[1], model.B[2]])

    key, k = jax.random.split(key)
    obs = env.reset(k)
    print(f"{'t':>3s} {'regime':>7s} {'P(bull)':>7s} {'pos':>5s} {'cap':>3s} "
          f"{'γ(M2)':>6s} {'rel_sig(M3)':>11s} {'P(trend)(M4)':>13s}")
    for t in range(args.steps):
        key, k = jax.random.split(key)
        result = agent.step(obs_to_list(obs, env.modality_names), k)

        p_bull = float(np.asarray(result.qs[0][0, -1])[BULL])  # 推断出的牛概率
        gamma = result.self_state["gamma"]
        rel_sig = result.precision["modality_reliability"][2]  # 分析师信号可靠性
        p_trend = agent.structure.posterior.get("trending", 0.0)
        print(f"{t:3d} {REGIME_NAMES[env.regime]:>7s} {p_bull:7.2f} "
              f"{POS_NAMES[env.position]:>5s} {env.capital:3d} "
              f"{gamma:6.2f} {rel_sig:11.3f} {p_trend:13.3f}")

        key, k = jax.random.split(key)
        obs, _done = env.step(k, result.action)


if __name__ == "__main__":
    main()
