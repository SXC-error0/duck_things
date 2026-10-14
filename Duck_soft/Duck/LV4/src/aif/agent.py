"""SelfAwareAgent：统一闭环，包装 pymdp ``Agent``。

一步主动推断：

    感知 → 推断状态（含内部）→ 三层自我 → 推断策略 → 采样动作

M1 只是对 pymdp 的薄包装；三层自我以占位方式注入，分别在 M2（内感受）、M3
（元认知）、M4（结构学习）填实。在线滤波通过把上一步后验作为当前先验实现。

M2 起，内感受层的 allostatic 精度通过动态 ``gamma`` 回填进策略推断（见
``_infer_policies``）。pymdp 的 ``gamma`` 构造期静态，故这里直接复刻
``Agent.infer_policies`` 的 vmap 调用、以 ``gamma`` 为参数传入（对齐 pymdp
函数式风格，见 ``docs/pymdp_api_notes.md`` §6）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import lax, nn
from pymdp import control
from pymdp.agent import Agent
from pymdp.algos import all_marginal_log_likelihood
from pymdp.maths import compute_log_likelihood_per_modality, log_stable

from .interfaces import AgentSpec
from .model.generative import GenerativeModel
from .self.interoception import Interoception, normalize_prob, softmax_log_pref
from .self.metacognition import Metacognition
from .self.structure import StructureLearning


def _weighted_fpi(
    A: list[jnp.ndarray],
    obs_vec: list[jnp.ndarray],
    prior: list[jnp.ndarray],
    A_dependencies: list[list[int]],
    weights: Sequence[float],
    num_iter: int,
) -> list[jnp.ndarray]:
    """带逐模态精度的因子化不动点迭代（复刻 ``algos.run_factorized_fpi``）。

    与原生版本唯一的区别：每个模态的对数似然先乘上其可靠性权重 ``w_m``（精度），
    等价于 ``A_m → A_m ** w_m``。``w_m=1`` 退化为原生推理。批量自由（batch 维在外）。
    """
    nf = len(prior)
    log_likelihoods = compute_log_likelihood_per_modality(obs_vec, A, distr_obs=True)
    log_likelihoods = [w * ll for w, ll in zip(weights, log_likelihoods, strict=True)]

    if nf == 1:
        log_q = sum(log_likelihoods) + log_stable(prior[0])
        return [nn.softmax(log_q, axis=-1)]

    log_prior = jtu.tree_map(log_stable, prior)
    log_q = jtu.tree_map(jnp.zeros_like, prior)

    def scan_fn(carry: list[jnp.ndarray], t: jnp.ndarray) -> tuple[list[jnp.ndarray], None]:
        log_q = carry
        q = jtu.tree_map(nn.softmax, log_q)
        marginal_ll = all_marginal_log_likelihood(q, log_likelihoods, A_dependencies)
        log_q = jtu.tree_map(lambda m, p: m + p, marginal_ll, log_prior)
        return log_q, None

    res, _ = lax.scan(scan_fn, log_q, jnp.arange(num_iter))
    return jtu.tree_map(nn.softmax, res)


@dataclass
class StepResult:
    """一步感知-动作产出的全部信息，供诊断/日志。"""

    qs: list[jnp.ndarray]       # 状态后验（含内部因子）
    q_pi: jnp.ndarray           # 策略后验
    neg_efe: jnp.ndarray        # -期望自由能
    action: jnp.ndarray         # 采样出的动作（逐状态因子一个下标，非控制因子恒 0）
    self_state: dict            # 推断出的内部状态及其精度（M2+）
    precision: dict             # 元认知精度估计（M3+）


class SelfAwareAgent:
    """域无关的主动推断智能体，带三层可插拔的「自我」能力。"""

    def __init__(
        self,
        spec: AgentSpec,
        model: GenerativeModel,
        batch_size: int = 1,
        setpoints: Mapping[int, jnp.ndarray] | None = None,
        **interoception_kwargs,
    ):
        self.spec = spec
        self.model = model
        self.batch_size = batch_size

        self._pymdp = Agent(
            A=list(model.A),
            B=list(model.B),
            C=list(model.C),
            D=list(model.D),
            A_dependencies=[list(d) for d in model.A_dependencies],
            B_dependencies=[list(d) for d in model.B_dependencies],
            batch_size=batch_size,
            policy_len=spec.policy_len,
        )

        # 三层自我。内感受层的稳态 setpoint 缺省由 C 偏好归一化得到。
        self.interoception = Interoception(
            spec, self._resolve_setpoints(model, setpoints), **interoception_kwargs
        )
        self.metacognition = Metacognition(spec)
        self.structure = StructureLearning(
            spec,
            A=list(model.A),
            A_dependencies=[list(d) for d in model.A_dependencies],
            B_dependencies=[list(d) for d in model.B_dependencies],
        )

        # 在线滤波：上一步后验作为下一步先验；初始为 D。
        self._prior: list[jnp.ndarray] | None = None
        # 结构学习（M4）：记录上一步后验/动作，供本步「转移解释」累积 evidence。
        self._prev_qs: list[jnp.ndarray] | None = None
        self._prev_u_t: list[int] | None = None

    def _resolve_setpoints(
        self, model: GenerativeModel, setpoints: Mapping[int, jnp.ndarray] | None
    ) -> dict[int, jnp.ndarray]:
        """内部因子下标 -> 稳态 setpoint（内部状态上的目标分布）。"""
        if setpoints is not None:
            return {f: normalize_prob(sp) for f, sp in setpoints.items()}
        # 缺省：内部因子 f 的 setpoint = softmax(C[其内感受模态])。
        # C 是 log-偏好；内感受模态与内部因子按下标平行对应，未标注则退化为
        # 「因子 f 的观测即模态 f」。
        resolved: dict[int, jnp.ndarray] = {}
        intero = list(self.spec.intero_modalities)
        for i, f in enumerate(self.spec.internal_factors):
            m = intero[i] if i < len(intero) else f
            resolved[f] = softmax_log_pref(model.C[m])
        return resolved

    def _empirical_prior(self) -> list[jnp.ndarray]:
        if self._prior is None:
            return list(self._pymdp.D)
        return self._prior

    def _action_to_u_t(self, action: jnp.ndarray) -> list[int]:
        """动作 → 全因子动作向量 u_t。

        pymdp 的 ``sample_action`` 返回的是**逐因子**的动作下标向量（长度
        ``num_factors``，非控制因子恒为 0），与 ``control.compute_expected_state``
        期望的 ``u_t`` 直接同形，原样取下标即可。
        """
        return np.asarray(action).reshape(-1).astype(int).tolist()

    def _predict_next(self, qs: list[jnp.ndarray], action: jnp.ndarray) -> list[jnp.ndarray]:
        """把当前后验经动作转移成下一时刻的经验先验（在线滤波的正确姿势）。

        ``infer_states`` 的 ``empirical_prior`` 是「观测前」的先验，因此必须是上一步
        后验**经转移矩阵推进**后的分布，而不是上一步的原始后验——否则位置变化时，陈旧
        先验会与新鲜观测打架，把信念越洗越散（M2 实测踩中）。这里复刻 pymdp 在 EFE
        里做转移的方式：``control.compute_expected_state``（批量自由，batch 维在外面
        剥离/重建）。
        """
        u_t = self._action_to_u_t(action)

        qs_nb = [q[0] for q in qs]          # 剥离 batch
        B_nb = [b[0] for b in self._pymdp.B]
        qs_next = control.compute_expected_state(
            qs_nb, B_nb, u_t, self._pymdp.B_dependencies
        )
        return [q[None, :] for q in qs_next]  # 重建 batch

    def _infer_policies(
        self, qs: list[jnp.ndarray], gamma: float
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """复刻 ``Agent.infer_policies``，但以显式 ``gamma``（策略精度）调用。"""
        latest_belief = jtu.tree_map(lambda x: x[:, -1], qs)
        infer_policies = partial(
            control.update_posterior_policies_inductive,
            self._pymdp.policies.policy_arr,
            A_dependencies=self._pymdp.A_dependencies,
            B_dependencies=self._pymdp.B_dependencies,
            use_utility=self._pymdp.use_utility,
            use_states_info_gain=self._pymdp.use_states_info_gain,
            use_param_info_gain=self._pymdp.use_param_info_gain,
            use_inductive=self._pymdp.use_inductive,
        )
        gamma_arr = jnp.broadcast_to(
            jnp.asarray(gamma, dtype=jnp.float32), (self.batch_size,)
        )
        q_pi, neg_efe = jax.vmap(infer_policies)(
            latest_belief,
            self._pymdp.A,
            self._pymdp.B,
            self._pymdp.C,
            self._pymdp.E,
            self._pymdp.pA,
            self._pymdp.pB,
            I=self._pymdp.I,
            gamma=gamma_arr,
            inductive_epsilon=self._pymdp.inductive_epsilon,
        )
        return q_pi, neg_efe

    def _infer_states_weighted(
        self, obs: Sequence[jnp.ndarray], prior: list[jnp.ndarray]
    ) -> list[jnp.ndarray]:
        """加权感知：逐模态可靠性作为精度乘到对数似然上，跑因子化 FPI。

        与 ``pymdp.Agent.infer_states`` 等价的感知，但额外把「不可靠模态」降权
        （M3 元认知层）。批量剥/建（batch_size=1 时即为 (1,1,num_states)）。
        """
        weights = self.metacognition.modality_reliability
        o_vec = self._pymdp.make_categorical(list(obs))  # 离散 → one-hot (batch, num_obs)
        A_nb = [a[0] for a in self._pymdp.A]             # 剥离 batch 维
        prior_nb = [p[0] for p in prior]                 # (num_states,)
        o_vec_nb = [o[0] for o in o_vec]                 # (num_obs,)
        qs_nb = _weighted_fpi(
            A_nb,
            o_vec_nb,
            prior_nb,
            self._pymdp.A_dependencies,
            weights,
            self._pymdp.num_iter,
        )
        return [q[None, None, :] for q in qs_nb]          # (1, 1, num_states)

    def step(self, obs: Sequence[jnp.ndarray], rng_key: jax.Array) -> StepResult:
        """推进一步。``obs`` 为按模态顺序排列的数组列表（pymdp 原生格式）。"""
        prior = self._empirical_prior()
        # 感知（M3 加权）：不用 return_info —— pymdp 1.0.4 的 VFE 诊断在 batch + 单因子
        # 模态 + one-step fpi 下有装配坑（见 docs/pymdp_api_notes.md §8c），逐模态
        # accuracy 在元认知层用 maths 原语自行计算。
        qs = self._infer_states_weighted(list(obs), prior)
        # qs 形如 (batch, time, num_states)；取最后一个时间步、去掉时间维，作为当前后验。
        qs_marginals = [q[:, -1] for q in qs]

        # 内感受（M2）：更新自模型 + 得到 allostatic 精度。
        gamma_allo = self.interoception.update(qs, info=None)
        # 策略精度（M3）：allostatic 精度 × 元认知策略精度（低置信度时压低 → 多探索）。
        gamma_eff = gamma_allo * self.metacognition.policy_precision

        q_pi, neg_efe = self._infer_policies(qs, gamma_eff)

        # 元认知（M3）：用本步观测/后验更新逐模态可靠性 + 策略精度（供下一步用）。
        self.metacognition.update(
            [q[0] for q in qs_marginals],
            [o[0] for o in obs],
            [a[0] for a in self._pymdp.A],
            self._pymdp.A_dependencies,
            neg_efe[0],
        )
        # 结构学习（M4）：在已注册的候选模型上，用「上一步后验 → 本步后验」的转移
        # 解释力累积 log-evidence。首次（无上一步）跳过。
        if self._prev_qs is not None and self._prev_u_t is not None and self.structure.models:
            self.structure.update(
                [q[0] for q in qs_marginals],  # 当前后验（批量自由）
                [o[0] for o in obs],           # 当前观测（离散下标）
                self._prev_qs,                 # 上一步后验
                self._prev_u_t,                # 上一步动作
            )

        # 采样动作。rng_key 须带 batch 维（batch_size=1 时为 (1,2)）。
        key, _ = jax.random.split(rng_key)
        action = self._pymdp.sample_action(
            q_pi, rng_key=jax.random.split(key, self.batch_size)
        )

        # 记录本步后验/动作（批量自由），供下一步结构学习做转移解释。
        self._prev_qs = [q[0] for q in qs_marginals]
        self._prev_u_t = self._action_to_u_t(action)

        # 下一时刻的经验先验 = 当前后验经动作转移（在线滤波）。
        self._prior = self._predict_next(qs_marginals, action)

        return StepResult(
            qs=qs,
            q_pi=q_pi,
            neg_efe=neg_efe,
            action=action,
            self_state=self.interoception.state,
            precision=self.metacognition.precision,
        )

    def reset(self) -> None:
        """清空在线滤波状态（回到初始先验 D）。"""
        self._prior = None
        self._prev_qs = None
        self._prev_u_t = None
