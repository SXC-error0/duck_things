# pymdp 1.0.4 API 内省笔记（M0 交付）

本文件锁定框架所依赖的 pymdp 真实接口。所有结论来自本机实际运行的内省
（`scratch_introspect*.py`，用完已删），非文档猜测。

## 1. 包与依赖

- PyPI 包名 `inferactively-pymdp`，import 名 `pymdp`，版本 **1.0.4**。
- **JAX-first 重写**：jax 0.10.2 / equinox 0.13.8 / mctx 0.0.71，numpy 2.4.6。
- 本机 Python 3.11.9，Windows 11，uv 解析安装成功，可跑通。

## 2. 模块布局（与旧版差异大）

| 模块 | 内容 | 备注 |
|---|---|---|
| `pymdp.agent` | `Agent`（equinox module） | 核心 |
| `pymdp.control` | EFE 原语 + `Policies` 类 + `sample_action/sample_policy` | EFE 移到这里 |
| `pymdp.maths` | `calc_vfe` / `spm_wnorm`(softmax) / `digamma` / `dirichlet_*` | |
| `pymdp.utils` | `random_A_array` / `random_B_array` / `list_array_uniform` 等 | |
| `pymdp.envs` | `Env` / `GridWorld` / `TMaze` / `GraphEnv` / `rollout` | 自带 GridWorld |

- **没有 `pymdp.functions`**（旧版的 `expected_free_energy` 迁到了 `control`）。
- **没有** `update_A` / `update_B` / `update_D` / `reset`（学习改为函数式
  `infer_parameters`）。

## 3. `Agent.__init__` 关键参数（完整签名见内省输出）

- 生成模型：`A`（似然）、`B`（转移）、`C`（偏好）、`D`（初始先验）、`E`（习惯）、
  `pA`/`pB`（A/B 的 Dirichlet 先验）。
- 结构：`num_controls`、`control_fac_idx`、`policy_len=1`、`policies`（可预传
  `Policies`）。
- **精度（元认知挂载点）**：
  - `gamma=1.0` → **策略精度**（策略后验软最大化的反温度）。
  - `alpha=1.0` → **动作选择精度**（从 q_pi 采样动作的反温度）。
  - 两者都 `jnp.broadcast_to(..., (batch_size,))`，可传标量或数组。
- EFE 开关：`use_utility=True`（务实项）、`use_states_info_gain=True`（状态信息增益/
  认知项）、`use_param_info_gain=False`（参数信息增益）。
- 其他：`action_selection='deterministic'`（或 `'stochastic'`）、
  `sampling_mode='full'`（或 `'marginal'`）、`inference_algo='fpi'`（定点迭代）、
  `num_iter=16`、`batch_size=1`、`learning_mode='online'`、`learn_A/B/C/D/E`。

## 4. 核心方法（实际签名）

- `infer_states(obs, empirical_prior=agent.D, return_info=False, ...) -> (qs, info)`
  - `qs` 形状 `[(batch, 1, num_states_f)]`（含策略/时间维）。
  - `info = {'vfe', 'vfe_components', 'vfe_t'}` —— **`vfe_components` 是逐模态 VFE
    分解**，作为逐模态精度 γ 估计的原料。
- `infer_policies(qs) -> (q_pi, neg_efe)`，`q_pi` 形状 `(batch, num_policies)`，
  `neg_efe = -G`（越小越「好」？注意符号约定）。
- `sample_action(q_pi, rng_key) -> action`。
  - **rng_key 必须带 batch 维**：`batch_size=1` 时传 `jr.split(key, 1)`（形状 `(1,2)`），
    **不能**传 `jr.split(key)[0]`（形状 `(2,)`，会 vmap 尺寸不匹配报错）。
- `infer_parameters(...) -> Agent`：**函数式**，返回一个新 Agent（不原地改）。
- `get_model_dimensions() -> dict`：含 `num_obs/num_states/num_controls/num_modalities/
  num_factors/num_policies/policy_len/A_dependencies/B_dependencies`。

## 5. EFE 原语（`pymdp.control`，可直接复用/包装）

- `compute_neg_efe_policy(qs_init, A, B, C, pA, pB, A_dependencies, B_dependencies,
  policy_i, use_utility=True, use_states_info_gain=True, use_param_info_gain=False) -> Array`
- `compute_expected_utility(qo, C, t=0) -> Array`
- `compute_info_gain(qs, qo, A, A_dependencies) -> Array`
- `compute_expected_state(qs_prior, B, u_t, B_dependencies=None) -> list[Array]`
- `update_posterior_policies(policy_matrix, qs_init, A, B, C, E, pA, pB, A_dependencies,
  B_dependencies, gamma=16.0, use_utility=True, ...) -> (q_pi, neg_efe)`
  - **`gamma=16.0` 默认在这里**；`Agent` 会把自己构造期的 `gamma=1.0` 传入。
- `construct_policies(num_states, num_controls=None, policy_len=1, control_fac_idx=None)`
- `sample_policy` / `sample_action`（`action_selection`, `alpha` 反温度）。

## 6. 精度语义（决定性结论）

- `gamma` = **策略精度**：EFE 差值 → 策略分布的锐度（反温度）。低 gamma = 平 =
  探索；高 gamma = 确定 = 贪婪。→ 框架「策略精度 β」。
- `alpha` = **动作选择精度**：从 q_pi 采样动作的锐度。→ 框架「动作温度」。
- 二者构造期静态；**要做逐步动态，直接调 `control.update_posterior_policies(gamma=…)`
  + `control.sample_policy(alpha=…)`**，而非 `agent.infer_policies/sample_action`。
  这与 pymdp 函数式风格一致。
- 包内**没有任何 `beta`**；「元认知/精度估计」须由本框架实现（计划已如此设计）。

## 7. 随机数与数组约定

- 全部 `jax.random.PRNGKey`（禁止 `np.random`）。
- 数组为 JAX Array；与 numpy 用 `np.asarray` 互通。观测直接喂 JAX 数组即可。
- `utils.random_A_array(key, num_obs, num_states)` / `random_B_array(key, num_states,
  num_controls)` / `list_array_uniform(shapes)` 用于造示例模型。

## 8. 最小端到端（已验证可跑）

```python
key = jr.PRNGKey(0)
num_obs = [3, 5]; num_states = [3, 2]; num_controls = [3, 1]
A = utils.random_A_array(key, num_obs, num_states)
B = utils.random_B_array(key, num_states, num_controls)
C = utils.list_array_uniform([[n] for n in num_obs])
agent = Agent(A=A, B=B, C=C, batch_size=1)
obs = [jnp.array([1]), jnp.array([4])]
qs, info = agent.infer_states(obs, empirical_prior=agent.D, return_info=True)
q_pi, neg_efe = agent.infer_policies(qs)
action = agent.sample_action(q_pi, rng_key=jr.split(key, 1))  # 注意 (1,2)
```

## 8b. 依赖推断（构建自定义生成模型的坑）

- 传**裸数组** `A`/`B` 时，`Agent._construct_dependencies` 走
  `utils.resolve_a_dependencies(num_factors, num_modalities)` → **默认所有模态依赖
  所有因子**（2 因子 2 模态 = `[[0,1],[0,1]]`）。
- 传 `Distribution` 对象时才用 `get_dependencies` 从结构推断。
- **结论**：构建稀疏依赖模型（如 position 观测只依赖 position 因子、energy 观测只
  依赖 energy 因子）时，**必须显式传 `A_dependencies` / `B_dependencies` /
  `B_action_dependencies`**，否则维度对不上（A 的 shape 会按全因子展开）。
- `B_dependencies` 默认 `[[f] for f in range(num_factors)]`（每因子只依赖自身）；
  `B[f]` shape = `(num_states_f, num_states_f, num_controls_f)`，未控因子
  `num_controls_f = 1`。

## 8c. `infer_states(return_info=True)` 的 VFE 诊断坑

- `Agent.infer_states(..., return_info=True)` 会走 `calc_vfe` 装配 VFE 分量，但 batch +
  one-step fpi 下，`qs` 带多余单例维（`(1,1,1,·)`），`calc_vfe` 的 `is_sequence = qs[0]
  .ndim > 1` 被误判，进而 `expand_dims(o_m, range(1, A_m.ndim))` 因 A 未剥 batch 而
  报 `axis N out of bounds`。
- **结论**：框架不用 `return_info=True`；M3 元认知层需要的逐模态 VFE 分量，用
  `maths.calc_vfe`（已验证对任意 A shape 可用）或 `compute_log_likelihood_per_modality`
  自行计算。

## 8d. D 的 batch 维 + 单因子模态 A 的两个坑（M1 实际踩中）

1. **D 不能带 batch 维传入**。`Agent.__init__` 对 B 的广播分支里执行
   `D[f] = broadcast_to(D[f], (batch_size,) + D[f].shape)`，它**假设传入的 D 是
   `(num_states,)`**（无 batch 维）。若传 `(1, num_states)`，会被二次加维成
   `(1,1,num_states)`，后续 `infer_states`/`infer_policies` 的 einsum 全部炸
   （`subscript 'b' does not contain the correct number of indices`）。
   **正确做法：D 传 `(num_states,)`，不带 batch 维。**
2. **单因子观测模态（`A[m]` 只依赖一个状态因子）会崩**。`compute_log_likelihood_
   single_modality` 对单因子 A 返回 1-D 似然，但 `algos.marginal_log_likelihood`
   期望带因子维 → `factor_dot` 的 assert / einsum 崩。**正确做法：让每个模态依赖
   全部因子（`A_dependencies=[[0,1],[0,1]]`），无关因子在 A 里复制摊平**
   （如位置观测 `A[o,pos,eng] = P(o|pos)` 沿 eng 轴复制）。这是经典 pymdp 的标准姿势。

## 8e. C 必须是一维 `(num_obs,)`（M2 实际踩中）

`compute_expected_utility` 的核心是 `util += (o_m * C_m).sum()`，其中 `o_m` 是
预测观测分布 `(num_obs,)`、`C_m` 是 log-偏好。

若把 `C` 构造成 `(num_obs, 1)`（尾单例维），则 `o_m (num_obs,) * C_m (num_obs,1)`
广播成 `(num_obs, num_obs)`，`.sum()` 得到 `Σ_o C_m[o]`——一个**与观测无关的常数**，
于是效用不随状态/动作变化，策略推断退化为纯先验（M2 里表现为「无论什么动作，效用恒为
`-3.0`」）。

**正确做法：C 的每个模态都是一维 `(num_obs,)`**（如 `C_eng = np.array([-4., -1., 0.])`），
不能带 `(num_obs, 1)` 尾单例维。`C` 是 **log-偏好**（正值=偏好，负值=惩罚），稳态
setpoint 由 `softmax(C)` 得到（见 `self/interoception.py` 的 `softmax_log_pref`）。

## 9. 对框架设计的锁定影响

1. **元认知层**输出动态 `gamma`（策略精度）+ `alpha`（动作精度），逐模态 γ 用
   `info['vfe_components']` 驱动 —— 全部走 `control` 原语，不重建 Agent。
2. **结构学习**（模型比较）无原生支持，用 `maths.calc_vfe` 累积 log-evidence 自实现。
3. **内感受** 就是「多一个内部状态因子 + 其观测模态」，写入 C 偏好即 setpoint；
   allostatic 驱动 = 把内部状态偏离转成对恢复策略的 `gamma`/EFE 加权。
4. 示例可用 `pymdp.envs.GridWorld` 作参考或直接复用。
