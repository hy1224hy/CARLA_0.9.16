# MAPPO 算法正确性纠正计划

> 创建日期：2026-06-08
> 范围：`train_no_rllib_mappo.py`（标准 MAPPO）与 `train_adversarial_ppo.py`（对抗单体 PPO）
> 目标：修复 4 个影响算法正确性的缺陷 + 若干次要问题

---

## 一、问题清单（按严重程度）

| 编号 | 严重度 | 问题 | 位置 |
|------|--------|------|------|
| #1 | 🔴 P0 | 单一共享 `V(s)` 在"每 agent 不同 reward"下被喂入互相矛盾的回归目标，critic 收敛到 per-agent return 的均值，baseline 有偏 | `train_no_rllib_mappo.py` critic 训练 / `actor_critic.py:CentralizedCritic` |
| #2 | 🔴 P0 | 时间截断（`current_step >= max_steps`）被当成真正终止，GAE 不 bootstrap，系统性低估 value | `carla_multi_agent_env.py:_check_dones` + `compute_gae` |
| #3 | 🔴 P0 | rollout 边界（回合未结束就到 rollout_steps）`next_value=0`，把未完成回合最后一步当终止 | `compute_gae` 初值 |
| #4 | 🟠 P1 | tanh 策略 `old_logp` 通过 `atanh` 反推 z 重算，饱和动作下首个 epoch ratio≠1，PPO clip 基准错误 | `actor_critic.py:evaluate_actions` |
| #5 | 🟡 P2 | entropy 用 base Normal 熵，未做 tanh 雅可比修正（近似，保留） | `actor_critic.py` |
| #6 | 🟡 P2 | `set_seed` 不完整（缺 `random`、cudnn determinism） | `train_no_rllib_mappo.py:set_seed` |

---

## 二、修复方案

### #1 → Agent-id 条件化集中式 Critic（用户选定方案）

**根因**：标准 MAPPO（Yu et al. 2021）能用单一 `V(s)` 是因为完全合作下所有 agent 共享团队 reward → 各 agent return 相等。本项目每个 agent 有独立 reward，同一 `global_state` 对应 N 个不同 return，MSE 回归到均值。

**方案**：critic 输入改为 `[global_state, one_hot(agent_id)]`，输出该 agent 专属的 `V(s, i)`。每个 agent 的 value/return 各自回归，互不冲突。

- 新增 `models/actor_critic.py:AgentConditionedCritic`（**不改** `CentralizedCritic`，避免破坏 robust / multi_adversarial 等共享该类的脚本）。
- `train_no_rllib_mappo.py`：
  - rollout 每步对每个 agent 调 `critic(s, aid)` 得到 per-agent value 存入 buffer。
  - critic 训练时把 `(state, return, agent_id)` 三元组拼接，前向 `critic(states, agent_ids)`。

### #2 → 环境区分 terminated / truncated

- `carla_multi_agent_env.py:_check_dones`：在原有 `dones` 之外，额外计算
  - `terminated[i]`：到达终点 / 自身严重碰撞 / 车辆销毁（真正终止，value 不 bootstrap）
  - `truncated[i] = dones[i] and not terminated[i]`（超时或被全局 override 连带结束，仍需 bootstrap）
- `step()` 的 `info` 增加 `info['terminated']`、`info['truncated']`（纯增量，旧脚本忽略）。

### #2+#3 → GAE 双 mask + bootstrap

`compute_gae` 改为接收显式数组，区分两种 mask：

```
next_nonterminal_value = 1 - terminated[t]   # 是否 bootstrap 价值
next_nonterminal_trace = 1 - done[t]         # 是否延续 GAE trace（回合边界一律切断）
delta = r[t] + gamma * next_values[t] * next_nonterminal_value - V[t]
gae   = delta + gamma * lambda * next_nonterminal_trace * gae
```

`next_values[t]` 构造规则（每个 agent 各自）：
- `done[t]`：用 reset 前对终止状态算出的 bootstrap 值（terminated 时该值被 mask 归零，truncated 时生效）；
- 最后一步且非 done：用 rollout 结束时 `critic(final_state, aid)` 作最终 bootstrap；
- 其余：用下一步存储的 `V[t+1]`。

→ 同时解决"超时不 bootstrap"和"rollout 尾部当终止"。

### #4 → buffer 存 pre-tanh z

- `actor_critic.py` 新增 `sample_action_z`（返回 `action_tanh, log_prob, z`）与 `evaluate_actions_z(obs, z)`（直接用 z，不走 atanh）。**保留**旧 `sample_action`/`evaluate_actions` 兼容其他脚本。
- 训练脚本 buffer 存 `z`，PPO 评估用 `evaluate_actions_z`，保证未更新策略时首个 epoch ratio 严格为 1。

### #6 → 完善 set_seed

补 `random.seed`、`torch.backends.cudnn.deterministic=True`、`benchmark=False`。

### #5

保留近似（tanh-squashed Gaussian 的精确熵无闭式解，业界普遍用 base 熵近似）。仅在文档记录。

---

## 三、影响范围与兼容性

| 文件 | 改动 | 兼容性 |
|------|------|--------|
| `models/actor_critic.py` | 新增 `AgentConditionedCritic`、`sample_action_z`、`evaluate_actions_z` | ✅ 纯新增，旧接口不变 |
| `environment/carla_multi_agent_env.py` | `_check_dones` 计算 terminated/truncated，`step` info 增键 | ✅ 纯增量 |
| `training/train_no_rllib_mappo.py` | 重写 GAE / buffer / critic 调用 / set_seed | ⚠️ 自洽；checkpoint critic 类型变更（eval 脚本只加载 actor，不受影响） |
| `training/train_adversarial_ppo.py` | 修 #2/#3/#4（单体 PPO，无 #1） | ⚠️ 自洽 |

### 第二批迁移（2026-06-08 已完成）

按相同模式迁移到其余 MAPPO 变体脚本：

| 文件 | #1 | #2 | #3 | #4 | #6 | 备注 |
|------|----|----|----|----|----|------|
| `train_no_rllib_mappo_robust.py` | ✅ | ✅ | n/a | ✅ | ✅ | 每 rollout 收集完整 episode，无 rollout 中途截断（#3 不适用） |
| `train_no_rllib_mappo_robust_v2.py` | ✅ | ✅ | ✅ | ✅ | ✅ | 已有 per-episode GAE+bootstrap，但旧实现 `mask=1-done` 把截断 bootstrap 抹掉，本次用 terminated 修正 |
| `train_multi_adversarial_mappo.py` | n/a | ✅ | ✅ | ✅ | ✅ | **联合(团队)奖励 + 单一共享 V**，G_i 相等无冲突 → #1 不适用，保留 `CentralizedCritic` |
| `train_multi_adversarial_mappo_v2.py` | ✅ | ✅ | ✅ | ✅ | ✅ | per-agent advantage + 共享 V → #1 适用，按**对抗团队内索引**条件化 |

关键设计点：
- #1 的 agent-id 条件化在 multi_adversarial_v2 中按"对抗团队内索引 0..M-1"（`adv_index`）条件化，`num_agents=len(adversarial_ids)`。
- multi_adversarial_mappo(v1) 的 #2 用"场景级 terminated"：任一 agent 真终止（碰撞/到达）→ 场景终止不 bootstrap；无人终止（纯超时）→ 截断需 bootstrap。
- robust_v2 / multi_v2 的隐藏 bug：原 per-episode bootstrap 逻辑被 `mask=1-done` 抵消（env 在 max_steps 也置 done=True），本次通过引入 terminated 双 mask 真正生效。
- 旧 checkpoint 的非条件化 critic 与新结构不兼容，加载处已加 try/except 跳过（actor 不受影响）。

**仍未纳入**：`train_no_rlib_sharedpolicy.py`（自带独立的 Actor/critic 定义与 shared-policy 结构，与本套模型接口差异较大，如需可单独处理）。

---

## 四、验收

1. `python -c "from models.actor_critic import Actor, CentralizedCritic, AgentConditionedCritic, PerAgentCritic; print('OK')"`
2. 静态 import 两个目标训练脚本无语法错误。
3. （需 CARLA 运行）跑若干 update，确认：
   - 首个 PPO epoch ratio≈1（#4 生效）；
   - critic value 不再在 rollout 尾 / 超时回合系统性塌向低值（#2/#3 生效）；
   - 各 agent value 估计可区分（#1 生效）。
