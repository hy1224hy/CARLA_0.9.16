# 实现计划：将单对抗智能体 PPO 升级为多对抗智能体 MAPPO

## 目标

将当前 `train_adversarial_ppo.py`（单对抗智能体 + 标准 PPO）升级为**多对抗智能体 + MAPPO（CTDE）**训练。所有对抗智能体使用统一的奖励函数。

核心变更：对抗团队训练架构从 `单Agent PPO` → `多Agent MAPPO (CTDE)`。

---

## 总体架构对比

```
当前架构                               目标架构
┌──────────────────┐                 ┌──────────────────────────┐
│ Agent 0 (合作)    │ 冻结            │ Team Cooperative (冻结)    │
│ Agent 1 (合作)    │ 冻结            │  Agent 0, Agent 1         │
│ Agent 2 (对抗)    │ 单PPO训练       │                           │
│  - Actor(19→3)   │                │ Team Adversarial (MAPPO)  │
│  - Critic(19→1)  │                │  Agent 2, Agent 3, ...    │
└──────────────────┘                 │  - 各独立Actor(19→3)      │
                                     │  - 共享CentralizedCritic  │
                                     │  - 统一对抗奖励函数       │
                                     └──────────────────────────┘
```

---

## Phase 1: 统一模型定义（消除代码重复）

### 问题
`Actor`/`Critic`/`CentralizedCritic` 在 3 个训练脚本中各自重复定义，且 `evaluate_actions` 返回值不一致（有的2个有的3个）。

### 方案
创建 `models/actor_critic.py`，提供统一实现：

```python
class Actor(nn.Module):
    """2层MLP (128, Tanh) → mu_head + log_std, 用于 tanh-squashed Gaussian 策略"""
    def forward(obs) -> (mu, log_std)
    def sample_action(obs) -> (action_tanh, log_prob)
    def evaluate_actions(obs, action_tanh) -> (log_prob, entropy)

class CentralizedCritic(nn.Module):
    """输入全局状态(所有agent拼接) → 标量 value, CTDE用"""
    
class PerAgentCritic(nn.Module):
    """输入单agent观察(19维) → 标量 value, 单agent训练用"""
```

### 需更新的文件

| 文件 | 变更 |
|------|------|
| `models/actor_critic.py` | **新建**，统一模型定义 |
| `models/__init__.py` | 添加导出 |
| `training/train_no_rllib_mappo.py` | 移除内联 Actor/CentralizedCritic 定义，改为 import |
| `training/train_no_rllib_mappo_robust.py` | 同上；修复 `evaluate_actions` 3返回值→2返回值 |
| `training/train_adversarial_ppo.py` | 移除内联 Actor/Critic 定义，改为 import |
| `evaluation/evaluate_adversarial_ppo.py` | 移除内联 Actor，改为 import |
| `evaluation/evaluate_robustness.py` | 同上 |

---

## Phase 2: 环境层改造

### 2.1 替换硬编码对抗ID

**当前** (`carla_multi_agent_env.py` L1034)：
```python
ADVERSARIAL_AGENT_ID = 2
```

**改为**：构造函数接受 `adversarial_agent_ids: List[int]`，`_compute_rewards()` 中遍历该列表。

### 2.2 `_find_straight_road()` 通用化

当前只有 `num_agents==3` 有特殊位置分配。改为按 `adversarial_agent_ids` 的角色信息进行分配：

- 合作车均匀分布在道路前段（20%~50%位置）
- 对抗车穿插在合作车间隙中，略偏前方便于干扰

### 2.3 对抗团队全局状态

对抗团队集中式 Critic 所需。复用现有格式（每agent 6维：x, y, speed, sin(yaw), cos(yaw), dist_to_goal），可仅包含对抗团队的智能体状态。

---

## Phase 3: 新建多对抗 MAPPO 训练脚本（核心）

新建 `training/train_multi_adversarial_mappo.py`。

### 3.1 关键差异

| 维度 | 旧 `train_adversarial_ppo.py` | 新 `train_multi_adversarial_mappo.py` |
|------|------|------|
| 对抗智能体数 | 固定 1 个 (Agent 2) | 可配置 M 个 (--num-adversarial) |
| 对抗训练算法 | 单Agent PPO | MAPPO (CTDE) |
| Critic | `PerAgentCritic(obs_dim=19)` | `CentralizedCritic(state_dim)` 共享 |
| RolloutBuffer | `List[Transition]` 单列表 | `Dict[int, List[Transition]]` 按agent索引 |
| PPO 更新 | 对抗agent独自更新 Actor+Critic | 集中式 Critic 更新（合并所有对抗agent数据）+ 各 Actor 独立更新 |

### 3.2 训练循环

```
for update in range(updates):
    # ── Rollout ──
    for step in range(rollout_steps):
        # 合作团队推理 (deterministic, frozen)
        for aid in cooperative_ids:
            action = pretrained_actors[aid](obs)  # mu → tanh

        # 对抗团队推理 (stochastic, training)
        global_state = env.get_adversarial_global_state()
        for aid in adversarial_ids:
            action_tanh, logp = adversarial_actors[aid].sample_action(obs)
            value = adversarial_critic(global_state)
            cache[aid] = (obs, global_state, action_tanh, logp, value)

        # 环境 step
        next_obs, rewards, dones = env.step(actions)

        # 计算每个对抗智能体的奖励（统一公式）
        for aid in adversarial_ids:
            adv_reward = rewards[aid]  # 环境层已计算（含cut_in+blocking）
            adv_reward += coef * (-avg(rewards[0], rewards[1])) * 0.01
            buffer[aid].add(Transition(..., adv_reward, ...))

    # ── MAPPO 更新 (对抗团队) ──
    # GAE (各对抗agent独立计算)
    for aid in adversarial_ids:
        advantages[aid], returns[aid] = compute_gae(buffer[aid])

    # 集中式 Critic 更新
    all_states  = concat([buffer[aid].global_states for aid in adversarial_ids])
    all_returns = concat([buffer[aid].returns for aid in adversarial_ids])
    for epoch in range(ppo_epochs):
        for mb in minibatches of (all_states, all_returns):
            critic_update(mb)

    # 各 Actor 独立更新
    for aid in adversarial_ids:
        for epoch in range(ppo_epochs):
            for mb in minibatches of buffer[aid]:
                ppo_update_actor(mb)
```

### 3.3 奖励函数（所有对抗智能体统一）

与当前 `_compute_adversarial_reward()` 完全相同：

```python
adv_env_reward = base_driving (progress*0.3 + speed*0.3 + centering*0.2 + ...)
               + cutting_in_reward   # 侧方近距离行驶
               + blocking_reward     # 前方近距离阻挡

adv_final_reward = adv_env_reward + adversarial_coef * (-avg(cooperative_team_rewards)) * 0.01
```

### 3.4 命令行参数

```
--pretrained-ckpt PATH        # 预训练合作团队checkpoint (必需)
--num-cooperative N           # 合作智能体数 (默认2)
--num-adversarial M           # 对抗智能体数 (默认2)
--freeze-cooperative BOOL     # 是否冻结合作团队 (默认True)

# 其余参数复用现有 --updates, --rollout-steps, --ppo-epochs, --lr-*, --gamma, --gae-lambda 等
```

### 3.5 Checkpoint 格式

```python
{
    "update": update,
    "cooperative_actors": {aid: state_dict, ...},     # 合作团队 Actor
    "adversarial_actors": {aid: state_dict, ...},     # 对抗团队各 Actor
    "adversarial_critic": state_dict,                 # 对抗团队集中式 Critic
    "adversarial_actor_optims": {aid: state_dict, ...},
    "adversarial_critic_optim": state_dict,
    "args": vars(args),
}
```

---

## Phase 4: 评估脚本更新

### 4.1 `evaluation/evaluate_adversarial_ppo.py`

- 支持加载新 checkpoint 格式（`"adversarial_actors"` dict 和 `"adversarial_actor"` 单个，兼容旧格式）
- 支持评估 M 个对抗智能体
- 输出：每个对抗智能体的平均回报、合作团队平均回报

### 4.2 `evaluation/evaluate_robustness.py`

- 适配多对抗智能体评估
- 新增对比：不同对抗数量下的自车成功率

---

## 执行顺序

```
Phase 1 (模型统一)
    │
    ├─→ Phase 3 (多对抗MAPPO训练脚本)
    │       │
    │       └─→ Phase 4 (评估更新)
    │
Phase 2 (环境改造)
    │
    └─→ Phase 3
```

Phase 1 和 Phase 2 无相互依赖，可并行进行。Phase 3 是核心交付物，依赖前两者。

---

## 验证

### 每个 Phase 完成后的检查

**Phase 1**：
```bash
cd PythonAPI\examples\mappo_training
python -c "from models.actor_critic import Actor, CentralizedCritic; print('OK')"
python -m training.train_no_rllib_mappo --render
```

**Phase 2**：
```bash
python -c "
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
env = CarlaMultiAgentEnv(num_agents=4, adversarial_agent_ids=[2,3], spawn_vehicles=False)
print('agents:', env.num_agents, 'adversarial:', env.adversarial_agent_ids)
"
```

**Phase 3**：
```bash
python -m training.train_multi_adversarial_mappo \
    --pretrained-ckpt ./logs/minimal_mappo_ckpts/ckpt_100.pt \
    --num-cooperative 2 --num-adversarial 2 \
    --curriculum-learning --render
```

观察点：
- 2 个对抗智能体是否均正常行驶
- 对抗团队总奖励是否收敛
- 合作车（自车）是否经历"被干扰→学习应对"的过程

**Phase 4**：
```bash
python -m evaluation.evaluate_adversarial_ppo \
    --pretrained-ckpt <path> --adversarial-ckpt <path> --episodes 100
```
