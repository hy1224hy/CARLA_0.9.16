# MAPPO Robust v2 不收敛 — 诊断与修复计划

## Context

[train_no_rllib_mappo_robust_v2.py](../training/train_no_rllib_mappo_robust_v2.py) 训练不收敛。用户观察到的症状：

- `success_rate` 大幅震荡，TensorBoard 上的成功率"看起来不对"
- agent 能开起来但效果差、成功率低且随机
- PPO `approx_kl` 偏大、`clip_frac` 高
- 各项 loss/指标都在波动，没有明显下降或上升趋势

目标：把训练拉到能稳定收敛、TB 指标可信的状态。按"零成本→需重训→深度改造"分阶段给出修复方案。

---

## 诊断：6 个相互叠加的根因

### R1【P0】`ep_success` 把 timeout 当成功 → 指标 + 课程双双污染

[train_no_rllib_mappo_robust_v2.py:591-594](../training/train_no_rllib_mappo_robust_v2.py#L591-L594)

```python
if all(dones.values()):
    ep_success = not any(info.get('collisions', {}).values())
```

环境在 `current_step >= max_steps` 时把所有 agent 的 `done` 置 True（[carla_multi_agent_env.py:1409](../environment/carla_multi_agent_env.py#L1409)）。此时若没碰撞，`ep_success=True`，但其实 agent 根本没到终点。

直接后果：
- TB 上 `success_rate_window` 失真 → 用户感觉"成功率不对"
- `CurriculumLearning.scenario_success_history` 被污染 → "困难场景"被误判为已掌握，加权采样反而降权 → 课程学习反向工作
- `weighted` 采样 + 错误成功率 = 训练数据分布持续动荡，是震荡的主因之一

### R2【P0】严重碰撞广播 done 时，"被牵连" agent 的 GAE bootstrap 被错算成 0

[train_no_rllib_mappo_robust_v2.py:597-607](../training/train_no_rllib_mappo_robust_v2.py#L597-L607)

```python
episode_truncated = (ep_length >= args.max_steps)
...
if episode_truncated:
    last_gs = env.get_global_state(); ... add_episode_bootstrap(aid, last_gs)
else:
    ... add_episode_bootstrap(aid, None)  # bootstrap = 0
```

环境在任一 agent 发生 severe collision 时把全员 `done=True`（[carla_multi_agent_env.py:1437-1447](../environment/carla_multi_agent_env.py#L1437-L1447)）。对未撞那台车而言，这是"被截断"而非"自然终止"，但 v2 的 `episode_truncated` 只检测 max_steps 截断，导致未撞 agent 的最后一步 GAE bootstrap = 0，等于错误地告诉 Critic：episode 在这里终止了、未来价值为 0。

直接后果：
- Critic 系统性低估 V(s)，advantage 估计偏负
- 未撞 agent 的策略学到"过度防御"，episode 越长越被惩罚
- v2 changelog 里宣称的 "GAE 截断 bootstrap 修复" 只覆盖了 max_steps 这一种情况

### R3【P0】MAPPO Critic：单个 V(global_state) 拟合多个 agent 不同 return → advantage 噪声极大

[train_no_rllib_mappo_robust_v2.py:537-538, 674-675](../training/train_no_rllib_mappo_robust_v2.py#L537-L538)

```python
with torch.no_grad():
    global_value = float(critic(global_state_t).item())   # 每步只算一次 V
...
critic_states.append(state_arr)   # 同一个 global_state 被两个 agent 各加一遍
critic_returns.append(ret_arr)    # 但每个 agent 的 return 不同
```

每个 step `global_state` 是共享的，但每个 agent 的 `reward` 和 `return` 不同。`per_agent_tensors[aid]['state']` 把同一个 state 复制两份放进 critic 训练集，目标值分别是两个 agent 的 return → critic 实际在拟合**两个 return 的平均**。

直接后果：
- Critic 拟合不了任何单 agent 的真实价值，`explained_variance` 长期接近 0
- advantage = per-agent reward 累积 - 平均 V → 高方差、有偏
- Actor 看到的优势信号充满噪声 → `approx_kl` / `clip_frac` 爆表（与用户观察一致）

### R4【P1】奖励量级失衡：progress 主导，碰撞信号被淹没

[carla_multi_agent_env.py:1186, 1271](../environment/carla_multi_agent_env.py#L1186)

```python
progress_reward = progress * 20.0 * (0.3 + 0.7 * efficiency)   # 每步可达 +8 ~ +20
collision_penalty = -100.0 if collision else 0.0               # 一次 -100
success_reward    = 100.0; time_bonus = (max-cur)/max * 50.0   # 终点 +100~+150
```

按 200 步正常行驶估算：累积 progress ≈ +1600～+4000，碰撞 -100 在 episode return 中只是个零头，Critic 学不到"靠近碰撞=危险"。

### R5【P1】PPO 更新过激进：lr × ppo_epochs × clip_eps × log_std 初始值 共同放大每步 KL

- `lr_actor=3e-4`、`ppo_epochs=5`、`clip_eps=0.2`、`Actor.log_std=0` (std=1.0 → tanh 后输出极易饱和到 ±1)
- 没有 **target KL 早停**：每次更新固定跑满 5 个 epoch，KL 失控也不停
- 没有 **lr 衰减**：训练后期 lr 仍然 3e-4，相对策略已稳定的尺度太大

用户观察到的 `approx_kl 偏大、clip_frac 高` 直接对应这一组。

### R6【P1】Action 互斥在 Actor 采样之外做 → log_prob 与实际动作不一致

[train_no_rllib_mappo_robust_v2.py:108-121](../training/train_no_rllib_mappo_robust_v2.py#L108-L121)

Actor 输出 3D Gaussian `(throttle, steer, brake)`，但 `action_tanh_to_env` 之后把 throttle/brake 强制互斥。存进 buffer 的 `log_prob` 是原始 3D 动作的概率，PPO ratio 计算与实际生效动作有系统性偏差。

---

## 修复计划（分阶段执行，按性价比排序）

### Phase A：零重训成本的指标 + 训练稳定性修复（半天～1 天）

目标：先把"假信号"和"PPO 步长失控"两个最便宜的洞堵上，让接下来任何实验的结果都可信。

**A1. 修 `ep_success` 误判 + 拆分三类终止指标**

```python
ep_success_to_goal = all(env.reached_target.get(aid, False) for aid in agent_ids)
ep_collision = any(
    env.collision_events.get(aid, False) or env.crosswalk_invasion_events.get(aid, False)
    for aid in agent_ids
)
ep_timeout = (ep_length >= args.max_steps) and not ep_success_to_goal and not ep_collision
ep_success = ep_success_to_goal
```

TB 上新增 3 个独立指标：`episode/rate_success_to_goal`、`episode/rate_collision`、`episode/rate_timeout`。

**A2. 修 collision 广播 done 时的 GAE bootstrap**

"自然终止" 仅限于：该 agent 自己撞了、或自己到了 goal。其它一切（max_steps 截断、被别人撞牵连）都用 V(last_global_state) bootstrap。

**A3. PPO 步长可控化**

- Actor `log_std` 初始值改 `-0.5`（std≈0.6）
- 新增 `--target-kl`（默认 0.02），单个 agent 当 epoch 内 `approx_kl > 1.5 * target_kl` 触发 break
- 新增 lr 线性衰减
- `--clip-eps` 默认 0.2 → 0.15

**A4. 最佳 checkpoint 跟踪**

每个 update 结束后维护 `best_success_rate`，保存 `best.pt`。

**A5. 多 seed**

`--seed 42 / 123 / 2024` 各跑 1 次，TB 路径用 seed 区分。

---

### Phase B：必须重训的算法 / 奖励结构性修复（2～3 天）

**B1. Critic 改造（R3）**——二选一：

- 方案 B1-a（推荐，改动小）：critic_states 去重 + team-reward critic target
- 方案 B1-b（更标准）：CentralizedCritic 多头 `[V_0, V_1]`

**B2. 奖励重平衡（R4）**

| 项 | 当前 | 建议 |
|---|---|---|
| `progress * 20.0` | 8~20/步 | `progress * 5.0`，1~5/步 |
| collision penalty | -100 | -50 |
| safety_lidar 权重 | 0.3 | 1.0 |
| success_reward + time_bonus | 100 + 50 | 50 + 25 |
| TTC_WEIGHT | 2.0 | 5.0 |

或在训练脚本里加 reward normalization（running mean/std）。

**B3. Action 空间改 2D（R6）**

Actor `act_dim=3` → `act_dim=2 (longitudinal, steer)`，longitudinal>0 映射 throttle，<0 映射 brake。

---

### Phase C：超参 + 数据量微调（重训时同步）

- `--rollout-episodes 16 → 24`
- `--ppo-epochs 5 → 3`
- `--ent-coef 0.01 → 0.005` + 线性衰减到 0
- `--minibatch-size 256 → 512`
- 课程：`--warmup-updates 20`、`--full-scenario-updates 80`

---

## 验证 / 验收

1. **Phase A 完成后**：30 update sanity，TB 上：
   - `rate_success_to_goal` + `rate_collision` + `rate_timeout` ≈ 1.0
   - `approx_kl` < 0.03，`clip_frac` < 0.3
   - `rate_success_to_goal` 近似单调上升
2. **Phase B 完成后**：100 update × 3 seed：
   - `explained_variance` > 0.5
   - 3 seed 的 `rate_success_to_goal` 在 80 update 时 mean > 0.5、std < 0.15
3. 用 [compare_reward_versions.py](../evaluation/compare_reward_versions.py) 跑修复前后 best.pt 对比。

---

## 不在本计划范围内

- Held-out 测试集 / 跨地图泛化
- IPPO / shared-policy MAPPO baseline 对照
- Co-evolution 对抗训练
- 超参 Optuna sweep
- LiDAR ray_length 调整



当前成功率/Done 完整链路梳理
1. Env 侧 — _check_dones()（每个 step 末尾）
先对每个 agent 单独判定：


if in_goal_region:                          # 车辆中心在 6m × 3.5m 矩形内
    self.reached_target[i] = True           # ← sticky，只在 reset 清零
    dones[i] = True
elif current_step >= max_steps or is_collided:
    dones[i] = True
else:
    dones[i] = False
然后全员强制 done 的两种情况：

条件	行为	打印
all(reached_target)	全员 dones=True	"所有智能体都已到达终点，结束本回合"
any(collision/crosswalk AND not reached)	全员 dones=True	"检测到严重碰撞，结束本回合"
2. Train 脚本侧 — rollout 循环

for step in range(args.max_steps):
    ...
    next_obs, rewards, dones, info = env.step(action_dict)
    ...
    if all(dones.values()):     # ← 关键：所有 agent dones 都 True 才 break
        break
Break 的唯一条件是 all(dones.values())，不直接看 reached_target。

3. Episode 末分类（train 脚本:691-720）

ep_success_to_goal = all(env.reached_target[aid] for aid in agent_ids)
ep_collision_per_agent = ep_collision_sticky                 # rollout 里逐步累积
ep_collision = any(...)
ep_timeout = (ep_length >= args.max_steps) and not ep_success_to_goal and not ep_collision
4. 累计统计
字段	计算
total_successes	每个 ep_success_to_goal=True 时 +1
cum_success	total_successes / total_episodes
window_successes (列表)	当前 update 16 个 episode 的 0/1 标志
TB episode/rate_success_to_goal	np.mean(window_successes)
"到达终点但回合没停" — 三个可能原因
原因 A（最大概率，设计预期）：一个 agent 到了，另一个还没
agent1 到点：reached_target[1]=True, dones[1]=True
agent0 还在跑：dones[0]=False
all(dones.values()) = False → train 脚本不 break，继续跑
CARLA 里看到 agent1 停在终点（hand_brake 锁死），agent0 还在挪
等 agent0 也到/撞/超时，才结束
这是 MAPPO 多 agent 训练的合理设计：必须所有 agent 完成才算一个完整 episode。但代价是：success 严格要求双双到达，所以即使 leader 频繁到点，success 仍可能长期 0%。

原因 B（潜在 bug）：reached_target 后被惯性滑出 goal_zone
行 1527-1536 的 if/elif/else 结构有个漏洞：


if in_goal_region:           # 第一帧进 zone，reached_target=True, dones=True
    self.reached_target[i] = True
    dones[i] = True
elif max_steps or is_collided:
    dones[i] = True
else:
    dones[i] = False         # ← 滑出 zone 后这里执行：reached_target 仍 True，但 dones 又变回 False
滑出 goal_zone 那一帧 dones[i] 会从 True 变回 False，导致 train 脚本不 break，episode 继续。直到 all(reached_target) 全员都到点才走第二层判定强制 done。

刚加的 hand_brake + 物理速度清零修复应该把这个降到很罕见，但不严谨。

原因 C：env 打印了"所有智能体都已到达"但你看到延迟
print 是在 _check_dones 末尾执行，return dones 后回到 step()，再 return 回 train 脚本，train 脚本的 print 在下一个 update 边界才出现，中间有视觉延迟。