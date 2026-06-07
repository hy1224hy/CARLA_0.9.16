# TensorBoard 指标记录优化

**日期**: 2026-05-14  
**状态**: ✅ 已完成  
**相关文件**: `training/train_no_rllib_mappo_robust.py`, `environment/carla_multi_agent_env.py`

---

## 一、修改背景

### 修改前的问题

| 问题 | 严重程度 | 具体表现 |
|------|----------|----------|
| Episode 丢弃 | **严重** | 只有 `all(dones)==True` 的 episode 被记录，max_steps 截断的 episode 的回报被静默丢弃，`avg_return` 基于极小样本（甚至为空→显示 0） |
| 成功率累计 | **严重** | 全程累计不清零，到 100 个 update 时近期改善几乎不动曲线，看不出"最近学得怎样" |
| 无 PPO 诊断 | **严重** | 没有 loss/entropy/KL/clip_frac，训练是否在发散、策略是否探索崩溃完全不可知 |
| 场景指标水平线 | **中等** | 场景成功率也是累计值，且采样稀疏，TB 上几乎是一条水平线 |
| 环境 TB 混乱 | **中等** | 环境自己创建 TB writer，x 轴 `current_step` 每 episode 归零，曲线重叠不可读；与训练脚本的 TB writer 并存产生两套日志 |
| 碰撞检测空壳 | **严重** | `_check_collisions()` 永远返回 `False`，导致 `ep_success` 判断恒为 `True`，成功率虚高为 100% |

---

## 二、修改内容

### 2.1 环境修复 (`carla_multi_agent_env.py`)

- 新增 `disable_tensorboard` 参数，训练脚本传入 `True` 避免双写
- 新增 `_total_env_steps` 全局计数器，跨 episode 不重置，保证 TB x 轴单调递增
- `_log_to_tensorboard()` 和 `reset()` 中的 episode 日志均改用 `_total_env_steps`
- **修复 `_check_collisions()` 空壳方法**（2026-05-17）：原方法永远返回 `{0: False, 1: False}`，导致训练脚本中 `ep_success = not any(info['collisions'].values())` 永远为 `True`，所有碰撞终止的 episode 都被误判为成功。改为读取传感器回调实际设置的 `self.collision_events` 和 `self.crosswalk_invasion_events`

### 2.2 训练循环改造 (`train_no_rllib_mappo_robust.py`)

- 每个 episode 无论终止方式都记录回报/episode 长度/是否成功
- PPO 更新循环中收集每 epoch 的 loss/entropy/approx_kl/clip_frac/grad_norm，跨 epoch 取均值后记录
- 场景指标改用课程学习器自带的滑动窗口（最近 20 次），至少采样 3 次才记录
- 新增 per-agent 回报记录和 reward 分量记录

---

## 三、TensorBoard 指标详解

所有指标 x 轴均为 `update`（训练更新次数），在 `log_interval` 的 update 上记录。

### 3.1 Episode 性能指标 (`episode/`)

这些指标回答"任务做得好不好"。

| 指标 | 含义 | 健康范围 | 异常信号 |
|------|------|----------|----------|
| `episode/avg_return` | 当前 update 窗口内所有 episode 的平均回报（不再丢弃未完成 episode） | 持续上升 | 长期不升或剧烈波动 |
| `episode/std_return` | episode 回报的标准差 | 先大后小（探索→收敛）| 持续增大→训练不稳定 |
| `episode/avg_length` | 平均 episode 步数 | 逐渐缩短（更快到达）或稳定在合理值 | 始终接近 max_steps→学不会到达；始终很短→频繁碰撞 |
| `episode/success_rate_window` | **当前 update 窗口**的成功率 | 持续上升 | 停滞在低位 |
| `episode/success_rate_cumulative` | 从训练开始累计的成功率 | 缓慢上升，作为长期趋势参考 | 不要只看这个——它严重滞后 |
| `episode/return_agent_0` | Agent 0 单独的平均 episode 回报 | 两个 agent 趋势相近 | 某个 agent 显著低于另一个→策略退化 |
| `episode/return_agent_1` | Agent 1 单独的平均 episode 回报 | 同上 | 同上 |

**使用建议**：主要看 `success_rate_window` 和 `avg_return`，用 `cumulative` 仅作长期趋势参考。`avg_length` + `success_rate_window` 组合可以区分"跑不到终点就撞了"vs"跑不到终点也跑不完"。

### 3.2 奖励分量 (`reward/`)

这些指标回答"回报由什么构成"，来自环境 `info['reward_details']` 中每个 step 的奖励分解。

| 指标 | 含义 | 正值驱动行为 |
|------|------|-------------|
| `reward/progress` | 前进奖励（沿目标方向位移 × 系数） | 鼓励朝终点移动 |
| `reward/speed` | 速度奖励（理想速度区间内为正） | 鼓励保持合理车速 |
| `reward/centering` | 居中奖励（横向偏差小为正） | 鼓励车道居中 |
| `reward/yaw` | 航向对齐奖励（车头朝向与道路方向一致） | 鼓励朝向正确 |
| `reward/collision` | 碰撞惩罚（负值，碰撞时大幅惩罚） | 惩罚碰撞 |
| `reward/lane_invasion` | 压线惩罚（负值） | 惩罚偏离车道 |
| `reward/step` | 每步固定惩罚（负值） | 施加时间压力 |
| `reward/success` | 到达终点奖励（正值，仅终点触发） | 稀疏引导到达 |
| `reward/jerk` | 急动度惩罚（加速度变化率，v2 独有） | 鼓励平滑驾驶 |
| `reward/steer_rate` | 转向速率惩罚（v2 独有） | 鼓励稳定转向 |
| `reward/safety_lidar` | 安全距离奖励（基于激光雷达，v2 独有） | 鼓励保持安全距离 |
| `reward/ttc` | 碰撞时间奖励（TTC，v2 独有） | 鼓励避免紧迫碰撞风险 |

**使用建议**：如果 `progress` 很高但 `success_rate` 很低，说明 agent 在前进但不精确；如果 `collision` 反复出现大的负值，agent 还没学会避障；`safety_lidar` 和 `ttc` 可以看出风险感知能力是否在改善。

### 3.3 PPO 训练诊断 (`ppo/`)

这些指标回答"学习过程是否健康"，是**最重要的调参依据**。

#### Critic 指标

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `ppo/critic_loss` | Value 函数的 MSE 损失 | 初期较高，逐步下降并稳定 |
| `ppo/critic_grad_norm` | Critic 梯度范数（clip 前） | < `max_grad_norm`（0.5），频繁触顶说明梯度过大 |
| `ppo/explained_variance` | Value 函数对实际回报的解释方差：`1 - Var(y_true - y_pred) / Var(y_true)` | **越接近 1.0 越好**；< 0 表示 Critic 比瞎猜还差 |

#### Actor 指标（每个 agent 独立记录）

| 指标 | 含义 | 健康范围 | 异常信号 |
|------|------|----------|----------|
| `ppo/actor_loss_{id}` | PPO clipped objective + entropy bonus 的总损失 | 初期较高，逐步下降 | 持续上升→策略退化 |
| `ppo/entropy_{id}` | 策略熵（动作分布的随机性） | 初期 ~1.0，逐步下降 | **趋近 0 = 探索崩溃**，agent 不再尝试新动作，学习停滞 |
| `ppo/approx_kl_{id}` | 新旧策略的近似 KL 散度 `mean((logp_new - logp_old)^2)` | 0.001~0.01 | > 0.05 策略更新过猛→不稳定；< 1e-5 更新太小→学不到东西 |
| `ppo/clip_frac_{id}` | PPO ratio 被 `clip_eps` 裁剪的比例 | 0.05~0.2 | > 0.5 → `clip_eps` 太小或学习率太大；接近 0 → clip 没起作用 |
| `ppo/grad_norm_actor_{id}` | Actor 梯度范数（clip 前） | < `max_grad_norm`（0.5） | 频繁触顶 → 学习率太大或网络不稳定 |

**调参指南**：

- **entropy 过快趋零** → 增大 `ent_coef`（如 0.01→0.02）或减小学习率
- **approx_kl 过大 + clip_frac 过高** → 减小学习率或减小 `clip_eps`
- **explained_variance 长期为负** → Critic 没学到东西，检查 `critic_loss` 和 `lr_critic`
- **actor_loss 震荡发散** → 减小 `lr_actor`，增大 `minibatch_size`

### 3.4 场景指标 (`scenarios/{name}/`)

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `success_rate_window` | **滑动窗口**成功率（最近 ≤20 次该场景的回放） | 各场景趋于一致且上升 |
| `sample_count` | 该场景被采样的累计次数 | 均匀增长说明课程学习进展正常 |

**使用建议**：对比不同场景的成功率可以识别"哪些场景是瓶颈"。如果某个场景成功率始终很低，考虑：
- 该场景是否难度过高（检查 JSON 配置的起始位置）
- 是否需要增加该场景的采样权重

---

## 四、TB 层级总览

```
episode/
  avg_return                    # 窗口平均 episode 回报
  std_return                    # 回报标准差
  avg_length                    # 平均 episode 步数
  success_rate_window           # 窗口成功率（主要看这个）
  success_rate_cumulative       # 累计成功率（长期参考）
  return_agent_0                # Agent 0 平均回报
  return_agent_1                # Agent 1 平均回报

reward/
  progress / speed / centering / yaw / collision / lane_invasion / step / success
  jerk / steer_rate / safety_lidar / ttc    # v2 独有

ppo/
  critic_loss                   # Critic MSE
  critic_grad_norm              # Critic 梯度范数
  explained_variance            # Value 解释方差
  actor_loss_agent0             # Agent 0 Actor 损失
  entropy_agent0                # Agent 0 策略熵
  approx_kl_agent0              # Agent 0 近似 KL
  clip_frac_agent0              # Agent 0 裁剪比例
  grad_norm_actor_agent0        # Agent 0 梯度范数
  actor_loss_agent1             # (Agent 1 同理)
  entropy_agent1
  approx_kl_agent1
  clip_frac_agent1
  grad_norm_actor_agent1

scenarios/{name}/
  success_rate_window           # 滑动窗口成功率
  sample_count                  # 采样次数
```

---

## 五、使用时注意事项

1. **TB 日志位置**：训练脚本的 TB 日志在 `--save-dir` 下的 `tb_logs/` 子目录；环境的独立 TB（如未禁用）在 `logs/carla_training_reward/`。
2. **查看命令**：`tensorboard --logdir PythonAPI/examples/mappo_training/logs/`
3. **x 轴一致性**：训练脚本的所有指标 x 轴为 `update`（1, 2, 3...）；环境的 TB（如启用）x 轴为 `_total_env_steps`（全局仿真步数）。
4. **log_interval 影响**：只有 `update % log_interval == 0` 时记录，频率由该参数控制。
