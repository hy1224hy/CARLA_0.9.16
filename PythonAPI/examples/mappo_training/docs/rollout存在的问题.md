## 一、Buffer 容量问题

当前每次 update 收集的数据量是：

```
4 episodes × 最多1000步 × 2 agents = 最多 8000 条 Transition
```

episode的更新原则：
- 随机选一个场景 JSON，调用 `env.set_scenario()` → `env.reset()`。
- `reset()` 内部：先销毁旧车辆和传感器，再重新 `_spawn_vehicles()`，初始化 `prev_distances`、`reached_target`，获取初始观测。
- 
但实际上 episode 很少跑满1000步（碰撞或到达终点会提前结束），真实数据量可能只有几百到两千条。用这么少的数据跑10个 PPO epoch、minibatch=256，**有效梯度更新次数极少，且同一条数据被反复使用导致 off-policy 偏差积累**。

4个 episode 的样本量根本无法覆盖状态空间，优势估计的方差极高，每次更新的梯度方向噪声很大。

整个生命周期是：**新建 → 收集4个episode → PPO更新 → 丢弃**，循环往复。没有 `buffer.update()` 或优先级采样，也没有经验回放。这是标准 on-policy PPO 的做法，每次更新后旧数据就过期了，因为策略已经改变。

PPO 更新阶段对同一批数据做了 `ppo_epochs=10` 轮重复利用，但这仍属于同一批数据内的 mini-batch 迭代，不是跨 update 的经验复用。


global state是怎么存的？

---

## 二、场景切换的灾难性影响

```
阶段切换
  → 新 buffer 全是新场景数据
  → 策略梯度完全由新场景主导
  → 旧场景的经验全部丢失
  → 网络权重向新场景过拟合
  → 下次碰到旧场景时性能崩塌
  → TensorBoard 上表现为剧烈震荡
```

并且当前的 `random.choice` 在同一 update 内只抽4个 episode，极小的样本下场景分布几乎是随机的——有可能4个 episode 全是同一个场景，也可能全不同，这让训练信号本身就不稳定。

---

## 三、改进方向

针对这两个问题，可以从以下几个角度入手：

**1. 引入 Replay Buffer，解耦数据收集与策略更新**

从 on-policy 转向 off-policy 或混合式，保留一个固定大小的循环缓冲区（如 50000 条），每次 update 从中随机采样，而不是只用当前 rollout 的数据。代价是需要引入重要性采样权重（IS weight）来修正分布偏差，或者改用 SAC/TD3 这类天然支持 replay 的算法。

如果坚持 PPO 框架，可以考虑**混合 buffer**：保留最近 N 个 update 的数据，用 clip ratio 自然过滤掉过时样本（ratio 偏离1太远的样本贡献趋近于0）。

**2. 场景分层采样，保证每次 update 的场景多样性**

```python
# 改为分层采样，保证每个场景至少出现一次
episodes_per_scenario = max(1, rollout_episodes // len(available_scenarios))
for scenario in available_scenarios:
    for _ in range(episodes_per_scenario):
        ...
```

同时对成功率低的场景**上采样**，利用已有的 `scenario_success_history` 做加权采样：

```python
weights = [1.0 - curriculum.get_scenario_success_rate(s) + 0.1
           for s in available_scenarios]
scenario = random.choices(available_scenarios, weights=weights, k=1)[0]
```

**3. 课程切换时做软过渡，而不是硬切换**

阶段切换时不立即丢弃旧场景，而是用概率混合：

```python
# 阶段1场景保留30%权重，新场景占70%
old_stage_weight = 0.3
new_scenarios = stage2_scenarios
old_scenarios = stage1_scenarios
```

或者更简单地，把 `warmup_updates` 和 `full_scenario_updates` 之间的过渡区间拉长，让新场景缓慢引入。

**4. 增大 rollout_episodes，稳定优势估计**

在不改算法的前提下，把 `rollout_episodes` 从4提高到16甚至32，用更多数据平均掉场景随机性带来的方差。代价是每次 update 的墙钟时间变长，但训练稳定性会有明显提升。

---

总结来说，当前设计是一个"数据量少 + 无记忆 + 硬切换"的三重叠加，2000回合训练效果差是必然结果。优先级最高的改动是**增大有效 buffer 容量**和**场景加权采样**，这两项改动最小但收益最大。