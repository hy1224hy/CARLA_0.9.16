# MAPPO 多智能体样本处理与 active-mask 原理

> 用途：解释 MAPPO 如何分别处理多个智能体，并将其迁移到本项目的 CARLA 多车训练；同时作为毕业论文“方法基础”和“工程实现”部分的写作素材。
>
> 当前状态：`P0-00C` 已于 2026-07-22 完成代码修改和 mock/unit 验收；当前基础策略唯一入口仍为 `training/train_no_rllib_mappo_robust_v2.py`。真实 CARLA smoke 统一留在 `P0-00G`，因此本文中的测试结果不等价于真实仿真性能结论。

## 1. 先给出核心结论

MAPPO 中的基本训练单位不是“某一帧只有一条联合样本”，而是有效的 **agent-time transition**：智能体 (i) 在时刻 (t) 确实处于可决策状态，依据自己的局部观测 (o_t^i) 选择动作 (a_t^i)，环境随后返回该智能体的奖励和生命周期边界。

因此，一个环境步可以产生两条、只产生一条或不产生训练样本：

- 两辆车都活跃：产生两条 transition；
- 一辆车已经到点或碰撞失活，另一辆仍在运行：只为仍活跃车辆产生一条 transition；
- 全局回合已经结束：不再产生 transition，应退出 rollout。

`active_mask` 回答的是“这个智能体在当前状态能否作出下一次决策”；`terminated/truncated` 回答的是“刚完成的 transition 为什么形成边界”。两类信号位于 transition 的两端，不能互相替代。

## 2. 从 PPO 到 MAPPO

### 2.1 单智能体 PPO

PPO 是 on-policy actor-critic 算法。Actor 根据当前观测给出动作分布，Critic 估计当前状态的期望折扣回报。采样阶段保存旧策略的动作对数概率，更新阶段使用概率比

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{\theta_{old}}(a_t\mid o_t)}
$$

构造 clipped surrogate objective：

$$
L^{clip}(\theta)=\mathbb{E}_t\left[
\min\left(
\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

PPO 的关键约束是：更新所使用的 ((o_t,a_t,\log\pi_{old},\hat A_t)) 必须来自旧策略真正执行过的有效决策。人为补出的全零观测和无效动作不满足这个条件。

### 2.2 MAPPO 的 CTDE 结构

MAPPO 把 PPO 扩展到多智能体场景，典型结构是集中训练、分散执行（Centralized Training with Decentralized Execution, CTDE）：

- 执行时，第 (i) 个 Actor 只能使用自身可获得的局部观测 (o_t^i)，输出 (a_t^i\sim\pi_{\theta_i}(\cdot\mid o_t^i))；
- 训练时，Critic 可以使用所有车辆的联合状态 (s_t)，以缓解其他智能体行为变化造成的非平稳性；
- 每个智能体仍需要自己的奖励、边界和 advantage。集中式 Critic 不等于把所有车辆强行合成一条 return。

对本项目更合适的价值函数写法是

$$
V_\phi(s_t,i),
$$

其中 (i) 是目标智能体标识。同一联合状态下，两辆车可能距离终点不同、即时奖励不同、最终结局不同，因此它们的回报目标也可能不同。将 agent ID 作为 Critic 条件，可以共享参数，同时避免让一个无条件标量 (V(s_t)) 拟合相互冲突的 per-agent return。

### 2.3 参数共享不是 MAPPO 的必要定义

同质智能体常共享 Actor 参数，这有利于样本复用和排列泛化；异质智能体或角色差异明显时也可以使用独立 Actor。MAPPO 的关键是 on-policy PPO 更新和 CTDE，而不是必须共享 Actor。

当前 `robust_v2` 的选择是：

| 模块 | 当前设计 | 含义 |
|---|---|---|
| Actor | 每辆红车一个独立 Actor | `actor[0]` 只使用 agent 0 的局部观测和有效样本，`actor[1]` 同理 |
| Critic | 一个参数共享、agent-ID 条件化的集中式 Critic | 输入联合状态和目标 agent ID，分别预测 (V(s,0))、(V(s,1)) |
| Reward/GAE | 按 agent 分开 | 每辆车使用自己的奖励序列、边界和 return，不跨 agent 传播 GAE |
| PPO batch | 按 agent 更新 Actor；有效样本合并更新 Critic | 失活车辆的填充观测不能进入任一损失 |

该结构是合理的 MAPPO 变体。是否将同质红车的 Actor 改成参数共享属于后续消融问题，不是 `P0-00C` 的修复目标。

## 3. MAPPO 如何处理不同智能体

### 3.1 每个时刻先确定“谁有决策权”

令

$$
m_t^i=\mathbb{1}[\text{agent }i\text{ 在状态 }s_t\text{ 仍可决策}].
$$

只有 (m_t^i=1) 时，才应为智能体 (i)：

1. 计算 (V_\phi(s_t,i))；
2. 从 (pi_{\theta_i}(\cdot\mid o_t^i)) 采样动作；
3. 保存本步的 observation、action、old log-probability 和 value；
4. 在 `env.step()` 后，用同一条 step cache 补齐 reward 与边界并写入 buffer。

如果 (m_t^i=0)，该智能体已经没有决策权。此时即便为了保持张量形状，环境仍返回全零 observation，也不能把它解释成“车辆观察到了一个全零世界并选择了动作”。

### 3.2 当前 mask、下一步 mask 与终端 transition

正确时序是：

```text
current_active_mask
        │
        ├─ 只为当前 active agent 计算 value、采样动作、建立 step_cache
        │
        └─ env.step(actions)
                 │
                 ├─ 从 step_cache 写入刚完成的 transition
                 │    （包括本步刚到点/碰撞的有效终端 transition）
                 │
                 └─ info['active_mask'] 变成 next_active_mask，控制下一步
```

最重要的边界是：**不能用 `next_active_mask` 反向过滤本步 transition。** 一辆车在本步动作之后到达终点，说明它在动作前是 active；这最后一条成功 transition 是有效学习信号，必须保留。`next_active_mask=False` 只表示下一步不再为它采样。

### 3.3 固定形状 buffer 与变长 buffer

官方 MAPPO 实现常采用形如 `[time, rollout_thread, agent, feature]` 的固定形状 buffer。为了保持形状，死亡或失活智能体的槽位可以继续存在，但会使用 `active_masks`：

- 优势统计时排除 inactive 槽位；
- Actor loss 乘 `active_masks`；
- 可配置的 value loss 同样乘 `active_masks`。

本项目的 `RolloutBuffer` 是 `Dict[agent_id, List[Transition]]`，天然支持不同智能体拥有不同长度的轨迹。因此更直接、也更不易出错的做法是：在 collector 阶段就不为失活智能体生成 transition。

在本项目已经冻结的 **agent-specific 回报边界** 下，两种实现的目标相同：一个 agent 自身到点或碰撞被定义为 T，此后的奖励不再属于它的个体回报，因此失活槽位对 GAE 统计、Actor loss、Critic loss 和 transition 数的贡献都必须为零。当前项目若只在最后的 loss 上临时乘 mask，仍会让伪样本污染 GAE、优势标准化、最小样本数门槛和日志，所以不能作为完整修复。

如果未来改成“智能体死亡后仍累计队伍后续奖励”的团队 return 定义，Actor 与 Critic 的有效性可能不再相同：死后动作仍须从策略损失排除，但 Critic 可能需要学习吸收态或 death state 的团队价值。届时应拆分 `actor_active_mask` 与 `critic_valid_mask`，不能直接照搬本阶段的个体回报处理。该情况不是当前 `P0-00C` 的语义。

### 3.4 death mask 与 active mask 不是一回事

MAPPO 文献中的 death masking 主要讨论 **Critic 输入表示**：以某个死亡智能体为估值对象时，可以用“零向量 + agent ID”明确表示死亡状态，而不是继续使用含混的普通全局状态。它服务于仍把死后团队回报纳入价值目标的固定长度合作任务。

本项目中的 `active_mask` 讨论的是 **样本有效性**：目标车辆还能不能选择动作、能不能产生新的 transition。

对当前项目而言：

- 在 agent 0 已到点后，agent 1 的 centralized state 中允许保留 agent 0 的全零槽位，因为“队友已离场”是 agent 1 所处环境的一部分；
- 但不能再以 agent 0 为目标，使用它的全零 local observation 计算 Actor 动作或 (V(s,0)) 并写训练样本。

## 4. 五类信号的职责边界

| 信号 | 时间含义 | 应控制的内容 | 不应被拿来代替什么 |
|---|---|---|---|
| `active_mask_t[i]` | 动作前：agent (i) 当前是否可决策 | value、action、step cache、transition 生成 | 不能代替 termination/truncation 原因 |
| `done_t[i]` | 动作后：该 agent 的本条轨迹是否在这里形成边界 | GAE trace 是否跨过该边界 | 不能单独决定是否 bootstrap |
| `terminated_t[i]` | 动作后：agent-specific MDP 回报是否真正结束 | `1-terminated` 控制 value bootstrap | 不能控制本步样本是否有效 |
| `truncated_t[i]` | 动作后：任务本可继续，但被外部边界打断 | 选择同帧有效 (s_{t+1}) 做 bootstrap | 不能让 agent 在下一步继续采样 |
| `episode_end_reason` | 动作后：联合 CARLA 场景是否还运行 | 是否退出全局 rollout 循环 | 不能覆盖每辆车各自的 T/X 类型 |

离散动作环境中还可能存在 `available_action_mask`，它表示“活跃 agent 当前有哪些离散动作合法”。本项目使用连续车辆控制，不需要该 mask；它与“agent 是否活跃”也不是同一概念。

## 5. terminated、truncated 与 GAE 的关系

对智能体 (i)，一步 TD residual 应为

$$
\delta_t^i=r_t^i+\gamma(1-\text{terminated}_t^i)V_\phi(s_{t+1},i)-V_\phi(s_t,i).
$$

GAE 在单个 agent、单个 episode 内递推：

$$
\hat A_t^i=\delta_t^i+gamma\lambda(1-\text{done}_t^i)\hat A_{t+1}^i.
$$

两个 mask 分工不同：

- `1 - terminated`：决定最后一步是否允许从 (V(s_{t+1},i)) bootstrap；
- `1 - done`：决定 GAE trace 是否跨过当前 episode 边界。

因此：

- 真正终止 T：`done=1, terminated=1`，不 bootstrap，trace 也停止；
- 合法截断 X：`done=1, terminated=0`，使用同帧有效最终状态 bootstrap，但 trace 不得串到下一个 reset episode；
- 正常运行 A：`done=0, terminated=0`，正常 bootstrap 并继续 trace。

这与 `active_mask` 的关系是：本条 transition 是否存在由动作前的 `active_mask_t` 决定；本条 transition 如何计算回报由动作后的 T/X 决定。

## 6. 迁移到本项目的两红车基础策略训练

设 agent 0 先到点，agent 1 继续行驶三步后到点。正确的数据流示例如下：

| 决策时刻 | 当前 active mask | 本步采样 Actor | 本步写入 transition | step 后状态 |
|---|---|---|---|---|
| (t_0) | `{0: True, 1: True}` | 0、1 | 0、1 各一条 | 均继续 |
| (t_1) | `{0: True, 1: True}` | 0、1 | 0、1 各一条 | 0 到点：agent 0 为 T，下一 mask 为 False |
| (t_2) | `{0: False, 1: True}` | 仅 1 | 仅 1 一条 | 1 继续 |
| (t_3) | `{0: False, 1: True}` | 仅 1 | 仅 1 一条 | 1 继续 |
| (t_4) | `{0: False, 1: True}` | 仅 1 | 仅 1 一条 | 1 到点，全局 episode 结束 |

最终 agent 0 有 2 条 transition，agent 1 有 5 条。agent 0 在 (t_1) 的成功 transition 必须保留，但 (t_2,t_3,t_4) 不能再添加零奖励、重复 `done=True` 的伪样本。

类似地：

- 一辆红车自身碰撞：碰撞车最后一条 transition 有效且为 T；若场景立即全局关闭，未碰撞红车的同帧 transition 有效且为 X；
- `max_steps`：当步所有仍活跃车辆都写入最后一条有效 transition，并按 X 使用同帧 `next_obs` bootstrap；
- reset 时若某辆车未成功生成、`active_mask=False`：从第一步起就不得为它采样或写 buffer。

## 7. 当前 `robust_v2` 的具体问题和因果链

### 7.1 环境已经发布权威信号

`CarlaMultiAgentEnv` 在 reset 和每次 step 后都会返回：

- `info['active_mask']`；
- `info['terminated']`；
- `info['truncated']`；
- `info['episode_end_reason']`。

同时，环境为了保持 observation 字典形状稳定，会继续保留所有 agent key；已销毁车辆对应的 observation 是全零向量。所以 `aid in obs_dict` 只能表示“这个槽位存在”，不能表示“这个 agent 仍活跃”。

### 7.2 训练入口没有消费 `active_mask`

当前 `train_no_rllib_mappo_robust_v2.py` 的 collector 存在以下断链：

1. reset 后取得了 `info`，但没有保存 `info['active_mask']`；
2. 每步为全部 agent 计算 centralized value；
3. 遍历 `obs_dict.items()` 为所有 key 采样动作并创建 `step_cache`；
4. 从 `step_cache` 无条件写入所有 transition；
5. step 后没有把新的 `info['active_mask']` 用作下一步 mask；
6. 仅用 `all(dones.values())` 判断全局退出，没有以 `episode_end_reason` 校验场景状态。

因此，一辆车提前到点后，另一辆车继续 (N) 步，前者会多出 (N) 条无效 transition。

### 7.3 第一条伪样本还会形成不可能的状态配对

环境在 `step()` 中先生成 `next_obs`，随后检查 done 并将到点车辆移除。这样做有合理原因：合法 truncation 的 bootstrap 需要保存同一物理帧的有效最终观测。

但当前训练入口下一轮会同时：

- 使用上一步返回的、移除前生成的非零 local observation 为失活 Actor 采样；
- 重新调用 `env.get_global_state()`，得到车辆已经移除、相应槽位为零的 centralized state。

于是会出现“目标 Actor 看见车辆仍存在，Critic 却看见该车已经消失”的不可能 ((o_t^i,s_t)) 配对。再往后的伪样本才会变成全零 local observation。这进一步说明，不能靠检查 observation 是否全零来推断生命周期，必须使用环境发布的 `active_mask`。

### 7.4 伪样本如何污染 MAPPO

| 污染位置 | 直接后果 |
|---|---|
| Actor action/log-prob | 策略被迫拟合从未真正执行过的“死后动作” |
| Critic value/return | Critic 学习失活目标 agent 的伪价值，稀释真实驾驶状态 |
| GAE/优势标准化 | 伪 advantage 改变有效样本的均值和方差；当前逐 episode 标准化还会放大边界问题 |
| transition 总数 | 可能虚假达到 `min_transitions_per_update`，让数据不足的 rollout 被误判为可更新 |
| reward component 日志 | 失活车的零分量进入均值，掩盖真实车辆行为 |
| episode 长短与样本权重 | 越早到点的车反而可能被补入越多重复终止样本，产生与任务目标相反的权重偏差 |

当前 `compute_gae()` 中“termination 不 bootstrap、truncation bootstrap”的基本公式本身没有造成上述问题；问题发生在它接收到的数据之前。先修 active-mask 数据流，再在 `P0-00D` 修正优势标准化口径，顺序是合理的。

## 8. `P0-00C` 的实施边界

本阶段只改变“哪些 agent-time transition 有资格进入训练”，不改变网络结构、奖励定义和已经冻结的 T/X 语义。实际实施为：

1. reset 后读取并校验当前 `active_mask`；
2. 只为当前 active agent 计算 Critic value、Actor action 和 `step_cache`；
3. step 后只遍历本步 `step_cache` 写 transition，保留刚形成的终端/截断 transition；
4. 写完本步后，再将 `info['active_mask']` 更新为下一步 mask；
5. 若回合仍在运行但 active 集合发生变化，重新读取车辆清理后的下一决策观测，防止 surviving Actor 使用退场前观测而 Critic 使用退场后全局状态；环境原始返回的同帧最终状态仍单独保留给合法 truncation bootstrap；
6. 以 `episode_end_reason != 'running'` 作为全局退出主条件，并校验不存在活跃 agent 被静默遗漏；
7. 按 episode、按 agent 记录 `active_decision_steps` 和 `buffer_transition_count`，要求严格相等；
8. 用“一车提前终止、另一车继续 (N) 步”的 mock rollout 验证提前终止车辆零追加。

这一阶段不应：

- 删除最后一条真实终端 transition；
- 用 `next_active_mask` 过滤刚完成的 transition；
- 改写 `terminated/truncated` 场景矩阵；
- 改动 GAE 优势标准化方式（归入 `P0-00D`）；
- 改成共享 Actor 或重构奖励函数；
- 顺带修复已经停用的 RLlib/旧训练入口。

## 9. 修改后的五项评估依据

1. **决策一致性**：只有 `current_active_mask=True` 的 agent 执行 value 和 action 前向计算。
2. **样本守恒**：每个 episode/agent 的 `active_decision_steps == buffer_transition_count`。
3. **终端步完整性**：本步由 active 变为 inactive 的 agent 恰好保留一条最终 transition，不丢失也不重复。
4. **失活隔离性**：agent 提前结束后，无论其他 agent 继续多少步，其 action、value、buffer 长度、GAE 和 PPO batch 均不再增长。
5. **协议一致性**：全局退出由 `episode_end_reason` 驱动，逐 agent 的 T/X 和合法 bootstrap 仍严格符合 `P0-00B`，没有 active agent 被静默丢弃。

### 9.1 实施验收结果（2026-07-22）

| 评估依据 | 结果 | 自动化证据 |
|---|---|---|
| 决策一致性 | PASS | 提前结束 mock 中 Actor 调用次数为 `{0: 1, 1: 4}`，Critic 目标 agent 序列为 `[0, 1, 1, 1, 1]` |
| 样本守恒 | PASS | `validate_episode_transition_counts()` 强制逐 episode/agent 对账，决策数与 buffer 增量均为 `{0: 1, 1: 4}` |
| 终端步完整性 | PASS | agent 0 只保留一条非零终端样本，`done=1, terminated=1, reward=10` |
| 失活隔离性 | PASS | agent 0 退场后动作、价值和 buffer 均不再增长；所有样本保持 Actor/critic/action 特征维度 `19/38/3`，仅 batch 行数动态变化 |
| 协议一致性 | PASS | 陈旧返回观测中的标记值 `99` 在下一决策前刷新为清理后值 `3`；矛盾 reason/mask、`active != not done` 均 fail-fast；3 项 active-mask 与 15 项 lifecycle/GAE 测试共 18 项通过 |

以上结果证明训练入口已经按预定协议消费 `active_mask`，但只覆盖 mock/unit 和静态入口检查。真实 CARLA 中的传感器时序、车辆销毁与 world tick 联动仍须在 `P0-00G` smoke 中确认。

## 10. 可直接用于论文的方法表述草稿

> 本文采用集中训练、分散执行的多智能体近端策略优化框架。执行阶段，每个车辆智能体依据局部观测独立生成连续控制动作；训练阶段，集中式价值网络接收由所有车辆观测拼接得到的联合状态，并通过智能体标识输出对应的个体状态价值 (V_\phi(s_t,i))。该设计在共享全局交通信息的同时，保留不同车辆奖励与回报目标之间的差异。
>
> 为处理车辆异步完成任务的情况，本文将训练样本定义为有效的 agent-time transition，并引入逐智能体活跃掩码 (m_t^i)。仅当 (m_t^i=1) 时，才为智能体 (i) 执行动作采样、价值估计及轨迹写入。当车辆在本步到达目标或发生碰撞时，本步 transition 仍被保留，而环境返回的下一时刻活跃掩码用于禁止该车辆继续产生后续样本。该机制避免了固定多智能体接口中全零占位观测被误当作有效状态的问题。
>
> 回报估计进一步区分任务终止与外部截断。真实终止切断价值 bootstrap；时间上限或同伴事件造成的外部截断保留同帧最终状态的价值 bootstrap；两者均切断 GAE 在 episode 间的递推。由此，活跃掩码控制“样本是否存在”，终止/截断信号控制“样本如何计算回报”，二者共同保证异步多车轨迹的语义一致性。

论文正式使用时，还应补充真实 CARLA 实验设置、符号表、消融结果和最终 Git 版本；当前可引用算法定义与 mock/unit 验收方法，但不能把它直接表述为真实仿真性能结论。

## 11. 参考资料

1. Schulman, J., et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017. <https://arxiv.org/abs/1707.06347>
2. Schulman, J., et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438, 2015. <https://arxiv.org/abs/1506.02438>
3. Yu, C., et al. *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*. NeurIPS Datasets and Benchmarks, 2022. <https://arxiv.org/abs/2103.01955>
4. MAPPO authors. *Official MAPPO implementation (`marlbenchmark/on-policy`)*. <https://github.com/marlbenchmark/on-policy>
5. MAPPO official trainer: inactive slots are excluded from advantage statistics and can be masked out of policy/value losses. <https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py>
6. MAPPO official shared buffer: separates `masks`, `bad_masks`, `active_masks` and `available_actions`. <https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/utils/shared_buffer.py>
7. Farama Foundation. *Handling Time Limits*. Gymnasium Documentation. <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>
