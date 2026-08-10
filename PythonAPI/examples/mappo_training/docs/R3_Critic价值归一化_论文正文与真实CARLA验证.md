# R3 Critic 价值归一化：问题、方法与真实 CARLA 验证

> 状态：2026-08-10 实现与机制验收完成。R3 只修改 Critic 的目标尺度、稳健损失、状态保存和诊断；Actor、R2 reward-v3、场景、网络宽度、学习率、PPO/GAE 参数与梯度裁剪阈值保持不变。

## 1. 修改原因与主要问题

旧 Run 的 Critic 长期表现为：解释方差接近 0、预测标准差显著小于 return 标准差、更新前梯度几乎持续超过 `max_grad_norm=0.5`。这说明单一学习率要同时拟合小幅稠密 progress 与稀疏 success/collision 大目标时，原始 MSE 的尺度条件很差。继续放大奖励、学习率或梯度阈值只会掩盖问题，也会破坏 R2 的可解释奖励预算。

R3 的研究问题是：在不根据新策略性能追调参数的前提下，能否让 Critic 在数值稳定的目标尺度上训练，同时保证 rollout、bootstrap、GAE、指标与外部接口仍使用真实奖励单位？

## 2. 论文依据与方案选择

MAPPO 论文将 value normalization 列为影响 on-policy 多智能体训练稳定性的关键实现细节：网络回归归一化 value target，而计算 GAE 时使用反归一化 value。官方实现默认启用 ValueNorm，并提供 Huber value loss。R3 采用这一最小改动路径，而没有改 Critic 架构或引入额外容量。

- MAPPO 论文：[The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://papers.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)
- 官方 MAPPO ValueNorm：[on-policy/valuenorm.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/utils/valuenorm.py)
- 官方 MAPPO value loss 路径：[on-policy/r_mappo.py](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py)
- 目标自适应归一化背景：[Learning values across many orders of magnitude](https://papers.neurips.cc/paper_files/paper/2016/file/5227b6aaf294f5f027273aebf16015f2-Paper.pdf)

冻结参数为 `beta=0.99999`、`epsilon=1e-5`、方差下限 `1e-2`、归一化空间 Huber `delta=10`。前三项与官方 ValueNorm 默认值一致，Huber 阈值与官方 MAPPO 默认配置一致。它们在查看新 pilot 结果前写入训练契约，不以成功率或碰撞率做选择。

## 3. 解决方式

1. Critic 网络输出定义为归一化 value；每个完整 PPO update 只用全部 active-agent 原始 return 更新一次共享标量 ValueNorm，避免 minibatch 顺序改变 target。
2. rollout 当前值和截断 bootstrap 值先反归一化，再进入 buffer 与 GAE。因此 `advantage = reward + gamma * V_raw(next) - V_raw(current)` 的量纲没有改变。
3. Critic 在归一化 target 上使用 Huber loss。极端误差超过 10 个归一化标准差后转为线性增长，碰撞/成功稀疏样本不会无限放大单批梯度。
4. checkpoint schema 升至 v4，强制保存 running mean、running mean square、debiasing term、update count 和 critic spec；strict resume 在连接 CARLA 前恢复并核验这些状态。
5. metrics 升至 v5，同时记录原始单位的预测/target/MAE/解释方差和归一化空间统计，避免“loss 变小”被误读为真实拟合必然变好。
6. R0/R1/R2 checkpoint 与 C0 契约保持只读历史路径；R3 训练与 C0 v4 使用新的训练契约 ID 和 SHA，不允许把旧 checkpoint 当作 strict resume。

## 4. 真实 CARLA 验证

验证连接 CARLA 0.9.16 的 `127.0.0.1:2000`，固定 Town02、seed 42、reward-v3、`scenario_01_east_basic_launch_a0_rear.json`。报告为 `diagnostics/r2_r3_real_carla/20260810_200111/report.json`。该实验是机制诊断，不是策略性能 pilot，也没有用于调参。

### 4.1 R2 reward-v3 实车轨迹

| 脚本轨迹 | 终局 | Agent 0 折扣回报 | Agent 1 折扣回报 | 关键证据 |
|---|---|---:|---:|---|
| stopped，80 steps | 诊断截断 | 0.0000 | 0.0000 | progress 与全部成本均为 0，不能停车刷分 |
| smooth | 两车成功，91 steps | 13.9143 | 14.9965 | completion 分别为 0.9762、0.9958；两车 success `+10`，无碰撞 |
| oscillatory | 合作碰撞，74 steps | 0.6211 | -13.8813 | 每车 longitudinal-rate 与 steer-rate 累计均为 `-0.07`，并产生额外 lane cost |
| rear_collision | 合作碰撞，33 steps | -15.8667 | -18.1656 | 两车均收到 collision `-20`；后车另有 TTC cost |

所有 536 个 active transition 的 `reward == sum(components)` 最大绝对残差小于 `1e-6`。真实动力学下同时满足 `smooth > stopped > controlled collision`；静止为严格零；碰撞事件被传感器与终局原因共同确认。这为 R2 的数学预算补上了真实 CARLA 信号链验证，但不等价于“训练后的 Actor 已经有效”。

### 4.2 同一真实 batch 的 Critic A/B

从上述轨迹得到 536 个真实 global-state/agent/discounted-return 样本，固定相同初始化、Adam `3e-4`、300 次 full-batch update 与 `max_grad_norm=0.5`，仅比较原始 MSE 和 R3 ValueNorm+Huber。80% 拟合，20% 留出。

| 指标 | 原始 return + MSE | R3 ValueNorm + Huber |
|---|---:|---:|
| 原始 target 标准差 | 11.7820 | 11.7820 |
| 归一化训练 target 标准差 | — | 1.0000 |
| 裁剪前梯度范数均值 | 9.6822 | 0.1161 |
| 梯度裁剪激活率 | 100.00% | 0.33% |
| 留出集原始单位 MSE | 63.8371 | 57.1734 |
| 留出集解释方差 | 0.5309 | 0.5897 |

R3 将梯度裁剪率降低 99.67 个百分点，留出 MSE 降低约 10.44%，解释方差提高 0.0588。CARLA 环境关闭后同步模式与 fixed-delta 恢复到诊断前状态，车辆 actor 数没有增加。

## 5. 预期效果与边界

预期效果不是保证成功率立即上升，而是让 Critic 具备可学习的尺度：预测方差能够随 return 方差增长，梯度不再被持续硬裁剪，优势估计噪声下降，Actor 接收到的更新方向更可靠。正式 pilot 中应继续观察原始单位 explained variance、prediction/target std 比值、MAE、梯度裁剪率以及 brake/throttle/coast 行为，不能只看归一化 loss。

R3 不解决早停门槛和行为塌缩判据；那属于 R4。真实脚本轨迹证明奖励信号链和 Critic 数值机制成立，但未证明学习收敛或泛化。R4 冻结 GO/HOLD v2 后，还需 R5 用最终训练入口做 checkpoint/resume 与成功链路 smoke，之后才能放行 fresh pilot。
