# P0-BASE Town05 v3 seed 42：u50 制动局部最优诊断

## P0-T05-09/R6 u160 后续追踪（2026-08-23）

> 本节是对原 u50 诊断的后续追踪，不改写旧实验结论。R6 诊断 Run `P0_DIAG_MAPPO_Town05R6U50_s42_20260820_123634` 已在同一冻结合同的注册边界内完成至 u160；该 Run 仍只允许作为诊断/失败回归证据，禁止进入基础策略池，也不得在新奖励或新合同下 resume。

### 追踪身份与完整性

| 项目 | 记录 |
|---|---|
| RunID | `P0_DIAG_MAPPO_Town05R6U50_s42_20260820_123634` |
| 代码身份 | `63d62be026a8b27e67b00f8c04626db678552d38` |
| 合同 | `P0_DIAG_MAPPO_TOWN05_R6_U50_V1` |
| 合同 SHA256 | `b50df18c2f1eac0b904c55f022efd31fac64cf1d64617900ad638cc91adf82c4` |
| 注册训练边界 | `planned_total_updates=160`；`lr_decay_horizon=160`；`entropy_anneal_updates=160` |
| 完整边界 | u160，2,560 个完整 episode |
| outcome | 93 success、525 collision、363 offroad、1,579 timeout |
| 训练健康 | 160 次 PPO update；未发现 nonfinite、skipped update 或 reward residual 超容差 |
| u160 checkpoint SHA256 | `ckpt_000160.pt`: `e918b9a0cd84cea29eb64a7d7ac9ccc449ca3bef126127db5f957de885437d40`；`latest.pt`: `d4335279236203b45a9e5fa561c1c6c476de40168b4db808674cc4f5d24f6c5d` |

### u160 暴露的两个主问题

1. **策略没有重新落入 u50 的纯制动局部最优，但学到了不安全的车道捷径。** u151–u160 的 success rate 为 16.25%，高于 u101–u110 的 2.50%；与此同时，collision+offroad 从 46.25% 升至 71.25%，不能把 success 上升解释为合理驾驶收敛。agent1 的均速约从 0.85 m/s 升至 4.10 m/s、brake mode 从 0.393 降至 0.200、确定性纵向动作从 +0.105 升至 +0.443，但 legal-lane-change 代理均值约从 1.37% 升至 6.99%，lane-deviation exemption rate 从 10.84% 升至 43.43%。这说明主要矛盾已由“停车避险”转为“用跨线/豁免走捷径”。
2. **critic 由有效拟合退化为接近均值预测器。** explained variance 从 u20 的 0.965 降至 u160 的 0.0226；同期 value target std 从 2.797 增至 9.207，而 prediction std 仅为 1.367。u160 critic grad norm 约 0.361、clip rate 为 0、relative update norm 约 0.001071，因此没有证据支持梯度爆炸；更符合“目标分布变宽、学习率已退火到底、critic 输出方差跟不上”的欠拟合/非平稳退化。

### critic 退化里程碑

| update | explained variance | target std | prediction std | critic loss | critic LR |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.9650 | 2.797 | 2.741 | 0.0087 | 2.64e-4 |
| 80 | 0.2740 | 2.313 | 1.194 | 0.0767 | 1.51e-4 |
| 100 | 0.1030 | 4.294 | 1.351 | 0.3648 | 1.13e-4 |
| 120 | 0.0620 | 8.384 | 1.639 | 1.3025 | 7.55e-5 |
| 140 | -0.0086 | 9.816 | 1.520 | 1.3996 | 3.77e-5 |
| 160 | 0.0226 | 9.207 | 1.367 | 0.9736 | 1.50e-5 |

### 当前根因排序

1. **reward/终止合法性缺口：** 当前 `offroad` 不能识别“仍位于 `Driving` 类型、但已进入对向车道”的状态；合法换道豁免又可能覆盖不合理横向行为，导致违规捷径仍能保留 progress/success 收益。
2. **回报分布非平稳：** 策略由低速 timeout 转向高速 collision/offroad/跨线，value target 的尺度和混合分布在训练后半段显著扩大。
3. **critic 学习率时序不匹配：** 160-update 线性退火恰好在 target variance 上升阶段把 critic LR 降至 1.5e-5，使其难以追踪新分布。
4. **critic 审计不足：** 当前汇总 EV 不足以区分单 agent、scenario 与 outcome 子分布的失配，也无法判断是在线 LR 时序还是函数拟合能力不足。虽然部分 reward/终止状态未直接出现在 critic 输入中，但按用户决定，本轮不通过增加车道观测来处理。R6 checkpoint/JSONL 没有保存原始 rollout state/target，不能事后伪造 R6 固定 batch 重放；R7 应在现有输入下记录在线 pre/post-update 分组诊断，并从首个完整 checkpoint 开始保存可重放 probe。
5. **长任务信用分配仍然困难：** 500-step 任务配合 `gamma=0.999`、`lambda=0.95` 时，GAE 权重按 `(gamma*lambda)^l` 衰减，特征尺度约为 19.6 step；u160 对复杂任务仍只是早期诊断，不是充分收敛预算。

### 拟定解决方法（待审批，尚未实施）

- 新建 **R7 / reward-v5 / 新合同 / 新 RunID**，fresh 训练；R6 u160 不跨合同恢复。
- 不构造手工“车道池”，也不向 actor 的 20 维观测增加车道字段。环境侧直接使用 CARLA waypoint 的 `road_id/section_id/lane_id/lane_type/lane_change`、航向和拓扑关系判定同向合法换道、对向占用与逆行。
- 用有限状态机描述换道：原车道保持与合法换道完成后在新车道内行驶都获得相同的 progress/alignment 基础奖励；换道走廊内只免除合理横向偏差成本，不额外发放可刷取的正奖励；越入对向/错误方向属于强违规，并进入终止或强失败路径。
- success 必须同时通过目标、方向、可驾驶车道及非对向占用合法性门控，防止跨线捷径换取 success。
- critic 先增加逐 agent、逐 outcome/scenario 的 pre/post-update EV、target/prediction std 比、bias、MAE 和相关性审计；R7 checkpoint 保存有界确定性 probe，供后续同输入回放比较。R6 只能保留历史汇总证据，不能声称做过原始 fixed-batch 拟合。得到 R7 probe 后再决定是否调整容量、更新轮数、ValueNorm/PopArt。
- 新长训练预算在合同创建时一次性注册，建议先按 u600 设计；critic LR 不再在 u160 附近提前衰减到底。建议候选为前 300 update 保持约 3e-4，再退火至 u600 的约 1e-4，最终数值须由离线拟合探针确认。
- 本轮 actor 与 critic 的观测输入均保持现状，不增加车道、lane id、标线或 wrong-way 字段；critic 修复只从诊断、学习率时序和经拟合探针证实的优化能力入手。
- 共享 actor 与行为克隆继续保留为 P1 消融，不与本轮 reward/critic 修复同时实施。

### 下一阶段 HOLD 条件

- 任一 agent 持续出现制动、饱和动作、lane/opposing-lane 违规或 critic EV 塌缩，即全局 HOLD。
- success 若伴随 collision/offroad、对向占用、wrong-way 或 lane invasion 同步上升，不计为 mastery。
- 在无 CARLA 数学/状态机/协议/全量回归和真实 CARLA 脚本化车道语义 smoke 全部通过前，不启动正式长训练。

> 状态：`INTERRUPTED / DIAGNOSTIC HOLD`（2026-08-14）。
> 本文记录 Run `P0_BASE_MAPPO_Town05V3_s42_20260814_125226` 在研究者要求下停止于最后完整 update 50 的原因、证据和下一版优化边界。该 Run 不再用于能力训练、选模、基础策略池或多 seed 放行。

## 1. 停止边界与证据完整性

- 训练身份保持为 origin Git `5dad4d4f84412d76a7d4316fd85778c660f4f8df`、合同 `P0_BASE_MAPPO_TOWN05_V3`、seed 42、同一 RunID 严格恢复；训练期间没有修改奖励、PPO、场景或课程参数。
- 最后完整边界是 update 50，共 800 个完整 episode。周期 checkpoint 为 `logs/P0_BASE_MAPPO_Town05V3_s42_20260814_125226/periodic/ckpt_update_000050.pt`，大小 832961 bytes，SHA256 `20b86366f3246a5603ac602652e4f86e904752ba63d3ca2416b2990cdd92adba`；checksum 和 schema-v5 header 校验通过。
- 停止命令到达时 update 51 已写入 1 条 episode，但尚未形成该 update 的 metrics、PPO 更新或 checkpoint。原始 `episodes.jsonl` 保留这条中断证据，不通过删除记录伪造自然停点；因此对原目录直接运行 strict validator 会如实报告 episodes 包含 u51 而 metrics 只到 u50。所有统计均显式排除该未完成前缀。
- u1–u50 的 50 次 PPO 均执行；两车 pre-update ratio 最大绝对误差均为 0，reward step residual 为 0，episode residual 最大约 `1.10e-13`，nonfinite 和 skipped update 均为 0。停止原因不是数值故障、CARLA 断连或 PPO 实现异常。
- 人工终止后 CARLA 一度保留同步设置；恢复异步时处理了终止前尚未提交的生成命令，出现 2 辆训练车和 6 个训练传感器残留。按 actor ID 清理后最终状态为 async/None、vehicle/sensor `0/0`，服务无需重启。
- 交付前再次只读查询时 2000 端口已无 CARLA 进程/TCP listener；这发生在训练停止和 actor 清理证据完成之后，不影响 u1–u50 结果。下一次真实机制测试前需要由研究者重新启动 CARLA。

## 2. u1–u50 行为趋势

| update 窗口 | success | collision | offroad | timeout | A0 速度 m/s | A0 完成度 | A0 brake mode | A0 纵向动作 | A1 速度 m/s | A1 完成度 | A1 brake mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–5 | 0/80 | 6/80 | 31/80 | 43/80 | 0.1903 | 0.0926 | 0.5424 | -0.0976 | 0.3428 | 0.1373 | 0.4827 |
| 6–10 | 0/80 | 5/80 | 0/80 | 75/80 | 0.0700 | 0.0430 | 0.6512 | -0.2647 | 0.0821 | 0.0485 | 0.6230 |
| 11–20 | 0/160 | 3/160 | 7/160 | 150/160 | 0.1092 | 0.0677 | 0.6138 | -0.2091 | 0.0815 | 0.0515 | 0.6329 |
| 21–30 | 0/160 | 3/160 | 5/160 | 152/160 | 0.0429 | 0.0320 | 0.7498 | -0.4267 | 0.1122 | 0.0721 | 0.5948 |
| 31–40 | 0/160 | 4/160 | 5/160 | 151/160 | 0.0143 | 0.0176 | 0.9282 | -0.7629 | 0.1557 | 0.1037 | 0.5775 |
| 41–50 | 0/160 | 2/160 | 0/160 | 158/160 | 0.0101 | 0.0138 | 0.9901 | -0.9328 | 0.1156 | 0.0793 | 0.6036 |

全段 800 episode 的结局为 0 success、23 collision、48 offroad、729 timeout。offroad 从早期大幅下降到 u41–u50 的 0，但 agent0 同时变为近乎全程制动，因此不能把低越界/碰撞解释成安全驾驶能力。

u50 单点中 agent0 的 brake mode 为 99.75%，平均纵向动作约 -0.964，速度约 0.011 m/s；u41–u50 窗口已连续稳定在同一量级。长训已经否定“只是 u5 训练轮次太短”的解释。

## 3. 为什么 PPO 会强化停车

### 3.1 当前回报排序实际偏向静止 agent

u41–u50 的平均 episode return 为 agent0 `-5.7736`、agent1 `-8.3456`。agent0 虽未完成任务，却通过停车把每回合纵向变化成本从 u31–u40 的 `-1.2227` 降到 `-0.5266`，车道偏离从 `-0.0077` 降到 0，并避免了 agent1 行驶时的车道/压线成本。agent1 仍有更高速度和完成度，却得到更低回报。PPO 因此是在正确优化当前回报排序，而不是随机误更新。

### 3.2 任务信号与 500 步时域不匹配

reward v3 的路线势能满足 `10 * (gamma * q_next - q_prev)`。这可以阻止往返刷分，但折扣后的整段任务信号只保留 `10 * gamma^T * q_T`。当前 `gamma=0.997` 的有效视野约 333 步，而 episode 上限为 500 步；`gamma^500=0.2226`。因此晚到的部分进度和成功信号显著衰减，而 LiDAR、控制变化和车道成本仍在沿途累计。随机初始化又从未采到成功样本，actor 很容易先发现“固定制动可减少变化/越界成本”。

### 3.3 pre-tanh 熵与实际动作熵不一致

`Actor.evaluate_actions_z()` 当前使用基础 `Normal` 的 `dist.entropy()`，但环境执行的是 `tanh(z)` 后的动作。tanh 饱和时，pre-tanh Normal 仍可具有较高熵，实际动作却几乎固定在边界。合同同时使用 `init_log_std=0.0`、`ent_coef=0.03`，而模型只限制 `log_std` 下界、没有上界。

实测两车 log-std 从 u5 的约 `0.006` 持续升到 u50 的 `0.032/0.052`（agent0）和 `0.059/0.057`（agent1）；训练熵从 u1–u5 的约 2.845 升至 u41–u50 的 2.919/2.945。与此同时 agent0 的实际 brake mode 升至 99%。这说明“探索熵健康”是 pre-tanh 指标假象：策略可以把均值推向负饱和区，在保留 Normal 熵奖励的同时输出近乎固定的全制动动作，并降低控制变化成本。

### 3.4 独立 actor 固化了角色分化

两辆合作车辆使用独立 actor、各自 optimizer；共享的是 agent-conditioned centralized critic，而不是动作参数。agent1 的少量行驶经验不会直接更新 agent0。随机早期分化后，agent0 更快进入制动吸引域，随后因相对回报更高而自我强化。critic 在 u41–u50 的 EV 升至均值 0.767、u50 为 0.854，只表示它越来越会预测窄化的全超时回报，不代表产生了到达能力，也不是本次失败的直接根因。

### 3.5 课程按 update 推进而非按能力推进

合同在 warmup 20 后按 update 数加入更难场景，不要求基础场景达到掌握门槛。u21 后任务变难会放大探索成本，但 u22 即使采到的仍是旧基础场景也没有成功，因此课程推进不是唯一根因。它需要改成能力门控，但不能单独修复奖励/熵造成的制动吸引域。

## 4. 优化方案（先冻结新协议，再 fresh 验证）

### P0：修复策略分布与熵目标

1. 为 tanh-squashed Gaussian 增加动作空间熵的 Monte Carlo 估计，训练和日志都使用带 Jacobian 修正后的 `-log pi(a|s)`，不得继续把基础 Normal 熵当作实际动作熵。
2. 为 `log_std` 增加合理上界，重新评估 `init_log_std` 和 `ent_coef`；优先比较较低初始方差与随 update 衰减的熵系数。验收必须同时看 pre-tanh 参数、post-tanh 熵、纵向/转向饱和率和确定性 actor mean。
3. 增加“高 pre-tanh 熵 + 高 post-tanh 饱和”的显式 reward-hacking/健康告警。不能只因 Normal entropy 未下降就宣称探索健康。

### P0：重做任务—安全奖励预算

1. 保留 collision/offroad 的强终局优先级，禁止为了让车动起来直接降低道路安全约束。
2. 在真实平滑前进、静止、振荡、碰撞、offroad、部分进度超时和成功轨迹上重新计算折扣回报；硬性要求 `成功 > 安全平滑前进 > 静止/无进度超时 > 碰撞/offroad`，并给安全平滑前进相对静止留下非零裕量。
3. 联合校准 `gamma`、episode 长度和 task scale。候选方向是把折扣视野覆盖完整任务（例如评估 `gamma=0.999`）或缩短训练路线/分阶段目标；不能只改一项而忽略沿途成本同时被重新折扣。
4. 评估“任务截止失败 + 剩余距离”终局项，或在明确无交通阻塞时增加持续无进度终止。该项必须有上界，且不能让移动但越界的轨迹优于合法静止；具体数值只能由上述轨迹预算和敏感性测试决定。
5. 复核 LiDAR 静态距离成本：平行同速车辆不应仅因距离近而长期累计与碰撞风险同向的成本；优先用 closing speed/TTC 门控静态邻近风险。

### P1：课程与参数共享

1. 课程从 update 驱动改为 mastery gate：仅当基础场景的双 agent 最低完成度、最低速度、成功率和 collision/offroad 上界同时满足预注册窗口，才加入下一难度；未通过时保持当前阶段而不是自动升级。
2. 对称合作任务优先做“共享 trunk/共享 actor + agent/角色 embedding”的消融，与当前独立 actor 对比。目标是让两车共享样本效率，同时保留不同出生车道/角色条件；结构变化必须新建合同和 fresh Run。
3. 如随机初始化在修复后仍无法产生成功，使用 autopilot/脚本合法轨迹做短行为克隆 warm start，再由 MAPPO 微调。行为克隆只能在奖励排序修复后使用，否则仍可能被 PPO 重新推回停车解。

## 5. 下一轮验收顺序

1. 无 CARLA：新增 squashed-entropy、log-std 上下界、折扣预算、stall/timeout 语义和共享策略专项测试，并完成全量回归。
2. 真实 CARLA 机制测试：固定简单场景分别执行平滑前进、静止、振荡和越界，核对动作空间熵、奖励排序、终局和守恒。
3. fresh 1-update 工程 smoke：只验 schema/checkpoint/resume/metrics/清理，不作收敛结论。
4. fresh seed-42 小规模 pilot：先只在基础课程运行到预注册停点，要求两辆车都没有持续制动塌缩；未通过立即 HOLD，不再用延长至 u100 代替机制修复。
5. 只有基础阶段通过 mastery gate，才扩展完整课程、双 C0、GO/HOLD 和多 seed。当前 u50 checkpoint 只保留为失败对照和回归输入，禁止 resume 成为基础策略。

## 6. 结论

本次停止是有充分证据的计算止损。问题不是 CARLA 随机性、训练轮次不足或 critic 数值崩溃，而是 task horizon、奖励预算、pre/post-tanh 熵定义和独立 actor 共同形成的制动吸引域。下一步应先形成新的 action/reward/curriculum 协议并 fresh 小规模验证；直接恢复 u50 或继续到 u100 预计只会把错误局部最优固化得更深。
