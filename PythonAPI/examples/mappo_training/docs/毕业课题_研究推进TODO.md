# 基于对抗学习的多车协作驾驶算法研究：研究推进 TODO

> 本文档是本课题的主任务清单、代码修改记录表和实验登记表。执行任务时以本文档的阶段门槛为准；理论设计细节参考 [对抗指标、奖励设计与阶段计划](./adversarial_metrics_reward_design_and_plan_20260705.md)、[MAPPO 多智能体样本处理与 active-mask 原理](./MAPPO多智能体样本处理与active-mask原理.md)、[MAPPO GAE、优势标准化与 PPO 更新原理](./MAPPO_GAE优势标准化与PPO更新原理.md) 和 [MAPPO checkpoint 与恢复训练协议](./MAPPO_checkpoint与恢复训练协议.md)。

## 0. 项目总览

| 项目 | 当前值 |
|---|---|
| 课题 | 基于对抗学习的多车协作驾驶算法研究 |
| 研究点 1 | 使用多智能体强化学习训练对抗车辆，主动生成能够干扰基础驾驶策略的长尾场景 |
| 研究点 2 | 使用 PSRO（不是 PRSO）维护红蓝策略池并交替演进，研究策略泛化和循环克制 |
| 扩展研究点 | 在多辆对抗车之间引入显式通信，研究通信对协同攻击能力的增益 |
| 当前工程基线 | `P0-00F`：版本化 checkpoint、原子保存、严格 resume、RNG/课程/LR 连续及实验来源记录已完成 unit 验收；下一项为 P0-00G 真实 CARLA smoke |
| 当前研究阶段 | P0-00 基础 MAPPO 算法数据流闭环 -> Phase 0 指标可信化 -> Phase 1 单对抗车有效性验证 |
| 当前训练入口 | 基础策略仅使用 `training/train_no_rllib_mappo_robust_v2.py`；单蓝车在 Phase 1 使用 `training/train_2red_1blue_ppo.py` |
| 当前有效自动化测试 | 原 35 项 + checkpoint/resume 16 项，共 51 项通过（2026-07-27）；新增覆盖原子保存、SHA256、immutable periodic、全部 MAPPO 模型/optimizer 与 RNG 连续、协议/配置拒绝、fresh/pretrained/resume 分离和训练期无 `best.pt` |
| 最近训练产物 | 2026-07-05 的 `train_2red_1blue_ppo` 中断实验；产生于生命周期修复前，只能作历史参考 |
| 本文档创建日期 | 2026-07-20 |
| 最后更新日期 | 2026-07-27 |

### 阶段状态

| 阶段 | 状态 | 结论 |
|---|---|---|
| 基础 CARLA 多车环境与 MAPPO | UNIT VERIFIED | P0-00B/C/D/E/F 已闭合生命周期、有效样本、GAE/PPO、生产更新和恢复协议；下一项 P0-00G 在真实 CARLA 验证三更新与一次 resume |
| 对抗 episode 生命周期 | UNIT VERIFIED | P0-00B 已按红/蓝 agent-specific 任务边界适配；15 项测试通过，真实 CARLA 联调归入 P0-00G |
| Phase 0 指标与归因 | IN PROGRESS | 仍缺精确碰撞对象、转向量解析、C0/C1/C2 统一评估器 |
| Phase 1 单蓝车对抗 | BLOCKED | 等待 Phase 0；旧训练不能证明修复后的对抗有效性 |
| Phase 2 红车策略池 | PARTIAL | PSRO 池代码已存在，但质量、数量和行为多样性尚未达标 |
| Phase 3 双蓝车协同 | PARTIAL | 训练脚本已存在，但奖励和生命周期语义尚未完全统一 |
| Phase 4 PSRO | PARTIAL | runner、population、payoff、meta-solver 和 JPC 已有；历史结果退化为碰撞攻击 |
| Phase 5 显式通信 | NOT STARTED | 代码中尚无消息编码、聚合、带宽或通信消融 |
| Phase 6 红蓝双边演进 | NOT STARTED | 必须建立在单蓝、双蓝和固定池 PSRO 的可信结果之上 |
| Phase 7 论文主实验 | NOT STARTED | 等待前述阶段形成稳定方法和冻结协议 |

## 1. 使用规则

1. 每次开始工作前先阅读“下一步唯一主线”，不要同时推进多个研究阶段。
2. 完成任务后立即勾选，并在“代码修改记录”或“实验登记表”留下证据路径。
3. 没有代码提交、测试输出、JSON/CSV 指标或检查点路径的任务，不标记为完成。
4. 训练日志只用于诊断；论文结论必须来自冻结检查点上的独立评估。
5. 所有实验使用唯一 `RunID`，禁止覆盖已有目录。推荐格式：`P{phase}_{method}_{scenario}_s{seed}_{YYYYMMDD_HHMM}`。
6. 每个 Run 至少记录：Git SHA、完整命令/配置、随机种子、基础检查点、输出目录、CARLA 版本和结果摘要。
7. 修改环境语义、观测、动作、奖励或网络结构后，默认旧检查点不兼容；必须显式验证后才能恢复训练。
8. 阶段验收门槛未通过，不进入下一阶段。失败实验也要登记，不能只保留正结果。
9. 任何“对抗效果”首先报告 `C2 - C1`，不能用 `1 - C2成功率` 代替对抗贡献。
10. 本文中的碰撞率均按 episode 统计；若同时保留 step 级指标，名称必须显式带 `_per_step`。
11. 当前基础策略训练唯一入口是 `train_no_rllib_mappo_robust_v2.py`；旧 RLlib、shared-policy PPO、旧 minimal/robust/multi 脚本不在本轮修复范围，禁止混入正式实验。

### 标签约定

- `[x]`：已完成且有证据。
- `[ ] NEXT`：当前阶段应立即执行。
- `[ ] BLOCKED`：被前置任务阻塞。
- `[ ] LATER`：后续阶段，不应提前实现。
- `[ ] VERIFY`：已有实现，但尚未经过足够测试或实验验证。

## 2. 不可破坏的研究不变量

- **三组对照**：C0 无额外交通；C1 有非学习/自然交通；C2 有学习型对抗车。
- **主效应**：`AttackEffect = SuccessRate(C2) - SuccessRate(C1)`，预期为负；使用相同场景和相同 seed 的配对差值。
- **自然交通效应**：`TrafficEffect = SuccessRate(C1) - SuccessRate(C0)`，避免把普通交通难度误认为对抗学习效果。
- **互斥结局**：每个 episode 只能属于成功、碰撞、超时/其他失败中的一个；各比例之和为 1。
- **安全约束**：蓝车自身碰撞和 kamikaze 不能成为降低红车成功率的主要方式。
- **精确归因**：墙体碰撞、红蓝碰撞和蓝蓝碰撞必须区分；无法归因时不得发放诱导碰撞奖励。
- **独立评估**：评估使用确定性动作、冻结权重、固定协议，且不复用训练 rollout 作为最终结果。
- **可复现性**：开发实验至少 3 个训练 seed；论文最终实验建议 5 个 seed，并报告均值、标准差和 95% 置信区间。
- **统计口径冻结**：最终比较开始后不得修改指标定义；如必须修改，旧结果全部标记为旧协议。
- **检查点选择**：最佳模型必须满足安全约束，再按独立验证集上的攻击效果选取，不能按训练期 `1 - red_success` 选取。
- **终局与截断**：按 agent-specific 回报边界判定；自身到点/碰撞及红车结局对应的蓝车对抗结果为 `terminated`，同伴红车碰撞对未碰撞红车属于外部 `truncated`，`max_steps` 对仍存活车辆同样为 `truncated`；只有合法 truncation 允许 value bootstrap。
- **失活样本隔离**：`active_mask=False` 的 agent 不得继续动作采样、写入 rollout、参与 GAE 或增加 transition 计数；全零 observation 不能被当成有效状态训练。
- **GAE 与优化目标隔离**：每个 agent/episode 独立计算原始 GAE；同一 agent 仅在完整 rollout 上标准化一次；Actor 使用标准化 advantage，Critic return 始终为 `raw_advantage + old_value`。

## 3. 下一步唯一主线

以下任务严格按顺序执行。在 `P0-00G` 完成前不运行任何训练 smoke；在 `P0-06` 完成前，不启动新的基础策略或对抗策略长训练。

- [x] **P0-00A 冻结训练入口**：基础策略只维护 `train_no_rllib_mappo_robust_v2.py`，Phase 1 单蓝车只维护 `train_2red_1blue_ppo.py`；README、运行命令和文件索引已标记 RLlib 和其他旧入口停止使用。
- [x] **P0-00B 冻结并适配 terminated/truncated 语义**：采用 agent-centric 回报边界；同伴红车碰撞时未碰撞红车按 truncation bootstrap，红车成功/碰撞时蓝车按对抗任务结局 termination；环境、两个当前入口和 15 项测试已同步。
- [x] **P0-00C 修复 robust_v2 失活样本采集**：reset/step 的 `info['active_mask']` 已驱动 value、动作和 buffer；一车提前终止、另一车继续 3 步的 mock 中 transition 数为 `{0: 1, 1: 4}`，失活车零追加。
- [x] **P0-00D 修正 GAE 与优势标准化口径**：每个 episode 独立计算原始 GAE，同一 agent 拼接全部 episode 后在 rollout 上统一标准化；Critic 保持原始 return，PPO 更新前强制 `ratio≈1` 并检查数值有限。
- [x] **P0-00E 补 MAPPO/PPO 算法回归测试**：生产 batch 准备、clipped actor loss 与完整 optimizer update 已抽取并由 10 项新测试直接调用；多 agent 时间顺序、Critic int/batch agent-id、空 agent、异常输入零污染和全量回归均通过。
- [x] **P0-00F 修正基础策略 checkpoint/resume 协议**：训练只保存原子 `latest`、不可覆盖 periodic 和安全 `interrupted`；补齐 schema/protocol、RNG、课程、LR horizon、RunID/Git/config/source/manifest，并以 16 项测试验证严格恢复。
- [ ] NEXT **P0-00G 运行 robust_v2 三更新 smoke**：真实 CARLA 下核对 transition 数、终局、PPO 健康值、checkpoint/resume 和 actor 清理；通过前不得长训练。
- [ ] BLOCKED **P0-01 精确记录碰撞对象**：等待 P0-00G；随后从 CARLA collision event 保存对方 actor id/type，并区分 `red-blue`、`blue-blue`、`static/world`。
- [ ] BLOCKED **P0-02 修复转向量提取**：等待 P0-01；`extract_kinematics()` 正确处理 NumPy action `[throttle, steer, brake]`，不能把正常步骤的 steer 恒记为 0。
- [ ] BLOCKED **P0-03 补指标和奖励单元测试**：等待 P0-02；创建 `tests/test_adversarial_metrics.py`、`tests/test_adversarial_reward.py`。
- [ ] BLOCKED **P0-04 实现 C0/C1/C2 统一评估器**：等待 P0-03；创建 `evaluation/evaluate_adversarial_c012.py`，输出逐 episode 原始记录和汇总统计。
- [ ] BLOCKED **P0-05 修正 best checkpoint 规则**：等待 P0-04；训练期保存 `latest/periodic`，独立验证后再生成 `best_valid.pt`。
- [ ] BLOCKED **P0-06 运行 Phase 0 基准评估**：等待 P0-05；验证互斥率、C0 基础能力和 C1 自然交通效应。
- [ ] BLOCKED **P1-01 启动修复后的单蓝车 smoke run**：使用新 RunID，不恢复 2026-07-05 的旧 run。
- [ ] BLOCKED **P1-02 完成单蓝车 3 seed 正式训练与配对评估**。
- [ ] LATER **P2-01 建立合格且多样的红车池**。
- [ ] LATER **P3-01 统一双蓝车训练语义并验证协同对抗**。
- [ ] LATER **P4-01 重跑固定池 PSRO**。
- [ ] LATER **P5-01 增加显式通信与消融实验**。
- [ ] LATER **P6-01 研究红蓝双边策略演进**。
- [ ] LATER **P7-01 冻结最终协议并完成论文实验**。

## 4. Phase 0：指标、归因与评估可信化

### P0-00 基础 MAPPO 算法数据流闭环

本节是所有后续实验的前置门槛。只修改当前基础策略入口 `training/train_no_rllib_mappo_robust_v2.py`、共享环境/模型及对应测试；不花时间修复已停用的 RLlib 和旧训练入口。若共享环境语义发生变化，当前单蓝车入口必须同步通过回归测试，但不在此阶段启动单蓝车训练。

#### P0-00A 入口与兼容性冻结

- [x] 基础策略正式入口冻结为 `training/train_no_rllib_mappo_robust_v2.py`。
- [x] Phase 1 单蓝车入口冻结为 `training/train_2red_1blue_ppo.py`。
- [x] `training/train_rllib.py`、`environment/carla_rllib_wrapper.py`、`training/train_no_rlib_sharedpolicy.py` 标记为历史入口，不再修复或用于论文实验。
- [x] `train_no_rllib_mappo.py`、robust v1 和 multi v1/v2 暂不作为本轮基础策略训练入口；未来阶段需要时单独迁移当前协议。
- [x] 在 README、`RUN_COMMANDS.md` 和 `FILE_INDEX.md` 中补充 CURRENT/DEPRECATED 标识、阶段门槛与唯一有效 smoke 命令，避免误启动旧入口。

#### P0-00B 终止与截断语义

> **实施状态：COMPLETED（2026-07-21）。** 状态简写：`A` = `done=False, terminated=False, truncated=False, active_mask=True`；`T` = `done=True, terminated=True, truncated=False, active_mask=False`；`X` = `done=True, terminated=False, truncated=True, active_mask=False`。

| 场景 | 修改前：直接到点/碰撞车辆 | 修改前：其他红车 | 修改前：蓝车 | P0-00B 已适配语义 | 合法 bootstrap |
|---|---|---|---|---|---|
| 正常运行，无终局事件 | `A` | `A` | `A` | 保持全部 `A` | 不涉及 |
| 单辆红车先到点，仍有红车未到点 | 到点红车 `T` | `A` | `A` | 保持：到点红车自然终止，其他车辆继续 | 到点红车否；其他车辆尚未形成边界 |
| 蓝车自行到点，红车任务仍在运行 | 到点蓝车 `T` | `A` | 其他蓝车 `A` | 暂保持局部 `T`；蓝车“到点”是否应作为有效对抗终局留到奖励审计确认 | 到点蓝车否 |
| 蓝车自身碰撞，且 `adversary_collision_ends_episode=False`（当前 Phase 1 配置） | 碰撞蓝车 `T` | `A` | 其他蓝车 `A` | 保持：只终止碰撞蓝车，红车继续 | 碰撞蓝车否 |
| 基础策略中一辆红车碰撞，触发 `cooperative_collision` | 碰撞红车 `T` | `X` | 不适用 | **已确认**：未碰撞红车的个人驾驶任务只是被同伴事件打断，保持 `X` | 未碰撞红车是；必须使用同帧有效最终状态 |
| 对抗训练中一辆红车碰撞，触发 `cooperative_collision` | 碰撞红车 `T` | `X` | 修改前为 `X` | **已确认并适配**：碰撞红车 `T`、未碰撞红车 `X`、全部蓝车按本轮对抗结局设为 `T` | 仅未碰撞红车是 |
| 基础策略中全部红车到点，触发 `all_agents_reached` | 全部到点红车 `T` | `T` | 不适用 | 保持全部红车 `T` | 否 |
| 对抗训练中全部红车到点，触发 `cooperative_success` | 全部到点红车 `T` | `T` | 修改前存活蓝车为 `X` | **已确认并适配**：蓝车未完成对抗目标，改为 `T` | 红车、蓝车均否 |
| 蓝车碰撞且 `adversary_collision_ends_episode=True`（当前有效入口未启用） | 碰撞蓝车 `T` | 修改前为 `X` | 其他蓝车修改前为 `X` | **确认本轮不适配**：保留旧兼容语义，并由回归测试防止误改；启用前再单独决策 | 碰撞蓝车否；其他车辆沿用旧分支 bootstrap |
| 达到 `max_steps`，无其他自然终止 | 所有存活车辆 `X` | `X` | `X` | 保持：外部时间限制截断 | 是，使用同帧有效最终状态 |
| 某 agent 已先自然终止，之后其他 agent 达到 `max_steps` | 已终止 agent 保持 `T` | 当时仍存活者 `X` | 当时仍存活者 `X` | 保持个体边界：既有 `T` 不得被覆盖成 `X` | 仅最终仍存活且被截断者是 |
| 所有 agent 均已分别自然终止，触发 `all_agents_terminated` | `T` | `T` | `T` | 保持全部 `T` | 否 |

- [x] 保留“蓝车自身碰撞只终止该蓝车、红车继续运行”的局部生命周期语义。
- [x] 基础策略 `cooperative_collision` 中，碰撞红车为 `T`，未碰撞红车为 `X`，后者保留个人驾驶回报 bootstrap。
- [x] 对抗训练 `cooperative_success` 中，全部到点红车和仍存活蓝车均为 `T`；蓝车不 bootstrap。
- [x] 对抗训练 `cooperative_collision` 时全部蓝车为 `T`；`adversary_collision_ends_episode=True` 未启用分支本轮不适配并保持旧语义。
- [x] 训练端直接消费并校验环境给出的 `terminated/truncated`，不再通过 `reached_target` 或碰撞 sticky 标志二次猜测 bootstrap 类型。
- [x] 所有 `X` 保存 `env.step()` 同帧返回的 `next_obs` 拼接状态；不得读取 reset 后状态或 despawn 后重新计算的状态。
- [x] 生命周期测试覆盖局部蓝车终止、单红车先到点、同伴红车碰撞导致 truncation、红车结局导致蓝车 termination、max_steps、旧兼容分支和 GAE bootstrap，共 15 项通过。

**验收门槛**：每个场景与确认后的矩阵完全一致；`T` 的 return 不 bootstrap，`X` 的 return 从同帧有效最终状态正确 bootstrap；任意 agent 的 `terminated` 与 `truncated` 不同时为真；已自然终止 agent 不得被后续全局事件改写。

#### P0-00C robust_v2 active-mask 数据流

> **实施状态：COMPLETED（2026-07-22）。** 本阶段只改变有效 agent-time transition 的采集资格；Actor 观测维度 19、centralized critic 状态维度 38、动作维度 3、奖励和 P0-00B T/X 语义均未改变。这里的 COMPLETED 是 mock/unit 验收，不代替 P0-00G 的真实 CARLA smoke。

- [x] `env.reset()` 后立即保存并严格校验 `active_mask = info['active_mask']`。
- [x] centralized critic value、actor action 和 `step_cache` 只为动作前 `active_mask=True` 的 agent 计算。
- [x] transition 只从当步 `step_cache` 写入；失活 agent 的全零 observation 不进入 buffer。
- [x] 每次 `env.step()` 后先写完本步 transition，再用 `info['active_mask']` 更新下一步 mask，不通过 `aid in obs_dict` 猜测存活状态。
- [x] 回合仍运行且 active 集合变化时，重新读取清理后的下一决策观测；原始 step 返回状态仍只作为合法 truncation bootstrap，避免 Actor/Critic 使用不同时序的状态。
- [x] episode 退出以 `episode_end_reason != 'running'` 为主；`running` 时无 active agent、终局时仍有 active agent、`active != not done` 均 fail-fast。
- [x] 每个 episode/agent 向 TensorBoard 记录 `active_decision_steps` 与 `buffer_transition_count`，运行时强制两者完全相等。
- [x] agent 提前到点或碰撞后，其 transition 数保持不变；导致其终止的最后一条真实 transition 保留且只保留一次。

| 五项评估依据 | 通过条件 | 验收结果/证据 |
|---|---|---|
| 1. 决策一致性 | 仅 `current_active_mask=True` 执行 value/action | PASS：mock 中 Actor 调用 `{0: 1, 1: 4}`，Critic 目标 agent 序列为 `[0, 1, 1, 1, 1]` |
| 2. 样本守恒 | 每 episode/agent 的 active 决策数等于新增 buffer 数 | PASS：`validate_episode_transition_counts()` 强制对账；mock 两侧均为 `{0: 1, 1: 4}` |
| 3. 终端步完整性 | active 变 inactive 时恰保留一条最终 transition | PASS：agent 0 唯一样本为非零观测，`done=1, terminated=1, reward=10` |
| 4. 失活隔离性 | 提前结束后 action/value/buffer/GAE/PPO batch 均零增长 | PASS：后续 action key 为 `{1},{1},{1}`，agent 0 Actor/Critic 不再调用、buffer 维持 1 条；所有后续训练数组只从该有效 buffer 构造 |
| 5. 协议一致性 | reason 驱动退出，T/X/bootstrap 回归不退化，无 active agent 静默丢弃 | PASS：陈旧返回观测标记值 `99` 在下一决策前刷新为清理后值 `3`；3 项 active-mask + 15 项 lifecycle/GAE 测试共 18 项；矛盾 reason/mask 和逐 agent `active != not done` 均触发异常 |

**验收结论**：构造“一车第 1 步终止、另一车继续 3 步”的 mock rollout，提前终止车辆仅保留 1 条真实终端 transition，额外 transition 数为 0；Actor/Critic 特征维度保持 `19/38`，仅 batch 行数动态变化。正式训练循环已接入同一组 helper 与运行时不变量，真实 CARLA 行为留待 P0-00G 验证。

#### P0-00D GAE、优势标准化与 PPO 更新

> **实施状态：COMPLETED（2026-07-23）。** 当前完成的是 `robust_v2` 的 synthetic/unit 验收；未启动 CARLA smoke。为避免越过阶段边界，仍导入旧 helper 的对抗/PSRO 入口暂时保留兼容语义，后续在各自阶段显式迁移。

- [x] 保留 episode 边界，逐 episode 计算**未标准化** GAE，禁止跨 episode 或跨 agent 传播 trace。
- [x] `mask_value = 1 - terminated`，`mask_trace = 1 - done`；只有合法 truncation 使用该 episode 的最终 value bootstrap。
- [x] 将同一 agent 的所有 episode advantage 拼接后，在整个 rollout 上统一标准化一次。
- [x] 不再逐 episode单独减均值/除标准差，避免抹掉成功、碰撞和稀疏 impact episode 之间的相对强弱。
- [x] critic return 使用原始 `advantage + old_value`，不得使用标准化后的 advantage 构造 return。
- [x] PPO 首次更新前用存储的 pre-tanh `z` 重算 log-prob，强制 `max_abs(ratio-1) < 1e-6`。
- [x] Actor/Critic 每个 minibatch 在 optimizer step 前后检查 loss、ratio、KL、clip fraction、梯度与参数有限；Actor gradient norm 按全部 minibatch 汇总。
- [x] 保留 agent-conditioned centralized critic；每个 agent 使用自己的 reward/GAE/return，不改成共享团队 return。

| 五项评估维度 | 合格标准 | 验收结果/证据 |
|---|---|---|
| 1. GAE 边界正确性 | 手算一致；T 不 bootstrap，X 使用最终 value；episode/agent 间零传播 | PASS：三步双 episode 轨迹得到 advantage `[1.985, 1.75, 5.6]`、return `[2.485, 2.0, 6.6]`；改变第二 episode 不影响第一 episode |
| 2. 标准化范围正确性 | 每个 agent 的完整 rollout 只标准化一次；非退化时均值 0、标准差 1 | PASS：成功/普通两 episode 拼接后统一标准化；结果不等于逐 episode 的 `[1,-1,1,-1]`；第二 agent 独立统计 |
| 3. Actor/Critic 目标分离 | `critic_return == raw_advantage + old_value`，标准化不改变 return | PASS：生产代码运行时强制 `allclose(..., atol=1e-6)`，synthetic test 保留原始 return |
| 4. PPO 概率与数值健康 | 更新前 `max_abs(ratio-1)<1e-6`；loss/KL/clip/梯度/参数有限 | PASS：未变 Actor 通过，篡改参数会 fail-fast；一次 Actor/Critic optimizer step 全部有限，clip fraction 在 `[0,1]` |
| 5. 集成与兼容性 | 原 18 项测试不退化，特征/权重形状不变，当前入口正常 | PASS：新增 7 项 + 原 18 项共 25 项；兼容 wrapper 保持未迁移入口的旧输出；`py_compile`、`robust_v2 --help`、`git diff --check` 通过；网络与 checkpoint 张量形状未变 |

**验收结论**：P0-00D 的五项门槛全部通过。真实 CARLA 下的 advantage/KL/clip/gradient 分布以及性能变化仍须在 P0-00G smoke 验证；本结论不等价于训练效果已经提升。

#### P0-00E 算法回归测试

> **实施状态：COMPLETED（2026-07-26）。** 本阶段只完成无 CARLA 的算法级回归；真实环境数据分布、训练吞吐与三更新端到端行为仍归入 P0-00G。

目标文件：`tests/test_mappo_algorithm.py`，必要时为 rollout 提取可注入 mock 的纯函数/小型 collector。

- [x] 测试真正终止不 bootstrap、max_steps 截断 bootstrap、episode 边界不串联。
- [x] 测试多 agent 数据不交错计算 GAE，每个 agent 的数组长度和时间顺序正确。
- [x] 测试失活 agent 不再采样动作、不写 buffer、不参与后续 transition 计数；无样本 agent 也不进入 Actor/Critic 更新。
- [x] 测试 `sample_action_z()` 与 `evaluate_actions_z()` 在参数不变时 ratio 为 1。
- [x] 测试 `AgentConditionedCritic` 的 batch/int agent id 调用、输出 shape、有限值和非法输入拒绝。
- [x] 用纯 synthetic ragged batch 执行完整生产 actor/critic optimizer path，断言输入不变、参数按归属更新且全部指标有限。
- [x] 运行 `python -m unittest discover -s tests -v`，旧生命周期测试和新增算法测试共 35 项全部通过。

| 五项评估维度 | 合格标准 | 验收结果/证据 |
|---|---|---|
| 1. 数据归属、维度与顺序 | 每个 agent 的 Actor batch 独立且时间顺序不变；Critic 拼接顺序与 agent-id 一一对应；特征维度固定 | PASS：ragged 样本数 `{0:3, 1:5}`，Actor 为 `[N_i,19]/[N_i,3]`，Critic 为 `[8,38]`；哨兵序列和 id `[0,0,0,1,1,1,1,1]` 精确匹配 |
| 2. GAE、标准化与目标口径 | T/X bootstrap、episode 隔离、per-agent 标准化及 raw critic return 不回退 | PASS：复用 lifecycle/GAE 既有 22 项相关断言；生产 batch 再校验 `ret == raw_adv + old_value`，非退化 Actor advantage 均值 0、标准差 1 |
| 3. Actor 概率与 PPO clipping | 未更新 ratio≈1；裁剪目标与手算一致；KL/clip/entropy/loss 有限 | PASS：生产更新前逐 agent ratio 误差 `<1e-6`；手算四样本 clipped loss=`0.15`、clip fraction=`1.0`；阈值边界受测 |
| 4. 共享 Critic 的 agent 条件语义 | int 与 batch id 结果一致；mixed id 逐行等价；空 agent 不制造 Critic 样本 | PASS：输出 shape `[B]` 且有限；越界、长度、dtype、state 维度错误 fail-fast；空 agent 的 Critic id 仅含有效 agent |
| 5. 更新健康、故障隔离与集成 | 完整生产路径更新正确模型、不改写 rollout；异常在任何 step 前失败；旧测试不退化 | PASS：两轮 ragged update 的 Actor/Critic 参数均有限并按样本数更新，输入 tensor 不变；NaN batch/非法 clip 均零参数污染；全量 35 项通过 |

**验收结论**：P0-00E 的五项门槛全部通过。测试不再复制一套“看起来相同”的 PPO 公式，而是直接调用 `robust_v2` 训练主循环使用的 batch preparation、loss 与 optimizer helper；因此后续生产逻辑改动会直接触发算法回归。该结论不替代 P0-00G 的真实 CARLA smoke。

#### P0-00F 基础策略 checkpoint 与恢复训练

> **实施状态：COMPLETED（2026-07-27）。** 当前完成无 CARLA 的协议与状态连续性验收；CARLA/Traffic Manager 跨进程不承诺轨迹逐位一致，真实恢复行为归入 P0-00G。

- [x] 移除 `robust_v2` 按训练 rollout 窗口成功率自动覆盖 `best.pt` 的规则。
- [x] 训练期只保存原子 `latest.pt`、安全边界 `interrupted.pt` 和不可覆盖的 `periodic/ckpt_update_*.pt`，并为每个文件写 SHA256。
- [x] checkpoint 保存 Actor、agent-conditioned Critic、全部 optimizer、update、args、累计统计、课程状态、LR schedule、协议版本及 Python/NumPy/Torch/CUDA RNG。
- [x] resume 在环境初始化前检查 schema、RunID、manifest、checksum、Git SHA 和显式配置漂移；环境创建后继续校验 observation/action/state 维度、agent 数量、reward/lifecycle、场景池和网络/optimizer 完整性。
- [x] `best_base_valid.pt` 只保留为后续独立 C0/C1 验证产物；训练代码与测试明确禁止创建 `best.pt`。
- [x] checkpoint 写入 RunID、Git SHA/dirty paths、关键源码指纹、完整命令、配置路径/哈希、来源 checkpoint、Python/Torch/NumPy/CUDA/CARLA 版本；代码漂移历史只追加不清除；旧 checkpoint 只能作 pretrained weights/历史评估，不能正式 resume。
- [x] fresh、pretrained、resume 三种语义互斥；`--resume latest` 不存在或损坏时 fail-fast，不再静默从零开始。
- [x] 将 LR decay horizon 与本次 `--updates` 停止位置解耦；resume 可延长停止 update，但不能隐式改变既有 LR 曲线。
- [x] 同一 Run TensorBoard 固定写入 `tb/`，resume 使用 `purge_step=next_update` 保持步数连续。

**验收门槛**：中断恢复后 update、optimizer、课程和 RNG 连续；同一 periodic checkpoint 的哈希固定；训练脚本不再创建含混的 `best.pt`。

| 五项评估维度 | 合格标准 | 验收结果/证据 |
|---|---|---|
| 1. 状态完整性 | 模型、全部 optimizer、统计、课程、LR 和 RNG 无缺项，缺项不得部分恢复 | PASS：schema 强制 required fields 和 Actor/optimizer agent-id 全覆盖；缺 Critic optimizer/Actor 测试均 fail-fast |
| 2. 恢复连续性 | Python/NumPy/Torch RNG 下一组相同；optimizer 恢复后的下一步与不中断路径一致；update/LR horizon 连续 | PASS：三类 RNG 数组逐值相等，Adam round-trip 下一步参数逐 tensor `torch.equal`；LR horizon 从 checkpoint 恢复且显式漂移被拒绝 |
| 3. 兼容安全性 | schema、维度、agent、reward/lifecycle、训练配置、场景池或 Git 不兼容时不得进入训练 | PASS：协议、state dim、gamma、agent-id、checksum、RunID/manifest 均有严格校验；resume/pretrained 冲突在 CARLA 初始化前失败 |
| 4. 文件可靠性 | latest 原子替换，periodic 不覆盖，interrupted 复制最近完整边界，损坏可检测 | PASS：Windows 临时文件+fsync+`os.replace` 受测；periodic 二次保存抛错且哈希不变；篡改文件在 `torch.load` 前被 SHA256 拒绝 |
| 5. 实验可追溯与集成 | 无训练期 best；Run/source/version/命令可追溯；原算法测试不退化 | PASS：源码断言不存在 `best.pt/best_success_rate`；manifest/metadata 完整；新增 16 项 + 原 35 项共 51 项通过 |

**验收结论**：P0-00F 的五项单元门槛全部通过。中断恢复只发生在完整 update 事务边界，未完成 rollout/PPO 数据明确丢弃；软件随机序列和 optimizer 可精确续接。P0-00G 必须进一步验证真实 CARLA 下 `latest/periodic/interrupted`、next update、LR、课程、TensorBoard 和车辆清理，不得据本阶段直接启动长训练。

#### P0-00G 真实 CARLA smoke

- [ ] 使用唯一 RunID 运行 `robust_v2`：1 seed、3 updates、每 update 4 个 episode，不恢复旧 checkpoint。
- [ ] 保存每个 episode 的 `episode_end_reason`、per-agent active steps、transition 数、terminated/truncated 和 bootstrap 是否启用。
- [ ] 验证 `active_decision_steps == buffer_transition_count`，且失活 agent 后续增量为 0。
- [ ] 验证首次 PPO epoch ratio≈1，KL/clip fraction/entropy/value loss/grad norm 均有限且数量级可解释。
- [ ] 验证 `latest.pt` 每 update 原子更新、`periodic/ckpt_update_*.pt` 不覆盖，checksum/manifest 一致，且 payload 含 Actor、agent-conditioned Critic、optimizer、update、课程、LR、RNG、RunID/Git 和协议版本。
- [ ] 使用同一 `save_dir --resume latest --updates 4` 恢复 1 个 update，检查 next update、LR、课程、累计统计和 TensorBoard `tb/` 连续；中断测试确认 `interrupted.pt` 等于最近完整边界。
- [ ] 人工查看至少 3 个 episode，确认提前到点、碰撞和 max_steps 至少覆盖两类；未覆盖的类型另用固定场景补测。

**P0-00 完成定义**：P0-00B 至 P0-00G 全部通过，测试输出和 smoke Run 已登记；此时仅证明基础 MAPPO 训练数据流可信，不代表基础策略性能已经达标。

### P0-A 已实现基础与复审项

- [x] 环境输出 `active_mask` 和 `episode_end_reason`。
- [x] 蓝车碰撞后只结束/销毁该蓝车，红车可继续到达终局。
- [x] 红车成功是蓝车对抗任务终局，仍存活蓝车按 terminated 处理，不再 bootstrap。
- [x] 红车严重碰撞时：碰撞红车 terminated、未碰撞红车 truncated、全部蓝车 terminated；仅未碰撞红车从有效同帧状态 bootstrap。
- [x] 最大步数使用 truncation 语义。
- [x] 保存蓝车碰撞前运动学快照，避免销毁后指标丢失。
- [x] `tests/test_episode_lifecycle.py` 的 15 项测试通过，P0-00B 新语义、协议不变量和 GAE bootstrap 已完成 mock 验收。
- [ ] VERIFY 在真实 CARLA 运行中验证上述生命周期；统一归入 P0-00G smoke，不阻塞 P0-00C 静态修复。

### P0-B 碰撞归因

涉及文件：[环境](../environment/carla_multi_agent_env.py)、[奖励](../training/adversarial_reward.py)、[指标](../utils/adversarial_metrics.py)。

- [ ] 在 collision callback 中保存 `other_actor.id`、`type_id`、语义类别和碰撞帧。
- [ ] 将 actor id 映射到 `agent0/agent1/...`，并写入当步 `info`。
- [ ] 蓝车碰墙：只记蓝车自身碰撞，不记 `impact_success`。
- [ ] 蓝车撞红车：根据碰撞前状态判定 induced/collateral/kamikaze，但不靠“最近红车”猜测对象。
- [ ] 蓝蓝碰撞：双方承担自身碰撞代价，不发放对红车的 impact 奖励。
- [ ] 多个同帧碰撞事件采用确定性的去重规则。
- [ ] 为红蓝、蓝蓝、墙体和未知 actor 四类情况添加测试。

**验收门槛**：构造事件中的碰撞类型与输出标签 100% 一致；墙体和蓝蓝测试中 `impact_reward == 0`。

### P0-C 运动学与奖励测试

- [ ] 修复 ndarray/list/dict/`VehicleControl` 四种 control 输入的统一解析。
- [ ] 测试 `steer`、纵向速度、横向速度、相对距离和 TTC 的数值方向。
- [ ] 测试威胁门控：不在红车前方、距离过远、出生保护期内均不得累积错误 threat。
- [ ] 测试单 episode threat cap。
- [ ] 测试 impact、threat、base-safe、自碰撞惩罚各分量可分别关闭。
- [ ] 测试所有奖励与指标均为有限值，不产生 NaN/Inf。
- [ ] 保存一个固定 synthetic trajectory 作为回归样例。

**验收门槛**：新增测试全部通过；同一固定轨迹的奖励分解和指标输出可重复。

### P0-D C0/C1/C2 统一评估器

目标文件：`evaluation/evaluate_adversarial_c012.py`。

- [ ] 定义 C0：仅两辆冻结红车，无第三车。
- [ ] 定义 C1：两辆冻结红车 + 非学习交通车；固定行为参数和生成位置。
- [ ] 定义 C2：两辆冻结红车 + 学习型蓝车。
- [ ] 三个条件复用同一 `scenario_id + episode_seed` 配对表。
- [ ] 支持 `--conditions c0,c1,c2`、`--seeds-file`、`--checkpoint`、`--output-dir`。
- [ ] 每 episode 输出 JSONL：条件、seed、场景、终局、红车成功、红车碰撞、蓝车碰撞、超时、最小间距、forced braking、swerve、impact、威胁量和 episode 长度。
- [ ] 汇总输出 JSON/CSV：比率、均值、标准差、配对差值和 bootstrap 95% CI。
- [ ] 失败原因互斥，并保留 `unknown` 计数用于发现环境漏洞。
- [ ] 记录评估协议版本、Git SHA、检查点 SHA256 和命令行参数。
- [ ] 生成可重放的失败 episode seed 清单。

**验收门槛**：

- 成功、碰撞、超时/其他比例之和为 1，允许的浮点误差 `< 1e-6`。
- 相同检查点、seed 和协议重复两次，终局与离散事件一致。
- C0 能复现基础红车已知能力；若 C0 明显失败，停止对抗训练并先修复基础策略。
- C1 与 C0 的差值被单独报告。

### P0-E 检查点选择和实验注册

- [ ] 移除或弃用 `adv_effectiveness = 1 - win_succ` 的 best 选择语义。
- [ ] 训练脚本只自动保存 `latest.pt`、`interrupted.pt` 和 `ckpt_update_*.pt`。
- [ ] 独立验证器按以下顺序选择模型：安全门槛 -> 配对攻击效果 -> 次级行为指标。
- [ ] 创建 `experiments/registry.csv`，每个 Run 一行。
- [ ] 创建 `configs/`，将正式实验参数固化为可版本控制的 YAML/JSON。
- [ ] 创建 `scripts/run_experiment.ps1`，负责生成 RunID、保存命令和 Git SHA。
- [ ] 创建 `scripts/validate_run.py`，检查必备元数据、检查点和结果文件是否齐全。

**Phase 0 完成定义**：P0-00、P0-B 至 P0-E 全部完成，测试通过，且至少完成一次 C0/C1/C2 基线评估。

## 5. Phase 1：单蓝车对抗有效性

### P1-A 训练前冻结项

- [ ] 选择一个由 P0-00 修复后的 `train_no_rllib_mappo_robust_v2.py` 产生、且 C0 成功率稳定的红车 checkpoint，并记录 SHA256。
- [ ] 固定首轮场景为 `scenario_01_basic.json`；泛化评估再覆盖其他场景。
- [ ] 固定训练 seeds：`42, 123, 2024`。
- [ ] 固定评估 seed 文件，训练过程不得修改。
- [ ] 固定 reward version 和全部权重到配置文件。
- [ ] 确认旧目录 `logs/train_2red_1blue_ppo` 不被覆盖。
- [ ] 确认 2026-07-05 旧 checkpoint 不作为新实验的 resume 起点。

### P1-B 逐级运行

- [ ] 先运行 3 updates smoke test：检查进程、显存、日志、episode 终局和 checkpoint。
- [ ] smoke 后抽取至少 10 个 episode，确认蓝车不是出生即撞或长期静止。
- [ ] 分别训练 3 个 seed 到 30/60/120 updates，并保留所有里程碑 checkpoint。
- [ ] 每个里程碑运行 C0/C1/C2 配对评估，不在训练 rollout 上选模型。
- [ ] 开启 TensorBoard 监控 policy loss、value loss、entropy、KL、clip fraction、collision 和 reward components。
- [ ] 出现 NaN/Inf、终局比例不守恒或日志缺失时立即停止。
- [ ] 连续 3 个评估窗口蓝车碰撞率超过 30% 时停止并检查奖励，不继续堆训练时长。

### P1-C 主结果与消融

- [ ] 主方法：impact + gated threat + base-safe + 自适应自碰撞惩罚。
- [ ] 消融 A：去掉 impact。
- [ ] 消融 B：去掉 threat gate/cap。
- [ ] 消融 C：去掉 base-safe。
- [ ] 消融 D：去掉 KL 正则或比较不同 `beta-kl`。
- [ ] 对每个方法报告 C0、C1、C2 和 `C2-C1`，不能只报蓝车 reward。
- [ ] 保存成功 cut-in、blocking、forced braking 各至少 3 个可复现 seed/视频。
- [ ] 保存失败样例：kamikaze、无效尾随、自撞、静止投机。

**Phase 1 验收门槛**：

- 配对 `C2-C1` 红车成功率差的 95% CI 不包含 0，且方向为负。
- 蓝车自身碰撞率 `<= 5%`。
- kamikaze 比例 `<= 10%`。
- 至少一种非碰撞干扰机制在重放中可确认：切入、阻挡、迫使制动或迫使偏转。
- 效果能在至少一个未参与训练的场景上保留，而非只记忆出生位置。

若未通过：优先检查指标和奖励投机，再调整奖励；不得直接进入双蓝车或 PSRO。

## 6. Phase 2：红车基础策略池

目标：构建既“会正常驾驶”又“行为有差异”的红车集合，为对抗泛化和 PSRO 提供有效对手分布。

- [ ] 核对 [鲁棒红车训练](../training/train_no_rllib_mappo_robust.py) 的 episode 生命周期与当前环境一致。
- [ ] 校验 17 个场景 JSON 的 schema、出生点合法性和难度标签。
- [ ] 使用 seeds `42, 123, 2024` 训练红车，并保留中期与最终 checkpoint。
- [ ] 对每个候选红车运行无对抗 C0 评估，淘汰基础驾驶能力不足者。
- [ ] 对每个候选运行 C1 评估，淘汰普通交通下极不稳定者。
- [ ] 提取行为特征：速度、加速度、制动率、转向幅度、车间距、到达时间和碰撞类型。
- [ ] 计算策略两两行为距离矩阵，删除近重复 checkpoint。
- [ ] 每个 pool member 记录来源 RunID、训练 seed、update、场景分布和质量指标。
- [ ] 冻结 `red_pool_manifest.json`，后续固定池实验不得静默替换成员。

**Phase 2 验收门槛**：

- 至少 6 个合格红车策略，而不是同一次训练的近重复快照。
- 每个成员满足预先冻结的 C0/C1 质量门槛。
- 行为距离矩阵显示多个可解释的驾驶风格簇。
- 池成员、权重文件和评估结果可由 manifest 完整追溯。

## 7. Phase 3：双蓝车协同对抗

涉及文件：[多对抗训练](../training/train_multi_adversarial_kl_risk.py)、[多对抗评估](../evaluation/evaluate_multi_adversarial.py)。

- [ ] 将 `AdversarialRewardComputer` 作为单蓝和双蓝的唯一奖励实现。
- [ ] 删除 `train_multi_adversarial_kl_risk.py` 中旧的内联 reward 公式。
- [ ] 所有 rollout、GAE 和 done 判断使用 `active_mask`、termination/truncation 和 `episode_end_reason`。
- [ ] 蓝车死亡后不再进入动作采样、buffer、价值 bootstrap 或团队奖励平均。
- [ ] 实现每辆蓝车独立的 impact/threat/self-collision 归因。
- [ ] 团队奖励只共享对红车的有效影响；蓝蓝碰撞不能变成正向团队信号。
- [ ] 记录两辆蓝车各自的有效影响率，检查是否出现一辆工作、一辆闲置。
- [ ] 先训练“无显式通信”的双蓝基线，为 Phase 5 建立对照。
- [ ] 统一多对抗评估器与 C0/C1/C2 指标协议。
- [ ] 比较：单蓝车、双蓝车独立策略、双蓝车参数共享策略。

**Phase 3 验收门槛**：

- 两辆蓝车均有非零、可重复的有效影响贡献。
- 双蓝车相对单蓝车的 `C2-C1` 改善显著，而蓝车碰撞约束不恶化。
- 蓝蓝碰撞率和闲置率低于冻结门槛。
- 结果来自至少 3 个 seed，并能重放协同夹击、前后阻挡或接力干扰行为。

## 8. Phase 4：固定策略池 PSRO

已有代码位于 [training/psro](../training/psro/README.md)。先完成固定红池的蓝方 PSRO，再考虑双边更新。

- [ ] 将 [蓝方 oracle](../training/psro/train_blue_oracle.py) 迁移到统一奖励与生命周期协议。
- [ ] 重新定义 payoff：保留完整向量指标，并在安全门槛内构造标量效用；避免单纯使用 `1 - red_success`。
- [ ] payoff 每个单元先用开发期 `>=20` 个配对 episode，最终用 `>=40` 个。
- [ ] 验证 payoff 表行列均有方差；若退化为常数，停止并检查池或指标。
- [ ] 验证 meta-solver 输出合法概率分布且随新 oracle 改变。
- [ ] 每轮保存 population manifest、payoff matrix、meta strategy、oracle config 和评估原始记录。
- [ ] 运行 5-7 轮蓝方 PSRO，记录 exploitability 或近似 best-response gap。
- [ ] 比较 InRL/单一对手训练、uniform pool sampling 和 PSRO meta-strategy sampling。
- [ ] 明确 JPC 的适用语义；若只有固定红池和扩增蓝池，不把普通泛化矩阵错误称为完整双边 JPC。
- [ ] 对历史 PSRO 碰撞退化结果保留记录，但不与新协议结果混合汇总。

**Phase 4 验收门槛**：

- 至少 2 轮新增 oracle 在独立评估上提供非冗余收益。
- meta strategy 不长期退化为单一固定策略，或能解释为何理论上应退化。
- PSRO 相对 uniform 和单一对手基线提升跨红策略泛化，并满足安全约束。
- payoff、策略池和所有权重能够完整复现。

## 9. Phase 5：对抗车辆显式通信

本阶段研究“通信是否带来额外协同收益”，不能把 centralized critic 当作显式通信贡献。

### P5-A 最小可验证架构

- [ ] 每辆蓝车用本地观测编码 `h_i = Encoder(o_i)`。
- [ ] 生成受限维度消息 `m_i = MessageHead(h_i)`，首版建议 4-8 维。
- [ ] 聚合其他蓝车消息：两车场景先用 mean/sum，后续再考虑 attention。
- [ ] actor 输入为本地表示和接收消息；执行阶段不能访问红车全局特权状态。
- [ ] centralized critic 仍可访问训练期全局状态，但与消息通道分开实现和记录。
- [ ] 记录消息均值、方差、相关性和带宽，检查消息是否塌缩为常数。

### P5-B 必需对照

- [ ] No-Comm：与 Phase 3 相同的双蓝基线。
- [ ] Comm：可学习消息通道。
- [ ] Zero-Message：结构相同但消息置零，排除参数量影响。
- [ ] Shuffled-Message：打乱队友消息，验证语义依赖。
- [ ] Dropout/Noise：模拟通信丢包和噪声。
- [ ] Bandwidth：比较 2/4/8/16 维消息。
- [ ] 可选：CommNet、TarMAC/attention 等方法只在最小架构有效后加入。

**Phase 5 验收门槛**：

- Comm 相对 No-Comm 的配对攻击效果有统计改善。
- Zero/Shuffled 对照不能保持相同收益，否则不能证明消息被有效使用。
- 收益在合理丢包/噪声下不过度崩溃。
- 通信没有依赖执行期不可得的全局特权信息。

## 10. Phase 6：红蓝双边演进

这是研究点 2 的完整形态，风险和计算成本最高，放在固定红池 PSRO 后执行。

- [ ] 定义红方 best response：在蓝方 meta-strategy 上继续提高鲁棒性。
- [ ] 红方更新时混入 C0/C1 rehearsal，避免只会躲当前蓝车而丧失基础驾驶能力。
- [ ] 使用 KL/行为克隆约束限制红方灾难性遗忘。
- [ ] 红方新策略必须先过 C0/C1 质量门槛才能加入池。
- [ ] 蓝方新策略必须先过安全门槛才能加入池。
- [ ] 每轮交替添加一个合格 best response，更新双边 payoff 和 meta-strategy。
- [ ] 记录策略循环、支配关系、池增长和近似 exploitability。
- [ ] 设置最大轮数和无改进停止条件，防止无限扩池。

**Phase 6 验收门槛**：红方鲁棒性提升不以明显牺牲 C0/C1 为代价；蓝方仍能持续发现新的非碰撞失败模式；双边池优于单轮 adversarial training。

## 11. Phase 7：论文最终实验与材料

### P7-A 冻结实验协议

- [ ] 冻结代码 commit、CARLA 版本、地图、天气、场景、seed 文件和指标版本。
- [ ] 冻结所有方法的训练预算，保证基线与主方法比较公平。
- [ ] 最终实验使用建议 5 个训练 seed；每个条件至少 40 个配对评估 episode。
- [ ] 预先写明主要指标、次要指标和失败判据，避免看结果后改口径。

### P7-B 最小论文实验矩阵

- [ ] 基础红车：C0/C1 能力和鲁棒性。
- [ ] 单蓝车：自然交通 vs 学习对抗，含奖励消融。
- [ ] 双蓝车：单蓝 vs 双蓝，无通信基线。
- [ ] 通信：No-Comm、Comm、Zero/Shuffled、带宽和鲁棒性消融。
- [ ] 策略池：单一对手、uniform pool、PSRO。
- [ ] 双边演进：单轮 adversarial training vs alternating PSRO/co-evolution。
- [ ] 泛化：未见场景、未见红策略和必要时的天气/扰动变化。

### P7-C 论文产物

- [ ] 汇总表：成功率、碰撞率、超时率、蓝车自碰撞、kamikaze、forced braking、swerve、最小间距。
- [ ] 统计表：均值、标准差、95% CI、配对差值和效应量。
- [ ] 曲线：训练过程、PSRO payoff/meta-strategy、池规模和泛化性能。
- [ ] 图示：CTDE、奖励归因、PSRO 循环、通信架构。
- [ ] 案例：成功与失败轨迹、俯视图、关键帧和视频。
- [ ] 附录：超参数、seed、场景配置、硬件、运行时和复现实验命令。
- [ ] 威胁与局限：sim-to-real、地图单一性、规则型 C1、计算预算和碰撞传感器误差。

**课题完成定义**：两个研究点均有明确方法、强基线、消融、独立评估、统计结果和可复现证据；通信作为独立变量得到验证，而不是只作为网络结构描述。

## 12. 实验运行等级与停止条件

| 等级 | 用途 | 建议规模 | 是否可形成结论 |
|---|---|---|---|
| L0 单元测试 | 验证纯函数和环境语义 | 秒级，无 CARLA 或最小 mock | 否 |
| L1 Smoke | 检查端到端可运行 | 1 seed，3 updates，少量 episode | 否 |
| L2 Dev | 比较方向和排查奖励 | 3 seeds，20-30 配对评估 episode | 仅内部决策 |
| L3 Milestone | 阶段验收 | 3 seeds，>=40 配对 episode | 可作阶段结果 |
| L4 Final | 论文最终结果 | 建议 5 seeds，>=40 配对 episode | 是 |

立即停止当前实验并登记原因的条件：

- 任何 loss、observation、action、reward 或 metric 出现 NaN/Inf。
- 成功/碰撞/超时比例不守恒，或出现无法解释的大量 `unknown`。
- 检查点、配置、Git SHA 或 seed 文件未保存。
- 蓝车通过高自撞/kamikaze 获得主要收益。
- C0 基础红车能力失效。
- CARLA 同步模式丢 tick、sensor frame 严重错位或 actor 未清理导致结果不可比。
- 代码在实验运行期间发生改变但 Run 元数据仍指向旧 SHA。

## 13. 标准命令

在 `PythonAPI/examples/mappo_training` 目录执行。运行 CARLA 的长实验前，应先完成不依赖 CARLA 的测试。

```powershell
# 当前已验证：episode 生命周期测试
..\..\..\venv\Scripts\python.exe -m unittest tests.test_episode_lifecycle -v

# 后续应扩展为完整单元测试
..\..\..\venv\Scripts\python.exe -m unittest discover -s tests -v

# P0-00G 基础 MAPPO smoke；只允许在 P0-00B 至 P0-00F 全部通过后运行
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --run-id P0_00G_robust_v2_smoke_s42_<timestamp> `
  --updates 3 `
  --lr-decay-horizon 3 `
  --rollout-episodes 4 `
  --min-transitions-per-update 64 `
  --max-steps 300 `
  --seed 42 `
  --save-dir logs/P0_00G_robust_v2_smoke_s42_<timestamp>

# 同一 Run 严格恢复第 4 个 update；冻结的 LR horizon 仍为 3
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --resume latest `
  --updates 4 `
  --save-dir logs/P0_00G_robust_v2_smoke_s42_<same-timestamp>

# 单蓝车 smoke 示例；替换 checkpoint 与唯一输出目录
..\..\..\venv\Scripts\python.exe -m training.train_2red_1blue_ppo `
  --pretrained-ckpt <red_checkpoint.pt> `
  --scenario scenario_01_basic.json `
  --updates 3 `
  --rollout-episodes 4 `
  --no-viz `
  --seed 42 `
  --save-dir logs/P1_single_blue_basic_s42_<timestamp>

# C0/C1/C2 评估命令在 evaluate_adversarial_c012.py 实现后固定于此
..\..\..\venv\Scripts\python.exe -m evaluation.evaluate_adversarial_c012 `
  --checkpoint <blue_checkpoint.pt> `
  --conditions c0,c1,c2 `
  --seeds-file configs/eval_seeds_v1.json `
  --output-dir <run_dir>/evaluation
```

每次长实验前检查：

```powershell
git status --short
git rev-parse HEAD
```

## 14. 实验登记表

每次实验结束后追加一行。详细原始数据留在 Run 目录，本表只记录决策所需摘要。

| 日期 | RunID | Phase | 假设/目的 | Commit | 脚本/配置 | 基础 ckpt | Seeds | C0/C1/C2 | 蓝碰撞/Kamikaze | 结果 | 证据路径 | 下一步 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-06-24 | `HIST_PSRO_20260624` | P4 历史 | 验证 PSRO 工程链路 | `6a332da` 前后版本 | `training.psro.*` | 历史红池 | 历史配置 | 非当前统一协议 | 碰撞攻击明显 | runner/pool/payoff 可运行，但科学结果退化，不能作主结论 | `logs/psro*`、`training/psro/` | 奖励和评估统一后重跑 |
| 2026-07-05 | `HIST_P1_20260705_s42` | P1 历史 | 2 红 1 蓝训练 | `6a332da` | `training.train_2red_1blue_ppo` | 见旧 run | 42 | 未完成正式 C012 | 旧生命周期口径 | 目标 120 updates，约 update 4 中断；不可恢复为新协议实验 | `logs/train_2red_1blue_ppo/` | 新 RunID 重新训练 |
| 2026-07-14 | `FIX_LIFECYCLE_20260714` | P0 | 统一 episode 生命周期 | `9cc4cb6` | 环境/训练/PSRO/tests | N/A | N/A | N/A | N/A | 7 个单元测试通过；尚无修复后 CARLA 实验 | `tests/test_episode_lifecycle.py` | 完成 P0-01 至 P0-06 |
| YYYY-MM-DD | `<RunID>` | `<P?>` | `<一句话可证伪假设>` | `<sha>` | `<config path>` | `<sha256/path>` | `<seeds>` | `<rates/deltas>` | `<rates>` | `<pass/fail>` | `<run path>` | `<action>` |

## 15. 代码修改记录

每个具备独立行为意义的修改追加一行；纯格式化可不单列。

| 日期 | Commit/分支 | 修改文件 | 行为变化 | 验证方式 | checkpoint 兼容性 | 关联任务/Run |
|---|---|---|---|---|---|---|
| 2026-07-14 | `9cc4cb6` | env、metrics、reward、3 个训练脚本、lifecycle tests | 蓝车碰撞不再直接终止红车 episode；增加终止原因与碰撞前快照 | 7 个 lifecycle 单元测试通过 | 需用新实验验证，不默认兼容旧训练语义 | P0-A |
| 2026-07-20 | `P0-00A` | `README.md`、`RUN_COMMANDS.md`、`FILE_INDEX.md`、研究 TODO | 冻结 `robust_v2`/`train_2red_1blue_ppo` 为当前唯一入口；旧 RLlib/shared/minimal/robust v1/旧对抗入口显式停用 | 文档入口与命令一致性检查 | 不修改 checkpoint；旧入口产物仍不得用于新协议 resume | P0-00A |
| 2026-07-21 | `P0-00B-design-draft` | 研究 TODO | 增加逐场景 agent 状态矩阵；记录同伴红车碰撞的 agent-centric truncation 与红车全部到点时蓝车 termination；尚未修改运行时 | 人工核对状态矩阵，待用户最终确认 | 尚无运行时或 checkpoint 变化 | P0-00B |
| 2026-07-21 | `P0-00B` | 共享环境、`lifecycle_semantics.py`、`robust_v2`、单蓝车入口、lifecycle tests、`FILE_INDEX.md`、研究 TODO | 按确认矩阵发布逐角色 T/X；训练端强校验协议并只为 X 保存同帧 bootstrap state；边界与 bootstrap 数量 fail-fast | 15 项 lifecycle/GAE 测试通过；两个当前入口 `--help` 正常；`git diff --check` 通过 | 网络结构未变；旧权重可评估，但旧 run 不得按新终止协议直接 resume | P0-00B |
| 2026-07-22 | `P0-00C` | `robust_v2`、`tests/test_mappo_active_mask.py`、MAPPO active-mask 原理文档、研究 TODO | collector 以动作前 active mask 过滤 value/action/cache；写完本步后更新 next mask；局部退场时刷新清理后的下一决策观测；保留最终 transition；逐 episode/agent 对账决策数与写入数 | 3 项 active-mask + 15 项 lifecycle/GAE，共 18 项通过；`robust_v2 --help`、重定向缓存的 `py_compile`、`git diff --check` 通过 | Actor/Critic 网络与特征维度未变，权重形状兼容；修复前 run 不作为新协议正式续训基线 | P0-00C |
| 2026-07-23 | `P0-00D` | `robust_v2`、`tests/test_mappo_gae_ppo.py`、MAPPO GAE/PPO 原理文档、研究 TODO | 逐 episode 计算原始 GAE；同一 agent rollout 统一标准化；Critic 使用 raw return；更新前 ratio 对账；minibatch 数值 fail-fast | 7 项 GAE/PPO + 原 18 项，共 25 项通过；兼容 wrapper 测试、`py_compile`、`robust_v2 --help`、`git diff --check` 通过 | 网络/权重形状兼容，但优化目标口径改变；修复前 run 不作为新协议正式 resume 基线 | P0-00D |
| 2026-07-26 | `P0-00E` | `robust_v2`、`models/actor_critic.py`、`tests/test_mappo_algorithm.py`、MAPPO GAE/PPO 原理文档、研究 TODO | 抽取并复用生产 batch preparation/clipped loss/完整 optimizer update；严格校验 ragged agent 顺序、Critic id、维度、有限值、optimizer 归属与超参数；空 agent 跳过更新 | 新增 10 项 production algorithm tests；全量 35 项通过，AST 语法检查、`robust_v2 --help`、`git diff --check` 通过 | Actor/Critic 网络参数形状不变，权重张量兼容；训练更新执行路径重构，旧 run 仍不作为新协议正式 resume 基线 | P0-00E |
| 2026-07-27 | `P0-00F` | `checkpoint_protocol.py`、`robust_v2`、`tests/test_checkpoint_resume.py`、checkpoint 协议/RUN_COMMANDS/README/FILE_INDEX/研究 TODO | 版本化 checkpoint；latest 原子更新、periodic immutable、interrupted 安全边界；严格 resume 恢复模型/optimizer/课程/LR/RNG；记录 Run/Git/config/source/runtime；移除训练期 best | 新增 16 项 checkpoint tests；与原 35 项共 51 项通过；全部 MAPPO 模型/optimizer round-trip、RNG/Adam 下一步一致、SHA 损坏与配置漂移 fail-fast；AST、`--help`、`git diff --check` 通过 | 新格式保留顶层 actors/critic/args 供权重读取；P0-00F 前 checkpoint 不具备正式 resume 资格，只能 pretrained/历史评估 | P0-00F |
| YYYY-MM-DD | `<sha/branch>` | `<paths>` | `<observable behavior>` | `<tests/eval>` | `<yes/no/reason>` | `<task/run>` |

## 16. 决策日志

用于记录“为什么这样做”，防止数周后重复争论或按结果改规则。

| 日期 | 决策 | 依据 | 被否决方案 | 影响范围 |
|---|---|---|---|---|
| 2026-07-20 | 基础策略唯一训练入口冻结为 `train_no_rllib_mappo_robust_v2.py` | 当前计划使用 robust v2 训练基础策略；缩小修复面可避免旧脚本分叉继续污染实验 | 同时修复 RLlib、shared-policy PPO、minimal、robust v1 和 multi 旧入口 | P0-00、基础策略训练 |
| 2026-07-26 | 算法测试必须直接调用基础训练入口的生产 batch/loss/update helper，且所有可预检错误必须在首个 optimizer step 前失败 | 复制 PPO 公式的测试可能与实际训练路径同时漂移；Critic 先更新后才发现非法 Actor 参数会造成部分参数污染 | 在测试内重新实现简化 PPO；仅断言 loss 有限；出错后依赖人工丢弃模型 | `robust_v2`、P0-00E 回归矩阵、后续算法改动 |
| 2026-07-27 | Resume 只表示同一 Run 在完整 update 边界的严格续接；从旧/更早权重分叉必须使用 pretrained 和新 RunID | 半个 rollout 无法安全恢复；部分加载会混合新旧模型；训练窗口 best 会泄漏训练噪声；CARLA 服务器内部状态无法完整序列化 | 缺字段时跳过；latest 不存在便从零；保存半个 update；按训练成功率覆盖 best.pt | P0-00F、基础策略所有后续 Run、P0-00G smoke |
| 2026-07-20 | **SUPERSEDED（未实施）**：任务内全局成功/碰撞统一按 terminated 处理 | 原判断将联合 episode 边界直接等同于所有 agent 的任务边界 | agent-centric 的逐车辆回报边界 | 已由 2026-07-21 P0-00B 草案替代 |
| 2026-07-21 | **CONFIRMED/IMPLEMENTED**：终止/截断按 agent-specific 任务边界判定，而不是按全局 reset 一刀切 | 同伴红车碰撞后，未碰撞红车理论驾驶回报仍存在，按 `X` bootstrap；红车成功或碰撞均使蓝车对抗任务产生结局，按 `T` 切断 bootstrap | 全局事件一律 `T`；非直接事件一律 `X` | 环境、robust_v2、单蓝车、生命周期测试；未启用的 adversary 全局碰撞兼容分支保持旧语义 |
| 2026-07-22 | **CONFIRMED/IMPLEMENTED**：MAPPO 保持固定 Actor/Critic 特征维度，允许每个 agent 的有效 transition 数动态不同；失活槽位只可作为仍活跃队友的 centralized-state 占位，不再以失活 agent 为目标生成样本 | 网络输入结构必须固定；当前 `Dict[agent_id, List[Transition]]` 天然支持变长轨迹，符合 agent-specific 回报边界 | 动态改变网络输入维度；用全零伪 transition 补齐；只在 loss 端临时 mask | `robust_v2` collector、buffer、critic batch、后续 GAE/PPO |
| 2026-07-23 | **CONFIRMED/IMPLEMENTED**：GAE 逐 episode 计算原始值，优势在 per-agent rollout 拼接后统一标准化；旧导入入口暂留显式兼容 wrapper | 保留 episode 边界同时避免逐 episode 标准化抹掉不同结局的相对强弱；不在 P0-00D 越权改变对抗/PSRO 入口 | 保持逐 episode 单独标准化；一次混合所有 agent；隐式改变所有导入方 | `robust_v2` 与 P0-00D tests；其他入口后续显式迁移 |
| 2026-07-20 | 先完成 C0/C1/C2 和精确归因，再启动长训练 | 当前指标仍可能误判 steer 与碰撞对象，旧训练无法形成可信结论 | 直接延长 7 月 5 日旧 run | P0-P1 |
| 2026-07-20 | 通信放在无通信双蓝基线之后 | 否则无法把多车数量收益与消息通信收益分离 | 直接在现有双蓝脚本加入 attention | P3-P5 |
| 2026-07-20 | 先固定红池 PSRO，再做双边演进 | 降低非平稳性，先验证 pool 和 meta-solver 的科学有效性 | 一开始同时训练红蓝双方 | P4-P6 |
| YYYY-MM-DD | `<decision>` | `<evidence>` | `<alternatives>` | `<scope>` |

## 17. 每周维护清单

- [ ] 本周所有代码修改已提交到独立、可描述的 commit。
- [ ] `git status --short` 中没有来源不明的生成文件或大模型权重。
- [ ] 新 Run 已加入实验登记表，失败 Run 也已记录。
- [ ] 重要 checkpoint、JSONL、CSV、TensorBoard 和视频已备份。
- [ ] Run 元数据包含 Git SHA、命令、配置、seed 和 checkpoint hash。
- [ ] 当前完成项已有测试或结果证据链接。
- [ ] 下一周只有一个主阶段和不超过 3 个最高优先任务。
- [ ] 更新本文档“最后更新日期”和阶段状态。
- [ ] 检查论文方法描述是否仍与实际代码一致。

## 18. 推荐的近期两周执行顺序

### 第 1 周：消除结论污染源

- [x] 第 1 天：完成 P0-00B，修正 terminated/truncated 语义并改写生命周期测试（15 项通过）。
- [x] 第 2 天：完成 P0-00C，`robust_v2` 全链路接入 `active_mask`；一车提前终止 mock 与 18 项回归测试通过。
- [x] 第 3 天上半：完成 P0-00D，修正 GAE/标准化/PPO 健康保护，25 项测试通过。
- [x] 第 3 天下半：完成 P0-00E，生产更新 helper 与完整 MAPPO/PPO 算法回归矩阵共 35 项测试通过。
- [x] 第 4 天上半：完成 P0-00F，冻结 checkpoint/resume 协议，51 项无 CARLA 回归测试通过。
- [ ] 第 4 天下半：执行 P0-00G 三更新 CARLA smoke，并从 latest 恢复第 4 个 update。
- [ ] 第 5-6 天：实现精确碰撞对象、steer/control 解析及 P0-B/P0-C 测试。
- [ ] 第 7 天：复查全部测试、smoke 原始记录和代码 diff；任一门槛失败则不进入评估器开发。

### 第 2 周：完成评估闭环，再决定是否进入单蓝车

- [ ] 第 1-3 天：实现 C0/C1/C2 evaluator、互斥结局校验和逐 episode JSONL。
- [ ] 第 4 天：建立 config、registry、latest/periodic 和独立 best-checkpoint 选择流程。
- [ ] 第 5 天：运行一次小规模 C0/C1/C2，复查原始记录、重复性和 C0 基础能力。
- [ ] 第 6 天：仅当 P0-06 通过时运行单蓝车 3-update smoke，手工重放典型 episode。
- [ ] 第 7 天：决定是否允许启动 3 seed 单蓝车训练；未通过则回到对应 P0 任务，不以延长训练替代修复。

日期是建议节奏，不是完成标准；CARLA 实验遇到环境 bug 时优先修复可复现性，不用训练时长掩盖问题。

## 19. Vibe Coding 与代码库约束

- 一次任务只修改一个可验证行为，例如“修正碰撞归因”，不要把奖励、网络和 PSRO 同时重写。
- 给 Codex 的任务应包含：目标、允许修改范围、不可改变的语义、验收测试和期望产物。
- 每次要求 Codex 先读相关实现和 `git diff`，再修改；不要只根据口头描述生成整套新框架。
- 训练启动前先提交代码；运行中的实验目录不要与代码提交混在一起。
- 大型 checkpoint、视频和 TensorBoard event 不提交 Git；提交 manifest、配置、汇总 CSV/JSON 和小型图表。
- 每个研究结论单独建任务，示例：`实现 C012 评估器并用固定轨迹测试互斥结局`。
- 让自动循环遵循“读代码 -> 写计划 -> 小改动 -> 单元测试 -> smoke -> 登记结果 -> 决定下一步”，禁止跳过失败步骤。
- 自动任务不得自行改变主指标、验收门槛、训练预算或删除历史产物；这些属于需要人工确认的研究决策。
- 建议在 `mappo_training/` 增加局部 `AGENTS.md`，固化本文件中的研究不变量、测试命令和禁止事项。

### 推荐给 Codex 的单任务模板

```text
目标：<一个可验证的代码或实验目标>
背景：<关联的研究假设和当前问题>
允许修改：<文件/目录>
禁止修改：奖励总定义、观测维度、旧实验目录（按任务调整）
验收：<具体测试命令和数值/行为门槛>
产物：代码、测试、配置、实验登记表更新
流程：先审计现状和 git diff；实施最小改动；运行单元测试；需要 CARLA 时先 smoke；
失败时保留日志并说明阻塞，不得通过放宽指标或删除测试来通过。
```

## 20. 当前最重要的判断

本项目已经超过“从零搭框架”的阶段，环境、MAPPO、对抗训练、鲁棒训练和 PSRO 都已有代码。P0-00B/P0-00C/P0-00D/P0-00E/P0-00F 已闭合终止/bootstrap、active-mask、GAE/PPO、生产更新及 checkpoint/resume 协议；当前唯一下一项是 P0-00G 的真实 CARLA 三更新 smoke 与一次严格恢复。P0-00G 和后续 Phase 0 通过后，才允许基础策略正式长训练并进入 Phase 1；Phase 1 通过后，双蓝、PSRO 与通信才可能成为可辩护的研究贡献。
