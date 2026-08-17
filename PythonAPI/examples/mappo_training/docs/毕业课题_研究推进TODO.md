# 基于对抗学习的多车协作驾驶算法研究：研究推进 TODO

> 本文档是本课题的主任务清单、代码修改记录表和实验登记表。执行任务时以本文档的阶段门槛为准；理论设计细节参考 [对抗指标、奖励设计与阶段计划](./adversarial_metrics_reward_design_and_plan_20260705.md)、[MAPPO 多智能体样本处理与 active-mask 原理](./MAPPO多智能体样本处理与active-mask原理.md)、[MAPPO GAE、优势标准化与 PPO 更新原理](./MAPPO_GAE优势标准化与PPO更新原理.md)、[MAPPO checkpoint 与恢复训练协议](./MAPPO_checkpoint与恢复训练协议.md)、[MAPPO TensorBoard 指标协议](./MAPPO_TensorBoard指标协议.md)、[R1 二维有符号纵向动作协议论文正文](./R1_二维纵向动作协议_论文正文与实验设计.md)、[R2 奖励预算校准论文正文](./R2_奖励预算校准_论文正文与实验设计.md) 和 [R3 Critic 价值归一化与真实 CARLA 验证](./R3_Critic价值归一化_论文正文与真实CARLA验证.md)。

## 0. 项目总览

| 项目        | 当前值                                                                                                                                       |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 课题        | 基于对抗学习的多车协作驾驶算法研究                                                                                                                         |
| 研究点 1     | 使用多智能体强化学习训练对抗车辆，主动生成能够干扰基础驾驶策略的长尾场景                                                                                                      |
| 研究点 2     | 使用 PSRO（不是 PRSO）维护红蓝策略池并交替演进，研究策略泛化和循环克制                                                                                                  |
| 扩展研究点     | 在多辆对抗车之间引入显式通信，研究通信对协同攻击能力的增益                                                                                                             |
| 当前工程基线    | `a33a573 / TOWN05 V3 U5 CONTINUE_WITH_WARNING`：正式 fresh u5、strict、双 32-case C0 和唯一 judge 已闭环；C0 v3 入口漏接已修复并通过 211/211。旧 u5 与旧 HOLD 不改写 |
| 当前研究阶段    | P0-BASE 基础 MAPPO 策略训练与选择 -> Phase 0 对抗指标可信化 -> Phase 1 单对抗车有效性验证                                                                          |
| 当前训练入口    | 基础策略仅使用 `training/train_no_rllib_mappo_robust_v2.py`；单蓝车在 Phase 1 使用 `training/train_2red_1blue_ppo.py`                                   |
| 当前有效自动化测试 | 211/211 全量通过（2026-08-14）；Town05 v2 32/32 真实出生点和四方向 offroad 正反例 PASS；正式 u5 strict、双 C0 v3、重复性和 CARLA 清理均 PASS |
| 最近训练产物    | 正式 Run `P0_BASE_MAPPO_Town05V3_s42_20260814_125226`：5 updates/80 episodes，u5 checkpoint SHA `cb79242e...10a1a`；主 C0 为 0/32 success、2/32 collision、0 offroad、30 timeout；judge=`CONTINUE_WITH_WARNING`，尚不可多 seed或入池 |
| 本文档创建日期   | 2026-07-20                                                                                                                                |
| 最后更新日期    | 2026-08-14                                                                                                                                |

### 阶段状态

| 阶段 | 状态 | 结论 |
|---|---|---|
| 基础 CARLA 多车环境与 MAPPO | REAL SMOKE + SUCCESS VERIFIED | P0-00G 自动链路、用户人工视觉及固定场景 success 两层真实测试通过；不代表 MAPPO 性能/收敛已验证 |
| 基础 MAPPO 策略训练与选择 | TOWN05 V3 U5 CONTINUE_WITH_WARNING | 正式 u5 工程与证据链 PASS；策略仍 0 success、制动主导，尚未证明收敛。当前只允许同一 Run strict resume 到下一登记停点，多 seed/选模/入池仍阻塞 |
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
5. 所有实验使用唯一 `RunID`，禁止覆盖已有目录。推荐格式：`P{phase}_{method}_{scenario}_s{seed}_{YYYYMMDD_HHMMSS}`。
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
- **道路合法性**：车辆车体只能占用 OpenDRIVE `Driving` lane 的并集；经 Broken 虚线在相邻 `Driving` lane 之间并线仍合法，但驶入 Shoulder、Sidewalk、Parking、Biking、Median 等非 `Driving` 区域必须成为显式安全失败，且不能因接近目标或获得进度奖励而被记为成功。
- **精确归因**：墙体碰撞、红蓝碰撞和蓝蓝碰撞必须区分；无法归因时不得发放诱导碰撞奖励。
- **独立评估**：评估使用确定性动作、冻结权重、固定协议，且不复用训练 rollout 作为最终结果。
- **可复现性**：开发实验至少 3 个训练 seed；论文最终实验建议 5 个 seed，并报告均值、标准差和 95% 置信区间。
- **统计口径冻结**：最终比较开始后不得修改指标定义；如必须修改，旧结果全部标记为旧协议。
- **检查点选择**：最佳模型必须满足安全约束，再按独立验证集上的攻击效果选取，不能按训练期 `1 - red_success` 选取。
- **终局与截断**：按 agent-specific 回报边界判定；自身到点/碰撞及红车结局对应的蓝车对抗结果为 `terminated`，同伴红车碰撞对未碰撞红车属于外部 `truncated`，`max_steps` 对仍存活车辆同样为 `truncated`；只有合法 truncation 允许 value bootstrap。
- **失活样本隔离**：`active_mask=False` 的 agent 不得继续动作采样、写入 rollout、参与 GAE 或增加 transition 计数；全零 observation 不能被当成有效状态训练。
- **GAE 与优化目标隔离**：每个 agent/episode 独立计算原始 GAE；同一 agent 仅在完整 rollout 上标准化一次；Actor 使用标准化 advantage，Critic return 始终为 `raw_advantage + old_value`。

## 3. 下一步唯一主线

以下任务严格按顺序执行。2026-08-12 起实验背景由 Town02 切换为 Town05；旧 seed-42 u5、C0 和 GO/HOLD 结论只保留为 Town02 历史证据。Town05 改变地图几何和任务分布，必须 fresh 训练，禁止恢复旧 Run。完整场景坐标、奖励语义、合同与实测证据见 [Town05 正式实验迁移与 16 场景设计](./Town05_正式实验迁移与16场景设计.md)。

- [x] **P0-T05-00 完成实验背景迁移**：新增四方向 × 四家族 16 个正式训练场景和 Road 43/6 的 8 个留出场景；训练、C0、GO/HOLD、当前 Phase 1 入口与常用工具默认改为 Town05。Town02 目录、合同、checkpoint 和裁决未删除且保持只读历史身份。
- [x] **P0-T05-01 冻结车道/目标语义**：居中与偏离按每辆车当前 `Driving` lane 中心计算；D1–D3 逐车保持出生车道目标，D4 使用共享窄目标强制并道；Town05 Broken 虚线变道合法，实线/路缘/未知事件仍违规。
- [x] **P0-T05-02 完成迁移链路诊断**：离线训练池 16/16、留出池 8/8；真实 CARLA 出生位置/速度 16/16；旧诊断训练 1 update/4 episodes/960 transitions；Town05 C0 smoke 协议链路 PASS。该旧训练 smoke 后续发现 PBRS 实际 gamma 为 0.99，只证明链路可运行，不作为奖励协议一致性证据。
- [x] **P0-T05-03A 修复并冻结 reward gamma 数据流**：`robust_v2` 现在把合同/PPO 的 `args.gamma` 显式传给环境 `reward_gamma`；环境创建后 mismatch 立即关闭并 fail-fast；checkpoint runtime 写入 `environment_reward_gamma`；Town05 C0 还会拒绝缺失/错误实际 gamma 的 checkpoint；新增连接级回归，连同当时全量 183/183 通过，并以 Git `2d18f68` 提交。Town02 正式 u5 和旧 Town05 smoke 因实际训练 gamma 0.99 降级为历史诊断证据，禁止续训/选模。验收中的首次 `numpy` import 失败由在项目目录误用 `.\venv`、选中内层空环境引起，改用规范根环境 `..\..\..\venv` 后全部通过，属于命令路径错误而非代码/算法失败。
- [x] **P0-T05-03B 提交初始化修复并在精确 commit 上重跑 Town05 真实训练 smoke**：首次完整逐场预检为 14/16，场景 09 的 agent0 与场景 11 的 agent1 偶发停在 0.5 m 预偏移位置且速度为 0；隔离重复表明同一场景有时正常、有时丢失首次非零速度请求，故判定为 CARLA 异步 transform/target-velocity 状态提交瞬态，不是训练指令、Critic warning、梯度或策略移动缓慢。修复以 `a11cf70` 提交：全车先清零速度，再分同步帧提交预偏移 transform 和目标速度，按位置/速度容差核验，最多五次且失败关闭。提交后 training/C0/GO-HOLD 三合同 critical-source clean PASS；真实 16/16 初始状态通过。Run `T05_parallel_smoke_s42_20260812_161632` 完成 1 update、4 episodes、960 transitions，strict validator 核对 197 个必需指标和 582 个 TB scalar；两车 preupdate ratio 误差均 0、PPO 未跳过、reward 最大逐步残差 0、训练/环境 gamma 均 0.997，退出后 async/None、vehicle/sensor 0/0。随机一更新 4/4 timeout 只说明尚未训练，不作性能结论。checkpoint 记录正确 origin/last Git `a11cf70`；全仓 `git_dirty=true` 仅来自合同 critical paths 之外的用户既有 `AGENTS.md`、`Pipfile`、Obsidian 状态，三套正式预检均明确允许且未污染受控源码身份。
- [x] **P0-T05-04 运行 Town05 正式 seed-42 u5**：Run `P0_BASE_MAPPO_Town05_s42_20260812_164411` 从随机初始化完成 5 updates、80 episodes、65178 transitions，并自然停止，未启动 u6。strict validator、schema-v4 periodic checkpoint、训练/环境 gamma 0.997 和 CARLA 清理均 PASS。主/重复 C0 各自均为 0/16 success、12/16 collision、4/16 timeout 且完整性 PASS；两轮宏观结局一致，但场景 07/08/09/10/15 的 collision episode length 相差 1–2 tick，冻结离散重复性为 HOLD。唯一 Town05 GO/HOLD 因该证据硬失败裁决 `HOLD`；训练健康和 R4 行为塌缩评估 PASS，另有 8/16 lane invasion warning。完整记录见 [Town05 seed-42 u5 正式训练与 GO/HOLD](./P0-BASE-Town05_seed42_u5正式训练与GO-HOLD记录.md)。
- [x] **P0-T05-05 修复并实测非 Driving 区域终止及证据记录**：204/204 离线回归、Town05 v2 32/32 出生点、四方向合法相邻 Driving/强制越界/offroad+goal、fresh 1-update、strict metrics、schema-v5 checkpoint/SHA256 与退出清理全部 PASS。随机初始策略的 10/16 offroad 均在第 3 个连续 raw tick 正确终止并带首证据，不属于合法轨迹假阳性；旧 u5 仍不得 resume。
- [x] **P0-T05-06 前瞻冻结 C0 结果级重复性口径**：已发布 training/C0/GO-HOLD Town05 v3 三套新合同；旧 v2 双 C0、GO/HOLD 报告和 `HOLD` 原样只读。新版严格比较 scenario/结局/end reason、success/collision/offroad、T/X/done、压缩 active-mask 生命周期转换和首个 offroad 证据存在性；episode length/各 Agent active steps 允许最多 2 tick，单 Agent episode return 在 `1e-5` 以上至 `0.1` 记 `WARNING`，超过 `0.1` 为 `HOLD`。不少于 8 cases 且告警案例超过 50% 时按系统性漂移 `HOLD`。C0 validator 和 GO/HOLD judge 只把 `gate_pass=false` 当结构失败，容差内告警仍写入报告；全量 210/210 离线回归通过。新合同必须在本阶段提交后 fresh 运行，不能追认旧产物。
- [x] **P0-T05-07 完成 Town05 v3 正式 seed-42 u5 与机器裁决**：Run `P0_BASE_MAPPO_Town05V3_s42_20260814_125226` fresh 完成 5 updates/80 episodes；strict validator 核对 208 必需指标、682 TB scalar 和 schema-v5 checkpoint。首次 C0 暴露生产入口漏接 v3 常量，修复 `a33a573` 不改变算法/合同/checkpoint，专项 17/17、全量 211/211 和三合同 preflight PASS。主/重复 C0 各 32 cases，安全/结局无 hard mismatch；13/32 return 容差告警且比例 40.625%，重复性 gate PASS。主 C0 为 0 success、2 collision、0 offroad、30 timeout，8 lane invasion；唯一 judge=`CONTINUE_WITH_WARNING`，允许同 Run 续下一停点但不放行多 seed。完整记录见 [Town05 v3 seed-42 u5 记录](./P0-BASE-Town05V3_seed42_u5正式训练与GO-HOLD记录.md)。
- [x] **P0-T05-08 停止 u50 制动局部最优并封存诊断**：同一 Run 曾按冻结身份恢复，研究者于最后完整 u50 要求停止；800 个完整 episode 为 0 success、23 collision、48 offroad、729 timeout，另保留 1 条未完成 u51 中断记录。u41–u50 的 agent0 brake mode `0.9901`、速度 `0.0101 m/s`、完成度 `0.0138`，且其平均回报 `-5.7736` 高于仍运动的 agent1 `-8.3456`，证明停车是当前目标下的稳定相对最优。50 次 PPO 无 ratio/reward/nonfinite/skipped 故障；u50 checkpoint SHA256 `20b86366...d92adba`，仅作失败对照，禁止 resume/选模。CARLA 最终 async/None、actor 0/0。下一步转为 squashed-action entropy、task horizon/reward budget、mastery curriculum 新协议，详见 [u50 局部最优诊断](./P0-BASE-Town05V3_seed42_u50局部最优诊断.md)。
- [x] **2026-08-17 真实 CARLA 重复性/断点续训抽查**：同一 u40 checkpoint、同一 smoke case 和 seed 连续执行两次 C0，独立产物 validator 与 repeat comparator 均 `PASS`，hard fields 无差异、2-tick 容差字段无 mismatch、return 无 warning；随后诊断 Run `DIAG_RESUME_REPRO_s42_20260817_115800` 完成 `fresh u1 → u2 首 episode 后受控中断 → 从 latest 恢复并完整提交 u2`。strict validator 核对 updates `[1,2]`、每 update 4 episodes、总计 8 条记录和 PPO 指标 `PASS`；latest 为 schema-v5/u2，保存双 actor optimizer、critic optimizer、Python/NumPy/Torch RNG 与 ValueNorm `update_count=2`，CARLA 最终 async/None、车辆/传感器 0/0。裸 `--resume latest` 因回落到默认 16 场景池被 checkpoint 指纹保护拒绝；复用原 32 场景配置后恢复成功，因此正式续训必须使用冻结合同或完整原参数。此项只证明**结果级 C0 重复性和完整 update 边界续训链路**，不证明 fresh-to-fresh 逐位轨迹一致，也不改变 u50 失败结论或 P0-T05-09 的修复优先级。

#### P0-T05-09 / R6 下一版基础策略修复计划（待审核后实施）

> **状态：PLANNED / DESIGN REVIEW（2026-08-17）**。实现基线为分支 `codex/mappo-experiment-baseline`、提交 `2c4d088`。本阶段必须建立新协议、新合同、新 Git 提交和新 RunID；旧 u50 checkpoint 只作失败回归输入，禁止 resume、选模、进入基础策略池或追认为新版本结果。开始实现和无 CARLA 回归时不需要启动 CARLA；只有进入真实机制测试和 fresh smoke 时才提醒猹师傅启动 `127.0.0.1:2000`。

**目标与边界**

1. 修复 tanh 前 Normal 熵与 CARLA 实际执行动作熵不一致造成的饱和漏洞，阻止“pre-tanh 熵上升、post-tanh 动作却近乎恒定全制动”。
2. 联合校准 500 步任务时域、路线/成功信号和沿途风险/舒适成本，使合法平滑前进严格优于静止超时，同时保持碰撞/offroad 为强失败。
3. 把课程推进从 update 数驱动改成能力掌握门控；基础驾驶未掌握时不得自动加入更难交互场景。
4. 把行为 HOLD 从“双 agent 同时塌缩”扩展为“任意单 agent 持续塌缩”，避免团队均值掩盖 agent0 一类退化。
5. 本轮 P0 不同时引入共享 actor 或行为克隆，先保证当前独立 actor 架构下的目标函数正确，以便归因；参数共享和 BC 仅作为后续 P1 消融。

**P0-A：策略分布与动作空间熵**

- 保留当前带 tanh Jacobian 修正的 action log-prob；新增基于**当前策略 fresh sample** 的 squashed-entropy Monte Carlo 估计 `H[pi(a|s)] ≈ -E[log pi(tanh(z)|s)]`。PPO 熵奖励不得复用 rollout 中由旧策略采出的 `z` 估计当前熵，避免把交叉熵误当策略熵。
- `dist.entropy()` 只保留为 `pre_tanh_entropy` 诊断；正式优化和健康判断使用 `post_tanh_entropy`。增加每一动作维度的 `log_std`、确定性 actor mean、longitudinal/steer 饱和率和“高 pre-tanh 熵 + 高 post-tanh 饱和”告警。
- Actor 增加 `max_log_std`，并让熵系数支持冻结的线性退火。首轮离线候选为 `init_log_std=-0.5`、`min_log_std=-2.0`、`max_log_std=0.0`、`ent_coef=0.01→0.001`；这些只是待敏感性测试的候选，必须在 fresh pilot 前冻结，禁止看结果后追调。
- 专项测试覆盖：Jacobian log-prob 一致性、未更新 ratio 严格为 1、fresh-sample entropy 梯度有限、均值进入 ±饱和区时 post-tanh entropy 正确下降、log-std 上下界、CPU/GPU 可重复张量形状、checkpoint 保存/恢复熵调度状态。

**P0-B：task horizon、奖励预算与超时语义**

- 保留 reward v3 的 gamma-consistent 路线势能、防往返刷分、成功/碰撞/offroad 互斥和非正安全/舒适成本；不恢复正 speed/alive/centering 奖励，也不得通过简单降低道路安全惩罚强迫车辆移动。
- 联合比较 `gamma`、episode 长度、初级路线长度和 route/success task scale。当前 `gamma=0.997` 在 500 步末只剩 `gamma^500=0.2226`；候选方向包括评估 `gamma=0.999`、缩短 Stage-0 路线或分阶段子目标。任何候选都必须连同沿途负成本重新计算，不能只比较末端正奖励。
- 区分 `mission_timeout` 与纯 `rollout_cutoff`：固定任务截止失败应有明确失败语义和有界剩余距离项，例如 `r_timeout=-w_timeout*(1-final_completion)`，并切断任务终局后的 value bootstrap；纯采样截断才允许 bootstrap。具体权重由轨迹预算决定，不先拍脑袋写死。
- 增加 `stalled` 诊断：低速、路线完成度长期无增加、持续超过预注册窗口。首版默认只记录；只有简单、无需停车让行的 Stage-0 场景在真实校准后才可升级为终止。复杂交互必须加入前车阻塞/合理让行豁免，防止把安全停车误杀。
- 重点复核 `safety_lidar` 对平行同速或静止邻车的持续计费，优先评估用 ego motion、closing speed 或 TTC 门控；控制变化成本必须用 post-tanh 实际控制轨迹重新预算。
- 奖励预算的硬排序为：`成功 > 安全平滑前进 > 有进度但未完成 > 静止/无进度超时 > 碰撞或 offroad`；同进度下 `平滑 > 振荡`，且 `移动但越界 < 合法静止`。固定平滑前进、静止、振荡、部分进度超时、成功、碰撞和 offroad 七类轨迹均需通过基准与核心参数敏感性测试。

**P0-C：mastery curriculum 与单 agent HOLD**

- Stage 0 仅使用自由行驶、平行车道和静止起步等简单合法场景，不包含强制并线或明显冲突；只有连续两个冻结评估窗口同时满足双车最低完成度、最低速度、成功/timeout 和 collision/offroad 上界，才加入下一难度。
- 候选门槛为：每个 agent 平均速度至少 `0.2–0.5 m/s`、完成度至少 `0.10–0.25`、brake mode 不高于 `0.80`、collision/offroad 各不高于 `0.10`，并连续两个窗口成立。准确数值必须根据合法脚本轨迹和 Stage-0 任务长度在第一次 fresh pilot 前冻结。
- GO/HOLD 增加 per-agent collapse：任意 agent 在相邻两个正式停点持续满足高制动、低速度、低完成度、零成功和团队高 timeout，即 `BEHAVIORAL_COLLAPSE/HOLD`；不得要求两个 agent 同时塌缩。
- 增加 post-tanh 饱和、pre/post entropy 背离和课程越权升级检查；mastery 未通过时课程必须保持当前阶段，并留下原因和窗口证据。

**P1 消融（P0 验收后再决定）**

- 若 P0 修复后仍出现独立 actor 的角色分化，再比较当前独立 actor、共享 trunk + agent/角色 embedding、共享 actor + 独立输出 head；保持 centralized critic 和 decentralized execution。结构变化必须再次升级模型/合同身份。
- 若目标函数修复后随机初始化仍无法获得成功样本，再用 CARLA autopilot 或合法脚本轨迹做短行为克隆 warm start，然后由 MAPPO 微调；禁止在奖励排序仍失配时用 BC 掩盖问题。

**版本、兼容性与拟修改范围**

- 新增 policy-distribution protocol 身份；reward 改动发布为 reward v4，课程发布为 curriculum v2，同时升级 metrics、checkpoint schema、Town05 training/C0/GO-HOLD 合同和 critical-source 指纹。
- 旧 schema-v5 checkpoint 继续可按历史协议只读检查，但新训练入口必须拒绝 resume；C0/GO-HOLD 不得把旧 u50 产物追认为新协议结果。
- 预计涉及 `models/actor_critic.py`、`training/train_no_rllib_mappo_robust_v2.py`、`training/metrics_protocol.py`、`utils/training_metrics_recorder.py`、`environment/base_reward_v3.py` 或新的 reward-v4 模块、生命周期/课程/experiment/checkpoint 协议、C0/GO-HOLD evaluator、版本化合同、测试和文档。实施前先审计精确调用链，再确定最小文件集合。

**执行与验收顺序**

1. 不启动 CARLA：输出代码位置审计、数学定义、拟修改文件、协议版本和测试矩阵，猹师傅审核后再实施。
2. 实现 P0-A，先通过 squashed-entropy/log-std/ratio/checkpoint 专项和全量无 CARLA 回归。
3. 实现 P0-B，完成七类轨迹排序、折扣预算、timeout/bootstrap、守恒与敏感性测试；不得以删除安全断言换取通过。
4. 实现 P0-C，完成 mastery 状态机、恢复一致性、单 agent HOLD 和旧证据只读兼容测试；随后冻结新 training/C0/GO-HOLD 合同。
5. 需要真实验证时提醒猹师傅启动 CARLA；依次执行平滑前进、静止、振荡、碰撞、offroad、部分进度超时和成功轨迹，验证动作空间熵、奖励排序、终局、bootstrap 和清理。
6. 在精确 clean commit 上运行 fresh 1-update 工程 smoke，只验 metrics/checkpoint/resume/证据/清理，不作收敛结论。
7. 新 RunID、seed 42、fresh 运行 Stage-0 小规模 pilot；若任一 agent 再次持续制动或 mastery 未通过，立即 HOLD，不用延长至 u100 代替机制修复。
8. 只有 Stage-0 连续通过 mastery gate，才扩展完整课程、双 C0、GO/HOLD 和多 seed；只有多 seed 与 held-out 能力通过后，才能建立基础策略池并进入对抗训练。

**交付物**

- 新策略分布/奖励/课程/metrics/checkpoint/C0/GO-HOLD 协议与版本化合同。
- 单元、集成、恢复、旧证据只读兼容和真实 CARLA 机制报告。
- fresh smoke 与 Stage-0 pilot 的 RunID、Git/config/scenario 指纹、checkpoint SHA、strict validator、双 C0/GO-HOLD（到达对应正式停点时）和 CARLA cleanup 审计。
- 同步更新基础策略实验记录、机器 registry 与本 TODO；失败 Run 和未完成前缀同样保留，不删除、不覆盖、不冒充正式能力证据。

#### P0-T05-05 非 Driving 区域终止及证据记录（真实 CARLA 验收完成）

> **状态：LIVE VERIFIED / COMPLETED（2026-08-13）**。环境、reward、训练、metrics、checkpoint、C0 与 GO/HOLD 已按审核语义实现；旧 u5 checkpoint 和旧 C0/GO-HOLD 证据保持只读且不会被新协议接续。离线全量 204/204、真实 Town05 四方向正反例、32/32 出生点和 fresh 1-update 机制 smoke 均 PASS。该结论只证明道路约束和训练链路正确，不宣称随机初始化策略已经学会驾驶；正式训练仍需在受控源码 commit 和新重复性协议冻结后 fresh 启动。

**为什么必须修复**

- 当前 `_get_driving_waypoint()` 使用 `project_to_road=True`，即使车辆已经离开可行车面，也会得到“最近的 Driving waypoint”；因此当前 `lateral_distance` 只能描述离最近车道中心有多远，不能证明车辆仍在 Driving 区域。
- 当前 episode 只会因到达目标、碰撞/斑马线或 `max_steps` 结束；lane invasion 和连续横向偏离主要是奖励项，不会直接终止，存在用非道路区域绕行或获得进度的策略漏洞。
- Town05 seed-42 u1–u5 的训练记录中，每轮最大横向偏离约为 `4.33/4.15/3.48/3.18/2.94 m`，累计记录到 179 个非法 lane-invasion step；双 C0 各有 8/16 case 超过 `1.75 m` 且出现非法压线，但通常直到后续碰撞才结束。现有聚合指标缺少首个越界坐标、持续 tick 和车体占用比例，所以这些数据只足以触发安全 HOLD，不能精确重建每次 offroad。

**拟冻结的行为定义（审核项）**

1. 以车辆**车体占用**而非仅中心点为准：在 bounding box 内取中心点和 8 个内缩边界采样点；每个点使用 CARLA `Map.get_waypoint(location, project_to_road=False, lane_type=Driving)` 检查是否精确落在任一 `Driving` lane。采样点可以分属相邻 lane，不要求 lane id 相同，因此 Broken 虚线并线不会被误判。
2. `offroad_raw=True` 的拟议条件为：中心点找不到精确 `Driving` waypoint，或 8 个边界点中至少 3 个不在 `Driving` lane。该“3/8”阈值允许地图边界和单点采样误差，但当车辆已有实质部分进入路肩/人行道等区域时会触发。
3. 连续 `3 tick`（当前同步步长下为 `0.3 s`）满足 `offroad_raw` 才确认 offroad；中途恢复合法即清零计数。中心、3/8 和 3 tick 都必须在四方向真实道路正反例中校准；如需调整，只能在首次正式训练前形成新协议版本和证据，不能依据训练结果事后放宽。
4. 不设置 junction 自动豁免，也不把“投影后距离较近”当合法。当前 Town05 直道路由先做强验收；若未来路线穿过 junction，必须用单独合法轨迹证明 OpenDRIVE Driving 多边形无空洞后才能放行。
5. Solid 实线跨越若两侧仍都是 `Driving` lane，继续归类为 lane-rule violation，而不是伪装成 offroad；Curb 或车体进入任何非 `Driving` lane 则属于 offroad。是否把“实线跨 Driving lane”也升级为立即终止，是独立交通规则决策，不在本任务中偷偷扩大范围。

**生命周期与奖励方案**

- 同一 tick 的结局优先级冻结为 `collision/crosswalk > offroad > success > max_steps`，避免驶出道路碰到目标区后被记为成功，也避免真实碰撞被 offroad 标签覆盖。
- 基础双红车环境中，offroad 车辆自身 `terminated=True`；该合作任务以 `cooperative_offroad` 结束，未越界同伴为 `truncated=True`，可从同帧合法最终状态 bootstrap。终止车辆随后必须 `active_mask=False`，不得再采样动作、写 rollout、参与 GAE 或累计 transition。
- 对抗环境保持 agent-specific 语义：蓝车越界时仅该蓝车 `terminated` 并从后续交互中移除；不得让已越界蓝车留在场内充当免费障碍物。红车是否因合作 offroad 全局结束沿用上项语义。
- 新增独立且只发放一次的 `offroad` 终端奖励分量，拟定为 `-20`，与当前 collision 终端量级对齐；若同 tick 已归类 collision，则不重复叠加 offroad penalty。连续 lane-deviation shaping 保持不变，防止借本任务重调其他 reward。

**必须写入的证据**

- step/info：每车 `driving_center_valid`、`footprint_outside_count`、`footprint_outside_fraction`、`offroad_raw`、`offroad_consecutive_ticks`、`offroad_terminated`、`world_frame`、`episode_step`、位置/yaw、最近 Driving `road_id/lane_id/lane_width` 和 episode end reason。
- episode/JSON/TensorBoard：`offroad_count`、`offroad_rate`、每车 `first_offroad_step/frame/location`、`max_consecutive_offroad_ticks`、`max_footprint_outside_fraction`、最终 `terminated/truncated/active_mask`；C0 必须把 `cooperative_offroad` 显式列为安全失败，不能只藏在 `other`。
- checkpoint/contract/manifest：记录 offroad 协议版本、9 点采样定义、`3/8` 阈值、3-tick persistence 和 penalty；strict validator 对正式 Run 缺字段、协议漂移或 reward 不守恒必须 fail-fast。

**测试与验收门槛**

1. 纯单元/集成回归覆盖：全点 Driving；Broken 虚线跨相邻 Driving；Solid 跨 Driving 只记 lane violation；Shoulder/Sidewalk/Parking/Median；1–2 tick 瞬态恢复；第 3 tick 终止；collision/offroad/success 同 tick 优先级；合作同伴 truncation；蓝车局部终止；零 bootstrap/零额外 transition；offroad penalty 只记一次且 reward 守恒。
2. 真实 CARLA 对 Town05 四个正式方向分别执行“车道内直行、合法 Broken 并线、强制驶向道路外侧、offroad 后进入目标区”正反例。合法用例 offroad 必须为 0；强制越界用例必须 100% 在确认阈值后的 3 tick 内终止；offroad+goal 必须为 offroad failure，不能 success。
3. 对冻结 v1 16 个场景和正在建设的 v2 场景池做初始车体合法性核验，要求零假阳性；D4 合法并线轨迹必须可完整执行且 offroad=0。CARLA 退出后仍要求 async/None、vehicle/sensor 0/0。
4. 在新协议精确 commit 上 fresh 跑 1-update smoke，strict 指标、checkpoint、reward conservation、active-mask 和清理全部 PASS，再执行新的双 C0/GO-HOLD。必须区分“机制验收”和“策略质量”：随机/早期策略尝试 offroad 本身不等于代码失败，只要它被 100% 正确终止、处罚、隔离和记录，便可继续 fresh 训练；漏判、offroad 后继续产生 transition、证据缺失或合法并线假阳性才是机制硬 `HOLD`。最终 `best_base_valid.pt` 的冻结 C0 资格仍要求 `offroad_rate=0`。新机制通过前不得启动正式 fresh 训练；旧 u5 因 MDP 不兼容始终不得 resume。

**2026-08-13 离线实施记录**：

- 新增 `environment/offroad_protocol.py`，冻结 90% 内缩车体边界、中心+8 边界点、中心无精确 Driving 或边界至少 3/8 越界、连续 3 tick 确认和 `-20` 单次终端成本；环境只用 `project_to_road=False + LaneType.Driving` 做合法性判定，最近车道投影仅保留为诊断字段。
- `CarlaMultiAgentEnv` 已在物理 tick 后、奖励和终止判定前更新 offroad 状态；终局优先级为 collision/crosswalk > offroad > success > max_steps。基础双车中越界车 `terminated=True`、同伴 `truncated=True`；局部蓝车越界会终止并移除，不再成为场内障碍。
- reward v3 新增独立 `offroad` 分量和车体占用 diagnostics；训练 episode JSONL/TensorBoard 增加 offroad rate、首个越界证据、最大连续 tick 和最大车体越界比例。checkpoint 升至 schema v5，metrics、lifecycle、reward 与 offroad 协议均使用新身份，旧 R1–R5 checkpoint 不能 resume，但仍可按历史协议严格读取。
- 新增 Town05 v2 C0 与 GO/HOLD 合同：C0 逐 agent 保存 offroad 终止证据；32-case gate 中任何 offroad 都是显式结构性 HOLD，不能被 return、success 或 `other` 聚合掩盖。旧 Town05 v1/Town02 合同和报告未改写。
- 离线专项与全量 `python -B -m unittest discover -s tests -v` 共 204/204 PASS，覆盖 1–2 tick 恢复、第 3 tick 确认、去抖期间禁止 offroad 目标捷径、collision/offroad/success 优先级、合作同伴 truncation、蓝车局部终止、reward 单次惩罚、schema/checkpoint/C0/GO-HOLD 及历史协议兼容；真实 CARLA 验收与 fresh smoke 结果见下节。

**2026-08-13 真实 CARLA 验收记录**：

- `utils.smoke_training_scenarios_live` 对 `training_scenarios_town05_v2` 完成 32/32 reset：两车中心均为精确 `Driving`、边界越界 0/8、`offroad_raw=0`，位置/速度、跟驰间距与 LiDAR 均在注册容差内。
- `utils.validate_offroad_live` 最终报告 `diagnostics/offroad_live/20260813_town05_four_direction_v4.json` 为 PASS。北/南/东/西四方向的合法相邻 Driving 跨线连续 3 tick 均未误判；强制路外在前 2 tick 不成功、不处罚、不结束，第 3 tick 均为 `cooperative_offroad`，越界车 `terminated`、同伴 `truncated`、offroad 分量 `-20` 且首证据完整。测试夹具 v1 的 API 名称错误、v2 的候选点落到反向 Driving lane、v3 西向静态碰撞回调污染均保留为失败诊断；v4 以 actor 落地后的真实车体占用和异步传感器冲洗筛选负例，未修改或放宽 3/8、3-tick 生产判据。
- fresh 机制诊断 Run `diagnostics/offroad_live/DIAG_OFFROAD_TOWN05V2_U1_20260813_V2` 完成 1 update/16 episodes/7386 transitions，PPO 未跳过；结局为 10 `cooperative_offroad`、2 collision、4 max_steps。10 个 offroad episode 均有 `offroad_terminated` agent、首证据和至少 3 个连续 raw tick，最大越界比例范围 0.375–0.5；随机初始策略 offroad 率高说明门禁正在截断非法探索，不是训练效果或假阳性结论。
- `utils.validate_tensorboard_run` 核对 208 个必需 tag、682 个 TensorBoard scalar、16 条 `mappo_episode_record_offroad_v2` 后 PASS；`latest.pt` 为 schema v5/update 1，SHA256 正确，metrics 全部有限。训练和四方向脚本退出后均为 async/None、vehicle/sensor 0/0。第一次训练命令因外部 120 秒工具时限中断，残留同步模式已显式恢复，半截目录保持诊断身份；全新 V2 Run 自然完成，不复用半截数据。
- 完成真实验收后再次运行 `python -B -m unittest discover -s tests -v`，结果仍为 204/204 PASS。当前 dirty worktree 含用户既有改动及本阶段未提交源码，因此该 1-update 只具机制诊断资格；下一次正式 seed-42 Run 必须在明确的受控源码 commit 上 fresh 开始。

**兼容性结论**：该修复同时改变环境终止、reward 分解和证据 schema，虽然 Actor/Critic 张量形状可不变，旧 u5 checkpoint 所对应的 MDP 已不同，因此只能保留为历史诊断权重，不能正式 resume 或参与新协议选模。正在建设的 Town05 v2 本来就要求 fresh 训练，应在其正式 Run 前合入本修复，避免再产生一批道路语义不合格的策略。

#### P0-T05-06 HOLD 分级与结果级重复性（已提交，等待正式取证）

- **结构性 HOLD**：道路/碰撞/成功语义错误、结局或 T/X/active-mask 翻转、reward 不守恒、证据缺失、checkpoint 损坏、actor/同步模式泄漏。此时不得继续正式长训练、选模或论文比较；允许且应当运行有界诊断实验定位问题。
- **正式资格 HOLD**：当前产物未满足其预注册合同（例如旧 Town05 双 C0 要求离散签名逐项完全相同）。这表示该产物不能升级为正式证据，不等于算法“禁止再实验”。可以修复流程，也可以在查看新结果前发布新的协议版本并从 fresh Run 重新取证，但不得用新阈值追认旧产物。
- **非阻塞告警**：若重复运行的 scenario、结局、collision/offroad/success、安全证据和生命周期完全一致，只出现 1–2 tick episode length 或小幅连续状态漂移，则保留原值并告警；在新协议中可不作为统计型多 seed 训练的硬前置。若漂移改变 return/transition 很多、集中于特定场景或导致安全/结局翻转，则升级为结构性 HOLD。
- **冻结的新口径**：硬比较所有行为与安全离散字段；episode length 与各 Agent active decisions 注册 `±2 tick`；return `≤1e-5` 为数值等价、`(1e-5, 0.1]` 为告警、`>0.1` 为 HOLD；至少 8 cases 且告警比例 `>50%` 为系统性 HOLD。正式性能主要依靠独立训练 seeds、配对场景和区间估计，而非要求两次物理轨迹逐 tick/逐浮点完全相等。阈值已写入 `p0_base_c0_town05_v3.json`，后续不得按结果扩大。
- **工程优先级**：CARLA 官方的理想物理复现流程仍应尽量做到同步固定步长、加载前同步、每次重复 reload world、批量命令和唯一 tick client。当前训练已使用同步固定 `0.1 s`，但旧 C0 的 1–2 tick 碰撞回调差异不影响本次 offroad 语义验收。可以暂缓深挖逐 tick 根因；若以后出现语义翻转，再恢复 P0-T05-06 的底层诊断。

**实现与验收记录（2026-08-13）**

- 新训练合同：`configs/base_mappo/p0_base_town05_v3.json` / `P0_BASE_MAPPO_TOWN05_V3`；科学训练参数和 Town05 v2 的 32 场景池不变，仅更换正式运行身份并绑定 C0 v3。
- 新 C0 合同：`configs/evaluation/p0_base_c0_town05_v3.json` / `P0_BASE_C0_TOWN05_V3`；episode schema 升级为 `p0_base_c0_town05_offroad_episode_v3`。
- 新 GO/HOLD 合同：`configs/evaluation/p0_base_go_hold_town05_v3.json` / `P0_BASE_GO_HOLD_TOWN05_V3`；证据字段由 `require_repeatability_pass` 攵称为 `require_repeatability_gate_pass`，明确 `WARNING` 不等于失败。
- 回归：重复性/C0/合同/judge 专项 51/51；全量离线 210/210；系统性漂移合成样例为 `HOLD / gate_pass=false`；`git diff --check` 无空白错误。
- 当前 CARLA 已关闭；受控实现提交为 `69252d4`，提交后 training/C0/GO-HOLD v3 critical-source preflight 全部 PASS。开始 fresh seed-42 到 u5 前，需猹师傅重新启动 `127.0.0.1:2000`。

- [x] **P0-00A 冻结训练入口**：基础策略只维护 `train_no_rllib_mappo_robust_v2.py`，Phase 1 单蓝车只维护 `train_2red_1blue_ppo.py`；README、运行命令和文件索引已标记 RLlib 和其他旧入口停止使用。
- [x] **P0-00B 冻结并适配 terminated/truncated 语义**：采用 agent-centric 回报边界；同伴红车碰撞时未碰撞红车按 truncation bootstrap，红车成功/碰撞时蓝车按对抗任务结局 termination；环境、两个当前入口和 15 项测试已同步。
- [x] **P0-00C 修复 robust_v2 失活样本采集**：reset/step 的 `info['active_mask']` 已驱动 value、动作和 buffer；一车提前终止、另一车继续 3 步的 mock 中 transition 数为 `{0: 1, 1: 4}`，失活车零追加。
- [x] **P0-00D 修正 GAE 与优势标准化口径**：每个 episode 独立计算原始 GAE，同一 agent 拼接全部 episode 后在 rollout 上统一标准化；Critic 保持原始 return，PPO 更新前强制 `ratio≈1` 并检查数值有限。
- [x] **P0-00E 补 MAPPO/PPO 算法回归测试**：生产 batch 准备、clipped actor loss 与完整 optimizer update 已抽取并由 10 项新测试直接调用；多 agent 时间顺序、Critic int/batch agent-id、空 agent、异常输入零污染和全量回归均通过。
- [x] **P0-00F 修正基础策略 checkpoint/resume 协议**：训练只保存原子 `latest`、不可覆盖 periodic 和安全 `interrupted`；补齐 schema/protocol、RNG、课程、LR horizon、RunID/Git/config/source/manifest，并以 16 项测试验证严格恢复。
- [x] **P0-00F-TB 修正并模块化 TensorBoard 指标**：环境发布固定奖励分解与原始诊断；独立 recorder 按 active agent 聚合奖励、PPO/Critic/动作/模型/耗时，TB 与 JSONL 同源且可由 checkpoint 重放；动作诊断已在开训前升级至 v3，将互斥产生的踏板零值与真正上界饱和拆开。
- [x] **P0-00G-PREP 补齐 smoke 证据协议**：新增 `episodes.jsonl` 与安全边界裁剪，严格 validator 对账逐 episode/agent 生命周期；终局结果由 `episode_end_reason` 唯一分类；smoke 固定 `save_interval=1`，保存/日志/histogram 间隔纳入 resume 不可漂移配置；运行规范已冻结，全量 71 项无 CARLA 测试通过。
- [x] GO **P0-00G 运行 robust_v2 三更新 smoke**：前两 Run 暴露的课程顺序/清理残留和 precheck batch 几何假阳性已修复；第三个 Run 的 fresh 1–3、update 4 中断/恢复、16 episode、strict validator、checkpoint、清理与用户人工视觉确认全部通过。未提供视觉编号/录像，success 路径转入 C0 补验；详见 [测试记录](./P0-00G_真实CARLA_smoke测试记录.md)。
- [x] GO **P0-00G-SUCCESS 策略无关地补验真实 success 路径**：`teleport` 在 step 1/2、`scripted` 在 step 32/55 依次触发两车 success；每车 `success=100`、`success_event=1`、terminated 且不 truncated，最终 `all_agents_reached`，collision 0。两次清理均恢复异步/None、actor 0/0；专项 8 项、全量 93 项回归通过。详见 [规范与记录](./P0-00G-SUCCESS_真实success逻辑测试规范与记录.md)。
- [x] GO **P0-BASE-00 冻结可训练源码基线**：success smoke 脚本、8 项专项测试和文档已审阅并提交为 `345ea04`；`Pipfile` 与 `.obsidian/workspace.json` 未纳入。全量 93 项、Town02 16 场景离线 validator、两个入口 `--help`、历史四更新 strict validator 和 `git diff --check` 均通过。
- [x] GO **P0-BASE-01 冻结正式实验契约**：机器真值为版本控制 JSON；固定 Git/源码/场景指纹、seed 42 pilot、100-update LR horizon、课程、单全批 PPO、停点、证据链和 HOLD 规则；12 项专项与全量 105 项回归通过。
- [x] GO **P0-BASE-02 建立基础策略专用 C0 evaluator**：冻结无蓝车/无额外交通的 smoke/gate/validation 三套 case，确定性 Actor mean、训练/验证集内容 SHA 隔离、逐 episode JSONL、互斥结局、逐车奖励与安全诊断、checkpoint provenance、严格恢复/离线重算和双 Run 离散重复性比较均已实现；历史 evaluator 继续只作诊断。
- [x] **P0-BASE-04 预注册 GO/HOLD 数值门槛**：机器 JSON 已冻结证据硬失败、KL/clip/梯度/critic/探索/真实动作饱和健康线、8 个停点能力线及 P0-BASE-06 held-out 下界；离线 judge 绑定 Run origin Git 中的阈值/代码 blob，输出不可覆盖裁决，累计回报不参与 GO。
- [x] **P0-BASE-03R 基础策略制动塌缩修复（R4）**：GO/HOLD v2 已冻结双 Agent 确定性 brake、低速、低 progress、timeout、零成功的联合行为门槛，并要求相邻两个里程碑持续成立；Critic HOLD 要求四项退化信号连续五个 update。旧 Run 真实 C0 回放在 u5 仅 warning、u10 `BEHAVIORAL_COLLAPSE` HOLD；零成功但仍行进的早期策略不被误杀。v1 历史裁决不变，179 项回归通过。
- [x] **P0-BASE-03R 修复版最终入口 smoke（R5）**：真实 CARLA 10 项 R2/R3 机制诊断全部通过；生产入口 fresh u1、u2 受控中断/严格恢复、u3 完成，共 12 episodes。strict validator 对 `[1,2,3]`、197 个必需标签和 575 个 TB scalar 标签 PASS；schema-v4 checkpoint、ValueNorm/RNG/optimizer、前缀裁剪与 async/actor 清理均通过，未发生服务卡死。详见 [R5 记录](./R5_最终训练入口真实CARLA_smoke验证.md)。
- [x] **P0-BASE-03R 正式训练前全链路检查**：R1/R2/R3/R4 专项 38/38、全量 179/179 PASS；训练/C0/GO-HOLD 三合同 SHA 互绑、16 个训练场景、8 个 held-out 场景、R5 strict/checkpoint/真实机制报告和 CARLA 空闲状态均复核通过。正式命令 dry preflight 强制解析为 fresh、seed 42、u5、LR horizon 100、16 episodes/update、reward v3，并显式使用与历史正式 Run/R5 一致的 `--cpu`，避免和 CARLA 争用 GPU；修复了主动操作文档残留 C0/GO-HOLD v1/v3 的问题。详见 [开训前记录](./R1-R5_正式训练前全链路检查记录.md)。
- [x] **P0-BASE-03 修复版 seed 42 正式 pilot 到 update 5**：Run `P0_BASE_MAPPO_R3_Town02_s42_20260811_112458` 已按 R3 训练合同从随机初始化完成 5 updates/80 episodes，strict validator、schema-v4 u5 periodic checkpoint、主/重复 C0 v4 与唯一 GO/HOLD v2 全部闭环。裁决为 `CONTINUE_WITH_WARNING`，仅有 9/16 gate lane-invasion warning；未启动 u6。详见 [u5 正式记录](./P0-BASE-R3_seed42_u5正式训练与GO-HOLD记录.md)。
- [x] ARCHIVED **P0-BASE-03 Town02 同 Run恢复到 update 10**：因正式实验背景迁移至 Town05，该续训计划取消。Town02 u5 checkpoint、双 C0 与 `CONTINUE_WITH_WARNING` 裁决完整保留，但禁止跨地图恢复或用于 Town05 正式选模。
- [ ] BLOCKED **P0-BASE-05 扩展多 seed 正式训练**：仅在 P0-BASE-04 GO 后，以同一 commit/config 运行 seeds `42,123,2024`；不得依据中间结果修改预算、奖励或场景，失败 Run 同样登记。
- [ ] BLOCKED **P0-BASE-06 独立评估并选择 checkpoint**：用冻结 C0 协议评估全部 periodic checkpoint，先安全门槛、再任务成功与稳定性；生成 `best_base_valid.pt`，登记 SHA256、来源 Run/commit/config，禁止按训练 rollout 选模。
- [ ] BLOCKED **P0-01 精确记录碰撞对象**：基础策略试训不依赖对抗 impact 归因；取得可用基础 checkpoint 后进入本项，从 CARLA collision event 保存对方 actor id/type，并区分 `red-blue`、`blue-blue`、`static/world`，禁止继续用“最近红车”猜测 impact 对象。
- [ ] BLOCKED **P0-02 修复转向量提取**：等待 P0-01；`extract_kinematics()` 正确处理 NumPy action `[throttle, steer, brake]`，不能把正常步骤的 steer 恒记为 0。
- [ ] BLOCKED **P0-03 补指标和奖励单元测试**：等待 P0-02；创建 `tests/test_adversarial_metrics.py`、`tests/test_adversarial_reward.py`。
- [ ] BLOCKED **P0-04 实现 C0/C1/C2 统一评估器**：等待 P0-03；创建 `evaluation/evaluate_adversarial_c012.py`，输出逐 episode 原始记录和汇总统计。
- [ ] BLOCKED **P0-05 修正 best checkpoint 规则**：等待 P0-04；训练期保存 `latest/periodic`，独立验证后再生成 `best_valid.pt`。
- [ ] BLOCKED **P0-05M 迁移单蓝车训练证据与动作协议**：等待 P0-05；`train_2red_1blue_ppo.py` 已迁移 Town05 场景默认值，但仍按旧 3 维 Actor 加载，不能直接消费当前 Town05 基础策略的 2 维 signed-longitudinal checkpoint。必须先迁移 Actor/action mapping，再补原子 latest/periodic/interrupted、checksum/manifest、RNG/LR/RunID/协议版本、对抗 `episodes.jsonl`/`metrics.jsonl` 和严格 validator；禁止直接 `torch.save` 半个 update。
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

#### P0-00F-TB TensorBoard 指标正确性与模块化

> **实施状态：COMPLETED（2026-07-28）。** 本阶段完成无 CARLA 的协议、聚合和持久化验收；真实 sensor 触发率、奖励分布与训练健康值仍归入 P0-00G。

- [x] 新增独立 `training/metrics_protocol.py`，冻结 v1/v2 奖励分量和 raw diagnostics schema；active agent 任一字段缺失、多余、NaN/Inf 或 `reward != sum(components)` 时 fail-fast。
- [x] 新增 `utils/training_metrics_recorder.py`，由 `robust_v2` 统一调用；环境和训练循环不再各自拼接当前 Run 的 TensorBoard 标签。
- [x] 所有奖励分量按 agent 有效 transition、episode sum、pooled transition 和 team episode 四种口径聚合；失活 agent 不补 0、不扩大分母。
- [x] 将 `progress/alive` 标为 v2 有意禁用项；将 LiDAR/TTC/碰撞/成功/time bonus 标为稀疏或门控项；`lane_invasion` 因含连续偏离惩罚保持常规项。
- [x] 环境发布 progress、速度、横向/yaw 误差、LiDAR 最小值/触发 ray、TTC 候选/触发、sensor 事件等原始诊断，不改变原奖励数值公式。
- [x] 接入 advantage、Actor/Critic loss、entropy、KL、clip、ratio、梯度裁剪、value 分布、动作零值/饱和/模式/冲突率、参数/更新范数、LR、课程和耗时；参数 histogram 间隔可配置。
- [x] 每个完整 update 将同一 payload 写入 `tb/` 与 `metrics.jsonl`，并保存为 checkpoint 的 `last_metrics`；checkpoint 已提交但日志未提交时由 resume 精确重放。
- [x] 新增 `utils.validate_tensorboard_run`，离线检查协议、必需标签、有限值、奖励守恒、update 连续和 TensorBoard/JSONL 数值一致。
- [x] 新增 9 项专项测试并运行全量 60 项测试；无需 CARLA 的奖励路径、ragged agent、稀疏事件、PPO 指标、模型更新和恢复故障窗口均已覆盖。

**验收门槛**：奖励字段与总奖励可对账；agent 生命周期不污染聚合；合理的 0 可由 enabled/sparse/raw diagnostic 解释；训练更新是否发生可由 PPO 和模型曲线交叉验证；日志可恢复、可离线审计。

| 五项评估维度 | 合格标准 | 验收结果/证据 |
|---|---|---|
| 1. 奖励完整性与守恒 | 每个 active transition 字段精确、数值有限、逐步残差不超过 `1e-6`；episode 再次守恒 | PASS：缺失/未知/NaN/Inf/错误总和全部 fail-fast；v2 环境 mock 的总奖励与 15 个分量在 `1e-6` 内一致 |
| 2. agent 与生命周期隔离 | 只统计 transition owner；一车早退不补零，per-agent active count 与训练 buffer 相等 | PASS：构造 agent0 仅 1 步、agent1 2 步，计数与均值严格保持 `1/2`，稀疏碰撞只归 agent1 |
| 3. 零值可解释性与诊断 | 禁用、稀疏和连续项显式区分；LiDAR/TTC 等有独立原始触发证据 | PASS：schema 标记 `progress/alive enabled=0`；LiDAR mock 同时产生非零 safety reward、最小距离和触发 ray；`lane_invasion` 未误标稀疏 |
| 4. PPO/模型训练健康 | Actor、Critic、advantage、ratio/KL/clip/entropy、梯度、LR、样本和参数更新均按 agent 命名且有限 | PASS：非跳过 PPO payload 的必需命名空间受测；模型参数范数和 update norm 可交叉判断是否真实更新；ratio 极值按全 update 聚合 |
| 5. 持久化、恢复与集成 | TB/JSONL 同源、update 连续、checkpoint 崩溃窗口可重放；既有算法/恢复测试不退化 | PASS：EventAccumulator 与 JSONL 逐标签/逐 update 对账，离线 validator 返回 PASS；模拟 checkpoint 后崩溃可重放；全量 `60/60` 通过 |

**验收结论**：P0-00F-TB 五项单元门槛全部通过。当前 `robust_v2` 已具备模块化、可守恒、可解释和可恢复的指标数据流；旧入口未自动获得该协议，尤其 `train_2red_1blue_ppo.py` 必须在 Phase 1 前按自己的奖励 schema 显式迁移。P0-00G 仍需用真实 CARLA 证明 sensor、场景终局、PPO 数值和日志恢复符合预期。

#### P0-00G 真实 CARLA smoke

> **执行状态：PASS / GO（2026-08-02）。** 主 Run `P0_00G_robust_v2_smoke_s42_20260802_164211` 的 fresh、中断、恢复、strict validator、checkpoint、清理门槛与用户人工视觉确认通过；71 项无 CARLA 回归通过。未提供视觉编号/录像，success 路径转入 C0 补验。完整门槛见 [运行规范](./P0-00G_真实CARLA_smoke运行规范.md)，逐次证据见 [测试记录](./P0-00G_真实CARLA_smoke测试记录.md)。

- [x] 使用唯一 RunID 运行 `robust_v2`：1 seed、3 updates、每 update 4 个 episode，不恢复旧 checkpoint。
- [x] 检查 `episodes.jsonl` 保存每个 episode 的 `episode_end_reason`、per-agent active steps、transition 数、terminated/truncated 和 bootstrap 是否启用；未完成 update 的领先记录在 resume 时被裁剪。
- [x] 验证 `active_decision_steps == buffer_transition_count`，且失活 agent 后续增量为 0。
- [x] 验证 optimizer step 前 `preupdate_max_abs_ratio_error < 1e-6`，并检查后续 PPO epoch 的 KL/clip fraction/entropy/value loss/grad norm 均有限且数量级可解释。
- [x] 验证 `latest.pt` 每 update 原子更新、`periodic/ckpt_update_*.pt` 不覆盖，checksum/manifest 一致，且 payload 含 Actor、agent-conditioned Critic、optimizer、update、课程、LR、RNG、RunID/Git 和协议版本。
- [x] 检查 `metrics_schema.json`、`metrics.jsonl` 和 TensorBoard：各奖励分量字段齐全且有限；`reward_audit/max_abs_step_residual <= 1e-6`；progress/alive 的 0 与 disabled 标记一致；LiDAR/TTC 的 0 或非零能由 raw diagnostic 解释。
- [x] 使用 `--expected-updates 1 2 3 4 --expected-episodes-per-update 4 --require-ppo-updates --max-preupdate-ratio-error 1e-6` 运行严格 validator 并取得 PASS；结构化审计 per-agent reward、active transition、episode return、PPO/Critic、动作和模型 update norm。
- [x] 使用同一 `save_dir --resume latest --updates 4` 恢复 1 个 update，检查 next update、LR、课程、累计统计、JSONL 和 TensorBoard `tb/` 连续；中断测试确认 `interrupted.pt` 等于最近完整边界。
- [x] 用户确认人工视觉检查无出生重叠、瞬移、方向错误、异常冻结或重置问题；未提供 episode 编号/录像，success 路径未出现并转入 C0 补验。

#### P0-00G-SUCCESS 策略无关的真实 success 补验

> **执行状态：PASS / FROZEN BY P0-BASE-00 (`345ea04`)。** 该补验于 2026-08-02 运行，不加载 Actor/checkpoint，避免把策略未收敛混入 success 逻辑判断；纯逻辑与短距离集成层均在真实 CARLA 中无污染通过。

- [x] 实现 `teleport` 隔离模式：每次只把一辆 active 车移入 goal zone，仍通过真实 `env.step()` 观察奖励和终局。
- [x] 实现 `scripted` 近终点模式：首车距终点近边界 5 m、后车间隔 10 m，固定低油门直行并记录逐步位置。
- [x] 固定门槛：success 同步产生正 success reward/`success_event=1`/terminated、移出 active；最终严格为 `all_agents_reached`；碰撞/横道污染拒绝 PASS。
- [x] 每次输出 `effective_scenario.json`、`steps.jsonl`、`result.json`；专项 8 项和全量 93 项回归通过。
- [x] REAL `teleport` PASS：Run `P0_00G_SUCCESS_teleport_s42_20260802_222443`，两车在 step 1/2 success，最终 `all_agents_reached`，collision 0，清理通过。
- [x] REAL `scripted` PASS：Run `P0_00G_SUCCESS_scripted_s42_20260802_222523`，两车在 step 32/55 success，先到车退出后后车继续 active，collision 0，清理通过。

**P0-00 完成定义**：P0-00B 至 P0-00G 全部通过，测试输出和 smoke Run 已登记；此时仅证明基础 MAPPO 训练数据流可信，不代表基础策略性能已经达标。

### P0-BASE 基础 MAPPO 策略训练与选择

> **审计结论（2026-08-02）**：下一条研究主线应是训练并独立验证基础 MAPPO 策略。当前没有发现必须继续修改 MAPPO/环境算法逻辑后才能试训的阻塞；但“可以开始受控试训”不等于“可以直接批量长训”。先冻结源码、实验契约和基础策略专用 C0 evaluator，才能让训练结果可复现、可比较、可选模。

#### P0-BASE-00 源码与证据冻结

- [x] 审阅 success smoke 脚本、专项测试、README/命令/索引和 P0-00G 文档；真实 Run 的状态、步数、奖励、生命周期、清理信息及 SHA256 与记录一致。
- [x] 提交训练相关源码与文档；个人 `Pipfile` 镜像变更、Obsidian workspace、训练日志、checkpoint、视频和 TensorBoard event 均未提交。
- [x] 在提交候选树上重跑 93 项无 CARLA 回归（20.499 s）、Town02 v3 场景离线 validator、训练/success 两入口 `--help`、历史四更新 strict validator 和 `git diff --check`，全部 PASS。
- [x] 冻结源码 Git SHA：`345ea044e5d0071b3be8bf9896bb18bfffaa6d36`。后续若训练相关源码变化，必须使用新 RunID 并重新通过本门槛。

> **执行状态：COMPLETED / GO TO P0-BASE-01（2026-08-04）。** 本阶段只冻结已验收源码与证据，没有改变 Actor/Critic、环境、奖励、GAE/PPO 或 checkpoint schema；因此不新增 checkpoint 不兼容性。尚未完成的正式配置和 C0 evaluator 分别由 P0-BASE-01/02 负责。

#### P0-BASE-01 正式实验契约

- [x] 建立版本控制配置 [`p0_base_pilot_v1.json`](../configs/base_mappo/p0_base_pilot_v1.json)、机器 registry 与[独立人工实验台账](./P0-BASE_基础策略实验记录.md)；正式模式禁止只在命令行历史或 argparse 默认值保存参数。
- [x] 冻结 Town02 v3 场景池及运行时 fingerprint、weighted/metadata 采样课程、seed 42 pilot、100-update LR horizon、16 episodes/update、`max_steps=1000` 和完整 PPO/GAE/LR 参数；登记停点为 5/10/20/30/40/60/80/100。
- [x] 冻结 checkpoint/log/histogram/C0 milestone；每个 fresh Run 在连接 CARLA 前写 `resolved_config.json`、实验 manifest 和 registry，checkpoint 继续记录 RunID、Git、命令、配置/契约 SHA、运行时版本和输出目录。
- [x] 冻结停止规则：NaN/Inf、奖励/ratio 守恒、生命周期/active transition 对账、空/跳过 update、actor 泄漏、同步模式恢复、checkpoint/JSONL/validator/必需产物任一失败立即 HOLD；`0.65` 动作 warning 只对真正的 steer/踏板上界饱和率解释，互斥映射产生的踏板零值不得触发饱和警报。
- [x] 正式 CLI 只允许运行端点、设备、登记停点、RunID/目录和 strict resume；科学参数覆盖、未知 JSON 字段、非登记停点、关键源码 dirty、RunID/目录或契约 SHA 漂移均在 CARLA/optimizer 前 fail-fast。

> **执行状态：COMPLETED / GO TO P0-BASE-02（2026-08-04）。** 完整设计、参数依据、命令、风险边界与验收见 [P0-BASE-01 正式实验契约](./P0-BASE-01_基础策略正式实验契约.md)。本阶段用 MAPPO/PPO 一手资料和 P0 smoke 健康指标校准参数：保留 5 epochs、`clip=0.15`、`target_kl=0.02` 等已验收参数，将正式 `minibatch_size` 设为 32768，使最大两车 rollout 的 critic 每 epoch 仍为单全批；未运行 CARLA 或查看策略成功结果后调参。专项 12 项契约/证据测试已通过；全量回归和提交状态在本阶段代码修改记录中登记。P0-BASE-02 仍是启动 seed 42 正式 pilot 前的硬门槛。

#### P0-BASE-02 基础策略 C0 评估协议

- [x] 为两辆红车定义独立 C0 评估：冻结 periodic checkpoint 里程碑、确定性动作、gate/held-out 固定场景与固定 episode seeds，不加入蓝车或额外交通。
- [x] 输出逐 episode JSONL 和汇总 JSON/CSV；结局严格互斥为 success、collision、timeout/other，并保留每车 success、碰撞、active 长度、回报分量和安全诊断。
- [x] 记录评估协议版本、Git SHA、配置 hash、checkpoint SHA256 和命令；提供同 checkpoint/profile 双 Run 的离散签名比较。
- [x] 独立 validator 从 JSONL 重算 summary/CSV，检查 case 顺序、奖励/transition 守恒、SHA、CARLA actor 和 timing 清理；中断只允许相同身份的精确 case 前缀恢复。
- [x] 旧 `evaluation/evaluate_no_rllib_mappo.py` 仅作历史诊断；不得用于正式 checkpoint 选择。

> **执行状态：COMPLETED / GO TO P0-BASE-04（2026-08-04）。** 机器真值为 [`p0_base_c0_eval_v1.json`](../configs/evaluation/p0_base_c0_eval_v1.json)，设计、命令、证据 schema、失败边界和风险复盘见 [P0-BASE-02 基础策略 C0 评估协议](./P0-BASE-02_基础策略C0评估协议.md)。7 项专项与全量 112 项无 CARLA 回归、8 个 held-out 场景离线验证、两个入口 `--help` 和差异检查通过。当前没有正式基础 checkpoint，因此本阶段没有伪造真实 success 结果；首次真实性能数据只能在 P0-BASE-04 阈值预注册后由 P0-BASE-03 产生。

#### P0-BASE-04-PREP 动作空间与边界指标澄清

- [x] 核对 CARLA `VehicleControl` 与自带纵向 controller：保留油门/刹车互斥作为物理约束。
- [x] 确认 `winner_take_all_v1` 是固定 latent-to-env 映射，PPO rollout/log-prob/ratio 始终使用同一映射前 action；不构成概率口径错误，Actor/Critic/action dimension 均保持不变。
- [x] 指标协议升级为 `mappo_metrics_p0_00g_v3`：拆分 throttle/brake zero 与 upper saturation，增加 steer saturation、throttle/brake/coast mode 和 longitudinal conflict rate。
- [x] C0 逐回合 action 证据升级为 `p0_base_c0_episode_v2`，validator 强制模式率守恒且 `longitudinal_conflict_rate=0`。
- [x] 旧 throttle/brake combined boundary 只保留追溯用途；P0-BASE-04 不得用它触发 GO/HOLD。

> **执行状态：COMPLETED（由本次 Git 提交冻结）。** 完整原理、方案比较、当前映射限制、指标定义和兼容性见 [MAPPO 车辆动作空间与边界指标协议](./MAPPO_车辆动作空间与边界指标协议.md)。新增专项及全量 116 项无 CARLA 回归通过。本项不等于 P0-BASE-04 数值门槛已经完成；下一步仍需把 `*_saturation_rate` 等正确指标写入机器可读 GO/HOLD 契约。

#### P0-BASE-04 GO/HOLD 数值门槛预注册

- [x] 冻结机器契约 [`p0_base_go_hold_v1.json`](../configs/evaluation/p0_base_go_hold_v1.json)，绑定 P0-BASE-01 训练契约 SHA、P0-BASE-02 C0 契约 SHA、metrics v3、C0 episode v2、seed 42、gate/validation profile 和两次重复性审计。
- [x] 证据硬门槛逐 update 检查：16 episodes、奖励残差 `<=1e-6`、ratio 误差 `<1e-6`、缺失/非有限/跳过 PPO/纵向冲突/other 终局均为 0；训练/C0 validator、periodic checkpoint、manifest、清理和重复性必须全部 PASS。
- [x] 数值健康门槛冻结：KL `>0.02` warning、单次 `>0.10` 或连续 2 update `>0.05` HOLD；clip fraction `>0.50` warning、连续 2 update `>0.80` HOLD；梯度裁剪、critic EV、log_std 和三通道真实上界饱和只按登记连续窗口 warning。
- [x] 能力门槛冻结：update 5–40 以健康为主，60 至少 `1/16` success 且 collision `<=8`，80 候选 `12/16` success，100 只有 `>=14/16` success、0 collision、`<=2` timeout、各族 `>=3/4` success 且 0 lane invasion 才为 `PILOT_GO`。
- [x] 预先冻结 P0-BASE-06 held-out 下界：`18/24` success、0 collision、`<=6` timeout、0 other/lane invasion、各族 `>=4/6` 且每场景 `>=1/3`；当前不打开 held-out 结果。
- [x] 实现 `base_go_hold_protocol.py` 与 `judge_base_go_hold.py`：不提供阈值 CLI override；只读取已完成的训练/C0 产物；性能只计第一套 C0，第二套仅检查离散重复性；Run origin Git 必须含有同 blob 的阈值契约与判定源码；裁决目录不可覆盖。
- [x] 13 项专项覆盖未知字段/绑定 SHA 篡改、KL/clip/ratio/reward 临界值、连续 update、动作饱和 warning、缺失/非有限、各里程碑边界、重复性/other HOLD、lane invasion 和 validation 边界。

> **执行状态：COMPLETED / GO TO P0-BASE-03（2026-08-04）。** 详细口径、风险边界、标准命令和五维验收见 [P0-BASE-04 GO/HOLD 数值门槛预注册](./P0-BASE-04_GO-HOLD数值门槛预注册.md)。本阶段没有运行 CARLA、没有查看或制造正式 checkpoint 性能结果，也没有改动 MAPPO/PPO、网络、环境、奖励、课程、动作映射或 checkpoint schema。完成仅表示“判定规则已预注册”，不表示 seed 42 已 GO；现在只解除 P0-BASE-03 update 5 的启动阻塞。

#### P0-BASE-03-PREP CARLA 冷启动可靠性

- [x] 预实验启动时复现 Windows packaged CARLA 冷加载 Town02 超过原硬编码 10 秒 RPC timeout；失败发生在环境构造、首个 episode 和 PPO update 之前，不属于学习速度或算法失败。
- [x] 将连接/地图加载抽取为 `_connect_and_load_world()`，统一使用 `CARLA_RPC_TIMEOUT_SECONDS=60.0`；该值只控制真实墙钟等待，不改变 `fixed_delta_seconds=0.1`、`max_steps=1000`、场景、奖励、动作或训练预算。
- [x] 新增 mock 测试验证 Client -> timeout -> `load_world("Town02")` 调用顺序；全量 130 项无 CARLA 回归通过。
- [x] 按用户要求删除未完成 update 1、零 episode/transition/PPO update 的无效预实验日志与 registry 行；不将它保留为性能实验或纳入任何统计。
- [x] 后续 CARLA 服务器由用户手动启动；训练端只访问 `127.0.0.1:2000`，一旦服务卡死/超时立即停止并通知用户重启，不再由自动流程重启服务器。

> **执行状态：READY FOR FRESH UPDATE 5（2026-08-09）。** 修复需先形成新 Git commit，再用新 RunID fresh 开始；禁止恢复已删除的 update-0 初始化 checkpoint。

#### P0-BASE-03-EVIDENCE 正式证据 seed 完整性

- [x] 首次 Run `P0_BASE_MAPPO_Town02_s42_20260809_173321` 完成 5 updates、80 episodes 和 37014 个 agent-transition；strict validator PASS，update 5 训练内为 0 success、0.8125 collision、0.1875 timeout。
- [x] 两次 C0 gate 均为 0/16 success、0/16 collision、16/16 timeout，逐 case 离散重复性 PASS；确定性策略主要制动并近乎静止，因此不把无碰撞解释为基础能力达标。
- [x] GO/HOLD 在性能判定前因 `resolved_config.json` 缺少 seed 返回来源预检 HOLD；冻结契约、registry 和 checkpoint 均确认实际 seed 为 42，问题属于证据接口而非实际 seed 漂移。
- [x] 完成环境/指令/代码三分排查：CARLA client/server 0.9.16、5 updates、双 C0 和两次 cleanup audit 排除环境中断；正式命令只指定契约/endpoint/device/stop/RunID/目录且没有 `--seed` 覆盖，排除指令传参错误；checkpoint `run_metadata.seed=42` 但 checkpoint 专用最小 `training_config` 不含 seed，确认旧证据生成代码错误复用了该子集。
- [x] `resolved_training_config` 改为记录 `TRAINER_ARG_NAMES` 全量契约参数，而 checkpoint 恢复参数仍保持原有最小集合；fresh/resume 额外逐项核对，两侧分别增加 seed 存在性和 judge 绑定集成测试。
- [x] 科学参数、MAPPO/PPO、环境、奖励、动作、课程、场景和 GO/HOLD 数值门槛均未修改；专项 27 项通过，全量 132 项通过。
- [x] 修复已提交为 `8c6d582`，新 Run `P0_BASE_MAPPO_Town02_s42_20260809_204729` 的 resolved seed、训练、双 C0、repeatability 与 judge 来源预检均通过；旧 Run 保留 `EVIDENCE_HOLD` 且未篡改。
- [x] 同一正式 Run 已严格 resume 至 update 10；updates 1–10、160 episodes、双 C0、repeatability 和唯一 judge 全部形成可追溯证据，裁决为 `CONTINUE_WITH_WARNING`。
- [x] update 10 诊断确认：随机训练动作 brake mode 约 68%，确定性 C0 两车均为 `throttle_mode=0/brake_mode=1`；timeout 的直接原因是 Actor 制动策略。Critic explained variance 近 0、value prediction 方差远小于 return target 且连续 10 update 裁剪，属于学习质量 warning；ratio/KL/clip/finite 与生产算法回归均正常，当前无证据判定 GAE/PPO 梯度实现错误。
- [x] update 20 续训在 update 11 的 2/16 episode 后由研究者停止，尚未发生 PPO update/metrics commit/checkpoint；安全边界仍为 u10，registry/manifest 为 `INTERRUPTED`，两条临时 episode 不进入实验统计。
- [x] 冻结 judge 的 `CONTINUE_WITH_WARNING` 保留不变；另以研究者 `DIAGNOSTIC HOLD` 封存 v1 Run，避免把事后发现的 brake collapse 偷换成原机器门槛的 HOLD。

#### P0-BASE-03R 基础策略制动塌缩修复

> **执行状态：R0/R1/R2/R3/R4/R5 COMPLETED / PREFLIGHT PASS / FORMAL U5 PILOT NEXT（2026-08-11）。** 本阶段已完成逐项实现、真实 CARLA 机制复验、最终生产入口 smoke 和开训前全链路检查。action/reward/value/门槛任一变化都会使旧 checkpoint 失去正式 resume/pretrained 资格；正式 pilot 必须使用冻结契约、新 RunID 和随机初始化。

- [x] **R0 诊断封存与基线提交**：u5/u10 双 C0 与重复性已离线复验 PASS；judge、registry、实验记录和本修复清单由本次证据提交冻结。checkpoint/TensorBoard/过程日志以及用户 `Pipfile`、Obsidian 状态均排除；本次提交后的 HEAD 是 R1 唯一父基线。
- [x] **R1 动作协议 v2**：基础 Actor 从三维 `[throttle, steer, brake]` 改为二维 `[signed_longitudinal, steer]`；`d=0.05` 连续死区使正值映射 throttle、负值映射 brake、中心映射 coast。训练采样/buffer/PPO log-prob 均使用二维 pre-tanh `z`，CARLA 仍接收三维环境动作；checkpoint schema v3 保存完整 `action_spec`，metrics v4 与 C0 v2 同时审计策略/环境动作。旧 WTA/C0/metrics 只保留显式历史读取，新 resume/pretrained 严格拒绝；正式 R1 契约要求 fresh。专项与全量 140 项无 CARLA 回归通过，reward/Critic 未修改，论文推导、参考文献和测试计划见 [R1 文档](./R1_二维纵向动作协议_论文正文与实验设计.md)。
- [x] **R2 奖励预算校准**：基础 reward v3 以折扣一致的路线完成度势能为唯一正稠密任务信号，成功 `+10`、碰撞 `-20`，其余超速、实际纵向命令/转向变化、LiDAR、TTC 与车道成本全部非正且有界；移除可诱发停车局部最优的正 speed/alive/centering/yaw 与固定 step/time bonus。五类轨迹满足 `success > smooth-forward > stopped > collision` 和同进度 `smooth-forward > oscillatory`，核心权重独立 ±25% 后排序不变；油门/刹车率成本对称，传感器条目数不放大奖励。训练、环境、metrics、checkpoint 契约与 C0 v3 已同步，160 项无 CARLA 回归通过。论文依据、精确预算和局限见 [R2 文档](./R2_奖励预算校准_论文正文与实验设计.md)。
- [x] **R3 Critic 稳定化**：采用官方 MAPPO 路径的共享 ValueNorm 与归一化 Huber；rollout/bootstrap/GAE 使用反归一化 value，checkpoint v4 保存完整 running moments，metrics v5 同时记录原始与归一化尺度。真实 CARLA 536-transition A/B 中裁剪率从 100% 降至 0.33%，留出 MSE 从 63.8371 降至 57.1734，解释方差从 0.5309 升至 0.5897；R2 实车脚本同时通过守恒、静止、成功与碰撞排序。详见 [R3 文档](./R3_Critic价值归一化_论文正文与真实CARLA验证.md)。
- [x] **R4 协议与算法回归**：GO/HOLD v2 精确绑定 R3 训练合同、C0 v4、metrics v5 与 checkpoint v4；C0 judge 汇总确定性 brake/throttle/coast、平均速度、每回合 progress 和 timeout。行为 HOLD 最早 u10 且要求相邻两里程碑联合塌缩；Critic HOLD 要求低 EV、近常数预测、高相对 MAE、近连续裁剪四项连续五次。旧 u5/u10 C0 反事实回放、误判边界、ValueNorm、前序报告绑定和 v1 兼容均通过；全量 179 项无 CARLA 回归通过。真实 CARLA teleport/scripted success smoke 分别在 step 1/2 与 32/55 完成两车到达，终局、无碰撞和清理审计 PASS；旧 smoke 的 `+100/time_bonus` 不冒充 reward-v3 证据。详见 [R4 文档](./R4_GO-HOLD-v2早期塌缩门槛与回归验证.md)。
- [x] **R5 修复版 smoke**：真实 CARLA 策略无关机制诊断 10/10 PASS；生产入口完成 fresh u1、受控中断/恢复 u2、恢复 u3，12 条 episode、3 次 PPO 均落盘且 `train/update_skipped=0`。动作 conflict 和 pre-update ratio error 均为 0；ValueNorm count `1/2/3`，Critic clip activation `0.50/0.08/0.00`；全部 checkpoint/manifest/checksum、严格恢复、TB/JSONL 对账和最终 async/actor 清理通过。该诊断 Run 不执行正式 C0/judge、不用于 pretrained/选模；详见 [R5 记录](./R5_最终训练入口真实CARLA_smoke验证.md)。

| 修复后五项验收维度 | 必须满足 |
|---|---|
| 1. 动作语义 | signed longitudinal 映射互斥、中心可 coast、边界连续；训练采样、PPO log-prob 和 C0 deterministic action 使用同一版本 |
| 2. 奖励可学习性 | 五类轨迹排序通过；前进收益、平滑与安全惩罚量级有审计表；刹车变化不再漏罚，停车不再成为明显局部最优 |
| 3. Actor 行为健康 | smoke/C0 不出现两车确定性 100% brake；速度、progress、mode rate 与 timeout 同时报告，不能仅用 0 collision 判好 |
| 4. Critic/GAE 健康 | value normalization 状态完整可恢复；value prediction 不再近常数，梯度裁剪不连续 100%；GAE T/X、raw return 与反归一化边界测试不退化 |
| 5. 实验可追溯 | 新 action/reward/value/GO-HOLD 版本、Git SHA、契约 SHA、RunID、checkpoint SHA 与双 C0 证据齐全；旧 v1 Run/裁决不覆盖、不混训 |

#### P0-BASE-03 至 P0-BASE-06 运行与选模

- [x] **预注册判定**：C0 输出口径冻结后、启动 pilot 前已写明 GO/HOLD 数值门槛；累计回报不参与 GO，最终用自主 success、0 collision/lane invasion、终局/奖励/transition 守恒、场景覆盖和训练数值稳定联合判定。每个停点只应用同一契约，不得重写门槛。
- [x] **修复版首段 u1–u5**：新 RunID、R3 训练合同、随机初始化、双 C0 v4 和 GO/HOLD v2 已完成；裁决 `CONTINUE_WITH_WARNING`，安全边界 u5，未混用旧 checkpoint 或 R5 smoke 权重。
- [ ] **修复版分段 u6–u10**：仅在 C 盘空间稳定恢复后按同 Run strict resume；u10 后重复主/复跑 C0 和带 u5 前序报告的裁决。当前因主机空间阻塞，不得启动。
- [ ] **多 seed 扩展**：P0-BASE-04 GO 后，才使用完全相同 commit/config 扩展 seeds `42,123,2024`；先完成开发级 3 seed，论文最终是否扩为 5 seed 另行冻结。
- [ ] **独立选模**：对 periodic checkpoints 使用同一冻结 C0 验证集；安全门槛优先，其次比较 success、collision/timeout 和跨 seed 稳定性，输出 `best_base_valid.pt` 及 manifest。
- [ ] **产物登记**：每个成功或失败 Run 都更新实验登记表；登记 checkpoint SHA、配置、C0 结果、停止原因和下一动作。

**P0-BASE 进入训练门槛**：P0-BASE-00/01/02 全部完成，且 P0-BASE-04 的数值阈值已在看结果前冻结后，仅允许 P0-BASE-03 单 seed 分段试训；P0-BASE-04 再按同一阈值作 GO/HOLD 判定。

**P0-BASE 批量放行门槛**：P0-BASE-04 明确 GO 后才允许多 seed；P0-BASE-06 生成独立验证通过的基础 checkpoint 后，才把它作为 Phase 1 红车冻结策略。

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

# P0-00G 基础 MAPPO smoke；只允许在 P0-00B 至 P0-00F-TB 全部通过后运行
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --run-id P0_00G_robust_v2_smoke_s42_<timestamp> `
  --updates 3 `
  --lr-decay-horizon 3 `
  --rollout-episodes 4 `
  --min-transitions-per-update 64 `
  --max-steps 300 `
  --save-interval 1 `
  --tb-histogram-interval 1 `
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
| 2026-08-11 | `P0_BASE_MAPPO_R3_Town02_s42_20260811_112458` | P0-BASE-03R u5 | 验证 R1–R4 修复版在 seed 42 首里程碑的训练健康、确定性行为与裁决 | `ff5ddeb` | `train_no_rllib_mappo_robust_v2.py` / `p0_base_critic_v4.json` | fresh | 42 | C0 主/重复均 0 success、5 collision、11 timeout；逐病例重复性 PASS | N/A | 历史机器输出 `CONTINUE_WITH_WARNING`；2026-08-12 发现训练 PBRS gamma 0.99 与合同/PPO/C0 0.997 不一致，降级为历史诊断 | `logs/P0_BASE_MAPPO_R3_Town02_s42_20260811_112458`；`evaluations/P0_BASE_C0_gate_u000005_20260811_115047`；`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260811_120658` | 禁止 resume/选模；Town05 修复后 fresh 重做 |
| 2026-08-12 | `T05_MIGRATION_SMOKE_20260812_121336` | P0-T05 migration | 证明 Town05 16 场景、当前 MAPPO 和 C0 链路能在现有主机运行 | uncommitted diagnostic | `p0_base_town05_v1.json` / 诊断覆盖 1 update、4 episodes、120 steps | fresh | 42 | C0 smoke 链路 PASS；随机一更新模型在 1 个 case 碰撞，不作性能结论 | N/A | 16/16 live spawn、960 transitions、PPO/checkpoint/metrics 均可运行；但训练 PBRS gamma 为 0.99，协议偏差使产物只具链路诊断价值 | `diagnostics/town05_migration_smoke/T05_MIGRATION_SMOKE_20260812_121336`；`evaluations/P0_BASE_C0_smoke_u000001_20260812_121700` | 提交 gamma 修复，fresh 重跑真实 1-update smoke |
| 2026-08-12 | `T05_parallel_smoke_s42_20260812_161632` | P0-T05-03B | 验证提交后的 Town05 初始化、gamma、MAPPO 更新、指标/checkpoint 与清理链路 | `a11cf70` | `town05_parallel_pilot_smoke_v1.json` / 1 update、4 episodes、120 steps | fresh | 42 | N/A；随机策略训练内 0 success、0 collision、4 timeout，不作性能结论 | N/A | 16/16 live spawn；960 transitions；strict 197 required/582 TB scalar；gamma 0.997/0.997；PPO 执行；清理 0/0 | `diagnostics/town05_gamma_alignment_smoke_clean/T05_parallel_smoke_s42_20260812_161632` | P0-T05-04：正式 seed 42 fresh 到 u5，随后 strict/C0×2/GO-HOLD |
| 2026-08-12 | `P0_BASE_MAPPO_Town05_s42_20260812_164411` | P0-T05-04 | 验证 Town05 正式 seed-42 u5 的训练健康、C0 行为与复现性 | `6c28a91` | `p0_base_town05_v1.json` / 5 updates、16 episodes/update、500 max steps | fresh | 42 | 主/重复 C0 均 0 success、12 collision、4 timeout；5/16 collision length 相差 1–2 tick | N/A | 训练/单次 C0 证据 PASS；GO/HOLD=`HOLD`；8/16 lane invasion warning，且既有聚合证据不能排除车辆进入非 Driving 区域 | `logs/P0_BASE_MAPPO_Town05_s42_20260812_164411`；`evaluations/P0_BASE_C0_gate_u000005_20260812_170923`；`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260812_171550` | P0-T05-05：先修复 offroad 终止/证据；P0-T05-06 再诊断 tick 漂移；禁止 u6、多 seed和选模 |
| 2026-08-13 | `DIAG_OFFROAD_TOWN05V2_U1_20260813_V2` | P0-T05-05 | 验证新 offroad 语义在 Town05 v2 真实 rollout、PPO、metrics/checkpoint 和清理链路中闭环 | uncommitted/dirty-tree diagnostic | `p0_base_town05_v2` 科学参数 / 1 update、16 episodes、500 max steps | fresh | 42 | 10 offroad、2 collision、4 timeout；不作性能结论 | N/A | 7386 transitions、PPO 未跳过；所有 offroad 有 3-tick/首证据；strict 208 required/682 TB scalar；schema-v5/SHA/清理 PASS | `diagnostics/offroad_live/DIAG_OFFROAD_TOWN05V2_U1_20260813_V2`；`diagnostics/offroad_live/20260813_town05_four_direction_v4.json` | 冻结/提交受控源码；前瞻发布新重复性协议后 fresh 正式训练 |
| 2026-08-14 | `P0_BASE_MAPPO_Town05V3_s42_20260814_125226` | P0-T05-07 / u5 | 验证 Town05 v3 seed-42 首里程碑的训练健康、道路语义、确定性行为和结果级重复性 | origin `5dad4d4`；C0 接线修复 `a33a573` | `p0_base_town05_v3.json` / 5 updates、16 episodes/update、500 max steps | fresh | 42 | 主/重复各 32 cases；主为 0 success、2 collision、0 offroad、30 timeout；安全/结局一致，13/32 return 容差告警 | N/A | strict、双 C0、来源和重复性 gate PASS；judge=`CONTINUE_WITH_WARNING`；8 lane invasion，多 seed未放行 | `logs/P0_BASE_MAPPO_Town05V3_s42_20260814_125226`；`evaluations/P0_BASE_C0_gate_u000005_20260814_132002`；`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260814_134018` | 经猹师傅批准后同 Run strict resume 到 u10；绑定 u5 decision 重做双 C0/judge |
| 2026-08-17 | `DIAG_RESUME_REPRO_s42_20260817_115800` | P0-T05 checkpoint/repeatability spot-check | 在当前真实 CARLA 上抽查结果级 C0 重复性与完整 update 边界恢复 | unregistered dirty-tree diagnostic | u40 C0 smoke×2；训练 4 episodes/update、120 max steps，fresh u1→受控中断→resume u2 | fresh diagnostic；C0 使用正式 Run u40 只读 checkpoint | 42 / C0 case seed 61001 | C0 repeat `PASS`、无 mismatch/warning；训练 strict updates `[1,2]`、8 episode records `PASS` | N/A | checkpoint 配置漂移 fail-fast；完整原配置 resume 成功；optimizer/RNG/ValueNorm 恢复链路与 cleanup PASS；不作性能或逐位复现结论 | `diagnostics/resume_repro_live/DIAG_RESUME_REPRO_s42_20260817_115800`；`evaluations/P0_BASE_C0_smoke_u000040_20260817_115219`；`evaluations/P0_BASE_C0_smoke_u000040_20260817_115301` | 保持 u50 禁止 resume；按 P0-T05-09 新协议修复后再 fresh pilot |
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
| 2026-07-28 | `P0-00F-TB` | 环境、`metrics_protocol.py`、`training_metrics_recorder.py`、validator、`robust_v2`、checkpoint schema、9 项 tests、指标协议及项目文档 | 固定奖励/诊断 schema 与守恒审计；按 active agent 聚合；模块化记录奖励、PPO/Critic、动作、模型、场景和耗时；TB/JSONL/checkpoint 同源并可重放 | 9 项指标专项 + 全量 60 项通过；EventAccumulator/JSONL 对账和离线 validator PASS；AST、两个 `--help`、`git diff --check` | checkpoint schema 升为 v2 并强制 `last_metrics`；P0-00F 旧 schema 只能 pretrained/历史评估，不能正式 resume | P0-00F-TB |
| 2026-08-04 | `345ea04` | success smoke evaluator、8 项专项测试、P0-00G/SUCCESS 证据、README/RUN_COMMANDS/FILE_INDEX/研究 TODO | 冻结策略无关的真实 success 验证入口及证据；统一基础训练前置顺序为 P0-BASE；不修改训练/环境算法 | 全量 93 项；Town02 16 场景离线 validator；训练/success `--help`；历史四更新 strict validator；`git diff --check` 全部 PASS | 不改变任何 checkpoint 字段、网络或训练语义 | P0-00G-SUCCESS、P0-BASE-00 |
| 2026-08-04 | `P0-BASE-01` | `experiment_contract.py`、正式 JSON config、registry、12 项 tests、契约/实验记录文档与项目入口文档 | 增加 contract-owned 参数、关键源码预检、resolved config/manifest/registry 证据链；正式 minibatch 按最大 rollout 冻结为单全批 32768 | 12 项契约专项及全量回归；JSON 严格 schema/CLI 覆盖/身份漂移/原子台账 fail-fast；入口 help、场景 validator、diff check | 网络、reward/lifecycle、GAE/PPO 公式和 checkpoint schema 不变；正式 checkpoint `run_metadata` 追加契约身份；源码指纹已变化，正式 Run 必须 fresh | P0-BASE-01 |
| 2026-08-04 | `P0-BASE-02` | C0 JSON 契约、8 个 held-out 场景、协议/evaluator/validator、7 项 tests、评估协议与项目入口文档 | 正式评估只接收登记 periodic checkpoint；确定性逐 active-agent 执行；原子逐 episode 证据、精确恢复、summary 重算、防泄漏、清理和双 Run 重复性审计 | 7 项 C0 专项及全量 112 项；场景 validator、两个入口 help、AST/diff check；无正式 checkpoint，未运行或虚构性能结果 | 不改变网络、训练、环境、奖励或 checkpoint schema；仅收紧正式评估资格和证据口径 | P0-BASE-02 |
| 2026-08-04 | `P0-BASE-04-PREP` | `metrics_protocol.py`、训练 recorder、C0 evaluator/validator、动作专项测试、动作协议/TensorBoard/checkpoint/C0/TODO 文档 | 保留三维 WTA 互斥映射；将踏板 zero 与 upper saturation 分离，增加 longitudinal mode/conflict 守恒；combined pedal boundary 降为追溯指标 | 动作 helper、训练聚合、C0 严格 schema 和 WTA 映射专项测试；全量 116 项、两个入口 help、场景 validator、diff check 通过 | 指标协议升为 v3、C0 episode evidence 升为 v2；网络和环境 action 不变；旧指标 checkpoint 不作正式 strict resume | P0-BASE-04-PREP |
| 2026-08-04 | `P0-BASE-04` | GO/HOLD JSON 契约、纯判定协议、离线 judge、13 项边界测试、门槛协议与入口/TODO 文档 | 冻结证据/数值/里程碑/held-out 门槛；只用主 C0 计性能、repeat C0 查重复性；origin Git blob 防事后改阈值；报告不可覆盖 | 契约 preflight、judge `--help`、13 项专项及全量 129 项、训练/C0 contract preflight、场景 validator、diff check | 不改网络、环境、奖励、训练更新、动作映射或 checkpoint schema；只新增判定层；当前无性能 GO | P0-BASE-04 |
| 2026-08-09 | `P0-BASE-03-PREP` | `carla_multi_agent_env.py`、lifecycle tests、研究 TODO/实验记录 | 冷启动/地图加载 RPC timeout 从硬编码 10 秒提升为 60 秒并抽取可测试 helper；CARLA 改为用户手动维护，超时只通知不自动重启 | 连接专项 1 项及全量 130 项通过；Town02/训练参数/仿真时间语义不变 | 网络、reward/lifecycle、动作、checkpoint schema 和训练契约参数不变；源码指纹变化，正式 Run 必须 fresh | P0-BASE-03-PREP |
| 2026-08-09 | `P0-BASE-03-EVIDENCE` | `experiment_contract.py`、experiment/GO-HOLD tests、registry、研究 TODO/实验记录 | 正式 resolved evidence 记录全部 contract trainer args（含 seed），并在 resume 逐项校验；保留 checkpoint 最小恢复配置，不篡改失败 Run | 27 项契约/裁决专项及全量 132 项通过；首次 Run strict/C0/repeat PASS，但 judge seed 来源预检 HOLD | 不改训练算法、模型、奖励、动作、场景、科学参数或阈值；旧 Run 只作失败诊断，新 commit 后必须 fresh | P0-BASE-03 |
| 2026-08-10 | `P0-BASE-03R-R1` | `action_protocol.py`、基础 `robust_v2`、checkpoint/metrics/C0 v2、R1 训练/C0 契约、动作/算法/恢复/指标/C0 tests、论文与协议文档 | 基础策略改为二维 signed-longitudinal/steer；连续死区映射三维 CARLA command；二维 pre-tanh PPO 概率、策略/执行动作双层审计和 v1 只读兼容 | 动作协议专项、生产 PPO、active-mask、checkpoint、metrics、C0 v1/v2 与全量 140 项无 CARLA测试通过；compileall、契约加载和 diff check 纳入提交前复核 | Actor 输出层与 optimizer 形状不兼容旧权重；checkpoint/metrics/C0 协议升级。reward、Critic、环境动力学和 CARLA 均未改；R2 前禁止启动正式 pilot | P0-BASE-03R-R1 |
| 2026-08-10 | `P0-BASE-03R-R2` | `base_reward_v3.py`、基础环境/训练/metrics、reward-v3 训练与 C0 v3 契约、C0 evaluator/protocol、reward/environment/metrics/contract/C0 tests、R2 论文与项目协议文档 | 以 gamma 一致 route-completion potential 替代失衡稠密奖励；终局层级与所有辅助成本有界，实际 throttle/brake 平滑对称；基础 reward v3 与历史/对抗 reward v2 显式隔离 | 五类轨迹排序、望远镜、防投机、边界、±25% 敏感性及环境/metrics/契约/C0 集成；全量 160 项无 CARLA 测试通过 | 模型张量形状不变但 reward/config/critical-source 身份改变；R1/R0 checkpoint 只能作历史读取，R2 正式契约仅 fresh；R3–R5 前仍禁止 pilot | P0-BASE-03R-R2 |
| 2026-08-10 | `P0-BASE-03R-R3` | `value_normalization.py`、基础训练/checkpoint/metrics、R3 训练与 C0 v4 契约、真实诊断脚本/报告、value/checkpoint/metrics/C0 tests、R3 论文与项目协议文档 | Critic 输出归一化 value，rollout/GAE 反归一化；完整 return batch每 update 更新一次 ValueNorm；归一化 Huber 抑制稀疏大误差；normalizer 为强制 checkpoint 状态 | 真实 CARLA 536-transition A/B 的裁剪率 100%→0.33%、留出 MSE 63.8371→57.1734、EV 0.5309→0.5897；R2 实车轨迹 10 项机制验收全过 | 不修改 Actor、reward、场景、LR/PPO/GAE/grad clip；真实脚本不是性能 pilot；旧 schema v2/v3 仅历史读取，R4/R5 前仍禁止 pilot | P0-BASE-03R-R3 |
| 2026-08-11 | `P0-BASE-03R-R4` | GO/HOLD v2 JSON、双版本 protocol/judge、C0 行为摘要、前序报告链、12 项 v2 tests、R4 说明与项目台账 | 行为 HOLD 使用双 Agent brake/低速/低 progress/timeout/零成功联合条件并要求相邻两里程碑持续；Critic HOLD 使用四信号连续五 update；报告输出机器原因码 | R4 专项 12 项、v1+v2 GO/HOLD 26 项、全量 179 项通过；旧真实 u5/u10 C0 回放分别为 warning 与 `BEHAVIORAL_COLLAPSE` HOLD；移动但零成功样本不 HOLD；真实 teleport/scripted success 与清理审计 PASS | 不改 Actor/Critic 网络、reward、PPO/GAE、场景或训练参数；旧 v1 裁决不改写；success smoke 旧奖励只验生命周期，不作为 reward-v3 数值证据；R4 不是性能 pilot | P0-BASE-03R-R4 |
| 2026-08-11 | `P0-BASE-03R-R5` | 最终生产入口真实 CARLA 3-update smoke、R2/R3 机制复验、checkpoint/恢复/TB/JSONL 审计与 R5 记录 | 不改算法与科学参数；验证 action-v2、reward-v3、ValueNorm-Huber、metrics-v5、schema-v4 checkpoint 和 R4 所需字段在唯一入口端到端连通 | 10 项机制诊断 PASS；fresh u1、u2 受控中断/严格恢复、u3 完成；12 episodes、197 必需标签、575 TB scalar 标签、全部 checkpoint SHA/manifest/ValueNorm/RNG/optimizer 与清理审计 PASS；全量 179 项回归通过 | 诊断 LR horizon/episode 规模仅用于 smoke；不执行正式 C0/judge，不复用权重，不声称性能或收敛；未出现 CARLA 卡死，无需重启 | P0-BASE-03R-R5 |
| 2026-08-11 | `P0-BASE-03R-PREFLIGHT` | R1–R5 端到端审计、三合同/场景/运行时预检、正式 u5 命令 dry preflight、C0/GO-HOLD 操作文档与独立检查记录 | 不改算法、配置或阈值；统一当前 C0 v4/GO-HOLD v2 操作入口，明确 u5 后双 C0 与唯一 judge 停止顺序 | R1–R4 专项 38 项、全量 179 项、训练/C0/GO-HOLD preflight、16 场景 validator、R5 strict validator、5 checkpoint 与 CARLA 0.9.16 空闲审计均 PASS | 只放行 seed 42 fresh 到 u5；尚无正式性能结果，不放行 u10、多 seed、选模或对抗训练 | P0-BASE-03R-PREFLIGHT |
| 2026-08-12 | `2d18f68` / `a11cf70` | Town05 迁移/gamma 数据流、初始状态同步核验、正式合同/C0/GO-HOLD、场景/工具/tests/docs | 训练环境 reward gamma 与 PPO/GAE 同源且 fail-fast；checkpoint/C0 审计实际 gamma；CARLA 初始 transform/速度整组同步提交、有限核验重试并在失配时拒绝 rollout | 全量 184/184；训练池 16/16、留出池 8/8 离线；三合同 clean preflight；真实 16/16；clean-commit 1-update/960-transition strict/checkpoint/cleanup PASS | Town02 与旧 gamma=0.99 smoke 仅历史诊断；网络张量未因初始化修复改变；Town05 正式训练只允许 `a11cf70` 后 fresh，不得复用诊断 checkpoint | P0-T05-03A/B、`T05_parallel_smoke_s42_20260812_161632` |
| 2026-08-13 | `69252d4` | offroad protocol、环境/reward、训练/metrics/checkpoint、C0/GO-HOLD v2、Town05 v2 合同/场景、live validator、tests/docs | 车辆连续 3 tick 实质占用非 Driving 区域时成为显式安全失败；禁止 offroad 目标捷径；记录车体占用和首证据；合法相邻 Driving 并线不误判 | 204/204；32/32 live spawn；四方向 v4 PASS；fresh u1 strict/checkpoint/SHA/PPO/cleanup PASS | 终止/reward/schema/协议身份均变化，旧 u5 仅历史读取且禁止 resume；正式新 Run 必须在本修改提交后 fresh | P0-T05-05、`DIAG_OFFROAD_TOWN05V2_U1_20260813_V2` |
| 2026-08-13 | `69252d4` | Town05 training/C0/GO-HOLD v3、结果级重复性比较器、validator/judge、边界 tests/docs | 安全/结局/生命周期零容忍；episode length/active steps ±2 tick 和 return ≤0.1 分级告警；至少 8 cases 且告警 >50% 升级系统性 HOLD | 专项 51/51、全量 210/210、合成系统漂移 HOLD、diff check PASS；提交后 training/C0/GO-HOLD clean preflight PASS | 旧 v2 HOLD 不改写；新 v3 必须 commit 后 fresh；本阶段只改判定协议身份，不改训练超参数或网络 | P0-T05-06 |
| YYYY-MM-DD | `<sha/branch>` | `<paths>` | `<observable behavior>` | `<tests/eval>` | `<yes/no/reason>` | `<task/run>` |

## 16. 决策日志

用于记录“为什么这样做”，防止数周后重复争论或按结果改规则。

| 日期 | 决策 | 依据 | 被否决方案 | 影响范围 |
|---|---|---|---|---|
| 2026-07-20 | 基础策略唯一训练入口冻结为 `train_no_rllib_mappo_robust_v2.py` | 当前计划使用 robust v2 训练基础策略；缩小修复面可避免旧脚本分叉继续污染实验 | 同时修复 RLlib、shared-policy PPO、minimal、robust v1 和 multi 旧入口 | P0-00、基础策略训练 |
| 2026-07-26 | 算法测试必须直接调用基础训练入口的生产 batch/loss/update helper，且所有可预检错误必须在首个 optimizer step 前失败 | 复制 PPO 公式的测试可能与实际训练路径同时漂移；Critic 先更新后才发现非法 Actor 参数会造成部分参数污染 | 在测试内重新实现简化 PPO；仅断言 loss 有限；出错后依赖人工丢弃模型 | `robust_v2`、P0-00E 回归矩阵、后续算法改动 |
| 2026-07-27 | Resume 只表示同一 Run 在完整 update 边界的严格续接；从旧/更早权重分叉必须使用 pretrained 和新 RunID | 半个 rollout 无法安全恢复；部分加载会混合新旧模型；训练窗口 best 会泄漏训练噪声；CARLA 服务器内部状态无法完整序列化 | 缺字段时跳过；latest 不存在便从零；保存半个 update；按训练成功率覆盖 best.pt | P0-00F、基础策略所有后续 Run、P0-00G smoke |
| 2026-07-28 | 当前训练指标采用固定 schema、active-agent ownership 和同源 TB/JSONL/checkpoint payload；缺失值不以 0 代替 | 历史全零曲线无法区分禁用项、稀疏未触发、传感器故障和字段漏传；混合 agent 均值还会稀释局部事件 | 各训练脚本直接 `add_scalar`；`.get(name, 0)`；只保留 pooled mean；checkpoint 与日志分别提交 | `robust_v2`、P0-00F-TB/P0-00G；其他当前入口迁移时复用协议模块 |
| 2026-08-02 | P0-00G 增加逐 episode 审计、严格 validator，并冻结 `save_interval/log_interval/tb_histogram_interval` | update 均值无法追溯单个 agent 的 T/X/bootstrap；默认 save interval=5 在四 update smoke 中不会生成 periodic；resume 默认值会改变同一 Run 的证据节奏 | 仅保留控制台文本；跑完后人工补表；无 periodic 也判 PASS；用“首次 epoch ratio≈1”作为模糊标准 | P0-00G、指标协议、checkpoint/resume |
| 2026-07-20 | **SUPERSEDED（未实施）**：任务内全局成功/碰撞统一按 terminated 处理 | 原判断将联合 episode 边界直接等同于所有 agent 的任务边界 | agent-centric 的逐车辆回报边界 | 已由 2026-07-21 P0-00B 草案替代 |
| 2026-07-21 | **CONFIRMED/IMPLEMENTED**：终止/截断按 agent-specific 任务边界判定，而不是按全局 reset 一刀切 | 同伴红车碰撞后，未碰撞红车理论驾驶回报仍存在，按 `X` bootstrap；红车成功或碰撞均使蓝车对抗任务产生结局，按 `T` 切断 bootstrap | 全局事件一律 `T`；非直接事件一律 `X` | 环境、robust_v2、单蓝车、生命周期测试；未启用的 adversary 全局碰撞兼容分支保持旧语义 |
| 2026-08-13 | **CONFIRMED/IMPLEMENTED FOR FRESH V3**：新正式 Run 的重复性门禁采用“安全与结局严格、tick/连续量预注册容差”，逐 tick 根因修复降为非阻塞诊断 | CARLA 官方理想确定性流程要求加载前同步、每次 reload 和批量命令；深度 RL 方法比较通常依赖多 seed 与区间估计。旧 C0 只出现结局相同的 1–2 tick 漂移，未构成本次 offroad 语义翻转。新协议另设 >50% 告警比例的系统性 HOLD，避免把普遍异常当噪声 | 事后直接放宽旧 HOLD；完全删除重复性检查；坚持逐浮点相同并无限期阻塞统计实验 | P0-T05-06、Town05 v3 C0/GO-HOLD；旧报告和裁决保持只读 |
| 2026-07-22 | **CONFIRMED/IMPLEMENTED**：MAPPO 保持固定 Actor/Critic 特征维度，允许每个 agent 的有效 transition 数动态不同；失活槽位只可作为仍活跃队友的 centralized-state 占位，不再以失活 agent 为目标生成样本 | 网络输入结构必须固定；当前 `Dict[agent_id, List[Transition]]` 天然支持变长轨迹，符合 agent-specific 回报边界 | 动态改变网络输入维度；用全零伪 transition 补齐；只在 loss 端临时 mask | `robust_v2` collector、buffer、critic batch、后续 GAE/PPO |
| 2026-07-23 | **CONFIRMED/IMPLEMENTED**：GAE 逐 episode 计算原始值，优势在 per-agent rollout 拼接后统一标准化；旧导入入口暂留显式兼容 wrapper | 保留 episode 边界同时避免逐 episode 标准化抹掉不同结局的相对强弱；不在 P0-00D 越权改变对抗/PSRO 入口 | 保持逐 episode 单独标准化；一次混合所有 agent；隐式改变所有导入方 | `robust_v2` 与 P0-00D tests；其他入口后续显式迁移 |
| 2026-07-20 | **SUPERSEDED FOR BASE TRAINING**：先完成 C0/C1/C2 和精确归因，再启动任何长训练 | 该顺序适用于对抗策略，但若把 P0-01 至 P0-06 同样前置于基础策略，会因 C0/Phase 1 都需要合格基础 checkpoint 而形成循环依赖 | 直接延长 7 月 5 日旧 run | 对抗 P0-P1 仍有效；基础训练改由 P0-BASE 门槛控制 |
| 2026-08-02 | 基础策略训练前置于对抗碰撞归因，但必须先冻结源码、正式配置和基础 C0 evaluator，再按单 seed 分段试训 -> GO/HOLD -> 多 seed -> 独立选模推进 | P0-00G-SUCCESS、93 项回归和 Town02 v3 的 16 场景真实验证已关闭基础算法/环境工程风险；P0-01/02 只影响蓝车 impact 归因，不阻塞无蓝车 C0；C0 与 Phase 1 又必须先有合格基础 checkpoint | 等 P0-01 至 P0-06 全部完成才训练基础策略；或跳过配置/评估直接批量长训 | P0-BASE、P0-01 至 P0-06、Phase 1 |
| 2026-08-04 | 基础策略正式参数只由版本化 JSON 契约提供；seed 42 先按登记停点分段，PPO 每 epoch 使用单全批并在 P0-BASE-02/04 前不按成功结果调参 | MAPPO 对过多 epoch/minibatch 更敏感；当前最大 critic rollout 为 32000 条，32768 可保证单全批；现有 smoke 只足以证明数值有限，不能提供性能调参依据 | 继续使用 argparse 默认 `minibatch=256`；把命令行历史当配置；在看到成功曲线后挑 seed/停点/阈值 | P0-BASE-01 至 P0-BASE-04、所有正式基础 Run |
| 2026-08-04 | 基础 C0 将训练内健康 gate 与 held-out checkpoint validation 分离；两者均固定完整场景目录和 seed，正式动作只用 Actor mean | 用训练场景选模会产生泄漏；随机动作会混入策略噪声；只保存汇总无法审计奖励和 agent-specific 生命周期 | 继续使用历史 evaluator；只跑有利场景；评估时从策略分布采样；手工填写结果表 | P0-BASE-02 至 P0-BASE-06、后续 C0/C1/C2 迁移 |
| 2026-08-04 | 保留基础 MAPPO 三维 `winner_take_all_v1` 油门/刹车互斥映射；GO/HOLD 只使用拆分后的上界饱和率，踏板零值只解释控制模式 | 互斥符合车辆纵向控制；映射是 PPO latent policy 的确定性环境适配，不破坏 ratio；combined boundary 将正常零踏板误报为饱和 | 用 combined throttle/brake boundary 直接 HOLD；训练中途改二维 action；允许油刹同时生效 | P0-BASE-04、基础训练/C0；二维 signed-longitudinal 只能作为新协议 fresh Run |
| 2026-08-04 | P0-BASE-04 的 GO/HOLD 不使用累计回报；update 5–40 健康优先，60 设最低学习信号，80 只作候选，只有 update 100 的成功/安全/场景族门槛可放行多 seed | return 易受 shaping 与刷分影响；早期硬设高成功率会误杀学习，允许在 80/100 间择优又会引入 checkpoint cherry-picking；0 collision/lane invasion 防止以危险行为换成功 | 按最高回报选模；update 5 即要求 success；update 80 达标就批量训练；看完曲线再调门槛 | P0-BASE-03 至 P0-BASE-06、基础策略实验结论 |
| 2026-08-10 | **SUPERSEDE FOR NEW RUNS**：WTA v1 与原 GO/HOLD v1 保留为历史协议，但修复版改用二维 signed-longitudinal 并在新 pilot 前增加确定性制动塌缩/速度/progress 早期门槛 | u1→u10 brake mode 持续上升、速度下降；两次 u10 C0 均为两车 100% brake 和 16/16 timeout，而 v1 judge 因未注册该行为仍 `CONTINUE_WITH_WARNING`。这证明协议数学可用不等于当前探索/奖励组合适配 | 继续把 u10 当正常早期探索跑到 u20；事后篡改 v1 judge；只调一个权重后 resume 旧权重 | P0-BASE-03R 及所有后续基础策略新 Run；v1 历史证据不变 |
| 2026-08-10 | **IMPLEMENTED R1**：二维策略动作冻结为 `[signed_longitudinal, steer]`，死区 `d=0.05`；PPO 概率只定义在二维 tanh-squashed Gaussian 上，三维 CARLA command 是确定性后适配 | 车辆平面控制可按纵向需求与转向分解；该参数化有自动驾驶 DRL 文献先例。相较 WTA，它消除纵向维度冗余、切换跳变和零 latent 半油门；R1 离线测试证明数学/协议不变量，但不越界声称性能已提升 | 在三维物理动作上计算伪 log-prob；继续加载 WTA Actor/optimizer；把 `d=0.05` 伪装成文献给定常数；在 R1 直接运行 CARLA | 基础 robust_v2、checkpoint/metrics、C0 v2、论文方法章节；R2/R3 仍分别负责奖励与 Critic |
| 2026-08-10 | **IMPLEMENTED R2**：基础 reward v3 冻结为路线完成度势能 + 互斥终局 + 有界风险/控制成本；取消正 speed/alive/centering/yaw 和固定 step/time bonus | PBRS 的 gamma-consistent 形式保证 progress 折扣和望远镜化；自动驾驶 reward 文献支持 route completion 主任务和安全/舒适分离。具体权重不照搬论文，而由本项目 0.1 s、1000-step 与五类轨迹预算预注册，并通过 ±25% 敏感性 | 只减小旧负权重；按 ray/marking 数求和；把控制变化称为物理 jerk；观察 pilot 后追调权重；声称离线排序等于策略性能提升 | 基础 robust_v2、环境 reward、metrics/checkpoint/C0 v3、R2 论文正文；R3 继续负责 Critic，R4/R5 负责门槛和真实行为 |
| 2026-08-11 | 修复版 seed 42 在 u5 执行 v2 机器裁决后保留 `CONTINUE_WITH_WARNING`，不因 0 success 或两车 brake mode=1 人工改判 HOLD | 训练健康、来源、checkpoint、双 C0 和重复性全部 PASS；agent1 progress `4.713 m > 3 m`、整体 timeout `68.75% < 95%`，不满足 R4 all-agent 联合塌缩；唯一 warning 为 9/16 lane invasion | 因短训练观感直接 HOLD；忽略行为证据直接宣称模型可用；跳过 u10 或提前多 seed | 当前 Run 允许同身份 strict resume 到 u10；不等于 `PILOT_GO`，多 seed/选模/对抗训练仍阻塞；C 盘空间是启动 u10 前的主机硬前置 |
| 2026-08-14 | Town05 v3 u5 保留机器裁决 `CONTINUE_WITH_WARNING`，不因 0 success、两车 brake mode=1 或 agent0 单车塌缩信号人工改写；同时不把 CONTINUE 解释为收敛或多 seed GO | 主 C0 的 agent1 progress `3.545 m > 3 m`，timeout `93.75% < 95%`，不满足 all-agent 联合塌缩；双 C0 安全/结局一致，return 告警 13/32=`40.625% < 50%`，无 hard failure | 看到停车观感后事后收紧 HOLD；因 v3 入口漏接而丢弃有效训练 checkpoint；直接跑长训练或多 seed | 同一 Run 只可 strict resume 到 u10；C0 入口修复 `a33a573` 不改训练算法/合同/权重，211/211 与来源校验 PASS |
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

### 第 1 周：冻结基础策略训练契约并完成单 seed 试训

- [x] 第 1 天：完成 P0-00B，修正 terminated/truncated 语义并改写生命周期测试（15 项通过）。
- [x] 第 2 天：完成 P0-00C，`robust_v2` 全链路接入 `active_mask`；一车提前终止 mock 与 18 项回归测试通过。
- [x] 第 3 天上半：完成 P0-00D，修正 GAE/标准化/PPO 健康保护，25 项测试通过。
- [x] 第 3 天下半：完成 P0-00E，生产更新 helper 与完整 MAPPO/PPO 算法回归矩阵共 35 项测试通过。
- [x] 第 4 天上半：完成 P0-00F，冻结 checkpoint/resume 协议，51 项无 CARLA 回归测试通过。
- [x] 第 4 天中段：完成 P0-00F-TB，冻结并模块化奖励/TensorBoard 指标协议，全量 60 项无 CARLA 回归测试通过。
- [x] 第 4 天下半准备：完成 P0-00G-PREP 与三项真实 smoke 修复，补齐互斥终局、逐 episode 证据、periodic/resume、课程顺序、清理和 ratio 数值几何，全量 71 项测试通过。
- [x] 第 4 天下半：P0-00G 自动 smoke、strict validator、update 4 恢复与用户视觉确认通过；success 补验与源码证据已由 P0-BASE-00 正式冻结。
- [x] 第 5 天：完成 P0-BASE-00，提交 `345ea04`；排除个人 Pipfile/Obsidian 状态，93 项回归、场景 validator、入口和历史 strict validator 复核通过。
- [x] 第 6 天上半：完成 P0-BASE-01，固化正式 JSON contract、单全批 PPO 数据使用、registry/manifest、停止规则和独立实验台账。
- [x] 第 6 天下半：完成 P0-BASE-02，冻结基础策略专用 C0 evaluator、场景/episode seeds、输出 schema、恢复/validator 与双 Run 重复性比较。
- [x] 第 7 天前半：在查看正式 pilot 结果前完成 P0-BASE-04，冻结成功/碰撞/timeout、安全和训练健康数值门槛，并实现不可覆盖离线 judge。
- [x] 第 7 天下半：seed 42 的 update 5 fresh、update 10 strict resume、各停点双 C0 与 GO/HOLD judge 已完成；两次均为 `CONTINUE_WITH_WARNING`。

### 第 2 周：决定基础策略批量放行，再进入对抗可信化

- [x] 第 1 天前半：修复版 seed 42 u5 正式训练、双 C0 v4 与 GO/HOLD v2 已完成，结果 `CONTINUE_WITH_WARNING`；未事后放宽阈值。
- [x] 第 1 天后半：Town05 正式 seed-42 u5、strict validator、双 C0 和唯一 GO/HOLD 已完成；机器裁决因 5/16 collision case 的 episode length 漂移 1–2 tick 为 `HOLD`，未启动 u6。
- [x] 第 2 天前半：P0-T05-05 非 Driving 终止与证据修复完成；204/204、32/32 出生点、四方向真实正反例和 fresh 1-update strict smoke 均 PASS。P0-T05-06 的逐 tick 根因修复可顺延，旧裁决不改写。
- [x] 第 2 天后半：猹师傅审核后完成 P0-T05-06、Town05 v3 三合同、210/210 离线回归并提交 `69252d4`；提交后三套 critical-source preflight PASS。CARLA 关闭不影响本项。
- [x] 第 3 天前半：Town05 v3 seed-42 fresh u5、strict validator、双 32-case C0 v3 和唯一 GO/HOLD v3 已完成；裁决 `CONTINUE_WITH_WARNING`，未启动 u6，未放行多 seed。
- [x] 第 3 天后半：猹师傅随后批准同一 Run 长训观察；Run 最终在完整 u50 因 agent0 99% 制动局部最优人工停止，未继续浪费算力到 u100。原始 u51 单条中断记录保留，安全 checkpoint 为 u50；没有伪造 u50 C0 或机器 judge。
- [ ] 第 4 天前半：审核并冻结新的 P0-T05-09 协议；优先修复 post-tanh 动作熵、log-std 上界/熵退火和饱和告警，再按真实轨迹重做 task horizon/reward budget，并把课程切换为 mastery gate。协议变化后必须新 Git、新 RunID、fresh 训练，禁止恢复 u50。
- [ ] 第 2-4 天：仅在 GO 后以同一 commit/config 扩展 seeds `42,123,2024`；失败 Run 也登记。
- [ ] 第 5 天：用冻结 C0 验证全部 periodic checkpoints，生成带 SHA/manifest 的 `best_base_valid.pt`。
- [ ] 第 6 天：开始 P0-01/P0-02，修复对抗碰撞对象和 steer/control 解析，并补相应测试。
- [ ] 第 7 天：继续 P0-03 至 P0-06 与 P0-05M；只有这些门槛通过后才决定是否运行单蓝车 smoke。

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

本项目已经超过“从零搭框架”的阶段，环境、MAPPO、对抗训练、鲁棒训练和 PSRO 都已有代码。R1 二维 signed-longitudinal、R2 有界 reward v3、R3 ValueNorm+Huber、R4 早期塌缩门槛和 R5 生产入口均已闭环，但 Town05 v3 正式长训给出了新的反证：同一 seed-42 Run 到 u50 仍为 0 success，agent0 已稳定到约 99% 制动、0.01 m/s，停车回报还高于尝试行驶。训练健康、Critic、checkpoint 和 CARLA 链路正确不等于任务目标正确；pre-tanh entropy 与 post-tanh 饱和、500 步折扣时域、奖励预算、独立 actor 和 update 驱动课程共同形成了制动吸引域。当前没有可用于多 seed、独立选模、基础策略池或对抗训练的合格 checkpoint。唯一 NEXT 是先审核并实现 P0-T05-09 新协议，完成无 CARLA 回归、真实机制测试和 fresh 小规模 pilot；禁止直接恢复 u50 或继续大规模训练。
