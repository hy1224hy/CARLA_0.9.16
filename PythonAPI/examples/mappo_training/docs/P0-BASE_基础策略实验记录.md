# P0-BASE 基础策略实验记录

> 本文是人工可读实验台账；机器可读索引为 [`experiments/base_mappo_registry.csv`](../experiments/base_mappo_registry.csv)。
> 科学参数真值为对应版本化 JSON 契约，不在本文复制后手工改值。
> 当前状态：Town05 v3 正式 seed-42 Run `P0_BASE_MAPPO_Town05V3_s42_20260814_125226` 在最后完整 u50 因 agent0 持续制动局部最优停止，u50 checkpoint 只保留为失败对照。P0-T05-09/R6 已完成 249/249 离线回归、32/32 live spawn 和完整真实机制 smoke；尚未产生 R6 训练 checkpoint。用户随后要求运行 fresh 50-update 奖励趋势诊断，使用独立诊断合同且禁止选模。
> 运行约束（2026-08-09）：CARLA 服务由用户手动启动并维护，训练程序仅访问 `127.0.0.1:2000`；发现无响应时停止实验并通知用户重启。此前零 episode/update 的无效启动记录已按用户要求从日志和 registry 删除，不进入实验统计。

## 1. 登记规则

每次 fresh Run 在启动前增加一行，失败 Run 也不得删除。恢复训练继续使用同一 RunID，在“分段记录”追加一行。人工记录至少回答：执行了什么、输入身份是什么、得到什么、是否触发 HOLD、下一步是什么。

自动 registry 负责 RunID、契约/Git/源码/场景指纹、当前停点和状态；本文补充运行环境异常、CARLA 人工观察、C0 结论和研究决策。日志、checkpoint、视频等大文件只记录路径与 SHA，不提交到 Git。

## 2. 契约登记

| 契约 ID | 配置 | 用途 | seed | 计划停点 | 状态 |
|---|---|---|---:|---|---|
| `P0_BASE_MAPPO_TOWN02_PILOT_V1` | `configs/base_mappo/p0_base_pilot_v1.json` | Town02 两红车基础 MAPPO 单 seed 工程 pilot | 42 | 5/10/20/30/40/60/80/100 | FROZEN；允许 update 5 fresh |
| `P0_BASE_C0_TOWN02_V1` | `configs/evaluation/p0_base_c0_eval_v1.json` | 无蓝车/无额外交通的 gate 与 held-out validation | gate: 42001；validation: 51001/51002/51003 | 对应训练 update 5/10/20/30/40/60/80/100 | FROZEN；等待正式 checkpoint |
| `P0_BASE_GO_HOLD_TOWN02_V1` | `configs/evaluation/p0_base_go_hold_v1.json` | 训练健康、分段能力、批量放行及 held-out 数值裁决 | pilot seed 42；C0 seeds 继承上行 | 每个登记停点；只有 update 100 可 `PILOT_GO` | FROZEN；SHA256 `c529edf363cdbeaec93f481a8c70ead4a041bae4d4fc59fdb63cdfd9f8053dde` |
| `P0_BASE_MAPPO_TOWN02_CRITIC_V4` | `configs/base_mappo/p0_base_critic_v4.json` | R1–R3 修复版 seed 42 正式 fresh pilot | 42 | 5/10/20/30/40/60/80/100 | FROZEN；SHA256 `1cab6f44faf65daae2cca22fb806c425d81fe386cbc625d51e30339170dac2c1`；R5 已验证生产入口 |
| `P0_BASE_C0_TOWN02_CRITIC_V4` | `configs/evaluation/p0_base_c0_eval_v4.json` | schema-v4 checkpoint 的 gate 与 held-out validation | gate/validation seeds 继承合同 | 对应训练 update 5/10/20/30/40/60/80/100 | FROZEN；SHA256 `7fdb797b704b9ff53587a529c57e68eb9659b3043921dad9312aad60e66dcc68` |
| `P0_BASE_GO_HOLD_TOWN02_R4_V2` | `configs/evaluation/p0_base_go_hold_v2.json` | R3 修复版证据/数值/Critic/持续行为塌缩/能力裁决 | pilot seed 42；C0 v4 seeds 继承评估合同 | u5 首次行为塌缩仅 warning；u10 起连续两里程碑才可行为 HOLD；u100 才可 `PILOT_GO` | FROZEN；SHA256 `668799076a7e08bfd37457457929df3f1bbf40fbc8ce64ef068f97e839925140`；R4 提交与 R5 入口均已验收 |
| `P0_BASE_MAPPO_TOWN05_V1` | `configs/base_mappo/p0_base_town05_v1.json` | 当前 Town05 四方向/四家族基础 MAPPO fresh pilot | 42 | 5/10/20/30/40/60/80/100 | FROZEN；SHA256 `eeb12304d9a262ac4297ede90bd651717d7d59cb31cb8ba0f384835e0ed84bde`；`a11cf70` 真实入口门禁 PASS，允许正式 fresh u5 |
| `P0_BASE_C0_TOWN05_V1` | `configs/evaluation/p0_base_c0_town05_v1.json` | 当前 Town05 gate 与 Road 43/6 held-out validation | gate 42001；validation 51001/51002/51003 | 对应 Town05 登记停点 | FROZEN；SHA256 `ea2e17f633e3b26eaedbbcf5f42cbaf9b14bc7077790e3dfcba438c0c276cab8` |
| `P0_BASE_GO_HOLD_TOWN05_V1` | `configs/evaluation/p0_base_go_hold_town05_v1.json` | 当前 Town05 证据、健康、行为与能力裁决 | 继承 Town05 训练/C0 合同 | u5 首门；u100 才可 `PILOT_GO` | FROZEN；SHA256 `bbe6903c1ba9fb5932eafc0ba4dd250f0f6254d2e9f9066d0da4c5d940c7e9c4` |
| `P0_BASE_MAPPO_TOWN05_R6_V4` | `configs/base_mappo/p0_base_town05_r6_v4.json` | P0-T05-09/R6 fresh seed-42 pilot；post-tanh entropy、reward-v4、mastery gate | 42 | 5/10/20/30/40/60/80/100/120/140/160 | FROZEN；SHA256 `53397a3c13ace5c014149127ab3191e38f2db1c85a35bdb86a1bfa8056f95f8e`；OFFLINE PASS，未运行 CARLA |
| `P0_BASE_C0_TOWN05_R6_V4` | `configs/evaluation/p0_base_c0_town05_r6_v4.json` | schema-v6、20维 time-aware R6 gate/held-out evaluation | gate 42001；validation 51001/51002/51003 | 对应 R6 登记停点 | FROZEN；SHA256 `3772d368d892f524c98b0494fa64c6f072061a8b2ec0dbcb4366c4368502866f` |
| `P0_BASE_GO_HOLD_TOWN05_R6_V4` | `configs/evaluation/p0_base_go_hold_town05_r6_v4.json` | R6 分布审计、mastery、任意单 agent 持续塌缩与能力裁决 | 继承 R6 训练/C0 合同 | u5 首门；u10 起同一 agent 相邻里程碑持续塌缩即 HOLD；u160 才可 `PILOT_GO` | FROZEN；SHA256 `c094b6744211d455737166057deb14dde08a23913cb2550d695c5336e6f89425` |
| `P0_DIAG_MAPPO_TOWN05_R6_U50_V1` | `configs/base_mappo/p0_diag_town05_r6_u50_v1.json` | 用户授权的 R6 fresh 50-update 奖励/塌缩趋势诊断 | 42 | 5/10/20/30/40/50；合同长期边界仍保留到160 | FROZEN；SHA256 `b50df18c2f1eac0b904c55f022efd31fac64cf1d64617900ad638cc91adf82c4`；不受正式 C0 合同绑定，禁止选模/策略池 |

## 3. Run 总表

| 日期 | RunID | Git SHA | 契约 SHA256 | 当前 update | 训练状态 | strict validator | C0 | GO/HOLD | 下一动作 |
|---|---|---|---|---:|---|---|---|---|---|
| 2026-08-14 | `P0_BASE_MAPPO_Town05V3_s42_20260814_125226` | origin `5dad4d4`；C0 入口修复 `a33a573` | `d39956e0...a08e92` | 50 | `INTERRUPTED / DIAGNOSTIC HOLD`；800 完整 episodes，另保留 1 条未完成 u51 中断记录 | u1–u50 PPO/ratio/reward/finite/checkpoint 健康；原目录 strict 如实因 u51 单条 episode 无对应 metrics 而拒绝；u50 checkpoint schema/SHA PASS | u5 双 C0 原结论保留；没有伪造 u50 C0。训练窗口 u41–u50 为 0 success、2 collision、0 offroad、158 timeout，A0 brake mode 99.01% | u5 机器输出仍为 `CONTINUE_WITH_WARNING`；u50 研究结论为局部最优 `DIAGNOSTIC HOLD`，不续训、不放行多 seed | 设计 squashed-action entropy、奖励时域/预算和 mastery curriculum 新协议；fresh 小规模验证 |
| 2026-08-12 | `P0_BASE_MAPPO_Town05_s42_20260812_164411` | `6c28a9141d3ede87f7f54734cc277973381e6f05` | `eeb12304...84bde` | 5 | `SEGMENT_COMPLETED / EVIDENCE_HOLD`；80 episodes、65178 transitions，未启动 u6 | PASS：updates 1–5、197 required tags、575 scalar tags；PPO/ratio/reward/finite/ValueNorm/checkpoint/gamma/cleanup 均通过 | 两轮各 0 success、12 collision、4 timeout；各自 validator PASS；5/16 碰撞 case 的长度相差 1–2 tick，repeatability HOLD | `HOLD`：唯一 hard failure 为 C0 离散重复性；训练健康和 R4 行为评估 PASS；8/16 lane-invasion warning | 冻结本 Run；诊断 collision callback/world frame/episode step 的 1–2 tick 漂移，禁止 resume、多 seed和调参 |
| 2026-08-12 | `T05_parallel_smoke_s42_20260812_161632` | `a11cf70721529d8960d0457a9cab219b54b2d563` | N/A：冻结诊断 smoke 配置 | 1 | `CLEAN-COMMIT SMOKE PASS`；4 episodes/960 transitions，0 collision/4 timeout | PASS：197 required tags / 582 scalar tags；ratio precheck 0/0；PPO 执行；reward step residual 0；训练/环境 gamma 0.997；清理 0/0 | 未运行：只验证训练入口 | 不裁决 | P0-T05-04：正式新 RunID、随机初始化 seed 42 到 u5 |
| 2026-08-12 | `T05_parallel_smoke_s42_20260812_160851` | `2d18f68` + 未提交初始化修复 | N/A：诊断 smoke | 1 | `PRECOMMIT DIAGNOSTIC PASS`；4 episodes/954 transitions，1 collision/3 timeout | PASS：197 required tags / 583 scalar tags；训练与环境 gamma 均 0.997；清理 0/0 | 未运行：只验证训练入口 | 不裁决 | 提交初始化修复，在精确 commit 上重复完整 live spawn 与单更新 smoke |
| 2026-08-12 | `T05_MIGRATION_SMOKE_20260812_121336` | 未提交 Town05 迁移工作树 | N/A：诊断 smoke | 1 | `DIAGNOSTIC ONLY / PROTOCOL DEVIATION`；4 episodes/960 transitions | 旧链路指标通过，但无法检查环境实际 gamma | C0 smoke 工具链 PASS；评估环境为 0.997 | 不裁决 | 旧产物封存；修复环境 gamma 注入、提交后 fresh 重跑 smoke |
| 2026-08-11 | `P0_BASE_MAPPO_R3_Town02_s42_20260811_112458` | `ff5ddeb` | `1cab6f44...ac2c1` | 5 | `HISTORICAL DIAGNOSTIC`；当时完成 u5，后续审计发现 reward-gamma 协议偏差 | 旧 strict validator PASS，但未记录/核对环境实际 gamma | 主/重复均 0/16 success、5/16 collision、11/16 timeout；C0 gamma 0.997，与训练实际 0.99 不一致 | 原机器输出 `CONTINUE_WITH_WARNING` 保留，不再具有续训放行效力 | 禁止 resume/选模；Town05 修复提交后 fresh 重跑 smoke 与正式实验 |
| 2026-08-11 | `P0_R5_MAPPO_smoke_s42_20260811_104100` | `00e16f3` | N/A：诊断 smoke，不冒充正式合同 Run | 3 | `R5 PASS / DIAGNOSTIC ONLY` | PASS：3 updates / 12 episodes / 197 required tags / 575 scalar tags；u2 受控中断后严格恢复 | 未运行：u1–u3 不是登记停点，禁止伪造 C0 | 未裁决：不满足正式 update 5/10 入口 | 永久禁止 resume/pretrained/选模；新 RunID、随机初始化运行正式 seed 42 到 u5 |
| 2026-08-09 | `P0_BASE_MAPPO_Town02_s42_20260809_204729` | `8c6d582` | `342a1f...f6f` | 10 | `INTERRUPTED / DIAGNOSTIC HOLD`；安全 checkpoint 仍为 u10 | PASS：10 updates / 160 episodes / 583 scalar tags；u11 的 2 条临时 episode 不计入 | 两次 gate 均 0/16 success、0/16 collision、16/16 timeout；离散重复性 PASS；确定性两车 100% brake mode | 冻结 judge 保留 `CONTINUE_WITH_WARNING`；研究者因未注册的制动塌缩失败模式停止 | 禁止 resume；设计 v2 动作/奖励/Critic 与早期门槛，fresh 新 Run |
| 2026-08-09 | `P0_BASE_MAPPO_Town02_s42_20260809_173321` | `8a0bd9d` | `342a1f...f6f` | 5 | `EVIDENCE_HOLD`；模型仅诊断 | PASS：5 updates / 80 episodes / 570 scalar tags | 两次 gate 均 0/16 success、0/16 collision、16/16 timeout；离散重复性 PASS | 来源预检 HOLD：resolved config 漏记 seed，未进入性能裁决 | 提交证据修复后新 RunID fresh 重跑 update 5；禁止 resume 本 Run |

## 4. 已运行记录

### P0-T05-09/R6：离线与真实机制 smoke 完成，尚无训练 Run

- 新协议不加载失败 u50：训练合同 `source_mode=fresh`，新 RunID 前缀为 `P0_BASE_MAPPO_Town05R6_s42_`；共享 actor 和行为克隆均未接入。
- tanh 后动作熵使用当前策略新鲜样本的 Jacobian-corrected `-log π(a|s)`；Normal entropy 降为诊断。`log_std∈[-2,0]`，熵系数 u1 `0.01` 到 u160 `0.001`，并记录 pre/post entropy gap、raw/effective `log_std` 和确定性动作饱和。
- reward-v4 以 `γ=0.999` 的路线完成度 PBRS 为唯一正稠密项；任务 500 tick、剩余时间进入第 20 维观察；成功 `+20`、碰撞/offroad `-50`、timeout `-(10+10×remaining)`。车道中心 0.5 m 内为 0，外侧只有最大 `-0.002/tick` 的封顶成本，合法跨虚线后豁免 10 tick，不发中心驻留正奖励。
- stall 只诊断且对 blocker/复杂交互豁免；课程必须连续两个全 rollout mastery window 通过才晋级；R6 GO/HOLD 改为同一任意单 agent 在相邻里程碑持续塌缩即可 HOLD。
- 全量离线回归 `249/249 PASS`。2026-08-20 的 32/32 live spawn 及 R6 机制 smoke 全部 PASS：20 维倒计时、reward residual `0`、timeout T/X、两类 stall 豁免、LiDAR approach/projected TTC、Broken grace 和 cleanup `0/0` 均闭环。第一份 FAIL 报告只暴露验证夹具车辆重合，PPO 未启动；修正后报告为 `diagnostics/r6_live/P0_T05_09_R6_MECHANISM_SMOKE_20260820_121926/report.json`。下一步按用户要求运行独立 fresh u50 诊断，当前仍无性能结论。

### `P0_BASE_MAPPO_Town05V3_s42_20260814_125226`：u50 人工停止与局部最优诊断

- 训练从 u5 严格恢复并保持 origin Git、合同、seed、RunID、PPO、奖励与课程参数不变；最后完整边界 u50 共 800 episodes，另有 1 条未完成 u51 episode 原样保留。u50 checkpoint SHA256 为 `20b86366f3246a5603ac602652e4f86e904752ba63d3ca2416b2990cdd92adba`。
- 全段为 0 success、23 collision、48 offroad、729 timeout。u41–u50 的 160 episodes 为 0 success、2 collision、0 offroad、158 timeout；agent0 平均速度 `0.0101 m/s`、完成度 `0.0138`、brake mode `0.9901`、signed-longitudinal `-0.9328`，已经稳定进入全制动局部最优。
- u41–u50 的平均 episode return 为 agent0 `-5.7736`、仍在尝试行驶的 agent1 `-8.3456`。停车通过减少纵向变化、车道偏离和违规成本获得更高相对回报，因此 PPO 是在优化失配目标；50 次更新均无 skipped、nonfinite、ratio 或 reward 守恒错误。
- 模型使用 pre-tanh Normal entropy，而环境执行 tanh 后动作；训练熵和 log-std 上升时，agent0 的实际动作仍可饱和为 99% 制动。`gamma=0.997` 在 500 步末只保留 `0.2226` 的任务信号，加上独立 actor 和 update 驱动课程，共同固化该吸引域。critic EV 上升只说明能预测全超时窄分布，不代表驾驶能力。
- 研究者要求停止后，训练进程退出；CARLA 恢复 async/None，并清理终止前待提交命令产生的 2 辆训练车与 6 个传感器，最终 actor `0/0`。registry/manifest 更新为 `INTERRUPTED`、安全边界 u50，禁止 resume 本 checkpoint 作能力训练。
- 完整数值、数学原因和优化顺序见 [u50 制动局部最优诊断](./P0-BASE-Town05V3_seed42_u50局部最优诊断.md)。下一步不是继续到 u100，而是新建 squashed-action entropy、task horizon/reward budget、mastery curriculum 协议，回归通过后 fresh 小规模验证。

### `P0_BASE_MAPPO_Town05_s42_20260812_164411`

- 输入：Git `6c28a9141d3ede87f7f54734cc277973381e6f05`，训练合同 `P0_BASE_MAPPO_TOWN05_V1`，seed 42，fresh 随机初始化，目标只到 update 5；训练、C0、GO/HOLD 三合同 critical-source clean preflight 通过。
- 训练：自然结束于 u5，未启动 u6；5 updates、80 episodes、65178 agent-transitions。success 始终为 0，collision 从 u1 的 12/16 降至 u5 的 1/16，timeout 从 4/16 升至 15/16。strict validator、PPO/ratio、reward residual、finite/schema、ValueNorm、checkpoint provenance、训练/环境 gamma `0.997/0.997` 全部 PASS。
- checkpoint：`logs/P0_BASE_MAPPO_Town05_s42_20260812_164411/periodic/ckpt_update_000005.pt`，SHA256 `b1c93b14d731ffc62e57ec294e09ff9332f1b358de6f8ea7b2b9569a336eeb23`。
- C0：主 `P0_BASE_C0_gate_u000005_20260812_170923`、重复 `P0_BASE_C0_gate_u000005_20260812_171236`；两轮均为 0 success、12 collision、4 timeout，且各自 identity/schema/reward/summary/cleanup validator PASS。Agent 0 的确定性 brake mode 为 1.0、均速 0.1588 m/s、progress 2.8379 m；Agent 1 近乎全程 coast、均速 0.8026 m/s、progress 10.2560 m。4 个静止场景全部 timeout，其余 12 个交互场景全部 collision。
- 重复性：两轮宏观结局和逐 case outcome/end reason/reached 均相同，但场景 07/08/09/10/15 的碰撞 episode length 分别相差 1/1/2/1/1 tick。冻结协议要求 `episode_length` 精确相等，因此 repeatability 为 `HOLD`，未事后放宽。
- judge：`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260812_171550/go_hold_decision.json`，结论 `HOLD`。训练健康与 R4 行为坍缩评估均 PASS；唯一 hard failure 为 `C0 discrete repeatability did not PASS`，另有 8/16 lane-invasion warning。该结论不是因 u5 零成功而误杀，也不表示策略已有能力；它首先阻断证据不稳定的续训决策。
- 环境：训练和双 C0 无 CARLA 卡死/RPC timeout，最终恢复 Town05 async/None、vehicle/sensor `0/0`。完整记录见 [Town05 seed-42 u5 正式训练与 GO/HOLD 记录](./P0-BASE-Town05_seed42_u5正式训练与GO-HOLD记录.md)。当前唯一动作是诊断 1–2 tick 终止漂移；禁止 u6、u10、多 seed、选模或调参。

### `T05_parallel_smoke_s42_20260812_161632`

- 输入身份：初始化修复提交 `a11cf70721529d8960d0457a9cab219b54b2d563`；训练、C0、GO/HOLD 三套冻结合同在启动 CARLA 前均通过 critical-source clean preflight。checkpoint 的 origin/last Git 与该提交一致；全仓 dirty 列表只含合同关键路径之外的用户既有 `AGENTS.md`、`Pipfile` 和 Obsidian workspace，不污染受控源码身份。
- 真实场景：Town05 正式训练池 16/16 逐场通过位置/初速度门槛；此前失败的 09/11 均为位置误差约 0.01 m、速度 4.89/5.00 m/s。环境退出后恢复 async/None，vehicle/sensor 0/0。
- 训练：1 update、4 个分层 episode、每车 480 transitions（总计 960）；随机策略为 0 success、0 collision、4 timeout，平均 return -4.13。该性能只说明模型尚未训练，不用于调参、成功率比较或选模。
- 算法/证据：strict validator 验证 197 个必需标签、582 个 TensorBoard scalar 和 4 条 episode；两车 preupdate max ratio error 都为 0，`update_skipped=0`，approx KL 为 0.00122/0.00177，reward 最大逐步残差 0、无 missing/nonfinite，checkpoint `training_config.gamma=environment_reward_gamma=0.997`，SHA256 `9882404b...d8775f7b`。
- 产物：`diagnostics/town05_gamma_alignment_smoke_clean/T05_parallel_smoke_s42_20260812_161632`。身份是 `CLEAN-COMMIT SMOKE`，只放行正式训练启动；其 checkpoint 禁止 resume、比较或选模。

### `T05_parallel_smoke_s42_20260812_160851`

- 首次完整 live preflight 在训练前以 14/16 失败：场景 09 agent0 和场景 11 agent1 均停留在 0.5 m 预偏移位置且速度为 0，而隔离重复同一场景可正常达到约 4.89/5.00 m/s。故障在 rollout、Critic 与 PPO 之前，归因为 CARLA 异步 transform/target-velocity 初始状态提交瞬态，不是训练指令、梯度 warning、初始探索移动缓慢或训练时长不足。
- 修复使用整组、有限的同步状态提交：清零线/角速度，分 tick 提交预偏移 transform 和目标速度，核对全部车辆位置/速度；最多 5 次，仍失败即拒绝 reset。修复后真实 Town05 16/16 初始状态通过，丢失首次速度请求的回归及全量 184/184 通过。
- 工作树诊断训练完成 1 update、4 episodes、954 agent-transitions，平均 return -4.77，1 collision/3 timeout；这是随机初始策略的短烟测结果，不用于成功率或收敛判断。strict validator 验证 197 个必需标签和 583 个 TensorBoard scalar 标签，checkpoint 中 `training_config.gamma=run_metadata.runtime.environment_reward_gamma=0.997`，退出后恢复 async/None 且 vehicle/sensor 0/0。
- 产物：`diagnostics/town05_gamma_alignment_smoke_precommit/T05_parallel_smoke_s42_20260812_160851`。它含未提交初始化修复，身份仅为 `PRECOMMIT DIAGNOSTIC`；不得 resume、比较或选模，最终门槛必须在修复提交后 fresh 重跑。

### `P0_BASE_MAPPO_R3_Town02_s42_20260811_112458`

- 追溯勘误（2026-08-12）：Git `ff5ddeb` 的训练入口没有把 `args.gamma=0.997` 传入 `CarlaMultiAgentEnv`，所以 reward-v3 PBRS 实际按默认 `0.99` 计算；PPO/GAE 和后续 C0 则按 0.997。现有 reward conservation 只校验分量求和，无法识别这一配置串线。该 Run 的算法/工具链诊断仍可复查，但“训练—评估奖励协议一致”和正式可续训身份不成立；原机器裁决作为历史输出不篡改，人工处置更新为禁止续训、比较和选模。

- 输入：R3 训练合同 `P0_BASE_MAPPO_TOWN02_CRITIC_V4`、seed 42、fresh 随机初始化、Git `ff5ddebbea7e36a2d6d080e827d1930c245ff586`，显式 `--cpu`，只访问用户维护的 CARLA `127.0.0.1:2000`。
- 训练：自然结束于 u5，未启动 u6；5 updates、80 episodes，success 始终为 0，collision 从 100% 降至 37.5%，timeout 升至 62.5%，u5 平均长度 865.81。5 次 PPO 均执行，ratio/reward/finite/schema 检查和 ValueNorm count `1..5` 均通过。
- checkpoint：`logs/P0_BASE_MAPPO_R3_Town02_s42_20260811_112458/periodic/ckpt_update_000005.pt`，SHA256 `4652aa5979575db0799bc70ba3fc9838efaf0c28476f35f87ecb38337e815e72`；schema-v4、checksum、模型/optimizer/RNG/ValueNorm 与合同身份审计 PASS。
- C0：主 `P0_BASE_C0_gate_u000005_20260811_115047`、重复 `P0_BASE_C0_gate_u000005_20260811_120044`；两轮均为 0 success、5 collision、11 timeout，逐病例离散字段无 mismatch，validator/summary/reward/cleanup 全部 PASS。主 C0 两车 brake mode 均为 1.0、平均速度 `0.0350/0.0714 m/s`、平均 progress `2.4963/4.7130 m`，9/16 episode 有 lane invasion。
- judge：`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260811_120658/go_hold_decision.json`，结论 `CONTINUE_WITH_WARNING`。唯一 warning 为 9 个 gate episode 出现 lane invasion；agent1 progress 高于 3 m、整体 timeout 低于 95%，故 R4 的 all-agent 联合塌缩不成立，Critic 联合退化也未成立。
- 环境：训练、双 C0 和 judge 无 CARLA 断连/卡死，结束后 async/None、vehicle/sensor `0/0`。第一轮 C0 后 C 盘被后台程序占满；经授权删除约 449 MB 百度自动更新临时缓存后完成余下验证，但 C 盘随后再次为 0。该主机风险不污染已完成证据，却要求 u10 前先稳定释放系统盘空间。
- 完整数值、身份、哈希和解释见 [u5 正式训练与 GO/HOLD 记录](./P0-BASE-R3_seed42_u5正式训练与GO-HOLD记录.md)。当前不得调参、fresh 重跑或多 seed；只允许在磁盘条件恢复后按同一身份严格续至 u10，并在 u10 judge 中链接本次 u5 决策报告。

### `P0_BASE_MAPPO_Town02_s42_20260809_204729`

- 输入：训练契约 `P0_BASE_MAPPO_TOWN02_PILOT_V1`，seed 42，fresh，来源 Git `8c6d582849c12c146a690e52772f05c8dcdcc835`；resolved config 已实测包含 seed 42 和全量 contract trainer args。
- update 1–5：80 episodes、44614 个 agent-transition，约 13 分 34 秒；训练内 success 始终 0，collision 从 1.0 降到 0.625，timeout 从 0 升到 0.375，平均回合从 130.75 增至 550.38。KL、clip、ratio、reward residual、finite/schema 和 skipped update 均通过硬门槛。
- checkpoint：`logs/P0_BASE_MAPPO_Town02_s42_20260809_204729/periodic/ckpt_update_000005.pt`，SHA256 `8a67c57d6755ebd421ea91e4d545aa77f1b260639a0d5fe208c2b6915d2f5054`。
- C0：主 `P0_BASE_C0_gate_u000005_20260809_210149`、重复 `P0_BASE_C0_gate_u000005_20260809_211015`；两次均 16 timeout、0 success、0 collision，离散签名完全一致，所有 identity/schema/reward/summary/cleanup 校验 PASS。主 C0 有 6/16 lane-invasion episode。
- judge：`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260809_211849/go_hold_decision.json`，预注册、Run origin、seed、checkpoint、训练、C0 和重复性证据均 PASS；结论 `CONTINUE_WITH_WARNING`。warning 为 Critic gradient clipping activation 连续 5 updates 大于 0.8，以及 6 个 gate episode 出现 lane invasion；无 hard/capability failure，多 seed 未放行。
- 判断：update 5 仍是健康停点，0 success 按预注册规则不触发 HOLD；确定性策略尚未形成到达能力，因此未调参并按原身份续至 update 10。

#### update 6–10 分段与诊断

- 严格恢复：沿用同一 RunID、Git `8c6d582`、resolved config 与 seed 42，从 update 5 完整边界恢复；updates 1–10 连续、160 episodes、583 个 TensorBoard scalar tags，strict validator PASS。
- 训练趋势：updates 6–10 success 始终为 0；collision rate 依次为 `0.6875/0.5625/0.4375/0.3125/0.4375`，timeout rate 依次为 `0.3125/0.4375/0.5625/0.6875/0.5625`，平均回合长度升至约 `607–836` 步。Actor 的 KL、clip、ratio、梯度与有限性检查保持健康。
- checkpoint：`logs/P0_BASE_MAPPO_Town02_s42_20260809_204729/periodic/ckpt_update_000010.pt`，SHA256 `4917f2259d4afd77cf621b862fbe17f2a30819961e86a5c08834a4408a677436`。
- Critic 诊断：update 10 未裁剪梯度范数约 `85.84`，裁剪上限 `0.5`、激活率 `1.0`；explained variance 约 `0.00037`，return target 为 `-15.19±7.24`，value prediction 为 `-4.90±0.08`。这表示 Critic 当前几乎输出常数、拟合不足，会使 advantage baseline 噪声较大并拖慢学习；但更新前 ratio 误差为 0，Actor KL/clip 正常，算法回归也已通过，故没有证据将其归类为 GAE、PPO 梯度方向或实现错误。
- 行为诊断：update 10 的随机训练动作中，两车 brake mode 约为 `68%`、平均速度仅约 `0.055 m/s`；主 C0 的确定性 Actor mean 更明确地退化为两车 `throttle_mode=0`、`brake_mode=1.0`，平均速度分别约 `0.016/0.024 m/s`。因此 timeout 的直接机制是 Actor 确定性策略选择全程刹车；Critic warning 是可能强化/延长该局部最优的间接学习质量问题，不等于梯度实现错误。
- C0：主 `P0_BASE_C0_gate_u000010_20260809_220248`、重复 `P0_BASE_C0_gate_u000010_20260809_221225`；两次均 16 timeout、0 success、0 collision、episode length 全为 1000，离散签名完全一致，全部 validator 与 cleanup audit PASS。主 C0 有 11/16 lane-invasion episode。
- judge：`decisions/base_mappo/P0_BASE_GO_HOLD_u000010_20260809_222131/go_hold_decision.json`；来源、训练健康、checkpoint、C0 与重复性全部 PASS，结论 `CONTINUE_WITH_WARNING`。warning 为 Critic gradient clipping activation 连续 10 updates 大于 0.8，以及 11 个 gate episode 出现 lane invasion；无 hard/capability failure，多 seed 未放行。
- 冻结 judge 判断：update 10 尚未证明基础驾驶能力，但 v1 门槛在 updates 5–40 只执行既有健康判定，0 success 与 brake mode 塌缩均未注册为 HOLD，因此机器裁决仍为 `CONTINUE_WITH_WARNING`。该结果必须保留，不得事后改写。

#### update 20 续训停止与诊断封存

- 2026-08-09 曾按 v1 裁决尝试从 update 10 严格恢复到 update 20；恢复身份检查通过，但在 update 11 仅完成 2/16 rollout episode、尚未执行本 update PPO、metrics commit 或 checkpoint 保存时，研究者要求先判断是否已可定位失败原因。
- 训练进程已停止；`latest.pt` 与 periodic checkpoint 的安全边界均保持 update 10。两条 update 11 episode 只属于安全 checkpoint 之后的未提交前缀，不纳入训练、C0、judge 或论文统计；任何后续 strict resume 原本也会按 recorder 协议自动裁掉该前缀。
- registry/manifest 记为 `INTERRUPTED`，不是 CARLA 环境导致的性能失败。强制停止后 CARLA timing 被人工恢复为 async/None，随后服务无响应，需用户重启；该服务器状态不改变 update 1–10 和两次 C0 已完成证据。
- 新增诊断结论：训练中 brake mode 从 u1 的约 `46%/55%` 增至 u10 的约 `69%/68%`，agent 0 平均速度从约 `0.692` 降到 `0.054 m/s`；同时 collision 下降而 timeout 上升。u10 的 agent 0 每回合 `steer_rate=-272.23`、`jerk=-123.14`、`step=-80.47`，远大于 `speed=+26.79`，说明当前奖励规模使带噪移动比保持制动更差。三维 WTA 的独立 throttle/brake 隐变量、`init_log_std=0` 和只惩罚 throttle 变化而不惩罚 brake 变化共同放大该局部最优。
- Critic 的连续裁剪、explained variance 近 0 与近常数 value output 会增加优势噪声，是必须修复的学习质量问题；但 ratio/KL/clip/finite 与生产算法测试均未显示 GAE/PPO 概率或梯度方向实现错误，因此不把 Critic warning 单独定为全程刹车的直接原因。
- 研究决策：不再消耗算力延长 v1 Run。冻结 judge 的历史结论继续保留为 `CONTINUE_WITH_WARNING`，人工研究结论记录为 `DIAGNOSTIC HOLD`；所有会改变 action dimension、奖励规模、value normalization 或早期门槛的修复必须形成新协议、新 Git 提交、新 RunID，并从随机初始化 fresh 训练，禁止继续利用 u10 权重作正式基线。
- R1 处置（2026-08-10）：动作协议已独立升级为二维 `[signed_longitudinal, steer]`，采用 `d=0.05` 连续死区映射 CARLA 三通道控制；同步升级 checkpoint schema、训练/metrics/C0 协议和只读 legacy validator。全量 140 项无 CARLA 回归通过，奖励与 Critic 未修改，也未启动 CARLA。该提交只关闭 R1 工程/理论验收，不恢复或改写本 Run 的任何训练、C0、judge、registry 或 checkpoint 证据；R2 奖励预算校准前仍禁止新正式 pilot。论文推导与测试计划见 [R1 二维有符号纵向动作协议](./R1_二维纵向动作协议_论文正文与实验设计.md)。
- R2 处置（2026-08-10）：基础奖励升级为 reward v3，以 gamma 一致路线完成度势能为唯一正稠密任务信号，冻结成功 `+10`、碰撞 `-20` 与有界超速/控制变化/风险/车道成本；油门和刹车按实际 signed-longitudinal command 对称计费。五类轨迹排序、预算边界和核心权重 ±25% 敏感性均通过，训练/环境/metrics/checkpoint/C0 v3 协议同步；全量 160 项无 CARLA 回归通过。该处置同样不改写旧 Run 证据、不声称真实性能提升，R3–R5 前仍禁止 pilot。方法、文献依据和局限见 [R2 奖励预算校准](./R2_奖励预算校准_论文正文与实验设计.md)。
- R3 处置（2026-08-10）：Critic 改为官方 MAPPO 风格的 ValueNorm target 与归一化 Huber，rollout/bootstrap/GAE 始终使用反归一化 value；normalizer 完整状态纳入 schema-v4 checkpoint，metrics-v5 同时审计原始与归一化尺度，C0 v4 绑定新契约。真实 CARLA 固定脚本确认 R2 守恒/静止/成功/碰撞信号链，并在 536 个真实 transition 的受控 A/B 中把梯度裁剪率从 100% 降至 0.33%、留出 MSE 从 63.8371 降至 57.1734。该诊断不是性能 pilot，不改写旧 Run，R4/R5 前仍禁止 fresh 长训。详见 [R3 文档](./R3_Critic价值归一化_论文正文与真实CARLA验证.md)。
- R4 处置（2026-08-11）：新增 GO/HOLD v2，精确绑定 R3 训练合同、C0 v4、metrics v5 和 checkpoint v4。行为 HOLD 要求双 Agent 确定性 brake `>=0.95`、速度 `<=0.1 m/s`、每回合 progress `<=3 m`、timeout rate `>=0.95`、零成功同时成立，并在相邻两个里程碑持续；u5 只 warning。Critic HOLD 要求低 EV、近常数预测、高相对 MAE 和近连续裁剪四项连续五 update。旧真实 C0 JSONL 回放在 u5 warning、u10 `BEHAVIORAL_COLLAPSE`，移动但零成功样本不 HOLD；v1+v2 专项和全量 179 项回归通过。用户重启 2000 端口后，teleport 两车在 step 1/2、scripted 两车在 step 32/55 成功到达；两次终局、无碰撞、terminated/truncated 和清理审计 PASS。该旧 smoke 的 `success=100/time_bonus` 只验生命周期，不作为 reward-v3 数值证据。旧 v1 judge 不改写，R4 不构成性能 pilot；详见 [R4 文档](./R4_GO-HOLD-v2早期塌缩门槛与回归验证.md)。
- R5 处置（2026-08-11）：在真实 CARLA 0.9.16/Town02 上复跑策略无关 R2/R3 诊断，10 项机制条件全部通过；随后以唯一生产入口完成 fresh u1、u2 episode 0 后受控中断、严格恢复完成 u2、再次恢复完成 u3。最终 12 条 episode、3 次 PPO、197 个必需标签、575 个 TB scalar 标签及全部 schema-v4 checkpoint/manifest/checksum/ValueNorm/RNG/optimizer 审计通过，未完成前缀被自动裁剪；最终 CARLA 为 async/None、vehicle/sensor `0/0`，无 RPC timeout 或卡死。该 Run 只验证工程链路，不执行正式 C0/judge、不用于 pretrained/选模，也不声称短训练已获得驾驶能力；详见 [R5 文档](./R5_最终训练入口真实CARLA_smoke验证.md)。
- 开训前处置（2026-08-11）：从 Git 祖先、三份合同 SHA、16+8 场景身份、R1–R4 38 项专项、179 项全集成、R5 strict/checkpoint/真实机制报告、CARLA 0.9.16 空闲状态、GPU/磁盘和正式 u5 命令解析逐项复核，全部通过。发现当前操作文档的 C0/judge 示例仍残留 v1、C0 协议页路径/SHA 残留 v3，已统一到 C0 v4 与 GO/HOLD v2；算法、配置和阈值未改。只放行新 RunID、随机初始化的 seed 42 到 u5，详见 [全链路检查记录](./R1-R5_正式训练前全链路检查记录.md)。

### `P0_BASE_MAPPO_Town02_s42_20260809_173321`

#### 预注册输入与训练结果

- 输入：训练契约 `P0_BASE_MAPPO_TOWN02_PILOT_V1`，seed 42，fresh，目标 update 5，来源 Git `8a0bd9d5edaa19cad100c6738145f7a8378f515c`，仅访问用户维护的 `127.0.0.1:2000` CARLA 0.9.16。
- 输出：`logs/P0_BASE_MAPPO_Town02_s42_20260809_173321`；周期 checkpoint `periodic/ckpt_update_000005.pt`，SHA256 `dc4df6134e8744a9eddacf3d96d2d2b66a022800bfea2a42244fe2c311e47a75`。
- 训练段：5 updates、80 episodes、37014 个 agent-transition，约 11 分 35 秒；strict validator 对 updates 1–5、每 update 16 episodes、PPO 更新、ratio 和指标 schema 全部 PASS。
- 训练行为趋势：success 始终为 0；collision rate 从 updates 1–4 的 1.0 降到 update 5 的 0.8125，同时 update 5 出现 0.1875 timeout，平均回合长度从 80.88 增至 401.75。KL、clip fraction、reward residual、非有限值和 skipped update 均未触发硬门槛。

#### 独立 C0 与失败原因

- 主 gate：`evaluations/P0_BASE_C0_gate_u000005_20260809_174729`；16/16 timeout、0 success、0 collision、两车均 0 次到达，validator 全项 PASS。
- 重复 gate：`evaluations/P0_BASE_C0_gate_u000005_20260809_175513`；离散签名逐 case 完全一致，repeatability PASS。第二次只验证重复性，不计作额外 16 个性能样本。
- 行为诊断：确定性 Actor mean 在 gate 中表现为制动/近乎静止，所有 case 跑满 1000 步；“0 collision”不能解释为已学会基础驾驶。
- GO/HOLD judge 在性能门槛前返回 `formal training seed differs from the pilot binding`。实际 seed 42 已由冻结契约、registry 与 checkpoint 交叉确认；根因是实验证据生成复用了不含 seed 的 checkpoint 恢复参数子集，导致 `resolved_config.json.resolved_training_config.seed` 缺失。
- 错误归类：**代码/证据序列化接口缺陷**，不是环境错误，也不是启动指令错误。环境侧 CARLA client/server 均为 0.9.16，训练完整结束且两次 C0 的 cleanup audit 均 PASS；指令侧完整命令使用冻结 `p0_base_pilot_v1.json`、未传 `--seed` 覆盖，契约固定 seed 42，checkpoint `run_metadata.seed=42`。只有 checkpoint 专用 `training_config` 按设计不含 seed，而旧实现错误地将这个最小恢复子集直接写成完整 resolved training config。
- 处置：不修改旧证据、不把本次模型判为性能 HOLD/GO、不从 update 5 resume；修复后新 RunID fresh 重跑。旧 Run 保留为失败证据，其 registry 状态改为 `EVIDENCE_HOLD`。

## 5. 单 Run 记录模板

### `<RunID>`

#### 预注册输入

| 字段 | 值 |
|---|---|
| 假设 | `<该分段要检验的可证伪命题>` |
| Git SHA / dirty 说明 | `<sha；任何非关键 dirty 路径及为何不影响运行>` |
| contract ID / SHA256 | `<id / sha256>` |
| scenario pool fingerprint | `<sha256>` |
| fresh / resume | `<fresh 或 resume latest；resume checkpoint SHA>` |
| 目标 stop update | `<5/10/...>` |
| 完整命令 | `<PowerShell 命令>` |
| CARLA / Python / Torch / device | `<版本和设备>` |
| 输出目录 | `<logs/RunID>` |

#### 分段结果

| update 区间 | wall time | episodes / transitions | success / collision / timeout | KL / clip fraction | entropy / action boundary | critic / grad clip | validator | C0 | 结论 |
|---|---:|---:|---|---|---|---|---|---|---|
| `<a-b>` | — | — | — | — | — | — | — | — | `<GO/HOLD>` |

#### 产物身份

| 产物 | 路径 | SHA256 / 状态 |
|---|---|---|
| resolved config | `<run>/resolved_config.json` | `<sha>` |
| experiment manifest | `<run>/experiment_manifest.json` | `<status>` |
| latest checkpoint | `<run>/latest.pt` | `<sha>` |
| periodic checkpoint | `<run>/periodic/ckpt_update_N.pt` | `<sha>` |
| metrics / episodes | `<run>/metrics.jsonl`、`episodes.jsonl` | `<validator PASS/FAIL>` |
| C0 summary | `<evaluation path>` | `<protocol/checkpoint/config sha>` |
| C0 repeat summary | `<second evaluation path>` | `<repeatability PASS/FAIL>` |
| GO/HOLD decision | `decisions/base_mappo/<DecisionID>/go_hold_decision.json` | `<contract sha / decision>` |
| 视频/截图（如有） | `<path>` | `<episode/update 对应关系>` |

#### 异常、判断与下一动作

- 异常：`<包括失败和 warning，不写“无”来掩盖未检查>`
- HOLD 规则触发：`<规则、证据位置、首次发生 update>`
- 人工观察：`<只描述可见行为，不替代数值门槛>`
- 结论：`<当前证据支持/不支持什么；不能越界推断什么>`
- 下一动作：`<继续到下一个登记停点 / 修复后新 RunID / 终止该契约>`

## 6. 变更纪律

若需要修改任一科学参数、场景文件、reward/lifecycle、模型结构或训练更新逻辑：

1. 当前 Run 标记 HOLD，不在原 Run 静默续跑。
2. 写清触发证据与修改假设。
3. 提升契约 ID/文件名，重新测试并提交 Git。
4. 使用新 RunID fresh 启动；旧 Run 保留作失败/对照证据。

仅 CARLA 临时断连且 checkpoint、指标、清理、代码与契约身份全部通过时，才允许同 Run 严格 `--resume latest`。
