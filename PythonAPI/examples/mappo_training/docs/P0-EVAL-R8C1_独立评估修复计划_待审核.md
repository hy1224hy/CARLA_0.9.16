# P0-EVAL-R8C1：独立评估修复计划（已批准实施）

> 2026-09-15 实施结果：A–G的代码、离线测试及42回合真实执行已完成，证据完整性PASS，但重复性验收HOLD。最新执行证据和待解决问题见 [真实评估与指标诊断](./P0-EVAL-R8C1_20260915_真实评估与指标诊断.md)。不重复运行本计划，不将已执行等同全部门槛通过。

原计划日期：2026-09-07。用户批准：2026-09-11。实施续接：2026-09-14。状态：`APPROVED / IMPLEMENTED / OFFLINE PASS / AWAITING USER-STARTED CARLA`。

用户已明确回复“请按照这份计划进行代码修复，如果需要carla环境，请让我启动”。本文文件名保留以兼容原链接；下文方案阶段措辞记录原计划，实施进度以本段及主 TODO 为准。已完成 337/337 离线回归和真实 u20/u60/u100 输入预检，尚未连接 CARLA。下一步由用户启动服务，先 2 smoke 再 40 诊断。详见 [实现与离线验收](./P0-EVAL-R8C1_实现与离线验收.md)。

原计划背景：用户要求先更新问题与进度，并在实现评估程序前审核方案。本轮已更新 [研究进度 TODO](./毕业课题_研究推进TODO.md) 与 [基础策略实验记录](./P0-BASE_基础策略实验记录.md)，没有修改评估/训练代码、JSON 合同或 checkpoint，也没有运行新评估。本文的文件名、协议和命令均为拟实施设计，不能当作已经可运行的功能。

**推荐方案是扩展现有 C0 评估链路，首版只支持 R8-C1 的确定性诊断。** 保留现有逐 active-agent 执行、证据落盘、独立 validator、重复执行与清理机制；新增独立 R8-C1 协议和配置。先回答“当前学到的策略在无探索噪声时能否驾驶、失败发生在哪些场景/agent”，本轮不追认基础能力 GO，也不启动选模、续训或对抗训练。

## 1. 当前问题及对应修复

| 编号 | 已确认问题 | 影响 | 拟修复方法 |
|---|---|---|---|
| E1 | `base_c0_protocol.py:426` 只注册到 R7；checkpoint 检查绑定旧 schema、reward、curriculum 和 protocol map | R8-C1 schema8/rewardv6/R8课程+采样会被拒绝；诊断开关不能解决结构不兼容 | 新注册 R8-C1 诊断协议，严格识别原始 schema8/header/model/protocol/合同身份，不冒充 R7、不跳过校验 |
| E2 | `_validate_scenario` 可选字段只有 goal_mode/agent_goal_zones，未允许 R8 场景的 `curriculum_stage` 与 `episode_max_steps` | 原始 R8 短场景会被入口直接拒绝 | 只对新协议扩展字段，按 R8 场景清单及逐文件 SHA 校验；验证 horizon 为正整数且不超过合同上限 |
| E3 | runner loop 及部分校验使用统一 max_steps；R8 场景实际分别是 200/500 tick | 删除 override 或统一500会改变任务期限、剩余时间观测和 pace reward 语义 | 展开 case 时固定 `effective_episode_max_steps`；apply/reset后断言 env.max_steps；循环、record、validator统一使用该值 |
| E4 | 五结局及道路证据已有部分实现，但重复性 hard signature 未完整包含道路违规字段 | 团队同为 road_violation 时，违规 agent/原因变化可能被遗漏 | 新版本明确比较逐agent道路确认/终止、原因、分类、参考车道、成功资格与首证据存在性；保留完整证据和浮点/计时容差分组 |
| E5 | 现 CLI、EvalID、contract、runner及validator多处仅允许 smoke/gate/validation；diagnostic限制为smoke | 只加 schema8后仍不能做完整短/长诊断 | 仅新协议允许 smoke/short_probe/long_probe，所有环节一致解析；历史协议保持原集合及限制 |
| E6 | 输出主要依赖 checkpoint formal_eligible 决定PASS；R8-C1原训练合同的评估/能力门槛仍reserved | checkpoint身份匹配可能被误解释为正式能力验收 | 把输入身份有效、评估完整性、研究用途、能力裁决拆开；本合同始终 diagnostic_only，能力裁决 NOT_EVALUATED，禁止自动选模 |
| E7 | 评估修复后的Git提交与原训练不同；仅比较整体Git或跳过所有source校验都不合适 | 可能错误拒绝合法评估更新，或漏掉共享推理/环境行为变化 | 分开记录训练provenance和评估provenance；重核共享Actor/action/env/reward/lifecycle/road-offroad文件身份；共享语义漂移fail-fast |

E3是需要统一的实现风险，不意味着现有所有case已经越过200tick：环境本身支持场景override并可能在200步正常终止。但依赖全局上限与隐式环境行为的实现无法充分证明整个证据链遵守场景期限，必须在新协议显式连接并测试。

## 2. 首版范围与输入身份

使用一个已完成的训练 Run：`P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224`。

| 输入项 | 拟固定值 |
|---|---|
| 训练来源 | Git `050fe9fc72b9639b3048f56384cc48ad0d8d0214` |
| 训练合同 | `P0_BASE_MAPPO_TOWN05_R8C1_PACE_V1` |
| 合同 SHA256 | `88bdb9dcac5b238d371d5718d346dc7e43b287b8cb725292ddb25b8862d053f4` |
| checkpoint | 原 Run 下 `periodic/ckpt_update_000020.pt`、`...000060.pt`、`...000100.pt`；u100为主结果，u20/u60为时间参照 |
| checkpoint身份 | schema8、`mappo_robust_v2_r8c1_pace_p0_t05_v8`、两个Actor、obs20/action2、central state40、hidden128；原始SHA与manifest逐项核验 |
| 环境 | Town05、CARLA0.9.16、两红车、无新增交通、timestep0.1、rewardv6、gamma0.999、LiDAR/TTC20m |
| 执行动作 | `tanh(mu)`，共用现有signed-longitudinal2维→CARLA3维适配；Actor eval/no_grad，权重执行前后哈希保持一致 |
| 场景 | 原始 `data/training_scenarios_town05_r8a_v1/` 01–12，保留原坐标、初速度、tags、目标和期限，不为评估修改场景 |
| 初始速度与时限 | 01–02短rolling为2m/s；03–04短stationary为0；01–04均200tick；05–12均500tick，使用各自JSON初速度 |
| 数值边界 | 成功、碰撞、offroad、road_violation、timeout及对应T/X沿用现环境定义；不在评估器改写终局或奖励 |

首版不支持把 R8-A/R7 失败 checkpoint 自动迁移成 R8-C1 身份；不扩展 stochastic、IPPO/BC通用适配、C1/C2、PSRO或GO/HOLD judge。旧R6/R7评估继续按旧合同运行。未来需要通用adapter时复用本次清晰的加载接口，本次不复制或重写完整训练/评估框架。

## 3. 拟修改文件与实现方式

| 文件 | 工作内容 |
|---|---|
| `evaluation/base_c0_protocol.py` | 接入新协议；新profile/ID规则；schema8、场景字段与有效时限；episode/resolved/summary新schema；五结局及道路重复性字段 |
| `evaluation/evaluate_base_policy_c0.py` | 复用原runner；新profile路由；source记录；按case horizon执行；固定诊断用途；checkpoint/Actor不变性及退出清理 |
| `evaluation/validate_base_c0_run.py` | 支持新schema/profile；独立重新读取输入和JSONL；重算哈希/汇总；核验horizon、来源、道路证据、resume一致性 |
| 拟新增 `evaluation/r8c1_diagnostic_protocol.py` | 放置R8-C1特定规格、用途/兼容性辅助检查，控制旧协议文件中的版本分支；不另写rollout/PPO或重复完整validator |
| 拟新增 `configs/evaluation/p0_base_c0_town05_r8c1_probe_v1.json` | 新诊断合同；固定Run/contract/checkpoint与场景身份、3个profile、seed、重复性、输出用途；通过schema再冻结 |
| 拟新增 `tests/test_base_c0_r8c1.py` | 真实输入离线加载、fake-env执行、horizon、negative tamper、五结局/道路repeatability、恢复和旧协议隔离 |
| 进度/实验/本计划文档 | 实施后逐项记录代码、测试和结果；本轮仅状态更新到待审核 |

预计不需要改 `training/checkpoint_protocol.py`：其现有header校验已经支持schema8。也不修改环境、模型、rewardv6、训练器、训练合同、历史评估JSON、旧裁决和原Run文件。若实现中发现必须修改共享行为文件，先提交新增问题与方案再让用户审核，不能扩大本次批准范围。

新协议拟命名 `p0_base_c0_diag_town05_r8c1_v1`；合同ID拟为 `P0_BASE_C0_TOWN05_R8C1_PROBE_V1`。这是新评估前冻结的事后诊断协议，不是训练前预注册；不得修改原训练合同的reserved字段来伪装已经预注册。

## 4. 输入来源、诊断资格和历史兼容

1. checkpoint文件、sidecar、manifest记录的SHA及update必须一致；来源Run、training contract id/hash、model spec、protocol map、actual reward gamma、R8课程/采样与critic probe证据必须通过。新合同登记20/60/100的精确文件SHA；不能按可变 `latest` 选择输入。
2. 原训练Git/源码fingerprint与本次评估Git/逐文件源码SHA分别写入resolved/manifest。评估程序改变提交是预期行为，不重写checkpoint metadata，也不使用训练resume的code-drift参数。
3. 审计共享推理语义文件。仅评估侧改动应不改变原训练关键源码身份；若共享文件哈希不符，列出差异并拒绝默认兼容，而不是用diagnostic开关忽略。
4. 新记录明确 `evaluation_purpose=diagnostic_only`、`selection_allowed=false`、`performance_decision=NOT_EVALUATED`。具体字段由新schema严格定义。合同/来源匹配是输入完整性的事实，不授予正式选模资格；程序完整性和重复性结果独立保存。
5. CLI可增加新profile选项，但旧合同遇到short_probe/long_probe仍拒绝。`--allow-diagnostic-checkpoint` 的旧smoke规则保留；R8-C1的诊断用途由新合同显式定义，不靠通用开关绕过结构或来源检查。
6. 新EvalID/profile必须与checkpoint update和输出目录一致；禁止覆盖已有结果。恢复评估只接续完整episode前缀，并核验同一checkpoint SHA、合同/场景SHA、case顺序、evaluator源码fingerprint和环境规格。改变实现或输入后使用新EvalID，不把两种实现写进同一Run。
7. validator始终重新读取输入并重算结果，不信任旧 `validator_report.json` 的PASS，不引入路径级结果缓存。新字段只对新schema必填，旧R6/R7记录保持原行为。

## 5. 首轮诊断矩阵：40回合，加2回合入口smoke

u100是已完成计划的终点，固定为主结果；u20和u60检查行为演变，不从三个checkpoint中挑“最好”的策略。全部使用确定性动作。主评估seed建议61001，独立smoke使用61002，批准后写入新合同；主/重复使用完全相同的seed、case顺序和输入SHA。

| 组别 | checkpoint | 场景 | 主执行 | 独立重复 | 总回合 |
|---|---|---|---:|---:|---:|
| 生产入口smoke | u100 | 01北向短rolling | 1 | 1 | 2 |
| 已训练分布：短rolling | u20/u60/u100 | 01–02北/南镜像 | 6 | 6 | 12 |
| 起步迁移：短stationary | u20/u60/u100 | 03–04北/南镜像 | 6 | 6 | 12 |
| 下一课程：长任务probe | 仅u100 | 05–12共8场景 | 8 | 8 | 16 |
| Road43/6 validation | 首轮不执行 | 原选模验证集 | 0 | 0 | 0 |

实现可用short_probe一次包含4短case，每个checkpoint各主/重复一次，但summary必须拆分rolling与stationary。40回合是主诊断的执行次数，2回合smoke单独记录，不并入能力汇总，也不是42个独立统计样本。

- R8-C1实际只采样过S0-A，01–02才是已训练分布；03–04虽然也在场景目录中，但未进入实际采样支持集，属于迁移探针。
- 长05–08同时引入四方向及异速等变化；09–12是长stationary，且不具free_flow+non_interactive双标签，pace项按原门控自然不生效。长短差异不能仅归因于路线长度；按这两个家族分别报告。
- 一个固定seed加相同配置重复用于稳定性审计，不宣称具有多训练seed统计效力。本轮不输出具有误导性的“跨seed泛化置信区间”。
- 01–04最多200tick、05–12最多500tick，42次总上界13200tick，即22分钟模拟时间，加上重置/连接开销；这不是墙钟耗时承诺，提前终止会降低实际步数。
- 首轮不实现stochastic模式。若确定性和既有训练行为差异显著，后续另行审核u100短rolling的动作采样诊断；其结果不能替换确定性主结论。

## 6. 输出与重复性规则

沿用现有output根目录与独立EvalID；每次执行保存manifest、resolved、episodes.jsonl、summary.json/CSV、validator报告，主/重复各自输出，比较结果独立落盘。新record应包含：

- 输入：Run/update/checkpoint SHA、训练合同SHA、评估合同/源码fingerprint、scenario SHA、episode seed、profile和case ordinal。
- 时序：effective horizon、实际episode长度、逐agent active decision steps、任务边界T/X和生命周期证据；记录reset后实际max_steps。
- 行为：五互斥结局、逐agent reached/完成度/速度/实际控制统计、道路合法性与首个安全失败证据；新增指标有明确来源，不以全局平均掩盖方向/agent差异。
- 奖励：v6全部分量和pace诊断字段、逐步/整回合守恒；训练奖励只作解释，不以不同奖励版本的return直接排名。
- 汇总：按checkpoint、场景、方向、agent、rolling/stationary/长任务家族分层；可保留全量计数用于完整性，但不以混合场景成功率代替主结论。

重复性按新版本固定：

1. 同一checkpoint、profile和case之间比较；五结局与逐agent终止/确认/成功资格、道路分类/原因/参考车道、首证据存在性和生命周期严格一致。记录完整road evidence，但不对整个含浮点字典直接做逐位相等。
2. episode length与active steps继续采用既有最多2tick容差；return继续沿用现有pass/warning/hold数值容差，不为本轮表现更改。道路方向cosine等连续量和debounce计数保留审计，不未经定义便纳入硬相等字段。
3. 当前系统性漂移规则要求至少8cases：每个short_probe只有4cases，该项明确输出 `NOT_APPLICABLE_INSUFFICIENT_CASES`；逐case硬失败/超容差仍正常失败。long_probe8cases按原系统漂移规则执行。不能将不同checkpoint或重复执行拼成独立样本凑门槛。
4. 42回合中即便全失败，只要证据完整也可判评估程序完整性通过；不得因此修改成功、安全或课程门槛。任何能力结论只描述该checkpoint和该测试范围，正式GO/HOLD另行设计审核。

## 7. 实施顺序与验收

| 步骤 | 工作 | 必须验证的内容 |
|---|---|---|
| A 用户审核 | 确认本计划的范围、矩阵与诊断用途 | 2026-09-11 已批准；B–E 于 2026-09-14 离线通过，F–G 等待用户启动 CARLA |
| B 协议与输入 | 新合同/spec接入，离线读取真实u20/u60/u100 | 文件SHA、model/protocol/训练身份均一致；42次矩阵可完全展开；不连接CARLA |
| C 执行与记录 | 原runner接入R8-C1、每case期限、record与source | fake-env验证200→500→200无串场；timeout恰在有效horizon；inactive agent不动作；五结局/T-X和v6字段一致 |
| D 独立校验 | 新validator/summary/repeatability/resume | 从JSONL重建汇总；篡改horizon、road确认位、pace分量、来源/模型/协议/SHA必定拒绝；重复目录/跨实现resume拒绝 |
| E 离线回归 | 新专项 + 既有315项回归/旧R6R7fixture | 旧协议、schema、profile限制不变；新测试覆盖关键行为，不以固定测试数量替代有效性 |
| F CARLA入口 | 2个独立smoke | 对接用户维护的127.0.0.1:2000；输入/初始化/行为/record/重复性/清理均通过后才执行矩阵 |
| G 诊断矩阵 | 24短+16长，串行连接同一world | 每一主/重复输出独立校验；分层分析；写回进度和实验记录，不自动训练/选模 |

CARLA由用户维护。连接失败时记录阻塞并通知，不自动重启服务；运行前确认无其他训练/评估客户端占用同步world。评估正常结束或异常退出均检查自己创建的车辆/传感器清理与原同步设置恢复；既有其他actor不删除，若无法确认空闲不开始诊断。

测试与真实执行使用根目录venv、从mappo_training目录以模块方式运行。实施后的预期入口形态如下，当前命令不可直接执行：

```text
python -m evaluation.evaluate_base_policy_c0
  --eval-contract configs/evaluation/p0_base_c0_town05_r8c1_probe_v1.json
  --checkpoint logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224/periodic/ckpt_update_000100.pt
  --profile short_probe
  --eval-id <按新协议生成的唯一ID>
  --output-dir <独立的新输出目录>
  --host 127.0.0.1 --port 2000
```

## 8. 本次请用户审核的决策

- 是否采用“扩展现有C0 + 新R8-C1诊断协议”的首版范围，先不实现R8-A批量兼容、stochastic、C012和通用算法框架。
- 是否采用u20/u60/u100的24短回合、u100的16长回合，以及先行2回合smoke；u100固定为主结果，Road43/6首轮不运行。
- 是否接受所有输出均为事后诊断，程序PASS不自动触发基础策略GO、选模、入池或续训；能力门槛保持原状，后续另行审核。

批准后的状态变化应明确记录日期、用户确认内容和计划版本，再进入B步骤；不能把用户要求撰写方案解释为已批准实现。

## 9. 实施状态（2026-09-14）

已完成本计划 B–E：337 项测试通过、3 个真实 checkpoint 预检通过；实施中新发现的版本降级、终止语义、矩阵约束及构造失败清理缺口已修复。证据见 [实现与离线验收](./P0-EVAL-R8C1_实现与离线验收.md)。本页第 8 节为原审核事项，用户已批准，无需重复审核。尚未执行 F–G，也没有修改训练/环境/模型/奖励/GO-HOLD 或启动 CARLA。
