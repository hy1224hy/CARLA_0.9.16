# P0-00G 真实 CARLA smoke 测试记录

> 记录状态：PASS / FROZEN BY P0-BASE-00 (`345ea04`)（用户确认人工视觉无异常；success 补验通过）
> 开始时间：2026-08-02 16:23:30 +08:00
> 裁决范围：当前未提交修复工作树的端到端工程 smoke；不构成策略性能或正式可复现版本的证明。

## 1. 固定边界

| 项目 | 本次取值 |
|---|---|
| 仓库基线 commit | `f17fd52e3e1c5c12327c139b2899ef27b60160d8` |
| 工作树 | DIRTY；保留既有用户文件 `docs/.obsidian/workspace.json`，本 smoke 不修改该文件 |
| Python | `D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe` |
| CARLA | client/server `0.9.16`，`127.0.0.1:2000` |
| 训练入口 | `training.train_no_rllib_mappo_robust_v2` |
| seed | `42` |
| 失败诊断 run | `P0_00G_robust_v2_smoke_s42_20260802_162851`；保留，不再续跑 |
| 失败数值门槛 run | `P0_00G_robust_v2_smoke_s42_20260802_163613`；保留，不再续跑 |
| 当前主 run | `P0_00G_robust_v2_smoke_s42_20260802_164211` |

工作树非 clean 是本次证据限制。fresh run 启动后，训练相关源码、场景、参数和规范冻结到 resume 完成；若需修复代码，原 run 保留并使用新 RunID 重跑完整流程。

## 2. 测试结果

### P0G-E0 环境与污染基线

- 状态：PASS（带工作树非 clean 限制）
- 时间：2026-08-02 16:23:30–16:24 +08:00
- 预期：2000 由本安装的 CARLA 监听；client/server 版本一致；测试前无车辆或传感器；磁盘不少于 10 GB。
- 观察：监听 PID `27296`，进程路径为 `CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe`；client/server 均为 `0.9.16`；当前地图 `Town10HD_Opt`、异步模式；actor 总数 23，`vehicle.*=0`、`sensor.*=0`；D 盘可用 `36,635,451,392` bytes（约 34.12 GiB）。
- 解释：Town10HD_Opt/异步模式是训练启动前状态；训练入口预计切换至 Town02 并接管同步模式。当前 actor 计数只建立泄漏比较基线，不能证明训练后的清理正确。
- 权限备注：`Get-NetTCPConnection` 被系统拒绝，已用 CARLA RPC 握手及 `netstat -ano` 双重替代；不影响结论。

### P0G-U0 无 CARLA 回归与中断钩子

- 状态：PASS
- 目的：确认新增确定性中断入口仅在 resume 生效，且既有 65 项回归不退化。
- 期望：全量测试通过；非法 fresh/越界配置被拒绝；不改变奖励、网络、rollout 或 checkpoint 冻结参数。
- 观察：新增 3 项配置测试全部通过；全量 68 项测试全部通过，耗时 5.847 秒。非法 fresh 使用、超过 rollout 的 episode 边界均被拒绝；合法 resume 边界返回 1。
- 结论：允许进入 fresh run。测试只覆盖钩子配置和既有算法/协议回归；真实中断路径仍由 P0G-S2 验证。

### P0G-S1a 首个 Run Fresh update 1–3

- 状态：PASS（工程链路；终局覆盖受限）
- 期望：每 update 4 episode；PPO 不跳过；安全 checkpoint、逐 episode 与指标记录连续。
- 命令关键参数：`updates=3`、`lr_decay_horizon=3`、`rollout_episodes=4`、`min_transitions_per_update=64`、`max_steps=300`、`save_interval=1`、`tb_histogram_interval=1`、`seed=42`。
- 启动前复核：输出目录不存在；`git diff --check` 通过（仅换行符警告）；CARLA 0.9.16 可握手，`vehicle.*=0`、`sensor.*=0`。
- 观察：进程正常退出，墙钟 92.4 秒。updates 1–3 的 transition 总数分别为 810、1030、1856，均执行 PPO；12 条 episode 和 3 条 metrics 顺序连续，manifest 为 latest=3、periodic=1,2,3。终局为 `cooperative_collision` 8 次、`max_steps` 4 次、success 0 次；长度 24–300 steps。
- 守恒与健康值：全部 agent 的 `active_decision_steps == transition_count`，不匹配数 0；两个 Actor 的 preupdate ratio error 每次均为 `9.5367431640625e-7 < 1e-6`；Actor/Critic update norm 每次均大于 0；最大逐 step 奖励残差 `2.8421709430404007e-14`；Actor LR 为 `3e-4, 1.5e-4, 1.5e-5`。
- 阶段 validator：expected updates 1–3、每 update 4 episode、强制 PPO 的严格模式返回 PASS；episode 12 条，必需 tag 209，TensorBoard scalar tag 529。
- 清理：退出后 Town02 恢复异步，`vehicle.*=0`、`sensor.*=0`。`fixed_delta_seconds=0.1` 仍保留，但异步模式下不驱动固定 tick；最终环境恢复问题另在结论中区分。
- 判断：允许进入受控中断。collision/max_steps 已满足“两类终局”覆盖，但没有 success，不能据此判断成功判定链路在真实 CARLA 中正确。

### P0G-S2a 首个 Run Update 4 受控中断

- 状态：FAIL（钩子尚未开始，resume 前置校验失败）
- 方法：resume 后写完 update 4 的第 1 个 episode，显式触发训练器现有 `KeyboardInterrupt` 清理路径；不关闭 CARLA、不强杀 Python。
- 期望：`latest=3`、`interrupted=3`；存在一个领先安全边界的 update 4 episode 记录，供下一次 resume 自动裁剪。
- 观察：命令在连接 CARLA 前以 `CheckpointProtocolError: curriculum stages differ from checkpoint` 退出；manifest 仍 latest=3、无 interrupted，episodes/metrics 仍为 12/3，CARLA actor 为 0/0，未污染原有安全数据。
- 根因：classic 课程第二阶段由 `list(set(stage2))` 生成；不同 Python 进程的 hash 随机化会改变同一场景集合的列表顺序，而 strict resume 按列表顺序比较，因此逻辑相同的 fresh/resume 被误判漂移。
- 最小修复：用保序去重 `list(dict.fromkeys(stage2))`，新增精确顺序回归；不放宽 checkpoint 校验。因为训练源码 fingerprint 将变化，本 Run 保留为失败证据，不再续跑。

### P0G-R0 针对性修复与回归

- 状态：PASS
- 观察：课程 stage2 改为保序去重；环境记录 Town02 进入时的 `synchronous_mode/fixed_delta_seconds`，退出时精确恢复，并关闭本环境启用的 Traffic Manager 同步。新增 2 项回归，全量 70 项测试通过，耗时 6.031 秒；`git diff --check` 通过（仅换行符警告）。
- CARLA 状态回滚：首个 Run 遗留状态由异步/0.1 恢复为已登记的异步/None，Traffic Manager 同步关闭；车辆/传感器保持 0/0，未重启或停止 CARLA。
- 判断：修复改变训练源码 fingerprint，必须使用新 RunID 从 fresh 重跑；新输出目录在启动前不存在。

### P0G-S1b 修复后 Fresh update 1–3

- 状态：FAIL（update 1 PPO 前置检查停止）
- 运行参数与 P0G-S1a 完全相同；训练源码从此冻结至第二个 Run resume 完成。
- 观察：4 个 rollout episode 已完成，但 optimizer 尚未执行；actor 0 的 `max_abs(ratio-1)=1.43051147e-6`，略高于冻结的 `1e-6`，训练器 fail-fast。manifest latest/interrupted 均为初始化安全边界 update 0，episodes=4、metrics=0。退出后异步/None、车辆/传感器 0/0，证明 I-006 的真实清理路径已生效。
- 判断：该 Run 没有完成 update 1，不能用于任何 PPO 或 resume 结论；原样保留，不放宽门槛后续跑。

### P0G-S2b 修复后 Update 4 受控中断

- 状态：NOT REACHED
- 期望：跨进程课程校验通过；写完 update 4 episode 0 后走 `KeyboardInterrupt` 清理，latest/interrupted 均保持 update 3。
- 观察：待记录。

### P0G-R1 PPO 前置 ratio 数值一致性修复

- 状态：PASS
- 根因证据：rollout 对每个 active agent 使用 batch=1；precheck 合并全部 transition 后再前向。相同参数下，float32 不同 GEMM 几何可产生 ulp 级差异。独立 4000 样本复现：合批重算最大 ratio 误差 `9.5367431640625e-7`，逐 transition 重算为 0；首个/第二个真实 Run 分别得到 `9.536743e-7` 和 `1.430511e-6`，说明 1e-6 门槛受数值几何支配。
- 修复：只让 fail-fast precheck 按 rollout 的 batch=1 几何逐 transition 重算；PPO minibatch、旧 log-prob、奖励、动作和 optimizer 均不改，门槛仍为 `1e-6`，参数被修改后的 stale-policy 测试仍必须失败。
- 验收：transitionwise 未更新策略误差严格为 0；原参数扰动 stale-policy 负例仍被拒绝；全量 71 项测试通过，耗时 6.043 秒。

### P0G-S1c 第二次修复后 Fresh update 1–3

- 状态：PASS（RunID 于 2026-08-02 16:42:11 +08:00 冻结）
- 运行参数仍与 P0G-S1a 完全相同；输出目录启动前不存在，`git diff --check` 通过（仅换行符警告），CARLA 为 0.9.16、异步/None、车辆/传感器 0/0。
- 观察：正常退出，墙钟 99.2 秒；updates 1–3 的 transition 总数为 1004、1504、1304，均执行 PPO；12 条 episode 为 `cooperative_collision` 9 次、`max_steps` 3 次，success 0。active/transition 不匹配数为 0；preupdate ratio error 在两个 Actor、三个 update 上均严格为 0。
- 阶段验收：strict validator PASS（12 episodes、required tags 209、TB scalar tags 529）；manifest latest=3、periodic=1,2,3。退出后异步/None、车辆/传感器 0/0。
- 重复性备注：与首个相同 seed Run 的轨迹统计不逐位一致；合理推断为持续 CARLA 服务内部物理/调度状态未完整序列化，不能据软件 seed 宣称跨 Run 轨迹确定性。该差异不用于筛选结果。

### P0G-S2c 第二次修复后 Update 4 受控中断

- 状态：PASS
- 期望：课程跨进程校验通过；update 4 episode 0 落盘后中断，latest/interrupted=3，领先 episode 在后续 resume 裁剪。
- 观察：跨进程课程校验通过，恢复信息为 next_update=4、total_ep=12、lr_horizon=3；update 4 episode 0 结束后按钩子进入原 `KeyboardInterrupt` 分支，墙钟 24.5 秒。
- 中断不变量：manifest latest=3、interrupted=3、periodic=1,2,3；metrics 仍 3 条；episodes 暂为 13 条，其中唯一领先记录是 update 4/index 0。`latest.pt` 与 `interrupted.pt` SHA256 同为 `1701802a09ed432a26face3f1bcef7d3e39f3480b47dd6565d00494bf3b6173f`，两侧 sidecar 均一致。
- 清理：异步/None、车辆/传感器 0/0。结论：最近完整 update 边界和真实清理路径成立；下一步必须证明领先 episode 会自动裁剪，而不是被重复计入。

### P0G-S3 严格恢复并完成 update 4

- 状态：PASS
- 期望：领先 episode 被裁剪后重采 update 4；最终 metrics 为 updates 1–4、episodes 恰为 16 条。
- 观察：不带中断钩子的 resume 从 next_update=4、total_ep=12、lr_horizon=3 启动，33.0 秒后完成 update 4；4 个 episode、558 transitions，PPO 未跳过，Actor LR 为 `1.5e-5`，生成 periodic 4 与 latest 4。
- 裁剪证据：中断态的 13 条 episodes 最终变为 16 条，而非 17 条；updates 1–4 各 4 条、index 均为 0–3，证明领先 update 4/index0 被丢弃后重采。metrics 恰为 updates 1–4。
- 清理：异步/None、车辆/传感器 0/0。

### P0G-V1 指标与 checkpoint 验收

- 状态：PASS
- 期望：严格 validator PASS；manifest/sidecar/SHA256 一致；latest=4、interrupted=3、periodic=1–4；Actor/Critic 确实更新且数值有限。
- validator：严格命令返回 `status=PASS`，updates 1–4，episode records 16，required tags 209，TensorBoard scalar tags 529。
- 生命周期与 PPO：13 次 `cooperative_collision`、3 次 `max_steps`；active/transition 不匹配 0；四次 update 均未跳过，preupdate ratio 全为 0，所有 Actor/Critic update norm 大于 0；最大逐 step 奖励残差 `2.8421709430404007e-14`。
- checkpoint：6 个记录全部通过 sidecar、manifest digest 和 header 校验；latest=4、interrupted=3、periodic=1,2,3,4。latest payload 含 Actor 0/1、各自 optimizer、`AgentConditionedCritic`/optimizer、课程、LR horizon=3、RNG、协议版本及 last_metrics=4。
- 来源：origin/current code fingerprint 均为 `cf6e9004aafeb21b26f74c75955846d21b2e51eeb8e945f5c6bd26e621474492`，`code_drift_override=false`；Git SHA 为 `f17fd52e3e1c5c12327c139b2899ef27b60160d8` 且工作树 dirty。

### P0G-A1 CARLA actor 清理与视觉证据

- 状态：PASS / USER CONFIRMED
- 期望：训练退出后 `vehicle.*` 和 `sensor.*` 恢复到基线 0；至少 3 个 episode 有可复核视觉观察，且终局至少覆盖两类。
- 自动观察：fresh、中断、最终 resume 三次退出后均为 Town02 异步、`fixed_delta_seconds=None`、车辆/传感器 0/0；训练过程中观察到同步模式与 2 辆车/6 个传感器，说明清理比较有真实创建基线。终局已覆盖 collision 与 max_steps 两类。
- 用户证据：2026-08-02 用户确认人工视觉检查未发现问题，据此关闭出生重叠、瞬移、方向错误、异常冻结和 episode 重置的操作门槛。
- 证据限制：未提供 episode 编号或录像，无法由第三方复核；第三个训练 Run 本身没有 success，因此它单独不能证明成功终局。下述策略无关补验随后关闭逻辑覆盖缺口，但不替代训练后 C0 性能评估。

### P0G-SUCCESS 策略无关的真实 success 补验

- 状态：PASS
- 前置环境：2026-08-02 22:24:43，用户重启的本安装 CARLA 监听 2000；client/server 均为 0.9.16；Town10HD_Opt、异步/None、车辆/传感器 0/0。
- 纯逻辑 Run：`P0_00G_SUCCESS_teleport_s42_20260802_222443`。两车在 step 1/2 分别获得 `success=100`、`success_event=1`，均 terminated 且不 truncated；第一车到点时第二车仍 active、全局仍 running，第二车到点后严格为 `all_agents_reached`；collision 0。证据：[result.json](../logs/P0_00G_SUCCESS_teleport_s42_20260802_222443/result.json)、[steps.jsonl](../logs/P0_00G_SUCCESS_teleport_s42_20260802_222443/steps.jsonl)。
- 近终点 Run：`P0_00G_SUCCESS_scripted_s42_20260802_222523`。两车从 `y=256/246` 以 throttle 0.30 直行，在 step 32/55 分别 success；step 31/54 的 `y=260.9079/260.9331` 均仍在终点近边界 `y=261` 外，下一步进入后正确触发。先到车辆退出后，后车继续 active 且无物理阻塞；最终 `all_agents_reached`、collision 0。证据：[result.json](../logs/P0_00G_SUCCESS_scripted_s42_20260802_222523/result.json)、[steps.jsonl](../logs/P0_00G_SUCCESS_scripted_s42_20260802_222523/steps.jsonl)。
- 连续运行与清理：两次 Run 使用同一 CARLA 服务且均成功；每次退出恢复异步/`fixed_delta=None`、车辆/传感器 0/0。当前证据不支持“只有第一次运行正确”或“第二次服务器卡死”。这不能追溯证明历史某次运行没有卡顿。
- 范围限制：补验确认 goal zone、success 奖励、生命周期、先到车移除和短距离直行链路；不证明 MAPPO 收敛、远距离成功率、跨 seed 鲁棒性或 checkpoint 质量。完整分析见 [success 规范与记录](./P0-00G-SUCCESS_真实success逻辑测试规范与记录.md)。

## 3. 问题登记

| ID | 问题 | 影响 | 最小验证/修复 | 状态 |
|---|---|---|---|---|
| I-001 | smoke 执行时工作树非 clean，尚未冻结到新 commit | 当次通过只能归属于记录的 dirty source fingerprint，不能直接宣称任意 commit 可复现 | 已审阅差异并将 success 工具、测试和证据冻结到 `345ea04`；93 项回归、场景 validator、入口与历史 strict validator 复核通过 | CLOSED BY P0-BASE-00 |
| I-002 | 工具执行无法可靠向后台 Windows 训练进程发送一次 Ctrl+C | 若用强杀模拟，会绕过清理逻辑并使 interrupted 证据无效 | 增加仅 resume 可用的确定性 episode 边界中断钩子，走原有 `KeyboardInterrupt` 分支；配置和真实路径均验证 | CLOSED |
| I-003 | 当前无法由自动化终端替代人工视觉观察 | 自动指标通过仍不能排除明显出生重叠、瞬移或错误驾驶方向 | 用户已确认人工视觉无异常；未提供编号/录像，保留证据强度限制 | CLOSED/USER CONFIRMED |
| I-004 | 当前主 Run 的 16 个已提交 episode 均未 success，且控制台聚合的 progress 为 0 | smoke 已覆盖 collision/max_steps，但没有真实 success 路径证据；progress=0 若被误读会造成指标失真结论 | 已确认 `reward_status/progress/enabled=0` 且 raw `diagnostic/progress_m` 非零；主 Run 不改，另以 teleport/scripted 两层真实补验关闭 success 覆盖缺口 | CLOSED BY SUPPLEMENT |
| I-005 | classic 课程 stage2 使用无序 `set`，跨进程重建顺序不稳定 | strict resume 对同一逻辑课程误报漂移，P0G-S2 无法开始，恢复性未成立 | 保序去重并新增顺序回归；新 RunID 重跑 fresh→interrupt→resume | CLOSED（真实跨进程 resume 通过） |
| I-006 | 环境退出只恢复异步，`fixed_delta_seconds` 留为 0.1，Traffic Manager 同步未关闭 | CARLA 全局状态污染后续客户端，actor=0 仍不能视为完整清理 | 记录并恢复原始 world timing，关闭本环境启用的 TM sync；mock 与 fresh/interruption/resume 真实路径验证 | CLOSED |
| I-007 | precheck 合批前向与 rollout batch=1 前向存在 float32 ulp 差异 | 固定 1e-6 门槛会随机接受 9.54e-7、拒绝 1.43e-6，造成相同未更新策略的假阳性 | precheck 匹配 rollout 的逐 transition 几何，保持 1e-6 和 stale-policy 负例；真实四 update 均为 0 | CLOSED |
| I-008 | 主 Run 的 16 个 episode 没有 success，无法把策略能力与终点逻辑分开 | 直接先训练或延长轮次仍无法单独证明 success reward/lifecycle 正确 | `teleport` 与近终点 `scripted` 连续真实运行均无污染 PASS；逐步 JSONL、结果和清理证据已保存 | CLOSED |

## 4. 当前裁决

### 已确认事实

- 主 smoke 当时的 71 项无 CARLA 回归通过；新增 success 测试后当前全量为 93 项通过；当前主 Run 的 fresh→controlled interrupt→resume→strict validator 全部通过。
- 16 条已提交 episode、四次 PPO、TensorBoard/JSONL、checkpoint/manifest/checksum 和 actor/world 清理满足自动门槛。
- 当前主 Run 只出现 collision 与 max_steps，没有 success；没有可复核视觉记录；工作树为 dirty。
- 后续策略无关补验中，teleport 两车在 step 1/2、scripted 两车在 step 32/55 触发 success；success/lifecycle/终局/清理全部满足冻结门槛。

### 合理推断

- 基础 MAPPO 的真实 CARLA 工程数据流已具备进入下一项验证的条件。
- 同 seed 的不同 Run 轨迹统计不同，较可能来自 CARLA 服务内部物理/调度状态未完整冻结；这不是“随机种子无效”或“恢复错误”的直接证明。

### 尚未验证与缺失信息

- 尚未验证：策略性能/收敛、远距离场景成功率、跨 seed 鲁棒性和精确碰撞对象。用户已确认人工视觉未见出生重叠/瞬移/方向错误，但没有 episode 编号或录像可供第三方复核。
- 缺失信息：至少 3 个训练 episode 的编号/录像记录；当前修复树对应的正式 commit。

### Judge 裁决

**PASS / FROZEN BY P0-BASE-00 (`345ea04`)**。自动工程 smoke、用户人工视觉以及策略无关的 teleport/scripted success 两层补验均通过，基础 MAPPO 的真实 CARLA 数据流和固定场景成功链路门槛闭合。P0-BASE-00 已冻结可训练源码基线；继续完成 P0-BASE-01/02 后才允许单 seed 分段试训，P0-BASE-04 GO 前仍禁止多 seed 批量训练。策略性能、远距离成功率和独立 C0 尚未验证。
