# 基础红策略 InstantPaceV2 实验记录

日期：2026-10-13，Asia/Shanghai。用户已启动CARLA并授权按评估结果修订、验证奖励与设定监控。沿用当前codex/mappo-experiment-baseline，不提交/推送、不切分支。当前阶段：u456独立行为筛选完成；新版代码与离线检查完成；首个完整update已派发，必须以新安全断点确认运行成功。

## 1. 冻结u456真实筛选

目录：diagnostics/reward_instant_v2_20261013/baseline_u456_s62001_v1。源u456 SHA为8b9cb173cd70aacc201f0cb68183fdfeeb593cc8f29863c096a5f59076954545。12个既有R8-A案例、seed62001、确定性Actor；manifest COMPLETED，trace SHA/冻结Actor/active生命周期一致。最终分析见behavior_analysis_v2.json。

- 12/12两车均到达，未记录碰撞/越界/道路违规/超时；这是已知场景单seed工程筛选，不是总体驾驶资格或留出泛化结论。
- active步加权均速agent0/1为26.47/12.16km/h，峰48.70/28.14km/h；agent0超30占43.31% active步，累计23.3秒。
- 平均到达时间4.48/9.38秒；南/东长静止案例agent1首次运动3.9/4.4秒，agent0到达时agent1进度仅0.080/0.006。记录事实，不推断“故意等待队友”。
- 旧pace累计0/−3.749，旧overspeed累计−1.717/−0.636。主要问题是快车超速与部分慢车起步迟缓，不能仅凭12例成功宣称驾驶合格。
- 评估结束正常移除全部车辆/传感器并恢复原异步世界；画面空场是阶段清理，不是任务停止。评估、代码修订和训练串行，2000只有一个时钟所有者。

## 2. 已实现的新奖励

profile=instant_pace_v2；environment/instant_pace_reward.py，保留frozen_v6和route_pace_v1原分支。v6中仅替换free_flow_pace_deficit与overspeed，其他进度/安全/道路合法/终局项保持。

- 即时前进速度缺口：参考2.5m/s；最大低速成本0.5/秒，按dt积分；当前前进速度恢复就归零，没有累计路线欠账。用目标方向投影速度，不用横向速度刷效率。
- 起步2秒、前车受阻、复杂交互、终点5米和终局豁免效率项。仍使用原12例畅通白名单和既有前车判定，不声称已覆盖所有静态障碍/合理让行。
- 二次超速成本：15km/h起算，以15km/h为尺度，0.2/秒系数；30km/h约−0.02/0.1s步，50km/h约−0.109/步，持续增长，无30km/h封顶。任务偏好阈值不等于道路法定限速。
- 20/2观测动作、网络、PPO、场景分布保持，首轮只筛奖励方向。隐藏累计时钟移除，不因此宣称所有奖励门控完全可观测。
- 元数据/schema记录m/s速度缺口及无[0,1]封顶的超速risk；新实现/派生工具进入源码指纹。反事实重算旧轨迹的成本不算新策略性能。

## 3. 新实验与恢复

实际run：logs/R8C1_InstantPaceV2_u456_to476_s42_20261013_v2。

源为固定u456，S2、12场景、16回合/update；计划20个新updates，绝对边界u476。先u457完整保存后退出，再同run严格恢复至u476。不可变seed.pt初始SHA为aada1aadbae2e054579cccf2b40813f326f14eba856e7452394e695feb7ff85d；manifest最后发布。

完整继承Actor/Critic/optimizer/ValueNorm/RNG/课程/采样/LR与entropy相位。旧Critic/probe还对应旧奖励，必须通过新rollout适应，不称重置或已适配。新实验统计只从u457开始；历史累计计数不当新奖励结果。

准备阶段发现安全STOP_REQUESTED尚需接线，已在启动前补齐。因此先前_v1目录只PREPARED，未训练，保留为未使用准备记录；当前只使用_v2，不覆盖原始种子和u456。恢复拒绝来源、源码指纹、场景、配置或profile漂移；改奖励必须新run。

23项必要离线检查通过：6纯函数、8mock集成、6派生身份/状态与拒绝路径、3所有权/纯文件监控。语法编译通过，未扩全仓长测试；离线结果不证明策略效果。

## 4. 镜头与20分钟监控

utils.reward_fork_session独立后台监督，复用OS run锁与共享端口锁。跟踪镜头只读位置并set_spectator，不tick/reset/spawn/load。文件快照每1200秒及启动/退出记录一次；只统计manifest安全边界以内的新run数据。监控文件：session/launch.json、exit.json、camera.json、monitor_latest.json、monitor_history.jsonl及唯一segment日志。

Codex heartbeat carla-20已创建ACTIVE，每20分钟纯文件巡检，正常无变化保持安静；异常/停滞/完成/需要用户行动时通知。不自动重启、改参数或连接CARLA评估。首段u457正常退出不是失败。数值/完整性检查包括nonfinite、跳过更新、ratio误差、奖励残差、真实终局、逐车速度和超速、KL/熵与Critic拟合。STOP_REQUESTED在完整update边界生效，恢复前需明确核实与归档。

启动派发不等于训练成功；须结合训练日志、新安全checkpoint、镜头新鲜记录及真实退出码判断。后续结果在此文件追加，不在训练期间改源码。

## 5. 已准备的指令

从PythonAPI/examples/mappo_training以venv运行：

```powershell
# 严格预检，不连接CARLA
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m training.reward_fork_v2 --run-dir logs/R8C1_InstantPaceV2_u456_to476_s42_20261013_v2 --stop-update 476 --validate-only
# 确认首段全部退出后才恢复；不要重复运行
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m utils.reward_fork_session launch --run-dir logs/R8C1_InstantPaceV2_u456_to476_s42_20261013_v2 --stop-update 476
# 纯文件查询
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m utils.reward_fork_session inspect --run-dir logs/R8C1_InstantPaceV2_u456_to476_s42_20261013_v2
```

u476训练结束并确认清理后，使用diagnostics.run_reward_screen在相同12例/seed62001评估latest，再以diagnostics.analyze_reward_screen分析。比较完成/真实安全、两车起步/速度/超速与到达；总回报不能跨不同奖励直接比较。只有一条新奖励微调，不作奖励因果或收敛证明；需要时另做同u456同预算route_pace_v1控制，避免重复旧长跑。

## 6. 首段完成及真实恢复（2026-10-13）

北京时间10:23首段启动，10:25:19监督记录returncode0且镜头正常EXITED。完整u457断点SHA为efe19930a8dd0938749a1b98c6e60ef24cc04ee88206894cb5cb3e1697cfcbae，16训练回合12成功、1碰撞、1越界、2道路违规、0超时；奖励残差/更新前ratio误差/跳过更新均0。这些来自采样训练，不与确定性12例筛选直接比较效果。

10:27恢复至u476的第二段，已见实际新回合车辆生成及FOLLOWING镜头记录。10:29确认安全断点推进到u458，SHA为87f99a446ae28ecdf3c022e4c88c6ef66a63a5b3c308908454662c742acae691，说明保存—退出—严格恢复—继续提交已在真实CARLA中跑通。当前运行的是双红基础MAPPO奖励微调，蓝方Mut四车实验尚未启动。

反事实成本报告counterfactual_baseline_v2.json：同一旧轨迹agent0超速成本−1.717→−12.586，agent1−0.636→−0.961；agent1有73个有效步骤在速度恢复后旧欠账仍负、新即时成本已0。门控数据完整；这些只是重算旧轨迹的奖励机制证据，不是新策略变好了。

## 7. V2已完成，独立评估未通过速度目标

20个新updates完成，320训练回合：263成功、14碰撞、14越界、23道路违规、6超时；全部update的ratio误差、奖励残差与跳过更新为0。u476 SHA为382436d7c36ad99909b70f8109aa23b1eda21f7cbd6a0e962e481957fb7b2153；训练北京时间10:54正常退出。20分钟纯文件监督实际于10:47记录u471，无完整性问题。25份受控源码已归档source_snapshot，不覆盖旧实验。

冻结u476同12例/seed62001评估COMPLETED，清理0车辆/传感器并恢复世界。配对resolved_config与12场景/初始配置SHA一致，分析与逐case比较见diagnostics/reward_instant_v2_20261013/after_u476_s62001_v1/behavior_analysis.json及paired_comparison.json。

| 指标 | agent0 u456→u476 | agent1 u456→u476 |
|---|---|---|
| active均速km/h | 26.47→29.34 | 12.16→12.19 |
| 峰值km/h | 48.70→54.22 | 28.14→26.87 |
| 超30 active步比例 | 43.31%→52.67% | 0→0 |
| 首次运动均值秒 | 0.325→0.317 | 1.467→1.367 |
| 到达均值秒 | 4.483→3.908 | 9.375→9.283 |
| 到达案例 | 12→12 | 12→12 |

agent0全部12例均速与峰值上升，agent1最慢启动仍4.3秒。两次确定性筛选都未记录安全终止，但不能忽略明显超速。结论：V2是未达正常速度目标的工程候选，不宣称奖励有效或红驾驶合格。额外20次PPO更新、旧Critic适应及采样等因素未由同预算控制分离，不能单独证明权重弱就是原因；加强阈值附近与高速区间成本是下一项可测试假设。

## 8. V3短修订（当前运行）

用户授权奖励实验与修复继续有效。V3仅改超速为−dt×(0.5e+0.5e²)，e=max(0,v−15km/h)/(15km/h)。30km/h约−0.1/0.1s步、50km/h约−0.389；即时低速、所有豁免及其他驾驶/安全项沿用V2。阈值右侧加入线性斜率，不把soft reward称物理硬限速。

单独run：logs/R8C1_InstantPaceV3_u456_to461_s42_20261013_v1。仍从同一u456完整状态开始，先5个新updates至u461作工程筛选，不与V2的20update作公平算法效果排名。seed SHA为c9f81c607f3e140ff652f5062778ce8ebf1628d8296e40c5eb8c2f7a4b670a71。V2权重/证据/源码快照保持。

相关检查累计32项通过，包含V3纯数学、实际mock环境30km/h成本−0.1/V2−0.02、其他项保持、runtime/schema系数与profile、固定u456派生身份及源码指纹。未加全仓长测试。11:07 V3首段派发，运行是否成功仍看安全断点与日志；启动后源码冻结。carla-20已更新到当前V3目录/目标，仍纯文件每20分钟、只在有意义变化时通知，不自动重启或改奖励。

V3结束后做相同12例冻结评估。5updates不足证明收敛或新奖励合理性，只用于筛选方向；不根据安全训练平均值代替逐车行为验收。

## 9. V3完成、效果复盘与V4候选（当前未训练）

V3已完成5updates/80回合并exit0，u461 SHA ae5ae7abd0e2240e6b0bbcc52a0deee9b64bfce57b7fa8627f4e2a33bf993245。匹配12例评估COMPLETED，团队10/12；新增SC01碰撞和SC05终点侧界外5.3cm停车超时。agent0峰37.02，配对共同成功10例均速26.41→22.29km/h；不能用停车稀释后的pooled均速12.99当成功降速。agent1最慢起步5.7秒。V3仍未合格。

用户要求分析先前效果，当前没有继续训练。V4先修近终点blanket豁免，允许0.5m/s慢行但目标外停车仍扣费；保留V3超速及其他项。40项相关必要检查分批通过，V4固定u456离线准备与实际预检通过，未CARLA运行。carla-20已因本轮完成设PAUSED，保留供下次运行恢复。详细实证、成本回放、侧向接近及起步问题见[训练效果与奖励优化复盘](./基础红策略_训练效果与奖励优化复盘_20261013.md)。

## 10. V4真实完成及V5状态（最新）

用户随后授权继续。V4于13:33启动，13:41完成5个新updates并exit0，u461 SHA d87cb4cd9099548158a839a77ee926e92ede9a5211ecb777246addcfafba5610，80训练回合68成功、2碰撞、5越界、5道路违规、0超时，数值完整性检查0。27份源码已归档。匹配冻结评估共同完成11/12，0碰撞/0超时、1道路违规；SC05未进近终点区域就驶入反向邻道，不能把失败变类写成停车修复成功。V4报告见v4_after_u461_s62001_v1和paired_comparison_v4.json。

V5实现持续参考车道偏离成本及非法宽限取消，沿用V4其余数学。49项相关必要检查分批通过。目录logs/R8C1_InstantPaceV5_u456_to461_s42_20261013_v1初始u456 SHA70772348e3b3d178a5b2c2694a713ddcca2c6a7edbac109cf6ba66e13e569272。此前仅准备、未启动，监控只读记录状态；本次发现并纠正启动漏项，18:12真实派发5updates至u461，等待新安全断点及镜头/退出证据。当前并无V5效果结论。

## 11. V5已结束，仍未合格

18:19:34 V5完成5新updates/80训练回合并exit0；最终u461 SHA cbe5bca2d7abe6f6633ae1eb05ef76f2160a2c4e980f4b606b00dfafd457ae33。74训练成功、2碰撞、2越界、2道路违规、0超时；数值与概率/奖励一致性检查均0。真实FOLLOWING镜头记录与28份source_snapshot已保存，旧source与权重没有覆盖。

同12例/seed62001独立确定性评估共同成功10/12，1碰撞、1道路违规，NOT_QUALIFIED。agent1启动改善，但速度/侧向风险仍在；agent0峰46.30、agent1峰34.29km/h。SC05 agent0到达、agent1反向邻道违规终止；新lane成本生效不能等同于行为合格。见paired_comparison_v5.json。监控随本轮完成暂停，当前不存在本次继续训练或评估。后续优先控制/信息与学习诊断，暂不将短预算奖励候选归为可用正常红基线。
