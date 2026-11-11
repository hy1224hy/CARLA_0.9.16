# 研究点一：五种子批量实验工具

正式计划见[实验报告](../highway/doc/2026-11-11-research1-experiment-report.md)。本工具不修改冻结任务包：L1 style-v4、L2 behavior-v3、L3 hybrid-v1。

算法为原生混合动作实现：PPO-Clip、同步A2C（RMSprop、单次全批on-policy更新）、TRPO（联合分布KL、共轭梯度、回溯线搜索）、Mut-PPO。不是直接调用Stable-Baselines3；Dict混合动作统一使用高斯加速度及掩码分类换道。TRPO无熵奖励；学习率控制其critic，actor由KL限幅。

所有方法都有两个副本，500k为A+B合计，约250k/副本；PPO双副本是独立训练，不能称作集成执行。所有方法相同选模机会，最终A/B先在seed内平均，跨5seed统计。单向蒸馏、self-KL是机制对照，不充作独立算法。

采样器跨rollout保留未结束回合，检查点包含物理环境及采样cursor、优化器、teacher和RNG，恢复与不中断训练有一致性测试。这是新训练协议，不能直接续用旧包模型或把旧运行混入新五种子统计。

## 准备与启动

以下从`PythonAPI/examples`运行；Python使用仓库venv。

```powershell
& ../../venv/Scripts/python.exe -m highway_research plan --root highwaytest/runs/research1_5seeds_20261111_v1
& ../../venv/Scripts/python.exe -m highway_research batch --root highwaytest/runs/research1_5seeds_20261111_v1 --arms main --jobs 4 --dry-run
& ../../venv/Scripts/python.exe -m highway_research batch --root highwaytest/runs/research1_5seeds_20261111_v1 --arms main --jobs 4
```

准备过的目录不可覆盖；代码变动后必须建立新版本。默认155组，`--optional`增加15组self-KL和L2的10组预热/刷新周期对照。仅准备配置不会训练。

PowerShell快捷脚本（以下命令从仓库根目录调用；脚本内部自动切换工作目录）：

```powershell
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action Preflight
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action TrainMain
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action TrainAblations
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action TensorBoard -Port 6012
```

main未通过行为验收时，先不要启动消融。批跑最多4进程，每进程单线程Torch。失败状态在control/batch_status.json，日志在logs；重跑同一命令会跳过已完成、恢复已有latest。只有完整、版本匹配的latest.tmp才可恢复；不删失败目录，不自动改变配置。

## 调参

公平调参使用独立seed32/33与development，不接触test：

```powershell
& ../../venv/Scripts/python.exe -m highway_research plan --root highwaytest/runs/research1_tuning_v1 --tuning --seeds 32 33 --steps 100000
& ../../venv/Scripts/python.exe -m highway_research batch --root highwaytest/runs/research1_tuning_v1 --arms tuning --jobs 4
& ../../venv/Scripts/python.exe -m highway_research audit-study --root highwaytest/runs/research1_tuning_v1 --arms tuning
& ../../venv/Scripts/python.exe -m highway_research select-tuning --root highwaytest/runs/research1_tuning_v1 --output highwaytest/runs/tuning_selected.json
& ../../venv/Scripts/python.exe -m highway_research plan --root highwaytest/runs/research1_formal_tuned_v1 --tuning-selection highwaytest/runs/tuning_selected.json
```

共72组：4方法×3候选×3等级×2seed。PPO/Mut/A2C比较lr=1e-4/3e-4/7e-4，TRPO比较KL=.005/.01/.02。默认主矩阵是起始超参，不代表已完成公平调参；select-tuning只根据development导出设置，再用tuning-selection建立正式新目录，不能修改已冻结配置。

## 独立审计

```powershell
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action AuditDev
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action AuditSelected
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action AuditRobust
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action AuditStochastic
./PythonAPI/examples/highway_research/scripts/experiment.ps1 -Action Report
```

最终方案冻结后才运行AuditTest（100场景）。随机执行固定action seed 0/101/202，先平均动作seed，再平均A/B，再跨训练seed；和确定性结果分开。OOD包含IDM时距1/2秒、较保守换道，以及原生recovery的横向/航向扰动；实际与参考场景同步修改。

保存init、latest、selected及每100k模型快照ckpt_100000等。快照文件名是名义预算，实际步数以train_steps为准。可用audit-study --checkpoint ckpt_200000重评策略稳定性。init是随机初始化网络的对照，不等同于均匀随机动作；仅init审计会复用主PPO初始化，不重复训练基线。

每模型有episodes、trajectories和冻结checkpoint；审计不是训练validation。A/B缺一或5seed缺失时，正式report拒绝汇总；`--allow-partial`只能生成明确标记的探索结果。没有结果时不会填0或生成伪结果图。

## 奖励消融与可比性

L1去style/impact仅去密集项，保留对应终局定义；去geometry只去塑形；去lane_change去实际换道费用，但保留换道预算/冷却。L2 risk_weight=0去时距密集项，保留控制代价；event_weight=0同时去事件及终局奖金，准确称event_and_terminal。

环境返回的return是优化目标；canonical_return用统一完整奖励重算。L2直接改risk/event权重时，通过相同物理frame上的完整权重observer恢复canonical，不更改物理状态或成功判据。跨消融以物理指标与canonical return比较，不能只看被移除后的原始return。

## 收集结果

report输出逐副本、逐seed、组统计、配对bootstrap、覆盖情况、后期回撤/达阈步数/耗时、OOD性能变化及全部曲线CSV。PNG/PDF按arm/metric/split/checkpoint/condition/execution分别命名，不混合口径。

```powershell
& ../../venv/Scripts/python.exe -m highway_research replay --run <运行目录> --checkpoint latest --team A --case-seed 4000003 --family adjacent --output <新GIF路径>
```

默认固定场景回放，不按胜负挑选；生成GIF、物理轨迹图和JSON。主文优先约8～10张图，不必把每个loss都作为有效性证据。
