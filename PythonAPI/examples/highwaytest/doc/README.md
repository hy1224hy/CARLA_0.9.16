# Highway 文档索引

Highway 相关设计、操作说明和验证记录当前统一存放在 `PythonAPI/examples/highwaytest/doc/`，与 `PythonAPI/examples/mappo_training/docs/` 中的 CARLA 文档分开。

文档迁移不改变训练入口；现有源码仍位于 [PythonAPI/examples/highwaytest](../README.md)。

## 当前一对一直道研究

混合动作适配见[highway_hybrid 使用说明](../../highway_hybrid/README.md)与[适配计划](混合动作MutPPO_适配计划_20261110.md)：连续加速度、离散换道、分头互学习、独立训练与评估协议。旧连续转向实验继续保留。

当前奖励开发与训练批次见[PPO 三级奖励修复与训练执行](PPO三级奖励修复与训练执行_20261110.md)。独立模型评估工具见[highway_eval 使用说明](../../highway_eval/README.md)：固定检查点、独立开发/test 场景、分场景指标及轨迹；当前 r3 另增加了两个过程标志，使用 26 维观测。

第七版运行诊断见[第七版奖励与指标诊断（2026-11-10）](第七版奖励与指标诊断_20261110.md)：包含 a7 的阶段性验证结果、L2 边界阻挡复查、奖励问题及快速验证矩阵。

最新整体建模见[研究点一整体实验建模报告（2026-11-10）](研究点一_整体实验建模报告_20261110.md)：覆盖当前 24 维观测、三级奖励、同任务双副本互学习、共享初始化、实验矩阵、评价统计与策略池；L2 默认一次有效切入达到次数门槛，前四次分别奖励。以下 V1 文档用于理解历史设计，参数与语义以当前报告及实际运行配置为准。

旧代码审查见[代码与当前计划一致性审查](代码与当前计划一致性审查_20261109.md)。新增独立一对一协议的入口与命令见[训练操作说明](一对一直道_训练操作说明_20261109.md)；旧训练器继续保留原协议。

1. [观测、动作与三级奖励 V1](一对一直道_观测动作与三级奖励V1_20261109.md)：22维观测、连续控制、三级奖励及开发集校准。
2. [分级对抗奖励与策略池设计](分级对抗奖励与策略池_设计细化_20261109.md)：奖励分级、同配置双训练副本互学习、基础驾驶初始化及策略池。
3. [两日实施方案](研究点一_互学习对抗驾驶_两日实施方案_20261109.md)：早期时间预算与实验组织；任务设定以以上更新设计为准。

## 既有 highwaytest 工程文档

以下保留各版本原有实验范围，不与当前一对一直道方案混用。

| 范围 | 文档 |
|---|---|
| 项目总说明 | [原项目 README](HIGHWAYTEST_README.md) |
| 一红两蓝瓶颈 v4 | [设计](BOTTLENECK_V4_DESIGN.md)、[服务器流程](BOTTLENECK_V4_SERVER.md)、[验证记录](BOTTLENECK_V4_VALIDATION.md) |
| staged_v1 分阶段训练 | [设计](STAGED_DESIGN.md)、[服务器流程](STAGED_SERVER.md)、[验证记录](STAGED_VALIDATION.md) |
| v3 研究协议 | [研究与评价](RESEARCH_V3.md)、[验证记录](VALIDATION_V3.md)、[CARLA 复验边界](ALGORITHM_TRANSFER.md) |
| 初期迁移与运行 | [迁移记录](MIGRATION.md)、[服务器指南](SERVER.md)、[服务器审查](SERVER_AUDIT_20261107.md)、[初期验证](VALIDATION.md) |

## 迁移记录

2026-11-09：迁入原 `highwaytest/docs/` 的13份文档、原项目README全文以及原 `mappo_training/docs/` 中本次新增的3份Highway研究文档，共17份；项目原README改为文档入口。
