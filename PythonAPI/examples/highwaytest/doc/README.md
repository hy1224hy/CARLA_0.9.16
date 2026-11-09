# Highway 文档索引

Highway 相关设计、操作说明和验证记录统一存放在仓库根目录的 `highway/doc/`，与 `PythonAPI/examples/mappo_training/docs/` 中的 CARLA 文档分开。

文档迁移不改变训练入口；现有源码仍位于 [PythonAPI/examples/highwaytest](../../PythonAPI/examples/highwaytest/README.md)。

## 当前一对一直道研究

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
