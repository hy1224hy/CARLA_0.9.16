# 研究点一：五种子实验设计与实施计划

用户确认：500k合计步/组，seed42～46，最多4组并行；只准备脚本，不启动全矩阵。

分别检验任务可学习性、算法最终效果、训练稳定性、互学习机制、奖励必要性及扰动泛化。保留负结果，不通过弱化基线保证Mut-PPO领先。

核心60组：PPO、A2C、TRPO、Mut-PPO × 3等级 × 5seed。PPO自身KL和单向蒸馏属于机制对照，不算独立算法。所有方法均用两个副本，匹配A+B总交互预算、网络、混合动作和选模机会。

任务版本：L1=highway_l1_style_v4，L2=highway_behavior_v3，L3=highway_hybrid；旧包不修改。跨等级成功定义、状态维度不同，只做同等级方法比较。全矩阵前先验收L1慢行可学习性、L2切入归因与L3碰撞计数。

beta30组：.005/.08 × 3等级 × 5seed，复用主实验.02及PPO beta=0。
机制30组：one_way、acc_only、lane_only × L1/L2 × 5seed。
奖励35组：L1去style_dense、impact_dense、potential、lane_cost各5seed；L2去risk_dense、event_and_terminal、potential各5seed。精确标记移除内容，不将去单个密集项称为去所有密集奖励。
共155组，7750万训练决策，不含调参和评估。可选PPO_selfKL再15组。

独立highway_research包实现配置、环境包装、更新、训练、恢复、独立审计、编排和报告。先写测试，核验TRPO实际KL约束/拒绝步回滚、A2C单次on-policy更新、冻结teacher、one_way方向、恢复与审计。

主统计：最终latest的A/B各40development场景，先在seed内平均A/B，再跨5seed取均值与样本SD。selected另列，禁止事后选副本。冻结方案后使用100个test场景。扰动、随机执行分条件统计，固定动作随机seed。

曲线包含训练ASR、selection ASR/约束、收益、beta/KL；均值±样本SD，不画A/B逐轮择优包络。收益只在相同奖励定义内直接比较，奖励消融用共同物理指标。

seed才是独立重复；A/B、场景不能当额外训练seed。配对差与bootstrap区间只作n=5的探索性分析。缺失不填0，正式汇总须种子齐全。保存源码/配置/版本/检查点哈希和预算，按阶段执行，最多4组并行。
