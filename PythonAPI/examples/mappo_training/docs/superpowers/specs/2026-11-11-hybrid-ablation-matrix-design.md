# 混合动作协议消融矩阵设计(hybrid ablation matrix)

日期:2026-11-11
协议:`highway_hybrid_twin_v1`(连续加速度 + 离散换道,原生控制器执行横向跟踪)

## 意图与背景

研究点一 = **奖励函数设计**(L1 温和阻碍 / L2 反复切入 / L3 碰撞攻击,事件优先、物理子步判据、密集过程项)与**互学习机制**(同任务 A/B 双副本、冻结教师、Gaussian+Categorical 双头 KL、β warmup/ramp)。

旧连续转向协议(`straight_twin_v1`)已给出证据:L2 mutual 0.433 vs independent 0.387(跨 seed SD 缩小 4.5×),L3 mutual 0.970 vs 0.947,L1 互学习无改善;β=0.02 为峰值;各奖励分量剔除均有效。2026-11-11 动作表示适配为混合动作后(commit 2108f25),pilot 已证明三级奖励在新协议下可学习(L1 52.5% / L2 15% / L3 15%,100k 单 seed)。**旧结论不能直接迁移,必须在新协议内重新验证**。

本矩阵目标:在混合动作协议内,用 5 seeds 冻结口径补齐研究点一的完整证据链——PPO vs Mut-PPO 收益对比、β 消融曲线、奖励分量消融。结果直接支撑论文表格。

## 目标

1. **A 主对比**(30 runs):3 级 × {independent, mutual} × 5 seeds × 500k 步。回答:互学习在新协议下是否带来成功率和跨 seed 稳定性收益,在哪些等级生效。
2. **B β 消融**(10 runs):L2 × mutual × 5 seeds × β∈{0.005, 0.08}(β=0.02 在主对比中,β=0 即 independent 臂)。回答:β 灵敏度曲线,峰值位置。
3. **C 奖励消融**(15 runs):L2 × mutual × 5 seeds × {no_risk, no_event, no_dense}。回答:risk 项(THW 窗口风险反馈)、event 项(切入事件奖励)、dense 项(密集效果 d 项,effect_weight=0)各自对反复切入学习的贡献。

## 非目标

- **今晚不做 SAC/Mut-SAC**(用户已裁定)。SAC 适配混合动作空间的兄弟模块(新协议 `highway_hybrid_sac_v1`)另行设计,明晚跑。
- 不改变任何 `highway_hybrid/` 源码、不改变奖励函数、不改变评估协议。
- 不用 test 集调参;test 100 场景终审在口径冻结确认后跑。
- 不恢复 `reward_repair_20261110` 旧直线矩阵(与本矩阵无关,保持原状)。

## 反目标(什么算失败)

- 任何 run 出现非有限 loss / 崩溃后无法断点续跑,导致矩阵残缺。
- 配置与 main_v1 口径不一致(预算、warmup、β、seed 范围之外混入差异)。
- 修改 frozen 源码导致 pilot checkpoint 失去可评估性。
- 用 validation(selection split)结果当作最终证据;证据必须是冻结模型的 development/test 独立审计。

## 约束(冻结口径)

| 项 | 值 |
|---|---|
| 总预算 | `total_steps=500000`(A+B 两副本合计,用户裁定) |
| warmup/ramp | 各 20000(plan 命令公式 `min(20000, steps//10)`) |
| β | 0.02(mutual 默认,主对比) |
| seeds | 42, 43, 44, 45, 46(5 个,用户裁定) |
| 学习超参 | 与 pilot 一致:rollout 512、epochs 4、minibatch 128、hidden 128、lr 3e-4、γ .99、λ .95、clip .2、entropy_coef .003、lane_entropy_coef .01、KL 乘子各 1.0、refresh_rounds 1 |
| 物理/任务 | duration 20s、policy_frequency 5、simulation_frequency 15、target_speed 18、target_time_gap 1.5 |
| selection 契约 | eval_interval 20000、eval_cases 32(冻结要求,不可改)、场景 seed 从 2,000,000 起 |
| 评估 | development 40 固定场景(follow/adjacent 各 20,seed 从 4,000,000 起);test 100 场景(seed 从 3,000,000 起) |
| 奖励参数 | 除消融臂明确改动的单个分量外,全部与 pilot 冻结值一致(effect/event/risk=1,l2_effect_coef=.5,lane_cross_bonus=0,cut_reward_cap=4,cut_success_min=1,approach/guidance/geometry=0) |
| 代码 | commit 2108f25,`highway_hybrid/` 及共享源文件零改动;每 run 自动记录 source sha256 |

## 实验矩阵明细

| 臂 | 配置名模板 | 配置差异 | 数量 |
|---|---|---|---|
| A | `L{1,2,3}_{independent,mutual}_s{seed}` | 无(基线口径) | 30 |
| B | `L2_beta{0.005,0.08}_s{seed}` | method=mutual, beta 改 | 10 |
| C | `L2_no_risk_s{seed}` | risk_weight=0 | 5 |
| C | `L2_no_event_s{seed}` | event_weight=0(联合消融:切入事件奖励 +2 与终局成功奖金 +5 同源,均随 event_weight 归零;口径沿用 2026-11-09 实施计划与 2026-11-10 建模报告既有约定) | 5 |
| C | `L2_no_dense_s{seed}` | effect_weight=0(l2_effect_coef 受 validate 约束必须 >0,故剔除密集 d 项用 effect_weight=0) | 5 |

合计 **55 runs × 500k 步 = 2750 万 A+B 决策**。

注:用户贴的 `hybrid_manual_L2_s42` 与 pilot 的 `L2_independent_s42`(100k)完全同配置,已被本矩阵 A 臂(L2_independent_s42@500k)覆盖,不单独重跑。

## 文件规划

```
highwaytest/runs/hybrid_matrix_20261111_v1/
├── README.md                 # 研究定义、口径、声明(所有 advantage 均为假设)
├── study.json                # 冻结的研究定义
├── configs/
│   ├── main/                 # A 臂 30 个
│   ├── beta/                 # B 臂 10 个
│   └── reward/               # C 臂 15 个
├── train/
│   ├── main/                 # A 臂 30 run
│   ├── beta/                 # B 臂 10 run
│   └── reward/               # C 臂 15 run(与旧 straight 矩阵 train/{main,beta,reward} 布局一致)
├── logs/                     # 每 run 控制台日志(同布局)
├── control/                  # 编排脚本、监控、状态、配置生成与校验
├── audit_selected/
│   ├── main/  beta/  reward/ # development 40 场景冻结审计,`--runs train/<arm>` 分臂执行
├── audit_test/               # test 100 场景终审(口径冻结后)
└── report.md                 # 汇总表(分臂)
```

- 分臂子目录下仍用统一 evaluate 入口(`--runs <arm>/train` 指向该臂 run 集合),audit 输出分臂存放,便于逐臂核对。
- `control/` 内脚本不属于被冻结的源码集,修改不影响 checkpoint 可评估性。
- 每 run 命名与 config 文件名一致(评估要求 run 名唯一)。

## 编排与调度

- **臂序**:A 臂独占全部 8 workers 先跑完(≈7.5h,4 波);随后 B+C 合并入同一 worker 池(25 runs ≈ 4 波 ≈ 6.5h),总时长 ≈ 14h。A 优先保证明早至少有主对比完整证据。
- **并行度**:8 workers(16 逻辑核,每 run 1 线程;机器上仅空闲 tensorboard)。沿用 pilot.py 模式:ThreadPoolExecutor + 独立子进程 + `-m highway_hybrid train` + 自动 `--resume latest.pt` + 输出重定向到 logs/。
- **失败自动恢复**:每 run 失败后自动重试至多 2 次(resume 为事务性,按轮次回滚 JSONL,可安全续跑)。全程记录 batch_status.json / pipeline_status.json。
- **监控**:10 分钟周期 monitor 进程,记录每 run 最新 steps、loss、熵、KL、实际 β、成功率快照,写入 monitor.jsonl;TensorBoard 新端口(6011)指向 `train/` 根。
- **时间估算**:100k ≈ 0.39h(实测)→ 500k ≈ 1.95h/run;55 runs ≈ 107 核时;8 并行 + 争用 ≈ 14h。A 臂约 7.5h,B+C 约 6.5h。
- **验收**(明早):55 个 summary.json 全部 `status=complete`;分臂 development 审计完成;report.md 生成均值±SD 表。test 终审与 SAC 臂另行安排。

## 证据声明规范(写入 README)

- 各臂回答的假设事先写明;负结果/零结果如实报告。
- 跨 seed 均值±SD 是主证据;单 seed 数值只作趋势参考。
- development 审计用于汇总表;test 终审是论文口径,冻结后只跑一次。
- 互学习证据 = 同预算下 mutual vs independent 的均值差 + SD 收缩,不是 β 曲线本身。

## 留给构建者的决定

- 配置生成与校验脚本的具体写法(用 frozen `HybridConfig` dataclass 生成,校验:预算一致、eval_cases=32、命名唯一)。
- 编排脚本内部结构(参照 `hybrid_pilot_20261111_v1/control/pilot.py` 与 `highwaytest/straight/__main__.py` 的 batch 模式)。
- 监控脚本字段与周期(参照 pilot 的 monitor.py)。
- report 汇总脚本的具体表结构(分臂表:成功率/碰撞率/无效率/有效切入/blocking_time/return 的均值±SD)。
