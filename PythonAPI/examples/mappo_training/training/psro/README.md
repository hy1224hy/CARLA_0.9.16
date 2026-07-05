# training/psro/

PSRO (Policy-Space Response Oracles) 对抗训练框架。

- 设计计划：[docs/对抗训练_MAPPO_PSRO_设计计划.md](../../docs/对抗训练_MAPPO_PSRO_设计计划.md)
- **实现评审 + 完整用法指南 + 指标字典**：[docs/对抗训练_PSRO_实现评审与用法指南.md](../../docs/对抗训练_PSRO_实现评审与用法指南.md) ← 先读这份
- **首次运行问题诊断**：[docs/对抗训练_PSRO_首次运行问题诊断_20260622.md](../../docs/对抗训练_PSRO_首次运行问题诊断_20260622.md)（payoff 饱和 / 蓝车自杀，附基线判定）

## 文件清单

| 文件 | 作用 | 行数 |
|---|---|---|
| `population.py` | RED/BLUE 团队池加载、采样、保存、训练元信息 | ~210 |
| `meta_solver.py` | 元策略求解器（uniform + PRD） | ~120 |
| `payoff_table.py` | 经验收益张量 + JPC R_- 计算 | ~180 |
| `train_blue_oracle.py` | 内层 MAPPO oracle 训练（每 ep 从 σ_red 采样 RED） | ~500 |
| `jpc_eval.py` | JPC 交叉评估 + 泛化/鲁棒性诊断（独立工具，热图输出） | ~340 |
| `psro_runner.py` | 外层 PSRO 主循环（程序入口） | ~330 |
| `build_red_pool.py` | Phase 0 辅助：一键多 seed 重训 / 收编已有策略组 RED 池 | ~240 |
| `baseline_eval.py` | 基线评估：良性蓝车下判定「红车基础策略不足」还是「蓝车自杀局部最优」 | ~270 |

## 数据流

```
[输入]                 [PSRO 外层]                   [输出]

RED 池                 ┌───────────────────┐         BLUE 池
(N 个 MAPPO ckpt) ─┐   │                   │   ┌──→  (M 个 BLUE 团队 ckpt)
                   ├──→│  psro_runner.py   │───┤
σ_red 初始 uniform ─┘   │                   │   ├──→  PayoffTable JSON
                       │                   │   │
                       │  每轮：            │   ├──→  TensorBoard 日志
                       │   1. train_oracle │   │
                       │   2. eval pair    │   └──→  最终 σ_red / σ_blue
                       │   3. meta solve   │
                       │                   │
                       └───────────────────┘
                              ↓
                       [可选] jpc_eval.py  ───────→  JPC 矩阵 + R_- 报告
                                                    + 热图 PNG
```

## 起步命令

### Phase 0：组建 RED 池（前置条件）

RED 池本质就是"一个放若干兼容 `.pt` 的目录"（`RedTeamPool.from_dir` 加载目录下所有 `*.pt`，
只要求每个含 `actors[cooperative_ids]`）。两种组池方式：

**方式②：动态收编已有策略（不训练，秒级，推荐先用这个跑通）**

```bash
# 收多个 run 的 best.pt（彼此差异更大）
python -m training.psro.build_red_pool --from-ckpts \
    logs/exp/A0617_baseline_seed42/best.pt \
    logs/exp/A0610_baseline_seed42/best.pt \
    logs/exp/A0608_baseline_seed42/best.pt \
    --out logs/red_pool/ckpts

# 或用 glob 收同一 run 的多个快照（不同 update = 不同策略）
python -m training.psro.build_red_pool --from-ckpts \
    "logs/exp/A0617_baseline_seed42/mappo_robust_v2_update_1[0-9]00.pt" \
    --out logs/red_pool/ckpts
```

会逐个校验 `actors` 是否齐全、起唯一名收进 `--out`（同名默认跳过，`--overwrite` 覆盖，
`--link` 用软链省空间）。**多样性提醒**：若收编的都来自同一 seed（仅 update/日期不同），
RED 差异有限、PSRO 种群效应偏弱；追求多样性仍建议混入不同 seed 重训的策略。

**方式①：重训多 seed（最贵但多样性最高，需 CARLA）**

```bash
python -m training.psro.build_red_pool --seeds 42 123 2024 --updates 150 \
    --out logs/red_pool/ckpts
```

或手动等价：

```bash
for seed in 42 123 2024; do
    python -m training.train_no_rllib_mappo_robust_v2 \
        --seed $seed --updates 150 \
        --save-dir ./logs/red_pool/seed_$seed
done

# 把 best.pt 拷到 RED 池目录（统一目录）
mkdir -p logs/red_pool/ckpts
cp logs/red_pool/seed_42/best.pt   logs/red_pool/ckpts/red_seed42.pt
cp logs/red_pool/seed_123/best.pt  logs/red_pool/ckpts/red_seed123.pt
cp logs/red_pool/seed_2024/best.pt logs/red_pool/ckpts/red_seed2024.pt
```

### Phase 1.5：基线判定（强烈建议在 PSRO 前先跑）

确认在 4 车场景里红车**能不能跑通**，把「红车基础策略不足」与「蓝车自杀式局部最优」分开：

```bash
python -m training.psro.baseline_eval \
    --red-pool-dir logs/red_pool/ckpts \
    --scenario scenario_01_basic.json \
    --blue-mode natural --n-eval 5 \
    --output-dir logs/psro_runs/baseline
```

- `natural`（蓝车=良性自然司机）下 `coop_success` 高 → 红车/几何 OK，问题在对抗蓝车自杀；
- `coop_success` 低且 `coop_collision` 高 → 红车基础策略不足/生成过挤；低且不撞 → 终点/距离问题。
- 脚本会直接打印 A/B 判定。详见
  [首次运行问题诊断](../../docs/对抗训练_PSRO_首次运行问题诊断_20260622.md)。

### Phase 2-3：PSRO 主流程

```bash
python -m training.psro.psro_runner \
    --red-pool-dir logs/red_pool/ckpts \
    --scenario scenario_01_basic.json \
    --psro-iters 5 \
    --inner-updates 30 \
    --meta-solver prd \
    --prd-gamma 0.2 \
    --save-dir logs/psro_runs
```

### Phase 3：JPC / 泛化交叉评估

```bash
python -m training.psro.jpc_eval \
    --red-pool-dir logs/red_pool/ckpts \
    --blue-pool-dir logs/psro_runs/<timestamp>/blue_pool \
    --scenario scenario_01_basic.json \
    --n-eval 5 \
    --output-dir logs/psro_runs/<timestamp>/jpc
```

`jpc_report.json` 中 **PSRO 池看 `generalization`**（`per_blue.worst_eff` 越高越鲁棒、
`gen_gap` 越接近 0 越不过拟合；`best_single_blue_worstcase`、`uniform_mix_worstcase`）。

`jpc_R_minus` 仅在 **配对池**（`blue_i` 专攻 `red_i` 的 InRL 基线）下才按论文 JPC 解读——
那时加 `--paired-pool`。PSRO 产出的 BLUE 池是对 σ_red 混合训的，**非配对**，R_- 不适用，
报告里会标 `jpc_R_minus_valid_as_jpc=false`。详见
[实现评审与用法指南 §3.1 / §5 / §6](../../docs/对抗训练_PSRO_实现评审与用法指南.md)。
