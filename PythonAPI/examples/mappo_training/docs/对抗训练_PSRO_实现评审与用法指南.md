# 对抗训练 PSRO —— 实现评审与用法指南

**日期**：2026-06-21
**评审对象**：`training/psro/`（population / meta_solver / payoff_table / train_blue_oracle / jpc_eval / psro_runner）
**配套设计文档**：[对抗训练_MAPPO_PSRO_设计计划.md](对抗训练_MAPPO_PSRO_设计计划.md)
**论文依据**：Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent RL*, NeurIPS 2017（PSRO / 元策略 / JPC R_-）

---

## 0. 一句话结论

`training/psro/` **正确地把论文的 PSRO 思路迁移到了本项目**：外层 PSRO 调度 + 内层
MAPPO(CTDE) oracle + 元策略求解(uniform/PRD) + 经验收益张量 + JPC 交叉评估，分层关系
与论文一致，代码可导入、可运行（需 CARLA server + RED 池）。本次评审**修复了 4 处问题**
（其中 1 处关键：指标可解释性），并补充了对 PSRO 始终有效的泛化/鲁棒性指标。下文给出
完整复现步骤。

---

## 1. 设计是否符合论文思路 —— 逐条核对

| 论文要素                                  | 代码落点                                                      | 评审结论                                             |
| ------------------------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| PSRO 外层种群循环（Algorithm 1）              | `psro_runner.main` 的 `for psro_iter`                      | ✅ 每轮：训 oracle → 评估填表 → 元策略更新 → 记录，顺序正确           |
| 内层最优响应 oracle                         | `train_blue_oracle.train_blue_oracle`                     | ✅ MAPPO+CTDE，每个 episode 从 σ_red 采样 frozen RED 团队 |
| CTDE critic 只覆盖 BLUE 队（一般和约束，论文 §三注2） | `AgentConditionedCritic(num_agents=len(adversarial_ids))` | ✅ critic 只回归 BLUE 的 return，RED 仅作为环境进入 state     |
| 元策略求解器                                | `meta_solver.{UniformSolver,PRDSolver}`                   | ✅ PRD 在 BLUE/RED 单纯形上做投影复制动态，零和近似取 `U_red=-U`    |
| 经验收益张量                                | `payoff_table.PayoffTable`                                | ✅ 多指标矩阵，主指标 `adv_eff = 1 - coop_success`         |
| JPC 过拟合度量 R_-                         | `payoff_table.compute_jpc_R_minus`                        | ⚠️ 公式正确，但**对 PSRO 池的语义需修正**（见 §3.1，已处理）          |
| 防过拟合（对分布而非单点求响应）                      | σ_red 采样 + PRD γ-探索下界                                     | ✅                                                |
| InRL 退化判断                             | `psro_runner` 对 `len(red_pool)<2` 告警                      | ✅                                                |

**数值自检（无需 CARLA）**：
- PRD：给定"RED0 难攻击、RED1 易攻击"的收益矩阵，σ_red 正确集中到 RED0（γ=0.1→0.95，
  γ=0.4→0.8），即下一个 BLUE oracle 会去专攻最难对手——对抗 PSRO 的正确行为。
- JPC：对角强矩阵 R_-=0.625（高过拟合信号），均匀矩阵 R_-=0（无过拟合）。
- 收益张量子矩阵裁剪、NaN 先验填充均符合预期。

**结论**：分层架构、CTDE 约束、元策略方向、收益口径均与论文一致，**整体设计无误**。

---

## 2. 代码正确性核查（已执行）

| 检查项 | 方法 | 结果 |
|---|---|---|
| 语法 | `py_compile` 全部 7 个文件 | 全部 OK |
| 跨模块导入 | 实际 `import`（含 `carla`、6 个 psro 模块） | 全部 OK（此前 `.pyc` 缺失说明 runner/oracle/jpc 从未被导入过，现已验证可导入） |
| 依赖函数存在性 | grep 校验 `Transition/RolloutBuffer/compute_gae_per_episode/compute_blue_threat_scores/extract_*/gaussian_kl_diag/is_any_red_under_threat/avg_red_speed` | 全部存在且签名匹配 |
| Actor/Critic 接口 | `get_deterministic_action/sample_action_z/evaluate_actions_z/forward/log_std` | 全部匹配 |
| env 构造与方法 | `reward_version/disable_tensorboard/seed/set_scenario/get_global_state/step 四元组返回` | 全部匹配 |
| 奖励分量键 | `extract_base_driving_safe` 读取的键 vs `_compute_rewards_v2` 输出 | 完全对齐（speed/centering/yaw/jerk/steer_rate/safety_lidar/ttc/lane_invasion/step/collision） |
| 参数装配 | `psro_runner.build_parser` → `_build_inner_args` → `train_blue_oracle` 读取字段 | 0 缺失字段 |
| 场景装配 | `prepare_eval_scenario` 由 2 车 JSON 自动补出 4 车（红 0,1 + 蓝 2,3 居中） | 正确 |

> **RED ckpt 兼容性**：Phase 0 的 `train_no_rllib_mappo_robust_v2.py`（`num_agents=2`）保存
> `{"actors": {0,1}}`，`RedTeamPool` 按此读取——格式匹配。RED 在 2 车世界训练、在 4 车世界
> 被攻击，观测仍是 19 维（8 基础 + 11 雷达），维度兼容；雷达此时会看到蓝车——这正是对抗
> 鲁棒性测试的核心分布漂移，属设计意图。

---

## 3. 本次发现的问题与修改记录

### 3.1【关键·指标可解释性】PSRO 池上的 JPC R_- 语义不成立 → 已补充有意义指标

**问题**：论文的 JPC 要求矩阵对角线是"配对协同训练"的实例（player_i 与 player_j 一起训出来，
对角=匹配、非对角=错配，R_- 衡量错配掉点）。但本项目里：
- RED 池 = 各自独立预训练的合作团队；
- BLUE 池 = 对**演化中的 σ_red 混合分布**训出来的对抗团队，`blue_j` 与 `red_j` **无配对关系**。

因此把 `RED池 × BLUE池` 矩阵的对角线当作"配对效果"去算 R_-，**对 PSRO 池不可按论文解读**
（对角并不特殊，R_- 偏高/偏低都没有"联合策略相关性"含义）。

**修改**（`jpc_eval.py` / `population.py`）：
1. 新增 `compute_generalization_report()`，给出**与索引配对无关、对 BLUE 始终有意义**的指标：
   - `per_blue.mean_eff / worst_eff / std_eff`：每个 BLUE 对**整个 RED 池**的平均 / 最坏情况 /
     方差有效性。`worst_eff = min_i M[i,j]` 是**最坏情况鲁棒性**——PSRO 想最大化它。
   - `per_blue.gen_gap = Σ_i σ_red_train[i]·M[i,j] − mean_i M[i,j]`：**训练分布有效性 − 全池平均**。
     >0 越大说明该 BLUE 越是"只在自己训练的 σ_red 上强"，泛化越差——这才是本设置下真正
     对应"过拟合"的量。需要训练时的 σ_red 快照，已由 `BlueTeamPool.get_meta()` 暴露。
   - `best_single_blue_worstcase = max_j min_i M[i,j]`、`uniform_mix_worstcase = min_i mean_j M[i,j]`：
     衡量"是否存在对所有 RED 都不差的通用对手"以及"混合是否抬高了下界"。
2. R_- 仍保留，但**只有加 `--paired-pool` 开关**（即 BLUE 与 RED 按索引配对的 InRL 基线池）
   时才按 JPC 解读；否则报告里标注 `jpc_R_minus_valid_as_jpc=false` 并提示看泛化报告。

**如何做论文式 R_- 对比（M6 验收）**：构造 InRL 基线池——`blue_i` 专门只对 `red_i` 训一个 oracle
（`psro-iters` 与 RED 池等大、σ_red 退化为单点 one-hot），对该配对池跑 `jpc_eval --paired-pool`
得到基线 R_-；再对 PSRO 产出的 BLUE 池看 `worst_eff` 抬升 / `gen_gap` 下降。二者口径才一致。

### 3.2【缺陷】`--init-log-std` 静默失效 → 已修复

`train_blue_oracle` 用 RED 合作策略 `state_dict` 初始化 BLUE actor，`load_state_dict` 会把
`Actor(init_log_std=...)` 设的探索方差**覆盖**成 RED 收敛后的极小 `log_std`，使 `--init-log-std`
形同虚设、BLUE 起步几乎无探索，难以发现对抗动作。已在 load 之后显式
`actor.log_std.fill_(init_log_std)` 恢复（均值仍是自然驾驶，KL 锚点 std 同步），让该旋钮生效。

### 3.3【文档】`train_blue_oracle` docstring 误述 → 已更正

原 docstring 称返回值"已保存到 `args.save_path`"，实际本函数不落盘、仅返回 dict，由
`psro_runner` 经 `BlueTeamPool.add` 保存。已更正描述以免误导。

### 3.4【接口】新增 `BlueTeamPool.get_meta(idx)`

暴露 BLUE ckpt 的 `trained_against_sigma_red / psro_iter / stats / args`（不加载权重张量），
供泛化间隙诊断使用。

---

## 4. 已知近似与限制（非缺陷，使用时知悉即可）

- **PRD 投影是简化版**：`clip(p, lower)` 后归一化，并非 L2 最近的 γ-探索单纯形投影，归一化后
  个别概率可能略低于 `lower`。论文亦用类似实现，作为元策略求解足够；如需严格下界可改用
  精确单纯形投影。
- **确定性评估**：`evaluate_pair` 双方都用 `get_deterministic_action`，固定场景 + 播种 env 下
  各 episode 主要靠 CARLA 物理/TrafficManager 的非确定性产生差异；`n_eval` 增大主要降的是
  物理噪声方差，而非策略随机性。
- **inline JPC R_- 仅在 `len(red_pool)==psro_iter+1` 那一轮计算**（需方阵）；完整且权威的交叉
  评估请用独立的 `jpc_eval`（见 §5 Phase 3）。
- **范围**：固定 2v2、Town02 直道、Tesla Model 3，与设计文档 §10 一致。

---

## 5. 用法指南（可复现，端到端）

> 环境：仓库根 `D:\study\yan2\CARLA_0.9.16` 下的 **pipenv**（`pipenv shell` 或 `pipenv run`）。
> 所有命令在 `PythonAPI/examples/mappo_training/` 目录下、以模块方式运行。
> 先启动模拟器：`./CarlaUE4.exe`（训练建议 `-quality-level=Low`）。

### Phase 0 —— 组建 RED 池（PSRO 前置条件，必须 ≥ 3）

RED 池 = 一个放若干兼容 `.pt` 的目录（`RedTeamPool.from_dir` 加载目录下所有 `*.pt`，
只要求每个含 `actors[cooperative_ids]`）。**不一定要重训**——已有 ckpt 可直接收编。

**方式 B/C（推荐先用，不训练、秒级、不需 CARLA）：收编已有策略**

```bash
# 收多个 run 的 best.pt（差异更大）
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

逐个校验 `actors` 齐全、起唯一名收进 `--out`（同名默认跳过；`--overwrite` 覆盖；`--link` 软链）。

**方式 A（最贵但多样性最高，需 CARLA）：重训多 seed**

```bash
python -m training.psro.build_red_pool --seeds 42 123 2024 --updates 150 \
    --out logs/red_pool/ckpts
# 手动等价：for seed in 42 123 2024; do python -m training.train_no_rllib_mappo_robust_v2 \
#     --seed $seed --updates 150 --save-dir ./logs/red_pool/seed_$seed ; done  然后 cp best.pt 到池目录
```

**多样性提醒**：方式 B/C 若全来自同一 seed（仅 update/日期不同），RED 行为差异有限、PSRO
种群效应偏弱；追求多样性可混入不同 seed 重训的策略，或混合 best.pt（不同 run）+ 快照。

**验收**：每个 RED 单独 `coop_success_rate > 80%`（需 CARLA）；不达标的剔除，避免 σ_red 退化。

### Phase 2-3 —— PSRO 主流程（产出 BLUE 池 + 收益表）

```bash
python -m training.psro.psro_runner \
    --red-pool-dir logs/red_pool/ckpts \
    --scenario scenario_01_basic.json \
    --cooperative-ids 0,1 --adversarial-ids 2,3 \
    --psro-iters 5 \
    --inner-updates 30 \
    --rollout-episodes 16 \
    --n-eval-episodes 5 \
    --adv-min-gap 14 \
    --meta-solver prd --prd-gamma 0.2 \
    --save-dir logs/psro_runs
```

> **`--adv-min-gap`（默认 14m）**：相邻车最小纵向间距，拉开生成间距、防一上来追尾
> （首跑 payoff 全饱和的几何根因，见[首次运行问题诊断 §6](对抗训练_PSRO_首次运行问题诊断_20260622.md)）。
> **评估必须用同值**：`jpc_eval` / `baseline_eval` 默认也是 14，改了训练就要同步改评估，
> 否则蓝车在 A 间距训、在 B 间距评，payoff 失真。要复现旧紧凑布局用 `--adv-min-gap 0`。

产物在 `logs/psro_runs/<时间戳>_seed42/`：
- `blue_pool/psroNN_blue_team.pt`：每轮一个 BLUE 团队（含 2 actor + 共享 critic + 训练时 σ_red）
- `payoff_table.json`：收益张量 + 各轮 σ_red/σ_blue + （方阵轮的）R_-
- `tb_logs/`：TensorBoard

监控：`tensorboard --logdir logs/psro_runs/`

### Phase 3 —— JPC / 泛化交叉评估（权威诊断）

```bash
python -m training.psro.jpc_eval \
    --red-pool-dir logs/red_pool/ckpts \
    --blue-pool-dir logs/psro_runs/<时间戳>_seed42/blue_pool \
    --scenario scenario_01_basic.json \
    --n-eval 5 \
    --output-dir logs/psro_runs/<时间戳>_seed42/jpc
```

产物：`adv_effectiveness_heatmap.png` / `coop_success_heatmap.png` /
`blue_collision_heatmap.png` / `payoff_table.json` / `jpc_report.json`。

**读 `jpc_report.json`（PSRO 池，重点看泛化段）**：
- `generalization.best_single_blue_worstcase`：越高越好（存在对所有 RED 都强的通用对手）。
- `generalization.uniform_mix_worstcase`：BLUE 池均匀混合的最坏情况有效性，混合应抬高下界。
- `generalization.per_blue[*].worst_eff`：单个 BLUE 的最坏情况鲁棒性。
- `generalization.per_blue[*].gen_gap`：>0 越大越过拟合训练 σ_red；理想接近 0。
- `jpc_R_minus`：**非配对池忽略**（`jpc_R_minus_valid_as_jpc=false`）。

**做 InRL 基线 R_- 对比时**加 `--paired-pool`，且 BLUE 池须是"blue_i 专攻 red_i"的配对池。

### 元策略 γ 调参速查

| `--prd-gamma` | 行为 | 何时用 |
|---|---|---|
| 0.1 | 最强剥削（σ 集中到最优响应） | 想最大化对单一最难 RED 的攻击力 |
| 0.2 | 折中（默认） | 起步推荐 |
| 0.4 | 最稳鲁棒（保留弱策略权重） | `gen_gap` 偏高 / 想要跨 RED 更稳 |

---

## 6. 指标字典（记录什么、含义、想要的方向）

| 指标 | 出处 | 含义 | 期望方向 |
|---|---|---|---|
| `coop_success_rate` | payoff 主键 | 合作团队全员到点率 | BLUE 想↓ |
| `adv_effectiveness = 1 − coop_success` | meta_solver 输入 | BLUE 视角收益 | BLUE 想↑ |
| `blue_collision_rate` | 副指标 | 蓝车自毁率（自适应自碰要压低） | ↓（真实对手不该靠自杀式撞击） |
| `blue_threat_window_pct` | 副指标 | 红车处于蓝车前向威胁窗口的步数占比 | ↑（施压有效） |
| `red_relative_slowdown` | 副指标 | 红车受威胁时相对自由流的减速幅度 | ↑（迫使减速） |
| `avg_threat_score` | 副指标 | 每步威胁分（exp(−TTC/τ) 求和） | ↑ |
| `per_blue.worst_eff` | 泛化报告 | 对最难 RED 的有效性 | ↑（鲁棒） |
| `per_blue.gen_gap` | 泛化报告 | 训练分布 − 全池有效性 | →0（不过拟合） |
| `jpc_R_minus` | 仅配对池 | 论文 JPC 过拟合度 | ↓（仅 InRL 基线对比时看） |

训练期 TensorBoard（`train_blue_oracle` 写）：`coop_success_rate / blue_collision_rate /
blue_threat_window_pct / avg_threat_score / self_collision_coef / reward_base_safe /
reward_threat / lr_actor`，按 `psro{iter}/...` 分组。

---

## 7. 验收清单（对应设计文档 M1–M6）

- [ ] M1：RED 池 ≥ 3，每个 `coop_success > 80%`
- [ ] M2：`psro_runner` 单轮跑通（冒烟：`--psro-iters 1 --inner-updates 20`）
- [ ] M3：完整 5 轮 + uniform 跑通
- [ ] M4：PRD（γ=0.2）完整 5 轮，σ_red 不剧烈震荡
- [ ] M5：`jpc_eval` 出热图 + 泛化报告
- [ ] M6：PSRO BLUE 池 `worst_eff` 显著高于 InRL 基线 / `gen_gap` 显著更小（或配对池 R_- 下降 >20%）
