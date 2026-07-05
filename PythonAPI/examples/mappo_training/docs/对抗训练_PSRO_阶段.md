# 对抗训练 PSRO —— 阶段总结

**日期**：2026-06-25
**主题**：在 CARLA 0.9.16 / Town02 直道上，用 **PSRO（Policy-Space Response Oracles）+ MAPPO** 训练可泛化的对抗智能体团队
**配套文档**：
- 设计计划：[对抗训练_MAPPO_PSRO_设计计划.md](对抗训练_MAPPO_PSRO_设计计划.md)
- 实现评审与用法：[对抗训练_PSRO_实现评审与用法指南.md](对抗训练_PSRO_实现评审与用法指南.md)
- 首次运行诊断 + 修改记录：[对抗训练_PSRO_首次运行问题诊断_20260622.md](对抗训练_PSRO_首次运行问题诊断_20260622.md)
- 论文依据：Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent RL*, NeurIPS 2017

---

## 0. 一句话总览

PSRO 的全套代码框架（外层种群调度 + 内层 MAPPO oracle + 元策略求解 + 收益张量 + JPC 交叉评估）**已实现并能端到端跑通**。经过三轮关键修复（生成间距、对抗车自撞不全局终止、威胁奖励重塑），**最致命的 "payoff 信号饱和" 问题已被攻克**——PSRO 元博弈第一次真正运转起来（σ_red 自动集中到最难攻击的红队）。当前剩余的核心难题是**蓝车仍倾向 "靠撞取胜" 而非 "无碰撞挡道"**，下一步用终局目标奖励（优化 B）+ 提升红队池多样性来解决，然后进入完整 5 轮 + JPC 验收。

---

## 1. 任务背景与目标

| 项 | 设定 |
|---|---|
| 场景 | CARLA 0.9.16，Town02 单车道直道，固定 **2v2** |
| RED 团队（红车 / 合作方） | 2 个 agent，用 MAPPO 预训练好后**全程冻结**，目标是开到终点 |
| BLUE 团队（蓝车 / 对抗方） | 2 个 agent，用 MAPPO + CTDE 训练，目标是**阻止红车到达终点** |
| 车辆 | Tesla Model 3；观测 19 维（8 基础 + 11 雷达）；动作 3 维 `[throttle, steer, brake]` |
| 设计目标 | 对抗策略要**可泛化、队内协同、不自毁**（不能靠自杀式撞击取胜） |

**为什么用 PSRO 而不是直接 InRL**：直接对单一红队训练对抗策略（InRL）会过拟合到那个特定对手，换一个红队就失效。PSRO 让蓝队对**红队的概率分布 σ_red** 求最优响应，迫使其学到跨对手稳健的策略。

---

## 2. PSRO 训练方式（核心方法）

### 2.1 分层架构：外层 PSRO 调度 + 内层 MAPPO oracle

PSRO 与 MAPPO **不在同一层级**，而是嵌套使用：

```
┌──────────── PSRO 外层（种群调度，psro_runner.py）────────────┐
│  RED 池 = { red_1, ..., red_K }   ← 冻结的合作策略，事先组建     │
│  BLUE 池 = { blue_1, blue_2, ... } ← PSRO 每轮新增一个对抗团队   │
│  元策略 σ_red = 在 RED 池上的概率分布                            │
│                                                                │
│  for psro_iter in range(N):                                    │
│      ① 内层 MAPPO 训一个新 BLUE oracle（对 σ_red 求最优响应）     │
│      ② 用新 BLUE 跟每个 RED 交叉评估，填充经验收益张量            │
│      ③ 元策略求解器更新 σ_red（让下个 BLUE 去专攻最难的红队）      │
│      ④ 记录 JPC / 泛化指标 + TensorBoard                        │
└────────────────────────────────────────────────────────────────┘
```

| 层级 | 作用 | 代码落点 |
|---|---|---|
| 外层 PSRO | 决定"BLUE 跟谁训"，防过拟合到单一 RED | `training/psro/psro_runner.py` |
| 内层 MAPPO | 决定"BLUE 队内怎么协同"，产出一个对抗 oracle | `training/psro/train_blue_oracle.py` |
| 元策略求解 | uniform / PRD（投影复制动态），决定 σ_red | `training/psro/meta_solver.py` |
| 收益张量 | 存所有 (red, blue) 配对的多指标收益 | `training/psro/payoff_table.py` |
| 种群管理 | RED/BLUE 池加载、采样、快照 | `training/psro/population.py` |
| JPC 交叉评估 | |R|×|B| 完整矩阵 + 泛化报告 | `training/psro/jpc_eval.py` |
| 红队池组建 | 收编已有 ckpt 或多 seed 重训 | `training/psro/build_red_pool.py` |
| 基线判定 | 蓝车换良性驾驶，判定"红车本能否过" | `training/psro/baseline_eval.py` |

### 2.2 内层 MAPPO oracle 的关键设计（CTDE）

- **每个 episode 从 σ_red 采样一个冻结红队**进入环境，蓝队对这个分布训练；
- **BLUE 团队独立一个 CTDE critic**（`AgentConditionedCritic`），看到全员观测但**只回归 BLUE 的 return**——红队只作为环境的一部分进入 state（论文 §三注 2：不能用一个集中 critic 同时覆盖红蓝两个对立目标，否则会被平均掉）；
- **真实性约束**（让蓝车行为像真人，不靠 bug 取胜）：
  - **KL 锚定**：蓝车 actor 的策略对"自然驾驶 base 策略"做 KL 约束，保持自然驾驶外观；
  - **威胁奖励（threat reward）**：奖励蓝车对红车施加前向压制；
  - **自适应自碰惩罚（PID-I）**：动态调节自撞惩罚系数，压制自毁行为。

### 2.3 元策略与收益口径

- **主指标**：`adv_effectiveness = 1 − coop_success_rate`（蓝车视角的收益，越高说明越能阻止红车到点）；
- **PRD 求解器**：零和近似 `U_red = −U_blue`，在单纯形上做投影复制动态，带 **γ-探索下界**（`--prd-gamma`，0.1 最剥削 / 0.2 折中 / 0.4 最鲁棒）；
- **JPC R_-（过拟合度量）**：注意本项目 RED 池与 BLUE 池**无索引配对关系**，论文式 R_- 不能直接套用。已改用**对 PSRO 始终有意义的泛化指标**：
  - `worst_eff = min_i M[i,j]`：单个蓝队对**最难红队**的有效性（最坏情况鲁棒性，越高越好）；
  - `gen_gap`：训练分布有效性 − 全池平均（>0 越大越过拟合训练 σ_red，理想→0）。

### 2.4 端到端训练流程与命令

> 环境：仓库根 pipenv；在 `PythonAPI/examples/mappo_training/` 下以模块方式运行；先启动 `./CarlaUE4.exe -quality-level=Low`。

```bash
# Phase 0：组建 RED 池（≥3，每个需 coop_success>80%）
python -m training.psro.build_red_pool --from-ckpts <best1.pt> <best2.pt> ... --out logs/red_pool/ckpts

# Phase 2-3：PSRO 主流程（产出 BLUE 池 + 收益表）
python -m training.psro.psro_runner \
    --red-pool-dir logs/red_pool/a0617_pool \
    --scenario scenario_01_basic.json --adv-min-gap 14 \
    --psro-iters 5 --inner-updates 30 --rollout-episodes 16 --n-eval-episodes 5 \
    --meta-solver prd --prd-gamma 0.2 --save-dir logs/psro_runs

# Phase 3：JPC / 泛化交叉评估（权威诊断）
python -m training.psro.jpc_eval \
    --red-pool-dir logs/red_pool/a0617_pool \
    --blue-pool-dir <RUN>/blue_pool --adv-min-gap 14 --n-eval 5 --output-dir <RUN>/jpc
```

**关键约束**：`--adv-min-gap` 训练与评估必须用同值（默认 14），否则蓝车在 A 间距训、B 间距评，payoff 会失真。

### 2.5 当前实现 vs 论文 PSRO —— 单边 PSRO（红队冻结）

本项目实现的是论文 PSRO 的一个**特例：单边（one-sided）PSRO，红队那一侧冻结、不进化**。理解这一点对解读结果很关键。

**论文 PSRO（Lanctot et al. 2017）** 是 Double Oracle / fictitious play 的推广，**双边对称、两边的池都在长大（共演化）**：

```
两边各有一个策略池：Π_red, Π_blue
for iter:
    对每个玩家 p：固定对手元策略 σ_{-p}（对手池上的分布）
                  用 RL 训一个对它的最优响应 BR_p（oracle），加进 Π_p
    用 meta-solver（纳什 / α-Rank / PRD）在新的 |Π_red|×|Π_blue| 收益矩阵上重算 σ_red, σ_blue
最终解 = 池上的元纳什混合（meta-Nash mixture）
```

**逐条对照**：

| 维度 | 论文 PSRO | 当前实现 | 代码落点 |
|---|---|---|---|
| 红池 | **每轮也新增一个红队 BR**（共演化） | **冻结，不增长**（单边） | `RedTeamPool.from_dir`，全程只读 |
| 蓝池 | 每轮新增一个蓝队 BR | ✅ 一致 | `blue_pool.add(...)` 每轮一个 |
| oracle | 对对手元策略求 BR | ✅ 一致（对 σ_red 求 BR） | `train_blue_oracle` |
| meta-solver | 纳什 / PRD / α-Rank（一般和） | PRD + **零和近似** `U_red=−U_blue` | `meta_solver.PRDSolver` |
| 最终解 | 双边元纳什混合 | 蓝池 + σ_blue 混合（红侧固定） | `payoff_table.json` 的 σ_blue |

**两个澄清（针对常见误解）**：

1. **不是"逐一训练"**：蓝车 oracle 不是依次对每个红队各训一遍，而是对**混合分布 σ_red** 求最优响应——每个 rollout episode `np.random.choice(len(red_pool), p=σ_red)` **按概率随机抽一个红队**来打。一个 oracle 的 30 个 update 里会按概率见到各种红队。
2. **σ_red 不是一直均匀**：仅第 0 轮均匀，之后 PRD 把权重**集中到最难攻的红队**（实测 `[0.88, 0.04, 0.04, 0.04]`），逼下一个蓝 oracle 去专攻最硬的对手。

**为什么现在是单边**：红蓝交替共演化在设计文档里被列为 **Phase 4（可选，研究深度方向）**，需先保证蓝队这条线跑通、JPC R_- 稳定下降后再做。当前阶段做的是"**对一个固定红队种群求稳健最优响应**"，还不是完整双边博弈求解。共演化的已知风险是一般和非传递性导致 σ 震荡不收敛。

> 关于零和近似：论文严格是一般和、用纳什/α-Rank 求解；这里用 `U_red=−U_blue` 配 PRD。对抗驾驶大致零和（蓝车收益≈红车失败），作为近似可用，但不是论文原味。

### 2.6 蓝车最终的策略是几个 —— 产物语义

**数量 = `--psro-iters`**。跑 5 轮 → **5 个蓝队**，存在 `<RUN>/blue_pool/psro00_blue_team.pt … psro04_blue_team.pt`。

- **每个"蓝队"是团队捆绑，不是单个网络**：2 个 actor（agent 2、3）+ 1 个共享 CTDE critic + 训练时的 σ_red 快照 + stats。
- **K 个蓝队彼此独立**：每个 oracle 都**从"自然驾驶基策略"初始化**（默认取 `red_pool[0]` 的合作策略当起点），**不是**在上一个蓝队基础上接着练。它们是针对**不同 σ_red 分布**得到的不同最优响应。
- 单个蓝 oracle 训法：MAPPO + CTDE，30 个 PPO update；每 episode 按 σ_red 抽红队收 rollout（带 death-masking）；奖励 = `base_safe + threat_coef·威胁分 + 自适应自碰罚`，actor 对自然驾驶做 KL 锚定；共享 critic 只回归蓝队 return。

**"最终对手"到底是哪个**——严格说 PSRO 的产物**不是单一策略，而是"蓝池 + 元策略 σ_blue"这个混合体**。两种用法：

| 用法 | 含义 | 取自 |
|---|---|---|
| **混合部署** | 按 σ_blue 概率随机抽一个蓝队上场（理论解，抬高最坏情况下界） | `payoff_table.json` 的 `sigma_blue` |
| **挑单个最鲁棒** | 用 `worst_eff = min_i M[i,j]`（对最难红队的有效性）选单个蓝队 | `jpc_report.json` 的 `best_single_blue_worstcase` |

---

## 3. 最近的关键修改（按时间顺序）

这一阶段的工作主线是**逐层定位并解除 "payoff 信号饱和" 这个阻断 PSRO 的致命问题**。

### 修改①（06-22）几何修复：拉开生成间距 `--adv-min-gap`

**问题**：`scenario_01_basic` 红车 y=176/210，蓝车被自动塞在中间（y≈189/197），4 辆车（每辆 ~4.7m）挤在单车道 34m 内，中间车距仅 ~3.3m，一启动就追尾。
**修改**：给 `make_adversarial_positions` 增加 `min_gap` 参数，`min_gap>0` 时"最前红车锚定不动、整列向后等间距铺开"，保证相邻间距 ≥ min_gap。经 `prepare_eval_scenario` 透传，三个入口（runner/jpc/baseline）统一 `--adv-min-gap`，默认 **14m**。

### 修改②（06-23）基线判定：定位真正根因

新增 `baseline_eval.py`，把蓝车换成**良性驾驶**（用红队自己的合作策略开向终点 = 4 车自然车流），判定 coop_success≡0 到底是 (A) 几何过挤/红车策略不足，还是 (B) 对抗蓝车自杀式局部最优。
**结果**（逐红队拆开才是真相）：

| RED | coop_success | 解读 |
|---|---|---|
| A0608 | **0%** | 策略坏：无对抗车自己 30 步内就撞 |
| A0610 | 20% | 策略勉强 |
| A0617 | **100%**（827 步零碰撞） | 策略好：4 车自然车流一路到终点 |

三条独立根因：**① 间距修复已生效，几何不再是瓶颈；② RED 池质量参差（3 个里 2 个不合格）；③【最致命】"自杀"是统治性策略，根在环境终止规则**——任一 agent 碰撞→全局终止，蓝车 35 步自杀就把红车 827 步的旅程腰斩，coop_success=0，payoff 处处=1.0。

### 修改③（06-23）★核心：对抗车自撞 → 不全局终止回合

**这是当前唯一能根治 payoff 饱和的改动。** 改了 5 处：

| 文件 | 改动 |
|---|---|
| `carla_multi_agent_env.py` | 构造参数 `adversary_collision_ends_episode`（默认 True 保旧行为）；为 False 时**仅合作车(红)严重碰撞才全局终止**，蓝车自撞只 despawn 那辆蓝车、红车继续；despawn 在 `info` 记录完碰撞/terminated **之后**执行（保证奖励、自碰统计、GAE 终止都正确） |
| `train_blue_oracle.py` | rollout 加 **death-masking**（已终止的车不再采样假 transition），收尾改为"红车全终止 或 蓝车全终止" |
| `jpc_eval.py` / `baseline_eval.py` | 同 death-masking；eval 收尾改为"红车全终止"（让红车跑到底才能量准 coop_success） |
| 三入口 | 加 CLI `--adversary-collision-ends-episode`，PSRO/eval 路径**默认 False（新行为）** |

**激励方向被纠正**：蓝车不再能靠自杀免费终止回合，"不撞地持续挡道"才能累积更多 threat 步且无撞毁罚 → return 更高。

### 修改④（06-23）RED 池重建

旧池里 A0608/A0610 不合格已弃用。用唯一合格的 **A0617** 的 best + 稳定好区快照重建 `logs/red_pool/a0617_pool = {best(819), 900, 1000, 1100}`（1250/1410 因退化被剔除）。
> 局限：全来自同一 seed 快照，多样性偏低，适合"验证 payoff 去饱和"，做强种群结论仍需混入不同 seed 的合格红车。

### 修改⑤（06-24）优化 A：威胁奖励重塑（"奖励安全距离逼停" 取代 "奖励贴脸即撞"）

**问题**：旧 `threat = exp(−TTC/τ)` 在 TTC→0（即将相撞）时最大，等于直接奖励"贴到撞距离"。
**重塑**（`compute_blue_threat_scores`）：

```
威胁分 = band_kernel(纵向距离) × impede(红车被压慢的程度)
```
- `band_kernel`：[threat_near, threat_radius] 内的钟形核，峰在安全甜点；**纵向 < threat_near（撞距离）直接 0 分** → 怼上去不再有正回报；
- `impede = clamp((red_desired_speed − 红车前向速度)/red_desired_speed, 0, 1)`：红车被压得越慢越高 → 奖励"**持续把红车压慢**"而非"逼近瞬间"。

**默认值调整**：`--threat-coef` 2.0→1.0；`--target-blue-collision-rate` 0.05→0.03；新增 `--self-collision-coef-max`（默认 20）、`--threat-near`（默认 3.0m）。

### 其他配套修复

- **`--init-log-std` 静默失效已修**：load 红队 state_dict 会把蓝车探索方差覆盖成红队收敛后的极小值，已在 load 后显式恢复，使蓝车起步有探索；
- **JPC 语义修正 + 泛化指标补充**：见 §2.3，R_- 仅在配对池（InRL 基线）下才按论文解读，否则看 `worst_eff/gen_gap`。

---

## 4. 目前取得的成果

### 4.1 框架层面（已完成）

- ✅ **PSRO 代码实现跑通**：`training/psro/` 9 个模块（runner / oracle / meta_solver / payoff_table / population / jpc_eval / build_red_pool / baseline_eval），全部 `py_compile` + import 通过，数值自检符合预期（PRD 正确集中到最难对手，JPC 对角强矩阵 R_-=0.625、均匀矩阵 R_-=0）；
- ✅ **完整可复现的用法链路**：组建红池 → PSRO 主流程 → JPC 评估，三入口参数对齐；
- ✅ **基线判定工具**：能区分"红车本能否过"与"被蓝车撞停"。

### 4.2 实验层面：payoff 成功去饱和

修改③（对抗车自撞不全局终止）实跑后，PSRO 元博弈**第一次真正运转**：

| 信号                    | 修改前（06-22，10 轮空转）      | 修改后（06-24）                             |
| --------------------- | ---------------------- | -------------------------------------- |
| `coop_success`（按红队）   | 30 格**处处 0**           | **[1.0, 0.2, 0.0, 0.0]** 出现区分度         |
| `σ_red`               | [.33,.33,.33] **十轮不动** | **[0.88, 0.04, 0.04, 0.04]** 自动集中到最难红队 |
| `blue_collision_rate` | 全 1.0（训练前就撞）           | RED0 降到 **0**                          |
| `avg_threat_score`    | ~0.001（几乎无前向威胁）        | 优化 A 后升到 **~0.4**（RED1/2/3）            |

**意义**：死亡终止漏洞堵上后，红队之间出现了攻击难度区分度，元求解器有了梯度，σ_red 自动把权重集中到最难攻的 RED0（best）——这正是对抗 PSRO 期望的行为。**这是从"框架空转"到"博弈运转"的质变。**

---

## 5. 当前存在的问题

1. **【主要】蓝车仍倾向"靠撞取胜"**：优化 A 后威胁分上来了（0.4），但能赢的红队（RED1/2/3）仍是 `coop_collision ≈ blue_collision ≈ 1.0`——红蓝一起撞。根因是红队在 **2 车干净世界**训练、从没学过跟车刹车，被蓝车贴脸硬挡就追尾；蓝车自己也撞毁，违背"不自毁"目标。
2. **【机制】训练奖励是 threat 代理量**，不直接含"红车是否被拦下(coop_success)"，存在 train/eval 口径错配风险。
3. **【数据】RED 池多样性低**：当前 `a0617_pool` 全来自同一 seed/run 的快照，行为差异有限，PSRO 种群效应偏弱，难以支撑强泛化结论。
4. **【指标】JPC R_- 暂为 NaN**：当前为 4×2 非方阵且非配对池，需完整跑满 + 构造 InRL 配对基线池才能做论文式对比。

---

## 6. 下一步计划

### 近期（解除"靠撞取胜"，1-2 周）

1. **优化 B：终局目标奖励**（把目标直接写进梯度）——给蓝车加终局奖励：红车超时/未到 → +R，红车到点 → −R。让"拦住红车"成为直接梯度信号，而非仅靠 threat 代理量。
2. **冒烟验证优化 A/B 效果**：`--psro-iters 2 --inner-updates 30`，判读标准：`blue_collision_rate` 从 0.8~1.0 **降到 <0.3**，同时 coop_success 仍有区分度、`red_relative_slowdown` 为正且较大（蓝车靠**逼停/超时**而非撞停取胜）。
3. **（按需调参）**：若蓝车变消极 → 上调 `--threat-coef`（1.0→1.5）；若仍大量自撞 → 收紧 `--target-blue-collision-rate`（0.03→0.02）/ 上调 `--self-collision-coef-max`。

### 中期（提升红队质量与多样性）

4. **修复红队"不会跟车"的根因**：考虑在红队预训练中引入多车车流（而非 2 车干净世界），或混入不同 seed 重训的合格红车，提升 RED 池真实多样性。
5. **RED 池二次验收**：对 `a0617_pool` 的 900/1000 快照单独跑 `baseline_eval` 确认 4 个都 ≥80%。

### 验收（对应设计文档 M4-M6）

6. **完整跑 5 轮 PSRO + PRD（γ=0.2）**，确认 σ_red 不剧烈震荡（M4）；
7. **`jpc_eval` 出热图 + 泛化报告**（M5）；
8. **M6 验收**：PSRO 产出的 BLUE 池 `worst_eff` 显著高于 InRL 基线 / `gen_gap` 显著更小（或构造配对池后 R_- 下降 >20%）。

### 可选（研究深度方向）

9. **Phase 4 共演化**：红蓝交替训练（仅在 Phase 0-3 跑通、R_- 稳定下降后再做），注意一般和非传递性导致的 σ 震荡风险。

---

## 7. 里程碑进度

| 节点 | 标志 | 状态 |
|---|---|---|
| M0 | 设计文档 review | ✅ 完成 |
| M1 | RED 池 ≥3，每个 coop_success>80% | ⚠️ 部分（a0617_pool 已建，900/1000 待复核；多样性偏低） |
| M2 | `psro_runner` 单轮跑通 | ✅ 完成（payoff 已去饱和） |
| M3 | 完整 5 轮 + uniform | ⏳ 待跑（先解"靠撞取胜"） |
| M4 | PRD（γ=0.2）5 轮，σ_red 不震荡 | ⏳ 待跑 |
| M5 | `jpc_eval` 出热图 + 泛化报告 | ⏳ 工具就绪，待完整数据 |
| M6 | BLUE 池 worst_eff 显著优于 InRL 基线 | ⏳ 验收节点 |
| M7（可选） | Phase 4 共演化 | ⏳ 后期 |

**当前位置**：M2 已过（框架跑通 + payoff 去饱和），正卡在 M3 前的"蓝车靠撞取胜"问题上，优化 B 是打通后续验收的关键。

---

## 8. 复现命令速查

```bash
# 0. 启动模拟器
./CarlaUE4.exe -quality-level=Low

# 1. PSRO 冒烟（验证优化效果）
python -m training.psro.psro_runner --red-pool-dir logs/red_pool/a0617_pool \
    --scenario scenario_01_basic.json --adv-min-gap 14 \
    --psro-iters 2 --inner-updates 30 --rollout-episodes 8 --n-eval-episodes 5 \
    --meta-solver prd --prd-gamma 0.2 --save-dir logs/psro_runs

# 2. PSRO 正式（5 轮）
python -m training.psro.psro_runner --red-pool-dir logs/red_pool/a0617_pool \
    --scenario scenario_01_basic.json --adv-min-gap 14 \
    --psro-iters 5 --inner-updates 30 --rollout-episodes 16 --n-eval-episodes 5 \
    --meta-solver prd --prd-gamma 0.2 --save-dir logs/psro_runs

# 3. JPC / 泛化评估
python -m training.psro.jpc_eval --red-pool-dir logs/red_pool/a0617_pool \
    --blue-pool-dir <RUN>/blue_pool --adv-min-gap 14 --n-eval 5 --output-dir <RUN>/jpc

# 4. 基线判定（蓝车=良性驾驶）
python -m training.psro.baseline_eval --red-pool-dir logs/red_pool/a0617_pool \
    --scenario scenario_01_basic.json --blue-mode natural --adv-min-gap 14 --n-eval 5 \
    --output-dir logs/psro_runs/baseline_$(date +%Y%m%d)

# 监控
tensorboard --logdir logs/psro_runs/
```

| 关键开关 | 默认 | 说明 |
|---|---|---|
| `--adv-min-gap` | 14 | 相邻车最小纵向间距；训练与评估须一致 |
| `--prd-gamma` | 0.2 | PRD 探索下界（0.1 剥削 / 0.4 鲁棒） |
| `--threat-coef` | 1.0 | 威胁奖励权重（优化 A 从 2.0 下调） |
| `--threat-near` | 3.0 | 撞距离阈值，纵向 < 此值威胁分=0 |
| `--target-blue-collision-rate` | 0.03 | 自适应自碰 PID 目标自撞率 |
| `--self-collision-coef-max` | 20 | 自碰惩罚系数上限 |
| `--adversary-collision-ends-episode` | False(新) | 加开关恢复旧行为做消融 |
