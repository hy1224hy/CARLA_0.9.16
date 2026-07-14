# 对抗行为科学度量、奖励设计与反 Reward-Hacking 总设计 + 分阶段实验计划

- **创建日期**: 2026-07-05
- **状态**: 待评审
- **适用范围**: 2红+1蓝对抗快验 → 双蓝 MAPPO → PSRO 全流程
- **配套代码**: `training/train_adversarial_fixed_scenario.py`、`training/train_multi_adversarial_kl_risk.py`、`training/train_no_rllib_mappo_robust_v2.py`

## 变更记录（流程追踪）

| 日期 | 版本 | 变更内容 | 原因 |
|---|---|---|---|
| 2026-07-05 | v1.0 | 初版：度量框架 + 奖励设计 + 反hacking + 五阶段计划 | 2+2 训练未见对抗收益，决定退到 2红+1蓝 重建根基 |
| 2026-07-05 | v1.1 | 新增附录 A：各阶段运行指令（参数与各脚本 argparser 逐一核对） | 便于按计划直接执行 |
| 2026-07-05 | v1.2 | 决策：红车基础奖励 v2 **原样沿用**（不改）；蓝车对抗奖励按 §2.3 执行 | 红车须是固定的"正常驾驶"基准以保证可比性；对抗奖励与红车奖励正交 |
| 2026-07-05 | v1.3 | 新增 §2.5：对抗奖励抽成**单一共享模块**；决策保留 fixed_scenario 作为 §2.3 的快验代理 | 发现奖励公式在 fixed_scenario 与 train_blue_oracle 复制粘贴，改一处会分叉；PSRO 真正落点是 train_blue_oracle |
| 2026-07-05 | v1.4 | **落地**：① 补齐 `train_blue_oracle.py` PSRO TB 缺口（PPO 健康值 + 红车结局拆分 + 减速代理）；② 新建 `utils/adversarial_metrics.py`（归因工具，CARLA 解耦，附单测冒烟）；③ 新建 `training/adversarial_reward.py`（§2.3 单一真相源 `AdversarialRewardComputer`）；④ 新建 `training/train_2red_1blue_ppo.py`（2+1 PPO，`--num-adversarial 2` 即升 2+2） | 执行 §2.5 实现次序 metrics→reward→脚本；先让新脚本吃上正确奖励并端到端验证通过，旧两处（fixed_scenario / train_blue_oracle）迁移到共享模块留待下一轮 |

> 后续所有设计变更在此表追加一行，并在正文对应小节标注 `[vX.X 变更]`。

> **v1.4 实现状态（§2.5）**：`adversarial_reward.py` 已成为单一真相源，但目前仅 `train_2red_1blue_ppo.py` 接入。`train_adversarial_fixed_scenario.py:479` 与 `train_blue_oracle.py:296` 仍是内联复制的旧公式（`base_safe + threat_coef*threat + self_coll`，无 r_impact/门控/封顶）——**待迁移**，迁移前二者与新脚本的奖励口径不一致。

---

## 0. 术语与约定

以代码为准（`train_adversarial_fixed_scenario.py:164`）：

| 术语 | 含义 | 代码对应 |
|---|---|---|
| **红车 (RED)** | 基础驾驶车，预训练后**冻结**，被攻击方 | `cooperative_ids`，deterministic action |
| **蓝车 (BLUE)** | 对抗车，**在训**方 | `adversarial_ids`，KL锚定 + 威胁奖励 |
| 基础策略 | robust_v2 训出的红车驾驶策略 | `--pretrained-ckpt` 加载 |
| 红车池 | PSRO 中作为蓝方对手的多个红车策略 | `psro/build_red_pool.py` |

---

## 1. 如何科学地衡量对抗行为

### 1.1 第一性原则：三个问题必须分开回答

"对抗有效"混淆了三个独立命题，度量必须分层：

1. **结果层（Outcome）**：红车的任务表现变差了吗？
2. **行为层（Behavior）**：蓝车做出了对抗性动作吗？
3. **归因层（Attribution）**：红车变差是蓝车**造成**的吗？

三者缺一不可的原因：
- 只看结果层 → 红车自身失败（本来就 30% 碰撞率）会被误记为对抗收益；
- 只看行为层 → 蓝车"看起来在干扰"但红车毫发无损（占位不生效）；
- 没有归因层 → 蓝车自杀式撞车也算"成功"，度量直接奖励了 hack。

### 1.2 评估协议：三条件反事实对照

**所有对抗效果评估必须跑三个条件，其余变量全部锁死**（同场景、同起点、同评估种子列表、红蓝均用 deterministic 动作）：

| 条件 | 配置 | 回答的问题 |
|---|---|---|
| **C0** | 只有红车（蓝车不生成） | 红车性能上限；红车自身失败率基线 |
| **C1** | 红车 + 蓝车（蓝车用**预训练权重**正常开车） | "路上多一辆普通车"本身造成多少干扰 |
| **C2** | 红车 + 蓝车（训练后的对抗策略） | 实验处理组 |

**核心定义：对抗收益 = C2 − C1**（配对差值）。

- C1 − C0 = 纯交通效应（不是对抗训练的功劳）；
- C2 − C0 会把交通效应冒算成对抗收益，**禁止用作主结论**；
- fixed_scenario 里 "update 1 ≈ 训练前基线" 的做法是 C1 的雏形，但它是随机采样动作 + 混在训练循环里，正式评估必须用独立评估脚本、deterministic、固定种子。

**统计规范**：
- 每条件 N ≥ 40 episodes，种子列表固定（如 `range(1000, 1040)`），三条件逐种子配对；
- 报告配对差值的 mean ± bootstrap 95% CI，CI 不跨 0 才可声称"有对抗收益"；小样本用 Wilcoxon 符号秩检验做辅助；
- 训练场景之外必须留 ≥1 个 **held-out 场景**（蓝车训练时从未见过的起点/终点组合），检测蓝车是否只会攻击一种几何布局。

### 1.3 指标体系（四层）

#### Layer A — 结果层（红车表现，全部报告 C2−C1 配对差）

| 指标 | 定义 | 备注 |
|---|---|---|
| `Δ success_rate` | 红车全员到达终点的回合占比之差 | 成功/碰撞/超时**三分互斥**，之和恒等于 1（修 robust_v2 文档 R1 的口径污染） |
| `Δ collision_rate` | 红车发生碰撞的回合占比之差 | 需配合 Layer C 归因拆分 |
| `Δ timeout_rate` | 超时回合占比之差 | 蓝车"逼停红车"体现在这里 |
| `Δ time_to_goal` | 仅统计双方条件下都成功的配对回合 | 删失数据不混入均值 |
| `Δ route_completion` | 失败回合的路程完成度之差 | 区分"起步就被搞死"和"临门一脚被搞死" |

#### Layer B — 行为层（蓝车做了什么，只在 C2 上测）

| 指标 | 定义 |
|---|---|
| `threat_window_pct` | 蓝车处于任一红车威胁窗口（前方 0.5~15m、横向 ≤2.5m，与 `is_any_red_under_threat` 同口径）的步数占比 |
| `cut_in_count` | 切入事件数：蓝车从红车侧方进入其前方窗口且横向速度超阈值 |
| `blocking_time` | 阻挡时长：蓝车在红车前方窗口内且自身速度 < 红车期望速度带下限（5 km/h）的累计时间 |
| `min_TTC 分布` | 红蓝间最小 TTC 的分布（报分位数 P10/P50，不只报均值） |
| 蓝车自身状态 | 蓝车均速、自碰率、压线率——验证蓝车不是退化策略 |

#### Layer C — 归因层（红车的失败是否由蓝车造成）★ 最重要

| 指标 | 定义 |
|---|---|
| `attributed_slowdown_rate` | 归因减速事件：红车朝目标闭合速度在 2s 内下降 ≥2 m/s，**且**当时蓝车在其威胁窗口内。对照口径：同样的减速事件在蓝车不在场时的发生率（C1 中测得） |
| `attributed_steer_rate` | 归因避让：红车方向盘突变（\|Δsteer\|>阈值）且蓝车在窗口内 |
| `attributed_collision_rate` | 归因碰撞，按碰撞几何分三类（见下） |
| `attributed_timeout_rate` | 归因超时：红车超时且该回合蓝车威胁窗口占比 ≥ 30% |
| `kamikaze_rate` | 自杀碰撞占比（hack 监测指标，越低越好） |

**碰撞归因三分法**（用碰撞瞬间接触点在双方车体坐标系的方位判定）：

1. **蓝车主动撞击**（蓝车车头接触红车）→ kamikaze，**不给对抗奖励、不计入对抗收益**，计入 `kamikaze_rate`；
2. **诱导碰撞**（红车车头接触蓝车尾/侧，蓝车处于正常车道行为中）→ 有效对抗，最有研究价值；
3. **连带碰撞**（红车因避让蓝车而撞上第三方/路沿，碰撞前 3s 内蓝车在其窗口内）→ 有效对抗。

#### Layer D — 训练诊断层（TB，每 update）

- 奖励分项独立曲线：`reward/base_safe`、`reward/threat`、`reward/impact`（新增）、`reward/self_collision`（已有大半）；
- 分项**占比**曲线：`reward/threat_share = |threat项| / Σ|各项|`——权重定标和 hack 监测都靠它；
- `ppo/kl_anchor`、`ppo/approx_kl`、`ppo/clip_frac`、entropy、explained_variance（已有）；
- hack 仪表盘指标（见 §3.3）。

### 1.4 模型选择准则（best.pt）

当前 `best = max(1 − coop_success_rate)` 有两个漏洞：把红车自身失败算进来、会选中自杀型策略。改为**约束满足前提下的归因收益最大化**：

```
best = max( attributed_impact_score )
       s.t. blue_collision_rate ≤ 5%  且  kamikaze_rate ≤ 10%
attributed_impact_score = Δsuccess_rate 的滑动窗口估计（以归因事件率作平滑辅助）
```

不满足约束的 update 不参与 best 评选。

---

## 2. 对抗奖励函数设计

### 2.1 设计原则

1. **奖励是归因指标的稠密代理**：训练奖励的每一项都必须能回答"它是 Layer C 哪个指标的代理"。答不上来的项就是 hack 温床。
2. **目标与约束分离**：蓝车"自己不出事"是**约束**（用自适应系数=拉格朗日对偶变量），"干扰红车"才是**目标**。两者混成固定权重加权和会在训练中互相淹没（R-3 的教训）。
3. **稠密项必须门控 + 有界**：任何每步可得的正奖励，要么乘以"红车实际响应"的指示函数，要么做势能形（PBRS），且设每回合封顶。
4. **KL 锚定是行为合理性信任域**：`β_KL·KL(π_blue ‖ π_pretrained)` 把蓝车限制在"像正常车"的策略邻域内，防止物理上有效但交通上荒谬的退化解。保留。
5. **量级用数据定，不拍脑袋**：先跑 rollout 实测各分项每步量级，再定权重（§2.4）。

### 2.2 当前奖励解析（fixed_scenario / kl_risk 版）

```
r_blue = base_safe + threat_coef(2.0) × threat_score + λ_t × collision_raw
actor_loss += β_KL(0.1) × KL(π_blue ‖ π_anchor)
λ_t 自适应：blue碰撞率 > 5% 则 ×1.2，否则 ×0.95，clip [0.1, 10]
```

评价：
- ✅ 自适应 λ_t 是正确的对偶上升结构；✅ KL 锚定方向正确；✅ 分项已进 TB。
- ❌ `threat_score` 是纯几何位置分——蓝车悬停在窗口内不影响红车也持续拿分（占位刷分漏洞，§3.1-H2）；
- ❌ 没有任何一项与"红车实际被影响"挂钩——奖励与 Layer C 完全脱钩；
- ❌ `base_safe` 与 threat 的量级关系未定标，可能重演"好好开车梯度压倒对抗梯度"（R-3）。

### 2.3 v-next 奖励结构

```
r_blue(t) = w_I · r_impact(t)          # 稀疏·归因事件（主目标）
          + w_T · r_threat_gated(t)     # 稠密·门控引导（学习脚手架）
          + w_B · base_safe(t)          # 小权重·基础驾驶能力
          − λ_t · c_self(t)             # 自适应约束（蓝车自身安全）
actor_loss += β_KL · KL(π_blue ‖ π_anchor)   # 行为合理性信任域
```

**r_impact — 稀疏归因事件奖励**（与 Layer C 指标一一对应）：

| 事件 | 触发条件（与 §1.3 Layer C 同口径） | 奖励 | 每回合上限 |
|---|---|---|---|
| 归因减速 | 红车闭合速度 2s 内降 ≥2 m/s 且蓝车在窗口内 | +5 | 3 次 |
| 归因避让 | 红车 steer 突变且蓝车在窗口内 | +3 | 3 次 |
| 诱导/连带碰撞 | 碰撞归因判定为第 2/3 类 | +30 | 1 次 |
| 归因超时 | 回合超时且威胁窗口占比 ≥30% | +15 | 1 次 |
| kamikaze | 碰撞归因判定为第 1 类 | **0**（不给分，c_self 照罚） | — |

**r_threat_gated — 门控稠密引导**：

```
r_threat_gated(t) = threat_score(t) × 1[红车正在响应(t)]
红车正在响应 := 红车闭合速度在下降 或 |红车steer|突增（滑窗判定）
每回合累计封顶：Σ r_threat_gated ≤ CAP（建议 CAP ≈ 单次诱导碰撞奖励的 2/3，即 ~20）
```

设计理由：纯位置分改成"位置 × 引起反应"，占位不再有收益；封顶保证稠密项永远不能超过稀疏主目标的价值。备选方案是把 threat 做成 PBRS 势能形（`γΦ' − Φ`，telescoping 保证无法刷分），但 PBRS 对"持续施压"给出轻微负奖励，与我们想要的行为不符，故选门控方案，PBRS 作为门控失效时的 fallback 记录在案。

**base_safe — 小权重保留**：防止蓝车丧失基本车辆控制。w_B 取值见 §2.4 定标，原则是 base_safe 在**交互区内**的量级不得超过门控威胁分。

**c_self — 约束（沿用自适应机制并扩展）**：自碰 + 压线 + 逆行都计入，λ_t 对偶更新规则不变，预算目标 `blue_collision_rate ≤ 5%`。

**明确不采用的项**：
- **全量零和项 `−r_red`**：红车的 shaped reward 含 13 项塑形噪声，取负后把噪声也传给蓝车。归因事件奖励（r_impact）已经是"零和的干净子集"，足够。
- **红车 success 的大额负奖励**：会制造强烈的"不惜一切阻止到达"梯度，是 kamikaze 的直接诱因。阻止到达的收益已由归因超时/碰撞事件承载。

### 2.4 权重定标流程（不拍脑袋）

1. 用当前 ckpt 跑 20 episodes，记录各分项每步量级分布（TB 已有大半）；
2. 定标目标：交互区内（蓝车在窗口内的步）`E|w_T·threat_gated| ≈ 1~2 × E|w_B·base_safe|`；全回合 `Σw_I·r_impact 的期望 ≥ Σw_T·threat_gated 封顶值`；
3. 训练全程监控 `reward/*_share` 占比曲线，任何一项占比 >70% 或 <5% 持续 10 个 update → 触发人工检查；
4. 每次调权在本文档变更记录表追加一行。

### 2.5 对抗奖励模块化（单一真相源）

**现状问题**：对抗奖励的"组装"逻辑（`combined = base_safe + threat_coef*threat + self_coll` + 自适应自碰系数更新）目前在多处**复制粘贴**，靠人工维持一致：

- `train_adversarial_fixed_scenario.py:479`（Phase 1 快验）
- `training/psro/train_blue_oracle.py:296`（Phase 4 PSRO 真正训练蓝 oracle 的落点）
- 组件函数（`compute_blue_threat_scores` / `extract_base_driving_safe` / `extract_collision_component`）已从 `train_multi_adversarial_kl_risk.py` 共享，但**组装步没有**。

风险：一旦按 §2.3 只改其中一处，PSRO 拿到的奖励与快验验证的奖励会**分叉**——快验就失去了代理意义。

**方案**：新建 `training/adversarial_reward.py`，把 §2.3 的完整奖励（r_impact + 门控 threat + base_safe + λ_t·c_self 组装 + 对偶系数更新）收敛为单一函数/类，由三处共同 import：

| 落点 | 阶段 | 角色 |
|---|---|---|
| `train_adversarial_fixed_scenario.py` | Phase 1 | 快验代理（**保留**，单场景几十 update 先确认奖励能学出行为 + hack 仪表盘正常） |
| `training/psro/train_blue_oracle.py` | Phase 4 | PSRO 蓝 oracle 真正训练落点 |
| `train_multi_adversarial_kl_risk.py` | Phase 3 | 双蓝车 MAPPO |

**收益**：奖励只有一个真相源，改一次三处生效；PSRO 训练的与快验验证的保证是同一个奖励（这也是 §3.2 反 hacking 第 3 条"奖励-度量对齐"能成立的前提）。

**实现次序**（因 r_impact 复用归因逻辑）：`utils/adversarial_metrics.py`（归因工具）→ `training/adversarial_reward.py`（奖励组装，import 归因工具）→ 三处训练脚本改为 import 该模块。

---

## 3. Reward Hacking 分析与防御

### 3.1 已知与可预见的 hack 清单

| # | Hack | 机制 | 征兆指标 | 防御 |
|---|---|---|---|---|
| H1 | **自杀式碰撞**（已实际发生） | 撞红车→红车失败→"对抗有效" | `kamikaze_rate`↑、`blue_collision_rate`↑ | 碰撞归因三分法（主动撞击零奖励）+ λ_t 对偶约束 + KL 锚定 |
| H2 | **威胁窗口占位刷分** | 悬停在红车前方拿几何分，红车实际无感 | `threat_window_pct`↑ 但 `attributed_slowdown_rate` 持平 | 门控（红车响应才给分）+ 每回合封顶 |
| H3 | **回合终止利用** | 蓝车自撞终止全员回合→红车成绩清零 | 回合长度骤降 + blue自碰率↑ | `adversary_collision_ends_episode=False`（已加）；蓝车死后红车继续跑，其后事件不计任何对抗奖励 |
| H4 | **出生点封锁** | 蓝车固定卡在红车出生位置 | 归因事件全部集中在回合前 N 步 | 起点随机化 + 出生保护期（前 3s 内的事件不给 r_impact）+ held-out 场景评估 |
| H5 | **利用红车过度保守** | 远处停车→红车冻结→归因超时 | `attributed_timeout`↑ 但 `min_TTC` 无恶化、blocking_time 却很高且蓝车速度≈0 | 这是**合法发现但退化的行为类**：单独归类统计（`passive_block_rate`），红车池多样化（Phase 2）从根上治 |
| H6 | **稠密项无界累积** | 类似红车旧 progress farming | 分项占比曲线失衡 | 所有稠密项封顶 + `reward/*_share` 监控 |
| H7 | **best.pt 选择被 hack** | `1−success` 准则选中自杀/占位型策略 | best ckpt 回放行为异常 | §1.4 约束满足 + 归因收益的选择准则 |
| H8 | **度量本身有 bug 的 wireheading** | timeout 记成功、口径漂移 | 三分指标之和 ≠ 1 | Phase 0 先修度量：三分互斥断言、单元测试、口径写死在共享工具函数里 |

### 3.2 通用防御体系（结构性，不依赖逐条封堵）

1. **目标-约束分离 + 对偶变量**（λ_t）：hack 大多寄生在"约束被定价成固定权重"上；
2. **KL 锚定**：把策略空间限制在"合理交通行为"邻域，大量荒谬解直接不可达；
3. **奖励-度量对齐**：训练奖励只代理归因指标 → 刷奖励 ≈ 提升真实指标，hack 空间被压缩；
4. **场景随机化 + held-out 评估**：防对布局过拟合；
5. **人工回放审计**：每个里程碑 ckpt，回放"对抗奖励最高的 5 个 episode"——高分回合是 hack 最可能藏身之处；
6. **红车池多样性**（Phase 2）：防蓝车过拟合单一红车的特定弱点。

### 3.3 Hack 监测仪表盘（TB 必备曲线）

```
hack/kamikaze_rate            # H1
hack/threat_vs_slowdown_gap   # H2: threat_window_pct − attributed_slowdown_rate（持续张大=占位刷分）
hack/avg_ep_length            # H3
hack/impact_events_first3s    # H4: 出生保护期内触发的事件占比
hack/passive_block_rate       # H5: 蓝车速度<1m/s 且在窗口内的步数占比
reward/*_share                # H6: 各分项占比
episode/rate_success + rate_collision + rate_timeout ≡ 1  # H8: 断言性检查
```

**规则：每新增一个奖励项，必须同时新增至少一个对应的 hack 监测指标，否则不允许进训练代码。**

---

## 4. 分阶段实验计划

> 各阶段的完整运行指令见**附录 A**。

### Phase 0 — 度量地基（先于一切训练，~2-3 天）

| 任务 | 内容 |
|---|---|
| 0.1 | 实现统一评估脚本 `evaluation/evaluate_adversarial_c012.py`：三条件 C0/C1/C2、固定种子、deterministic、输出 §1.3 全部四层指标 + 配对统计（bootstrap CI） |
| 0.2 | 归因判定工具函数（威胁窗口、减速事件、碰撞三分法、出生保护期）抽成 `utils/adversarial_metrics.py`，训练与评估**共用同一份口径** |
| 0.3 | 成功/碰撞/超时三分互斥口径修正 + 断言 |
| 0.4 | 用现有 ckpt 跑 C0/C1/C2 基线存档（这也是 2+2 实验"无对抗收益"结论的正式复核） |

**验收**：三分指标之和 ≡ 1；C0 红车成功率与既往评估一致；C1−C0 交通效应量化落盘；评估结果写入 §5 追踪表。

### Phase 1 — 2红+1蓝 固定场景快验 + 奖励迭代（核心，~1-2 周）

| 任务 | 内容 |
|---|---|
| 1.1 | `train_adversarial_fixed_scenario.py --num-adversarial 1`，先跑当前奖励 60 updates 作为 reward-v0 对照 |
| 1.2 | 实现 §2.3 v-next 奖励（r_impact + 门控 threat + 封顶），实现与 hack 仪表盘同步落地 |
| 1.3 | 权重定标（§2.4 流程），reward-v1 训练 60-150 updates |
| 1.4 | 每个里程碑：C0/C1/C2 评估 + 高分回合人工回放审计 |
| 1.5 | 消融：threat_coef、β_KL、有无门控（验证门控确实防住 H2） |

**验收**：C2−C1 的 `Δsuccess_rate` 配对 95% CI 不跨 0；`blue_collision_rate ≤ 5%`；`kamikaze_rate ≤ 10%`；回放审计确认行为是"切入/阻挡/逼刹"而非退化解。

### Phase 2 — 红车基础策略池构建（可与 Phase 1 并行，~1 周）

> **奖励决策（v1.2）**：红车基础奖励函数 v2 **原样沿用，不做任何修改**——红车须是固定的"正常驾驶"基准，改动会破坏 C0/C1/C2 对照的可比性。
>
> **前置修复（区别于奖励）**：训练循环 bug ≠ 奖励 bug。Phase 2 重训前须并入 `robust_v2_convergence_fix_plan.md` 的**零重训成本修复**，至少 R1（成功/碰撞/超时三分互斥口径，否则课程学习被污染）与 R2（碰撞广播 done 时用 V(last_state) bootstrap 而非 0）。这不改变红车奖励，只是让原奖励被正确优化，保证池成员质量稳定。

| 任务 | 内容 |
|---|---|
| 2.1 | 场景丰富：新增带不同 `goal_zone` 的场景 JSON（近/中/远三档终点 × 现有起点变化）。**随机采样混合，不做单调距离课程**（距离课程曾诱发冻结局部最优，已回退） |
| 2.2 | robust_v2 重训 3 seeds（42/123/2024），训练分布 = 丰富后的场景池 |
| 2.3 | 每个 seed 收割中期 + 后期 checkpoint（不同熟练度 = 不同难度对手），入池 |
| 2.4 | 入池筛选：每个成员在固定评估集上 success ≥ 阈值（质量关）；成员间轨迹特征距离（均速/横向偏移分布/TTC 分布）足够分散（多样性关） |

**验收**：池内 ≥ 6 个成员，全部过质量关，两两行为距离矩阵落盘。

### Phase 3 — 双蓝车 MAPPO 对抗训练（Phase 1 验收后，~1-2 周）

- 把 Phase 1 验证过的 reward-v1 迁移到 2 蓝车；
- 信用分配：r_impact / threat_gated 天然按 agent 归属（哪辆蓝车在窗口内谁拿分），**禁止**团队平分；
- 新增蓝-蓝互撞归类：蓝车互撞计入各自 c_self，不触发任何 r_impact；
- 验收：与单蓝 Phase 1 相比 `Δsuccess_rate` 进一步下降，且两辆蓝车的 impact 事件都非零（防"一车干活一车躺平"）。

### Phase 4 — PSRO 迭代（Phase 2+3 验收后）

- 红池 = Phase 2 产出；蓝 oracle = Phase 3 配方；
- 复核既有修复（min_gap 铺开、蓝撞不终止回合）后重启 PSRO 外层循环；
- 验收：payoff 矩阵行/列方差 > 0（打破首跑零方差空转），元策略随迭代变化。

### 依赖关系

```
Phase 0 ──→ Phase 1 ──→ Phase 3 ──→ Phase 4
       └──→ Phase 2 ──────────────↗
```

---

## 5. 实验追踪表

> 每个实验（含失败的）跑完立即填一行。RunID 命名：`P{阶段}_{脚本缩写}_{日期}_{seed}`，例 `P1_fixscn_0708_s42`。

| 日期 | RunID | 脚本/关键配置 | 假设 | 关键结果（指标+数值） | 结论/决策 | TB/ckpt 路径 |
|---|---|---|---|---|---|---|
| | | | | | | |

## 6. 遗留决策点（需实验数据后回答）

- [x] ~~红车基础训练是否沿用原奖励~~ → **已定（v1.2）：原样沿用 v2，不改；对抗奖励按 §2.3 执行**
- [ ] r_impact 各事件的具体奖励数值（§2.3 表格为初值，Phase 1.3 定标后修订）
- [ ] 门控方案失效时是否切换 PBRS 势能形 threat
- [ ] H5 被动阻挡是否作为独立的合法行为类保留进蓝方策略池
- [ ] Phase 2 终点变化档位的具体 goal_zone 坐标（依赖 Town02 直路几何测量）

---

## 附录 A：各阶段运行指令

**通用前置**（每条指令执行前）：

1. 先启动 CARLA 服务器（另开一个终端，训练时建议低画质）：
   ```bash
   # 仓库根目录 D:\study\yan2\CARLA_0.9.16
   ./CarlaUE4.exe -quality-level=Low
   ```
2. Python 一律走仓库根的 **pipenv**（不要直接调 venv）：
   ```bash
   export PIPENV_PIPFILE=/d/study/yan2/CARLA_0.9.16/Pipfile
   export PIPENV_VERBOSITY=-1
   cd /d/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training
   ```
3. 所有脚本以**模块**方式从 `mappo_training/` 运行（`python -m ...`）。
4. 监控：`pipenv run tensorboard --logdir logs/`。

> 图例：✅=脚本已存在，可直接跑；🔨=需在 Phase 0 先实现的新脚本；`<...>`=需替换的实参。

---

### Phase 0 — 度量地基

**0.1🔨 三条件评估脚本** `evaluation/evaluate_adversarial_c012.py`（本阶段实现，以下为目标 CLI 约定）：
```bash
# C0/C1/C2 一次跑齐，固定种子 range(1000,1040)，deterministic，输出四层指标 + 配对CI
pipenv run python -m evaluation.evaluate_adversarial_c012 \
    --pretrained-ckpt <red_base.pt> \
    --adversarial-ckpt <blue.pt> \
    --num-cooperative 2 --num-adversarial 1 \
    --n-episodes 40 --eval-seed-start 1000 \
    --scenarios scenario_01_basic.json \
    --held-out-scenario scenario_03_far.json \
    --output-file logs/eval_c012/<runid>.json
```

**0.2🔨 归因工具函数** `utils/adversarial_metrics.py`（训练与评估共用，无独立 CLI，用单测校验）：
```bash
pipenv run python -m pytest tests/test_adversarial_metrics.py -v   # 威胁窗口/减速事件/碰撞三分/出生保护期口径
```

**0.4✅ 复核 2+2「无对抗收益」结论**（用现有 v2 评估脚本先出一版基线，待 0.1 就绪再用新脚本复跑）：
```bash
pipenv run python -m evaluation.evaluate_multi_adversarial_v2 \
    --pretrained-ckpt <red_base.pt> \
    --adversarial-ckpt <blue_2plus2.pt> \
    --num-episodes-per-scenario 40 \
    --scenarios scenario_01_basic.json \
    --policy deterministic --seed 1000 \
    --output-file logs/eval_c012/baseline_2p2_<date>.json
```

---

### Phase 1 — 2红+1蓝 固定场景快验 + 奖励迭代

**1.1✅ reward-v0 对照**（当前奖励，`--num-adversarial 1` 即 2红+1蓝）：
```bash
pipenv run python -m training.train_adversarial_fixed_scenario \
    --pretrained-ckpt <red_base.pt> \
    --num-cooperative 2 --num-adversarial 1 \
    --scenario scenario_01_basic.json \
    --updates 60 --seed 42 \
    --save-dir logs/P1_fixscn_rv0_s42 --no-viz
```

**1.3✅→🔨 reward-v1**（§2.3 新奖励，需先在脚本内接入 r_impact/门控/封顶 + hack 仪表盘；CLI 沿用现有参数，新增权重开关待实现后补入本表）：
```bash
pipenv run python -m training.train_adversarial_fixed_scenario \
    --pretrained-ckpt <red_base.pt> \
    --num-cooperative 2 --num-adversarial 1 \
    --scenario scenario_01_basic.json \
    --updates 150 --seed 42 \
    --threat-coef <定标值> --beta-kl 0.1 \
    --target-blue-collision-rate 0.05 \
    --save-dir logs/P1_fixscn_rv1_s42 --no-viz
```

**1.4 里程碑评估**（每 ~30 update 后对 `best.pt` 跑 Phase 0.1 的 C0/C1/C2）：
```bash
pipenv run python -m evaluation.evaluate_adversarial_c012 \
    --pretrained-ckpt <red_base.pt> \
    --adversarial-ckpt logs/P1_fixscn_rv1_s42/best.pt \
    --num-cooperative 2 --num-adversarial 1 \
    --n-episodes 40 --eval-seed-start 1000 \
    --output-file logs/eval_c012/P1_rv1_s42.json
```

**1.5 消融**（示例：关掉门控验证 H2 防御是否生效——需脚本支持 `--no-threat-gating` 开关）：
```bash
pipenv run python -m training.train_adversarial_fixed_scenario \
    --pretrained-ckpt <red_base.pt> --num-cooperative 2 --num-adversarial 1 \
    --scenario scenario_01_basic.json --updates 150 --seed 42 \
    --no-threat-gating --save-dir logs/P1_ablation_nogate_s42 --no-viz
```

---

### Phase 2 — 红车基础策略池构建（可与 Phase 1 并行）

**2.2✅ robust_v2 多 seed 重训**（3 个 seed 各一条；场景丰富后 `data/training_scenarios/` 新增终点档位，采样用 weighted 非距离课程）：
```bash
for S in 42 123 2024; do
  pipenv run python -m training.train_no_rllib_mappo_robust_v2 \
      --updates 150 --seed $S \
      --scenario-sampling-mode weighted \
      --curriculum-mode classic \
      --warmup-updates 20 --full-scenario-updates 80 \
      --reward-version v2 \
      --save-dir logs/P2_redbase_s$S
done
```
> 注：`--curriculum-mode classic`（不用 `distance`，距离课程曾诱发冻结局部最优）。

**2.3 / 2.4✅ 收割 checkpoint 入池 + 筛选**（PSRO 工具，`--collect-only` 只归集不重训）：
```bash
pipenv run python -m training.psro.build_red_pool \
    --from-ckpts logs/P2_redbase_s42/best.pt logs/P2_redbase_s42/ckpt_075.pt \
                 logs/P2_redbase_s123/best.pt logs/P2_redbase_s123/ckpt_075.pt \
                 logs/P2_redbase_s2024/best.pt logs/P2_redbase_s2024/ckpt_075.pt \
    --cooperative-ids 0,1 \
    --collect-only --out logs/red_pool/ckpts

# 池成员质量+多样性体检
pipenv run python -m training.psro.baseline_eval \
    --red-pool-dir logs/red_pool/ckpts \
    --scenario scenario_01_basic.json \
    --blue-mode natural --n-eval 20 \
    --output-dir logs/red_pool/baseline
```

---

### Phase 3 — 双蓝车 MAPPO 对抗训练

**3.1✅ reward-v1 迁移到 2 蓝车**（`train_multi_adversarial_kl_risk.py`，`--num-adversarial 2`）：
```bash
pipenv run python -m training.train_multi_adversarial_kl_risk \
    --pretrained-ckpt <red_base.pt> \
    --num-cooperative 2 --num-adversarial 2 \
    --coop-start-config scenario_03_far.json \
    --updates 300 --seed 42 \
    --beta-kl 0.1 --threat-coef <与P1一致的定标值> \
    --target-blue-collision-rate 0.05 \
    --curriculum-learning --warmup-updates 50 --ramp-up-updates 100 \
    --scenario-sampling-mode weighted \
    --save-dir logs/P3_multiadv_s42 --no-viz
```

**3.✅ 评估**（沿用 v2 评估，2红2蓝）：
```bash
pipenv run python -m evaluation.evaluate_multi_adversarial_v2 \
    --pretrained-ckpt <red_base.pt> \
    --adversarial-ckpt logs/P3_multiadv_s42/best.pt \
    --num-episodes-per-scenario 40 \
    --policy deterministic --seed 1000 \
    --output-file logs/eval_c012/P3_s42.json
```

---

### Phase 4 — PSRO 迭代

**4.✅ 空转体检**（先确认 payoff 非零方差，用 Phase 2 的红池）：
```bash
pipenv run python -m training.psro.baseline_eval \
    --red-pool-dir logs/red_pool/ckpts \
    --scenario scenario_01_basic.json \
    --blue-mode adversarial --n-eval 10 \
    --cooperative-ids 0,1 --adversarial-ids 2,3 \
    --output-dir logs/psro_runs/precheck_<date>
```

**4.✅ PSRO 外层循环**：
```bash
pipenv run python -m training.psro.psro_runner \
    --red-pool-dir logs/red_pool/ckpts \
    --psro-iters 5 --inner-updates 30 --n-eval-episodes 5 \
    --meta-solver prd \
    --scenario scenario_01_basic.json \
    --cooperative-ids 0,1 --adversarial-ids 2,3
```

**4.✅ 联合最佳响应评估（JPC）**：
```bash
pipenv run python -m training.psro.jpc_eval \
    --red-pool-dir logs/red_pool/ckpts \
    --blue-pool-dir <psro产出的blue池目录> \
    --output-dir logs/psro_runs/jpc_<date> \
    --n-eval 5
```

> 附录内所有 `<red_base.pt>` 指 Phase 2 产出的红车基础策略；`--save-dir` 命名与 §5 追踪表的 RunID 保持一致，便于回溯。
