# 对抗智能体训练 — MAPPO + PSRO 设计计划

> **2026-06-21 实现评审更新**：`training/psro/` 已实现并通过导入/数值自检。评审结论、
> 4 处修改记录（关键：JPC R_- 对 PSRO 池语义修正 + 补充泛化指标）、完整可复现用法指南见
> [对抗训练_PSRO_实现评审与用法指南.md](对抗训练_PSRO_实现评审与用法指南.md)。

**日期**：2026-06-10
**任务定位**：在已训练好的 2 智能体合作策略（MAPPO 训出的 RED 团队）基础上，
训练 2 个对抗智能体（BLUE 团队，MAPPO + CTDE），产出**可泛化、协同、不自毁**的对抗策略。

**论文参考**：Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent
Reinforcement Learning*, NeurIPS 2017（[读书笔记](论文/A%20Unified%20Game-Theoretic%20Approach%20to%20Multiagent%20Reinforcement%20Learning.md)）。

---

## 1. 范围与约束

| 项       | 值                                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------------- |
| 团队规模    | **固定 2v2**（2 RED + 2 BLUE）                                                                                        |
| RED 团队  | MAPPO 训练完成（1 个 ckpt），训练后**全程冻结**                                                                                  |
| BLUE 团队 | 2 个 agent，MAPPO + CTDE 训练，团队内部协同                                                                                  |
| 仿真器     | CARLA 0.9.16, Town02 直道                                                                                           |
| 已有代码    | `train_multi_adversarial_kl_risk.py` / `train_adversarial_fixed_scenario.py` / `evaluate_multi_adversarial_v2.py` |
| 已有数据    | 1 个训练好的 RED ckpt + 12 个场景 JSON                                                                                    |

---

## 2. MAPPO 与 PSRO 的分工（必须先理清）

论文的核心洞察是 **PSRO 与 MAPPO 不在同一层级**，应嵌套使用：

```
┌─────────────────────────── PSRO 外层（种群调度）──────────────────────────┐
│                                                                          │
│   RED 池 = { red_v1, red_v2, ..., red_vK }   ← Phase 0 扩展              │
│   BLUE 池 = { blue_t1, blue_t2, ... }        ← PSRO 每轮新增一个         │
│   元策略 σ_red = 在 RED 池上的概率分布                                   │
│                                                                          │
│   for psro_iter in range(N):                                             │
│       ┌──────────── MAPPO 内核（产出一个 BLUE oracle）──────────┐         │
│       │                                                          │         │
│       │  每个 episode: 采样 frozen RED 团队 ~ σ_red               │         │
│       │  用 MAPPO + CTDE 训练 BLUE 2-agent 团队                   │         │
│       │  共享 critic 只覆盖 BLUE 队内（一般和约束）              │         │
│       │  Actor: KL 锚定到 base + threat reward + 自适应自碰      │         │
│       │                                                          │         │
│       └──────────────────────────────────────────────────────────┘         │
│       blue_team_new = trained MAPPO team                                  │
│       BLUE 池.add(blue_team_new)                                          │
│       更新 σ_red, σ_blue                                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**对应关系**：

| 层级 | 作用 | 我们用什么 |
|---|---|---|
| 外层 PSRO | 决定"BLUE 跟谁训"，防过拟合到单一 RED | 新建 `psro_runner.py` |
| 内层 MAPPO | 决定"BLUE 队内怎么协同" | 复用 `train_multi_adversarial_kl_risk.py`（改造） |
| 真实性约束 | 决定"BLUE 行为像真人" | KL anchor + threat reward + 自适应自碰（已有） |

**关键约束（来自论文 §三注 2）**：

- ❌ 不要用一个集中 critic 同时覆盖 RED + BLUE → 会把两个对立目标平均掉
- ✅ BLUE 团队**自己一个 critic**（CTDE），看到全员 obs 但只回归 BLUE 的 return
- ✅ RED 团队冻结，仅作为环境的一部分进入 critic 输入

---

## 3. 现状诊断 — 距离 PSRO 多远

**结论**：当前 `train_multi_adversarial_kl_risk.py` 实际上是 **PSRO 的 InRL 特例**
（即 σ_red 是单点分布，只对 1 个 RED 训练）。论文证明这种特例会过拟合：
对单一对手训出的 BLUE，在不同 RED 上效果可能塌缩到无效。

我们目前缺的就两件事：

1. **RED 池只有 1 个元素** → PSRO 退化为 InRL（必须扩 RED 池）
2. **没有外层调度逻辑** → 每次训练都是孤立的（必须写 PSRO runner）

下面分四个阶段补这两件事。

---

## 4. 训练方案分阶段

### Phase 0 ─ RED 池扩展（先决条件，2-3 天）

PSRO 至少需要 σ_red 有多元素才能"对分布求响应"。

#### 4.0.1 四种产生 RED 变体的途径

| 方法 | 难度 | 多样性 | 推荐度 |
|---|---|---|---|
| **A. 不同 seed 重训** | 低（改 `--seed` 即可） | 中-高 | ★★★★★ |
| **B. 同一训练的不同 checkpoint** | 极低（已有 ckpt 文件） | 低 | ★★★ |
| **C. 超参数扰动** | 中（多组配置） | 中 | ★★★ |
| **D. 加入规则驾驶员（IDM/MOBIL）作为非 RL ego** | 高（需新代码） | 极高 | ★★（研究深度方向） |

**推荐起步组合**：A（3 个 seed）+ B（同一 seed 取 update_100/150/200 三个快照）= 池大小 ≥ 3-5。

#### 4.0.2 具体操作

```bash
# A. 不同 seed
for seed in 42 123 2024; do
    python -m training.train_no_rllib_mappo_robust_v2 \
        --seed $seed \
        --updates 150 \
        --save-dir ./logs/red_pool/seed_$seed
done

# B. 已有 ckpt 不同步数（如果训练时多保存了快照）
cp logs/robust_v2/mappo_robust_v2_update_100.pt logs/red_pool/red_snap_100.pt
cp logs/robust_v2/mappo_robust_v2_update_150.pt logs/red_pool/red_snap_150.pt
cp logs/robust_v2/best.pt                        logs/red_pool/red_best.pt
```

#### 4.0.3 RED 池质量验收

每个 RED 单独评估，必须满足：

- `coop_success_rate > 80%`（合作策略本身要可用）
- 行为可观察差异（速度分布、跟车间距、对刹车反应有可见区别 → 用 `evaluate_multi_adversarial_v2.py` 的 baseline 模式分别评估并对比 TB）

不达标的 RED 剔除（避免 σ_red 退化）。

---

### Phase 1 ─ MAPPO 内核已完成（确认与小改）

**好消息**：`train_multi_adversarial_kl_risk.py` 已经是 MAPPO + KL 锚定 + 风险增量奖励的完整实现，作为 PSRO 内层 oracle **几乎不需要改动**。

#### 4.1.1 唯一必须改造的点

把"加载单个 pretrained ckpt → 冻结 RED"改造成"加载多个 RED ckpt → 每 episode 采样"：

```python
# 原：（train_multi_adversarial_kl_risk.py L233-244）
ckpt = torch.load(pretrained_path, ...)
pretrained_actors_state = ckpt.get("actors", {})
for cid in cooperative_ids:
    actor.load_state_dict(pretrained_actors_state[cid])

# 改：每 episode 从 σ_red 采样一个 RED 团队
red_pool = RedTeamPool(red_pool_paths)  # 加载所有 RED ckpt
def get_episode_red_team(sigma_red):
    idx = np.random.choice(len(red_pool), p=sigma_red)
    return red_pool[idx]   # 返回 {cid: Actor}

# 每个 episode reset 前调用：
current_red_team = get_episode_red_team(sigma_red)
for cid in cooperative_ids:
    cooperative_actors[cid] = current_red_team[cid]
```

#### 4.1.2 重命名为 PSRO oracle 角色

新文件：[training/psro/train_blue_oracle.py](../training/psro/train_blue_oracle.py)
基于 `train_multi_adversarial_kl_risk.py` 复制 + 上述改造 + 暴露 `train_oracle(args, sigma_red)` 函数供 runner 调用。

#### 4.1.3 BLUE 团队作为一个整体保存

每个 PSRO 轮次产出的 BLUE oracle 是一个**团队捆绑**，不是单个 actor：

```python
blue_team_ckpt = {
    "adversarial_actors": {bid: actor.state_dict() for bid in adversarial_ids},
    "critic": critic.state_dict(),   # CTDE 共享 critic
    "log_std": {bid: actor.log_std for bid in adversarial_ids},
    "trained_against_sigma_red": sigma_red.tolist(),
    "psro_iter": psro_iter,
    "args": vars(args),
}
torch.save(blue_team_ckpt, f"logs/blue_pool/psro{psro_iter}_blue_team.pt")
```

---

### Phase 2 ─ PSRO 外层骨架（3-5 天，本计划核心）

#### 4.2.1 新文件结构

```
training/psro/
  __init__.py
  population.py           ← RED 池 / BLUE 池管理（加载/采样/快照）
  meta_solver.py          ← 元策略求解器（uniform / PRD）
  payoff_table.py         ← 经验收益张量（增量更新）
  train_blue_oracle.py    ← MAPPO 内核（改造自 kl_risk）
  psro_runner.py          ← 外层主循环
```

#### 4.2.2 `psro_runner.py` 主流程

```python
def psro_main(args):
    red_pool = RedTeamPool.from_dir(args.red_pool_dir)
    blue_pool = BlueTeamPool(save_dir=args.blue_pool_dir)
    payoff = PayoffTable(red_pool_size=len(red_pool), max_blue=args.psro_iters)
    meta = make_meta_solver(args.meta_solver)   # uniform / PRD

    sigma_red = meta.uniform(len(red_pool))     # 初始 uniform

    for psro_iter in range(args.psro_iters):    # 5-10 轮
        print(f"\n=== PSRO iter {psro_iter} ===")
        print(f"σ_red = {sigma_red.round(3)}")

        # 步①：对 σ_red 训一个新 BLUE oracle（MAPPO）
        new_blue = train_blue_oracle(
            red_pool=red_pool, sigma_red=sigma_red,
            inner_args=args.inner,
            train_updates=args.inner_updates,   # 30-60
            save_path=blue_pool.next_save_path(psro_iter),
        )
        blue_pool.add(new_blue)

        # 步②：补全经验收益张量（增量评估）
        for ri in range(len(red_pool)):
            payoff.fill(ri, psro_iter, evaluate_pair(
                red_pool[ri], blue_pool[psro_iter],
                n_episodes=args.n_eval_episodes,    # 5
                scenarios=args.eval_scenarios,
            ))

        # 步③：更新元策略
        sigma_red, sigma_blue = meta.solve(payoff.matrix())
        print(f"new σ_red = {sigma_red.round(3)}, σ_blue = {sigma_blue.round(3)}")

        # 步④：JPC 矩阵 + log
        jpc_R_minus = compute_jpc_R_minus(payoff.matrix())
        log_to_tensorboard(payoff, sigma_red, sigma_blue, jpc_R_minus, psro_iter)
```

#### 4.2.3 元策略求解器（分两档实现）

**L0（必做）**：uniform mixture（fictitious play 风格）—— 30 行
**L1（推荐收尾）**：Projected Replicator Dynamics（PRD）—— 论文 §3.3 + 附录 A.3

```python
class PRDSolver:
    """投影复制动态 — 论文 Algorithm 2"""
    def __init__(self, gamma=0.1, lr=0.01, max_iter=10000):
        self.gamma = gamma   # 探索下界，0.1=偏剥削，0.4=偏鲁棒
        ...

    def solve(self, payoff_matrix):
        # 1. 复制动态迭代
        # 2. 每步投影到 γ-探索单纯形（每个策略概率 ≥ γ/(K+1)）
        # 3. 返回 (sigma_red, sigma_blue)
        ...
```

**γ 推荐取值**（论文 Leduc 实验）：
- `γ=0.4` 最稳鲁棒
- `γ=0.1` 最强剥削
- 起步用 `γ=0.2` 折中

#### 4.2.4 经验收益张量（2v2 + |R|×|B|）

```python
class PayoffTable:
    """
    存所有 (red_team, blue_team) 配对的多维收益：
      - coop_success_rate    (核心指标，BLUE 想让它降低)
      - blue_threat_window_pct
      - red_relative_slowdown
      - blue_collision_rate  (副指标，自毁度量)
    """
    def __init__(self, red_pool_size, max_blue):
        self.coop_success = np.full((red_pool_size, max_blue), np.nan)
        # 其他指标同理...

    def matrix(self):
        """返回 BLUE 视角的对抗收益（= 1 - coop_success_rate），供 meta_solver 用"""
        return 1.0 - self.coop_success
```

---

### Phase 3 ─ JPC 评估（1-2 天，必做）

PSRO 的价值需要 JPC 矩阵证明。借鉴论文 §3.2 (a)。

#### 4.3.1 评估流程

```python
def jpc_full_matrix(red_pool, blue_pool, n_episodes=10, scenarios=None):
    """
    生成 |R|×|B| 完整交叉评估矩阵。
      - 对角线（i=i 配对）= 配对训练的 (red_i, blue_i)
      - 非对角线 = 错配
    """
    matrix = np.zeros((len(red_pool), len(blue_pool)))
    for ri, red in enumerate(red_pool):
        for bi, blue in enumerate(blue_pool):
            result = evaluate_pair_multi_metric(red, blue, scenarios, n_episodes)
            matrix[ri, bi] = result["coop_success_rate"]

    # 论文核心指标 R_-
    diag_mean = np.mean(np.diag(matrix))           # 配对效果
    off_diag_mean = np.mean(matrix[~np.eye(...)])  # 错配效果
    R_minus = (diag_mean - off_diag_mean) / diag_mean
    return matrix, R_minus
```

#### 4.3.2 新文件 [evaluation/evaluate_jpc.py](../evaluation/evaluate_jpc.py)

输出：
- `jpc_matrix.csv`（|R|×|B| 数字表）
- `jpc_heatmap.png`（matplotlib 热图）
- `jpc_report.json`（含 R_-、各指标矩阵、配对效果汇总）

#### 4.3.3 判读规则

| R_- 范围 | 含义 | 处置 |
|---|---|---|
| > 0.30 | 严重过拟合（BLUE 只对训练 RED 有效） | 增加 PSRO 轮次或扩 RED 池 |
| 0.10–0.30 | 有过拟合迹象 | 调大 PRD 的 γ |
| < 0.10 | 泛化良好 | OK |

**目标**：PSRO 后的 BLUE 池 R_- 显著低于 InRL baseline（论文降幅 28.7%-56.7%）。

---

### Phase 4 ─ 共演化（可选，研究深度方向，5+ 天）

只有 Phase 0-3 跑通且 R_- 稳定下降后再做。

```python
for outer in range(N):
    train_blue_oracle_vs_red_mixture(steps=A)
    blue_pool.add(); update_payoff()

    if outer % K == 0 and outer >= warmup:
        train_red_oracle_vs_blue_mixture(steps=B)
        red_pool.add(); update_payoff()
```

**已知风险**（论文 §五 + 后续 PSRO 文献）：
- 一般和非传递性 → BLUE_t 克 RED_t 克 BLUE_{t-1} 克 RED_{t-1}... → 元策略震荡不收敛
- 缓解：保留所有历史快照在池里（PSRO 默认行为）；改用 α-Rank 而非纳什求解器

---

## 5. 文件结构总览

```
PythonAPI/examples/mappo_training/
├─ training/
│  ├─ train_multi_adversarial_kl_risk.py   (已有，InRL 内核 = PSRO 第 1 轮)
│  ├─ train_adversarial_fixed_scenario.py  (已有，固定场景快验)
│  └─ psro/                                 (新增)
│     ├─ __init__.py
│     ├─ population.py
│     ├─ meta_solver.py
│     ├─ payoff_table.py
│     ├─ train_blue_oracle.py
│     └─ psro_runner.py
├─ evaluation/
│  ├─ evaluate_multi_adversarial_v2.py     (已有，单对 BLUE-RED 评估)
│  └─ evaluate_jpc.py                       (新增，|R|×|B| 交叉矩阵 + R_-)
├─ logs/
│  ├─ red_pool/                             (新增，Phase 0 产物)
│  │  ├─ seed_42/best.pt
│  │  ├─ seed_123/best.pt
│  │  └─ seed_2024/best.pt
│  ├─ blue_pool/                            (新增，PSRO 产物)
│  │  ├─ psro0_blue_team.pt
│  │  ├─ psro1_blue_team.pt
│  │  └─ ...
│  └─ psro_runs/<timestamp>/                (新增，TB + 元策略历史)
└─ docs/
   └─ 对抗训练_MAPPO_PSRO_设计计划.md       (本文档)
```

**改动量预估**：
- 新文件 6 个 ≈ 800-1000 行
- 复用 `train_multi_adversarial_kl_risk.py` ≈ 70% 代码不动
- 修改 `evaluate_multi_adversarial_v2.py` 抽出 `evaluate_pair_multi_metric()` 公共函数

---

## 6. 里程碑与时间预算

| 节点 | 完成标志 | 时间 |
|---|---|---|
| M0 | 计划文档 review（本文档） | 已完成 |
| M1 | RED 池 ≥ 3 个 ckpt，每个 coop_success > 80% | 2-3 天 |
| M2 | `train_blue_oracle.py` 支持 σ_red 采样，单轮 PSRO 跑通 | 2 天 |
| M3 | `psro_runner.py` + uniform meta-solver 完整跑 3 轮 | 3 天 |
| M4 | PRD meta-solver 实现，完整跑 5 轮 | 2 天 |
| M5 | `evaluate_jpc.py` 完成，输出热图 + R_- 报告 | 2 天 |
| M6 | **验收**：PSRO 后 R_- 相对 InRL 下降 > 20% | 验收节点 |
| M7（可选） | Phase 4 共演化跑通 3 个交替轮 | 5+ 天 |

**核心实施总预算**：M1-M6 ≈ **11-12 个工作日**。

---

## 7. 关键设计决策

| 决策点 | 选项 | 建议 |
|---|---|---|
| RED 池规模 | 3 / 5 / 8 | **5**（够多元，CARLA 训练成本可控） |
| PSRO 轮次 | 5 / 10 / 20 | **5-7**（边际递减 + CARLA 成本） |
| 内层 PPO updates / 轮 | 20 / 30 / 60 | **30**（每轮约 1 小时） |
| 每对 eval episodes | 3 / 5 / 10 | **5**（统计 power 足够 + 成本可控） |
| 元求解器 | uniform / PRD / α-Rank | **uniform 起步，PRD 收尾** |
| γ（PRD 探索下界） | 0.1 / 0.2 / 0.4 | **0.2** |
| 共演化 | 现在做 / 后期做 | **后期**（先保证 BLUE 这条线跑通） |
| BLUE critic 共享性 | 共享 / agent-id 条件化 | **AgentConditionedCritic**（沿用 kl_risk） |

---

## 8. 与已有工作的依赖关系

| 已有组件 | 在新方案中的角色 |
|---|---|
| `train_multi_adversarial_kl_risk.py` 的 KL 锚定 | BLUE actor 始终保持自然驾驶（PSRO 不管真实性，论文 §四） |
| `compute_blue_threat_scores` | 内层奖励 + 评估指标 |
| 自适应自碰惩罚 (PID-I) | 防自毁，与 PSRO 多样性正交 |
| `AgentConditionedCritic` | BLUE 团队共享 CTDE critic |
| `evaluate_multi_adversarial_v2.py` 的 `_run_episode` | 提取为 `evaluate_pair_multi_metric()` 供 payoff_table 调用 |
| `make_adversarial_positions` | 蓝车摆位（PSRO 不改变此部分） |
| 场景文件初始化修复（M1-M3） | 前置条件（见 [对抗训练_初始化分析与修复计划.md](对抗训练_初始化分析与修复计划.md)） |

---

## 9. 风险与备选

| 风险 | 概率 | 影响 | 备选 |
|---|---|---|---|
| CARLA 太慢，PSRO 5 轮跑不完 | 中 | 时间超预算 | 缩 inner_updates 到 20；并行评估 |
| RED 池差异不够（seed 重训出来都差不多） | 中 | PSRO 退化 InRL | 加超参数扰动（C）；甚至加 IDM 规则 ego（D） |
| 非传递性导致 σ 震荡 | 低（无共演化时） | 元策略不稳定 | uniform 兜底；监控 σ 历史变化幅度 |
| JPC R_- 没下降 | 低 | 论文价值未实现 | 检查 RED 池真实差异（BC 行为聚类） |
| BLUE 团队 critic 拟合不上（多 RED 导致状态分布更广） | 中 | explained_variance 低 | 增大 critic hidden_dim 到 256；增大 rollout |

---

## 10. 不在本计划范围内（明确划定）

- 跨地图泛化（仍限 Town02）
- 不同车辆动力学（仍限 Tesla Model 3）
- 对抗车队规模 > 2（架构支持，但实验范围不展开）
- 实时部署到真实车辆
- 多 GPU 并行训练

---

## 11. 推荐起步顺序

1. **Phase 0 立刻可做**：跑 3 个 seed 重训 RED，2-3 天产出 RED 池
2. **同时可写**：`population.py` + `meta_solver.py`（uniform 版）+ `payoff_table.py`
3. **Phase 1 改造**：把 `train_multi_adversarial_kl_risk.py` 改造成 `train_blue_oracle.py`
4. **冒烟测试**：3 个 RED × 1 轮 PSRO × 20 update → 跑通即视为骨架 OK
5. **完整跑 5 轮 + PRD** → 出 JPC 报告 → 验收

