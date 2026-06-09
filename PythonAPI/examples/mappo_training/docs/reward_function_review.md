# 奖励函数设计复审与规则总结

> 日期：2026-06-08
> 范围：`environment/carla_multi_agent_env.py` 的 `_compute_rewards_v2`（默认）+ `_compute_adversarial_reward`
> 关联：`mappo_holistic_review.md`（算法/工程复审）

---

## 一、发现清单（按严重度 / 含状态追踪）

| 编号 | 级别 | 问题 | 位置 | 状态 |
|------|------|------|------|------|
| R-1 | 🔴 高 | `approach_reward` 是 level 奖励、每步发放、未 motion_gate → 停在终点外可刷分，累计可压过 success；且与 progress 重复 | `_compute_rewards_v2` 1b 段 | ✅ 已修复（PBRS） |
| R-2 | 🟠 中 | `safety_lidar` 与 `ttc` 重复惩罚"邻近"，窄路多车时压制 progress，且与对抗切入目标对冲 | 第 5/6/7 段 | ⬜ 待修（合作侧待讨论） |
| R-3 | 🟠 中 | env 级对抗奖励（cut_in≤0.5 / blocking≤0.3）弱于 base 驾驶（progress×0.3≈+1/步）；zero-sum 项又被 ×0.01 缩小 | `_compute_adversarial_reward` + 训练脚本 | ⬜ 待修（对抗侧，暂缓） |
| R-4 | 🟡 低 | 车道惩罚只有"压线"离散事件，偏离车道中心无连续惩罚 → S 形画龙/变道几乎免费 | 第 9 段 | ✅ 已修复（连续横向偏离惩罚 + 加大转向率惩罚） |
| R-5 | 🟡 低 | success(+100) 在长 episode 被 γ 折到近 0，终点信号几乎只能靠 progress 间接学到 | 第 11 段 | ⬜ 待评估 |
| R-6 | 🟡 doc | 注释 `COLLISION_PENALTY=-300` 与实际常量 -150 不符 | 第 8 段注释 | ✅ 已修复 |

> 当前焦点：**合作奖励**修改（R-2 / R-4 / R-5 + 速度奖励等塑形项）。对抗侧（R-3）暂缓。

---

## 二、本轮修复

### R-1（已修复）：approach 改为势能塑形（PBRS）
**旧实现**（farmable）：
```python
approach_ratio = (d0 - current_dist) / d0       # 当前位置的 level 量
approach_reward = APPROACH_WEIGHT * approach_ratio   # 每步发放、未 gate
```
停在终点区外几米：speed=0 使 centering/yaw/alive 归零，但 approach 仍每步 +0.47，
净 +0.37/步；剩余 800 步 ≈ **+296**，而进入终点仅一次性 `success+time_bonus≈+140` 且回合结束
→ "到终点不进、在边上刷分"成为局部最优。

**新实现**（farm-proof，Ng et al. 1999 势能塑形，保证最优策略不变）：
```python
ratio_prev = clip((d0 - prev_dist) / d0, 0, 1)
ratio_curr = clip((d0 - current_dist) / d0, 0, 1)
approach_reward = APPROACH_WEIGHT * (reward_gamma * ratio_curr - ratio_prev)   # F = γΦ' − Φ
```
`reward_gamma` 为新增构造参数（默认 0.99，应与训练 gamma 一致）。

**数值验证**：
| 行为 | 旧 approach/步 | 新 approach/步 | 800步累计(新/旧) |
|------|------|------|------|
| 停在终点外 5m | +0.475 | **−0.0047** | −3.8 / +380 |
| 向终点移动 1m | +0.475 | +0.0002~0.0025 | — |
| 远离终点 1m | +0.465 | −0.0097 | — |

farm 被消除，"吸向终点"梯度仍存（方向正确，主驱动仍是 progress×5）。

### R-6（已修复）：同步陈旧注释 -300 → -150。

> R-2 / R-3 涉及权重调参、会改变训练动力学，**未自动修改**，留待确认。

---

## 三、合作智能体奖励设计规则（reward_version='v2'）

合作（含普通 MAPPO、robust）智能体奖励 = 以下各项之和：

| 分量 | 形式 | 量级/步 | 作用 |
|------|------|---------|------|
| progress | `(prev_dist−cur_dist)×5` | ±(0~3) | **主引导**，telescoping 势能，朝终点 |
| approach (R-1后) | `0.5·(γ·ratio'−ratio)` | ≈±0.01 | 辅助吸引，farm-proof |
| speed | 高斯峰@8m/s，减基线 | 0~+0.43 | 合理车速；speed=0 时为 0 |
| alive | ramp 到 0.3 @2m/s | 0~+0.3 | 鼓励"动起来" |
| centering | `0.5·exp(−d²/2σ²)`×motion_gate | 0~+0.5 | 车道居中（停车不发） |
| yaw | `0.15·(cos+1)/2`×motion_gate | 0~+0.15 | 航向对齐（停车不发） |
| jerk / steer_rate | `−w·|Δctrl|/dt` | 小负 | 平滑性 |
| safety_lidar | 11 ray，<0.5 线性罚 | 0~−3.3 | 近距避障（软） |
| ttc | 前向同车道 `−2·e^(−ttc)` | 0~−数 | 前向碰撞风险（软） |
| collision | −150（含压人行横道） | 0/−150 | 硬安全 |
| lane_invasion | 虚线−3/实线−12/路沿−25 | 0/负 | 车道纪律 |
| step | −0.1 | −0.1 | 时间压力 |
| success | +100（进 goal_zone） | 0/+100 | 终点（硬） |
| time_bonus | `(剩余步/总步)·50` | 0~+50 | 越快到越多 |

**设计规则**：
1. **势能塑形主导**：progress 是 telescoping 形式，对 γ 折扣鲁棒，承担主要引导；
2. **全连续可微**：速度/居中/航向用高斯/指数/余弦，消除 v1 的二元阶梯；
3. **运动门控防 farm**：姿态类(centering/yaw/alive) × `min(1, speed/2)`，停车归零；
4. **安全分层**：软（lidar/ttc 连续梯度）→ 硬（collision/lane 离散重罚）；
5. **回报排序约束**（设计目标）：撞车(−150) < 原地不动(−100) < 正常行驶 < 成功。

**合理性评估**：
- ✅ 主体合理：势能塑形 + 运动门控 + 连续化是正确的工程方向，历史上的"原地不动 farm"和"撞车 return>0"两类致命 bug 已被针对性修复（见代码注释 R4/R5/B2）。
- ⚠️ **塑形项偏多（13 项）**：多项小信号（speed/alive/centering/yaw/approach）方向部分重叠，存在"信号竞争/相互稀释"，调参敏感、可解释性下降。建议精简或合并同类项。
- ⚠️ **安全项窄路过罚（R-2）**：Town02 窄路 + 多车时 safety_lidar(−3.3) + ttc 可淹没 progress，需在 TB 上监控 `safety_lidar`/`ttc` 分量占比。
- ⚠️ **终点信号弱（R-5）**：success 被 γ 折没，实际靠 progress 间接学到终点；可考虑把 success 也做成接近终点的高势能。

---

## 四、对抗智能体奖励设计规则

对抗智能体（Agent N..N+M-1）最终奖励 = **env 级对抗奖励** + **训练脚本零和项**：

**env 级**（`_compute_adversarial_reward`）= `base_driving + cutting_in + blocking`：

| 分量 | 形式 | 上限/步 | 作用 |
|------|------|---------|------|
| base_driving | 各 base 分量乘系数（progress 0.3 / speed 0.3 / centering 0.2 / yaw 0.1 / **collision 1.0** / lane 0.5 / success 0.3 / ...） | — | 保留基本驾驶能力 + **全额自身碰撞惩罚** |
| cutting_in | 侧面 1~3m 且纵向 −5~+10m | +0.5/对手 | 鼓励"贴边切入" |
| blocking | 正前 <1.5m 且 0~8m | +0.3/对手 | 鼓励"前向阻挡" |

**脚本级零和项**（`train_adversarial_ppo` / `train_multi_adversarial_mappo*`）：
`+ coef × (−avg_coop_reward) × 0.01`（v1 写死 ×0.01；v2 用 `--zero-sum-coef` 参数化），
配合课程：warmup 只驾驶（coef=0）→ ramp 渐增对抗系数。

**设计规则**：
1. **保留基础驾驶**：避免对抗 agent 退化成"乱撞/不动"，仍需会开车才能实施干扰；
2. **全额保留自身碰撞惩罚**（×1.0）：对抗 ≠ 自毁，鼓励"逼停/挤压对手"而非同归于尽；
3. **几何塑形干扰行为**：用相对位置（横/纵向）显式定义"切入"和"阻挡"区域；
4. **零和目标**：把"压低对手收益"直接计入对抗奖励；
5. **课程过渡**：先学会开车再逐步加对抗，避免早期对抗信号破坏基础策略。

**合理性评估**：
- ✅ 框架合理：基础驾驶 + 自损惩罚 + 几何塑形 + 零和 + 课程，是对抗驾驶的标准组合，"全额自损惩罚"尤其关键（防自杀式攻击）。
- 🟠 **R-3 干扰信号偏弱**：cut_in(≤0.5)/blocking(≤0.3) 显著小于 base 的 progress×0.3(≈+1/步)，且 zero-sum 被 ×0.01 缩小 → 对抗 agent 单看奖励"正常开车"优于"切入/阻挡"，对抗行为难以学出。建议把 cut_in/blocking 提到与 base 同量级，或调大 zero-sum 缩放并依赖课程。
- 🟠 **R-2 与切入对冲**：对抗 agent 仍吃 `safety_lidar×0.3 + ttc×0.5` 的近距惩罚，而它的任务正是贴近对手 → 自相矛盾。建议对抗 agent 下调/豁免 safety/ttc。
- ⚠️ **v1 路径无对抗塑形**：v1 把 cut_in/blocking 置 0，对抗训练必须用 `reward_version='v2'`（multi_v2 文件头已注明）。

---

## 八、速度奖励改为 Toromanoff desired-speed（2026-06-08，最新）

**动机**：理论分析发现，无界 progress（`0.5×speed`/步，边际 +0.5/(m/s)）会压过超速惩罚（−0.2/(m/s)），"加大惩罚"需 `β>0.5 且去 floor 且更大以压时间偏好` → 高方差不稳定（曾出现 −4711 return）。根因是 progress 对速度无上界线性 → 最优是 bang-bang（顶满速度）。

**方案（采用 Toromanoff et al. 2020, CVPR, CARLA Challenge 冠军）**：用**期望速度带**取代无界 progress 作为主前进信号——带内取峰、过快/过慢都减，**没有独立的无界 progress 项**。方向由 centering/yaw 维持（直道上=朝终点）。

**改动**（`_compute_rewards_v2`）：
- `PROGRESS_WEIGHT` 5.0 → **0.0**（停用无界 progress；保留为可回退旋钮，>0 即混合模式）；
- 速度奖励改为梯形 desired-speed：
```python
SPEED_LOW  = 5.0/3.6    # 1.39 m/s (5 km/h)
SPEED_HIGH = 15.0/3.6   # 4.17 m/s (15 km/h)
SPEED_PEAK = 1.0        # 带内峰值（主前进信号）
OVERSPEED_SLOPE = 0.4   # 超上界每 m/s 下降，可转负
SPEED_FLOOR = -1.0
if   speed < SPEED_LOW:  speed_reward = SPEED_PEAK*(speed/SPEED_LOW)      # 0→峰，起步梯度
elif speed <= SPEED_HIGH: speed_reward = SPEED_PEAK                        # 5~15km/h 平台
else: speed_reward = SPEED_PEAK - OVERSPEED_SLOPE*(speed-SPEED_HIGH)       # 超速降转负
speed_reward = max(SPEED_FLOOR, speed_reward)
```

**曲线**：

| km/h | 0 | 2.5 | **5** | 10 | **15** | 18 | 20 | 24 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|
| speed_reward | 0 | +0.5 | **+1.0** | +1.0 | **+1.0** | +0.67 | +0.44 | 0 | −0.67 | −1.0 |

5–15 km/h 平台等价最优，24 km/h 起净惩罚。按时间偏好车会贴上沿 ~15km/h 巡航 = "稳定 ≤15km/h"。

**远距停滞复核（重点）**：desired-speed + centering + yaw 全部**与距离无关、每步即时**：
- 带内行驶（~15km/h，居中对准）每步净 ≈ **+1.55**；原地不动每步净 ≈ **−0.10** → 前进严格压倒不动，且远近车一致；
- 方向由 yaw（朝目标）+ centering（沿车道）维持，直道上即朝终点 → **不存在"远车看不到终点奖励而停滞"**；
- 配合已修的 截断 bootstrap + agent-id critic（远状态 value 无偏），far-vehicle 学习信号干净。

**注意 / 可调**：
- 平台 → 车按时间偏好贴 **上沿 15km/h**；若想稳定在更低（如 10km/h），把带改窄/单峰（`SPEED_LOW=SPEED_HIGH=10/3.6`）即可。
- progress 停用后**目标方向完全依赖 yaw/centering**；直道任务足够，若换复杂路线或发现冲终点意愿弱，可（a）适当加大 `yaw` 权重，或（b）回退为"封顶 progress + desired-speed"混合（`PROGRESS_WEIGHT>0` 且给 progress 按 `SPEED_HIGH` 封顶）。
- 改动改变训练动力学，需重训。

---

## 六、速度奖励简化（2026-06-08，已被第八节取代）

**动机**：原速度奖励 = 高斯曲线（峰 8 m/s≈29km/h）+ 独立 alive_bonus，偏复杂且目标速度偏快、无超速惩罚。需求：降目标速度让车更稳 + 加超速惩罚 + 简化。

**改动**（`_compute_rewards_v2`）：
- 删除高斯公式与 `speed_baseline`、删除独立 `alive_bonus`（折叠进新曲线，起步段即提供脱困梯度）；
- 改为**分段线性**：
```python
SPEED_TARGET = 15.0/3.6   # 4.17 m/s (15 km/h)
SPEED_PEAK   = 0.4
OVERSPEED_PENALTY = 0.2   # 每 m/s
SPEED_FLOOR  = -1.0
if speed <= SPEED_TARGET:
    speed_reward = SPEED_PEAK * (speed / SPEED_TARGET)          # 0→峰，线性，含起步梯度
else:
    speed_reward = SPEED_PEAK - OVERSPEED_PENALTY*(speed-SPEED_TARGET)  # 超速线性下降转负
speed_reward = max(SPEED_FLOOR, speed_reward)
```
- `motion_gate` 阈值常量化为 `MOTION_GATE_SPEED=2.0`（原引用 alive 的常量）；
- `details["alive"]` 保留为 0.0 占位（`train_no_rllib_mappo_robust_v2.py` 日志会读该键）。

**曲线**（验证值）：

| 速度 | 0 | 5 | 10 | **15(峰)** | 20 | 22 | 25 | 30 | 40 km/h |
|---|---|---|---|---|---|---|---|---|---|
| speed_reward | 0 | +0.13 | +0.27 | **+0.40** | +0.12 | +0.01 | −0.16 | −0.43 | −0.99 |

性质：静止=0（防 farm 保留）；峰值在 15 km/h；约 **22 km/h 为零点**，超过转净惩罚，下限 −1.0。

**注意**：此改动改变了合作智能体的训练动力学，旧 checkpoint 的策略会处于不同奖励地形下，需重新训练或微调。

---

## 七、车道保持改进（R-4，2026-06-08 已修改）

**现象**：训练出的车容易 **S 形画龙 / 频繁变道前进**。

**根因**：
1. 旧设计车道惩罚**只有"压线"离散事件**（lane_invasion 传感器，Broken −3/Solid −12/Curb −25），**偏离车道中心本身无连续惩罚**；车道内左右摆、压虚线变道几乎免费。
2. `map.get_waypoint()` 取**最近车道**中心，车滑入相邻车道后居中奖励按新车道"重新居中" → 变相允许变道。
3. 转向变化率惩罚过小（`STEER_RATE_WEIGHT=0.02`），压不住来回打方向。

**参考**：Toromanoff 2020(CARLA 冠军)/ Kendall 2019 / Stanley 代价 —— 车道保持普遍用**连续横向偏移 + 航向误差惩罚**，在压线前就给回正梯度。

**改动**（`_compute_rewards_v2`）：
- **新增连续横向偏离惩罚**（并入车道纪律项 `lane_invasion_penalty`，不新增 details 键）：
```python
LANE_DEV_WEIGHT = 0.5      # 每米
LANE_DEV_DEADZONE = 0.3    # 死区(米)
lane_dev_penalty = -LANE_DEV_WEIGHT * max(0.0, lateral_dist - LANE_DEV_DEADZONE)
lane_invasion_penalty += lane_dev_penalty     # 与离散压线惩罚叠加
```
- **加大转向变化率惩罚**：`STEER_RATE_WEIGHT` 0.02 → **0.05**（直接压制方向盘来回打）。

**效果**（居中奖励 + 偏离惩罚 合成的车道保持净信号）：

| 横向偏移 m | 0 | 0.3 | 0.5 | 0.8 | 1.0 | 1.5 | 1.75(压线处) |
|---|---|---|---|---|---|---|---|
| 净信号 | +0.50 | +0.30 | +0.03 | −0.24 | −0.35 | −0.60 | −0.73(+离散压线) |

形成围绕车道中心的"势能阱"：偏离越多越罚，S 形画龙不再免费。

**注意**：
- 横向距离用的是**最近车道**中心（`lateral_dist`），对"车道内画龙"和"压线/straddle 时刻"都有效；若需严格"锁定原车道、禁止任何变道"，可改用目标车道轴（target waypoint 朝向）做参考，告知我即可换。
- 改动改变训练动力学，需重训。可调旋钮：`LANE_DEV_WEIGHT / LANE_DEV_DEADZONE / STEER_RATE_WEIGHT`。

---

## 五、后续建议（未自动改）
1. **R-3**：提高 cut_in/blocking 权重或 zero-sum 缩放（改动训练动力学，需实验确认）。
2. **R-2**：safety_lidar 限定侧/后扇区或下调权重；对抗 agent 豁免 safety/ttc。
3. **精简合作塑形项**：合并 speed/alive、approach 已弱化可考虑并入 progress。
4. **R-5（可选）**：success 势能化，强化显式终点偏好。
