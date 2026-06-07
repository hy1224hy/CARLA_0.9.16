# 奖励函数优化方案

**日期**: 2026-05-13  
**状态**: Phase 1 ✅ + Phase 2 ✅ + A/B 实验基础设施 ✅  
**目标**: 将原始二元阶梯式奖励函数升级为全连续、提供稠密梯度的奖励函数

---

## 一、原始奖励函数诊断

### 当前奖励（`environment/carla_multi_agent_env.py:_compute_rewards`）

| 分量 | 计算方式 | 值域/步 | 核心问题 |
|------|---------|--------|---------|
| progress | `Δ距离 × 20.0` | 0 ~ +40 | 仅依赖位移量，不区分快慢 |
| speed | 三段 if/elif/else（停/-0.1, 正常/+0.1, 超速/-1.0） | -1.0 ~ +0.1 | **二元阶梯**，无梯度过渡 |
| centering | 横向偏差 < 5cm → +0.5, else 0 | 0 或 +0.5 | **全或无**，6cm与2m惩罚相同 |
| yaw | 偏差 < 0.2rad → +0.1, else 0 | 0 或 +0.1 | **二元阶梯** |
| collision | -100 | -100 或 0 | 缺少碰撞风险的中间信号 |
| lane_invasion | -5 | -5 或 0 | 不区分压虚线/实线/路沿 |
| step | -0.05 | 固定 | 太小，无有效时间压力 |
| success | +100 | 稀疏 | 快到达和慢到达奖励相同 |

### 根本问题

四个形状分量（speed, centering, yaw, lane_invasion）都是**二元阶梯函数**，Critic 无法学习精准的 value 估计——"差一点就对"和"完全不对"在价值上无法区分。

---

## 二、优化方案：三维度设计

### 设计原则
- **处处连续可导**：所有形状分量提供从好到坏的平滑梯度
- **scale 分离**：progress 主导（8~20/步），辅助信号合计 0.3~1.0/步，安全惩罚仅在风险时激活
- **向后兼容**：reward_details 的 key 集合只增不减

### 2.1 驾驶效率 (Driving Efficiency)

#### 2.1.1 效率加权进度奖励
```python
target_speed = 8.0  # m/s ≈ 29 km/h
efficiency = min(speed / target_speed, 1.0)
progress_reward = progress * 20.0 * (0.3 + 0.7 * efficiency)
# 1 m/s → 每米 7.7 分（保底）
# 8 m/s → 每米 20.0 分（全额）
```

#### 2.1.2 连续速度奖励（高斯曲线）
```python
speed_reward = 0.5 * exp(-(speed - 8.0)^2 / (2 × 4.0^2))
# speed=0  → 0.07
# speed=4  → 0.30
# speed=8  → 0.50（峰值）
# speed=12 → 0.30
# speed=20 → 0.005
```

#### 2.1.3 时间效率终点奖励
```python
time_bonus = max(0, (max_steps - steps_used) / max_steps) × 50.0
```

### 2.2 行驶质量 (Driving Quality)

#### 2.2.1 连续车道居中（指数衰减）
```python
centering_reward = 0.5 × exp(-lateral_dist^2 / (2 × 0.3^2))
# 0.00m → 0.500
# 0.10m → 0.473
# 0.30m → 0.303
# 0.60m → 0.068
```

#### 2.2.2 驾驶平滑性惩罚
```python
jerk_penalty    = -0.05 × |Δthrottle| / Δt    # 加速度突变
steer_penalty   = -0.02 × |Δsteer| / Δt       # 转向速率
```

#### 2.2.3 连续航向奖励（余弦相似度）
```python
yaw_reward = 0.15 × (cos(angle_diff) + 1) / 2
# 0°   → 0.150
# 90°  → 0.075
# 180° → 0.000
```

### 2.3 安全裕度 (Safety Margin)

#### 2.3.1 LiDAR 安全距离惩罚
```python
safe_threshold = 0.5  # 实际距离 2.0m
for each lidar_ray:
    if ray_distance < safe_threshold:
        penalty += -0.3 × (safe_threshold - ray_distance) / safe_threshold
```

#### 2.3.2 TTC 前向碰撞风险惩罚 ✅ 已实现
```python
for each vehicle in front (same lane, within 10m):
    rel_speed = ego_speed - other_speed
    if rel_speed > 0:
        ttc = longitudinal_dist / rel_speed
        if ttc < 3.0s: penalty += -2.0 × exp(-ttc / 1.0)
# TTC=0.5s → -1.21, TTC=1.0s → -0.74, TTC=2.0s → -0.27, TTC=3.0s → -0.10
```

#### 2.3.3 分级车道偏离惩罚
```python
Broken (虚线) → -1.0
Solid  (实线) → -5.0
Curb   (路沿) → -10.0
```

---

## 三、权重平衡总览

| 分量 | 原 | 新 | 每步典型值 |
|------|----|----|-----------|
| progress | ×20.0 | ×20.0 × efficiency | +8 ~ +20 |
| speed | ±0.1 三段 | ×0.5 高斯 | +0.1 ~ +0.5 |
| centering | +0.5 二元 | ×0.5 exp衰减 | +0.01 ~ +0.5 |
| yaw | +0.1 二元 | ×0.15 余弦 | +0.05 ~ +0.15 |
| jerk_penalty | — | ×0.05 | 0 ~ -0.3 |
| steer_penalty | — | ×0.02 | 0 ~ -0.1 |
| safety_lidar | — | ×0.3 | 0 ~ -1.6 |
| ttc | — | ×2.0 exp | 0 ~ -2.0 |
| lane_invasion | -5 统一 | 分级(-1/-5/-10) | -1 ~ -10 |
| step | -0.05 | -0.05 | -0.05 |
| collision | -100 | -100 | -100 |
| success | +100 | +100 + time_bonus | +100 ~ +150 |
| **正常每步合计** | | | **+8 ~ +21** |

---

## 四、实现路线图

### Phase 1（本次实现）✅ 已完成
- [x] 连续速度奖励（高斯曲线）
- [x] 连续居中奖励（指数衰减）
- [x] 连续航向奖励（余弦相似度）
- [x] 效率加权进度奖励
- [x] 驾驶平滑性惩罚（jerk + steer rate）
- [x] LiDAR 安全距离惩罚
- [x] 分级车道偏离惩罚
- [x] 时间效率终点奖励

### Phase 2（后续）✅ 已完成
- [x] TTC 前向碰撞风险惩罚
- [x] 对抗智能体奖励函数的同步更新（已纳入所有新分量）
- [x] A/B 对比实验基础设施

### Phase 3（待执行）
- [ ] 运行 v1/v2 对比训练实验
- [ ] 分析 TensorBoard 学习曲线差异
- [ ] 根据实验结果调优超参数

---

## 五、实验记录

| 日期 | 实验 | 结果 | 备注 |
|------|------|------|------|
| 2026-05-13 | Phase 1 实现 | — | 完成全部8项奖励分量的连续化改造 |
| 2026-05-13 | Phase 2 TTC | — | 新增 TTC 前向碰撞风险惩罚 |
| 2026-05-13 | A/B 基础设施 | — | 添加 reward_version 切换 + 对比评估脚本 |

---

## 六、实现备注

### 修改的文件
- `environment/carla_multi_agent_env.py`
  - `__init__`: 新增 `self.prev_controls = {}`
  - `reset()`: 初始化 `self.prev_controls`，更新 episode 平均详情的 key 列表
  - `step()`: 每步末尾保存当前控制量到 `self.prev_controls`
  - `_compute_rewards()`: 完全重写，新增 5 个连续分量（efficiency-progress, gaussian-speed, exp-centering, cosine-yaw, lidar-safety），1 个 TTC 碰撞风险惩罚，2 个平滑性惩罚（jerk, steer-rate），分级 lane_invasion，time_bonus
  - `_compute_adversarial_reward()`: 纳入所有新分量（jerk/steer_rate/safety_lidar/ttc/time_bonus）

### 向后兼容性
- `reward_details` 的旧 key 全部保留，只新增了 `jerk`, `steer_rate`, `safety_lidar`, `ttc`, `time_bonus`
- 对抗奖励计算的 `.get(key, 0.0)` 使用默认值，兼容旧版 reward_details
- TensorBoard 日志通过 `details.items()` 动态迭代，自动覆盖新键

### TTC 惩罚设计细节
- **判定条件**: 同车道（横向 < 2m）、正前方、纵向距离 < 10m、自车更快
- **惩罚函数**: 指数衰减 `-2.0 × exp(-ttc / 1.0)`，TTC 越小惩罚越重
- **典型值**: TTC=0.5s → -1.21, TTC=1.0s → -0.74, TTC=2.0s → -0.27, TTC=3.0s → -0.10
- **与其他安全信号的互补关系**:
  - LiDAR safety: 全向360°，仅关心近距离（<2m），捕获侧面和后方的即时碰撞风险
  - TTC: 仅前向，但可提前感知远达 10m 的追尾风险
  - Collision: 稀疏的 -100 兜底

### 超参数调优建议
- 若策略过于保守（速度上不去）：增大 `TARGET_SPEED` 或减小 `SIGMA_SPEED`
- 若策略频繁压线：增大 `SAFETY_LIDAR_WEIGHT` 或减小 `SAFE_THRESHOLD`
- 若控制震荡：增大 `JERK_WEIGHT` / `STEER_RATE_WEIGHT`
- 若策略不愿前进：检查 progress 保底系数 `0.3`，可适当提高
- 若策略跟车过近（追尾风险）：增大 `TTC_WEIGHT` 或 `TTC_THRESHOLD`

### 修改的文件（Phase 3 更新）
- `environment/carla_multi_agent_env.py`:
  - `__init__`: 新增 `reward_version` 参数
  - `_compute_rewards()`: 变为调度器，根据 `self.reward_version` 分发
  - `_compute_rewards_v1()`: 原始二元阶梯奖励（保留用于 A/B 对比）
  - `_compute_rewards_v2()`: 优化版全连续梯度奖励
- `training/train_no_rllib_mappo.py`: 新增 `--reward-version` 参数
- `training/train_no_rllib_mappo_robust.py`: 新增 `--reward-version` 参数
- `evaluation/compare_reward_versions.py`: **新建** — A/B 对比评估脚本

---

## 七、A/B 对比实验指南

### 7.1 实验流程

```
  ┌─────────────────────────────────────────────────────┐
  │  Step 1: 用 v1 和 v2 分别训练两个 checkpoint        │
  │  Step 2: 用对比评估脚本进行 head-to-head 评估        │
  │  Step 3: 分析指标差异，判断优化有效性               │
  └─────────────────────────────────────────────────────┘
```

### 7.2 Step 1 — 并行训练

**终端 A — v1（原始奖励）训练：**
```bash
cd PythonAPI/examples/mappo_training
python -m training.train_no_rllib_mappo \
    --reward-version v1 \
    --save-dir ./logs/ab_v1_ckpts \
    --updates 500 \
    --seed 42 \
    --render
```

**终端 B — v2（优化奖励）训练：**
```bash
cd PythonAPI/examples/mappo_training
python -m training.train_no_rllib_mappo \
    --reward-version v2 \
    --save-dir ./logs/ab_v2_ckpts \
    --updates 500 \
    --seed 42 \
    --render
```

> **注意**: 两个终端需要连接**同一个 CARLA 服务器**，或分别启动两个 CARLA 实例（不同 port）。

**同时监控 TensorBoard：**
```bash
tensorboard --logdir ./logs/
# 对比指标曲线:
#   - Agent_0/Reward/Total（每步奖励趋势）
#   - Agent_0/Reward/Speed, Agent_0/Reward/Centering...（分量分布）
#   - Agent_0/Episode_Reward/Average（episode 回报收敛曲线）
#   - Agent_0/Reward_Percentage/...（各分量占比变化）
```

### 7.3 Step 2 — 对比评估

**取 v1 和 v2 训练完成后各自最优的 checkpoint，运行 head-to-head 对比：**

```bash
cd PythonAPI/examples/mappo_training

# 在默认直道上对比 50 个 episode
python -m evaluation.compare_reward_versions \
    --checkpoint-a ./logs/ab_v1_ckpts/ckpt_500.pt \
    --label-a "v1 (原始奖励)" \
    --checkpoint-b ./logs/ab_v2_ckpts/ckpt_500.pt \
    --label-b "v2 (优化奖励)" \
    --episodes 50

# 使用随机场景进行更全面的泛化性评估
python -m evaluation.compare_reward_versions \
    --checkpoint-a ./logs/ab_v1_ckpts/ckpt_500.pt \
    --label-a "v1 (原始奖励)" \
    --checkpoint-b ./logs/ab_v2_ckpts/ckpt_500.pt \
    --label-b "v2 (优化奖励)" \
    --episodes 100 \
    --use-random-scenarios

# 保存结果到 JSON 便于后续分析
python -m evaluation.compare_reward_versions \
    --checkpoint-a ./logs/ab_v1_ckpts/ckpt_500.pt \
    --checkpoint-b ./logs/ab_v2_ckpts/ckpt_500.pt \
    --episodes 100 --use-random-scenarios \
    --output-json ./logs/ab_results.json
```

### 7.4 评估指标说明

| 指标 | 含义 | 期望方向 |
|------|------|---------|
| 成功率 | 到达终点的比例 | ↑ 越高越好 |
| 碰撞率 | 发生碰撞的比例 | ↓ 越低越好 |
| 平均速度 (km/h) | 行驶速度均值 | ↑ 适度提高（不超速） |
| 平均横向偏差 (m) | 偏离车道中心线的平均距离 | ↓ 越小越居中 |
| 每ep车道偏离次数 | 压线/偏离频率 | ↓ 明显改善 |
| 平均episode步数 | 到达终点所用步数 | ↓ 越少越高效 |
| 平均episode return | v2 奖励衡量的综合得分 | ↑ 越高越好 |

### 7.5 判断优化有效的标准

以下指标**同时满足**即可认为 v2 优于 v1：

1. **安全改善**：碰撞率下降，或横向偏差/车道偏离下降
2. **效率不降**：平均速度不显著下降（下降 < 15%），episode 步数减少或持平
3. **学习效率**：TensorBoard 中 v2 的 episode return 收敛更快或最终值更高

### 7.6 单 checkpoint 评估（不对比）

```bash
# 单独评估一个 checkpoint 的质量指标
python -m evaluation.compare_reward_versions \
    --checkpoint ./logs/ab_v2_ckpts/ckpt_500.pt \
    --episodes 50 --use-random-scenarios \
    --output-json ./logs/v2_eval.json
```

### 7.7 稳健性训练评估

```bash
# v1 稳健性训练
python -m training.train_no_rllib_mappo_robust \
    --reward-version v1 --seed 42 --updates 500

# v2 稳健性训练
python -m training.train_no_rllib_mappo_robust \
    --reward-version v2 --seed 42 --updates 500
```

### 7.8 注意事项

1. **相同随机种子**：训练时两版本使用相同 `--seed` 确保初始化一致
2. **独立 CARLA 实例**：两终端训练需连接不同 port 的 CARLA（分别启动 `CarlaUE4.exe -carla-port=2000` 和 `2001`）
3. **评估时用相同场景**：对比评估脚本自动对两个 checkpoint 使用相同的 scenario 序列
4. **先验 checkpoint**：如果有 pretrained checkpoint，可同时作为两版本训练的起点加速对比
