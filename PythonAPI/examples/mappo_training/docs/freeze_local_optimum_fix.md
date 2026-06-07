# 破除「原地不动」局部最优 改动记录（R5）

> 日期：2026-06-04
> 涉及文件：
> - `environment/carla_multi_agent_env.py`（reward v2 + reset）
> - `models/actor_critic.py`（Actor log_std 下限）
> - `training/train_no_rllib_mappo_robust_v2.py`（课程回退 + 日志）

## 1. 问题背景

训练（尤其在引入「按距离分层的课程学习」后）反复收敛到一个**规避风险的局部最优：车辆原地不动**。
即使在最简单的短距离阶段也会发生，是非常糟糕的训练结果。

### 根因分析（v2 reward 下「一直停在原地」的收支）

| 分量 | 停止时取值 | 说明 |
|------|-----------|------|
| progress | 0 | 无位移 |
| speed / alive | 0 | speed=0，减 baseline 后 clamp 到 0 |
| centering / yaw | 0 | 被 `motion_gate=0` 清零 |
| safety_lidar | 0〜负 | 窄路前后车近时为负 |
| collision | 0 | 不撞即 0 |
| step | -0.1 | |
| **合计** | **≈ -0.1/步 → 每回合 -100（timeout）** | |

而「往前开然后撞车」是 **-300**。于是 **停在原地(-100) 比撞车(-300) 更"安全"**。
在尚未学会路线的探索初期，一动就被碰撞/跨线扣分，梯度被推向"刹车"方向；一旦
`log_std` 收缩，策略被锁死在"冻结"局部最优盆地里。

主要成因：
1. **风险不对称**：撞车 -300 ≫ 成功 +100（下行是上行的 3 倍）。
2. **progress 的 efficiency 门控**：低速前进只拿 30% 前进奖励，"胆怯前进"几乎无正反馈。
3. **缺少把车"吸"向终点的连续正向信号**：终点 +100 太稀疏，探索到不了就一次都体验不到。
4. **熵塌缩**：`log_std` 早期收缩后探索消失，无法逃出。
5. **距离课程**：短距离阶段同样诱发冻结（场景单一 + 上述动力学）。

## 2. 解决思路（约束）

按用户要求：
- **去掉距离课程**，回退到之前的 `basic / left|right / all` 课程。
- 终点**不使用"惩罚"形式**，改用**正向奖励**引导（不加"未到达扣分"）。
- 跨线惩罚**加重**（强化车道纪律）。

总策略：**强化正向前进/接近奖励 + 缩小负向风险 + 防熵塌缩 + 回退课程**，完全不含终端惩罚。

## 3. 具体改动

### 3.1 回退距离课程（`train_no_rllib_mappo_robust_v2.py`）
- 新增命令行参数 `--curriculum-mode {classic, distance}`，**默认 `classic`**。
- `CurriculumLearning` 增加 `mode` 参数；`_build_default_stages(..., use_distance=...)`：
  - `classic`：忽略 `dist_<N>m` 分层，**并把 dist_*.json 从场景池中排除**，
    使行为完全等同"加距离课程之前"的版本（basic / left|right / all 三段）。
  - `distance`：保留原距离分层逻辑（实验性，不推荐）。

### 3.2 降低撞车惩罚（`carla_multi_agent_env.py::_compute_rewards_v2`）
- `COLLISION_PENALTY`: **-300.0 → -150.0**。
- 缩小探索下行风险（是"减小惩罚"而非"增加惩罚"），仍保持
  **撞车(-150) < 正常行驶 < 成功** 的排序。

### 3.3 去掉 progress 的 efficiency 门控（同上）
- 原 `progress * PROGRESS_WEIGHT * (0.3 + 0.7*efficiency) * proximity_factor`
  → **`progress * PROGRESS_WEIGHT`（满额）**。
- 任何前进都立即拿到正比于位移的正奖励，使"移动"严格优于"停住"。
  （`proximity_factor` 此前 BOOST 已为 0，等价移除。）

### 3.4 新增连续「接近终点」正向势能奖励（同上）
- `approach_reward = APPROACH_WEIGHT * clamp((d0 - cur) / d0, 0, 1)`，`APPROACH_WEIGHT=0.5`。
  - `d0` = 回合起始到终点距离（reset 时记录于 `self.initial_distances`，已防除零）。
  - 越靠近终点该项越大（∈[0,0.5]），形成把车"吸"向终点的连续正向吸引子。
  - 势能塑形、被初始距离归一化、有界，**不会 farm、不是惩罚**。
- 已加入 reward 求和与 `reward_details["approach"]`（两个早返回分支也补了键）。

### 3.5 加重跨线惩罚（同上，第 9 项 `lane_invasion`）
| 线型 | 旧值 | 新值 |
|------|------|------|
| Broken（虚线） | -1.0 | **-3.0** |
| Solid（实线） | -5.0 | **-12.0** |
| Curb（路沿） | -10.0 | **-25.0** |
| 未知/默认 | -5.0 | **-12.0** |
- 注意：加重跨线惩罚与"破冻结"略有张力（静止车永不跨线 = 间接奖励不动）；
  靠 3.2–3.4 的正向前进/接近奖励 + 3.6 的探索下限来抵消，保证在车道内
  正常行驶的净收益仍显著为正。

### 3.6 Actor log_std 下限，防熵塌缩（`models/actor_critic.py`）
- `Actor.__init__` 新增 `min_log_std=-1.0`；`forward` 中 `log_std.clamp(min=min_log_std)`。
- `std ≥ 0.37`，保证持续探索噪声，避免策略过早收敛到"刹车/不动"。

### 3.7 日志增强（`train_no_rllib_mappo_robust_v2.py`）
- 打印行新增 `approach` 与 `lane` 两个 reward 分量均值。
- `reward/approach` 会自动写入 TensorBoard（沿用现有逐分量记录逻辑）。

## 4. 运行方式

```bash
cd PythonAPI/examples/mappo_training
# 默认即 classic（已回退距离课程）
python -m training.train_no_rllib_mappo_robust_v2 --render
# 如需对照实验性距离课程：
python -m training.train_no_rllib_mappo_robust_v2 --curriculum-mode distance
```

## 5. 验证指标（改前/改后对比）

短距离/基础场景跑若干 update，看 TensorBoard / 控制台：
- `avg_ep_len` 是否仍 ≈ `max_steps` 且 `progress≈0`（冻结特征）→ 应显著下降。
- `episode/rate_success_to_goal` 是否开始出现非零。
- `train/log_std_agent*` 是否塌缩（应被 -1.0 下限托住）。
- reward 分量：`progress / approach` 应为正且可观，`collision / lane` 触发频率应可控。

## 6. 后续可调旋钮

- `APPROACH_WEIGHT`（0.5）：接近奖励强度，若仍冻结可上调到 1.0。
- `COLLISION_PENALTY`（-150）：若出现"轰油门硬撞"可回调到 -200。
- 跨线 `severity_map`：若车辆过度保守贴右行驶可适度回调。
- `min_log_std`（-1.0）：探索不足可放宽到 -0.7。
