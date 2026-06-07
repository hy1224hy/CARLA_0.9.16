# MAPPO Robust v2 训练 — Bug 修复记录

按时间顺序列出（从最早到最新）。每条 bug 包含：现象 / 根因 / 改动位置 / 验证方式。

---

## 目录

| #                                                     | 主题                                          | 主要文件                                          |
| ----------------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| [1](#1-2025528-collision-统计恒为-0)                      | collision 统计恒为 0                            | `training/...v2.py`                           |
| [2](#2-2026528-phase-a-系列ep_success--gae--kl--bestpt) | Phase A 系列（ep_success / GAE / KL / best.pt） | `training/...v2.py`, `models/actor_critic.py` |
| [3](#3-2026528-phase-b2r4-reward-量级失衡)                | Phase B2：R4 reward 量级失衡                     | `environment/carla_multi_agent_env.py`        |
| [4](#4-2026528-safety_lidar--ttc-权重回退)                | safety_lidar / TTC 权重回退                     | `environment/carla_multi_agent_env.py`        |
| [5](#5-2026528-原地不动局部最优)                              | "原地不动"局部最优                                  | `environment/carla_multi_agent_env.py`        |
| [6](#6-2026528-lidar-self-detection-bug)              | Lidar self-detection bug                    | `environment/carla_multi_agent_env.py`        |
| [7](#7-2026528-proximity-加权-progress--探索超参)           | proximity 加权 progress + 探索超参                | `environment/...env.py`, `training/...v2.py`  |
| [8](#8-2026528-tensorboard-曲线重叠)                      | TensorBoard 曲线重叠                            | `training/...v2.py`                           |
| [9](#9-2026528-续训功能---resume)                         | 续训功能 `--resume`                             | `training/...v2.py`                           |
| [10](#10-2026531-issue-1到点后右滑姿态不正)                    | Issue 1：到点后右滑姿态不正                           | `environment/carla_multi_agent_env.py`        |
| [11](#11-2026531-issue-2人行横道压线漏判)                     | Issue 2：人行横道压线漏判                            | `environment/carla_multi_agent_env.py`        |
| [12](#12-2026531-_check_dones-状态机抖动)                  | `_check_dones` 状态机抖动                        | `environment/carla_multi_agent_env.py`        |

---

## 1. collision 统计恒为 0

### 现象

TB 上 `rate_collision` 一直 0%，但 CARLA 日志会打"检测到严重碰撞，结束本回合"。

### 根因

CARLA env 在每个 `step()` 末尾把 `self.collision_events[i] = False` 和 `crosswalk_invasion_events[i] = False` 清零（[carla_multi_agent_env.py:978-984](../environment/carla_multi_agent_env.py#L978-L984)），目的是"避免重复惩罚"。所以 episode 跳出 `for step` 循环后，读到的 `env.collision_events` 永远是 0。

### 改动位置

`training/train_no_rllib_mappo_robust_v2.py` rollout 循环：
```python
# 进 episode 前
ep_collision_sticky: Dict[int, bool] = {aid: False for aid in agent_ids}

# 每个 step env.step() 之后
step_collisions = info.get('collisions', {}) or {}    # env 清零前的快照
for aid in agent_ids:
    if step_collisions.get(aid, False):
        ep_collision_sticky[aid] = True

# Episode 末尾用 ep_collision_sticky 替代 env.collision_events
```

### 经验

**`env.reached_target` 是 sticky 的**（只在 spawn 时清零）→ 可放心在 episode 末尾读。
**任何 step 内 reset 的事件性字段**（collision / lane_invasion / crosswalk）→ 必须在循环里就累积。

`info['collisions']` 已经把 collision_events 和 crosswalk_invasion_events OR 在一起（见 `_check_collisions`），所以一个 sticky 字典就够。

---

## 2. Phase A 系列（ep_success / GAE / KL / best.pt）

### A1：`ep_success` 把 timeout 当成功

**根因**：env 在 `current_step >= max_steps` 时把所有 agent done=True；旧代码 `ep_success = not any(info['collisions'])`，超时无碰撞 = 误判为成功 → 课程学习反向工作。

**改动**：`training/train_no_rllib_mappo_robust_v2.py` 严格 3 类终止：
```python
ep_success_to_goal = all(env.reached_target.get(aid, False) for aid in agent_ids)
ep_collision = any(本回合内任一 agent 撞过)   # 见 #1 sticky 修复
ep_timeout = (ep_length >= args.max_steps) and not ep_success_to_goal and not ep_collision
ep_success = ep_success_to_goal
```

TB 新增 `episode/rate_success_to_goal` / `rate_collision` / `rate_timeout` 三条独立曲线。

### A2：collision 广播 done 时 GAE bootstrap = 0

**根因**：env 任一 agent 严重碰撞时全员 done=True。对未撞 agent 而言这是"被截断"，但原 v2 的 `episode_truncated` 只检测 `max_steps`，导致未撞 agent 的 GAE bootstrap = 0 → Critic 系统性低估 V → 学到"过度防御"。

**改动**：episode 末按 per-agent 自然终止判定：
```python
naturally_terminal = own_reached or own_crashed   # 只有自己撞 / 到点
if naturally_terminal:
    buffer.add_episode_bootstrap(aid, None)       # V=0
else:
    buffer.add_episode_bootstrap(aid, last_gs)    # V(last_global_state) bootstrap
```

### A3：PPO 步长失控

**根因**：`lr=3e-4, ppo_epochs=5, clip_eps=0.2, log_std=0`（std=1.0 → tanh 后频繁饱和到 ±1），且无 KL 早停、无 lr 衰减。

**改动**：
- `models/actor_critic.py`：`Actor.log_std` 初始值 `0.0 → -0.5`（std≈0.6）
- `training/train_no_rllib_mappo_robust_v2.py`：
  - 新增 `--target-kl`（默认 0.02），单 agent 单 epoch `approx_kl > 1.5 × target_kl` 触发 break
  - `approx_kl` 改用 Joschu 公式 `((ratio - 1) - log ratio).mean()`（无偏低方差近似）
  - 新增 lr 线性衰减（`--lr-decay/--no-lr-decay`，默认开）
  - `--clip-eps` 默认 `0.2 → 0.15`
  - TB 新增 `train/lr_actor`、`train/lr_critic`、`train/log_std_agentX`、`ppo/stop_epoch_agentX`

### A4：best.pt 跟踪

**改动**：每个 update 结束维护 `best_success_rate`，按窗口严格成功率取 max 时保存 `best.pt`。评估永远拿 `best.pt`。

---

## 3.  Phase B2：R4 reward 量级失衡

### 现象

Phase A 修复后日志里 `avg_return=119.70, success=0.00%, collision=100.00%, avg_ep_len=30`：30 步内必撞但 return 正。agent 学到"轰油门往前撞"。

### 根因

按 30 步 episode 算：
- 单步 progress 奖励 ≈ `0.8 × 20 × 0.72 ≈ 11.5`
- 30 步累积 ≈ +345
- 碰撞 -100
- 净 episode return ≈ +245

撞车比正常行驶（200 步走完）的中间状态收益还高，撞车是 agent 的理性选择。

### 改动位置

[carla_multi_agent_env.py:1145-1167](../environment/carla_multi_agent_env.py#L1145-L1167)

| 项 | 旧 | 新 | 理由 |
|---|---|---|---|
| `PROGRESS_WEIGHT` | 20.0 | **5.0** | 让 30 步速冲 progress 累积 < collision -300 |
| `COLLISION_PENALTY` | -100 | **-300** | 必须大于"撞前累积 progress" |

数量级目标（按 30 步 ep_len 算）：
- 撞车：progress +85 + collision -300 ≈ **-215**
- 完成（200 步）：progress +580 + success +120 ≈ **+700**
- 差距 ≈ 915

并新增控制台 reward 分量打印：
```
reward分量(均值/步): progress=0.243, safety+ttc=-0.082, collision=-1.250, ...
```

---

## 4. safety_lidar / TTC 权重回退

### 现象

[#3](#3-2026528-phase-b2r4-reward-量级失衡) 把 SAFETY_LIDAR_WEIGHT 调到 1.0、TTC_WEIGHT 调到 5.0 后，update 7 出现 `avg_return=-4711, safety+ttc=-11.000/步` 的灾难性数据。

### 根因

safety_lidar 是 **11 条 ray 求和**：
```python
for d in lidar_dists:
    if d < SAFE_THRESHOLD:
        safety_penalty += -SAFETY_LIDAR_WEIGHT * (SAFE_THRESHOLD - d) / SAFE_THRESHOLD
```

11 × 1.0 = max -11/步。Town02 窄路 + 两车靠近时几乎所有 ray 都触发，agent 被罚到无法动弹。

（事后发现根本原因是 [#6 lidar self-detection bug](#6-2026528-lidar-self-detection-bug)，但当时先做了权重回退）

### 改动位置

[carla_multi_agent_env.py:1145-1167](../environment/carla_multi_agent_env.py#L1145-L1167)

```python
SAFETY_LIDAR_WEIGHT = 0.3   # 回退到原值（曾经误调到 1.0）
TTC_WEIGHT_NEW = 2.0        # 回退到原值（曾经误调到 5.0）
COLLISION_PENALTY = -300.0  # 保留 R4 的主修
```

### 经验教训

R4（撞了反而赚）的核心修法是 **COLLISION_PENALTY: -100 → -300**，连续惩罚不需要也跟着放大。一次性惩罚和连续惩罚要分开调，混在一起会过度激进。

---

## 5. "原地不动"局部最优

### 现象

经过 [#4](#4-2026528-safety_lidar--ttc-权重回退) 和后来 [#6](#6-2026528-lidar-self-detection-bug) 修复后，距目标点近的 leader 车辆几乎不动。

### 根因

按当时 reward 设置算账：

| 策略 | 1000 步内 | progress | speed | step_pen | **总计** |
|---|---|---|---|---|---|
| 完全不动 | 1000 | 0 | +68 | -50 | **+18** |

- `step_penalty=-0.05/step` 跑满 timeout 只 -50
- `speed_reward = 0.5*exp(-(speed-8)²/32)` 在 speed=0 时仍是 +0.068/step，1000 步 = +68
- 两者相加 +18 → 站立比撞车（-300）和大部分早期探索都好
- 配合 R3（共享 critic 噪声 + log_std 持续下降），agent1 的策略坍缩到"踩死刹车"

leader 没有 TTC 压力（TTC 只看正前方，agent0 在它后面），后方 lidar 也是 4m 内才触发，所以 agent1 站着不动几乎零负反馈。

### 改动位置

**`environment/carla_multi_agent_env.py`**：

1. `speed_reward` 减 baseline 后 clamp，speed=0 严格为 0：
```python
speed_baseline = math.exp(-(TARGET_SPEED ** 2) / (2 * SIGMA_SPEED ** 2))
speed_reward = 0.5 * max(0.0,
    math.exp(-((speed - TARGET_SPEED) ** 2) / (2 * SIGMA_SPEED ** 2)) - speed_baseline)
```

2. 新增 `alive_bonus`（0~0.3/step 线性 ramp，2 m/s 封顶）：
```python
ALIVE_BONUS_MAX = 0.3
ALIVE_BONUS_SPEED = 2.0
alive_bonus = ALIVE_BONUS_MAX * min(1.0, speed / ALIVE_BONUS_SPEED)
```

3. `step_penalty: -0.05 → -0.5`（timeout 跑满 -500，明显比撞车 -300 还差）

4. 两处 early-exit 的 reward_details 加 `"alive": 0.0` 占位

**`training/train_no_rllib_mappo_robust_v2.py`**：控制台 reward 分量打印加 `speed / alive / step` 三项

### 修后数量级核对

| 策略 | step | progress | speed | alive | collision | success | step_pen | **总计** |
|---|---|---|---|---|---|---|---|---|
| 完全不动 1000 步 | 1000 | 0 | 0 | 0 | 0 | 0 | -500 | **-500** ✓ |
| 慢速 3 m/s 到点 | 293 | +247 | +47 | +88 | 0 | +135 | -146 | **+371** ✓ |
| 匀速 8 m/s 到点 | 110 | +440 | +48 | +33 | 0 | +145 | -55 | **+611** ✓ |
| 撞车 30 步 | 30 | +86 | +13 | +9 | -300 | 0 | -15 | **-207** ✓ |

层次：快到点 > 慢到点 > 撞车 > 不动。

---

## 6.  Lidar self-detection bug

### 现象

控制台 `safety+ttc` 分量稳定在 `-3.300/步`（完美 saturate），即使车辆远离任何障碍物也不下降。observation 里 11 维 lidar 全是 ~0。

### 根因

[carla_multi_agent_env.py:_get_ray_detection](../environment/carla_multi_agent_env.py#L707) 的"自过滤"逻辑：
```python
if detection.label == vehicle.id:    # ← 错误
    continue
```

CARLA 的 `LabelledPoint.label` 是 **CityObjectLabel 枚举**（小整数：Vehicles=10, Wall=12 等），不是 actor ID。所以这个比较**永远不匹配**，11 条 ray 全打在自己的车身上（车长约 2m、车宽 1m，ray 从 vehicle center 发出，每个方向都会先撞到自己的车身）。

### 影响

- observation 里 11 维 lidar 全是 ~0（无意义信号），critic / actor 拿不到真实"附近障碍"信息
- reward 永远 -3.3/step 误导策略向"原地不动"
- 解释了 [#4](#4-2026528-safety_lidar--ttc-权重回退) 为什么连原值 0.3 都会持续大幅负

### 改动位置

[carla_multi_agent_env.py:707-764](../environment/carla_multi_agent_env.py#L707-L764)

用 vehicle bounding box 推算车体外接圆半径，detection 在此半径内直接丢弃：
```python
try:
    extent = vehicle.bounding_box.extent
    self_radius = math.sqrt(extent.x ** 2 + extent.y ** 2) + 0.15
except Exception:
    self_radius = 1.5
...
for detection in detections:
    dist = start_loc.distance(detection.location)
    if dist < self_radius:
        continue   # 过滤自身
    if dist < det_dist:
        det_dist = dist
```

### 验证

跑 1~3 update 立刻看 `safety+ttc`：
- 修后应 ∈ [-0.5, 0]（只有侧方墙偶尔触发）
- 如果还在 -3.3 → `vehicle.bounding_box` 不可用，看脚本输出

---

## 7.  proximity 加权 progress + 探索超参

### 需求

"原地不动"局部最优修复 [#5](#5-2026528-原地不动局部最优) 后，leader 仍然走得慢。怀疑 progress 激励相对于"撞车 -300"的方差不够大，给 leader 一个"越靠近目标越值钱"的引力。

### 改动位置

**`environment/carla_multi_agent_env.py`**：progress 加 `proximity_factor`

```python
PROXIMITY_RANGE = 80.0
PROXIMITY_BOOST = 1.5
proximity_factor = 1.0 + PROXIMITY_BOOST * max(
    0.0, (PROXIMITY_RANGE - current_dist) / PROXIMITY_RANGE
)
progress_reward = progress * PROGRESS_WEIGHT * (0.3 + 0.7 * efficiency) * proximity_factor
```

- 距目标 ≥80m：factor=1.0（远端无加成）
- 距目标 40m：factor≈1.75
- 距目标 0m：factor=2.5

因为 `progress = prev_dist - current_dist` 是 delta，**站立不动整项=0，不可 farm**。

**`training/train_no_rllib_mappo_robust_v2.py`**：
- `--ent-coef` 默认 `0.01 → 0.03`（防 log_std 过早塌缩）
- `--init-log-std` 默认 `-0.5 → 0.0`（std=1.0，初期探索空间足）

### 验证

控制台 `progress` 分量明显上升（leader 在 ~50m 时 factor≈1.4，progress/step 加成约 40%）。

---

## 8.  TensorBoard 曲线重叠

### 现象

同一个 TB 图上看到多条扭曲的重叠曲线（"花线"），不是干净的一条线。

### 根因

同一个 `save_dir` 反复跑训练，`tb_logs/` 累积了多个 event 文件。TB 把它们合并成一个 run 显示，scalars 在相同 (name, step) 上得到多个点 → 曲线重叠。

### 改动位置

`training/train_no_rllib_mappo_robust_v2.py`：
```python
import datetime as _dt
run_tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
tb_log_dir = os.path.join(save_dir, 'tb_logs', run_tag)
```

每次启动用时间戳子目录隔离。

### 验证

启动时打印 `TensorBoard 日志: ./logs/.../tb_logs/<timestamp>_seedN`。新 run 在 TB 上是独立一条曲线，要对比就在 UI 里多选 run。

---

## 9.  续训功能 `--resume`

### 需求

恢复中断的训练 / 在已有 best.pt 基础上继续优化。原 `--pretrained-checkpoint` 只装权重，不带 optimizer / update 计数 / 累计统计 / 课程历史。

### 改动位置

`training/train_no_rllib_mappo_robust_v2.py`：
1. CLI 新增 `--resume <path>`（可传 `latest` 自动找 save_dir 下编号最大的 ckpt）
2. 训练循环前 resume 块：装载 actors / critic / actor_optims / critic_optim / update 计数 / total_episodes / total_successes / total_collisions / total_timeouts / best_success_rate / curriculum.scenario_success_history
3. 主循环改为 `for update in range(start_update, args.updates + 1)`
4. `_build_ckpt_dict()` 工厂函数：每个 ckpt（含 best.pt）都存全套 resume 所需字段

### 用法

```bash
# 模式 1：指定 ckpt 文件
python -m training.train_no_rllib_mappo_robust_v2 \
    --updates 200 --save-dir ./logs/foo \
    --resume ./logs/foo/mappo_robust_v2_update_60.pt

# 模式 2：自动找最新
python -m training.train_no_rllib_mappo_robust_v2 \
    --updates 200 --save-dir ./logs/foo --resume latest

# 模式 3：旧权重做种子开新实验（用 --pretrained-checkpoint）
python -m training.train_no_rllib_mappo_robust_v2 \
    --updates 100 --save-dir ./logs/new_run \
    --pretrained-checkpoint ./logs/foo/best.pt
```

### 注意

- `--updates N` 是**目标总数**，不是增量。要从 60 续到 200，传 `--updates 200`
- 两者同时给时 `--resume` 优先
- 如果 ckpt 是 reward 改动前保存的，恢复的策略在新 reward 下表现可能很差（V 估计不匹配）；这时建议用 `--pretrained-checkpoint`，让 optimizer 重新热身

---

## 10.  Issue 1：到点后右滑/姿态不正

### 现象

车辆到达 goal_zone 后停止姿态不像正常驾驶，总是有缓慢右转的趋势。

### 根因

step() 里已经在 `reached_target` 时应用了 `brake=1.0`，但**只是给刹车信号**。车辆原本的线速度和角速度还在物理引擎里，brake 要好几个 tick 才能压下去，期间转向轮的余量 + 转动惯量让车体侧滑/慢转。

### 改动位置

[carla_multi_agent_env.py:941-957](../environment/carla_multi_agent_env.py#L941-L957)

`reached_target` 分支多做两件事：
- 加 `hand_brake=True`（锁死车轮）
- 直接 `set_target_velocity(0,0,0)` + `set_target_angular_velocity(0,0,0)`，**绕过物理松弛立刻把刚体速度清零**

### 验证

CARLA 视图里 reached_target 那一刻车辆方向盘归正、立刻定住、不再右移。

---

## 11.  Issue 2：人行横道压线漏判

### 现象

车辆已经压在斑马线上、甚至开过了横道边缘，episode 都没有结束。

### 根因

原版仅依赖 `lane_invasion` 传感器的回调，且只在 `marking_type == 'NONE'` 时置位 `crosswalk_invasion_events`。问题：
1. lane_invasion 传感器**只在跨越标线那一刻回调一次**，进了横道后不再触发
2. CARLA 对人行横道边的标线类型分类不可靠，有时不是 `NONE`
3. 从未画线的一侧进入完全检测不到

### 改动位置

`environment/carla_multi_agent_env.py`：
1. `__init__` 加 `self._crosswalk_polys = None`（懒加载缓存）
2. [行 432-487](../environment/carla_multi_agent_env.py#L432-L487) 新增三个方法：
   - `_ensure_crosswalk_polygons`：调 `map.get_crosswalks()`，按"闭合点回到首点"切多边形
   - `_point_in_polygon`：射线法点-多边形包含判定
   - `_vehicle_on_crosswalk`：bbox 粗筛 + 精确判定
3. [step() 在 world.tick() 之后](../environment/carla_multi_agent_env.py#L953)，对每个未到点的活车主动判定，命中即置 `crosswalk_invasion_events[i] = True`

后续逻辑不变：`crosswalk_invasion_events` 在 `_compute_rewards` 给 -300 collision，在 `_check_dones` 触发"严重碰撞，全员 done"。

### 验证

CARLA 视图里车辆**一脚踏上斑马线就立刻 done**，不再开很远才结束。控制台 `collision%` 会上升（之前漏判被算作 timeout）。

### 性能成本

`map.get_crosswalks()` 只调 1 次（懒加载），后续每 step 是 O(N_polygons × N_vertices) 的点-多边形判定。Town02 一般 ≤30 个横道、每个 4 顶点，每个 step ~240 次浮点比较，相对 CARLA tick 开销可忽略。

---

## 12. `_check_dones` 状态机抖动

### 现象

CARLA 视图里有时看到 agent 已经停在 goal_zone 内（甚至已经触发 `reached_target=True`），但 episode 没有立即结束、train 脚本继续往下跑。

### 根因

[carla_multi_agent_env.py:1527](../environment/carla_multi_agent_env.py#L1527) 原判定结构：
```python
if in_goal_region:
    self.reached_target[i] = True
    dones[i] = True
elif current_step >= max_steps or is_collided:
    dones[i] = True
else:
    dones[i] = False   # ← 滑出 zone 那一帧走这里
```

车辆进 zone 设置 `reached_target=True, dones=True`，但下一帧若被物理惯性滑出 zone：
- `in_goal_region=False`、未碰撞、未超时
- `dones[i]` 又变回 False
- train 脚本 `all(dones.values())=False`，episode 不 break

### 改动位置

[carla_multi_agent_env.py:1527-1542](../environment/carla_multi_agent_env.py#L1527-L1542)

让 done 对 reached_target sticky：
```python
already_reached = self.reached_target.get(i, False)
if in_goal_region or already_reached:
    if not already_reached:
        self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.0)
    self.reached_target[i] = True
    dones[i] = True
elif self.current_step >= self.max_steps or is_collided:
    dones[i] = True
else:
    dones[i] = False
```

### 验证

CARLA 视图里 agent 一旦定在终点区，**接下来这台车的 dones 永远是 True**，不会出现"到了 → 抖动 → 又活"的循环。与 [#10](#10-2026531-issue-1到点后右滑姿态不正) 互为保险（#10 让物理上几乎不会滑出，#12 保证即使滑出状态机也不回退）。

### 关于"到点了但回合不停"的另一个常见原因

最大概率不是这个 bug，而是 **MAPPO 多 agent 设计预期**：episode 必须等**所有 agent** 完成才结束。
- agent1 到点 → `dones[1]=True`, agent0 还在跑 → `dones[0]=False`
- train 脚本 `all(dones.values())=False`，继续跑
- CARLA 里看到 agent1 停在终点（hand_brake 锁死），agent0 还在挪
- 等 agent0 也到/撞/超时才结束

这是合理设计但带来一个副作用：严格 success 要求**两个 agent 都到 goal_zone**，所以即使 leader 频繁到点，`rate_success_to_goal` 仍可能长期 0%。


