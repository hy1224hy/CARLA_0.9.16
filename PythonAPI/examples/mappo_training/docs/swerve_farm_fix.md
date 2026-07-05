# 抑制「左右摆动刷居中奖励」改动记录（R7）

> 日期：2026-06-16
> 涉及文件：`environment/carla_multi_agent_env.py`（`_compute_rewards_v2`）

## 1. 现象

训练中车辆出现**左右摆动 / 画龙（S 形）**，且看起来是在「骗取」居中/速度类奖励——
即摆动并不为了前进，而是为了刷分。

## 2. 根因：speed 用了「总速度模长」，把横向摆动当成「速度」刷分

`PROGRESS_WEIGHT` 已置 0（progress 停用，改用 Toromanoff 期望速度带），因此
**`speed_reward`（峰值 ~1.0/步）成了主前进信号**。但速度此前按
**总速度模长** `speed = sqrt(vx²+vy²+vz²)` 计算，被用于：

- `speed_reward`（期望速度带主信号）
- `motion_gate`（门控 `centering`+`yaw`）

于是**左右摆动产生的横向速度也被计入 `speed`**：车不接近终点，仅靠横向摆动
就能刷满 `speed_reward` 并让 `motion_gate=1`（居中 0.5 + 航向 0.15 也满额）。
摆动因此是**被奖励驱动**的——已有的居中死区、`lane_dev` 横偏惩罚、`steer_rate`
惩罚都没堵住这个「横向速度 = 奖励」的口子。

## 3. 修法：速度口径改为「朝终点的闭合速度」

把 `speed_reward` 和 `motion_gate` 的速度口径从总速度模长改为
**朝终点方向的闭合速度**（速度在「指向终点」单位向量上的投影，clamp≥0）：

```python
_gdx = target_loc.x - transform.location.x
_gdy = target_loc.y - transform.location.y
_gnorm = math.hypot(_gdx, _gdy)
closing_speed = max(0.0, (velocity.x*_gdx + velocity.y*_gdy)/_gnorm) if _gnorm>1e-6 else speed
```

| 运动 | 总速度模长(旧) | 朝终点闭合速度(新) |
|------|:---:|:---:|
| 前进开向终点 | 计入 | 计入 ✅ |
| 横向靠近终点 | 计入 | 计入 ✅ |
| 拐弯朝终点开 | 计入 | 计入 ✅ |
| 左右摆动 / 垂直于终点方向乱动 | 计入(被刷分) | ≈0 不计入 ✅ |

- `speed_reward`：仍是期望速度带（5–15 km/h 取峰、超 24 km/h 转负），只是带内变量
  从总速度换成闭合速度 → 摆动刷不到分，狂飙仍被罚。
- `motion_gate`：同样用闭合速度 → 摆动(闭合≈0)→gate≈0→不发放居中/航向；
  停车(闭合=0)→gate=0 仍堵住静态 farm（不重新诱发冻结）。
- `speed`（总速度模长）仅保留给 **TTC 车间相对速度**（碰撞风险），未改。

本质：**只有「让你更接近终点」的运动才算数；不让你更接近终点的摆动不算。**
这正是「有用运动 vs 无用摆动」的本质区分。

## 4. 为什么不会有副作用（评审中提出的两点担心）

1. **不抑制该拐的弯**：弯道上朝终点前进时，速度方向跟随车头（侧滑角小），
   闭合速度依然高；且「该不该拐」的引导信号是 `yaw`/`lane_dev`/居中，**完全未动**。
2. **不误伤横向接近终点**：横向靠近终点的闭合速度为正 → 照常奖励
   （这是最初用「沿车头投影」方案的盲区，故最终改用「朝终点投影」）。
3. **不切弯穿道够终点**：`lane_dev` + 跨线惩罚仍把车约束在车道内；
   且训练为直道，终点就在正前方车道里，闭合速度 = 车道内前进。
4. **不重新诱发冻结**：停车闭合=0 → gate=0，静态居中 farm 仍被堵。

## 5. 验证指标

CARLA 空闲时跑若干步看 TensorBoard / 控制台：
- 出现摆动时 `reward/speed`、`reward/centering` 应明显下降（不再被摆动刷起来）。
- 轨迹的横向抖动幅度 / `steer` 反复换向频率应下降。
- `episode/rate_success_to_goal` 不应因此下降（前进/接近终点不受影响）。

## 6. 可调旋钮

- 若仍残留小幅抖动：已有 `STEER_RATE_WEIGHT=0.05`、`lane_dev` 可适度上调。
- 期望速度带 `SPEED_LOW/HIGH`、`OVERSPEED_SLOPE` 控制巡航速度区间，与本次口径修改正交。
