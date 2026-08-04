# MAPPO 车辆动作空间与边界指标协议

> 适用入口：`training/train_no_rllib_mappo_robust_v2.py` 与基础策略 C0 评估器。
>
> 当前决定：保留三维 Actor action 和 `winner_take_all_v1` 环境映射；指标协议升级为 `mappo_metrics_p0_00g_v3`，C0 逐回合证据升级为 `p0_base_c0_episode_v2`。

## 1. 结论

油门与刹车互斥是合理的车辆控制约束。CARLA 的 `VehicleControl` 分别接受 `[0,1]` 油门、`[-1,1]` 转向和 `[0,1]` 刹车；CARLA 自带 navigation controller 也是先计算一个有符号纵向控制量，非负时施加油门、负值时施加刹车，不同时下发二者。参考：[CARLA Python API](https://carla.readthedocs.io/en/latest/python_api/#carla.VehicleControl) 与仓库内 [`controller.py`](../../../carla/agents/navigation/controller.py)。

自动驾驶强化学习中常见两种参数化：

1. 策略直接输出 `throttle/steer/brake`，再施加范围与互斥约束；
2. 策略输出一个有符号 longitudinal acceleration 和 steer，再将正负纵向量拆成油门/刹车。后者维度更低、语义更直接，也见于直接学习 acceleration/steering 的研究设计（[Maramotti et al., 2022](https://arxiv.org/abs/2207.02162)）。

当前基础 MAPPO 属于第一种。它不是 PPO 概率错误：rollout 保存和 PPO ratio 使用的是映射前同一个三维 latent action，环境映射是固定且不依赖参数的确定性函数。因此策略梯度仍对应“采样 latent action 后执行其环境映射”的策略。

## 2. 当前映射及保留原因

令 Actor 的 tanh action 为 `(a_t, a_s, a_b)`：

```text
throttle_raw = (a_t + 1) / 2
steer        = a_s
brake_raw    = (a_b + 1) / 2

if throttle_raw >= brake_raw:
    throttle = throttle_raw, brake = 0
else:
    throttle = 0, brake = brake_raw
```

它保证输出范围正确且 `throttle × brake = 0`。目前不把 Actor 改成二维 signed-longitudinal，原因是：

- 当前 MAPPO、checkpoint model spec、C0 契约均已冻结为 action dimension 3；
- 映射不破坏 PPO ratio、GAE 或 centralized critic 的正确性；
- 在正式 pilot 前临时改变动作维度会同时改变探索分布、网络参数量和 checkpoint 协议，属于新的实验变量，不是边界日志 bug 的最小修复。

该映射也有已知限制：两个纵向 latent 维度冗余；在两者接近时环境动作会切换；两者同为零时映射为半油门而不是滑行。P0-BASE pilot 必须通过 jerk、模式占比、真实 C0 轨迹和安全结局监控这些风险。若出现持续油刹切换或控制不平滑，应另建 action protocol v2，使用二维 signed-longitudinal，从 fresh Run 重新训练；不得在同一个正式 Run 中途更换映射。

## 3. 旧边界指标为什么会误报

旧指标把 `pedal <= 0.01` 和 `pedal >= 0.99` 合并为一个 boundary。互斥映射每步至少令一个踏板为零，因此某个 throttle/brake combined boundary rate 长期接近 1 完全可能是正常驾驶模式，不能据此判断策略饱和。

v3 保留以下旧标签用于历史追溯，但禁止将 throttle/brake combined boundary 直接用于 GO/HOLD：

- `throttle_boundary_rate = throttle_zero_rate + throttle_saturation_rate`
- `brake_boundary_rate = brake_zero_rate + brake_saturation_rate`
- `steer_boundary_rate = steer_saturation_rate`

## 4. v3 可用于判定的动作指标

每个 agent 新增：

| TensorBoard/JSONL 标签后缀 | 含义 |
|---|---|
| `throttle_zero_rate`、`brake_zero_rate` | 踏板处于零值的比例；主要解释互斥模式，不是饱和 |
| `throttle_saturation_rate`、`brake_saturation_rate` | 踏板达到上边界 `>=0.99` 的比例 |
| `steer_saturation_rate` | `abs(steer)>=0.99` 的比例 |
| `throttle_mode_rate` | 仅油门有效的比例 |
| `brake_mode_rate` | 仅刹车有效的比例 |
| `coast_mode_rate` | 油门和刹车都不生效的比例 |
| `longitudinal_conflict_rate` | 油门与刹车同时生效的比例；当前映射下必须为 0 |

以下守恒关系由 C0 validator 强制检查：

```text
throttle_mode_rate + brake_mode_rate
+ coast_mode_rate + longitudinal_conflict_rate = 1

longitudinal_conflict_rate = 0
```

P0-BASE-04 应使用真正的 `*_saturation_rate` 制定数值门槛。零值率只解释控制模式；最终是否存在静止刷分或激进控制，还必须联合 success/timeout、speed、closing speed、jerk、lane invasion、collision 和视频判断。

## 5. 兼容性

- Actor、Critic、环境三维 action 和 `winner_take_all_v1` 行为均未改变。
- 指标协议由 v2 升为 v3；旧指标 checkpoint 不允许作为新正式 Run 的 strict resume 来源。
- C0 核心 case、seed、确定性动作和 summary 口径未改变；逐回合 action 证据增加拆分字段并升为 `p0_base_c0_episode_v2`。
- 仓库中的历史 checkpoint 仍只能用于历史诊断，不能绕过当前正式契约。
