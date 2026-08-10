# MAPPO 车辆动作空间与边界指标协议

> 适用入口：`training/train_no_rllib_mappo_robust_v2.py` 与基础策略 C0 评估器。
> 当前决定：R1 使用二维 `[signed_longitudinal, steer]` 策略动作与 `signed_longitudinal_deadzone_v2` 环境映射；指标协议为 `mappo_metrics_action_v2_p0_00g_v4`，C0 逐回合证据为 `p0_base_c0_episode_v3`。三维 `winner_take_all_v1` 仅作历史读取与诊断。

完整理论推导、论文正文、文献与测试计划见 [R1 二维有符号纵向动作协议](./R1_二维纵向动作协议_论文正文与实验设计.md)。本文只保留工程协议真值和指标解释。

## 1. 当前动作真值

Actor 输出

```text
policy_action = [signed_longitudinal, steer] ∈ [-1,1]^2
deadzone d = 0.05
```

纵向重缩放为

```text
if longitudinal > d:
    command = (longitudinal - d) / (1 - d)
elif longitudinal < -d:
    command = (longitudinal + d) / (1 - d)
else:
    command = 0

throttle = max(command, 0)
brake    = max(-command, 0)
steer    = policy_steer
```

因此输出满足：

- `throttle ∈ [0,1]`、`brake ∈ [0,1]`、`steer ∈ [-1,1]`；
- `throttle * brake = 0`；
- `abs(signed_longitudinal) <= 0.05` 时为 coast；
- 在 `±0.05` 处连续，且 `±1` 保留满幅控制。

实现的唯一真值位于 `training/action_protocol.py`。训练、指标 recorder 和 C0 不允许各自复制另一份映射。

## 2. PPO 概率口径

Actor 在二维 pre-tanh 空间采样 `z`，令 `action_tanh=tanh(z)`；rollout 保存同一个二维 `z` 和经变量变换修正的标量 log-prob。PPO 更新继续用保存的 `z` 重算新 log-prob，策略参数未变化时 ratio 必须为 1。

二维动作映射成三维 CARLA command 是采样后的确定性环境适配，不在三维物理动作上重新计算概率。C0 使用 `tanh(actor_mean)`，再调用完全相同的当前映射。

## 3. WTA v1 历史协议

旧 Actor 输出 `(a_t,a_s,a_b)`，映射为

```text
throttle_raw = (a_t + 1) / 2
brake_raw    = (a_b + 1) / 2

if throttle_raw >= brake_raw:
    throttle = throttle_raw, brake = 0
else:
    throttle = 0, brake = brake_raw
```

该协议能保证油刹互斥，且旧 PPO 的 latent 概率口径在数学上并非无效；但它有两个纵向隐变量竞争一个物理自由度、切换边界不连续、零 latent 映射半油门等限制。R0 的制动塌缩证据触发了 R1 升级。

`legacy_wta_action_tanh_to_env()` 只服务历史 C0/证据读取。旧 Actor、optimizer 和 checkpoint 不得用于 R1 strict resume 或正式 pretrained。

## 4. 当前动作指标

每个 active agent 必须同时记录二维策略动作和三维环境动作；recorder 逐步重算映射，任何不一致立即失败。

| TensorBoard/JSONL 标签后缀 | 含义 |
|---|---|
| `policy_signed_longitudinal/mean` | 二维策略纵向输出均值 |
| `policy_steer/mean` | 二维策略转向输出均值 |
| `signed_longitudinal_deadzone_rate` | 策略纵向输出落入 `[-0.05,0.05]` 的比例 |
| `signed_longitudinal_saturation_rate` | `abs(signed_longitudinal)>=0.99` 的比例 |
| `throttle_zero_rate`、`brake_zero_rate` | CARLA 踏板处于零值的比例；主要解释互斥模式，不表示上界饱和 |
| `throttle_saturation_rate`、`brake_saturation_rate` | CARLA 踏板达到上边界 `>=0.99` 的比例 |
| `steer_saturation_rate` | `abs(steer)>=0.99` 的比例 |
| `throttle_mode_rate` | 仅油门有效的比例 |
| `brake_mode_rate` | 仅制动有效的比例 |
| `coast_mode_rate` | 油门与制动均不生效的比例 |
| `longitudinal_conflict_rate` | 油门与制动同时生效的比例；必须为 0 |

以下守恒关系由 metrics/C0 validator 强制检查：

```text
throttle_mode_rate + brake_mode_rate
+ coast_mode_rate + longitudinal_conflict_rate = 1

longitudinal_conflict_rate = 0
```

历史 combined boundary 标签仍保留，但只满足：

```text
throttle_boundary_rate = throttle_zero_rate + throttle_saturation_rate
brake_boundary_rate    = brake_zero_rate + brake_saturation_rate
steer_boundary_rate    = steer_saturation_rate
```

combined boundary 包含互斥造成的正常零踏板，禁止将其直接解释为策略饱和。动作健康必须联合策略死区/饱和、物理模式、speed、progress、jerk、success/timeout、lane invasion 和 collision 判断。

## 5. 协议版本与兼容性

| 层 | 历史 v1 | 当前 v2 |
|---|---|---|
| 策略动作 | 3 维 throttle/steer/brake latent | 2 维 signed-longitudinal/steer |
| 映射 | `winner_take_all_v1` | `signed_longitudinal_deadzone_v2` |
| checkpoint schema | 2 | 3，含完整 `action_spec` |
| training protocol | `mappo_robust_v2_p0_00g_v2` | `mappo_robust_v2_action_v2_p0_00g_v3` |
| metrics | `mappo_metrics_p0_00g_v3` | `mappo_metrics_action_v2_p0_00g_v4` |
| C0 | `p0_base_c0_eval_v1` / episode v2 | `p0_base_c0_eval_v2` / episode v3 |

- v1 checkpoint 和证据保持只读，不覆盖原 judge 或实验记录。
- 当前 resume/pretrained/C0 必须同时匹配 schema、protocol versions、`action_spec`、`model_spec.action_dim=2` 和严格 Actor state。
- R1 不修改奖励版本、Critic 类型/维度、观测协议、场景和生命周期协议。
- 当前动作适配器逐车独立，对更多车辆可复用；但本轮训练/C0 契约仍固定两车，多车 Critic 与评估协议必须另行升级。
