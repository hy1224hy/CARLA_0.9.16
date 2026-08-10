# MAPPO TensorBoard 指标与奖励审计协议

> 适用入口：`training/train_no_rllib_mappo_robust_v2.py`。指标协议版本：`mappo_metrics_action_v2_p0_00g_v4`；当前奖励协议：`base_route_completion_reward_v3`；逐 episode 协议：`mappo_episode_record_v1`。本协议不自动覆盖旧 RLlib、robust v1、旧对抗 PPO 或尚未迁移的 `train_2red_1blue_ppo.py`。历史 metrics/reward schema 仅保留离线验证兼容。

## 1. 目的

本阶段不是简单增加曲线，而是保证“环境产生什么、训练器使用什么、TensorBoard 显示什么”能够逐 update 对账：

1. 每个奖励分量使用固定 schema；缺字段、未知字段、NaN/Inf 不再被静默当成 0。
2. 奖励只归入真正拥有该 transition 的 agent，失活 agent 不补零、不增加分母。
3. 区分连续奖励、稀疏事件和有意禁用项，避免把合理的 0 误判为日志故障。
4. 同时记录 rollout、PPO、Critic、动作、模型变化、场景和耗时，定位“没有学习”发生在哪一层。
5. TensorBoard 与可审计 JSONL 使用同一份冻结 payload，并与安全 checkpoint 在 update 边界保持一致。
6. 每个 episode 独立保存 agent 生命周期、T/X、bootstrap 和 transition 对账证据，不能只保留 update 均值。

## 2. 模块边界

```text
CarlaMultiAgentEnv
  -> reward + reward_details + reward_diagnostics
  -> training/metrics_protocol.py（schema、有限值、精确字段、奖励守恒）
  -> utils/training_metrics_recorder.py（校验、聚合、持久化）
  -> episodes.jsonl + tb/ + metrics.jsonl + checkpoint.last_metrics
  -> utils/validate_tensorboard_run.py（离线验收）
```

环境只报告原始事实和加权后的奖励分量；训练入口只提供 rollout/PPO 上下文；独立 recorder 负责校验、聚合和持久化。以后其他当前训练入口接入时应复用这三个模块，并为自己的奖励版本显式增加 schema，不能复制临时 TensorBoard 代码。

## 3. Run 产物与事务顺序

```text
<save_dir>/
├── tb/                    # TensorBoard event
├── metrics.jsonl          # 每行一个完整 update 的同源 scalar payload
├── episodes.jsonl         # 每行一个 episode 的生命周期与 transition 原始记录
├── metrics_schema.json    # 版本、agent、奖励字段、聚合口径
├── latest.pt              # 含 last_metrics
└── ...
```

每个完整 episode 先以 append+flush+fsync 写入 `episodes.jsonl`；随后一个 update 的提交顺序是：recorder 冻结 payload；原子保存带 `last_metrics` 的安全 checkpoint；将同一 payload 写入 TensorBoard 和 `metrics.jsonl`。若 update 尚未形成安全 checkpoint 就中断，resume 会原子删除所有领先安全边界的 episode 记录；若 checkpoint 已保存但 update 日志未提交，则以 checkpoint 为权威重放最后一个 metrics payload。两类 JSONL 都必须按 update/episode 严格有序，且逐 episode 合计必须与 update 聚合相等。

## 4. 奖励字段语义

| 类别 | 当前 v3 字段 | 正确的 0 如何解释 |
|---|---|---|
| 正任务信号 | `progress` | gamma 一致的路线完成度势能；停车为 0，后退可为负，往返闭环不能刷正回报 |
| 连续有界成本 | `overspeed`、`longitudinal_rate`、`steer_rate`、`safety_lidar`、`ttc`、`lane_deviation` | 未进入风险区或实际控制未变化时为 0；每项都有冻结的单步幅值上界 |
| 稀疏事件 | `lane_invasion`、`collision`、`success` | 仅在事件发生时非零；collision 与 success 同 tick 时 collision 优先，不能同时发放 |

reward v3 显式移除 `alive`、`approach`、`centering`、`yaw`、`step` 和 `time_bonus`，速度本身不产生正奖励；历史 reward v2 记录仍按其原 schema 只读验证。`progress_m`、路线完成度、实际 signed-longitudinal command/rate、速度、横向偏移、LiDAR/TTC 候选与事件均作为 raw diagnostics。LiDAR 使用最近 ray、lane event 使用单次最严重 marking，因此 ray/marking 数量不会放大奖励预算。完整数学见 [R2 奖励预算校准](./R2_奖励预算校准_论文正文与实验设计.md)。

每个 active transition 强制满足：

```text
abs(env_reward - sum(reward_details.values())) <= 1e-6
```

episode 级再次校验 `episode_return` 与逐步分量和，容差上限为 `active_steps × 1e-6`。字段缺失、多余、非有限或守恒失败都会立即终止训练，不生成看似正常的 0 曲线。

## 5. 聚合口径

| 标签口径 | 含义 | 用途 |
|---|---|---|
| `reward/agent_<id>/*/mean_per_active_step` | 仅该 agent 有效 transition 的均值 | 判断个体策略和失活后的样本污染 |
| `reward/agent_<id>/*/sum_per_episode_mean` | 每个 episode 先求和，再等权平均 | 判断该分量对 episode return 的实际贡献 |
| `reward/pooled/*/mean_per_active_step` | 所有 agent transition 加权均值 | 观察整体采样分布；不能替代个体曲线 |
| `reward/team/*/mean_agent_episode_sum` | 各 agent episode-sum 均值再等权平均 | 观察合作团队回报口径 |

每个 per-agent 分量还记录 `nonzero_rate`、`nonzero_mean`、`minimum` 和 `maximum`。稀疏分量额外写入 `events/.../nonzero_count_per_update`，不能仅看普通均值判断是否触发。

## 6. 训练健康指标

- `rollout/*`：每 agent active transition、进入 PPO 的 transition 数和跳过更新标志。
- `episode/*`、`scenario/*`：回报、长度、成功/碰撞/超时、终局原因和分场景统计。
- `diagnostic/*`：速度、位置误差、LiDAR、TTC、碰撞/越线/成功原始信号。
- `action/*`：二维策略 `signed_longitudinal/steer` 均值、纵向死区率/饱和率，以及映射后三维油门、转向、刹车分布和控制模式率。recorder 会逐步验证二维策略动作与三维环境动作一致；旧 throttle/brake combined boundary 同时包含互斥产生的零值，不能直接作为饱和门槛。完整语义见 [车辆动作空间与边界指标协议](./MAPPO_车辆动作空间与边界指标协议.md)。
- `advantage/*`：标准化前后统计和样本数。
- `policy/*`：loss、entropy/bonus、ratio、KL、clip fraction、梯度裁剪、stop epoch、log_std。
- `critic/*`：value loss、explained variance、预测/return 分布、绝对误差、样本数和梯度裁剪。
- `model/*`：Actor/Critic 参数范数、更新范数和相对更新范数。
- `optimizer/*`、`timing/*`、`curriculum/*`：实际学习率、耗时和课程状态。
- `reward_audit/*`：逐步与 episode 守恒残差、缺字段和非有限值计数。

`policy/agent_<id>/preupdate_max_abs_ratio_error` 是 optimizer step 前按完整 agent batch 重算的 `max_abs(ratio-1)`，生产更新以 `<1e-6` 为硬门槛；它替代含糊的“首次 PPO epoch ratio≈1”。`ratio_min` 与 `ratio_max` 分别取本 update 所有 PPO epoch 的真实极小值/极大值；其他 epoch 级健康量按均值汇总。参数 histogram 默认每 10 updates 写一次，可用 `--tb-histogram-interval 0` 关闭。

## 7. 使用与离线验收

在 `PythonAPI/examples/mappo_training/` 下运行：

```powershell
tensorboard --logdir logs/<RunID>/tb

..\..\..\venv\Scripts\python.exe -m utils.validate_tensorboard_run `
  --run-dir logs/<RunID> `
  --expected-updates 1 2 3 4 `
  --expected-episodes-per-update 4 `
  --require-ppo-updates `
  --max-preupdate-ratio-error 1e-6
```

P0-00G 验证器必须返回 `"status": "PASS"`。它检查协议、update 顺序、必需标签、全部 scalar 有限、逐步奖励守恒、逐 episode/agent 生命周期与 transition 汇总，以及 TensorBoard 与 JSONL 在每个 update 的数值一致性。省略严格参数时只执行通用结构验证，不能替代 P0-00G 验收。

无 CARLA 测试只能证明指标协议、聚合逻辑与 reward-v3 数学正确。LiDAR、TTC、碰撞 sensor、真实 episode 终局和真实 PPO 数值是否符合预期，仍必须在 R5 CARLA smoke 中通过 validator、TensorBoard 和场景原始记录共同确认。

v4 随 R1 将当前基础 Actor 改为二维，但 CARLA 环境 action 仍为三维；指标同时审计这两个空间。R2 保持 metrics protocol v4 不变，由 `reward_version=v3` 和嵌入的 `reward_v3_spec` 精确区分奖励 schema/参数。由于 checkpoint 同时绑定动作、奖励、指标和训练契约，R0/WTA 与 R1/reward-v2 checkpoint 不得作为 R2 正式 Run 的 strict resume/pretrained 来源，但其历史 JSONL/TensorBoard schema 仍可由 validator 读取。
