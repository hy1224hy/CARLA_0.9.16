# MAPPO 基础策略 checkpoint 与恢复训练协议

> 适用入口：`training/train_no_rllib_mappo_robust_v2.py`。当前 checkpoint schema：v3；训练协议：`mappo_robust_v2_action_v2_p0_00g_v3`；动作协议：`tanh_gaussian_signed_longitudinal_2d_pre_tanh_z_v2`；奖励协议：`base_route_completion_reward_v3`；指标协议：`mappo_metrics_action_v2_p0_00g_v4`。R0/WTA 与 R1/reward-v2 checkpoint 只作历史诊断，不适用 R2 正式 resume/pretrained。

## 1. 三种操作必须区分

### Fresh run

从随机初始化开始一个新实验。`save_dir` 不得包含既有 checkpoint 产物；RunID、Git SHA、关键源码指纹、命令、配置、随机种子和运行时版本会写入 checkpoint。

### Pretrained weights

从已有且动作协议 v2 结构完全兼容的 Actor（以及结构兼容时的 Critic）权重开始一个新实验。它不会继承 optimizer、update、课程历史、累计统计或 RNG。来源路径和 SHA256 会记录在新 Run 中。R2 正式契约要求 fresh，禁止 pretrained；该入口只服务显式诊断/后续另行注册的实验。

### Resume

继续同一个 Run。必须完整恢复 Actor、agent-conditioned Critic、全部 optimizer、update、课程历史、统计和 RNG。缺字段、哈希损坏、协议/配置/Git SHA 不兼容时直接失败，不允许部分恢复。

## 2. Run 目录

```text
<save_dir>/
├── latest.pt
├── latest.pt.sha256
├── interrupted.pt
├── interrupted.pt.sha256
├── periodic/
│   ├── ckpt_update_000005.pt
│   ├── ckpt_update_000005.pt.sha256
│   └── ...
├── checkpoint_manifest.json
├── metrics_schema.json
├── metrics.jsonl
├── episodes.jsonl
└── tb/
```

- `latest.pt`：每个完整 update 后原子替换。
- `periodic/ckpt_update_*.pt`：按 `save_interval` 生成且禁止覆盖。
- `interrupted.pt`：Ctrl+C 或异常时复制最近完整 `latest.pt`，不会保存半个 rollout/PPO update。
- `checkpoint_manifest.json`：登记 RunID、latest、interrupted、periodic 的 update、路径和 SHA256。
- `metrics_schema.json`：指标协议、agent、奖励字段和聚合语义。
- `metrics.jsonl`：每个完整 update 的可审计 scalar payload。
- `episodes.jsonl`：每个 episode 的场景、终局和逐 agent 生命周期/transition/T-X/bootstrap 证据；resume 删除安全边界之后的未提交记录。
- `tb/`：与 JSONL 同源的 TensorBoard 连续目录；resume 使用 `purge_step` 并在需要时从 checkpoint 重放最后一个 update。

训练脚本不会创建 `best.pt`。未来的 `best_base_valid.pt` 只能由独立 C0/C1 验证流程从 immutable periodic checkpoints 中选择。

## 3. checkpoint 内容

- Actor、AgentConditionedCritic 及全部 optimizer state；
- 已完成 update 和累计 episode/success/collision/timeout；
- 课程参数、阶段列表、场景成功率历史和场景池文件指纹；
- Python、NumPy、Torch CPU、全部 CUDA RNG；
- 冻结的 LR decay horizon、初始/当前学习率；
- 完整 `action_spec`（协议、映射、策略/环境通道、维度和死区）；
- observation/action/state 维度、agent 数量、hidden dim、Critic 类型；
- schema、训练、观测、动作、reward 和 terminated/truncated 协议版本；
- 指标协议版本和最近完整 update 的 `last_metrics` payload；
- RunID、Git SHA/dirty paths、关键源码指纹、完整命令、配置路径/哈希、来源 checkpoint；
- Python、平台、Torch、NumPy、CUDA、CARLA client/server 版本。

## 4. Resume 兼容规则

以下配置会从 checkpoint 恢复。若命令行显式给出不同值，则 fail-fast：

- 地图、道路长度、timestep、max_steps、reward version；
- rollout episodes、PPO epochs、minibatch、最小 transition；
- Actor/Critic LR、gamma、GAE lambda、clip、entropy、gradient norm、target KL；
- hidden dim、log_std 初值、LR decay 开关；
- 场景采样和课程学习参数。
- checkpoint 保存间隔、控制台日志间隔和 TensorBoard histogram 间隔。

允许改变：本次停止 update、CARLA host/port、render 和输出操作参数。保存/日志/histogram 间隔属于同一 Run 的证据协议，不再允许 resume 时隐式回到默认值。LR decay horizon 在 fresh run 时冻结；resume 延长 `--updates` 不会改变既有 LR 曲线。

P0-00F-TB 起，resume 还要求 checkpoint 的 `last_metrics.update`、RunID 和指标协议与 checkpoint 边界一致。若 checkpoint 已保存但 TensorBoard/JSONL 尚未写入，恢复时会重放该 payload；日志领先 checkpoint、出现 update gap 或 schema 改变时直接失败。详细口径见 [MAPPO TensorBoard 指标协议](./MAPPO_TensorBoard指标协议.md)。

R1 起，resume/pretrained 还必须精确匹配 `action_spec` 与 `model_spec.action_dim=2`，Actor 使用严格 `state_dict` 加载。旧三维输出头、三维 `log_std` 和 optimizer state 不能部分加载，也不能使用 `strict=False` 绕过。旧 checkpoint 仍可由显式 legacy C0 v1 路径离线验证原证据。

R2 不改变 Actor/Critic 张量形状，因此 checkpoint schema 仍为 v3；但 reward version、训练契约 ID/SHA、metrics reward schema 和关键源码指纹都必须精确匹配。R1/action-v2/reward-v2 checkpoint 可以通过显式 C0 v2 历史路径读取，却不能 resume 或 pretrained 到 `P0_BASE_MAPPO_TOWN02_REWARD_V3`。正式 R2 只允许随机初始化 fresh，以隔离奖励改变的因果效应。

Git SHA 和关键源码指纹默认都必须一致。`--allow-resume-code-drift` 仅用于明确知道风险的诊断续训；每次漂移的前后 Git SHA 与源码指纹都会在元数据中永久累积，不能作为正式可复现实验的默认做法。

## 5. 标准命令

### 新训练

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --run-id P0_base_s42_20260727_1200 `
  --save-dir logs/P0_base_s42_20260727_1200 `
  --updates 100 `
  --lr-decay-horizon 100 `
  --seed 42
```

### 同一 Run 恢复

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --resume latest `
  --save-dir logs/P0_base_s42_20260727_1200 `
  --updates 100
```

未在恢复命令中重写的训练超参数会从 checkpoint 恢复。`--updates` 必须大于 checkpoint 中已完成的 update。

### 从权重开启新 Run

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --pretrained-checkpoint logs/source/periodic/ckpt_update_000050.pt `
  --run-id P0_finetune_s43_20260727_1400 `
  --save-dir logs/P0_finetune_s43_20260727_1400 `
  --updates 100 `
  --seed 43
```

## 6. 中断与故障处理

- 正常 Ctrl+C：当前未完成 update 被丢弃，`interrupted.pt` 指向最近完整 update。
- 进程强杀/断电：使用已经原子提交的 `latest.pt`。
- `latest.pt` 哈希不一致：拒绝加载，不静默回退；检查 immutable periodic checkpoint 后人工决定恢复点。
- 找不到 `latest.pt`：拒绝从零启动，避免覆盖实验身份。
- 需要从更早 periodic 分叉：使用 `--pretrained-checkpoint` 和新 RunID，不伪装成同一 Run 的 resume。

## 7. 可复现性的边界

软件层面可以恢复 Python/NumPy/Torch RNG 和 optimizer 的下一步连续性。CARLA/Traffic Manager 跨进程后无法保存服务器内部全部物理状态，因此不承诺轨迹逐位一致。正式口径是：只在完整 update 边界恢复，随后通过 P0-00G 验证 update、LR、课程、样本数和日志连续；不能把“同一随机种子”误写成 CARLA 轨迹逐位复现。
