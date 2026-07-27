# MAPPO 基础策略 checkpoint 与恢复训练协议

> 适用入口：`training/train_no_rllib_mappo_robust_v2.py`。协议版本：`mappo_robust_v2_p0_00f_v1`。旧 RLlib、robust v1 和旧独立 MAPPO checkpoint 不适用本协议。

## 1. 三种操作必须区分

### Fresh run

从随机初始化开始一个新实验。`save_dir` 不得包含既有 checkpoint 产物；RunID、Git SHA、关键源码指纹、命令、配置、随机种子和运行时版本会写入 checkpoint。

### Pretrained weights

从已有 Actor（以及结构兼容时的 Critic）权重开始一个新实验。它不会继承 optimizer、update、课程历史、累计统计或 RNG。来源路径和 SHA256 会记录在新 Run 中。

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
└── tb/
```

- `latest.pt`：每个完整 update 后原子替换。
- `periodic/ckpt_update_*.pt`：按 `save_interval` 生成且禁止覆盖。
- `interrupted.pt`：Ctrl+C 或异常时复制最近完整 `latest.pt`，不会保存半个 rollout/PPO update。
- `checkpoint_manifest.json`：登记 RunID、latest、interrupted、periodic 的 update、路径和 SHA256。
- `tb/`：同一 Run 的 TensorBoard 连续目录；resume 使用 `purge_step` 清理重复步。

训练脚本不会创建 `best.pt`。未来的 `best_base_valid.pt` 只能由独立 C0/C1 验证流程从 immutable periodic checkpoints 中选择。

## 3. checkpoint 内容

- Actor、AgentConditionedCritic 及全部 optimizer state；
- 已完成 update 和累计 episode/success/collision/timeout；
- 课程参数、阶段列表、场景成功率历史和场景池文件指纹；
- Python、NumPy、Torch CPU、全部 CUDA RNG；
- 冻结的 LR decay horizon、初始/当前学习率；
- observation/action/state 维度、agent 数量、hidden dim、Critic 类型；
- schema、训练、观测、动作、reward 和 terminated/truncated 协议版本；
- RunID、Git SHA/dirty paths、关键源码指纹、完整命令、配置路径/哈希、来源 checkpoint；
- Python、平台、Torch、NumPy、CUDA、CARLA client/server 版本。

## 4. Resume 兼容规则

以下配置会从 checkpoint 恢复。若命令行显式给出不同值，则 fail-fast：

- 地图、道路长度、timestep、max_steps、reward version；
- rollout episodes、PPO epochs、minibatch、最小 transition；
- Actor/Critic LR、gamma、GAE lambda、clip、entropy、gradient norm、target KL；
- hidden dim、log_std 初值、LR decay 开关；
- 场景采样和课程学习参数。

允许改变：本次停止 update、保存/日志间隔、CARLA host/port、render 和输出操作参数。LR decay horizon 在 fresh run 时冻结；resume 延长 `--updates` 不会改变既有 LR 曲线。

Git SHA 和关键源码指纹默认都必须一致。`--allow-resume-code-drift` 仅用于明确知道风险的诊断续训；每次漂移的前后 Git SHA 与源码指纹都会在元数据中永久累积，不能作为正式可复现实验的默认做法。

## 5. 标准命令

### 新训练

```powershell
......envScriptspython.exe -m training.train_no_rllib_mappo_robust_v2 `
  --run-id P0_base_s42_20260727_1200 `
  --save-dir logs/P0_base_s42_20260727_1200 `
  --updates 100 `
  --lr-decay-horizon 100 `
  --seed 42
```

### 同一 Run 恢复

```powershell
......envScriptspython.exe -m training.train_no_rllib_mappo_robust_v2 `
  --resume latest `
  --save-dir logs/P0_base_s42_20260727_1200 `
  --updates 100
```

未在恢复命令中重写的训练超参数会从 checkpoint 恢复。`--updates` 必须大于 checkpoint 中已完成的 update。

### 从权重开启新 Run

```powershell
......envScriptspython.exe -m training.train_no_rllib_mappo_robust_v2 `
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
