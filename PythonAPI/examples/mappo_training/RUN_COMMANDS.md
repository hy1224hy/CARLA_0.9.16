
# 运行指令参考

本文档保留项目脚本的历史运行方式，但新实验必须遵守下面的入口冻结规则。阶段顺序和是否允许运行以 [毕业课题研究推进 TODO](docs/毕业课题_研究推进TODO.md) 为准。

> **入口冻结（2026-07-20）**
>
> - 基础策略唯一入口：`training.train_no_rllib_mappo_robust_v2`。
> - Phase 1 单蓝车唯一入口：`training.train_2red_1blue_ppo`。
> - RLlib、shared-policy、minimal/robust v1、旧 adversarial PPO 和旧 multi 入口全部是 **DEPRECATED**，不得用于新训练、resume、正式 checkpoint 或论文结果。
> - 2026-08-12 起当前背景为 Town05：使用 `P0_BASE_MAPPO_TOWN05_V1`、Town05 C0 和 Town05 GO/HOLD。提交 `a11cf70` 已通过 16/16 和真实 1-update 门禁；当前只允许以新正式 RunID、随机初始化运行 seed 42 fresh pilot 到 update 5。Town02/诊断 smoke checkpoint 禁止 resume。u5 必须停止并执行 strict validator、两次 C0 gate 和唯一冻结 judge；对抗策略长训练仍需等待 P0-01 至 P0-06 和 P0-05M。

Town05 正式坐标、16 场景问题定义和迁移验证见 [Town05 正式实验迁移与 16 场景设计](docs/Town05_正式实验迁移与16场景设计.md)。

---

## 目录

1. [前置准备：启动 CARLA 服务器](#前置准备启动-carla-服务器)
2. [训练脚本](#训练脚本)
3. [评估脚本](#评估脚本)
4. [数据采集工具](#数据采集工具)
5. [常用参数速查表](#常用参数速查表)

---

## 前置准备：启动 CARLA 服务器

所有脚本都需要 CARLA 服务器正在运行：

**Windows:**
```cmd
CarlaUE4.exe
```

**Linux:**
```bash
./CarlaUE4.sh
```

可选参数：
```bash
# 修改端口（默认 2000）
./CarlaUE4.sh -carla-port=2000

# 降低画质以提升性能
./CarlaUE4.sh -quality-level=Low
```

---

## 策略无关的固定场景 success smoke

本测试不需要训练好的基础 checkpoint。先用真实环境中的顺序传送隔离验证 success/reward/lifecycle，再用近终点固定油门验证短距离运动链路；禁止把未收敛策略到不了终点误判为 success 实现错误。

```powershell
# A. 必须先通过：纯终点逻辑；预计两车依次 success
..\..\..\venv\Scripts\python.exe -B -m evaluation.smoke_test_success_logic `
  --mode teleport `
  --scenario data/training_scenarios_town05_v1/scenario_01_north_parallel_stationary_a0_left.json `
  --seed 42 `
  --follow-spectator

# B. A 通过后运行：近终点、低油门直行
..\..\..\venv\Scripts\python.exe -B -m evaluation.smoke_test_success_logic `
  --mode scripted `
  --scenario data/training_scenarios_town05_v1/scenario_01_north_parallel_stationary_a0_left.json `
  --lead-distance 5 `
  --following-gap 10 `
  --throttle 0.30 `
  --max-steps 250 `
  --seed 42 `
  --follow-spectator
```

两次 `result.json` 都必须为 `status=PASS`，并且路径中不能混入 collision/crosswalk 事件。产物与错误归因见 [P0-00G-SUCCESS 规范与记录](docs/P0-00G-SUCCESS_真实success逻辑测试规范与记录.md)。

---

## 训练脚本

### 当前有效入口

所有命令都在 `PythonAPI/examples/mappo_training/` 目录执行。

```powershell
# P0-00G：基础 MAPPO 三更新 smoke
# 只允许在 P0-00B 至 P0-00F-TB 全部通过后执行
# 完整环境检查、受控中断和验收见 docs/P0-00G_真实CARLA_smoke运行规范.md
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --run-id P0_00G_robust_v2_smoke_s42_<timestamp> `
  --updates 3 `
  --lr-decay-horizon 3 `
  --rollout-episodes 4 `
  --min-transitions-per-update 64 `
  --max-steps 300 `
  --save-interval 1 `
  --tb-histogram-interval 1 `
  --seed 42 `
  --save-dir logs/P0_00G_robust_v2_smoke_s42_<timestamp>

# 随后从 update 3 严格恢复第 4 个 update；LR horizon 仍保持 3
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --resume latest `
  --updates 4 `
  --save-dir logs/P0_00G_robust_v2_smoke_s42_<same-timestamp>

# Phase 1：单蓝车三更新 smoke
# 只允许在 P0-06 通过后执行
..\..\..\venv\Scripts\python.exe -m training.train_2red_1blue_ppo `
  --pretrained-ckpt <best_base_valid.pt> `
  --scenario scenario_09_north_stagger_closing_a0_rear.json `
  --scenario-dir data/training_scenarios_town05_v1 `
  --updates 3 `
  --rollout-episodes 4 `
  --no-viz `
  --seed 42 `
  --save-dir logs/P1_single_blue_basic_s42_<timestamp>
```

正式训练参数必须固化到版本控制配置中并登记 RunID；不要直接把 smoke 命令扩成上百 updates。

Town05 迁移与 gamma 对齐修复提交后，先运行诊断 smoke。它使用显式诊断参数，不冒充正式合同 Run；完成后必须核对 checkpoint runtime 中环境实际 gamma 为 0.997：

```powershell
..\..\..\venv\Scripts\python.exe -B -m utils.run_town05_pilot `
  --skip-live-smoke `
  --rollout-episodes 4 `
  --max-steps 120 `
  --output-root diagnostics/town05_gamma_alignment_smoke
```

该 smoke、strict metrics、checkpoint gamma 和 CARLA 清理通过后，下一次获准执行的正式命令才是 update 5。`<timestamp>` 必须替换为新的时间戳，RunID 与保存目录必须一致；不得加入 `--resume` 或加载任何 Town02 checkpoint：

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --host 127.0.0.1 `
  --port 2000 `
  --experiment-contract configs/base_mappo/p0_base_town05_v1.json `
  --run-id P0_BASE_MAPPO_Town05_s42_<timestamp> `
  --updates 5 `
  --cpu `
  --follow-spectator `
  --save-dir logs/P0_BASE_MAPPO_Town05_s42_<timestamp>
```

`--follow-spectator` 只在每次车辆初始化后移动 CARLA 服务器自带俯视镜头，不启动旧 pygame 可视化。无人观察的长训练可去掉该开关；它不改变观测、动作、奖励或采样时序。

该段完成后先运行 strict validator，再按 Town05 C0 v1 执行主/重复 gate，最后使用 Town05 GO/HOLD v1 唯一裁决；不要在看到训练 rollout 后临时修改奖励、参数、门槛或继续到 u10。

```powershell
..\..\..\venv\Scripts\python.exe -m utils.validate_tensorboard_run `
  --run-dir logs/P0_BASE_MAPPO_Town05_s42_<same-timestamp> `
  --expected-updates 1 2 3 4 5 `
  --expected-episodes-per-update 16 `
  --require-ppo-updates `
  --max-preupdate-ratio-error 1e-6
```

### 1. [DEPRECATED] train_no_rllib_mappo.py - 早期独立 MAPPO

**功能：** 不依赖 RLlib 的 MAPPO 训练，使用集中式批评家（CTDE）

**运行方式：**
```bash
# 从根目录运行（推荐）
python -m training.train_no_rllib_mappo

# 或进入子目录运行
cd training
python train_no_rllib_mappo.py
```

**常用参数：**
```bash
# 基础用法
python -m training.train_no_rllib_mappo \
    --host 127.0.0.1 \
    --port 2000 \
    --town Town02

# 开启可视化（需要 pygame）
python -m training.train_no_rllib_mappo --render

# 调整训练参数
python -m training.train_no_rllib_mappo \
    --updates 2000 \
    --rollout-steps 2048 \
    --lr-actor 1e-4 \
    --lr-critic 1e-4 \
    --save-dir ./my_checkpoints

# 仅使用 CPU 训练
python -m training.train_no_rllib_mappo --cpu

# 完整参数示例
python -m training.train_no_rllib_mappo \
    --host 127.0.0.1 \
    --port 2000 \
    --town Town02 \
    --straight-road-length 100.0 \
    --max-steps 1000 \
    --updates 1000 \
    --rollout-steps 1024 \
    --ppo-epochs 10 \
    --minibatch-size 256 \
    --lr-actor 3e-4 \
    --lr-critic 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-eps 0.2 \
    --ent-coef 0.01 \
    --max-grad-norm 0.5 \
    --hidden-dim 128 \
    --seed 42 \
    --render \
    --save-dir ./logs/minimal_mappo_ckpts \
    --save-interval 5
```

---

### 2. [DEPRECATED] train_adversarial_ppo.py - 早期对抗智能体训练

**功能：** 训练 Agent 2 作为对抗者，对抗冻结的预训练 Agent 0 & 1

**运行方式：**
```bash
# 基础用法（必须提供预训练 checkpoint）
python -m training.train_adversarial_ppo \
    --pretrained-ckpt ./logs/minimal_mappo_ckpts/ckpt_100.pt

# 同时启动可视化
python -m training.run_adversarial_training_with_viz \
    --pretrained-ckpt ./logs/minimal_mappo_ckpts/ckpt_100.pt
```

**常用参数：**
```bash
# 基础对抗训练
python -m training.train_adversarial_ppo \
    --pretrained-ckpt <checkpoint_path> \
    --adversarial-coef 1.0

# 启用课程学习（推荐）
python -m training.train_adversarial_ppo \
    --pretrained-ckpt <checkpoint_path> \
    --curriculum-learning \
    --warmup-updates 100 \
    --ramp-up-updates 200

# 自定义训练参数
python -m training.train_adversarial_ppo \
    --pretrained-ckpt <checkpoint_path> \
    --updates 2000 \
    --rollout-steps 1024 \
    --lr-actor 3e-4 \
    --adversarial-coef 2.0 \
    --no-viz

# 完整参数示例
python -m training.train_adversarial_ppo \
    --host 127.0.0.1 \
    --port 2000 \
    --town Town02 \
    --straight-road-length 100.0 \
    --timestep 0.1 \
    --max-steps 1000 \
    --pretrained-ckpt <checkpoint_path> \
    --updates 1000 \
    --rollout-steps 1024 \
    --ppo-epochs 10 \
    --minibatch-size 256 \
    --lr-actor 3e-4 \
    --lr-critic 3e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-eps 0.2 \
    --ent-coef 0.01 \
    --max-grad-norm 0.5 \
    --hidden-dim 128 \
    --seed 42 \
    --cpu \
    --save-dir ./logs/adversarial_ppo_ckpts \
    --save-interval 50 \
    --adversarial-coef 1.0 \
    --curriculum-learning \
    --warmup-updates 100 \
    --ramp-up-updates 200 \
    --no-viz
```

---

### 3. [DEPRECATED] train_rllib.py - 历史 RLlib PPO 训练

**历史功能：** 使用 Ray RLlib 的共享 PPO 配置；没有当前 MAPPO centralized critic，不得称为当前 MAPPO 基线。

**运行方式：**
```bash
python -m training.train_rllib
```

**常用参数：**
```bash
# 基础用法
python -m training.train_rllib \
    --num-gpus 0 \
    --num-workers 0 \
    --train-iter 1000

# 开启可视化（仅单进程模式）
python -m training.train_rllib \
    --num-workers 0 \
    --render

# 使用预训练模型初始化
python -m training.train_rllib \
    --pretrained-model ./pretrained_models/ckpt.pt
```

---

### 4. pretrain_with_manual_data.py - 手动数据预训练

**功能：** 使用手动驾驶数据进行行为克隆预训练

**运行方式：**
```bash
python -m training.pretrain_with_manual_data \
    --manual-data-dir ./data/manual_driving_data
```

**常用参数：**
```bash
python -m training.pretrain_with_manual_data \
    --manual-data-dir ./data/manual_driving_data \
    --output-dir ./pretrained_models \
    --pretrain-steps 2000 \
    --batch-size 256 \
    --learning-rate 1e-4
```

---

## 评估脚本

### 1. evaluate_base_policy_c0.py - 当前基础策略 C0 evaluator

**功能：** 按冻结 Town05 C0 契约对 schema-v4 periodic checkpoint 执行无蓝车/无额外交通的 gate 或 held-out validation，输出逐 episode 证据、汇总和清理审计。以下是正式 u5 的主/重复 gate 完整顺序；两个 EvalID 必须不同，但 checkpoint、合同和 profile 必须相同。

```powershell
$runId = "<P0_BASE_MAPPO_Town05_s42_YYYYMMDD_HHMMSS>"
$checkpoint = "logs/$runId/periodic/ckpt_update_000005.pt"
$c0Contract = "configs/evaluation/p0_base_c0_town05_v1.json"

# 主 C0
$evalA = "P0_BASE_C0_gate_u000005_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
..\..\..\venv\Scripts\python.exe -m evaluation.evaluate_base_policy_c0 `
  --eval-contract $c0Contract `
  --checkpoint $checkpoint `
  --profile gate `
  --eval-id $evalA `
  --output-dir "evaluations/$evalA"

..\..\..\venv\Scripts\python.exe -m evaluation.validate_base_c0_run `
  --output-dir "evaluations/$evalA" `
  --expected-contract $c0Contract `
  --expected-checkpoint $checkpoint `
  --expected-profile gate `
  --expected-eval-id $evalA

# 重复 C0；性能只计主 C0，第二次只用于离散重复性审计
$evalB = "P0_BASE_C0_gate_u000005_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
..\..\..\venv\Scripts\python.exe -m evaluation.evaluate_base_policy_c0 `
  --eval-contract $c0Contract `
  --checkpoint $checkpoint `
  --profile gate `
  --eval-id $evalB `
  --output-dir "evaluations/$evalB"

..\..\..\venv\Scripts\python.exe -m evaluation.validate_base_c0_run `
  --output-dir "evaluations/$evalA" `
  --expected-contract $c0Contract `
  --expected-checkpoint $checkpoint `
  --expected-profile gate `
  --expected-eval-id $evalA `
  --repeat-output-dir "evaluations/$evalB"
```

完整 profile、恢复、重复性和 HOLD 规则见 `docs/P0-BASE-02_基础策略C0评估协议.md`。

---

### 1B. judge_base_go_hold.py - P0-BASE-04 唯一离线数值裁决

**功能：** 同时校验完整训练证据、formal periodic checkpoint、两次 C0、离散重复性和预注册 Git blob，再按冻结数值门槛输出不可覆盖的 JSON 决策。累计回报不参与 GO。

```powershell
# 无 CARLA 契约预检
..\..\..\venv\Scripts\python.exe -m evaluation.base_go_hold_protocol `
  --contract configs/evaluation/p0_base_go_hold_town05_v1.json

# 当前示例是 update 5；EvalA/EvalB 必须是同 checkpoint 的两次独立 gate
$decisionId = "P0_BASE_GO_HOLD_u000005_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
..\..\..\venv\Scripts\python.exe -m evaluation.judge_base_go_hold `
  --mode milestone `
  --go-hold-contract configs/evaluation/p0_base_go_hold_town05_v1.json `
  --run-dir "logs/$runId" `
  --checkpoint $checkpoint `
  --c0-output-dir "evaluations/$evalA" `
  --c0-repeat-output-dir "evaluations/$evalB" `
  --output-dir "decisions/base_mappo/$decisionId" `
  --expected-update 5
```

u5 是 Town05 的第一个里程碑，不提供 `--previous-decision-report`；从 u10 开始必须用该参数绑定同一 Run 的相邻上一里程碑报告。`HOLD`、`PILOT_NO_GO`、`VALIDATION_REJECT` 返回非 0。Town02 v1/v2 文档只解释历史裁决；当前 Town05 语义见迁移设计文档。

---

### 2. evaluate_no_rllib_mappo.py - 历史 MAPPO 诊断

**功能：** 加载 checkpoint 并评估正常智能体的表现

**运行方式：**
```bash
python -m evaluation.evaluate_no_rllib_mappo --checkpoint <path>
```

---

### 3. evaluate_adversarial_ppo.py - 历史对抗模型诊断

**功能：** 评估训练好的对抗智能体

**运行方式：**
```bash
python -m evaluation.evaluate_adversarial_ppo --checkpoint <path>
```

---

### 4. evaluate_with_speedup.py - 加速场景评估

**功能：** 在加速场景中评估对抗智能体的鲁棒性

**运行方式：**
```bash
python -m evaluation.evaluate_with_speedup --checkpoint <path>
```

---

## 数据采集工具

### record_manual_control.py - 手动驾驶数据采集

**功能：** 通过 pygame 界面手动驾驶并记录数据

**运行方式：**
```bash
python -m utils.record_manual_control --agent-id 0 --output-dir ./data/manual_driving_data
```

**控制方式：**
| 按键 | 功能 |
|------|------|
| W / ↑ | 油门 |
| S / ↓ | 刹车 |
| A / ← | 左转 |
| D / → | 右转 |
| ESC | 结束当前 episode 并保存 |
| Q | 退出程序 |

**常用参数：**
```bash
# 采集 Agent 0 的数据
python -m utils.record_manual_control \
    --agent-id 0 \
    --output-dir ./data/manual_driving_data

# 采集 Agent 1 的数据
python -m utils.record_manual_control \
    --agent-id 1 \
    --output-dir ./data/manual_driving_data
```

---

## 常用参数速查表

下表主要对应历史入口，可能与当前 `robust_v2`/`train_2red_1blue_ppo` 默认值不同。执行前必须以目标模块的 `--help` 和版本化配置为准。

### 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 127.0.0.1 | CARLA 服务器地址 |
| `--port` | 2000 | CARLA 服务器端口 |
| `--town` | Town05 | 当前正式地图；Town02 仅历史显式加载 |
| `--straight-road-length` | 100.0 | 直道长度（米） |
| `--timestep` | 0.1 | 时间步长（秒） |
| `--max-steps` | 1000 | 每 episode 最大步数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--updates` | 1000 | 总更新次数 |
| `--rollout-steps` | 1024 | 每次更新的采样步数 |
| `--ppo-epochs` | 10 | PPO 迭代次数 |
| `--minibatch-size` | 256 | 小批量大小 |
| `--lr-actor` | 3e-4 | Actor 学习率 |
| `--lr-critic` | 3e-4 | Critic 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-eps` | 0.2 | PPO clip 阈值 |
| `--ent-coef` | 0.01 | 熵系数 |
| `--max-grad-norm` | 0.5 | 梯度裁剪范数 |
| `--hidden-dim` | 128 | 隐藏层维度 |
| `--seed` | 42 | 随机种子 |
| `--cpu` | False | 强制使用 CPU |

### 保存 & 日志

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save-dir` | 见各脚本 | Checkpoint 保存目录 |
| `--save-interval` | 5/50 | 保存间隔（updates） |
| `--log-interval` | 5 | 日志输出间隔 |
| `--render` / `--no-viz` | False | 是否启用可视化 |

### 对抗训练特有

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pretrained-ckpt` | **required** | 预训练模型路径 |
| `--adversarial-coef` | 1.0 | 对抗项权重 |
| `--curriculum-learning` | False | 启用课程学习 |
| `--warmup-updates` | 100 | 课程学习热身轮数 |
| `--ramp-up-updates` | 200 | 对抗系数渐增轮数 |

---

## 典型工作流程示例

### 当前流程 1：基础策略

```powershell
# 1. 启动 CARLA
# (在另一个终端)
CarlaUE4.exe

# 2. 先完成 TODO P0-00A 至 P0-00F-TB 和全部单元测试

# 3. 运行 P0-00G smoke（命令见“当前有效入口”）

# 4. R0/WTA v1 已在 update 10 诊断封存，禁止继续 resume/pretrained

# 5. R1–R5 已完成，最终生产入口、严格恢复、checkpoint/metrics 与 CARLA 清理验收 PASS。
#    R5 诊断 Run/权重永久禁止作为 resume/pretrained/选模来源。

# 6. 下一步只按“当前有效入口”中的正式命令，使用 p0_base_town05_v1.json、
#    新 RunID 和随机初始化运行 seed 42 到 u5；随后必须双 Town05 C0 + GO/HOLD。
#    未得到该停点裁决前不得运行 u6–u10，也不得放行多 seed。
```

#### 历史 v1 checkpoint/resume（封存，不得执行）

```powershell
# 不提供可复制执行的 resume/pretrained 命令，防止误续 R0。
# 当前代码会以 schema/action_spec/model_spec 严格拒绝 WTA v1 权重。
```

当前 schema v4 checkpoint 会绑定二维 `action_spec`、reward v3、ValueNorm 状态和 critic spec；`--resume latest` 不存在、哈希损坏、Git/动作/奖励/Critic/指标/训练配置/实验契约不一致时直接失败，不会从零启动或部分加载。训练期不产生 `best.pt`；未来仅由独立验证从 immutable periodic checkpoint 生成 `best_base_valid.pt`。R1/R2/R3 原理分别见对应论文正文，R4/R5 阶段证据见相应记录，checkpoint 文件结构见 `docs/MAPPO_checkpoint与恢复训练协议.md`。

### 当前流程 2：单蓝车对抗

```bash
# 1. 启动 CARLA
CarlaUE4.exe

# 2. 前置：P0-06 已通过，且已有 best_base_valid.pt

# 3. 先运行 Phase 1 三更新 smoke（命令见“当前有效入口”）

# 4. smoke 验收后，使用唯一 RunID 和冻结配置运行 train_2red_1blue_ppo
python -m training.train_2red_1blue_ppo \
    --pretrained-ckpt <best_base_valid.pt> \
    --scenario scenario_01_basic.json \
    <frozen-config-args>

# 5. 用 C0/C1/C2 独立评估器选择 best_valid.pt，不使用训练 rollout 选 best
```

---

## TensorBoard 监控

当前指标协议只对 `robust_v2` 生效。每个 Run 的 `tb/`、`metrics.jsonl` 与 checkpoint `last_metrics` 来自同一 update payload；旧训练入口的历史日志不得与当前协议混用。

```powershell
# 启动当前 Run 的 TensorBoard
tensorboard --logdir logs/<RunID>/tb

# P0-00G 严格验收逐 episode、PPO、update 连续及 TB/JSONL 对账
..\..\..\venv\Scripts\python.exe -m utils.validate_tensorboard_run `
  --run-dir logs/<RunID> `
  --expected-updates 1 2 3 4 `
  --expected-episodes-per-update 4 `
  --require-ppo-updates `
  --max-preupdate-ratio-error 1e-6
```

然后在浏览器访问 `http://localhost:6006`。完整标签和全零奖励分量的判断方法见 [MAPPO TensorBoard 指标协议](docs/MAPPO_TensorBoard指标协议.md)。

---

## 常见问题

**Q: 提示找不到模块？**
```
A: 确保从根目录以模块方式运行：
python -m training.train_no_rllib_mappo_robust_v2
而非：python training/train_no_rllib_mappo_robust_v2.py
```

**Q: 如何从 checkpoint 继续训练？**
```
A: 同一 Run 使用：
python -m training.train_no_rllib_mappo_robust_v2 --resume latest \
  --save-dir logs/<original-run-id> --updates <new-stop-update>

只允许从 P0-00F-TB 当前 schema v2 协议生成且 checksum、manifest、Git、结构、配置和 `last_metrics` 校验通过的 checkpoint resume。
旧 RLlib/minimal/robust v1/旧对抗 checkpoint 默认只能作历史评估或 pretrained weights，不能作为正式 resume 起点。
```

**Q: 显存不足？**
```
A: 尝试减小 batch size：
--minibatch-size 64
或使用 CPU：
--cpu
```
