
# 运行指令参考

本文档保留项目脚本的历史运行方式，但新实验必须遵守下面的入口冻结规则。阶段顺序和是否允许运行以 [毕业课题研究推进 TODO](docs/毕业课题_研究推进TODO.md) 为准。

> **入口冻结（2026-07-20）**
>
> - 基础策略唯一入口：`training.train_no_rllib_mappo_robust_v2`。
> - Phase 1 单蓝车唯一入口：`training.train_2red_1blue_ppo`。
> - RLlib、shared-policy、minimal/robust v1、旧 adversarial PPO 和旧 multi 入口全部是 **DEPRECATED**，不得用于新训练、resume、正式 checkpoint 或论文结果。
> - P0-00G 和 P0-BASE-00/01/02/04 已通过；当前只允许 seed 42 从 update 5 开始分段试训，每段必须执行 strict validator、两次 C0 gate 和冻结 judge；只有 update 100 `PILOT_GO` 后才允许多 seed 长训练。对抗策略长训练仍需等待 P0-01 至 P0-06 和 P0-05M。

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
  --scenario data/training_scenarios/scenario_01_basic.json `
  --seed 42 `
  --follow-spectator

# B. A 通过后运行：近终点、低油门直行
..\..\..\venv\Scripts\python.exe -B -m evaluation.smoke_test_success_logic `
  --mode scripted `
  --scenario data/training_scenarios/scenario_01_basic.json `
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
  --scenario scenario_01_basic.json `
  --updates 3 `
  --rollout-episodes 4 `
  --no-viz `
  --seed 42 `
  --save-dir logs/P1_single_blue_basic_s42_<timestamp>
```

正式训练参数必须固化到版本控制配置中并登记 RunID；不要直接把 smoke 命令扩成上百 updates。

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

**功能：** 按冻结 `p0_base_c0_eval_v1` 契约对 periodic checkpoint 执行无蓝车/无额外交通的 gate 或 held-out validation，输出逐 episode 证据、汇总和清理审计。

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evalId = "P0_BASE_C0_gate_u000005_$stamp"
..\..\..\venv\Scripts\python.exe -m evaluation.evaluate_base_policy_c0 `
  --checkpoint logs/<RunID>/periodic/ckpt_update_000005.pt `
  --profile gate `
  --eval-id $evalId `
  --output-dir "evaluations/$evalId"

..\..\..\venv\Scripts\python.exe -m evaluation.validate_base_c0_run `
  --output-dir "evaluations/$evalId" `
  --expected-checkpoint logs/<RunID>/periodic/ckpt_update_000005.pt
```

完整 profile、恢复、重复性和 HOLD 规则见 `docs/P0-BASE-02_基础策略C0评估协议.md`。

---

### 1B. judge_base_go_hold.py - P0-BASE-04 唯一离线数值裁决

**功能：** 同时校验完整训练证据、formal periodic checkpoint、两次 C0、离散重复性和预注册 Git blob，再按冻结数值门槛输出不可覆盖的 JSON 决策。累计回报不参与 GO。

```powershell
# 无 CARLA 契约预检
..\..\..\venv\Scripts\python.exe -m evaluation.base_go_hold_protocol `
  --contract configs/evaluation/p0_base_go_hold_v1.json

# 当前示例是 update 5；EvalA/EvalB 必须是同 checkpoint 的两次独立 gate
$decisionId = "P0_BASE_GO_HOLD_u000005_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
..\..\..\venv\Scripts\python.exe -m evaluation.judge_base_go_hold `
  --mode milestone `
  --go-hold-contract configs/evaluation/p0_base_go_hold_v1.json `
  --run-dir logs/<RunID> `
  --checkpoint logs/<RunID>/periodic/ckpt_update_000005.pt `
  --c0-output-dir evaluations/<EvalA> `
  --c0-repeat-output-dir evaluations/<EvalB> `
  --output-dir "decisions/base_mappo/$decisionId" `
  --expected-update 5
```

`HOLD`、`PILOT_NO_GO`、`VALIDATION_REJECT` 返回非 0。完整门槛、比较符和 update 60/80/100 状态含义见 `docs/P0-BASE-04_GO-HOLD数值门槛预注册.md`。

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
| `--town` | Town02 | 地图名称 |
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

# 4. 已完成 P0-BASE-00/01/02/04：冻结源码、训练契约、C0 evaluator 和 GO/HOLD 数值裁决

# 5. 只用冻结契约运行 seed 42 的第一个 5-update 分段
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runId = "P0_BASE_MAPPO_Town02_s42_$stamp"
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --experiment-contract configs/base_mappo/p0_base_pilot_v1.json `
  --updates 5 `
  --run-id $runId `
  --save-dir "logs/$runId"

# 6. 执行 strict validator、两次独立 C0 gate 和 P0-BASE-04 judge；只有裁决允许才 resume 到下一停点
# 7. 只有 update 100 PILOT_GO 后才进入多 seed；P0-BASE-06 再从 held-out validation 生成 best_base_valid.pt
```

#### 基础策略正式契约 checkpoint/resume（P0-00F + P0-BASE-01）

```powershell
# 同一正式 Run 严格恢复到下一个登记停点；契约 SHA、配置、seed、LR horizon 均校验
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --experiment-contract configs/base_mappo/p0_base_pilot_v1.json `
  --updates 10 `
  --run-id $runId `
  --save-dir "logs/$runId" `
  --resume latest

# 以下 pretrained 语义只供新契约设计或历史诊断；P0_BASE_MAPPO_TOWN02_PILOT_V1 明确禁止 pretrained

# 从已有权重开启新 Run；不会继承 optimizer/update/课程/RNG
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --pretrained-checkpoint logs/<source-run>/periodic/ckpt_update_000050.pt `
  --run-id <new-run-id> `
  --save-dir logs/<new-run-id> `
  --updates 100 `
  --seed 43
```

`--resume latest` 不存在、哈希损坏、Git/协议/训练配置/实验契约不一致时会直接失败，不会从零启动或部分加载。正式 CLI 只允许 endpoint/device/登记停点/RunID/目录/resume，科学参数覆盖会 fail-fast。训练期不再产生 `best.pt`；后续仅由独立 C0/C1 验证从 immutable periodic checkpoint 生成 `best_base_valid.pt`。正式参数和证据链见 `docs/P0-BASE-01_基础策略正式实验契约.md`，checkpoint 文件结构见 `docs/MAPPO_checkpoint与恢复训练协议.md`。

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
