
# CARLA 多智能体对抗训练项目

基于 CARLA 模拟器的多智能体强化学习项目，实现 MAPPO（Multi-Agent PPO）训练和对抗智能体训练。

## 当前研究入口冻结（2026-07-20）

新实验必须先阅读 [毕业课题研究推进 TODO](docs/毕业课题_研究推进TODO.md)，并且只能从以下入口开始：

| 用途 | 唯一有效入口 | 当前许可 |
|---|---|---|
| 基础红车策略 | `python -m training.train_no_rllib_mappo_robust_v2` | R0 Run 已诊断封存，R1 二维动作与 R2 有界奖励协议已完成；当前被 R3–R5 阻塞，禁止启动/恢复 CARLA 正式训练 |
| Phase 1 单蓝车 | `python -m training.train_2red_1blue_ppo` | P0-06 通过前禁止启动 |

以下入口均为 **DEPRECATED/历史参考**：`train_rllib.py`、`carla_rllib_wrapper.py`、`train_no_rlib_sharedpolicy.py`、`train_no_rllib_mappo.py`、`train_no_rllib_mappo_robust.py`、`train_adversarial_ppo.py`、`run_adversarial_training_with_viz.py` 以及旧 multi v1/v2。它们不得用于新训练、恢复修复后的训练、生成正式 checkpoint 或论文结果；未来确需使用时，必须先迁移当前生命周期、GAE、指标和 checkpoint 协议并重新验收。

---

## 快速导航

| 文档 | 用途 |
|------|------|
| **[RUN_COMMANDS.md](RUN_COMMANDS.md)** | ⭐ **必读！** 所有脚本的运行方式和参数参考 |
| **[FILE_INDEX.md](FILE_INDEX.md)** | 各文件功能详解索引 |
| **[MANUAL_DATA_USAGE.md](MANUAL_DATA_USAGE.md)** | 手动驾驶数据采集指南 |
| **[docs/毕业课题_研究推进TODO.md](docs/毕业课题_研究推进TODO.md)** | 当前唯一任务顺序、阶段门槛和实验登记 |
| **[docs/P0-BASE-01_基础策略正式实验契约.md](docs/P0-BASE-01_基础策略正式实验契约.md)** | 正式 JSON 参数、分段停点、HOLD 规则、Run 证据链和标准命令 |
| **[docs/P0-BASE-02_基础策略C0评估协议.md](docs/P0-BASE-02_基础策略C0评估协议.md)** | 基础 checkpoint 的固定 C0 case、确定性动作、证据 schema、validator 和重复性命令 |
| **[docs/P0-BASE-04_GO-HOLD数值门槛预注册.md](docs/P0-BASE-04_GO-HOLD数值门槛预注册.md)** | 训练健康、各停点能力、held-out 下界、Git 预注册证据和唯一离线裁决命令 |
| **[docs/P0-BASE_基础策略实验记录.md](docs/P0-BASE_基础策略实验记录.md)** | 基础策略 fresh/resume/C0/GO-HOLD 人工实验台账 |
| **[docs/MAPPO_TensorBoard指标协议.md](docs/MAPPO_TensorBoard指标协议.md)** | 当前基础 MAPPO 的奖励审计、聚合口径、训练健康指标和离线验证 |
| **[docs/MAPPO_车辆动作空间与边界指标协议.md](docs/MAPPO_车辆动作空间与边界指标协议.md)** | 二维 signed-longitudinal 动作、连续死区、二维策略/三维环境审计及 v1 只读兼容 |
| **[docs/R1_二维纵向动作协议_论文正文与实验设计.md](docs/R1_二维纵向动作协议_论文正文与实验设计.md)** | 可直接用于论文的车辆动力学、动作映射、PPO 概率推导、文献、风险与测试计划 |
| **[docs/R2_奖励预算校准_论文正文与实验设计.md](docs/R2_奖励预算校准_论文正文与实验设计.md)** | reward v3 的论文依据、预算不等式、轨迹排序、敏感性与测试计划 |
| **[docs/P0-00G_真实CARLA_smoke运行规范.md](docs/P0-00G_真实CARLA_smoke运行规范.md)** | P0-00G 环境门槛、冻结命令、受控中断、产物和 PASS/HOLD 标准 |
| **[docs/P0-00G_真实CARLA_smoke测试记录.md](docs/P0-00G_真实CARLA_smoke测试记录.md)** | P0-00G 逐次结果、失败根因、修复验证与最终 Judge 裁决 |
| **[docs/P0-00G-SUCCESS_真实success逻辑测试规范与记录.md](docs/P0-00G-SUCCESS_真实success逻辑测试规范与记录.md)** | 无 checkpoint 的固定场景 success/reward/lifecycle 隔离测试、命令与逐次记录 |

---

## 项目结构

所有脚本文件位于根目录，按功能分类如下：

```
mappo_training/
├── 环境模块
│   ├── carla_multi_agent_env.py    # CARLA多智能体环境（核心）
│   ├── base_reward_v3.py           # [CURRENT] 基础策略有界奖励数学（无 CARLA 依赖）
│   ├── carla_rllib_wrapper.py      # [DEPRECATED] RLlib环境包装器
│   └── speedup_env_wrapper.py      # 提速环境包装器
├── 训练脚本
│   ├── train_no_rllib_mappo_robust_v2.py # [CURRENT] 基础红车 MAPPO
│   ├── train_2red_1blue_ppo.py          # [CURRENT] Phase 1 单蓝车 PPO
│   ├── train_rllib.py                   # [DEPRECATED] RLlib训练
│   ├── train_no_rllib_mappo.py          # [DEPRECATED] 早期独立MAPPO
│   ├── train_no_rlib_sharedpolicy.py    # [DEPRECATED] 共享策略训练
│   ├── train_adversarial_ppo.py         # [DEPRECATED] 早期对抗PPO
│   ├── run_adversarial_training_with_viz.py  # [DEPRECATED] 早期启动器
│   └── pretrain_with_manual_data.py    # 手动数据预训练
├── 评估脚本
│   ├── evaluate_base_policy_c0.py      # [CURRENT] 基础策略冻结 C0 evaluator
│   ├── validate_base_c0_run.py         # [CURRENT] C0 产物重算/重复性 validator
│   ├── evaluate_rllib.py
│   ├── evaluate_rllib_checkpoint.py
│   ├── evaluate_no_rllib_mappo.py
│   ├── evaluate_adversarial_ppo.py
│   ├── evaluate_adversarial.py
│   └── evaluate_with_speedup.py
├── 可视化模块
│   ├── visualize_adversarial_training.py  # 对抗训练可视化
│   └── mappo_visualizer.py
├── 工具模块
│   ├── mappo_logger.py           # 日志工具
│   ├── manual_data_loader.py     # 手动数据加载器
│   └── record_manual_control.py  # 手动驾驶数据采集
├── 数据/配置
│   ├── custom_positions_example.json  # 自定义位置示例
│   ├── manual_driving_data/           # 手动驾驶数据（目录）
│   └── mappo_models/                  # MAPPO模型定义（目录）
├── 输出目录
│   ├── logs/                  # 训练日志
│   └── checkpoints/           # 检查点
├── 文档
│   ├── README.md             # 本文件
│   ├── FILE_INDEX.md         # 文件功能索引
│   ├── RUN_COMMANDS.md       # 运行指令参考（必读！）
│   └── MANUAL_DATA_USAGE.md  # 手动数据使用指南
└── requirements.txt          # 依赖
```

**详细文件功能索引请查看 [FILE_INDEX.md](FILE_INDEX.md)**

## 功能特性

### 1. 多智能体环境
- 支持 Town02 地图中的直道场景
- 2-3 个智能体同时训练
- 19维观察空间（8维基础状态 + 11维雷达感知）
- 当前基础策略为 2 维策略动作（signed-longitudinal, steer），经连续死区映射为 CARLA 3 维控制（throttle, steer, brake）
- 碰撞检测、车道偏离检测
- 自定义车辆起始位置

### 2. 奖励函数
- **唯一正稠密任务信号**：折扣一致、取值有界的路线完成度势能，往返运动不能刷正回报
- **终局预算**：成功 `+10`、碰撞 `-20`，同一 tick 冲线碰撞时碰撞优先
- **有界辅助成本**：超速、实际执行纵向命令变化、转向变化、LiDAR、TTC、车道偏移与越线事件
- **防停车投机**：取消正 speed/alive/centering/yaw、固定 step 与 time bonus；只有实际减少目标距离才产生正稠密奖励
- **协议隔离**：reward v3 仅用于基础红车策略；尚未迁移的对抗奖励仍显式使用历史 v2

### 3. 对抗智能体
Agent 2 作为对抗智能体，专用奖励函数包括：
- 基础驾驶奖励（低权重）
- 侧面切入奖励（在其他车辆侧面行驶）
- 阻挡奖励（在其他车辆前方行驶）
- 零和对抗项（降低其他智能体奖励）

### 4. 手动数据预训练
支持通过手动驾驶采集专家数据，用于行为克隆预训练：
- pygame 手动控制界面
- 数据加载和过滤
- 预训练模型初始化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行脚本

当前训练入口统一使用模块方式运行：

所有当前训练脚本都应从 `mappo_training/` 目录以模块方式运行：

```bash
python -m training.train_no_rllib_mappo_robust_v2 --help
python -m training.train_2red_1blue_ppo --help
python -m utils.record_manual_control --agent-id 0
```

不要从 `training/` 子目录直接执行文件，也不要从本文档历史段落复制 deprecated 命令。

`p0_base_pilot_v1.json`、WTA checkpoint 和 GO/HOLD v1 已全部封存为历史证据，不得继续 resume/pretrained。R1 已建立 action-v2，R2 已建立 reward-v3 训练/C0/metrics 协议，但仍不构成开训许可；必须按 TODO 完成 R3 Critic、R4 GO/HOLD v2 与 R5 真实 smoke，之后才允许用最终新契约、新 RunID 和随机初始化 fresh pilot。评估协议见 [P0-BASE-02 C0 协议](docs/P0-BASE-02_基础策略C0评估协议.md)，理论见 [R1 动作正文](docs/R1_二维纵向动作协议_论文正文与实验设计.md)与 [R2 奖励正文](docs/R2_奖励预算校准_论文正文与实验设计.md)。

依赖包括：
- torch &gt;= 1.9.0
- numpy &gt;= 1.19.0
- carla &gt;= 0.9.16
- pygame &gt;= 2.0.0
- tensorboard &gt;= 2.0.0
- ray[rllib]（仅历史 RLlib 入口需要；当前研究训练不使用）

## 快速开始

### 1. 启动 CARLA 服务器

```bash
# Windows
CarlaUE4.exe

# Linux
./CarlaUE4.sh
```

### 1.1 无 checkpoint 验证 success 逻辑

不要先用未收敛策略验证终点逻辑。下面的入口在真实 CARLA 中顺序把车辆传送到终点，直接审计 success 奖励、`terminated/truncated`、`active_mask` 和 `all_agents_reached`；通过后再运行 `--mode scripted` 的近终点低油门短距离测试。

```powershell
..\..\..\venv\Scripts\python.exe -B -m evaluation.smoke_test_success_logic `
  --mode teleport `
  --scenario data/training_scenarios/scenario_01_basic.json `
  --seed 42 `
  --follow-spectator

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

每次运行写入唯一的 `logs/success_logic_smoke_<timestamp>/`，完整门槛见 [success 逻辑测试规范与记录](docs/P0-00G-SUCCESS_真实success逻辑测试规范与记录.md)。该 PASS 只证明环境成功链路，不证明 MAPPO 收敛。

### 2. 训练正常智能体（2车）

```powershell
# 当前仅允许在 TODO 的 P0-00B 至 P0-00F-TB 通过后执行 P0-00G smoke
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
```

基础策略 checkpoint 从 P0-00F 起使用版本化严格协议：每个完整 update 原子更新 `latest.pt`，按间隔生成不可覆盖的 `periodic/ckpt_update_*.pt`，中断时只复制最近完整边界到 `interrupted.pt`。同一实验使用 `--resume latest --save-dir <原 Run 目录>`；从权重开启新实验使用 `--pretrained-checkpoint` 和新 RunID。详见 [checkpoint 与恢复训练协议](docs/MAPPO_checkpoint与恢复训练协议.md)。

### 3. 训练对抗智能体（3车）

P0-06 通过后，使用经过独立 C0/C1 验证的基础红车 checkpoint：

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_2red_1blue_ppo `
  --pretrained-ckpt <best_base_valid.pt> `
  --scenario scenario_01_basic.json `
  --updates 3 `
  --rollout-episodes 4 `
  --no-viz `
  --seed 42 `
  --save-dir logs/P1_single_blue_basic_s42_<timestamp>
```

### 4. 评估

基础策略专用、无蓝车 C0 evaluator 已冻结；对抗研究使用的统一 C0/C1/C2 evaluator 仍在 P0-04 计划中。

```powershell
python -m evaluation.evaluate_base_policy_c0 --help
python -m evaluation.validate_base_c0_run --help
```

下面的旧评估入口只可读取历史 checkpoint 做诊断，不得形成当前协议结论：

```bash
# [DEPRECATED] 评估历史 RLlib 智能体
python evaluate_rllib_checkpoint.py --checkpoint /path/to/checkpoint

# [DEPRECATED] 评估历史对抗智能体
python evaluate_adversarial_ppo.py --checkpoint /path/to/checkpoint
```

## 详细使用指南

### 手动驾驶数据采集

详细指南请参考 [MANUAL_DATA_USAGE.md](MANUAL_DATA_USAGE.md)

```bash
# 采集Agent 0数据
python record_manual_control.py --agent-id 0 --output-dir ./manual_driving_data

# 采集Agent 1数据
python record_manual_control.py --agent-id 1 --output-dir ./manual_driving_data
```

控制方式：
- W/↑: 油门
- S/↓: 刹车
- A/←: 左转
- D/→: 右转
- ESC: 结束当前episode
- Q: 退出程序

### 自定义起始位置

编辑 `custom_positions_example.json`:

```json
{
  "agent0": {"x": -7.45, "y": 176, "z": 0.5, "yaw": 90.0},
  "agent1": {"x": -7.45, "y": 210, "z": 0.5, "yaw": 90.0},
  "agent2": {"x": -3.0, "y": 185, "z": 0.5, "yaw": 90.0}
}
```

### TensorBoard 监控

当前 `robust_v2` 将同一 update payload 写入 `<RunID>/tb/` 和 `<RunID>/metrics.jsonl`，并将逐 episode 生命周期原始证据写入 `<RunID>/episodes.jsonl`。奖励、PPO/Critic、动作、模型更新、场景和耗时的完整口径见 [TensorBoard 指标协议](docs/MAPPO_TensorBoard指标协议.md)。

```powershell
tensorboard --logdir logs/<RunID>/tb

..\..\..\venv\Scripts\python.exe -m utils.validate_tensorboard_run `
  --run-dir logs/<RunID> `
  --expected-updates 1 2 3 4 `
  --expected-episodes-per-update 4 `
  --require-ppo-updates `
  --max-preupdate-ratio-error 1e-6
```

验证器返回 `"status": "PASS"` 后，仍需在 P0-00G 人工检查真实 CARLA 的 LiDAR/TTC、终局和 PPO 曲线。旧训练入口的历史 TensorBoard 不属于当前指标协议。

## 训练流程建议

### 阶段 0：代码与指标可信化
1. 严格完成 TODO 的 P0-00A 至 P0-00G
2. 完成 P0-BASE-00/01/02 后，用单 seed 分段试训并按预注册门槛作 GO/HOLD
3. P0-BASE-04 GO 后才能扩展多 seed，并通过独立 C0 选择基础 checkpoint
4. 基础 checkpoint 合格后完成 P0-01 至 P0-06 和 P0-05M，再进入对抗训练

### 阶段1：正常智能体训练
1. （可选）采集手动驾驶数据
2. （可选）使用手动数据预训练
3. 只使用 `train_no_rllib_mappo_robust_v2.py` 训练2个正常智能体
4. 通过独立 C0/C1 验证生成 `best_base_valid.pt`

### 阶段2：对抗智能体训练
1. 加载阶段1的预训练模型
2. 启用课程学习（前期只用基础奖励）
3. 逐渐增加对抗项权重
4. 训练对抗智能体

### 阶段3：评估和迭代
1. 在标准场景评估
2. 在提速场景评估（使用 `SpeedupEnvWrapper`）
3. 根据结果调整奖励函数

## 关键文件说明

### carla_multi_agent_env.py
核心环境类，主要方法：
- `__init__()`: 初始化CARLA连接、生成车辆、挂载传感器
- `reset()`: 重置环境
- `step(action_dict)`: 执行一步动作
- `_compute_rewards()`: 计算奖励
- `_compute_adversarial_reward()`: 对抗智能体专用奖励
- `set_custom_start_positions()`: 设置自定义起始位置

### train_2red_1blue_ppo.py
对抗训练脚本：
- Agent 0/1: 加载预训练权重，冻结
- Agent 2: PPO训练
- 使用统一对抗奖励、KL anchor 和 active-mask 生命周期
- 仅在 TODO 的 Phase 1 门槛开放后运行

### speedup_env_wrapper.py
提速环境包装器，用于评估对抗智能体在更激进场景下的表现：
- 固定提速模式
- 自适应提速模式

## 常见问题

### CARLA连接失败
检查：
- CARLA服务器是否已启动
- host/port 是否正确
- 防火墙设置

### 车辆生成失败
- 检查起始位置是否合法
- 尝试使用自定义位置

### 训练不收敛
- 调整学习率
- 检查奖励函数设计
- 增加训练步数
- 考虑使用手动数据预训练

## 许可证

本项目仅供学习研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request。
