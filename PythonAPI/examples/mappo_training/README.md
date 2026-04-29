
# CARLA 多智能体对抗训练项目

基于 CARLA 模拟器的多智能体强化学习项目，实现 MAPPO（Multi-Agent PPO）训练和对抗智能体训练。

---

## 快速导航

| 文档 | 用途 |
|------|------|
| **[RUN_COMMANDS.md](RUN_COMMANDS.md)** | ⭐ **必读！** 所有脚本的运行方式和参数参考 |
| **[FILE_INDEX.md](FILE_INDEX.md)** | 各文件功能详解索引 |
| **[MANUAL_DATA_USAGE.md](MANUAL_DATA_USAGE.md)** | 手动驾驶数据采集指南 |

---

## 项目结构

所有脚本文件位于根目录，按功能分类如下：

```
mappo_training/
├── 环境模块
│   ├── carla_multi_agent_env.py    # CARLA多智能体环境（核心）
│   ├── carla_rllib_wrapper.py      # RLlib环境包装器
│   └── speedup_env_wrapper.py      # 提速环境包装器
├── 训练脚本
│   ├── train_rllib.py                  # RLlib训练
│   ├── train_no_rllib_mappo.py         # 独立MAPPO训练
│   ├── train_no_rlib_sharedpolicy.py   # 共享策略训练
│   ├── train_adversarial_ppo.py        # 对抗智能体训练
│   ├── run_adversarial_training_with_viz.py  # 对抗训练启动器
│   └── pretrain_with_manual_data.py    # 手动数据预训练
├── 评估脚本
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
- 3维动作空间（throttle, steer, brake）
- 碰撞检测、车道偏离检测
- 自定义车辆起始位置

### 2. 奖励函数
- **进度奖励**: 向目标前进的距离
- **速度奖励**: 鼓励合理速度
- **居中奖励**: 保持车道居中
- **朝向奖励**: 朝向目标方向
- **碰撞惩罚**: -100
- **车道偏离惩罚**: -5
- **时间步惩罚**: -0.05（鼓励尽快到达）
- **成功奖励**: +100

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

有两种方式运行子目录中的脚本：

**方式1：从子目录运行**
```bash
cd training
python train_adversarial_ppo.py --pretrained-ckpt <path>
```

**方式2：从根目录使用模块运行（推荐）**
```bash
python -m training.train_adversarial_ppo --pretrained-ckpt <path>
python -m evaluation.evaluate_adversarial_ppo --checkpoint <path>
python -m utils.record_manual_control --agent-id 0
```

依赖包括：
- torch &gt;= 1.9.0
- numpy &gt;= 1.19.0
- carla &gt;= 0.9.16
- pygame &gt;= 2.0.0
- tensorboard &gt;= 2.0.0
- ray[rllib] (可选，用于RLlib训练)

## 快速开始

### 1. 启动 CARLA 服务器

```bash
# Windows
CarlaUE4.exe

# Linux
./CarlaUE4.sh
```

### 2. 训练正常智能体（2车）

```bash
# 使用RLlib训练
python train_rllib.py --town Town02 --num-agents 2

# 或使用独立MAPPO训练
python train_no_rllib_mappo.py
```

### 3. 训练对抗智能体（3车）

首先需要有预训练的正常智能体checkpoint，然后：

```bash
# 方式1：直接训练
python train_adversarial_ppo.py --pretrained-ckpt /path/to/checkpoint.pt

# 方式2：带可视化启动
python run_adversarial_training_with_viz.py --pretrained-ckpt /path/to/checkpoint.pt
```

### 4. 评估

```bash
# 评估正常智能体
python evaluate_rllib_checkpoint.py --checkpoint /path/to/checkpoint

# 评估对抗智能体
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

训练过程中自动记录TensorBoard日志：

```bash
tensorboard --logdir logs/
```

## 训练流程建议

### 阶段1：正常智能体训练
1. （可选）采集手动驾驶数据
2. （可选）使用手动数据预训练
3. 训练2个正常智能体直到收敛

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

### train_adversarial_ppo.py
对抗训练脚本：
- Agent 0/1: 加载预训练权重，冻结
- Agent 2: PPO训练
- 支持课程学习（curriculum learning）
- 支持课程学习参数：`--warmup-updates` 和 `--ramp-up-updates`

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

