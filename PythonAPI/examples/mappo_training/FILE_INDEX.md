
# 项目文件功能索引

本文档列出项目中所有文件的功能说明，帮助快速定位需要的文件。

---

## 目录

1. [环境模块](#环境模块)
2. [训练脚本](#训练脚本)
3. [评估脚本](#评估脚本)
4. [可视化模块](#可视化模块)
5. [工具模块](#工具模块)
6. [数据与配置](#数据与配置)
7. [文档](#文档)

---

## 环境模块

| 文件名 | 功能说明 | 关键类/函数 |
|--------|----------|-------------|
| **carla_multi_agent_env.py** | CARLA多智能体环境（核心文件） | `CarlaMultiAgentEnv` |
| | - 连接CARLA服务器、生成车辆 | `__init__()` |
| | - 观察空间：19维（8基础 + 11雷达） | `_get_observations()` |
| | - 动作空间：3维（throttle, steer, brake） | `step()` |
| | - 碰撞/车道偏离检测 | `_attach_collision_sensor()`, `_attach_lane_invasion_sensor()` |
| | - 正常智能体奖励函数 | `_compute_rewards()` |
| | - 对抗智能体专用奖励 | `_compute_adversarial_reward()` |
| | - 自定义起始位置 | `set_custom_start_positions()` |
| **carla_rllib_wrapper.py** | RLlib兼容的环境包装器 | `RLlibCarlaEnv` |
| **speedup_env_wrapper.py** | 提速环境包装器（用于评估） | `SpeedupEnvWrapper`, `SpeedupActionWrapper`, `AdaptiveSpeedupWrapper` |
| | - 固定油门放大模式 | `SpeedupActionWrapper` |
| | - 自适应速度调整模式 | `AdaptiveSpeedupWrapper` |

---

## 训练脚本

| 文件名 | 功能说明 | 使用场景 |
|--------|----------|----------|
| **train_rllib.py** | 基于Ray RLlib的MAPPO训练 | 大规模分布式训练 |
| **train_no_rllib_mappo.py** | 独立实现的MAPPO训练 | 轻量级训练，不依赖Ray |
| **train_no_rlib_sharedpolicy.py** | 共享策略的多智能体训练 | 参数共享的协同训练 |
| **train_adversarial_ppo.py** | 对抗智能体训练（核心） | 训练Agent 2作为干扰者 |
| | - 加载预训练Agent 0/1并冻结 | `--pretrained-ckpt` |
| | - Agent 2用PPO训练 | 对抗奖励 = 基础奖励 + 对抗系数 * -avg(others) |
| | - 支持课程学习 | `--curriculum-learning`, `--warmup-updates`, `--ramp-up-updates` |
| **run_adversarial_training_with_viz.py** | 对抗训练启动器（带可视化） | 同时启动训练和可视化窗口 |
| **pretrain_with_manual_data.py** | 手动驾驶数据预训练 | 用行为克隆初始化策略 |

---

## 评估脚本

| 文件名 | 功能说明 |
|--------|----------|
| **evaluate_rllib.py** | 评估RLlib训练的智能体 |
| **evaluate_rllib_checkpoint.py** | 从checkpoint加载并评估RLlib智能体 |
| **evaluate_no_rllib_mappo.py** | 评估独立MAPPO训练的智能体 |
| **evaluate_adversarial_ppo.py** | 评估对抗智能体 |
| **evaluate_adversarial.py** | 对抗场景综合评估 |
| **evaluate_with_speedup.py** | 在提速场景下评估（测试对抗智能体鲁棒性） |

---

## 可视化模块

| 文件名 | 功能说明 |
|--------|----------|
| **visualize_adversarial_training.py** | 对抗训练实时可视化 |
| | - pygame窗口显示CARLA俯视画面 |
| | - 实时显示各智能体奖励曲线 |
| **mappo_visualizer.py** | MAPPO训练可视化工具 |

---

## 工具模块

| 文件名 | 功能说明 |
|--------|----------|
| **mappo_logger.py** | 训练日志记录工具 |
| **manual_data_loader.py** | 手动驾驶数据加载器 |
| | - 加载、解析手动采集的数据 |
| | - 数据过滤、统计 |
| | - 转换为RLlib格式 |
| **record_manual_control.py** | 手动驾驶数据采集工具 |
| | - pygame控制界面 |
| | - 记录观察、动作、奖励轨迹 |

---

## 数据与配置

| 文件名 | 说明 |
|--------|------|
| **data/custom_positions_example.json** | 自定义车辆起始位置示例 |
| **data/manual_driving_data/** | 手动驾驶数据存储目录 |
| **models/mappo_models/** | MAPPO模型定义目录 |
| **logs/** | TensorBoard训练日志目录 |
| **checkpoints/** | 模型检查点保存目录 |
| **requirements.txt** | Python依赖包列表 |

---

## 文档

| 文件名 | 说明 |
|--------|------|
| **README.md** | 项目主文档（快速开始、安装、基本使用） |
| **FILE_INDEX.md** | 本文档：文件功能索引 |
| **MANUAL_DATA_USAGE.md** | 手动驾驶数据采集与使用详细指南 |

---

## 工作流程参考

### 正常智能体训练流程
1. `record_manual_control.py` → 采集手动数据（可选）
2. `pretrain_with_manual_data.py` → 预训练（可选）
3. `train_no_rllib_mappo.py` 或 `train_rllib.py` → 正式训练
4. `evaluate_no_rllib_mappo.py` → 评估

### 对抗智能体训练流程
1. （前置）完成正常智能体训练，得到checkpoint
2. `train_adversarial_ppo.py --pretrained-ckpt <path>` → 对抗训练
3. `evaluate_adversarial_ppo.py` → 标准场景评估
4. `evaluate_with_speedup.py` → 提速场景评估（鲁棒性测试）

### 文件依赖关系
```
train_adversarial_ppo.py
├── carla_multi_agent_env.py
└── visualize_adversarial_training.py (可选)

evaluate_with_speedup.py
├── carla_multi_agent_env.py
└── speedup_env_wrapper.py

pretrain_with_manual_data.py
├── manual_data_loader.py
└── carla_rllib_wrapper.py (可选)
```

