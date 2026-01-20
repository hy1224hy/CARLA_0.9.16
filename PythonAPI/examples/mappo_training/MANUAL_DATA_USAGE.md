# 手动驾驶数据采集与使用指南

本指南说明如何使用手动驾驶数据采集功能，并将采集的数据用于MAPPO训练。

## 功能概述

通过手动驾驶采集高质量的专家驾驶数据，可以：
1. 作为预训练数据，帮助智能体更快学习基本驾驶行为
2. 提供专家示范，减少探索时间
3. 帮助智能体避免早期训练中的危险行为（碰撞、偏离等）

## 使用流程

### 步骤1: 采集手动驾驶数据

首先，需要为每个智能体分别采集手动驾驶数据。

#### 采集Agent 0的数据

```bash
cd PythonAPI/examples/mappo_training
python record_manual_control.py --agent-id 0 --output-dir ./manual_driving_data
```

#### 采集Agent 1的数据

```bash
python record_manual_control.py --agent-id 1 --output-dir ./manual_driving_data
```

#### 控制说明

- **W/↑**: 油门
- **S/↓**: 刹车
- **A/←**: 左转
- **D/→**: 右转
- **ESC**: 结束当前episode并保存
- **Q**: 退出程序

#### 采集建议

1. **多次采集**: 建议为每个智能体采集至少5-10个episodes的数据
2. **质量优先**: 尽量保持平稳驾驶，避免碰撞和车道偏离
3. **多样性**: 尝试不同的驾驶风格和速度，增加数据多样性

### 步骤2: 查看数据统计

采集完成后，可以查看数据统计信息：

```bash
python manual_data_loader.py --data-dir ./manual_driving_data
```

这将显示：
- 每个智能体的episode数量
- 总步数
- 平均奖励
- 其他统计信息

### 步骤3: 使用手动数据进行预训练

使用采集的手动驾驶数据进行预训练（行为克隆）：

```bash
python pretrain_with_manual_data.py \
    --manual-data-dir ./manual_driving_data \
    --output-dir ./pretrained_models \
    --pretrain-steps 1000 \
    --batch-size 128 \
    --learning-rate 1e-4
```

参数说明：
- `--manual-data-dir`: 手动驾驶数据目录
- `--output-dir`: 预训练模型保存目录
- `--pretrain-steps`: 预训练步数（默认1000）
- `--batch-size`: 批次大小（默认128）
- `--learning-rate`: 学习率（默认1e-4）

### 步骤4: 使用预训练模型进行训练

使用预训练模型开始正式训练：

```bash
python train_rllib.py \
    --pretrained-model ./pretrained_models/pretrained_policy \
    --train-iter 10000 \
    --num-workers 1
```

注意：由于RLlib的架构限制，预训练模型的加载需要在训练配置中手动设置。当前版本提供了预训练脚本，您可以根据需要调整训练流程。

## 数据格式

手动驾驶数据以JSON或NPZ格式保存：

### JSON格式

```json
{
    "agent_id": 0,
    "num_episodes": 5,
    "total_steps": 1234,
    "episodes": [
        {
            "observations": [[...], [...], ...],
            "actions": [[...], [...], ...],
            "rewards": [0.5, 0.3, ...],
            "dones": [false, false, ...],
            "next_observations": [[...], [...], ...]
        }
    ]
}
```

### 观察空间

每个观察是19维向量：
- 前8维：基础状态（位置、速度、控制输入等）
- 后11维：雷达感知信息

### 动作空间

每个动作是3维向量：
- `[throttle, steer, brake]`
- `throttle`: [0, 1] 油门
- `steer`: [-1, 1] 转向
- `brake`: [0, 1] 刹车

## 文件结构

```
mappo_training/
├── record_manual_control.py      # 手动驾驶数据采集脚本
├── manual_data_loader.py          # 数据加载器
├── pretrain_with_manual_data.py   # 预训练脚本
├── train_rllib.py                 # 训练脚本（已更新）
├── manual_driving_data/           # 数据保存目录
│   ├── agent_0_data.json
│   └── agent_1_data.json
└── pretrained_models/             # 预训练模型目录
    └── pretrained_policy/
```

## 注意事项

1. **CARLA服务器**: 确保CARLA服务器在采集数据前已启动
2. **数据质量**: 尽量采集高质量的驾驶数据，避免碰撞和车道偏离
3. **数据量**: 建议每个智能体至少采集1000步以上的数据
4. **一致性**: 确保手动驾驶时使用的观察和动作空间与训练环境一致

## 故障排除

### 问题1: 无法连接到CARLA服务器

- 检查CARLA服务器是否运行
- 检查host和port参数是否正确

### 问题2: 数据采集时车辆不响应

- 确保pygame窗口处于活动状态
- 检查键盘输入是否被其他程序占用

### 问题3: 预训练损失不下降

- 检查数据质量
- 尝试调整学习率
- 增加预训练步数

### 问题4: 预训练模型加载失败

- 检查模型路径是否正确
- 确保模型格式与训练配置兼容

## 高级用法

### 数据过滤

可以使用`ManualDataLoader`的`filter_low_quality_data`方法过滤低质量数据：

```python
from manual_data_loader import ManualDataLoader

loader = ManualDataLoader("./manual_driving_data")
data = loader.load_agent_data(0)

# 过滤低质量数据
filtered_data = loader.filter_low_quality_data(
    data,
    min_reward=-10.0,  # 最小奖励阈值
    filter_collisions=True,  # 过滤碰撞数据
    filter_lane_invasions=True  # 过滤车道偏离数据
)
```

## 预期效果

使用手动驾驶数据进行预训练后，预期可以：
- 减少训练时间：智能体可以更快学习基本驾驶行为
- 提高成功率：避免早期训练中的危险行为
- 加速收敛：从专家示范中学习，减少随机探索

## 参考

- [MAPPO训练README](README.md)
- [CARLA文档](https://carla.readthedocs.io/)
