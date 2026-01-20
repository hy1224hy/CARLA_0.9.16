# MAPPO训练系统 - CARLA多智能体自动驾驶

本系统实现了基于MAPPO（Multi-Agent Proximal Policy Optimization）算法的多智能体强化学习训练，用于在CARLA仿真环境中训练两个自动驾驶车辆在Town02直道上进行直行训练。

## 特性

- **CTDE架构**: 集中式训练（Centralized Training），分布式执行（Decentralized Execution）
- **MAPPO算法**: 使用PPO算法进行多智能体训练
- **Town02直道训练**: 自动在Town02地图中找到合适的直道进行训练
- **双智能体系统**: 支持同时训练两个自动驾驶车辆

## 系统架构

### 文件结构

```
mappo_training/
├── carla_multi_agent_env.py    # CARLA多智能体环境包装器
├── mappo_networks.py            # MAPPO神经网络（Actor和Critic）
├── mappo_buffer.py              # 经验缓冲区
├── mappo_trainer.py             # MAPPO训练器
├── train_mappo.py               # 主训练脚本
├── requirements.txt             # 依赖包
└── README.md                    # 说明文档
```

### 核心组件

1. **CarlaMultiAgentEnv**: CARLA环境包装器
   - 自动在Town02中查找直道
   - 管理多个车辆智能体
   - 提供观察、奖励和完成信号

2. **MAPPOAgent**: MAPPO智能体
   - Actor网络：输出动作分布
   - Critic网络：评估全局状态价值（集中式训练）

3. **MAPPOBuffer**: 经验缓冲区
   - 存储训练经验
   - 计算GAE（Generalized Advantage Estimation）

4. **MAPPOTrainer**: 训练器
   - 实现PPO更新逻辑
   - 支持多智能体同时训练

## 安装

### 1. 确保CARLA服务器运行

首先启动CARLA服务器：
```bash
# Windows
CarlaUE4.exe

# Linux
./CarlaUE4.sh
```

### 2. 安装依赖

```bash
cd PythonAPI/examples/mappo_training
pip install -r requirements.txt
```

### 3. 安装CARLA Python API

确保CARLA Python API已正确安装：
```bash
cd PythonAPI/carla
pip install -e .
```

## 使用方法

### 基本训练（带可视化）

```bash
cd PythonAPI/examples/mappo_training
python train_mappo.py
```

训练过程中会自动打开一个pygame窗口，实时显示：
- **每个智能体的观测信息**（位置、速度、控制输入等）
- **奖励信息**（当前步奖励、episode总奖励、平均奖励）
- **训练进度**（episode和step进度条）
- **位置地图**（车辆和目标位置的可视化）
- **摄像头视角**（每辆车的实时视觉图像，类似manual_control.py）

### 禁用可视化

如果不想显示可视化窗口（可以提高训练速度）：

```bash
python train_mappo.py --no-visualize
```

### 高级参数

```bash
python train_mappo.py \
    --num-episodes 2000 \
    --max-steps 1000 \
    --update-frequency 100 \
    --save-frequency 100 \
    --save-dir ./models \
    --device cuda \
    --host 127.0.0.1 \
    --port 2000
```

### 参数说明

- `--num-episodes`: 训练episode数量（默认: 1000）
- `--max-steps`: 每个episode最大步数（默认: 1000）
- `--update-frequency`: 更新频率，每N步更新一次（默认: 100）
- `--save-frequency`: 保存频率，每N个episode保存一次（默认: 100）
- `--save-dir`: 模型保存目录（默认: ./mappo_models）
- `--device`: 设备，'cpu'或'cuda'（默认: cpu）
- `--host`: CARLA服务器地址（默认: 127.0.0.1）
- `--port`: CARLA服务器端口（默认: 2000）
- `--no-visualize`: 禁用可视化窗口（默认: 启用可视化）
- `--log-dir`: TensorBoard日志保存目录（默认: ./logs/mappo_training）

## 训练过程

1. **环境初始化**: 系统会自动连接到CARLA服务器，加载Town02地图，并查找合适的直道
2. **车辆生成**: 在直道上生成两个车辆，设置起始位置和目标位置
3. **可视化窗口**: 如果启用可视化，会打开一个pygame窗口显示训练过程
4. **训练循环**: 
   - 每个智能体根据当前观察选择动作
   - 执行动作并收集经验
   - 实时更新可视化窗口（如果启用）
   - 记录训练指标到TensorBoard
   - 定期使用PPO算法更新策略
5. **模型保存**: 定期保存训练好的模型
6. **日志记录**: 所有训练指标自动保存到TensorBoard日志文件

## 可视化功能

训练过程中的pygame可视化窗口包含以下信息：

### 观测信息面板
- **相对位置**: 车辆相对于目标的位置（X, Y）
- **到目标距离**: 当前到目标的距离
- **角度差**: 车辆朝向与目标方向的夹角
- **速度**: 当前速度（m/s）
- **控制输入**: 油门、转向、刹车的实时值和可视化进度条

### 奖励信息面板
- **当前步奖励**: 当前时间步获得的奖励
- **Episode总奖励**: 当前episode累计的总奖励
- **最近10步平均**: 最近10步的平均奖励

### 训练进度
- **Episode进度**: 当前episode和总episode数，以及进度条
- **Step进度**: 当前step和最大step数，以及进度条

### 位置地图
- **车辆位置**: 用不同颜色的圆点表示每个智能体
- **目标位置**: 用黄色圆点表示目标位置
- **连线**: 显示车辆到目标的连线
- **自动缩放**: 地图会自动缩放以显示所有车辆和目标

### 摄像头视角
- **实时图像**: 显示每辆车的第一人称视角图像（640x480分辨率）
- **多视角支持**: 同时显示所有智能体的摄像头画面
- **彩色边框**: 每个智能体的摄像头图像用不同颜色的边框区分
- **自动更新**: 摄像头图像实时更新，与训练同步

### 操作说明
- 按 **ESC** 键可以关闭可视化窗口（训练会继续进行）
- 关闭可视化窗口不会影响训练过程

## TensorBoard日志记录

训练过程中会自动记录各种指标到TensorBoard，方便分析和监控训练过程。

### 记录的指标

#### 每个智能体的指标

**奖励指标**:
- `Agent_X/Step_Reward`: 每步的即时奖励
- `Agent_X/Episode_Reward`: 每个episode的总奖励
- `Agent_X/Episode_Reward_Avg_10`: 最近10个episode的平均奖励
- `Agent_X/Episode_Reward_Avg_100`: 最近100个episode的平均奖励

**观测指标**:
- `Agent_X/Obs/Distance_To_Target`: 到目标的距离
- `Agent_X/Obs/Speed`: 当前速度
- `Agent_X/Obs/Angle_Diff`: 角度差（相对于目标方向）

**动作指标**:
- `Agent_X/Action/Throttle`: 油门值
- `Agent_X/Action/Steer`: 转向值
- `Agent_X/Action/Brake`: 刹车值

**价值估计**:
- `Agent_X/Value_Estimate`: Critic网络的价值估计

**损失指标**:
- `Agent_X/Loss/Actor_Loss`: Actor网络的损失
- `Agent_X/Loss/Critic_Loss`: Critic网络的损失
- `Agent_X/Loss/Entropy`: 策略熵（探索性）

**成功率**:
- `Agent_X/Success`: 是否成功到达目标（1.0表示成功，0.0表示失败）

#### 全局指标

- `Episode/Length`: 每个episode的长度（步数）
- `Episode/Length_Avg_10`: 最近10个episode的平均长度
- `Episode/Success_Rate`: 总体成功率

### 查看TensorBoard

训练开始后，可以使用以下命令启动TensorBoard：

```bash
tensorboard --logdir ./logs/mappo_training
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

### 日志目录结构

```
logs/
└── mappo_training/
    └── events.out.tfevents.xxxxx  # TensorBoard事件文件
```

### 使用建议

1. **实时监控**: 在训练过程中保持TensorBoard运行，实时查看训练进度
2. **对比实验**: 使用不同的`--log-dir`参数来保存不同实验的日志，方便对比
3. **分析训练**: 通过查看损失曲线和奖励曲线，判断训练是否正常
4. **调参参考**: 根据TensorBoard中的指标来调整超参数

### 示例：对比不同超参数的训练

```bash
# 实验1：默认参数
python train_mappo.py --log-dir ./logs/exp1_default

# 实验2：不同的学习率
python train_mappo.py --log-dir ./logs/exp2_lr_1e4 --learning-rate 1e-4

# 在TensorBoard中对比
tensorboard --logdir ./logs
```

## 观察空间

每个智能体的观察包括：
- 相对目标位置 (dx, dy)
- 到目标的距离
- 角度差（相对于目标方向）
- 当前速度
- 当前控制输入（throttle, steer, brake）

维度: 8

## 动作空间

每个智能体的动作包括：
- throttle: [0, 1] 油门
- steer: [-1, 1] 转向
- brake: [0, 1] 刹车

维度: 3

## 奖励设计

奖励函数包括：
- 距离奖励：越接近目标奖励越高
- 速度奖励：保持合理速度（5-15 m/s）
- 方向奖励：朝向目标方向
- 到达奖励：到达目标附近获得额外奖励

## CTDE架构说明

### 集中式训练（Centralized Training）
- Critic网络使用全局状态（所有智能体的状态拼接）进行价值估计
- 训练时可以使用全局信息，提高学习效率

### 分布式执行（Decentralized Execution）
- Actor网络只使用局部观察（单个智能体的观察）
- 执行时不需要全局信息，适合实际应用

## 模型保存

训练过程中，模型会保存在指定目录：
```
mappo_models/
├── agent_0_episode_100.pth
├── agent_1_episode_100.pth
├── agent_0_episode_200.pth
├── agent_1_episode_200.pth
...
├── agent_0_final.pth
└── agent_1_final.pth
```

## 注意事项

1. **CARLA服务器**: 确保CARLA服务器在训练前已启动
2. **地图加载**: 系统会自动加载Town02地图，如果地图不存在会报错
3. **性能**: 如果使用CPU训练，速度可能较慢，建议使用GPU（CUDA）
4. **内存**: 长时间训练可能占用较多内存，注意监控

## 故障排除

### 连接失败
- 检查CARLA服务器是否运行
- 检查host和port参数是否正确

### 车辆生成失败
- 检查Town02地图是否正确加载
- 尝试调整车辆生成位置

### 训练不稳定
- 调整学习率
- 调整PPO裁剪参数
- 调整奖励函数权重

## 扩展

可以进一步扩展的功能：
- 支持更多智能体
- 添加更多传感器（摄像头、LiDAR等）
- 实现更复杂的奖励函数
- 支持其他地图
- 添加评估脚本

## 参考文献

- MAPPO: Multi-Agent Proximal Policy Optimization
- PPO: Proximal Policy Optimization Algorithms
- CARLA: An Open Urban Driving Simulator

