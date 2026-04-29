
# 运行指令参考

本文档总结项目中所有脚本的运行方式和主要参数。

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

## 训练脚本

### 1. train_no_rllib_mappo.py - 独立 MAPPO 训练

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

### 2. train_adversarial_ppo.py - 对抗智能体训练

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

### 3. train_rllib.py - RLlib MAPPO 训练

**功能：** 使用 Ray RLlib 框架进行 MAPPO 训练

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

### 1. evaluate_no_rllib_mappo.py - 评估 MAPPO 模型

**功能：** 加载 checkpoint 并评估正常智能体的表现

**运行方式：**
```bash
python -m evaluation.evaluate_no_rllib_mappo --checkpoint <path>
```

---

### 2. evaluate_adversarial_ppo.py - 评估对抗模型

**功能：** 评估训练好的对抗智能体

**运行方式：**
```bash
python -m evaluation.evaluate_adversarial_ppo --checkpoint <path>
```

---

### 3. evaluate_with_speedup.py - 加速场景评估

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

### 完整流程 1：正常智能体训练

```bash
# 1. 启动 CARLA
# (在另一个终端)
CarlaUE4.exe

# 2. (可选) 采集手动数据
python -m utils.record_manual_control --agent-id 0 --output-dir ./data/manual_driving_data
python -m utils.record_manual_control --agent-id 1 --output-dir ./data/manual_driving_data

# 3. (可选) 预训练
python -m training.pretrain_with_manual_data \
    --manual-data-dir ./data/manual_driving_data \
    --output-dir ./pretrained_models

# 4. 训练
python -m training.train_no_rllib_mappo --render

# 5. 评估
python -m evaluation.evaluate_no_rllib_mappo --checkpoint ./logs/minimal_mappo_ckpts/ckpt_50.pt
```

### 完整流程 2：对抗智能体训练

```bash
# 1. 启动 CARLA
CarlaUE4.exe

# 2. (前置) 已有正常智能体的 checkpoint

# 3. 对抗训练（带课程学习）
python -m training.train_adversarial_ppo \
    --pretrained-ckpt ./logs/minimal_mappo_ckpts/ckpt_100.pt \
    --curriculum-learning \
    --warmup-updates 100 \
    --ramp-up-updates 200

# 4. 或用启动器同时开启可视化
python -m training.run_adversarial_training_with_viz \
    --pretrained-ckpt ./logs/minimal_mappo_ckpts/ckpt_100.pt

# 5. 评估
python -m evaluation.evaluate_adversarial_ppo --checkpoint ./logs/adversarial_ppo_ckpts/adversarial_ppo_update_100.pt

# 6. 加速场景测试
python -m evaluation.evaluate_with_speedup --checkpoint ./logs/adversarial_ppo_ckpts/adversarial_ppo_update_100.pt
```

---

## TensorBoard 监控

所有训练脚本都会自动记录 TensorBoard 日志：

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 或指定特定子目录
tensorboard --logdir logs/adversarial_training
```

然后在浏览器访问: `http://localhost:6006`

---

## 常见问题

**Q: 提示找不到模块？**
```
A: 确保从根目录以模块方式运行：
python -m training.train_no_rllib_mappo
而非：python training/train_no_rllib_mappo.py
```

**Q: 如何从 checkpoint 继续训练？**
```
A: 当前脚本需要手动修改代码来加载。
建议使用小的 --save-interval 以便频繁保存。
```

**Q: 显存不足？**
```
A: 尝试减小 batch size：
--minibatch-size 64
或使用 CPU：
--cpu
```
