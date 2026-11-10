# Mut-PPO 单车双学习器框架图

本图依据 `highwaytest/straight` 当前的 PPO 实现绘制。A、B 各自包含一个 Actor、一个标量 Critic、一个 Adam 优化器和独立的 Highway 环境副本。每个环境只有一个可学习的控制车辆；目标车使用固定控制器。图中采用现有高斯 KL 互学习，不包含门控、通信或多车集中式 Critic。

## 文件

- `mut_ppo_dataflow.png`：可插入论文或幻灯片的预览图。
- `mut_ppo_dataflow.svg`：可编辑文字的矢量源图。
- `render_framework.py`：确定性绘图源代码，无训练或仿真调用。

## 图的读取方式

1. 采样时，在线观测输入本方 Actor 和 Critic。Actor 给出高斯分布，采样得到 pre-tanh 的 z，再由 a=tanh(z) 得到归一化控制动作。
2. 本方 rollout 保存观测、z、旧对数概率、奖励、采样时的价值与下一状态价值，以及真终止和回合边界标记。Actor/Critic 的采样记录写入行为已标在 buffer 内；采样和更新共用同一组可训练网络。
3. GAE 只用已存储的采样价值计算优势与回报目标。只对优势标准化，价值目标保持原始奖励量纲。真终止切断价值 bootstrap；回合边界切断优势递推。
4. 更新时，本方存储观测输入当前 Actor、当前 Critic 和对方冻结 Actor。当前策略在存储 z 上重算动作对数概率，与已保存旧概率组成 PPO ratio。
5. 基础 PPO 损失为裁剪策略损失 + 0.5×价值 MSE − 熵系数×变换前高斯熵。
6. 同一观测输入下的当前高斯分布与对方冻结高斯分布计算 KL(student || frozen peer)。对动作维度求和，再对本方有效样本平均。KL 不经过 Critic。
7. 总损失为基础 PPO 损失 + β(n)×互学习损失。一个 Adam 更新本方 Actor/Critic；互学习项只对本方 Actor 参数产生梯度。
8. A、B 完成本轮更新后，每 F 轮统一复制教师快照。教师使用 deepcopy、requires_grad_(False)，且预测位于 no_grad 中。当前模型优化器不持有教师参数。

黑实线表示数据或损失输入，青色实线表示策略分布进入采样及后续动作执行，蓝虚线表示教师快照复制，红点线表示总损失对本方可训练网络组的优化。黑线与青线的无圆点交叉不表示连接。

## 程序依据

- `straight/model.py`：Actor 高斯分布、独立标量 Critic、tanh 动作及其对数概率。
- `straight/learning.py:18`：采样与 buffer 内容。
- `straight/learning.py:36`：同输入教师预测、GAE、PPO、MSE、熵、高斯 KL 及总损失。
- `straight/runner.py:153`：独立模型/优化器/随机流及冻结教师初始化。
- `straight/runner.py:235`：独立环境、A/B 依次采样更新、按周期同步刷新教师。

重绘：在仓库根目录使用本地 venv 执行本目录的 `render_framework.py`。仅修改图文件，不修改训练实现。
