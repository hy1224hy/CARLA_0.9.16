# 混合动作 Mut-PPO

策略网络输出连续加速度和离散车道决策，横向跟踪由 highway-env 的 `ControlledVehicle.steering_control` 执行。目标车仍为可跟车、可换道的 IDM/MOBIL。旧连续转向协议保留在 `highwaytest/straight/`。

## 动作与观测

- `acceleration ∈ [-1,1]`，物理映射 `a=5×acceleration` m/s²。
- `lane=0/1/2` 分别为保持、左换道、右换道。左为车道索引减1，右为加1。
- 网络决定何时换道和加减速；控制器不读取攻击等级，也不选择攻击行为。
- 边界外车道被掩码屏蔽。换道过程中只允许保持，纵向控制继续可用。
- 换道完成要求距目标中心小于0.35 m、航向小于8°并连续保持0.4 s。
- 观测是 `Dict(state, lane_mask)`：state 为28维（现有26维 + 目标车道横向误差 + 换道执行/回正进度）；最后一维空闲时为0，执行时为1加已连续回正时长比例，避免隐藏控制器计时状态。lane_mask 是3个合法性标志，采样和更新使用相同掩码。

## PPO 与互学习

Actor 共享特征网络后接一维 Gaussian 加速度头和三类 Categorical 车道头。加速度经 tanh 压缩，联合动作对数概率为带雅可比修正的连续项与类别项之和。PPO 对联合概率比进行一次裁剪。

每次实验维护同任务的 A/B 两副本，相同初始网络权重、独立轨迹/RNG/优化器。三个等级在同种子下共享初始化方式，由奖励分化。`independent` 和 `mutual` 的总预算及选模规则一致；independent 将实际互学习系数置零。

互学习对同一个状态和合法动作集合，分别计算 Gaussian KL 和 Categorical KL；对端教师快照冻结，默认每轮结束后刷新。两个 KL 均采用 student || peer。

\[
L=L_{PPO}+0.5L_V-c_aH_a-c_lH_l+
\beta(t)w_aD_{KL}^{acc}+\beta(t)w_lD_{KL}^{lane}.
\]

`beta` 是公共强度；`acceleration_kl_multiplier`、`lane_kl_multiplier` 分别缩放两项。默认 warmup/ramp 各20k A+B合计步数。日志分别记录两个实际系数、KL、熵及加权互学习损失。Gaussian 熵为 tanh 前熵。

## 奖励边界

三级事件、碰撞优先级、温和阻碍和有效切入判据复用 `highwaytest/straight/rewards.py`。默认保留任务密集项与事件/终局奖励，关闭此前为直接方向盘探索开发的 approach/guidance/geometry 过程项；这些参数仍可显式配置。

这是一项动作表示适配，不能与旧连续控制曲线直接拼接后归因于互学习或奖励。正式比较需在本协议内固定动作、初始化、奖励和总预算。控制器提供基础横向执行能力，因此不需要先克隆普通驾驶转向网络，也不接收旧连续动作检查点。

## 运行

从 `D:\study\yan2\CARLA_0.9.16\PythonAPI\examples` 运行：

```powershell
# 配置矩阵：3等级 × 2方法 × 3种子
& ..\..\venv\Scripts\python.exe -m highway_hybrid plan `
  --output highway_hybrid/configs/main_v1 --steps 200000 --seeds 42 43 44

# 单个实验；预算是两个副本合计
& ..\..\venv\Scripts\python.exe -m highway_hybrid train `
  --config highway_hybrid/configs/main_v1/L2_mutual_s42.json `
  --output highwaytest/runs/hybrid_v1/L2_mutual_s42

# 有界短程训练，结束后保留 latest.pt 和 selected.pt
& ..\..\venv\Scripts\python.exe -m highway_hybrid train `
  --config highway_hybrid/configs/main_v1/L2_independent_s42.json `
  --output highwaytest/runs/hybrid_v1/L2_independent_s42 --max-rounds 25

# 恢复时配置、代码来源和运行版本必须一致
& ..\..\venv\Scripts\python.exe -m highway_hybrid train `
  --config highway_hybrid/configs/main_v1/L2_independent_s42.json `
  --output highwaytest/runs/hybrid_v1/L2_independent_s42 `
  --resume highwaytest/runs/hybrid_v1/L2_independent_s42/latest.pt

# 不更新网络的随机策略覆盖诊断，不是学习效果
& ..\..\venv\Scripts\python.exe -m highway_hybrid probe `
  --level 2 --cases 12 --output highwaytest/runs/hybrid_probe_01

# 小预算端到端 smoke，使用4秒短回合，不能当作正式效果证据
& ..\..\venv\Scripts\python.exe -m highway_hybrid smoke `
  --level 2 --steps 512 --method mutual --output highwaytest/runs/hybrid_smoke_01

# 独立开发评估：模型先冻结，再使用固定场景
& ..\..\venv\Scripts\python.exe -m highway_hybrid evaluate `
  --runs highwaytest/runs/hybrid_v1 --checkpoint selected `
  --split development --cases 40 --output highwaytest/runs/hybrid_v1/audit_dev_01

# 同样支持统一评估入口，按检查点协议自动路由
& ..\..\venv\Scripts\python.exe -m highway_eval `
  --runs highwaytest/runs/hybrid_v1 --checkpoint latest --team B `
  --split development --cases 40 --output highwaytest/runs/hybrid_v1/audit_latest_B_01

# 模型冻结后最终评估；不反复用 test 调奖励
& ..\..\venv\Scripts\python.exe -m highway_hybrid evaluate `
  --runs highwaytest/runs/hybrid_v1 --split test --cases 100 `
  --output highwaytest/runs/hybrid_v1/audit_final_01
```

输出目录必须是新目录；恢复训练仅接收该运行自己的 latest.pt。未通过协议/来源/版本检查的模型拒绝加载。

## 日志与产物

每个训练运行包含 config、manifest、源码快照、metrics/episodes/validation JSONL、TensorBoard、latest.pt、selected.pt 和 summary.json。断点恢复保留两副本优化器、随机状态和冻结教师，并回滚未提交轮次的 JSONL。

TensorBoard 使用整个训练根目录，例如：

```powershell
& ..\..\venv\Scripts\python.exe -m tensorboard.main `
  --logdir highwaytest/runs/hybrid_v1 --host 127.0.0.1 --port 6010
```

优先看 validation 的攻击成功率、失败率、阻碍时间和有效切入次数；train 中看 `lane_decision_fraction`、分头熵、分头 KL、实际 beta、联合概率误差及裁剪比例。换道忙碌状态的类别熵与 KL 正确地为0，不代表策略塌缩。

独立评估保存固定 cases、冻结模型、逐回合 CSV/JSONL、完整轨迹及首次成功/未成功轨迹图，分别报告 follow/adjacent。零成功的检查点会标记为 `attack_capable=false`，不能当作有效策略池成员。

检查点协议为 `highway_hybrid_twin_v1`，与旧版 `straight_twin_v1` 分离。训练/selection/development/test 的场景 seed 范围分别为0、2,000,000、4,000,000、3,000,000起。独立评估输出的误差与边界时间诊断按策略频率采样，事件判据仍由物理子步跟踪器计算。

## 验证

```powershell
& ..\..\venv\Scripts\python.exe -m unittest discover -s highway_hybrid/tests -v
```

测试覆盖联合概率、合法掩码、忙碌状态零类别熵/KL、冻结教师及有限梯度、原生换道完成与反向执行、纵向连续控制、双副本恢复一致性、旧协议拒绝、冻结独立评估及统一入口路由。
