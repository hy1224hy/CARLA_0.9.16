# 迁移算法设计与性能结论，而非直接搬运权重

## 证据层级

1. **实现可复用**：相同PPO/GAE、角色独立网络、KL方向、教师快照/门控、消息协议和配对指标能够在两种后端运行。公共模块与模拟接口测试支持这一层。
2. **Highway内有效**：多seed主表、普通互学习/无互学习对照、通信2×2及机制控制支持某个方法在Highway任务上的收益。必须由真实训练结果给出，不能由代码测试或单seed选择集推出。
3. **性能结论在CARLA复现**：冻结在Highway确定的方法，再在CARLA独立训练和比较，观察收益方向、稳定性、安全性是否仍成立。这层尚未完成，不能用“统一接口”替代。

迁移成功并不要求两环境有相同的回报数值、成功率或策略权重，也不是Sim2Real。需要的是在各自公平基线下，核心改进的效果能重复出现。

## 共享与允许变化

固定共享：25维路线相对观测语义、29维critic状态、两层128隐藏层、8维消息、连续高斯策略、同角色KL、门控定义、PPO核心、C0/C1/C2配对、资格与所有案例分母、选模与最终测试隔离。

后端负责：场景生成、合法车道查询、碰撞观测、路线进度、动作执行和物理推进。允许根据CARLA场地的合法速度/距离重新标定任务，但所有CARLA算法组使用同一协议；物理参数变化必须在该后端最终测试前冻结，不用Highway的原始return直接跨环境排名。

建议CARLA先复验G0/G1/G2三个算法、各3seed；若通信是论文主张，再补G3/G4及相同消息干预。每个后端分别报告均值/标准差、相对基线差值与学习曲线，不把两种仿真器的episode当同一种独立重复。

若Highway有效而CARLA无效，应分析感知分布、动作闭环误差、路线拓扑、红车能力和安全资格差异；不能只选CARLA中好看的演示视频声称性能迁移。

## 已实现的接入路径

```
HighwayConditionBackend / CarlaConditionBackend
                  ↓ SceneFrame / VehicleFrame，SI单位
common.contract.inputs → 同一个Team网络
                  ↓
同一个PPO/GAE与互学习runner
common.objective + common.outcomes → 同一奖励与配对指标
```

- `common/contract.py`：局部旋转/平移不变特征、身份对应与物理动作定义。
- `common/objective.py`、`outcomes.py`：共享的有限时间目标、安全资格与额外攻击效应。
- `adapters/carla.py`：CARLA位置/速度/yaw/路线/车道/碰撞到公共快照；加速度反馈转换；单步同步后端。
- `adapters/environment.py`：先串行获取C0/C1参考，再加载C2，支持同一个CARLA服务。
- `adapters/train_carla.py`：复用现有训练runner，注入场景和控制器，A/B采样结束即释放世界，避免两个环境同时控制同一world。

## CARLA真实联调仍需完成的工作

没有启动或操作当前CARLA服务。已用假world/actor跑通共享训练器，验证身份、时钟、释放顺序和数据接口；这不是物理仿真或控制性能验证。

Town05场景拥有者需提供以下冻结对象：

```python
# 由现有场景管理器实现，不能同时保留旧训练循环在tick。
loader(condition, family, seed) -> (actors_by_physical_id, frame_adapter, cleanup)
traffic_control(identity, scene_frame) -> carla.VehicleControl  # 冻结红车/C1蓝车
acceleration_controls = {2: CarlaAccelerationControl(front_wheel_max_angle_rad),
                         3: CarlaAccelerationControl(front_wheel_max_angle_rad)}
```

loader必须做到C0仅加载红0/1、C1/C2加载0..3、各条件红车初始状态一致且C1/C2蓝车初始状态一致；坐标与初速会被评估器再次核对。路线使用冻结的不自交短折线/直路；复杂路口和闭环路线需要更完整的路线跟踪器，当前最近线段投影不保证解决所有拓扑歧义。

`CarlaFrameAdapter`需要Driving-lane地图和覆盖全部活动车辆的、按物理帧同步并持续锁存的碰撞flags回调。不能用缺失传感器默认为无碰撞。调用者必须验证传感器同步，尤其最后一帧事件。

CARLA固定dt必须与task.simulation_frequency一致且整除策略周期。例如物理0.1s、策略5Hz时配置simulation_frequency=10，每个动作执行两个物理tick。首版不自动启动服务器、切换地图、猜测出生点或复用旧20维actor权重。

先在固定场景标定加速度PI和最大转角映射，并校准红车/C1正常通行。若沿用Town05低速任务，不可照搬Highway的18m/s、180m/20s；应设置可完成的合法速度、距离和时窗，重新校准所有组。

在Config中显式指定backend="carla"，并记录backend_contract里的scene_manifest_sha256、red_controller_sha256、normal_controller_sha256。然后调用：

```python
from highwaytest.adapters.train_carla import train_carla
train_carla(config, output, world=world, loader=loader,
            traffic_control=traffic_control,
            acceleration_controls=acceleration_controls,
            control_factory=carla.VehicleControl)
```

训练与配对评估函数使用相同后端工厂。普通Highway CLI故意拒绝把CARLA配置静默当作Highway运行。真实传感器同步、路线/动作标定、Town05在线回合与跨seed性能复验完成后，才有资格声称性能结论迁移。
