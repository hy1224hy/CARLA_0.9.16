# L1/L2行为分化第二版

独立于`highway_hybrid`旧矩阵，协议`highway_behavior_twin_v2`。网络从随机初始化开始，原双副本互学习保持不变：相同初始参数、独立采样/优化器、冻结同伴的连续与离散KL。总步数为A+B合计，非每副本。

L1最多发起2次换道，每次完成后冷却3秒。稳定同车道前方0.6秒后，温和速度损失获得收益；每次换道代价0.05。蓝车在后方不会获得影响项收益。目标车仍可按原IDM/MOBIL绕行，成功不要求封堵整个回合。

L2切入在跨入目标车道且处于目标前方时锚定，入场THW为0.3至3秒；1秒内需新增速度下降至少1m/s或新增制动至少1.5m/s²持续0.2秒并完成对齐，或目标新增换道而参考车未同期换道。对比入场前状态及参考速度变化，扣除原有制动的线性延续。这里是时序归因代理，不是严格反事实因果证明。0.3至1.2秒另记危险切入；必须稳定退出至邻道0.5秒才可重置事件资格。一次有效事件4分，完整回合成功5分，碰撞-20。

L2删除持续速度损失收益；前向时距引导仅在首次切入前短暂提供，低速停车有持续代价。两个等级使用有界势函数差分进行接近引导，终止势函数归零，不对纯换道行为发正奖励。L1温和偏好保留，越级或碰撞终止。观测含剩余预算、冷却、待确认切入的基线和重置状态。

从`PythonAPI/examples`运行：

```powershell
& ../../venv/Scripts/python.exe -m highway_behavior_v2 train --config highway_behavior_v2/configs/pilot/L1_mutual_s42.json --output highwaytest/runs/behavior_v2_pilot_20261111_v1/train/L1_mutual_s42
& ../../venv/Scripts/python.exe -m highway_behavior_v2 train --config highway_behavior_v2/configs/pilot/L2_mutual_s42.json --output highwaytest/runs/behavior_v2_pilot_20261111_v1/train/L2_mutual_s42
& ../../venv/Scripts/python.exe -m highway_behavior_v2 evaluate --runs highwaytest/runs/behavior_v2_pilot_20261111_v1/train --checkpoint latest --team A --cases 40 --output highwaytest/runs/behavior_v2_pilot_20261111_v1/audit_A
```

同一训练程序不会覆盖现有输出；续跑使用同一配置并指定本目录`latest.pt`。审计输出目录必须不存在。新旧事件定义不同，成功率不直接横向替换。TensorBoard以试跑根目录`train`为logdir。
