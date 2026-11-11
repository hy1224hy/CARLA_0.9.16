# L1慢速驾驶风格v4

独立协议highway_l1_style_twin_v4，只训练L1；L2/L3保持原定版源码。53维观测，网络仍输出连续加速度及离散换道，低层车道跟踪不变，红车仍使用IDM/MOBIL。

统一奖励分项：style（风格）、geometry（势函数进展）、impact（交通影响）、dense/lane_change（控制与换道代价）、failure（约束失败）、success（终局达成）。没有正的纯换道事件奖励。

1. 准备：蓝车位于红车前方、同车道对齐、稳定0.3秒、净间距至少为红车1.5秒行驶距离，才进入慢行阶段。后方通过势函数获得接近引导，不能在后方直接领取慢行奖励。
2. 阶段锁存：进入后记录本车道；红车绕行不改变蓝车慢行目标，不再引导追堵。
3. 慢行目标：参考车速度的85%；高斯速度跟踪×车道保持质量为主收益，权重2。低于3m/s或参考低于8m/s无风格收益，停车另有代价。
4. 影响辅助：仅蓝车前方且红车非明显急刹时，速度损失项贡献，10%后饱和，权重0.3；不重复奖励累计进度损失。
5. 成本：动作变化、停车；最多2次换道、完成后3秒冷却；准备期每次-0.05，慢行期每次-0.25。碰撞/无效-20，危险切入/持续逼停越级-10。
6. 风格成功：安全跑满20秒、慢行阶段至少5秒、至少70%的阶段时间速度比在0.77～0.93之间、至少85%的阶段时间位于锁存车道中心0.5米且航向偏差10度内，终局+3。
7. 影响达成：前方非急刹速度影响≥5%累计0.5秒，或进度损失≥5米且曾处于前方0.5秒。联合成功=风格成功且影响达成，额外+1。

style_success_rate与attack_success_rate分开报告，不能把只会慢行计为对抗成功。阈值为开发候选；单seed实验不证明互学习鲁棒性。geometry使用有界势函数差分，终止势函数归零。

从PythonAPI/examples运行：

```powershell
& ../../venv/Scripts/python.exe -m highway_l1_style_v4 train --config highway_l1_style_v4/configs/L1_mutual_s42_500k.json --output highwaytest/runs/l1_style_v4_s42_500k_20261111_v1/train/L1_mutual_s42
```

每组500k是A+B总决策步，PPO基线也使用两个独立副本、相同采样预算和选模。新模型随机初始化，不加载旧网络。观测中包含风格阶段、锁存车道和累计计时。TensorBoard优先观察episode/A或B下style_active_rate、style_success_rate、attack_success_rate、style_speed_fraction、style_lane_fraction、collision_rate及grade_violation_rate。
