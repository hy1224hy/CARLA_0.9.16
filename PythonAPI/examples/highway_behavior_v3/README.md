# 行为分化v3：较易触发的L1与后方准备切入的L2

独立协议`highway_behavior_twin_v3`，不修改v2或旧hybrid实验；新的45维观测，随机网络初始化，双副本Mut-PPO机制不变。目标车仍使用原IDM/MOBIL。

L1：稳定前方0.3秒后才有影响收益，速度损失5%以上计合格时间，10%起奖励饱和，取消35%上限及对应越级终止。最多2次换道、完成后3秒冷却、发起扣0.05。全20秒累计1秒影响且无碰撞/无效/危险切入/持续逼停即成功；另记前10秒累计1秒的early_success，不提前结束回合。持续停车和危险切入约束保留。接近引导对后方同车道状态先偏好邻道，前方速度目标调整为参考速度90%。

L2：邻道稳定且前向时距0.7～2秒时进入准备完成状态；只有退到0.4秒以下、超过2.5秒或未重新取得资格时才退出，避免阶段抖动。准备完成后向目标车道发起换道时记录速度/制动/参考基线；实际跨入前方且THW在0.3～3秒时才建立事件，随后1秒确认新增制动/速度损失或参考未同步发生的新增避让。一份发起基线只用一次。没有合格准备基线的纯换道不计有效事件；后方无正影响收益。退出邻道0.5秒后重置资格。归因仍为时序代理，不是严格因果证明。

L2的准备状态、发起基线和事件状态均进入观测；不通过规则控制网络动作。新增initial_behind_asr/initial_ahead_asr按蓝车初始中心纵向位置划分，后方样本数同步记录。

## 本轮训练

两级各400k合计决策步（每副本约200k），seed42，beta=.02，warmup/ramp各20k，selection每20k步32场景，输出`highwaytest/runs/behavior_v3_400k_20261111_v1/`。新奖励从头训练，不把v2检查点混入。

从PythonAPI/examples运行：

```powershell
& ../../venv/Scripts/python.exe -m highway_behavior_v3 train --config highway_behavior_v3/configs/400k/L1_mutual_s42.json --output highwaytest/runs/behavior_v3_400k_20261111_v1/train/L1_mutual_s42
& ../../venv/Scripts/python.exe -m highway_behavior_v3 train --config highway_behavior_v3/configs/400k/L2_mutual_s42.json --output highwaytest/runs/behavior_v3_400k_20261111_v1/train/L2_mutual_s42
```

已存在的训练目录不可覆盖。续跑指定同一目录的latest.pt；独立评估使用evaluate子命令，latest的A/B分别评估，selected单独报告。
