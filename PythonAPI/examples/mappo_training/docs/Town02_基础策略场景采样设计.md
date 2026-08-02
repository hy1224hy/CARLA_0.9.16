# Town02 基础策略场景采样设计

## 结论

Town02 v2 场景池已重构为 **16 个配置、4 个累积课程阶段、4 个行为场景族**。场景不再主要依赖坐标镜像制造数量，而是按动力学难度递进：

```text
静止长距起步
  → 中速滚动 + 横向/yaw 扰动
  → 同速滚动近距跟驰
  → 后车以速度差逼近前车
```

每个阶段都覆盖东、西、南、北四个方向；每阶段 Agent 0/1 在后各 2 次，完整池角色比例为 8:8。

## 重构前备份

- 重构前 Git 提交：`34c355a`（`feat: harden MAPPO baseline and add Town02 curriculum`）。
- 可恢复备份：`logs/backups/20260802_before_scenario_redesign/`。
- 备份包含 HEAD 基线压缩包、tracked patch 和全部 untracked 文件。

旧的 12 场景设计已经保存在提交和备份中，不需要在新目录中保留重复 JSON。

## 场景协议 v3

默认目录：`data/training_scenarios_town02_v2/`。

场景 JSON 的核心字段：

```json
{
  "schema_version": 3,
  "town": "Town02",
  "difficulty": 2,
  "family": "offset_rolling_medium_gap",
  "tags": ["eastbound", "rolling_start", "agent1_rear"],
  "goal_zone": {},
  "agents": {
    "agent0": {
      "x": 100.0,
      "y": 108.95,
      "z": 0.5,
      "yaw": -4.0,
      "initial_speed_mps": 3.0
    },
    "agent1": {}
  }
}
```

`initial_speed_mps` 只在 reset 后设置车辆初始速度。episode 内仍完全执行策略动作，不覆盖学习车的油门、转向或制动，保证 PPO rollout 中的动作与 log-prob 一致。

## 四阶段课程

| difficulty | 累积场景数 | 新增 family | 初始状态 | 训练目标 |
|---:|---:|---|---|---|
| 1 | 4 | `basic_launch_long_gap` | 静止、长车距、居中 | 学会起步、直行、到达和基础角色解耦 |
| 2 | 8 | `offset_rolling_medium_gap` | 双车 3 m/s、中车距、±0.5m/±4° 扰动 | 学会滚动接管和姿态恢复 |
| 3 | 12 | `rolling_close_following` | 双车 5 m/s、14m 近距 | 学会高能量状态下的稳定跟驰 |
| 4 | 16 | `rear_closing_pressure` | 后车 5 m/s、前车 3 m/s、24m 间距 | 学会识别速度差并安全减速 |

各阶段内部结构：

| 维度 | 分布 |
|---|---|
| 方向 | 东/西/南/北各 1 个 |
| 后车角色 | Agent 0/1 各 2 个 |
| 路口 | 南北向直行穿越 Town02 路口，东西向覆盖横向道路 |
| family | 每阶段只新增一种能力机制，便于课程归因 |

## 为什么没有脚本急刹

直接在 `env.step()` 中覆盖某个学习 agent 的制动，会造成：

```text
rollout 保存：策略采样动作 a
环境实际执行：脚本覆盖动作 a'
PPO 更新使用：log π(a|s)
```

这会破坏 on-policy PPO 的动作—概率一致性。因此，本轮只增加初始速度分布。未来若要加入急刹或移动阻塞，应使用独立 NPC，或让环境把实际执行动作和接管 mask 显式返回并从该 agent 的 PPO 样本中剔除。

## 族级覆盖 + 失败率采样

默认仍为每 update 采集 16 个 episode，但不再强制每个 JSON 每批出现一次。

weighted 模式的顺序为：

1. 课程权重达到 0.5 的每个 `family` 至少选择一个场景；
2. family 内优先选择历史采样次数最少的成员，保证方向和角色轮换；
3. 剩余 episode 按 `curriculum_weight × (0.25 + posterior_failure)` 分配；
4. 单场景仍受批内最大次数约束，避免不可解场景吞掉整批。

完整课程阶段只有 4 个 family，因此 16 个 episode 中至少保留约 12 个自适应配额，解决了“16 场景 + 16 episode 时逐场景强制覆盖耗尽失败率加权预算”的问题。

未观测场景使用 Beta(2,2) 中性先验：

\[
p_{success}=\frac{successes+2}{count+4}
\]

困难度因子：

\[
w_{scenario}=w_{curriculum}(0.25+1-p_{success})
\]

## 地图与可解性约束

- 所有出生点和目标均使用 Town02 OpenDRIVE 离线校验；
- 横向偏移不超过 0.75m，yaw 误差不超过 5°；
- 初始中心距不低于 8m；
- `initial_speed_mps` 必须位于 `[0, 15]`；
- 终点必须位于车辆前方，并沿同一可行驶车道拓扑可达；
- 四个方向、四个 family、四个 difficulty 均保持 4:4:4:4 平衡；
- Agent 0/1 后车角色保持 8:8。

## 使用与校验

从 `PythonAPI/examples/mappo_training/` 运行：

```powershell
python -m utils.validate_training_scenarios `
  --scenario-dir data/training_scenarios_town02_v2

python -m utils.smoke_training_scenarios_live `
  --host 127.0.0.1 `
  --port 2000 `
  --town Town02

python -m training.train_no_rllib_mappo_robust_v2 `
  --port 2000 `
  --town Town02 `
  --scenario-dir data/training_scenarios_town02_v2 `
  --curriculum-mode metadata `
  --scenario-sampling-mode weighted `
  --rollout-episodes 16
```

`CarlaMultiAgentEnv(..., town="Town02")` 会调用 CARLA
`client.load_world("Town02")`，因此即使 2000 端口当前运行的是其他
地图，上述 smoke 工具和训练脚本也会在初始化时将其切换到
Town02。地图加载会清理旧 world 中的 actor，不要在其他 client
正在使用同一 CARLA 实例时执行。

预期离线校验摘要：

```text
场景数=16
difficulty={1:4, 2:4, 3:4, 4:4}
family=四类且各4个
方向=东/西/南/北各4个
后车角色=agent0/agent1各8个
速度档=静止4、同速滚动8、后车逼近4
```

2026-08-02 在 `127.0.0.1:2000` 的实机验收结果：

- 执行前地图为 Town10HD_Opt，工具自动切换为 Town02；
- 16/16 个场景出生成功，执行后地图查询为 `Carla/Maps/Town02`；
- JSON 标称 x/y 与实际 reset 坐标误差为 0–0.01m；
- 3m/s 与 5m/s 初速实测约为 2.96m/s 与 4.89m/s，误差小于 0.25m/s。

为避免将 z=0.5m 出生后的竖直落体速度混入首帧观察，reset
会先用同步帧让车辆落地，再分帧提交预偏移 transform 和初始水平速度。
这只构造初始状态，不会在 rollout 中覆盖策略动作。

## 训练兼容性

场景文件集合、JSON 内容和采样协议均已改变。不要把旧场景池产生的 periodic checkpoint 直接 strict-resume 到新池；`scenario_pool_fingerprint` 和代码指纹会拒绝这种不一致。

允许的使用方式：

- 为新场景池启动 fresh run；
- 将旧 checkpoint 显式作为 pretrained 权重来源，但创建新的实验目录和运行元数据；
- 重构前结果继续使用提交 `34c355a` 和备份中的旧协议复现。
