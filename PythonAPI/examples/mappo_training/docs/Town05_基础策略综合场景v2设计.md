# Town05 基础策略综合场景 v2 设计

> 状态：设计与实现完成；32/32 离线/真实初始化、offroad 四方向正反例、fresh 1-update 严格链路和 210/210 全量回归均通过。Town05 v1/v2 证据只读保留；当前正式 fresh 身份升级为 training/C0/GO-HOLD v3，用结果级重复性取代逐 tick 硬相同。CARLA 当前关闭，待本阶段提交后再启动 seed-42 u5。

## 1. 结论

当前 `training.train_no_rllib_mappo_robust_v2` 是课程学习。默认 `metadata` 模式读取每个 JSON 的 `difficulty`，构建**累计场景集合**：进入更高难度后，早期场景仍会继续采样。`weighted` 模式还会根据最近失败率提高困难场景的采样优先级，并保证各 `family` 长期轮换覆盖。

训练器没有把场景总数写死为 16。它会读取 `scenario_dir` 中全部 JSON；原来的 16 只属于冻结的 Town05 v1 训练/C0/GO-HOLD 合同，不是算法上限。综合 v2 因此使用独立目录、合同、指纹和 RunID，禁止拿 v1 checkpoint resume。

## 2. 32 场景矩阵

v2 采用 `8 个能力家族 × 4 个方向 = 32 个场景`。训练道路仍为 Road 44 南北向和 Road 3 东西向；Road 43、Road 6 保留给后续 v2 留出评估。

| 难度 | 家族 | 数量 | 初始结构 | 基础能力 |
|---|---|---:|---|---|
| D1 | `free_flow` | 4 | 相邻车道、`2/4 m/s` 角色镜像 | 独立巡航、不同初速恢复、车道保持 |
| D1 | `parallel_stationary` | 4 | 并排、`0/0 m/s` | 冷启动、基础纵向控制、身份/左右对称 |
| D2 | `parallel_rolling` | 4 | 并排、`3/3 m/s` | 同速并行、侧向感知、避免无谓抢道 |
| D2 | `car_following_steady` | 4 | 同车道、10 m 中心距、`3/3 m/s` | 稳态跟车与安全间距保持 |
| D3 | `stagger_rear_closing` | 4 | 相邻车道错位、后车 `5`、前车 `3 m/s` | 侧后方追近、跨车道速度协调 |
| D3 | `car_following_closing` | 4 | 同车道、12 m 中心距、后车 `5`、前车 `3 m/s` | 纵向闭合识别、主动减速、防追尾 |
| D4 | `car_following_near_gap` | 4 | 同车道、8 m 中心距、`3/3 m/s` | 近间距恢复、平滑拉开安全距离 |
| D4 | `required_merge` | 4 | 相邻车道错位、共享单车道窄目标 | 合法并道、让行时序、冲突消解 |

所有家族均覆盖北、南、东、西四方向。跟车家族的后车身份和使用车道在四方向间交替，防止 Actor 记住固定 Agent ID 或固定左右车道。

这里没有主动对抗场景：没有脚本车辆恶意急刹、别车、阻挡或针对策略动作。所有新增条件都是正常驾驶的初始速度、车距和目标。后续对抗训练继续负责学习/构造攻击行为。

## 3. 课程开放时间

正式 v2 合同计划 160 updates：

| Update | 当前阶段 | 累计可用场景 | 新增能力 |
|---:|---|---:|---|
| `0–19` | D1 | 8 | 自由流 + 静止并排 |
| `20–49` | D2 | 16 | 滚动并排 + 稳态跟车 |
| `50–79` | D3 | 24 | 跨车道追近 + 同车道追近 |
| `80–160` | D4 | 32 | 近间距恢复 + 强制并道 |

新阶段首次开放后用 10 updates 把新增场景的课程权重从 `0.1` 线性提高到 `1.0`；已有场景始终保持权重 `1.0`。在此基础上，失败率自适应采样会适度增加尚未掌握的场景，但设置批内集中上限，避免一个不可解场景吞掉整个更新。

`rollout_episodes=16` 小于最终场景数 32 是允许的。采样器每个 update 保证覆盖 8 个 family，并在 family 内优先选择历史累计采样最少的方向；跨多个 update 完成长周期均衡，而不是要求每次更新枚举全部 32 个场景。

## 4. 为什么同时修改纵向感知范围

原 19 维观察由 8 个自车/目标特征和 11 条全周射线组成，但射线物理量程只有 4 m；奖励的 TTC 候选也只检查 10 m。安全的同车道初始中心距为 8–12 m，如果只增加 JSON，Actor 在场景开始时看不到前车，稳态跟车会成为部分不可观测任务。

v2 保持网络输入仍为 19 维，不扩大模型和显存，只把 11 条归一化射线的物理量程注册为 20 m，并把 TTC 候选范围同步为 20 m。由于相同归一化数值对应的物理距离发生变化，checkpoint 的 observation protocol 升级为 `carla_obs_19_lidar20m_v2`，所以必须 fresh 训练。

当前射线角分辨率仍仅 11 条，这是计算开销与初步能力覆盖之间的折中。若真实 CARLA smoke 证明正前方车辆漏检率高，应另立协议提升射线数量，而不能在正式运行中静默修改。

## 5. 奖励与任务边界

- 继续使用 `reward v3`：路径完成势能、超速/控制变化/LiDAR/TTC/车道偏离成本和终止奖励。
- 不增加“必须保持某个固定车距”的正奖励，避免基础策略为了刷跟车分而拒绝到达终点；D2–D4 通过观测、TTC/LiDAR风险成本和碰撞结局学习安全纵向控制。
- 非并道场景保持出生车道目标；每辆车的居中基准始终是其当帧所在 `Driving` lane 中心。
- required-merge 保留共享窄目标车道，只有该家族要求主动跨线。

需要承认一个边界：两辆车都是学习策略，当前 JSON 只能冻结初始条件，不能让前车在回合中执行确定的速度曲线。因此 v2 可以训练稳态跟车、初始闭合和近间距恢复，但“前车在第 N 秒突然制动”的可重复测试应放入后续独立的扰动评估协议，或在环境中增加非对抗 scripted leader；不能假装当前 JSON 已覆盖动态急刹。

## 6. 文件与命令

| 文件 | 用途 |
|---|---|
| `data/training_scenarios_town05_v2/` | 32 个综合基础训练场景 |
| `utils/build_town05_base_scenarios_v2.py` | 从 Town05.xodr 和冻结 v1 可复现生成 v2 |
| `utils/validate_training_scenarios.py` | 通用 OpenDRIVE/平衡/角色离线校验 |
| `configs/base_mappo/p0_base_town05_v3.json` | 当前 fresh 训练合同；科学参数沿用 v2 |
| `configs/evaluation/p0_base_c0_town05_v3.json` | offroad-aware C0 与结果级重复性阈值 |
| `configs/evaluation/p0_base_go_hold_town05_v3.json` | 消费重复性 semantic gate 的冻结裁决合同 |

离线重建与校验：

```powershell
# 工作目录：PythonAPI/examples/mappo_training
..\..\..\venv\Scripts\python.exe -m utils.build_town05_base_scenarios_v2

..\..\..\venv\Scripts\python.exe -m utils.validate_training_scenarios `
  --scenario-dir data/training_scenarios_town05_v2 `
  --xodr ..\..\..\CarlaUE4\Content\Carla\Maps\OpenDrive\Town05.xodr `
  --town Town05 --expected-count 32
```

真实 CARLA 逐场初始化门禁：

```powershell
..\..\..\venv\Scripts\python.exe -m utils.smoke_training_scenarios_live `
  --scenario-dir data/training_scenarios_town05_v2 `
  --town Town05 --host 127.0.0.1 --port 2000 `
  --lidar-range-m 20 --ttc-max-range-m 20
```

正式入口必须等 v3 critical sources 提交后再执行；否则 formal preflight 会按设计拒绝。必须使用新 RunID 和随机初始化，禁止把任何 Town05 v1/v2、Town02 或诊断 checkpoint 作为 resume/pretrained 来源。

## 7. 下一门槛

1. 32/32 离线几何与平衡校验。
2. 32/32 真实 CARLA 出生位置、初速度和初始射线可见性检查。
3. 1-update 真实训练 smoke，检查 PPO、指标、checkpoint observation protocol 和 actor 清理。
4. 在 Road 43/6 构造同样八家族的 v2 held-out 子集，并冻结新的 C0/GO-HOLD 门槛。
5. 上述门禁全部通过且 critical sources clean 后，才启动 seed 42 fresh 正式训练。

## 8. 当前验证记录（2026-08-12）

- 场景生成：32 个 JSON；8 个 family 均为 4 个方向；D1–D4 均为 8 个场景。
- OpenDRIVE 离线校验：32/32 PASS；四方向各 8，三个同车道跟车 family 的 Agent 0/1 后车角色为 6/6。
- 代码回归：`python -m unittest discover -s tests -p 'test_*.py'`，186/186 PASS。
- 真实门禁第一次调用在生成任何 actor 前的 `client.load_world("Town05")` 等待 60 秒超时；随后 10 秒 RPC 探针也超时，TCP 2000 无监听，系统中没有 CARLA/UE4 进程。该结果不属于场景 PASS 或 FAIL，状态记为 `BLOCKED_SERVER_OFFLINE`，服务器恢复后必须从 32/32 全量门禁重新开始。

## 9. 2026-08-13 offroad 与重复性收口

- 真实 CARLA 32/32 出生点与四方向合法 Driving 并线/强制越界/offroad+goal 反例均通过；车辆实质占用非 Driving 区域连续 3 tick 后显式终止，并保存首证据。
- fresh 诊断 u1 完成 16 episodes、7386 transitions、PPO/metrics/schema-v5 checkpoint/SHA/清理闭环；随机策略产生 10 offroad、2 collision、4 timeout，仅作机制证据。
- Town05 v3 重复性硬字段覆盖 outcome/end reason、success/collision/offroad 和 done/T/X；episode length/active decisions 允许 `±2 tick`，episode return 在 `0.1` 内只告警。不少于 8 cases 且告警比例超过 50% 时仍按系统性漂移 HOLD。
- 新协议前瞻应用且全量 210/210 通过；旧 Town05 u5 的逐 tick `HOLD` 不重判。正式下一步是在 v3 受控 commit 上 fresh seed 42 到 u5。
