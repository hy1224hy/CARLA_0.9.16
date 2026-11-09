# Highwaytest：MUT-MAPPO 轻量多车实验

**新一红两蓝瓶颈v4**：[服务器操作说明](docs/BOTTLENECK_V4_SERVER.md)。真实两车道收窄为一车道，新增道路观测、进度保持率和停车资格、三条件校准与共同驾驶初始化；入口`python -m highwaytest.bottleneck`。保留场景级门控，同输入KL，不宣称逐状态门控或必然取得算法优势。新study重新训练，不与旧结果混表。

**保持通行能力的分阶段训练**：针对v3冷启动后停车的问题，新增[staged_v1服务器流程](docs/STAGED_SERVER.md)：IDM示范初始化 → 普通驾驶PPO → 合格检查 → 8组共享初始化与攻击课程。入口为`python -m highwaytest.staged`，一键脚本为`scripts/run_staged_s42.sh`。物理/评价保持v3，训练设置变化，应另建实验，不与旧结果混表。

**对抗验证请使用新的v3研究协议**：[实验流程与指标](docs/RESEARCH_V3.md)、[算法结论如何在CARLA复验](docs/ALGORITHM_TRANSFER.md)。新配置在 `configs/research_v3/normal` 和 `configs/research_v3/attack`；配对评估入口为 `python -m highwaytest.research`。下文旧v2说明保留供历史实验追溯，不应与v3混表。优势需要实验支持，代码不会保证算法胜出。

本目录是独立实验项目，使用 **Highway-env 1.10.2 的 Road、运动学车辆、碰撞检测和 IDM**。不连接 CARLA，不导入 `mappo_training`，不修改原实验。入口统一从 `PythonAPI/examples` 以 `python -m highwaytest ...` 启动，避免同名 `training/models/environment` 包相互覆盖。

当前为服务器审查后的 **v2**：新增TensorBoard、矩阵预检/批量运行、OS运行锁、完整回报记录和恢复日志修复；critic补充目标进度，改为28维状态，旧v1 checkpoint不兼容。服务器安装、批次运行、SSH查看曲线和GIF导出见 [服务器指南](docs/SERVER.md)。

需要实际训练车辆的实时画面时，单组训练增加 `--preview`，用 `serve-preview --run runs/你的目录 --port 8000` 提供浏览器预览；`--preview-speed 1` 可按仿真时间播放。具体服务器/本地端口转发见服务器指南第8节。预览读取实际采样轨迹，不运行第二个环境；默认关闭。

中断恢复用 `train --resume`；完成原预算后延长训练用 `extend --checkpoint runs/旧目录/latest.pt --steps 新累计总步数 --output runs/新目录`，继承完整optimizer、教师、门控和RNG状态，保留父实验。旧源码checkpoint需审查后显式 `--allow-source-change`。详见服务器指南第9节。

当前交付为可运行的算法迁移与实验框架。短测只验证工程闭环，不构成 MUT-MAPPO 或通信有效性的实验结论。场景、奖励和门控阈值仍须经过预实验冻结。

## 目录管理

```text
highwaytest/
  config.py                配置、组别、配置哈希
  __main__.py              train / evaluate / plan / report
  configs/                 可审查的 smoke、正常驾驶和对抗配置矩阵
  environment/             Highway-env 适配、场景划分、统一奖励
  models/                  独立角色 actor、消息编码器、集中 critic
  training/                PPO/GAE、冻结教师/门控、A/B runner、checkpoint
  evaluation/              独立评估、消息干预、跨 seed 汇总与曲线
  diagnostics/             全组短测、脚本驾驶可行性探针
  tests/                   信息边界、概率、GAE、恢复与预算检查
  docs/                    迁移口径与验证记录
  runs/                    自动生成，git 忽略；每次实验独立子目录
```

每个 run 保存 `config.json`、依赖/配置/源码哈希的 `manifest.json`、`source/` 源码快照、训练/回合/验证 JSONL、`latest.pt`、`selected.pt` 和 `summary.json`。评估存于 `evaluations/`，不覆盖已有报告。训练不覆盖已有 run 目录；模型只从验证集选取一个 A/B 团队。只能加载可信的本地 checkpoint。

## 环境与科学口径

- 固定两车道直路。红车 ID 0/1，固定参数 IDM，禁止换道，目标速度 18m/s；蓝车 ID 2/3，为两个可学习 actor。A/B 是两套团队学习器，每次物理仿真仍只有四辆车。
- `parallel`：蓝车在红车后方；`closing`：红车在较慢蓝车后方。每回合随机间距，按固定物理角色对应。首版没有匝道并线、真实遮挡或驾驶风格池，不能把它们写成已实现功能。
- 每蓝车 24 维局部观测：自身 6 维 + 另三辆车各 6 维；25m 外车辆特征全部置零。集中 critic 单独读取四车28维全状态，含每车目标进度。
- 动作 `[加速度/5, 转角/0.25] ∈ [-1,1]^2`，不是 CARLA 的油门/刹车协议。策略频率 5Hz，物理频率 15Hz；物理子步将车速限制在 [0,30]m/s，排除原生 Vehicle 允许的倒车。目标为两辆蓝车各前进 180m，最长 20s；任意车碰撞或驶离道路立即终止。
- 正常驾驶阶段 `attack_weight=0`。每步驾驶奖励为 `0.6*clip(前进距离/(20*dt),-1,1)+0.2*clip(speed/20,0,1)-0.1*clip(abs(横向误差)/2,0,1)-0.1*abs(sin(heading))-0.02`。碰撞团队每个角色 -10、驶离道路 -5。默认无需达到所有旧 CARLA 驾驶门槛才给进度信号。
- 对抗阶段追加 `attack_weight*clip(红车参考前进量-真实前进量)/(18*dt)`，clip 作用于整个比值、范围 [0,1]。参考为相同初始红车、相同固定控制器、无蓝车的独立道路，每物理步同步推进。违规当前步不发对抗正奖励。两阶段改变的是共同任务，所有算法组必须使用相同阶段/奖励。
- `red_delay_m` 是相同仿真时间下参考红车与真实红车的平均位置差；不能只比较因提前碰撞结束的累计回报。最终同时报告成功、碰撞/出界、蓝车进度和红车损失。
- TTC 为同横向带内的纵向保险杠距离/接近速度，子步检测；`safe_event` 为出现 TTC<2s 且回合无碰撞/出界，**不表示必然完成了有效避险**。`qualified_event` 额外要求两蓝车到达。`min_ttc_s=100` 表示未观测到更小的有效 TTC，不是精确无限时距。`hard_brakes` 统计红车制动<-3m/s² 的“车辆×决策步”数，不是去重制动事件数。
- 门控/选模 quality：正常驾驶为安全成功率；对抗阶段为 `安全*(成功+0.25*clip(red_delay_m/180,0,1))`。选模 quality 相同时依次优先较低违规率、较高蓝车进度，门控仍只按 quality 优势与安全条件决定。该标量是首版协议，不宣称最优。

## 算法与对照

|组别|互学习|队内通信|
|---|---|---|
|G0|关闭|关闭|
|G1|普通互学习|关闭|
|G2|质量门控互学习|关闭|
|G3|关闭|开启|
|G4|质量门控互学习|开启|
|G5（可选）|普通互学习|开启|

所有组保留 A/B、独立 optimizer、独立随机流、相同总采样预算和相同选模程序。A/B **不是两个独立 seed**。`total_train_steps` 为 A+B 合计环境决策步；每步控制两蓝车，所以蓝车 agent steps 是它的两倍。另记验证消耗，验证轨迹不进入 PPO。

同角色、同观测行上计算 `KL(student || stopgrad(peer))`，使用 tanh 前高斯分布，保留原项目 student-to-teacher 方向；以所有样本/角色为分母，不只平均开门样本。双方只使用自己采集的 on-policy 数据做 PPO，另一学习器只作冻结教师。首轮及每 `refresh_rounds` 轮更新教师与配对 gate 证据；先经过 `warmup_rounds`。门控要求教师安全违规率不高于阈值/学生、quality 优势达到 margin。门控可能全关，应检查 `gate_fraction`，不能把“实现了门控”写成“证明门控有效”。

消息为 8 维本地观测编码、单轮发送给唯一队友；接收距离默认80m，显式可用性 mask。没有退化的单邻居 softmax 注意力。PPO 训练时重算当前编码器与接收 actor，教师用自己的编码器重算消息。存储 pre-tanh `z`、旧 log_prob、联合局部观测和 mask；每次更新前检查 log_prob 一致性。无通信组保留同形状模型、编码器输出乘零以匹配分配参数量，同时须说明其有效使用参数量不同。

每轮 rollout 结束后主动重置下轮环境；未自然结束的尾段正常 bootstrap，但不计作完整回合。这样 checkpoint 不依赖序列化仿真活对象，可以从完整 A/B 边界恢复。训练自然 time-limit 使用 next-value bootstrap，真正终止置零，GAE 不跨回合。

## Windows 使用

以下命令从 `D:\study\yan2\CARLA_0.9.16\PythonAPI\examples` 执行。使用根目录现有 venv；Highway-env 已安装到该环境。

```powershell
$py = 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe'
& $py -m highwaytest --help
& $py -m unittest highwaytest.tests.test_contracts -v
& $py -m highwaytest.diagnostics.smoke
```

1. 先运行正常驾驶 G0 预实验，确认独立评估中出现稳定成功行为。10万步是起点，不是收敛保证。若基线不成立，先诊断任务与训练，暂不比较模块。

```powershell
& $py -m highwaytest train --config configs/driving_g0.json --output runs/driving_G0_s42
& $py -m highwaytest evaluate --checkpoint runs/driving_G0_s42/selected.pt --split selection --gif
```

2. 对抗矩阵已提供 G0–G4 × seeds 42/43/44。可以从头训练；也可给所有组显式设置相同正常驾驶边界初始化。共用预训练 checkpoint 的成本必须另记。`--initialize-from` 会复制两团队模型但重置 optimizer，切换通信时编码器尚未被正常驾驶训练。不得混合有/无预训练的组直接归因模块收益。

```powershell
& $py -m highwaytest train --config configs/adversarial/G0_s42.json --output runs/attack_G0_s42
& $py -m highwaytest train --config configs/adversarial/G2_s42.json --output runs/attack_G2_s42
& $py -m highwaytest train --config configs/adversarial/G4_s42.json --output runs/attack_G4_s42
# 可选，全部组采用同一方式：
# ... --initialize-from runs/driving_G0_s42/latest.pt
```

3. 暂停可用 `--max-rounds N`，只在完整边界退出。恢复使用 `latest.pt`，要求配置与 Python 源码一致；源码快照便于审查，入口不会自动执行旧快照。异常退出最多重做一个 A/B 轮次。

```powershell
& $py -m highwaytest train --resume runs/attack_G4_s42/latest.pt
```

4. 所有算法与奖励冻结后，才对 `selected.pt` 做最终测试。train/gate/selection/test 使用不同 seed 空间；禁止根据最终测试反复选择模型。通信干预包括置零、前一决策步过期消息和独立随机丢包；过期消息不是跨回合打乱，也不能替代从头训练的无通信组。

```powershell
& $py -m highwaytest evaluate --checkpoint runs/attack_G4_s42/selected.pt --gif
& $py -m highwaytest evaluate --checkpoint runs/attack_G4_s42/selected.pt --intervention zero
& $py -m highwaytest evaluate --checkpoint runs/attack_G4_s42/selected.pt --intervention stale
& $py -m highwaytest evaluate --checkpoint runs/attack_G4_s42/selected.pt --intervention drop --drop-probability 0.3
& $py -m highwaytest report --runs runs/attack_matrix --output runs/comparison
```

汇总命令扫描指定目录下的普通最终测试报告，要求任务/奖励/测试案例相同、每个 group/seed 只有一份报告；输出跨独立 seed 的均值、样本标准差、CSV 和训练曲线。将正式矩阵放在单独父目录（如 `runs/attack_matrix/G0_s42`），不要把 smoke、正常驾驶与对抗结果混在一个汇总根目录。少于3个 seed 时只能作探索分析。

可另生成新矩阵：`python -m highwaytest plan --output configs/new_matrix --steps 100000 --attack-weight 0.3`；默认不启动15组长实验。

## 与 CARLA 原实验的边界

迁移了 CTDE、角色独立 actor、双学习器、冻结教师、质量门控、普通互学习对照、公平总预算与独立选模等研究逻辑；PPO 核心是清晰独立的实现，未复用导入 CARLA 的训练脚本。旧的20维/2维 CARLA checkpoint、动作死区、危险几何协议与奖励数值不适用于本项目，也没有伪装为严格续训。

暂未迁移驾驶风格池、PSRO、门控 random/strength 机制对照、ValueNorm、复杂交互案例生成和 CARLA 的完整 C0/C1/C2 评估协议。此处仅使用“无蓝车参考”作为可解释延迟度量。普通/门控互学习和通信是否有效，仍由后续多 seed 实验回答。

参考：[Highway-env 官方源码](https://github.com/Farama-Foundation/HighwayEnv)、[多智能体接口说明](https://highway-env.farama.org/v1.10.1/multi_agent/)。首版直接组合其物理/控制组件并提供团队 Gymnasium 接口，没有声称使用未经修改的官方标准 benchmark。
