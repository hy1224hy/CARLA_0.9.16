# P0-BASE-01 基础策略正式实验契约

> 状态：IMPLEMENTED / CONTRACT FROZEN（2026-08-04）
> 唯一训练入口：`training.train_no_rllib_mappo_robust_v2`
> 机器可读真值：[`configs/base_mappo/p0_base_pilot_v1.json`](../configs/base_mappo/p0_base_pilot_v1.json)
> 实验台账：[`P0-BASE_基础策略实验记录.md`](./P0-BASE_基础策略实验记录.md) 与 [`experiments/base_mappo_registry.csv`](../experiments/base_mappo_registry.csv)

本阶段冻结的是“怎么启动、参数从哪里来、何时停、产物如何追踪”的实验契约，不证明策略已收敛，也不授权直接开始多 seed 批量训练。完成 P0-BASE-02 的独立 C0 evaluator 后，才允许执行 P0-BASE-03 的 seed 42 分段试训。

## 1. 目的与真值层级

正式实验最容易出现的污染不是公式写错，而是同名 Run 实际使用不同默认值、恢复时 LR horizon 被缩短、命令行参数无法追溯，或看到结果后再改停止标准。当前采用以下真值顺序：

1. 版本控制 JSON 契约是科学参数的唯一真值。
2. 每个 Run 的 `resolved_config.json` 是该 Run 启动时解析后的证据，不反向修改契约。
3. checkpoint `run_metadata`、`experiment_manifest.json`、`metrics.jsonl` 和 registry 是执行证据。
4. 本 Markdown 用于解释设计和记录人工判断；若文字表格与 JSON 不一致，以 JSON 契约和它的 SHA256 为准。

`argparse` 默认值只服务历史/诊断入口，不再被视为正式实验配置。正式模式通过 `--experiment-contract` 开启，科学参数若在 CLI 重复指定，即使值相同也会被拒绝，避免形成两个真值源。

## 2. 参数冻结与依据

### 2.1 训练规模和 PPO 数据复用

| 项目 | 冻结值 | 依据和边界 |
|---|---:|---|
| seed | `42` | 仅作为工程 pilot；不能据此声称跨 seed 稳定 |
| planned updates | `100` | LR horizon 和完整计划长度；不是要求一次跑完 |
| 分段停点 | `5,10,20,30,40,60,80,100` | 全部与 5-update periodic checkpoint 对齐 |
| rollout episodes/update | `16` | 保留已通过真实 smoke 的采样规模设定 |
| max steps/episode | `1000` | 与 Town02 当前任务和环境协议一致 |
| PPO epochs | `5` | 采用 MAPPO 困难任务建议的 5–10 epochs 中较保守端，限制同批数据反复更新造成的非平稳性 |
| minibatch size | `32768` | 上界覆盖 `2 agents × 16 episodes × 1000 steps = 32000` 条 critic 样本；实际样本不足时使用实际全批，不补伪样本 |
| min transitions/update | `512` | 低于门槛不做“凑数更新”，正式 Run 直接 HOLD |

PPO 原论文定义了“先采样、再对同一 on-policy 数据做多 epoch/minibatch 优化”的基本范式；MAPPO 论文和官方实现经验进一步指出，多智能体任务对过多数据复用更敏感，困难任务通常采用 5 或 10 epochs，并尽量减少 minibatch 数量。[PPO 论文](https://arxiv.org/abs/1707.06347)、[MAPPO 论文](https://arxiv.org/abs/2103.01955)、[MAPPO 官方实现](https://github.com/marlbenchmark/on-policy)、[BAIR MAPPO 实践说明](https://bairblog.github.io/2021/07/14/mappo/)

`32768` 不是从论文照抄的通用默认值，而是按本项目两车、16 回合、1000 步的样本上界推导出的“单 epoch 单全批”配置。它会改变相对 P0-00G smoke 中 `256` 小批的 optimizer step 数，因此本 pilot 与历史 smoke 不做学习曲线等价比较；算法目标、网络结构和 checkpoint 张量形状没有改变。

### 2.2 优化、时序和探索参数

| 参数 | 冻结值 | 理由与首段监控 |
|---|---:|---|
| actor / critic LR | `3e-4 / 3e-4` | 已在 P0 smoke 中保持有限；比 MAPPO 官方常见 `5e-4` 更保守，不因短 smoke 擅自上调 |
| gamma | `0.997` | 有效视野约 `1/(1-γ)=333` 步，保留长距离终点信号设计 |
| GAE lambda | `0.95` | 保留已验收的偏差—方差折中和边界实现 |
| clip epsilon | `0.15` | 比常见 `0.2` 更保守，又避免 `0.05` 级过小 clipping 导致学习过慢 |
| target KL | `0.02` | 作为 clip 之外的单 agent、单 epoch提前停止保护 |
| entropy coefficient | `0.03` | 当前任务早期需要探索；首段重点看 entropy、动作边界率、KL 和 clip fraction |
| init log std | `0.0` | 初始 `std=1`；不基于未收敛 smoke 的动作越界现象事后改小 |
| max grad norm | `0.5` | 与官方 MAPPO/PPO 常用保护一致，延续已有回归 |
| hidden dim | `128` | 保持两套独立 Actor 和 agent-conditioned centralized Critic 的现有结构 |
| LR decay | 开启，horizon `100` | 每次只改变停止 update，不允许把 LR 计划压缩到当前分段长度 |

这里区分三种依据：论文/官方经验决定方向，当前任务结构决定 `gamma` 和 batch 上界，P0 smoke 只用于排除明显数值不稳定。真实成功率还不存在，因此没有用训练结果反向挑选超参数。

### 2.3 场景和课程

| 项目 | 冻结值 |
|---|---|
| Town / 道路长度 / timestep | `Town02 / 100 m / 0.1 s` |
| 场景池 | `data/training_scenarios_town02_v2`；启动时计算文件级 pool fingerprint |
| 采样 | `weighted` |
| 课程模式 | `metadata` |
| warmup / full stage | update `10 / 40` |
| transition window | `10` updates |
| reward | `v2` |

契约同时要求场景目录和关键源码处于已提交状态。无关个人文件不参与关键源码指纹，但仓库整体 dirty 状态及路径仍写入 Run 元数据，人工台账必须说明例外。

## 3. 正式命令

以下命令必须从 `PythonAPI/examples/mappo_training` 执行。首次只登记到 update 5；在 P0-BASE-02 完成前不要运行。

提交后可先执行不连接 CARLA 的契约/Git 预检：

```powershell
..\..\..\venv\Scripts\python.exe -m training.experiment_contract `
  --contract configs/base_mappo/p0_base_pilot_v1.json
```

只有输出 `"status": "PASS"` 才允许继续；关键源码或场景 JSON 未提交时会直接失败。

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runId = "P0_BASE_MAPPO_Town02_s42_$stamp"

..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --experiment-contract configs/base_mappo/p0_base_pilot_v1.json `
  --updates 5 `
  --run-id $runId `
  --save-dir "logs/$runId"
```

同一 Run 在验证 update 5 后续接到 update 10：

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --experiment-contract configs/base_mappo/p0_base_pilot_v1.json `
  --updates 10 `
  --run-id $runId `
  --save-dir "logs/$runId" `
  --resume latest
```

正式 CLI 只允许改变 `host`、`port`、`cpu`、停止 milestone、RunID、输出目录和严格 resume 来源。`seed`、PPO、课程、日志间隔、reward 等参数必须修改 JSON、提升契约版本并重新提交，不能临时覆盖。

## 4. 运行产物和状态机

fresh Run 在连接 CARLA 前产生：

| 产物 | 用途 |
|---|---|
| `resolved_config.json` | 契约/解析配置、Git、源码指纹、场景池 fingerprint、HOLD 规则 |
| `experiment_manifest.json` | `PREPARED → RUNNING → SEGMENT_COMPLETED / INTERRUPTED / HOLD`；连接 CARLA 前失败记为 `ERROR_PRE_TRAIN` |
| `experiments/base_mappo_registry.csv` | 跨 Run 机器可读索引；相同 RunID 只更新一行 |

训练过程中继续使用已经冻结的 `latest.pt`、`periodic/ckpt_update_*.pt`、`checkpoint_manifest.json`、`metrics.jsonl`、`episodes.jsonl` 和 `tb/`。恢复时除 checkpoint schema、源码和训练参数外，还要求契约 SHA、resolved config 和 experiment manifest 身份一致。

## 5. 预注册停止规则

下列任一项出现时，当前分段标记为 `HOLD`，保留产物并停止继续增加 update：

- 任意 loss、KL、ratio、value、reward 或关键诊断出现 NaN/Inf。
- reward conservation 或 pre-update ratio 守恒绝对误差大于 `1e-6`。
- active decision、transition、episode boundary 或 lifecycle 对账出现任何不一致。
- update 被跳过、缺失任一 required metric，或没有达到最小有效 transition。
- CARLA actor 泄漏数大于 0，或同步模式/环境设置未恢复。
- checkpoint SHA/schema/manifest、JSONL、逐 episode 记录或 strict validator 任一不完整/失败。
- 首段 `steer_saturation_rate` 超过 `0.65` 时先发出 warning；throttle/brake 只检查各自的**上界饱和率**，不能把互斥映射产生的零值计为饱和。JSON 中保留的 `action_boundary_rate_warning=0.65` 由 P0-BASE-04 按此语义解释；结合 entropy、KL、clip fraction、驾驶模式率、回报分量和视频判断是否升级为 HOLD。

成功率、碰撞率和 timeout 的正式 GO/HOLD 数值阈值必须在 P0-BASE-02 evaluator 完成后、查看 seed 42 pilot 结果前于 P0-BASE-04 冻结。训练 JSON 中只预留外部 `p0_base_c0_eval_v1` 接口和 milestone，不会在 P0-BASE-02 回写或改变本契约 SHA；评估器和性能门槛使用各自版本化文件。当前阶段刻意不编造一个没有独立评估依据的成功率阈值。

## 6. 自我审查与未完成边界

- 单 seed 不能支持论文统计结论；seed 42 只是工程 pilot。
- `100 updates` 是计划和 LR horizon，不代表已经证明足够收敛；只能在预注册停点上追加，不能看到曲线后任意挑结束点。
- 全批更新符合 MAPPO 减少 minibatch 的经验，但相对历史 smoke 降低了每 epoch optimizer step 数；首段必须同时检查 KL、clip fraction、相对参数更新量和 critic 梯度裁剪率，防止更新过弱。
- `entropy=0.03`、`init_log_std=0` 是需要用首段健康信号检验的任务参数，不宣称是论文最优值。
- P0-BASE-01 不运行 CARLA 调参：C0 尚未冻结，此时依据训练成功结果调参会破坏预注册。P0-BASE-02 完成后先跑 update 5，再按本契约判定 GO/HOLD。

## 7. P0-BASE-01 验收

- JSON 严格 schema：缺字段、未知字段、非有限数、错误课程/停点/LR horizon均 fail-fast。
- 正式 CLI 科学参数覆盖、非登记停点、RunID/目录不一致均 fail-fast。
- 最大 critic batch 下仍为单全批；不生成 padding transition。
- fresh/resume 的契约 SHA 和证据文件闭环；registry 同 RunID 原子更新且不可漂移身份字段。
- 不改变 Actor/Critic 网络形状、环境 reward/lifecycle、GAE/PPO 公式或 checkpoint schema；新契约字段只追加到 `run_metadata`。训练源码指纹因本阶段实现发生变化，所以 P0-BASE 正式 Run 必须 fresh 启动，旧 Run 不再具有“相同源码”的 strict-resume 身份。
