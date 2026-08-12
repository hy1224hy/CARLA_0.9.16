# P0-BASE-02 基础策略 C0 评估协议

> 当前协议 ID：`P0_BASE_C0_TOWN05_V1`（R1 动作 v2 + R2 奖励 v3 + R3 ValueNorm Critic + Town05 场景）
>
> 机器真值：[`configs/evaluation/p0_base_c0_town05_v1.json`](../configs/evaluation/p0_base_c0_town05_v1.json)
>
> 契约 SHA256：文件 `ea2e17f633e3b26eaedbbcf5f42cbaf9b14bc7077790e3dfcba438c0c276cab8`
>
> 当前状态：Town05 v1 实现与 16+8 场景已验收；`a11cf70` 上 gamma/初始化修复后的 fresh 训练 smoke 已通过，当前等待正式 u5 periodic checkpoint
> 边界：本阶段冻结“如何评估”，不声称已有合格基础策略，也不预先宣称训练性能通过。

Town02 v1–v4 契约及既有证据保持只读，仍可离线复验原裁决；新执行默认使用 Town05 v1。动作、奖励与 Critic 升级原理见 [R1 动作](./R1_二维纵向动作协议_论文正文与实验设计.md)、[R2 奖励](./R2_奖励预算校准_论文正文与实验设计.md)和 [R3 Critic](./R3_Critic价值归一化_论文正文与真实CARLA验证.md)。

## 1. 目的

P0-BASE-02 的核心确实是统一基础策略的模型评估标准，但不只是统一几个统计公式。它同时固定以下问题：

1. 哪一种 checkpoint 具备正式评估资格；
2. 评估时 Actor 如何产生动作，能否混入采样噪声；
3. 使用哪些场景和 episode seed，训练内分布检查与独立验证如何隔离；
4. success、collision、timeout、other 如何从环境终局原因唯一映射；
5. 每一步奖励、生命周期和 active-mask 数据不合法时是否立即停止；
6. 逐回合原始证据如何生成汇总、如何中断恢复、如何由独立程序重算；
7. 相同 checkpoint、profile、场景和 seed 重复执行时，如何检查离散结局一致。

历史 [`evaluation/evaluate_no_rllib_mappo.py`](../evaluation/evaluate_no_rllib_mappo.py) 仍可用于查看旧 checkpoint，但它没有当前 checkpoint provenance、active-mask、奖励守恒、固定 case、恢复和证据重算约束，不参与 P0-BASE 正式选模。

## 2. C0 定义与冻结输入

C0 只包含两辆合作红车：

- CARLA client/server 版本固定为 0.9.16，地图固定为 Town05；
- `num_agents=2`，Agent 0 和 Agent 1 各自使用 checkpoint 中对应的独立 Actor；
- 不生成蓝车，不生成额外自然交通；
- observation、action、lifecycle、reward 和 metrics 版本必须与当前 `robust_v2` reward-v3 训练协议完全一致；
- Actor 使用 `tanh(mu)`，不从高斯分布采样；
- `tanh` 动作为二维 `[signed_longitudinal, steer]`，通过生产入口的连续死区映射转换为 CARLA 三通道动作；
- 环境步长 0.1 秒，最长 500 步，奖励版本 v3，`reward_gamma=0.997`；该值必须等于 checkpoint 训练配置和训练环境 runtime 实际值。

正式 checkpoint 必须同时满足：

- checkpoint schema 为 v4，包含精确 `action_spec`、ValueNorm 和可验证的 `.sha256` sidecar；
- 训练协议为 `mappo_robust_v2_valuenorm_p0_00g_v4`；
- Actor 恰好为 0、1，结构为 `obs=19/action=2/hidden=128`；
- 来源训练契约为 `P0_BASE_MAPPO_TOWN05_V1`，文件 SHA256 必须为 `eeb12304d9a262ac4297ede90bd651717d7d59cb31cb8ba0f384835e0ed84bde`；
- `checkpoint_kind=periodic`；
- update 属于 `5/10/20/30/40/60/80/100` 登记停点。

`--allow-diagnostic-checkpoint` 只允许配合 `smoke` profile 检查工具链，输出标为 `DIAGNOSTIC_PASS`，不得用作性能结论、继续训练判断或选模证据。

## 3. 三个 profile

| Profile | 场景 × seed | 用途 | 能否用于正式性能判断 |
|---|---:|---|---|
| `smoke` | 1 × 1 = 1 episode | 真实 CARLA 入口、模型加载和证据链诊断 | 否 |
| `gate` | 16 × 1 = 16 episodes | 每个登记 checkpoint 的训练内分布健康门 | 是，但不用于最终选模排序 |
| `validation` | 8 × 3 = 24 episodes | checkpoint 选择用的固定 held-out 验证 | 是 |

`gate` 完整枚举 [`data/training_scenarios_town05_v1`](../data/training_scenarios_town05_v1)；`validation` 完整枚举 [`data/evaluation_scenarios/town05_c0_validation_v1`](../data/evaluation_scenarios/town05_c0_validation_v1)。两个目录的“文件名 + 文件内容”pool fingerprint 直接写入冻结 JSON；协议加载时还会按单文件内容 SHA256 检查 held-out 未与训练场景重复，并禁止只选取目录中的有利子集。因此，后续即使把场景修改后正常提交 Git，也必须同时提升评估协议版本，不能沿用相同 contract ID/SHA。

8 个 held-out 场景在以下维度保持平衡：难度 1–4 各两个、东/西/南/北各两个、Agent 0/1 后车角色各四个，并覆盖静止起步、同速滚动和后车追近；具体坐标和 seed 一经提交即属于版本化评估输入。

## 4. 终局与逐车口径

| 环境 `episode_end_reason` | C0 outcome | 解释 |
|---|---|---|
| `cooperative_success`、`all_agents_reached` | `success` | 两辆红车均已到达 |
| `cooperative_collision`、`adversary_collision` | `collision` | 严重碰撞终局；C0 正常只应出现 cooperative 分支 |
| `max_steps` | `timeout` | 时间上限截断 |
| `all_agents_terminated` | `other` | 已知兜底终局；不具备 GO 资格 |

四组原因必须互斥，未知原因直接 HOLD。每辆车独立记录：

- episode return、active decision steps、到达和碰撞布尔值；
- 最终 done/terminated/truncated；
- 10 个 v3 奖励分量累计和及最大奖励守恒残差；
- 23 个原始任务/安全/运动诊断的 count/min/max/mean/last；
- 二维策略动作均值、纵向死区率/饱和率，以及映射后的油门、转向、刹车均值与控制模式率。逐回合证据版本为 `p0_base_c0_episode_v4`，同时绑定 reward v3 分量/诊断；策略/环境映射不一致直接拒绝，旧 combined boundary 只保留追溯用途。

评估器只为动作前 `active_mask=True` 的车辆计算和提交动作，也只审计这些车辆当步产生的奖励；网络维度保持固定，失活车辆不生成虚假 transition。每一步都要求：

- `done == terminated or truncated`，且 T/X 互斥；
- 下一步 active-mask 与当前 done 一致；
- 奖励字段和诊断字段完整且有限；
- 总奖励与奖励分量之和误差不超过协议容差。

## 5. 输出和证据链

每个 EvalID 使用独立目录 `evaluations/<EvalID>/`：

| 文件 | 作用 |
|---|---|
| `evaluation_manifest.json` | PREPARED/RUNNING/PASS/HOLD 状态、case 进度、清理审计和产物 SHA |
| `resolved_eval_config.json` | 契约/checkpoint/Git/命令/endpoint、Python/Torch/CARLA client/server、固定 case 列表和输入 SHA |
| `episodes.jsonl` | 逐 episode 原始记录；每条写入后 flush + fsync |
| `summary.json` | 从完整 JSONL 确定性重算的总体、逐场景族、逐场景和逐车汇总 |
| `summary.csv` | 与 summary JSON 同源的平面表格 |
| `validator_report.json` | 独立 validator 的身份、顺序、守恒、重算和清理结果 |

fresh 模式拒绝复用已有目录。`--resume` 只接受相同 EvalID、profile、契约 SHA 和 checkpoint SHA，并要求现有 JSONL 是冻结 case 顺序的唯一完整前缀；不能跳过失败 case、覆盖旧记录或改变 seed 后续跑。

正常或异常退出都会调用环境清理。车辆/传感器残留、同步模式或 fixed delta 未恢复均为 HOLD。summary 只有在所有 case 完成且清理通过后生成；validator 会从 JSONL 重算 summary JSON/CSV 并检查所有 SHA，手工修改汇总不能通过。

## 6. 命令

以下命令必须从 `PythonAPI/examples/mappo_training` 运行。先启动 CARLA 0.9.16 服务端。

### 6.1 正式 gate

```powershell
$checkpoint = "logs/<RunID>/periodic/ckpt_update_000005.pt"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evalId = "P0_BASE_C0_gate_u000005_$stamp"
..\..\..\venv\Scripts\python.exe -m evaluation.evaluate_base_policy_c0 `
  --eval-contract configs/evaluation/p0_base_c0_town05_v1.json `
  --checkpoint $checkpoint `
  --profile gate `
  --eval-id $evalId `
  --output-dir "evaluations/$evalId"
```

### 6.2 正式 held-out validation

```powershell
$checkpoint = "logs/<RunID>/periodic/ckpt_update_000005.pt"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$evalId = "P0_BASE_C0_validation_u000005_$stamp"
..\..\..\venv\Scripts\python.exe -m evaluation.evaluate_base_policy_c0 `
  --eval-contract configs/evaluation/p0_base_c0_town05_v1.json `
  --checkpoint $checkpoint `
  --profile validation `
  --eval-id $evalId `
  --output-dir "evaluations/$evalId"
```

中断后使用完全相同的 checkpoint、EvalID 和目录追加 `--resume`。

### 6.3 独立复核和重复性比较

```powershell
..\..\..\venv\Scripts\python.exe -m evaluation.validate_base_c0_run `
  --output-dir "evaluations/<EvalID>" `
  --expected-contract configs/evaluation/p0_base_c0_town05_v1.json `
  --expected-checkpoint "logs/<RunID>/periodic/ckpt_update_000005.pt" `
  --expected-profile validation `
  --expected-eval-id "<EvalID>"

# 同一 checkpoint/profile 独立跑两次后比较离散签名
..\..\..\venv\Scripts\python.exe -m evaluation.validate_base_c0_run `
  --output-dir "evaluations/<EvalID-A>" `
  --repeat-output-dir "evaluations/<EvalID-B>"
```

重复性签名固定为：case ID、outcome、终局原因、episode 长度和两车到达标志。两次 Run 必须使用相同协议、checkpoint SHA、profile、场景和 seed；EvalID 可以不同。

## 7. PASS/HOLD 边界

本阶段的工程 PASS 要求：

- 契约、场景和输出 schema 严格测试通过；
- checkpoint provenance、动作确定性、active-mask/lifecycle、奖励守恒、恢复前缀和 summary 重算测试通过；
- 入口 `--help` 和离线场景校验通过；
- 代码、契约和场景已提交，正式评估的 critical-source Git preflight 能通过。

下列内容不由 P0-BASE-02 宣称：

- 某 checkpoint 的 success rate 已经达标；
- MAPPO 已收敛；
- 可以跳过 seed 42 pilot 直接批量训练；
- validation 中的 24 episodes 足以替代最终论文多 seed 统计。

Town02 性能 GO/HOLD v1/v2 数值门槛及历史机器裁决保持原样，不做事后篡改；但 gamma 执行偏差使旧训练不再具有新实验的续训/选模资格。当前机器裁决只使用 `P0_BASE_GO_HOLD_TOWN05_V1`。修复后真实训练 smoke 通过才放行 Town05 seed 42 fresh pilot 到 u5，不代表已有性能 GO。C0 第二次重复执行只检查离散可复现性，性能计数只取第一套 case，避免把重复性运行伪装成额外样本量。

## 8. 风险复盘

| 风险 | 当前控制 | 仍需保留的限制 |
|---|---|---|
| 训练场景泄漏进选模集 | 目录隔离、完整枚举、内容 SHA 去重 | held-out 规模仍较小，论文最终需扩大统计 |
| 随机动作混入评估 | 只调用 `get_deterministic_action`，专项 fake-env 重复测试 | CARLA 物理/传感器仍可能有非确定性，故需双 Run 比较 |
| checkpoint 名称相同但内容变化 | `.sha256` sidecar、manifest 和逐回合记录绑定 digest | 移动产物时必须同时保留 checkpoint 与 sidecar |
| inactive agent 污染统计 | 动作前 active-mask 所有权、逐步 T/X 和 count 守恒 | 不应使用历史 evaluator 补正式结果 |
| 汇总表被手工修改 | validator 从 JSONL 重算并比对 SHA | JSONL 本身必须作为原始证据保留 |
| 中断后挑有利 case 重跑 | 只允许精确 case 前缀恢复 | 需要修复科学输入时必须升版并新建 EvalID |
| 清理失败影响下一 Run | actor 数和 timing 恢复为硬门 | 与其他 CARLA 客户端共用服务端会破坏该假设 |
