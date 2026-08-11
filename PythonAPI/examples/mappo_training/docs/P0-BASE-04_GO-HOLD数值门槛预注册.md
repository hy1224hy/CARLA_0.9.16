# P0-BASE-04 GO/HOLD 数值门槛预注册

> **历史 v1 协议说明（2026-08-11）：** 本文以下数值、状态和命令用于复现 R0/WTA v1 Run，不得复制到 R1–R5 修复版新训练。当前机器真值是 [`p0_base_go_hold_v2.json`](../configs/evaluation/p0_base_go_hold_v2.json)，方法和门槛见 [R4 GO/HOLD v2](./R4_GO-HOLD-v2早期塌缩门槛与回归验证.md)，可执行命令只以 [`RUN_COMMANDS.md`](../RUN_COMMANDS.md) 的 C0 v4/GO-HOLD v2 段落为准。
>
> 状态：`FROZEN / PRE-REGISTERED`（2026-08-04）
> 机器真值：[`configs/evaluation/p0_base_go_hold_v1.json`](../configs/evaluation/p0_base_go_hold_v1.json)
> 契约 ID：`P0_BASE_GO_HOLD_TOWN02_V1`
> 文件 SHA256：`c529edf363cdbeaec93f481a8c70ead4a041bae4d4fc59fdb63cdfd9f8053dde`
> 历史结论：v1 门槛在当时正式 pilot 前冻结；旧 Run 后续已因未注册的确定性制动塌缩被研究者诊断封存，机器裁决不改写。修复版当前仅放行 seed 42 fresh pilot 到 u5，多 seed 仍被阻塞。

## 1. 本阶段解决什么问题

P0-BASE-01 冻结了“怎样训练”，P0-BASE-02 冻结了“怎样评估”，P0-BASE-04 冻结“看到什么数值才能继续、停止或放行”。如果先看正式结果再设门槛，会产生事后调阈值、挑 checkpoint 和挑指标的自由度，实验结论不可追溯。

本阶段不调整 MAPPO/PPO、奖励、动作空间、课程或场景，也不运行 CARLA。它只完成四件事：

1. 将训练证据、数值健康、C0 能力和 held-out 验证门槛写成严格 JSON；
2. 用离线判定器统一生成 `CONTINUE/HOLD/CANDIDATE/PILOT_GO` 等结论；
3. 把阈值契约、判定代码的 Git blob 与正式 Run 的 origin Git SHA 绑定，拒绝事后换规则；
4. 用边界回归测试固定 `>`、`>=`、连续 update 和计数口径。

累计回报不参与 GO。原因是回报依赖 shaping 尺度，可能被静止、压线或高风险行为抬高；最终能力必须由“到达目标且无碰撞/车道侵入”的离散事件证明。

## 2. 判定状态

| 状态 | 含义 | 允许的下一步 |
|---|---|---|
| `HOLD` | 证据链、协议或硬数值健康失败，或 update 60 未达到最低学习信号 | 停止原 Run；诊断后按变更纪律决定 resume 或新契约 fresh Run |
| `CONTINUE` | 当前登记停点证据有效，尚不声称策略合格 | 同一 Run resume 到下一登记停点 |
| `CONTINUE_WITH_WARNING` | 可继续，但存在预注册 warning 或尚未达到候选能力 | 同一 Run resume，保留 warning，不得调参消除曲线观感 |
| `CANDIDATE` | update 80 达到临时候选门槛 | 只能继续到 update 100；不能放行多 seed |
| `PILOT_GO` | update 100 通过最终 gate | 放行 P0-BASE-05 多 seed |
| `PILOT_NO_GO` | 证据有效但 update 100 能力不足 | 不放行多 seed；保留失败证据并重新立项 |
| `VALIDATION_ACCEPT/REJECT` | P0-BASE-06 使用 held-out profile 的预注册判定 | 仅用于后续独立验证，不得提前打开结果选阈值 |

warning 不单独否决 `PILOT_GO`；只有契约列明的硬失败和最终能力条件否决。warning 必须进入报告并在实验台账解释。

## 3. 证据硬门槛

下列任一项失败立即 `HOLD`，不能被成功率或回报抵消：

| 证据 | 门槛 |
|---|---:|
| metrics update | 必须从 1 连续到当前登记停点 |
| 每 update episode 数 | 恰为 16 |
| 必需指标缺失 / NaN / Inf | 0 |
| reward 最大逐步守恒残差 | `<= 1e-6`，逐 update 检查 |
| pre-update PPO ratio 最大绝对误差 | `< 1e-6`，逐 agent、逐 update 检查 |
| PPO skipped update | 0 |
| active transition、episode、TensorBoard/JSONL | 严格 validator 全部对账 |
| throttle/brake 同时非零冲突率 | 0 |
| experiment manifest | `SEGMENT_COMPLETED` 且 update 身份一致 |
| checkpoint | 登记停点的 formal periodic checkpoint，哈希和 provenance 通过 |
| C0 | 两次独立执行均通过 strict validator |
| C0 重复性 | 冻结离散字段完全一致 |
| `other` 异常终局 | 0 |
| CARLA 清理 | C0 两次均 vehicle/sensor actor 为 0 且同步设置恢复 |

C0 第二次执行只用于离散重复性审计，不作为额外 16/24 个性能样本。性能计数只取第一套完整 case。

## 4. 训练数值健康门槛

| 指标 | warning | HOLD | 理由与口径 |
|---|---|---|---|
| `approx_kl`（每 actor） | 任一 update `> 0.02` | 单次 `> 0.10`；或连续 2 update `> 0.05` | `0.02` 与训练 target KL 对齐；硬阈值只拦截显著策略跃迁 |
| `clip_frac`（每 actor） | 任一 update `> 0.50` | 连续 2 update `> 0.80` | 高比例裁剪说明 rollout 与更新后策略偏离过大 |
| actor/critic grad clip activation | 连续 3 update `> 0.80` | 不单独 HOLD | 可能表示尺度/学习率压力，但裁剪本身仍在保护更新 |
| critic explained variance | 连续 5 update `< -0.50` | 不单独 HOLD | 长期负值提示 critic 比常数基线更差，但早期波动不应误杀 |
| throttle/steer/brake 真正上界饱和率 | 任一通道连续 3 update `> 0.65` | 不单独 HOLD | 只读 `*_saturation_rate`；踏板 zero 和旧 combined boundary 不进入该规则 |
| actor `log_std_min` | 连续 3 update `<= -0.99` | 不单独 HOLD | 接近当前下限 `-1.0`，提示探索塌缩 |
| actor `log_std_max` | 连续 3 update `> 1.0` | 不单独 HOLD | 提示探索噪声持续过大 |

阈值比较符号是契约的一部分。例如 KL 恰为 `0.10` 不触发单次硬失败，ratio 误差恰为 `1e-6` 则触发硬失败。

## 5. seed 42 gate 里程碑

每个停点使用 C0 `gate` profile：16 个训练分布内场景、seed `42001`、确定性 Actor mean、两次独立执行。

| update | 能力门槛 | 通过状态 | 未通过状态 |
|---:|---|---|---|
| 5/10/20/30 | 只检查证据与数值健康；允许 0 success | `CONTINUE`，有 warning 时 `CONTINUE_WITH_WARNING` | 硬失败为 `HOLD` |
| 40 | 同上；0 success 开始形成 warning | `CONTINUE` 或 `CONTINUE_WITH_WARNING` | 硬失败为 `HOLD` |
| 60 | success `>=1/16`，collision `<=8/16` | `CONTINUE` 或带 warning 继续 | `HOLD` |
| 80 | success `>=12/16`，collision `<=2/16`，timeout `<=4/16` | `CANDIDATE` | `CONTINUE_WITH_WARNING` 到 update 100 |
| 100 | success `>=14/16`；collision `=0`；timeout `<=2/16`；每个场景族 success `>=3/4`；发生 lane invasion 的 episode `=0` | `PILOT_GO`，放行多 seed | `PILOT_NO_GO`，不放行多 seed |

所有行同时受第 3、4 节约束，且 `other=0`。update 80 不能取代 update 100，也不能通过在 80/100 中挑更好 checkpoint 来放行多 seed。

## 6. held-out 门槛（现在冻结，P0-BASE-06 执行）

C0 `validation` profile 固定 8 个 held-out 场景 × 3 seeds，共 24 cases：

- success `>=18/24`；
- collision `=0`，timeout `<=6/24`，other `=0`；
- 四个场景族分别 success `>=4/6`；
- 每个具体场景 success `>=1/3`，防止总成功率掩盖单场景完全失败；
- 发生 lane invasion 的 episode `=0`；
- 两次完整执行的离散重复性必须 PASS。

该门槛是工程候选策略的 release gate，不是统计安全证明。P0-BASE-05 的多 seed 聚合规则和 P0-BASE-06 的最终 checkpoint 选择仍需各自契约，但不能放宽这里已经冻结的单 checkpoint C0 安全/能力下界。

## 7. 标准命令

先做无需 CARLA 的契约预检：

```powershell
cd D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training
$env:PYTHONDONTWRITEBYTECODE='1'
D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe -m evaluation.base_go_hold_protocol `
  --contract configs/evaluation/p0_base_go_hold_v1.json
```

某个训练停点完成 strict validator 和两次 C0 gate 后执行唯一裁决：

```powershell
$U = 5
$RunID = '<P0_BASE_MAPPO_Town02_s42_YYYYMMDD_HHMMSS>'
$EvalA = '<P0_BASE_C0_gate_u000005_YYYYMMDD_HHMMSS>'
$EvalB = '<P0_BASE_C0_gate_u000005_YYYYMMDD_HHMMSS>'
$DecisionID = '<P0_BASE_GO_HOLD_u000005_YYYYMMDD_HHMMSS>'

D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe -m evaluation.judge_base_go_hold `
  --mode milestone `
  --go-hold-contract configs/evaluation/p0_base_go_hold_v1.json `
  --run-dir "logs/$RunID" `
  --checkpoint "logs/$RunID/periodic/ckpt_update_000005.pt" `
  --c0-output-dir "evaluations/$EvalA" `
  --c0-repeat-output-dir "evaluations/$EvalB" `
  --output-dir "decisions/base_mappo/$DecisionID" `
  --expected-update $U
```

必须把示例中的 update 同步替换为当前登记停点的六位文件编号。判定目录不可覆盖；失败或重跑使用新时间戳。退出码 0 表示允许契约定义的下一步，`HOLD/PILOT_NO_GO/VALIDATION_REJECT` 返回非 0。机器结论位于：

```text
decisions/base_mappo/<DecisionID>/go_hold_decision.json
```

报告绑定训练 resolved/manifest、checkpoint SHA、两次 C0 validator、重复性、阈值 SHA、阈值 Git commit、Run origin Git SHA 和关键判定源码 blob。只有当前关键文件与 Run origin 中的版本完全相同时才能裁决，从而阻止事后换门槛或换判定代码。

## 8. 变更纪律与风险边界

- 本契约没有 CLI 阈值覆盖项；不要复制报告后人工改结论。
- 正式结果出现后，不得在同一 ID 下修改数值、比较符或里程碑含义。确需修改时，旧 Run 保留为失败/探索证据，新建 V2 契约并重新 fresh 训练。
- `PILOT_GO` 只证明 seed 42 在登记 gate 上达到工程放行线，不证明跨 seed 稳定，也不证明对抗鲁棒性。
- gate 使用训练分布内场景，只负责阶段健康与最低能力；held-out validation 才参与后续选模。
- C0 是无蓝车、无额外交通基线评估；不能外推到蓝车对抗阶段。

## 9. 本阶段验收

| 维度 | 验收结果 |
|---|---|
| 预注册性 | 阈值 JSON 在正式 pilot 前落盘；Run origin Git 固定其 blob |
| 证据完整性 | 训练、checkpoint、C0、重复性、清理和 Git provenance 全链路校验 |
| 数值正确性 | 临界值、连续 update、缺失/非有限和历史 update 污染均有自动测试 |
| 能力与安全 | success、collision、timeout、other、family/scenario 覆盖和 lane invasion 联合判定；return 不参与 GO |
| 可执行与可追溯 | 单命令生成不可覆盖的 JSON 裁决；P0-BASE 实验台账只引用其路径与结论 |
