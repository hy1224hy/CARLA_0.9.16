# P0-BASE 基础策略实验记录

> 本文是人工可读实验台账；机器可读索引为 [`experiments/base_mappo_registry.csv`](../experiments/base_mappo_registry.csv)。
> 科学参数真值为对应版本化 JSON 契约，不在本文复制后手工改值。
> 当前状态：第一次 seed 42 / update 5 分段已完成训练与双 C0，但 GO/HOLD 来源预检发现 `resolved_config.json` 漏记 seed，故该 Run 只作诊断、不得正式续跑；证据修复通过回归后以新 RunID fresh 重跑。
> 运行约束（2026-08-09）：CARLA 服务由用户手动启动并维护，训练程序仅访问 `127.0.0.1:2000`；发现无响应时停止实验并通知用户重启。此前零 episode/update 的无效启动记录已按用户要求从日志和 registry 删除，不进入实验统计。

## 1. 登记规则

每次 fresh Run 在启动前增加一行，失败 Run 也不得删除。恢复训练继续使用同一 RunID，在“分段记录”追加一行。人工记录至少回答：执行了什么、输入身份是什么、得到什么、是否触发 HOLD、下一步是什么。

自动 registry 负责 RunID、契约/Git/源码/场景指纹、当前停点和状态；本文补充运行环境异常、CARLA 人工观察、C0 结论和研究决策。日志、checkpoint、视频等大文件只记录路径与 SHA，不提交到 Git。

## 2. 契约登记

| 契约 ID | 配置 | 用途 | seed | 计划停点 | 状态 |
|---|---|---|---:|---|---|
| `P0_BASE_MAPPO_TOWN02_PILOT_V1` | `configs/base_mappo/p0_base_pilot_v1.json` | Town02 两红车基础 MAPPO 单 seed 工程 pilot | 42 | 5/10/20/30/40/60/80/100 | FROZEN；允许 update 5 fresh |
| `P0_BASE_C0_TOWN02_V1` | `configs/evaluation/p0_base_c0_eval_v1.json` | 无蓝车/无额外交通的 gate 与 held-out validation | gate: 42001；validation: 51001/51002/51003 | 对应训练 update 5/10/20/30/40/60/80/100 | FROZEN；等待正式 checkpoint |
| `P0_BASE_GO_HOLD_TOWN02_V1` | `configs/evaluation/p0_base_go_hold_v1.json` | 训练健康、分段能力、批量放行及 held-out 数值裁决 | pilot seed 42；C0 seeds 继承上行 | 每个登记停点；只有 update 100 可 `PILOT_GO` | FROZEN；SHA256 `c529edf363cdbeaec93f481a8c70ead4a041bae4d4fc59fdb63cdfd9f8053dde` |

## 3. Run 总表

| 日期 | RunID | Git SHA | 契约 SHA256 | 当前 update | 训练状态 | strict validator | C0 | GO/HOLD | 下一动作 |
|---|---|---|---|---:|---|---|---|---|---|
| 2026-08-09 | `P0_BASE_MAPPO_Town02_s42_20260809_173321` | `8a0bd9d` | `342a1f...f6f` | 5 | `EVIDENCE_HOLD`；模型仅诊断 | PASS：5 updates / 80 episodes / 570 scalar tags | 两次 gate 均 0/16 success、0/16 collision、16/16 timeout；离散重复性 PASS | 来源预检 HOLD：resolved config 漏记 seed，未进入性能裁决 | 提交证据修复后新 RunID fresh 重跑 update 5；禁止 resume 本 Run |

## 4. 已运行记录

### `P0_BASE_MAPPO_Town02_s42_20260809_173321`

#### 预注册输入与训练结果

- 输入：训练契约 `P0_BASE_MAPPO_TOWN02_PILOT_V1`，seed 42，fresh，目标 update 5，来源 Git `8a0bd9d5edaa19cad100c6738145f7a8378f515c`，仅访问用户维护的 `127.0.0.1:2000` CARLA 0.9.16。
- 输出：`logs/P0_BASE_MAPPO_Town02_s42_20260809_173321`；周期 checkpoint `periodic/ckpt_update_000005.pt`，SHA256 `dc4df6134e8744a9eddacf3d96d2d2b66a022800bfea2a42244fe2c311e47a75`。
- 训练段：5 updates、80 episodes、37014 个 agent-transition，约 11 分 35 秒；strict validator 对 updates 1–5、每 update 16 episodes、PPO 更新、ratio 和指标 schema 全部 PASS。
- 训练行为趋势：success 始终为 0；collision rate 从 updates 1–4 的 1.0 降到 update 5 的 0.8125，同时 update 5 出现 0.1875 timeout，平均回合长度从 80.88 增至 401.75。KL、clip fraction、reward residual、非有限值和 skipped update 均未触发硬门槛。

#### 独立 C0 与失败原因

- 主 gate：`evaluations/P0_BASE_C0_gate_u000005_20260809_174729`；16/16 timeout、0 success、0 collision、两车均 0 次到达，validator 全项 PASS。
- 重复 gate：`evaluations/P0_BASE_C0_gate_u000005_20260809_175513`；离散签名逐 case 完全一致，repeatability PASS。第二次只验证重复性，不计作额外 16 个性能样本。
- 行为诊断：确定性 Actor mean 在 gate 中表现为制动/近乎静止，所有 case 跑满 1000 步；“0 collision”不能解释为已学会基础驾驶。
- GO/HOLD judge 在性能门槛前返回 `formal training seed differs from the pilot binding`。实际 seed 42 已由冻结契约、registry 与 checkpoint 交叉确认；根因是实验证据生成复用了不含 seed 的 checkpoint 恢复参数子集，导致 `resolved_config.json.resolved_training_config.seed` 缺失。
- 错误归类：**代码/证据序列化接口缺陷**，不是环境错误，也不是启动指令错误。环境侧 CARLA client/server 均为 0.9.16，训练完整结束且两次 C0 的 cleanup audit 均 PASS；指令侧完整命令使用冻结 `p0_base_pilot_v1.json`、未传 `--seed` 覆盖，契约固定 seed 42，checkpoint `run_metadata.seed=42`。只有 checkpoint 专用 `training_config` 按设计不含 seed，而旧实现错误地将这个最小恢复子集直接写成完整 resolved training config。
- 处置：不修改旧证据、不把本次模型判为性能 HOLD/GO、不从 update 5 resume；修复后新 RunID fresh 重跑。旧 Run 保留为失败证据，其 registry 状态改为 `EVIDENCE_HOLD`。

## 5. 单 Run 记录模板

### `<RunID>`

#### 预注册输入

| 字段 | 值 |
|---|---|
| 假设 | `<该分段要检验的可证伪命题>` |
| Git SHA / dirty 说明 | `<sha；任何非关键 dirty 路径及为何不影响运行>` |
| contract ID / SHA256 | `<id / sha256>` |
| scenario pool fingerprint | `<sha256>` |
| fresh / resume | `<fresh 或 resume latest；resume checkpoint SHA>` |
| 目标 stop update | `<5/10/...>` |
| 完整命令 | `<PowerShell 命令>` |
| CARLA / Python / Torch / device | `<版本和设备>` |
| 输出目录 | `<logs/RunID>` |

#### 分段结果

| update 区间 | wall time | episodes / transitions | success / collision / timeout | KL / clip fraction | entropy / action boundary | critic / grad clip | validator | C0 | 结论 |
|---|---:|---:|---|---|---|---|---|---|---|
| `<a-b>` | — | — | — | — | — | — | — | — | `<GO/HOLD>` |

#### 产物身份

| 产物 | 路径 | SHA256 / 状态 |
|---|---|---|
| resolved config | `<run>/resolved_config.json` | `<sha>` |
| experiment manifest | `<run>/experiment_manifest.json` | `<status>` |
| latest checkpoint | `<run>/latest.pt` | `<sha>` |
| periodic checkpoint | `<run>/periodic/ckpt_update_N.pt` | `<sha>` |
| metrics / episodes | `<run>/metrics.jsonl`、`episodes.jsonl` | `<validator PASS/FAIL>` |
| C0 summary | `<evaluation path>` | `<protocol/checkpoint/config sha>` |
| C0 repeat summary | `<second evaluation path>` | `<repeatability PASS/FAIL>` |
| GO/HOLD decision | `decisions/base_mappo/<DecisionID>/go_hold_decision.json` | `<contract sha / decision>` |
| 视频/截图（如有） | `<path>` | `<episode/update 对应关系>` |

#### 异常、判断与下一动作

- 异常：`<包括失败和 warning，不写“无”来掩盖未检查>`
- HOLD 规则触发：`<规则、证据位置、首次发生 update>`
- 人工观察：`<只描述可见行为，不替代数值门槛>`
- 结论：`<当前证据支持/不支持什么；不能越界推断什么>`
- 下一动作：`<继续到下一个登记停点 / 修复后新 RunID / 终止该契约>`

## 6. 变更纪律

若需要修改任一科学参数、场景文件、reward/lifecycle、模型结构或训练更新逻辑：

1. 当前 Run 标记 HOLD，不在原 Run 静默续跑。
2. 写清触发证据与修改假设。
3. 提升契约 ID/文件名，重新测试并提交 Git。
4. 使用新 RunID fresh 启动；旧 Run 保留作失败/对照证据。

仅 CARLA 临时断连且 checkpoint、指标、清理、代码与契约身份全部通过时，才允许同 Run 严格 `--resume latest`。
