# P0-BASE 基础策略实验记录

> 本文是人工可读实验台账；机器可读索引为 [`experiments/base_mappo_registry.csv`](../experiments/base_mappo_registry.csv)。
> 科学参数真值为对应版本化 JSON 契约，不在本文复制后手工改值。
> 当前状态：P0-BASE-01 训练契约和 P0-BASE-02 C0 评估协议均已冻结；尚未启动正式基础策略 pilot，下一步先完成 P0-BASE-04 数值门槛预注册。

## 1. 登记规则

每次 fresh Run 在启动前增加一行，失败 Run 也不得删除。恢复训练继续使用同一 RunID，在“分段记录”追加一行。人工记录至少回答：执行了什么、输入身份是什么、得到什么、是否触发 HOLD、下一步是什么。

自动 registry 负责 RunID、契约/Git/源码/场景指纹、当前停点和状态；本文补充运行环境异常、CARLA 人工观察、C0 结论和研究决策。日志、checkpoint、视频等大文件只记录路径与 SHA，不提交到 Git。

## 2. 契约登记

| 契约 ID | 配置 | 用途 | seed | 计划停点 | 状态 |
|---|---|---|---:|---|---|
| `P0_BASE_MAPPO_TOWN02_PILOT_V1` | `configs/base_mappo/p0_base_pilot_v1.json` | Town02 两红车基础 MAPPO 单 seed 工程 pilot | 42 | 5/10/20/30/40/60/80/100 | FROZEN；等待 P0-BASE-04 门槛 |
| `P0_BASE_C0_TOWN02_V1` | `configs/evaluation/p0_base_c0_eval_v1.json` | 无蓝车/无额外交通的 gate 与 held-out validation | gate: 42001；validation: 51001/51002/51003 | 对应训练 update 5/10/20/30/40/60/80/100 | FROZEN；等待正式 checkpoint |

## 3. Run 总表

| 日期 | RunID | Git SHA | 契约 SHA256 | 当前 update | 训练状态 | strict validator | C0 | GO/HOLD | 下一动作 |
|---|---|---|---|---:|---|---|---|---|---|
| — | — | — | — | — | 尚未运行 | — | evaluator frozen；无 checkpoint 结果 | HOLD | 看结果前先冻结 P0-BASE-04 数值门槛 |

## 4. 单 Run 记录模板

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
| 视频/截图（如有） | `<path>` | `<episode/update 对应关系>` |

#### 异常、判断与下一动作

- 异常：`<包括失败和 warning，不写“无”来掩盖未检查>`
- HOLD 规则触发：`<规则、证据位置、首次发生 update>`
- 人工观察：`<只描述可见行为，不替代数值门槛>`
- 结论：`<当前证据支持/不支持什么；不能越界推断什么>`
- 下一动作：`<继续到下一个登记停点 / 修复后新 RunID / 终止该契约>`

## 5. 变更纪律

若需要修改任一科学参数、场景文件、reward/lifecycle、模型结构或训练更新逻辑：

1. 当前 Run 标记 HOLD，不在原 Run 静默续跑。
2. 写清触发证据与修改假设。
3. 提升契约 ID/文件名，重新测试并提交 Git。
4. 使用新 RunID fresh 启动；旧 Run 保留作失败/对照证据。

仅 CARLA 临时断连且 checkpoint、指标、清理、代码与契约身份全部通过时，才允许同 Run 严格 `--resume latest`。
