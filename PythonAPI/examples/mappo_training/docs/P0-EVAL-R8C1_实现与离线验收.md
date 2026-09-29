# P0-EVAL-R8C1 实现与离线验收

日期：2026-09-14。状态：`IMPLEMENTED / OFFLINE PASS / AWAITING USER-STARTED CARLA`。

> 2026-09-15 更新：用户启动 CARLA 后，已完成全部42回合；10个Run/42条记录完整性PASS，重复性HOLD，现场清理通过。最新状态、指标和下一步以 [真实评估与指标诊断](./P0-EVAL-R8C1_20260915_真实评估与指标诊断.md) 为准；下文保留9月14日离线验收时的状态。

用户于 9 月 11 日批准 [独立评估修复计划](./P0-EVAL-R8C1_独立评估修复计划_待审核.md)，并要求需要 CARLA 时由用户启动。本次完成评估代码、冻结诊断合同与离线验收；没有连接 CARLA，没有新训练或策略能力裁决。9 月 14 日的回归和真实 checkpoint 预检结果已保存，后续从本文继续，不必重做项目审查。

## 1. 完成内容

| 范围 | 已实现行为 |
|---|---|
| 原 C0 协议扩展 | 新注册 `p0_base_c0_diag_town05_r8c1_v1`；严格支持 schema8、rewardv6、obs20/action2/state40、R8 课程/采样及 critic probe；旧 R6/R7 规则保留 |
| 固定输入 | 原 u20/u60/u100 文件、sidecar 文件名与 SHA、manifest、原训练合同与模型/协议逐项核对；诊断开关不能绕过来源不匹配 |
| 场景时限 | 展开 case 时固定 200/500 tick；apply、reset、执行循环、回合记录、独立 validator 和恢复评估统一核验；提前 mission_timeout 被拒绝 |
| 执行行为 | Actor eval/no_grad/冻结梯度；只对 active agent 推理与统计；每回合和运行结束核验权重，结束重查 checkpoint 文件和源码身份 |
| 诊断用途 | record、resolved、manifest、summary JSON/CSV、成功的 validator/repeatability 报告明确 diagnostic_only、selection_allowed=false、NOT_EVALUATED；正式选模资格始终为 false |
| 五结局与归因 | success/collision/offroad/road_violation/timeout；逐 agent 终止原因、道路证据、v6 所有奖励分量及 pace 诊断；pace 奖励诊断和奖励分量总和一致 |
| 分层汇总 | 按场景、方向、家族和诊断分组输出；每个分组保留逐 agent 成功/安全失败、最终完成度、速度等诊断均值、动作统计与奖励分量。短 rolling、短 stationary、长 free-flow 和长 stationary 分开 |
| 重复性 | 保留既有 tick/return 容差；新增道路确认/终止、分类、原因、参考车道及成功资格硬比较；少于 8 cases 的系统性漂移项明确不适用，逐 case 检查仍有效 |
| 来源与恢复 | 原训练与新评估源码分别记录；39 个训练/共享文件与原 22 路径 fingerprint 核验；源码不同禁止续写同一 EvalID，仅接续完整回合前缀；validator 每次重读文件和重建汇总 |
| 清理 | 初始化前只读检查已有车辆/传感器/同步占用；初始化部分失败恢复世界时序；正常退出、异常退出均记录清理审计；不删除外部 actor |

实现文件为 `evaluation/base_c0_protocol.py`、`evaluation/evaluate_base_policy_c0.py`、`evaluation/validate_base_c0_run.py`、新增 `evaluation/r8c1_diagnostic_protocol.py`、新 JSON 合同与 `tests/test_base_c0_r8c1.py`。没有修改训练器、环境、模型、rewardv6 或原始 Run。

## 2. 实施中补充发现并修复的问题

| 问题 | 处理及验收 |
|---|---|
| 只检查 profile 数量，仍可修改 seed、formal_use、重复次数或未核验的逐场景 SHA | 固定 01 / 01–04 / 05–12 清单、61002/61001、诊断用途与 repeat=2；计算 42 次执行；逐场景 SHA 同时核对实际文件和训练源码记录；篡改测试通过 |
| R8 运行可能接受旧 R7 record，从而绕过 v6 和时限字段 | case 明确绑定 R8 record；独立重建 summary 的协议必须匹配合同；版本降级拒绝测试通过 |
| 最终记录可以自洽地写成 done=false，或把自然 timeout 改为外部截断 | R8 记录强制完整结束，依据自身到点/碰撞/offroad/road 与 horizon 验证 T/X；v6 mission_timeout 保持 MDP 内 terminated；篡改测试通过 |
| 环境构造中途失败时 env 尚未赋值，原清理分支无法恢复同步模式 | 新增无车辆构造失败的时序恢复；捕获关闭后审计 RPC 异常，保留原异常与 manifest；假环境实测构造失败→恢复时序→同 Run resume→完整评估及独立 validator PASS |

## 3. 离线证据

- 全量：`python -B -m unittest discover -s tests -q`，**337/337 PASS，46.949 秒**，退出码 0；其中新增 R8-C1 专项 22 项。旧 315 项仍通过。
- [测试日志](../diagnostics/r8c1_eval_offline/20260914_212000/unittest.log)、[命令/时刻/退出码](../diagnostics/r8c1_eval_offline/20260914_212000/unittest_result.json)。存在既有 pygame/pkg_resources 弃用提示，未影响测试。
- [真实 checkpoint 离线预检及完整执行矩阵](../diagnostics/r8c1_eval_offline/20260914_212000/checkpoint_preflight.json)：u20/u60/u100 全部 PASS；39 文件身份 MATCH；确定性推理和 Actor/文件 SHA 不变；`carla_connected=false`。
- `git diff --check` 通过。本次没有创建提交；训练 Git 与评估源码身份分开记录，评估运行不要求伪造原训练提交身份。

合同：[`p0_base_c0_town05_r8c1_probe_v1.json`](../configs/evaluation/p0_base_c0_town05_r8c1_probe_v1.json)，SHA256 `0654a931a0387ffdbf3884bf83ef5ae63899fcf0aa5ba922a9acc6f4748e736d`。

| checkpoint | SHA256 |
|---|---|
| u20 | `1cfd66872171427ecf68eddb530df5defe976503d5b172549dbb8291c5ec51be` |
| u60 | `8d702aae3a292553c2254aec22480a689e4b17df3930afa1b326f45561b9d280` |
| u100 | `c3519fb3ccb0b55f4b14f735849dd0093bc28914feee8ece60c556dbb417d1d6` |

原训练 Git：`050fe9fc72b9639b3048f56384cc48ad0d8d0214`；离线预检时评估源码 fingerprint：`494359502c7a93d9f4e944f9313d087c2abb709ea51751a03c18fec176a45c49`。此 fingerprint 识别代码内容，不代表 Git 已提交。

## 4. 下一步：真实 CARLA 验证

1. **用户启动 CARLA 0.9.16**，服务端口 `127.0.0.1:2000`，确保没有训练、手动驾驶或其他同步客户端。可在仓库根目录执行 `./CarlaUE4.exe -quality-level=Low`。本次尚未请求连接或启动模拟器。
2. 先用 u100、场景 01、seed61002 执行两个独立 smoke。每次使用不同 EvalID/目录，独立 validator 后再比较重复性；若初始化、证据、重复性或清理失败，先处理该问题，不能直接扩量。
3. smoke 完整性通过后，执行 short_probe 的 u20/u60/u100 各主/重复一次，共 24 回合；执行 u100 long_probe 主/重复，共 16 回合。总计 42 次执行，最多 13200 tick；模拟时间上界 22 分钟，不是墙钟耗时承诺。
4. 把真实逐回合结果和分层诊断写回主 TODO 和实验记录。u100 固定为主结果，u20/u60 仅作演变参照；Road43/6 本轮不运行；不生成 best、不自动续训/入池或授予 GO。

真实结果决定后续优化：先看已训练短 rolling 的确定性驾驶；再区分 stationary 起步迁移与长任务失败，并检查方向/agent 差异。当前不根据训练 return 或离线测试 PASS 提前决定修改奖励、加长训练或进入对抗阶段。

单次 smoke 的 PowerShell 入口（由评估执行者在 CARLA 启动后运行，当前尚未执行）：

```powershell
Set-Location 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
$evalId = 'P0_BASE_C0_smoke_u000100_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -B -m evaluation.evaluate_base_policy_c0 `
  --eval-contract configs/evaluation/p0_base_c0_town05_r8c1_probe_v1.json `
  --checkpoint logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224/periodic/ckpt_update_000100.pt `
  --profile smoke --eval-id $evalId --output-dir ('evaluations/' + $evalId) `
  --host 127.0.0.1 --port 2000
```

重复性入口为 `python -B -m evaluation.validate_base_c0_run --output-dir <主目录> --repeat-output-dir <重复目录>`。每次比较重新校验两个完整目录，不读取缓存 PASS 作为依据。
