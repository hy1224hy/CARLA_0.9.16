# R5：最终训练入口真实 CARLA smoke 验证

> 状态：PASS / R5 COMPLETE（2026-08-11）。
> 性质：最终生产入口的工程与证据链验证，不是策略性能 pilot，不产生可用于 pretrained/选模的 checkpoint。
> 基线提交：`00e16f3 feat(mappo): freeze R4 GO HOLD v2 gates`。

## 1. 目的与边界

R5 验证 R1 action-v2、R2 reward-v3、R3 ValueNorm-Huber Critic 与 R4 GO/HOLD v2 所需指标能否通过唯一生产入口 `training.train_no_rllib_mappo_robust_v2` 在真实 CARLA 中连通。验收范围包括 rollout、PPO、metrics、schema-v4 checkpoint、受控中断、严格恢复、证据前缀裁剪和 CARLA 清理。

本次不修改 Actor/Critic 网络、奖励、学习率、PPO/GAE、ValueNorm、场景或 GO/HOLD 阈值。工程 smoke 只完成 3 个 update，每 update 4 个 episode、最多 300 steps；不能据此判断收敛、成功率或长期制动塌缩。正式 C0 v4 与 GO/HOLD v2 只接受 update 5/10 等登记停点，因此不得给本 Run 伪造正式裁决。

## 2. 运行身份与参数

| 项目 | 值 |
|---|---|
| CARLA | client/server `0.9.16`，`127.0.0.1:2000`，Town02 |
| RunID | `P0_R5_MAPPO_smoke_s42_20260811_104100` |
| Git 基线 | `00e16f3`；仅用户 `Pipfile` 与 Obsidian workspace 非关键路径有修改 |
| 初始化 | seed 42，fresh，未加载 resume/pretrained |
| 更新边界 | fresh u1；u2 episode 0 后受控中断；resume 完成 u2；resume 完成 u3 |
| smoke 规模 | 4 episodes/update，max_steps 300，min transitions 64 |
| 训练协议 | action-v2、reward-v3、ValueNorm-Huber、metrics-v5、checkpoint schema v4 |
| LR | 初始 `3e-4`，diagnostic horizon 3；u1/u2/u3 为 `3e-4/1.5e-4/1.5e-5` |
| 保存 | save interval 1，TB histogram interval 1 |

输出位于 `logs/P0_R5_MAPPO_smoke_s42_20260811_104100/`，不提交 Git，不得作为正式基础权重。

## 3. 生产训练结果

| update | episodes | agent transitions | success | collision | timeout | 说明 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | `231+231=462` | 0% | 100% | 0% | fresh 随机策略；PPO 已执行 |
| 2 | 4 | `557+557=1114` | 0% | 100% | 0% | 从 u1 完整边界恢复后执行 |
| 3 | 4 | `1088+1088=2176` | 0% | 25% | 75% | episode 更长；不作性能解释 |

0 success 不属于 R5 失败，因为本阶段没有达到正式能力里程碑。随机策略的碰撞和 timeout 只用于覆盖 terminated/truncated、active-mask、bootstrap、奖励终局和长 episode 路径。

三个 update 均满足：

- `train/update_skipped=0`，Actor/Critic 实际更新；
- 两个 Agent 的 pre-update ratio max error 均为 `0`；
- throttle/brake longitudinal conflict rate 均为 `0`；
- reward-v3 分量由生产入口记录且 step reward 守恒；
- 每 update 恰好提交 4 条 episode，active transition 与 rollout 对账；
- metrics JSONL 与 TensorBoard 同源。

动作 mode 不是确定性 C0，而是训练随机采样。Agent 0 brake mode 从 `0.489/0.515/0.559` 变化，Agent 1 为 `0.329/0.404/0.484`；同时两车都有 throttle 与 coast，不能把它解释成旧 Run 的确定性 100% brake。正式行为塌缩判断仍必须等 update 5/10 的确定性双 C0。

## 4. Critic 与 ValueNorm

| update | EV | grad clip activation | prediction std | target std | relative MAE | normalized target std | ValueNorm count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0401 | 0.50 | 0.2086 | 4.9515 | 0.7494 | 1.0000 | 1 |
| 2 | 0.0675 | 0.08 | 0.4062 | 2.9121 | 0.4996 | 0.7104 | 2 |
| 3 | 0.1965 | 0.00 | 0.3540 | 1.1513 | 0.2502 | 0.3364 | 3 |

结果说明生产入口确实执行了 R3 双尺度通路：ValueNorm 每 update 更新一次，normalized/raw target 同时记录，EV 和误差均有限。三个 update 不满足 R4 的 Critic 联合退化条件，且裁剪压力由 50% 降至 0。由于 R4 要求最早 update 10 且四信号连续五次，本 smoke 不产生正式 Critic GO/HOLD。

此外重新运行了策略无关真实 CARLA 专用诊断，报告为 `diagnostics/r2_r3_real_carla/20260811_103903/report.json`。十项接受条件全部通过：reward 守恒、stopped 为 0、smooth 高于 stopped、collision 低于 stopped、碰撞事件存在、归一化 target 单位尺度、R3 裁剪压力下降、留出 MSE 改善、timing 恢复、无车辆泄漏。

## 5. 受控中断与恢复

完成 u1 后，从 `latest.pt` 恢复进入 u2，并在 episode 0 证据落盘后触发训练器内置 `KeyboardInterrupt` 钩子。

中断后：

- `latest.pt` 与 `interrupted.pt` 都停在完整 update 1；
- 两者 SHA256 均为 `62547fb12611f96012fe53bfd08f1b175a6562db2b3d22775924188bce6b3473`；
- schema 为 4，ValueNorm update count 为 1，total episodes 为 4；
- `episodes.jsonl` 暂时包含 u1 的 4 条正式记录和 u2 的 1 条未完成前缀；
- CARLA 已恢复 async/None，vehicle/sensor 为 0/0。

再次严格恢复完成 u2 后，未完成前缀被自动丢弃，episode 记录变为恰好 u1=4、u2=4。随后恢复到 u3，最终严格验证得到 updates `[1,2,3]`、12 条 episode，无重复、缺失或越过安全边界的记录。

这验证了模型、optimizer、Python/NumPy/Torch RNG、课程状态、LR horizon、ValueNorm 和 recorder 证据边界通过生产 resume 路径恢复，而不是部分权重加载。

## 6. Checkpoint 结果

| checkpoint | update | kind | SHA256 | ValueNorm count | total episodes |
|---|---:|---|---|---:|---:|
| `interrupted.pt` | 1 | safe latest copy | `62547f…3473` | 1 | 4 |
| `periodic/ckpt_update_000001.pt` | 1 | periodic | `2451d8…bdce` | 1 | 4 |
| `periodic/ckpt_update_000002.pt` | 2 | periodic | `1d5895…aad` | 2 | 8 |
| `periodic/ckpt_update_000003.pt` | 3 | periodic | `6582b8…e8e5d` | 3 | 12 |
| `latest.pt` | 3 | latest | `3fa3a4…5e259` | 3 | 12 |

全部文件通过 checksum、schema-v4 header、完整 payload、action/model/training/critic spec、ValueNorm strict load、双 Actor optimizer ID 和 RNG 字段检查。`checkpoint_manifest.json` 的 latest/interrupted/periodic 索引与文件 SHA、update 一致。

## 7. 自动验证与 CARLA 清理

最终 strict validator：

```text
updates: [1, 2, 3]
episode_record_count: 12
required_tag_count: 197
tensorboard_scalar_tag_count: 575
status: PASS
```

文档归档前再次执行完整无 CARLA 回归集：项目 venv 未安装 `pytest`，因此按仓库既有规范运行 `..\..\..\venv\Scripts\python.exe -B -m unittest discover -s tests -v`，结果为 `Ran 179 tests in 15.287s / OK`。这是测试运行器可用性差异，不是代码或 CARLA 故障；R5 没有修改训练代码或科学参数，R1–R4、checkpoint、metrics、C0、GO/HOLD 与生命周期测试均未退化。

每次 fresh、受控中断和 resume 退出后均恢复世界设置。最终审计：

```text
map: Carla/Maps/Town02
synchronous_mode: false
fixed_delta_seconds: null
vehicle actors: 0
sensor actors: 0
```

本次没有 CARLA RPC timeout、world tick 卡死、服务器断连或 actor 清理失败，不需要用户重启服务。

## 8. 结论与下一步

R5 判定为 PASS。没有发现需要修改代码或科学参数的问题，因此不创建新的 action/reward/value/GO-HOLD 版本。R3 合同 `P0_BASE_MAPPO_TOWN02_CRITIC_V4`、C0 v4 与 GO/HOLD v2 可以作为正式 seed 42 fresh pilot 的冻结组合。

下一步只能创建新的正式 RunID，从随机初始化训练到 update 5；不得复用本 smoke checkpoint。u5 完成 strict validator 后必须运行主 C0、重复 C0 和唯一 GO/HOLD v2 judge，再根据冻结结论决定是否继续到 u10。R5 不放行多 seed、checkpoint 选模或对抗训练。
