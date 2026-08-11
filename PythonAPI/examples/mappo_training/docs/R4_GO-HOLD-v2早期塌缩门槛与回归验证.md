# R4：GO/HOLD v2 早期塌缩门槛与回归验证

> 状态：实现完成；本文件随 R4 阈值提交形成后续 fresh pilot 的预注册祖先。
> 范围：只修改裁决合同、离线判定器、C0 行为汇总和测试；不修改 Actor、Critic 网络、奖励、PPO/GAE、场景或训练超参数。
> 机器真值：`configs/evaluation/p0_base_go_hold_v2.json`。

## 1. 为什么需要 R4

历史 v1 Run 在 update 10 的 PPO 数值指标仍然有限，KL、ratio 和 clip fraction 也没有越过硬门槛，因此冻结 judge 给出 `CONTINUE_WITH_WARNING`。但是两次确定性 C0 均为 `0/16 success`、`0/16 collision`、`16/16 timeout`，两车 `brake_mode=1.0`，车辆几乎不动。v1 将 updates 5—30 定义为数值健康门槛，不能识别“以停车换取零碰撞”的行为塌缩。

R4 不负责再次改变学习机制。R1、R2、R3 已分别修复动作、奖励和 Critic；R4 只回答：现有证据是否足以继续训练，以及 HOLD 的原因是什么。

## 2. HOLD 的语义与优先级

`HOLD` 表示暂停训练并诊断，不表示算法被证明永远无法学习。触发后保留 checkpoint、日志和 C0 证据，不删除失败样本；修复或补齐证据后必须以符合合同的方式重新裁决。

判定器按以下优先级处理：

1. **证据 HOLD**：配置/checkpoint/C0/metrics 身份不一致、episode 数不足、非有限值、reward 不守恒、ratio 误差、动作冲突、重复性失败或历史行为报告缺失。
2. **数值与算法 HOLD**：KL/clip 达到冻结硬门槛、PPO update 被跳过、ValueNorm 状态不一致，或者 Critic 联合退化持续达到规定长度。
3. **行为塌缩 HOLD**：确定性停止模式的完整联合条件在两个相邻里程碑持续成立。
4. **能力 HOLD/NO-GO**：到达中后期能力节点后，success/collision/timeout/场景覆盖未达到原冻结门槛。

报告输出稳定原因码：`EVIDENCE_OR_NUMERICAL_HOLD`、`VALUENORM_STATE_INVALID`、`CRITIC_DEGENERACY`、`BEHAVIOR_HISTORY_MISSING`、`BEHAVIORAL_COLLAPSE` 和 `CAPABILITY_NOT_MET`。

## 3. 防止短训练误判

早期 `0 success`、单个 Agent 低速、单次高 brake 或单次 C0 timeout 都不能独立触发行为 HOLD。v2 使用两层保护。

第一层是联合条件。当前 C0 只有同时满足以下全部条件，才被标记为一次 `collapsed_now`：

| 信号 | 冻结条件 | 含义 |
|---|---:|---|
| 确定性 brake mode | 每个 Agent `>= 0.95` | Actor mean 几乎总是选择制动，而不是随机探索噪声 |
| 平均速度 | 每个 Agent `<= 0.1 m/s` | 约 `0.36 km/h`，属于近似静止 |
| 每回合累计正向 progress | 每个 Agent `<= 3.0 m` | Town02 目标路段为 `100 m`；100 秒 episode 只完成不超过 3% |
| timeout rate | `>= 0.95` | 至少 15/16 gate case 用满时间预算 |
| success | 必须为 0 | 避免把已经完成任务的短 episode 判成停止塌缩 |

`progress <= 3 m` 不是单独性能线。历史制动车辆因初始速度和物理滑行仍会产生约 `1.6—2.4 m` 的累计进度，因此必须与双 Agent brake、低速、timeout 和零成功共同使用。

第二层是时间持续性：

- update 5 即使满足完整联合条件，也只输出 `WARNING`；
- 行为 HOLD 最早从 update 10 开始；
- 当前和相邻上一个里程碑必须都满足完整联合条件；
- 上一个报告必须属于同一 Run、同一 v2 合同，并且正好是相邻登记停点；
- 缺少上一个报告时只能给出 `BEHAVIOR_HISTORY_MISSING`，不能宣称已经证明持续塌缩。

因此，“尚未成功但已经产生 0.5 m/s 速度和 5 m 进度”的早期策略可以继续；“两个相邻节点均为双车全制动、近静止、近零进度和全 timeout”才会行为 HOLD。

## 4. Critic HOLD

Critic 不再只看 normalized loss 或一次 explained variance。v2 在 update 5 开始观察以下四项原始尺度信号：

| 信号 | 退化条件 |
|---|---:|
| explained variance | `<= 0.02` |
| `prediction_std / target_std` | `<= 0.05` |
| `MAE / target_std` | `>= 1.0` |
| gradient clipping activation | `>= 0.95` |

至少两项同时出现只产生 warning。只有四项在至少五个连续 update 同时成立，并且当前已达到 update 10，才触发 `CRITIC_DEGENERACY`。这对应“预测近常数、误差不小、几乎每个 minibatch 都被裁剪、EV 没有拟合能力”的联合失败，避免因一次回报方差变化或早期 EV 波动误杀训练。

ValueNorm 另有结构性硬约束：`value_normalizer_update_count` 必须等于训练 update，std 不得低于训练合同的 `0.1` 方差下限；归一化 target std 长期超出 `[0.25, 4.0]` 只警告，不单独 HOLD。

## 5. 版本与证据链

GO/HOLD v2 精确绑定：

- 训练合同 `P0_BASE_MAPPO_TOWN02_CRITIC_V4` 及其 SHA256；
- C0 合同 `P0_BASE_C0_TOWN02_CRITIC_V4` 及其 SHA256；
- metrics v5、C0 episode v5、checkpoint schema v4；
- seed 42、16 个 gate case、两次离散重复性执行；
- R3 的 action-v2、reward-v3 和 ValueNorm Critic 身份由上述合同继续向下绑定。

判定报告记录当前合同 SHA、RunID、checkpoint SHA、C0 主/重复输出以及前一里程碑报告 SHA。GO/HOLD 关键源码和合同必须在正式 Run origin 中存在、内容一致且工作区干净；阈值提交必须是 Run origin 的祖先。这样可以防止查看新 pilot 后移动门槛。

v1 配置和历史报告继续按 v1 reader 读取。旧 Run 的机器裁决仍是 `CONTINUE_WITH_WARNING`，不会被改写；v2 回放只作为反事实回归测试。

## 6. 历史失败 Run 的反事实回放

测试直接读取历史主 C0 episode JSONL，而不是手工填写汇总：

| 节点 | Agent 0 | Agent 1 | 结果 |
|---|---|---|---|
| update 5 | brake `1.0`，速度约 `0.023 m/s`，progress 约 `2.32 m` | brake `1.0`，速度约 `0.021 m/s`，progress 约 `2.15 m` | 16/16 timeout；v2 WARNING |
| update 10 | brake `1.0`，速度约 `0.016 m/s`，progress 约 `1.61 m` | brake `1.0`，速度约 `0.024 m/s`，progress 约 `2.32 m` | 16/16 timeout；v2 `HOLD/BEHAVIORAL_COLLAPSE` |

这说明 v2 能在 update 10 识别旧协议遗漏的失败模式，同时保留 update 5 的学习宽限期。

## 7. 真实 CARLA success smoke

2026-08-11 在用户维护的 `127.0.0.1:2000` CARLA 0.9.16 / Town02 上完成两层复验：

- teleport：两车分别在 step 1/2 到达，终局为 `all_agents_reached`；
- scripted：固定 throttle 通过真实车辆动力学行驶，两车分别在 step 32/55 到达；
- 两次均无碰撞，成功 Agent 为 `terminated=true`、`truncated=false`；
- 两次均通过 `success_protocol_pass` 和 `clean_run`；清理后恢复异步/`fixed_delta_seconds=None`，vehicle/sensor actor 为 `0/0`。

证据分别位于 `logs/r4_success_smoke_teleport_20260811/result.json` 和 `logs/r4_success_smoke_scripted_20260811/result.json`。日志不提交 Git。

该历史 smoke 入口默认使用旧 success 奖励口径，输出中的 `success=100` 和 `time_bonus` 只用于其既有生命周期断言，不能当作 reward-v3 数值证据。R2-v3 的真实守恒、stopped、smooth success 和 collision 排序仍以 R3 专用诊断为准；R4 没有修改奖励或环境终局逻辑。

## 8. 验收边界

R4 完成只证明裁决器能区分证据错误、数值退化、持续停车塌缩和能力不足，不证明 R1—R3 修复后的策略一定收敛。下一步仍需 R5：用最终入口完成 action/reward success smoke、1—3 update 工程 smoke、checkpoint/resume 和证据链验证。全部通过后，才能用新 Git 祖先、新 RunID 和随机初始化启动 seed 42 fresh pilot；多 seed 与对抗训练仍未放行。
