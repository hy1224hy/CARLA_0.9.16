# P0-BASE Town05 v3 seed 42：u5 正式训练与 GO/HOLD 记录

> 日期：2026-08-14
>
> RunID：`P0_BASE_MAPPO_Town05V3_s42_20260814_125226`
>
> 机器裁决：`CONTINUE_WITH_WARNING`
>
> 放行边界：只允许同一 Run 严格恢复到下一登记停点；不放行多 seed、选模、基础策略池或对抗训练。

## 1. 实验身份与目的

- 训练来源：Git `5dad4d4f84412d76a7d4316fd85778c660f4f8df`，合同 `P0_BASE_MAPPO_TOWN05_V3`，SHA256 `d39956e0ee1021d7e6d27057bc0a9877edc9dbee0bf9f6ebbf1224f9a5a08e92`。
- 初始化：seed 42、fresh 随机初始化；没有 `--resume`，没有加载历史或诊断 checkpoint。
- 环境：CARLA client/server `0.9.16/0.9.16`，Town05，固定步长 `0.1 s`。
- 训练规模：5 updates，每 update 16 episodes，`max_steps=500`；训练自然停止在 u5，未启动 u6。
- 本阶段只回答训练链路和早期学习健康，不用 5 个 update 声称算法已经收敛。

## 2. 训练与 strict validator

训练完成 80 episodes。各 update 的训练结局如下：

| update | success | collision | offroad | timeout | transitions | Critic loss | Critic EV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0/16 | 1/16 | 12/16 | 3/16 | 9544 | 0.5116 | -0.0041 |
| 2 | 0/16 | 1/16 | 9/16 | 6/16 | 10886 | 0.3901 | 0.0074 |
| 3 | 0/16 | 1/16 | 5/16 | 10/16 | 12342 | 0.2448 | 0.0021 |
| 4 | 0/16 | 2/16 | 3/16 | 11/16 | 12948 | 0.2488 | 0.0182 |
| 5 | 0/16 | 1/16 | 2/16 | 13/16 | 14608 | 0.1263 | 0.0231 |

训练期间 PPO 没有跳过，pre-update ratio error 为 0，reward step residual 为 0，无缺字段或非有限值；offroad 终止均带连续 3 tick 和首证据，未发现车辆进入非 Driving 区域后继续写 rollout 的情况。offroad 比例下降是早期趋势，不足以证明策略学会安全驾驶。

周期 checkpoint：`logs/P0_BASE_MAPPO_Town05V3_s42_20260814_125226/periodic/ckpt_update_000005.pt`，SHA256 `cb79242e0a4998dfd18bcd62856cb590fe478f481b81e1101bb6f28a0c410a1a`。

strict validator：updates `1..5`、80 条 episode 记录、208 个必需指标和 682 个 TensorBoard scalar 标签全部 PASS。训练和两次 C0 退出后均恢复异步模式、`fixed_delta=None`、车辆/传感器 `0/0`。

## 3. C0 v3 入口接线问题与修复

首次正式 C0 在创建 CARLA rollout 前被生产入口拒绝。根因是 `base_c0_protocol.py`、v3 合同和测试已支持 `p0_base_c0_eval_town05_v3`，但 `evaluate_base_policy_c0.run_evaluation()` 的正式可执行协议集合漏加对应常量。

修复没有改变 Actor/Critic、reward、PPO/GAE、场景、训练合同或 checkpoint：

- 将 Town05 v3 常量加入显式正式可执行集合；
- 增加防漏接回归测试；
- 专项 17/17、全量 211/211 通过；
- 修复提交：`a33a573`；
- 修复后三合同 critical-source preflight PASS；训练关键源码与 run-origin 一致，GO/HOLD 的预注册 blob 检查 PASS，因此本 checkpoint 无需重训。

失败的首次 EvalID 没有生成正式产物，也没有进入裁决。

## 4. 双 C0 v3

- 主 C0：`P0_BASE_C0_gate_u000005_20260814_132002`
- 重复 C0：`P0_BASE_C0_gate_u000005_20260814_133029`
- 合同 SHA256：`0ddf0efa3b00f170721f35f7e32878fae67d5f6c88cb954f6edafd164de38553`

两次各覆盖 32/32 场景，identity/hash、固定 case 顺序、episode schema、reward conservation、summary 重建和清理审计均 PASS。主 C0 结果：

- success `0/32`、collision `2/32`、offroad `0/32`、timeout `30/32`；
- lane invasion `8/32`；
- agent0/1 的 deterministic brake mode 均为 `1.0`；
- agent0：平均速度 `0.0466 m/s`，平均 progress `2.291 m/episode`；
- agent1：平均速度 `0.0781 m/s`，平均 progress `3.545 m/episode`。

结果说明 u5 策略尚未学会有效前进，更不能视为收敛模型；但没有确定性 offroad，且冻结的联合行为 HOLD 条件尚未满足。

重复性比较：安全、结局、end reason、success/collision/offroad、T/X/done、active-mask 生命周期和 offroad 证据存在性完全一致，无 hard mismatch。13/32 case 存在 return 漂移，最大约 `0.0364 < 0.1`；warning case fraction `40.625% < 50%`，因此为 `WARNING / gate_pass=true`，不是系统性 HOLD。

## 5. GO/HOLD v3 与结论

裁决：`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260814_134018/go_hold_decision.json`。

机器输出为 `CONTINUE_WITH_WARNING`：

- 训练健康、metrics、checkpoint、C0 完整性、重复性语义 gate 和预注册来源均 PASS；
- agent0 满足 brake dominant、near stationary、progress 不足三项；agent1 的 progress `3.545 m > 3 m`，不满足 all-agent collapse；
- timeout `93.75% < 95%`，也未满足 episode 联合条件；
- u5 没有上一里程碑，冻结规则要求从 u10 起连续两里程碑才可形成持久行为 HOLD；
- warnings：8 个 C0 case 有 lane invasion；13 个 case 有预注册范围内的 return 漂移。

该结论只允许保留同一 Run 身份继续下一登记停点。它不等于算法收敛、`PILOT_GO`、多 seed GO、基础策略池入池或对抗训练 GO。下一步如继续，应严格 resume 到 u10，立即停止并再次执行 strict validator、双 C0 v3 和 GO/HOLD v3，同时提供本次 u5 decision 作为上一里程碑证据。
