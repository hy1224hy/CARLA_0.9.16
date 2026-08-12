# P0-BASE Town05 seed 42：u5 正式训练与 GO/HOLD 记录

> 日期：2026-08-12
>
> RunID：`P0_BASE_MAPPO_Town05_s42_20260812_164411`
>
> 最终机器裁决：`HOLD`
>
> 处置：停止于 update 5；禁止运行 update 6、续到 update 10、多 seed、选模或 Phase 1，先诊断双 C0 的逐 tick 重复性。

## 1. 本次实验要回答的问题

本次是 Town05 冻结协议下的首个正式单 seed 工程 pilot，不以五个 update 证明收敛，而是回答三个更早的问题：

1. 正式训练入口能否在真实 CARLA 中连续完成 update 1–5，并留下可严格恢复、可审计的训练证据；
2. 冻结的 u5 检查点在独立、确定性 C0 gate 中究竟表现为有效行进、制动/静止退化，还是危险碰撞；
3. 同一检查点、同一场景和同一评估 seed 的两次 C0 是否满足预注册的逐病例离散重复性门槛，从而允许继续 pilot。

这里的 u5 是安全停点，不是基础模型能力验收点。只有 update 100 的 `PILOT_GO` 才能放行多 seed；多 seed 训练与独立验证后形成的是基础策略候选集，只有通过安全、能力、来源和哈希登记的检查点才可进入后续基础策略池/PSRO 流程。

## 2. 冻结输入身份

| 项目 | 值 |
|---|---|
| Git commit | `6c28a9141d3ede87f7f54734cc277973381e6f05` |
| 训练合同 | `P0_BASE_MAPPO_TOWN05_V1` |
| 训练合同 SHA256 | `eeb12304d9a262ac4297ede90bd651717d7d59cb31cb8ba0f384835e0ed84bde` |
| C0 合同 | `P0_BASE_C0_TOWN05_V1` |
| C0 合同 SHA256 | `ea2e17f633e3b26eaedbbcf5f42cbaf9b14bc7077790e3dfcba438c0c276cab8` |
| GO/HOLD 合同 | `P0_BASE_GO_HOLD_TOWN05_V1` |
| GO/HOLD 合同 SHA256 | `bbe6903c1ba9fb5932eafc0ba4dd250f0f6254d2e9f9066d0da4c5d940c7e9c4` |
| 初始化 | fresh、随机初始化，未加载 pretrained，未 resume |
| seed / 停点 | `42` / update `5` |
| 训练规模 | 16 episodes/update，max 500 steps，5 PPO epochs，单全批 `32768` |
| 折扣 | PPO/GAE 与环境 PBRS 均为 `gamma=0.997` |
| CARLA | client/server `0.9.16/0.9.16`，`Town05`，`127.0.0.1:2000` |
| 训练场景池 | `data/training_scenarios_town05_v1`，离线 16/16 PASS |
| C0 gate 场景池 | `data/training_scenarios_town05_v1`，冻结 16 cases；另有 held-out validation 池 `data/evaluation_scenarios/town05_c0_validation_v1` 离线 8/8 PASS，本次 u5 health gate 未使用 held-out 池选模 |

启动前训练、C0、GO/HOLD 三份合同的 critical-source clean preflight 均通过。工作树中已有的 `AGENTS.md`、`Pipfile` 和 Obsidian workspace 改动不属于合同关键源码，本次未改写或纳入实验提交。

## 3. 正式命令与训练产物

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --host 127.0.0.1 --port 2000 `
  --experiment-contract configs/base_mappo/p0_base_town05_v1.json `
  --run-id P0_BASE_MAPPO_Town05_s42_20260812_164411 `
  --updates 5 --cpu --follow-spectator `
  --save-dir logs/P0_BASE_MAPPO_Town05_s42_20260812_164411
```

训练约 1336.8 秒后自然结束于 update 5，没有启动 update 6。共完成 80 episodes、65178 个 agent-transitions；周期检查点为：

`logs/P0_BASE_MAPPO_Town05_s42_20260812_164411/periodic/ckpt_update_000005.pt`

其 SHA256 为 `b1c93b14d731ffc62e57ec294e09ff9332f1b358de6f8ea7b2b9569a336eeb23`。checkpoint 的 origin/last Git、合同、fresh 来源、update、训练 gamma、环境 reward gamma、模型、optimizer、RNG 和 ValueNorm 状态均通过严格加载审计。

## 4. update 1–5 训练健康结果

| update | success | collision | timeout | mean length | transitions | Actor KL（a0/a1） | Critic EV |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 0/16 | 12/16 | 4/16 | 314.44 | 10062 | 0.00413 / 0.00604 | -0.0099 |
| 2 | 0/16 | 12/16 | 4/16 | 324.00 | 10368 | 0.00278 / 0.00088 | -0.0016 |
| 3 | 0/16 | 4/16 | 12/16 | 419.25 | 13416 | 0.00243 / 0.00222 | 0.0079 |
| 4 | 0/16 | 2/16 | 14/16 | 491.56 | 15730 | 0.00077 / 0.00056 | -0.0070 |
| 5 | 0/16 | 1/16 | 15/16 | 487.56 | 15602 | 0.00203 / 0.00149 | 0.0443 |

严格训练校验结论为 `PASS`：updates `1..5` 连续、80 条 episode 记录完整、197 个必需指标和 575 个 TensorBoard scalar 标签齐全；两车 PPO 均未跳过，pre-update ratio error 为 0，reward step residual 为 0，无 missing/nonfinite。ValueNorm update count 为 5，最终 mean `-1.7579`、std `1.7863`。

训练 rollout 中碰撞率下降、超时率上升且成功仍为零，只能作为需要 C0 解释的趋势，不能单凭训练收益判断策略已经安全或有效。

## 5. 两次独立 C0 gate

| 项目 | 主 C0 | 重复 C0 |
|---|---|---|
| EvalID | `P0_BASE_C0_gate_u000005_20260812_170923` | `P0_BASE_C0_gate_u000005_20260812_171236` |
| cases | 16 | 16 |
| success / collision / timeout | 0 / 12 / 4 | 0 / 12 / 4 |
| evaluator 完整性 | PASS | PASS |
| identity/schema/reward/summary/cleanup | 全部 PASS | 全部 PASS |

主 C0 的冻结确定性行为摘要：

| Agent | brake mode | mean speed | mean progress/episode | 解释 |
|---:|---:|---:|---:|---|
| 0 | 1.0000 | 0.1588 m/s | 2.8379 m | 确定性策略全程选择制动；低位移，但速度尚未低到 R4 的 `<=0.1 m/s` 联合条件 |
| 1 | 0.0036 | 0.8026 m/s | 10.2560 m | 几乎全程 coast，不是双车共同静止/制动坍缩 |

四个 `parallel_stationary` case 全部跑满 500 steps 后 timeout；其余 rolling、rear-closing 和 required-merge case 共 12 个，全部以 collision 结束。两轮宏观结局、终止原因、成功标志及每个场景的结局均一致，说明 u5 策略能力确实不足，而不是一次评估偶然得到了差的成功率。

## 6. 重复性失败与正式裁决

双 C0 的严格比较要求以下离散字段逐病例完全一致：`case_id`、`outcome`、`episode_end_reason`、`episode_length`、两车 reached 标志。两轮有 5/16 个 collision case 仅 `episode_length` 不同：

| case | 主 C0 | 重复 C0 | 差值 |
|---|---:|---:|---:|
| scenario 07 east parallel rolling | 74 | 73 | 1 tick |
| scenario 08 west parallel rolling | 99 | 98 | 1 tick |
| scenario 09 north stagger closing | 83 | 81 | 2 ticks |
| scenario 10 south stagger closing | 16 | 17 | 1 tick |
| scenario 15 east required merge | 23 | 24 | 1 tick |

虽然五例的 `outcome=collision`、`episode_end_reason=cooperative_collision`、两车 reached 标志完全相同，但预注册协议没有允许 episode length 容差，因此重复性审计必须为 `HOLD`，不能在看见结果后人工放宽。

正式报告：

`decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260812_171550/go_hold_decision.json`

机器裁决为：

- `training.health = PASS`：PPO、Critic、ValueNorm、指标和 checkpoint 没有数值硬失败；
- `behavior_assessment = PASS`：u5 的 R4 行为坍缩条件没有满足，故不是因为短训练零成功而被误判；
- `C0 repeatability = HOLD`：唯一 hard failure 为 `C0 discrete repeatability did not PASS`；
- `decision = HOLD`，`multiseed_released = false`；
- 附加 warning：主 C0 中 8/16 episode 出现 lane invasion。

因此，本次 HOLD 是证据/复现门禁，而不是“u5 没学会成功所以立即判算法失败”，也不是 Critic 网络数值失败。与此同时，0/16 成功、12/16 碰撞和静止场景全部超时仍然是明确的早期能力不足证据，不能因重复性问题而忽略。

## 7. CARLA 与清理

训练和两次 C0 均未发生 RPC timeout、服务断连或卡死。每段结束后均恢复 Town05 异步模式、`fixed_delta_seconds=None`，并清理为 vehicle/sensor `0/0`。`Town05.bin` 导航缓存缺失警告不影响本次无行人车辆评测和已生成的正式证据。

## 8. 唯一下一步

1. 冻结本次 Run、checkpoint、两次 C0 与 GO/HOLD 报告，不运行 update 6，不改变训练科学参数；
2. 对上述 5 个 mismatch case 做最小化重复回放，记录碰撞 sensor callback、world frame、episode step、初始 transform/velocity 和终止提交顺序；
3. 先判断漂移来自 CARLA 物理/碰撞回调的正常 1–2 tick 非确定性，还是 evaluator 的 seed、同步帧或事件消费存在缺口；
4. 若是实现缺口，修复后补回归并以新 C0 协议/新身份重新评测；若是不可消除的仿真器噪声，则在不查看新模型性能的前提下预注册有科学依据的重复性口径，旧报告保持 HOLD；
5. 只有新的双 C0 和 GO/HOLD 通过后，才决定是否允许同一 Run 严格恢复至 update 10。多 seed 与基础策略池仍需等待 update 100 的 `PILOT_GO`。
