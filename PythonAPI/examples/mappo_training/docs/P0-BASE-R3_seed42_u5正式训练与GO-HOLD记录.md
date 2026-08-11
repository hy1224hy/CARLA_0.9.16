# P0-BASE-R3 seed 42 update 5 正式训练与 GO/HOLD 记录

> 执行日期：2026-08-11
> RunID：`P0_BASE_MAPPO_R3_Town02_s42_20260811_112458`
> 结论：`CONTINUE_WITH_WARNING`
> 安全训练边界：update 5；未启动 update 6
> 下一里程碑：在系统盘空间恢复后，以同一 RunID 从 u5 严格恢复到 u10；不得自动续训或改变科学参数

## 1. 本阶段目的与冻结输入

本阶段不是用 5 个 update 证明策略已经收敛，而是验证修复后的 R1 二维动作、R2 reward v3、R3 ValueNorm/Huber Critic 和 R4 GO/HOLD v2 能否在真实 CARLA 正式入口中形成完整、可追溯、可继续或可停止的首个里程碑证据。训练严格停在 u5 后，必须依次完成日志验证、主 C0、重复 C0 和唯一 GO/HOLD 裁决；在裁决前不得进入 u6。

| 身份项 | 冻结值 |
|---|---|
| 训练合同 | `P0_BASE_MAPPO_TOWN02_CRITIC_V4` |
| 训练合同 SHA256 | `1cab6f44faf65daae2cca22fb806c425d81fe386cbc625d51e30339170dac2c1` |
| C0 合同 | `P0_BASE_C0_TOWN02_CRITIC_V4`，SHA256 `7fdb797b704b9ff53587a529c57e68eb9659b3043921dad9312aad60e66dcc68` |
| GO/HOLD 合同 | `P0_BASE_GO_HOLD_TOWN02_R4_V2`，SHA256 `668799076a7e08bfd37457457929df3f1bbf40fbc8ce64ef068f97e839925140` |
| seed / source | `42` / fresh 随机初始化；未使用 resume、pretrained 或 R5 smoke 权重 |
| Run origin | Git `ff5ddebbea7e36a2d6d080e827d1930c245ff586`；critical-source fingerprint `d4d40c8095260302fc9a88fafda9ea73ec6201c16862a98d23d2d803ed9ad776` |
| CARLA | client/server `0.9.16`，Town02，`127.0.0.1:2000` |
| 训练设备 | 显式 `--cpu`；CARLA 使用 GPU |

实际命令：

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --host 127.0.0.1 `
  --port 2000 `
  --experiment-contract configs/base_mappo/p0_base_critic_v4.json `
  --run-id P0_BASE_MAPPO_R3_Town02_s42_20260811_112458 `
  --updates 5 `
  --cpu `
  --save-dir logs/P0_BASE_MAPPO_R3_Town02_s42_20260811_112458
```

训练约 1304.1 秒，进程退出码为 0，并在 u5 自然结束；不存在 u6 指标、episode 或 checkpoint。

## 2. u1–u5 训练结果

| update | episodes | agent transitions | success | collision | timeout | 平均长度 | A0/A1 平均回报 | Critic EV | Value MAE | ValueNorm count |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 16 | 4,124 | 0% | 100% | 0% | 128.88 | `-3.588 / -19.122` | -0.0097 | 1.9529 | 1 |
| 2 | 16 | 6,612 | 0% | 100% | 0% | 206.63 | `-4.902 / -21.032` | 0.0056 | 1.4819 | 2 |
| 3 | 16 | 9,978 | 0% | 93.75% | 6.25% | 311.81 | `-5.702 / -21.035` | 0.0197 | 1.0962 | 3 |
| 4 | 16 | 23,978 | 0% | 43.75% | 56.25% | 749.31 | `-7.371 / -21.308` | 0.0687 | 0.2534 | 4 |
| 5 | 16 | 27,706 | 0% | 37.5% | 62.5% | 865.81 | `-9.527 / -19.094` | 0.0922 | 0.1474 | 5 |

训练健康审计结果：

- 5 个 update 均执行 PPO，`train/update_skipped=0`；双 Actor 的 pre-update ratio error 始终为 0，纵向油刹冲突率为 0。
- reward step residual 始终为 0，最大 episode residual 为 `1.85e-13`；无缺字段或非有限值。
- Critic EV 从 `-0.0097` 改善到 `0.0922`，Value MAE 从 `1.9529` 降至 `0.1474`；GO/HOLD 的 Critic 四信号联合退化未成立。
- strict validator：updates `[1,2,3,4,5]`、80 条 episode、197 个必需标签、575 个 TensorBoard scalar 标签，`PASS`。

周期检查点为：

```text
logs/P0_BASE_MAPPO_R3_Town02_s42_20260811_112458/periodic/ckpt_update_000005.pt
SHA256 4652aa5979575db0799bc70ba3fc9838efaf0c28476f35f87ecb38337e815e72
```

其 schema-v4 header、checksum、periodic kind、seed、Run/Git/合同身份、双 Actor/Critic/optimizer、RNG 和 ValueNorm strict load 均通过。检查点记录 80 episodes、0 success、60 collision、20 timeout；ValueNorm count 为 5，反归一化均值/标准差为 `-2.7605 / 2.4029`。

## 3. 双 C0 v4

两次评测使用完全相同的 u5 periodic checkpoint、gate profile、16 个冻结病例和确定性 Actor mean。第二次只用于重复性审计，不计作额外性能样本。

| 角色 | EvalID | success | collision | timeout | 独立 validator | 摘要 SHA256 |
|---|---|---:|---:|---:|---|---|
| 主 C0 | `P0_BASE_C0_gate_u000005_20260811_115047` | 0/16 | 5/16 | 11/16 | PASS | `7364981e1486cfb2932c2c9ae0336ac68bc782c63ce3123b874c53628befed1e` |
| 重复 C0 | `P0_BASE_C0_gate_u000005_20260811_120044` | 0/16 | 5/16 | 11/16 | PASS | `ab8aa28e393f3a4044aad156168fc74cbd951fe1223b452464ae88c8de21d7d4` |

两轮的冻结 case 顺序、`outcome`、终局原因、episode length 和两车 reached 标志逐病例完全一致，repeatability 为 `PASS`、mismatch 为空。两次评测结束后的 CARLA 均恢复为 async、`fixed_delta_seconds=None`、vehicle/sensor `0/0`。

主 C0 的行为摘要：

| Agent | brake mode | throttle mode | mean speed | 每 episode 平均 progress | R4 单车塌缩判定 |
|---:|---:|---:|---:|---:|---|
| 0 | 1.0 | 0.0 | `0.0350 m/s` | `2.4963 m` | true |
| 1 | 1.0 | 0.0 | `0.0714 m/s` | `4.7130 m` | false：progress 高于 3 m 阈值 |

这说明 u5 的确定性策略仍明显偏向制动、尚未形成到达能力，但不能据此绕过预注册规则直接宣判 HOLD。

## 4. GO/HOLD v2 结果与解释

裁决产物：

```text
decisions/base_mappo/P0_BASE_GO_HOLD_u000005_20260811_120658/go_hold_decision.json
SHA256 c031da99670fd28538305fac3e911466ec54ed80da1cb6c68b904495f4bea7d1
```

机器结论为 `CONTINUE_WITH_WARNING`：

- 来源预注册、critical-source Git blob、训练证据、checkpoint、主/重复 C0 和重复性全部 `PASS`。
- 无 hard failure、capability failure、Critic failure 或行为塌缩 reason code；多 seed 未放行。
- 唯一正式 warning：主 C0 的 16 个 gate episode 中有 9 个发生 lane invasion。
- R4 行为塌缩的当前联合判定为 false。虽然两车 brake mode 都为 1.0、速度都低于 `0.1 m/s` 且 success 为 0，但 agent1 progress `4.7130 m > 3 m`，整体 timeout `0.6875 < 0.95`，没有满足“所有 Agent + episode 信号”联合条件；u5 也没有可供持续性判断的前一里程碑。
- 这正是 R4 防止短训练误杀的效果：保留明显不理想的行为证据，但不把首个里程碑上的未学习完成误报成持续性塌缩。

## 5. 运行异常与 CARLA 状态

训练、两次 C0 和裁决期间没有 CARLA RPC timeout、断连或卡死，因此无需重启 CARLA。所有阶段结束后均确认 Town02 为 async、vehicle/sensor `0/0`。

第一轮 C0 结束后，系统盘 `C:` 变为 0 字节可用，导致一次工具侧验证启动失败；项目盘 `D:` 当时仍约有 33.3 GB。经用户授权，删除了精确目录 `C:\Users\hy\AppData\Local\Temp\baidu` 中约 449 MB 的自动更新临时缓存，验证得以继续；该缓存不经过回收站。随后后台程序又把 C 盘占满，但正在 D 盘写入的第二轮 C0、validator 和 judge 均完整结束且证据通过。

该磁盘异常不改变本次科学结论，但会阻塞或危及下一训练段。启动 u6–u10 前必须由用户把 C 盘恢复到稳定的非零余量，并检查占盘后台程序；不能把临时释放的数百 MB 当作长期训练容量。

## 6. 下一步唯一动作

当前允许但尚未执行的动作是：在用户确认 C 盘空间已稳定释放后，以同一 RunID 和同一冻结合同从 `latest`/u5 完整边界严格恢复到 u10。不得新建 fresh Run、复用其他 checkpoint、修改 reward/Actor/Critic/PPO/场景/阈值或自动放行多 seed。

u10 结束后仍须立即停止，重新执行主 C0、重复 C0，并在 GO/HOLD v2 judge 中显式提供本次 u5 报告作为 `--previous-decision-report`。只有 u10 新裁决完成后，才能决定是否继续下一个里程碑；本次 `CONTINUE_WITH_WARNING` 不等于模型合格、PILOT_GO、多 seed GO 或对抗训练 GO。
