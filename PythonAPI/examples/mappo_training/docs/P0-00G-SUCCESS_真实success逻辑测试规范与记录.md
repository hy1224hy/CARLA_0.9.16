# P0-00G-SUCCESS 真实 success 逻辑测试规范与记录

> 当前状态：PASS / FROZEN BY P0-BASE-00 (`345ea04`)
> 建立日期：2026-08-02
> 裁决范围：验证固定场景中的终点区域、成功奖励和 episode 生命周期；不评价 MAPPO 收敛或基础策略能力。

## 1. 为什么不先训练基础策略

“环境能否正确识别 success”和“策略能否学会到达终点”是两个不同问题。若先用未验收的训练策略触发终点，失败可能来自策略未收敛、探索噪声、车辆控制、场景难度或 success 实现，无法归因。

因此本测试不加载 Actor 或 checkpoint，先用确定性干预建立 success 逻辑证据。只有此门槛通过后，正式训练产生的成功率才有可解释的指标基础。训练好的基础策略仍是后续 C0 能力评估的必要条件，但不是本逻辑测试的前置条件。

## 2. 两层测试与固定顺序

测试入口：`evaluation/smoke_test_success_logic.py`。

1. `teleport`：在真实 `CarlaMultiAgentEnv.step()` 前，每次只把一辆仍 active 的车顺序移入终点中心。它隔离验证终点判定、奖励和生命周期，不验证驾驶能力。
2. `scripted`：把两车放到同一固定终点前方，首车距终点近边界 5 m，后车再落后 10 m；两车执行固定低油门直行。它额外验证短距离车辆运动、先到车辆消失以及后车继续到点。

人工键盘驾驶不作为强制验收方法：人的操作和到点时机不可重复，适合失败后的视觉定位，不适合作为唯一通过证据。`--follow-spectator` 可把 CARLA 观察相机移到固定俯视位置。

## 3. PASS 门槛

同一个两车回合必须全部满足：

- reset 后两车均 `active=True`，且 `terminated=False`、`truncated=False`、全局原因为 `running`；
- 每辆车首次进入 goal zone 的同一步，`reward_details.success > 0` 且 `reward_diagnostics.success_event == 1`；
- 到点车辆必须 `done=True`、`terminated=True`、`truncated=False`、`active=False`；
- 第一辆车到点后，另一辆车仍 active，回合仍为 `running`；
- 两辆车全部到点后，全局原因严格为 `all_agents_reached`，全部 agent 关闭；
- 整条 success 路径没有 collision/crosswalk 事件；
- CARLA 清理成功。

`teleport` 和 `scripted` 必须分别取得 `result.json` 的 `status=PASS`，才关闭固定场景 success 覆盖缺口。任一模式失败都必须保留输出目录，不得通过放宽上述门槛改成 PASS。

## 4. 冻结命令

在 `PythonAPI/examples/mappo_training/` 目录运行。CARLA 0.9.16 必须由用户先启动并监听 `127.0.0.1:2000`。

```powershell
# A. 纯 success/reward/lifecycle 隔离测试
..\..\..\venv\Scripts\python.exe -B -m evaluation.smoke_test_success_logic `
  --mode teleport `
  --scenario data/training_scenarios/scenario_01_basic.json `
  --seed 42 `
  --follow-spectator

# B. 近终点、低油门、无策略短距离行驶测试
..\..\..\venv\Scripts\python.exe -B -m evaluation.smoke_test_success_logic `
  --mode scripted `
  --scenario data/training_scenarios/scenario_01_basic.json `
  --lead-distance 5 `
  --following-gap 10 `
  --throttle 0.30 `
  --max-steps 250 `
  --seed 42 `
  --follow-spectator
```

默认每次创建唯一目录 `logs/success_logic_smoke_<timestamp>/`：

- `effective_scenario.json`：本次真实应用的近终点场景；
- `steps.jsonl`：每一步的 success、终止/截断、active mask、碰撞和位置；
- `result.json`：最终 PASS/FAIL/ERROR、配置、逐车成功证据和错误栈。

## 5. 失败归因规则

| 观察 | 可以支持的判断 | 不能支持的判断 | 最小下一步 |
|---|---|---|---|
| `teleport=PASS`、`scripted=PASS` | 当前固定场景的 success/reward/lifecycle 和短距离直行链路成立 | MAPPO 已收敛或远距离成功率达标 | 进入 P0-BASE-00；冻结配置/C0 后做单 seed 分段试训 |
| `teleport=PASS`、`scripted=FAIL` | success 核心逻辑成立；问题位于出生/运动/道路/碰撞/步数一侧 | success 逻辑错误 | 只分析 scripted 的 `steps.jsonl` 和 CARLA 画面 |
| `teleport=FAIL` | 当前证据不足以信任 success 指标 | 失败由策略训练不足造成 | 按首个违反的 reward/lifecycle 字段修复并重跑 teleport |
| `ERROR` 且无法连接 2000 | 测试未执行 | success 通过或失败 | 用户手动重启 CARLA 后原命令重跑 |
| success 同步出现 collision/crosswalk | 路径被终局混杂污染 | 干净成功已验证 | 检查 goal zone 与横道/碰撞事件，不接受该次 PASS |

## 6. 运行记录

| 时间 | 层级 | 命令/环境 | 结果 | 证据与裁决 |
|---|---|---|---|---|
| 2026-08-02 | 无 CARLA 单元与全量回归 | `python -B -m unittest discover -s tests` | PASS | success 专项 8 项通过；最终全量 93 项通过，4.430 s。只证明测试器协议与几何构造，不替代真实 CARLA。 |
| 2026-08-02（程序完成后） | 首次环境可用性 | 进程与 `:2000` 监听只读检查 | NOT AVAILABLE（历史） | 当时 `CARLA_PROCESS=NONE`、`PORT_2000_LISTENER=NONE`；未启动、未 kill CARLA，故等待用户手动启动。随后 22:24 的重启后 preflight 已通过。 |
| 2026-08-02 22:24:43 | 重启后 preflight | CARLA `127.0.0.1:2000` | PASS | PID 10392 监听；client/server 均为 0.9.16；初始 Town10HD_Opt、异步、`fixed_delta=None`；vehicle/sensor 为 0/0。 |
| 2026-08-02 22:24:56–22:25:07 | real CARLA / teleport | 冻结命令 A；Run `P0_00G_SUCCESS_teleport_s42_20260802_222443` | PASS | 两车分别在 step 1/2 success；每车 `success=100`、`success_event=1`、terminated 且不 truncated；最终 `all_agents_reached`；collision 0；清理后 actor 0/0、异步/None。证据：[result.json](../logs/P0_00G_SUCCESS_teleport_s42_20260802_222443/result.json)、[steps.jsonl](../logs/P0_00G_SUCCESS_teleport_s42_20260802_222443/steps.jsonl)。 |
| 2026-08-02 22:25:35–22:25:47 | real CARLA / scripted | 冻结命令 B；Run `P0_00G_SUCCESS_scripted_s42_20260802_222523` | PASS | 低油门直行中 agent0/1 分别在 step 32/55 success；每车 `success=100`、`success_event=1`；先到车退出后后车保持 active；最终 `all_agents_reached`；collision 0；清理后 actor 0/0、异步/None。证据：[result.json](../logs/P0_00G_SUCCESS_scripted_s42_20260802_222523/result.json)、[steps.jsonl](../logs/P0_00G_SUCCESS_scripted_s42_20260802_222523/steps.jsonl)。 |
| 2026-08-02 22:26 | 最终 CARLA 污染检查 | 同一服务连续两次运行后 | PASS | Town02、异步、`fixed_delta=None`、vehicle/sensor 0/0；未 kill 或重启 CARLA。 |

## 7. 结果与问题分析

### 已确认事实

- `teleport` 在真实环境的两个连续 step 中分别触发两车 success，证明矩形 goal zone、`SUCCESS_REWARD=100`、`success_event`、terminated/truncated、active mask 和全局终局能够形成一致闭环。
- `scripted` 不是传送：两车从 `y=256/246` 以固定 throttle 0.30 行驶。step 31 时 agent0 位于 `y=260.9079`、尚未成功；step 32 入区并退出。step 54 时 agent1 位于 `y=260.9331`、尚未成功；step 55 入区并退出。终点近边界是 `y=261`，观测与矩形几何一致。
- agent0 在 step 32 到点后，agent1 从 step 33 至 54 一直保持 `active=True`、`terminated=False`，说明先到车辆的 disappear-at-target 没有误终止或物理阻塞后车。
- 两次 Run 连续使用同一个 CARLA 服务，均独立恢复世界 timing 并清空车辆/传感器。当前没有“只有第一次运行正确”或“第二次 CARLA 卡死”的证据。
- 两次 `effective_scenario.json` 的 SHA256 同为 `6838c6706187b7caa34560204959bc42b1d28a65cbe047a3624c34e37ea54530`，证明两种模式使用同一冻结几何。teleport 的 `result/steps` SHA256 为 `fc84943b4faa95a6259d614d530b340c84a414d1369ec21dbf1ec8a5824cf038` / `04774d73f624c19f9a718423b7e99dd9049f9a0e6b5fd93b60248704a488365a`；scripted 为 `8bcb02218ec41014d0986c8c1c033807d34ab7dc80702e34653b71b50cc21984` / `16e4046256a8868e0305a3cee298078181481f8942e6104fc7fcda1ac4853e23`。

### 合理推断

- 先前训练 smoke 没有 success，主要不能被归因为 success 判定逻辑损坏；更合理的候选是随机未训练策略的驾驶能力、远距离任务难度以及碰撞/超时分布。
- 本次只能说明当前服务器在这两个短测试中没有卡死，不能追溯证明此前某次历史运行一定没有服务器卡顿。

### 尚未验证

- 未验证 MAPPO 收敛、原固定长距离场景成功率、跨 seed 鲁棒性、C0/C1 基线能力或正式 checkpoint 质量。
- `teleport` 是逻辑隔离证据，`scripted` 是短距离集成证据；两者都不能替代训练后独立评估。

## 8. 当前 Judge 裁决

**PASS / FROZEN BY P0-BASE-00 (`345ea04`)**：策略无关的纯逻辑层和近终点集成层均在真实 CARLA 中无污染 PASS，固定场景 success 覆盖缺口关闭。继续完成正式配置和基础 C0 evaluator 冻结后，只允许单 seed 分段试训；它不证明基础策略性能，也不授权多 seed 批量训练，后者继续 HOLD 至 P0-BASE-04 明确 GO。

2026-08-04，P0-BASE-00 已将该工具、测试和证据冻结到 `345ea044e5d0071b3be8bf9896bb18bfffaa6d36`；复核未改变上述证据边界，下一项为 P0-BASE-01。
