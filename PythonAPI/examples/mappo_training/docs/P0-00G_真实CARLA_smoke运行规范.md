# P0-00G 真实 CARLA smoke 运行规范

> 状态：PASS / FROZEN BY P0-BASE-00 (`345ea04`)（2026-08-04 复核时全量 93 项测试通过，用户确认人工视觉无异常）。主 Run 的 fresh、中断、恢复、strict validator、checkpoint 与清理门槛通过；随后 `teleport` 纯逻辑和近终点 `scripted` success 补验均在真实 CARLA 中通过。未提供视觉 episode 编号/录像，策略性能与远距离成功率仍未验证。唯一适用训练入口为 `training.train_no_rllib_mappo_robust_v2`；逐次证据与失败 Run 处置见 [P0-00G 真实 CARLA smoke 测试记录](./P0-00G_真实CARLA_smoke测试记录.md)，success 补验见 [P0-00G-SUCCESS 规范与记录](./P0-00G-SUCCESS_真实success逻辑测试规范与记录.md)。

## 1. 目的与结论边界

P0-00G 只验证真实 CARLA 下从环境、生命周期、rollout、GAE/PPO、逐 episode 审计、TensorBoard/JSONL 到 checkpoint/resume 的端到端数据流。通过只表示基础 MAPPO 训练链路具备继续验证的工程条件，不表示策略已收敛、性能提升、场景鲁棒或碰撞归因正确。

本 smoke 的 update 1-4 均处于默认 warmup，实际只采样 `scenario_01_basic.json`。它不覆盖多场景状态泄漏、精确碰撞对象、对抗 steer 指标、C0/C1/C2 或 held-out 泛化。

## 2. 2026-08-02 环境快照

| 条件 | 当前证据 | 状态 | 执行前要求 |
|---|---|---|---|
| CARLA 可执行文件 | `D:\study\yan2\CARLA_0.9.16\CarlaUE4.exe` 存在 | PASS | 不更换安装目录 |
| 根 Python 环境 | Python 3.10.8，`D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe` | PASS | 只使用该解释器 |
| 关键包 | CARLA 0.9.16、Torch 2.9.1、NumPy 2.2.6、Pygame 2.6.1、TensorBoard 2.20.0 可导入 | PASS | 修复后重新跑全量测试 |
| 计算设备 | Torch 为 CPU build，CUDA 不可用 | WARN | smoke 可运行但较慢；不是通过/失败条件 |
| 磁盘 | D 盘可用约 34.12 GB | PASS | smoke 启动前至少保留 10 GB；长训练另行预算 |
| CARLA 进程 | 2026-08-02 16:23，启动器 PID 23232、Win64 服务 PID 27296，均来自本安装目录 | PASS | 不并行启动第二个 CARLA 服务 |
| `localhost:2000` | PID 27296 监听；RPC 握手 client/server 均为 0.9.16；修复后重跑前为异步、`fixed_delta_seconds=None`、actor 0/0 | PASS | 连接或 tick 卡死时 HOLD，并提示用户手动重启 |
| 多重虚拟环境 | 项目目录内另有 `mappo_training/venv` | RISK | 禁止使用嵌套 venv，防止依赖漂移 |
| Git/代码冻结 | 修复过程中工作树非 clean | EXPECTED/NOW | 评审通过后提交代码；Run 必须记录准确 commit/dirty paths |

环境快照只说明检查时刻的状态。CARLA 进程、端口、磁盘、Git 和输出目录必须在每次 smoke 前重新检查。

## 3. 硬性前置条件

1. 在仓库根目录启动 CARLA：

   ```powershell
   .\CarlaUE4.exe -quality-level=Low
   ```

2. 在 `PythonAPI/examples/mappo_training/` 目录确认客户端和服务器版本：

   ```powershell
   ..\..\..\venv\Scripts\python.exe -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(3); print('client=',c.get_client_version()); print('server=',c.get_server_version())"
   ```

   两者都必须报告 0.9.16。连接失败、版本不同或另有客户端控制同步 tick 时 HOLD。

3. 使用根虚拟环境运行全量测试，必须全部通过：

   ```powershell
   ..\..\..\venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

4. 正式复现实验应把训练相关代码、场景和本规范冻结到明确 commit。当前修复 smoke 的工作树非 clean，必须记录基线 SHA、dirty paths 和源码 fingerprint；因此结果只归属于当前修复树。不得使用 `--allow-resume-code-drift` 绕过 fresh/resume 的来源检查。
5. 使用从未存在过的唯一 RunID 和空输出目录。不加载旧 checkpoint，不使用 `--pretrained-checkpoint`。
6. CARLA 运行期间只允许当前训练客户端推进同步世界；禁止第二个客户端调用 `world.tick()`。
7. 明确人工观察方式：可使用 CARLA 可见窗口、预先声明的 `--render` 或录像。没有可复核视觉证据时不得勾选“人工查看 episode”。

## 4. 冻结命令与执行顺序

以下变量只在当前 PowerShell 会话使用：

```powershell
$p0Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$p0RunId = "P0_00G_robust_v2_smoke_s42_$p0Stamp"
$p0RunDir = "logs/$p0RunId"
```

### 4.1 Fresh run：完成 update 1-3

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --run-id $p0RunId `
  --updates 3 `
  --lr-decay-horizon 3 `
  --rollout-episodes 4 `
  --min-transitions-per-update 64 `
  --max-steps 300 `
  --save-interval 1 `
  --tb-histogram-interval 1 `
  --seed 42 `
  --save-dir $p0RunDir
```

`--save-interval 1` 是 P0-00G 的证据条件；否则 update 1-4 不会产生 periodic checkpoint。`save_interval`、`log_interval` 和 `tb_histogram_interval` 已纳入 resume 不可漂移配置。

### 4.2 受控中断：验证最近完整边界

完成 update 3 后启动：

```powershell
..\..\..\venv\Scripts\python.exe -m training.train_no_rllib_mappo_robust_v2 `
  --resume latest `
  --updates 4 `
  --p0-smoke-interrupt-after-episodes 1 `
  --save-dir $p0RunDir
```

工具执行无法可靠向后台 Windows 进程发送一次 Ctrl+C，因此本次使用仅允许在 resume 生效的确定性钩子：update 4 的第 1 个 episode 审计落盘后，主动抛出 `KeyboardInterrupt` 并等待训练器自行清理。钩子不修改奖励、网络、rollout 数据或 checkpoint 冻结配置。不得关闭 CARLA、强杀 Python 或删除文件来模拟此步骤。预期 `interrupted.pt` 与中断时的安全 `latest.pt` 都指向 update 3；未完成 update 4 的逐 episode 记录在下次 resume 时必须自动丢弃。

### 4.3 严格恢复：完成 update 4

再次运行 resume 命令，但移除 `--p0-smoke-interrupt-after-episodes 1`，完整完成 update 4。`--updates 4` 表示总进度到 4，不是额外训练 4 个 update。不得改变 seed、LR horizon、save interval、rollout 或指标配置。

## 5. 必需产物

```text
<RunID>/
├── checkpoint_manifest.json
├── latest.pt + latest.pt.sha256
├── interrupted.pt + interrupted.pt.sha256
├── periodic/
│   ├── ckpt_update_000001.pt + .sha256
│   ├── ckpt_update_000002.pt + .sha256
│   ├── ckpt_update_000003.pt + .sha256
│   └── ckpt_update_000004.pt + .sha256
├── metrics_schema.json
├── metrics.jsonl
├── episodes.jsonl
└── tb/events.out.tfevents.*
```

最终 `metrics.jsonl` 必须恰有 update 1、2、3、4；`episodes.jsonl` 必须恰有 16 条已提交记录，每个 update 的 `episode_index` 为 0-3。每条记录保存场景、终局、长度、结果，以及每个 agent 的 return、active steps、transition、done、terminated、truncated 和 bootstrap。

## 6. 自动验收

完成 update 4 后运行严格 validator：

```powershell
..\..\..\venv\Scripts\python.exe -m utils.validate_tensorboard_run `
  --run-dir $p0RunDir `
  --expected-updates 1 2 3 4 `
  --expected-episodes-per-update 4 `
  --require-ppo-updates `
  --max-preupdate-ratio-error 1e-6
```

只有返回 `"status": "PASS"` 才满足自动指标门槛。验证器同时检查：

- TensorBoard 与 `metrics.jsonl` 的 update/标量一致；
- `episodes.jsonl` 的协议、顺序、数量和 run identity；
- 逐 episode active steps、transition、return、长度、结果与 update 聚合一致；
- 每个 update 都实际执行 PPO；
- 更新前 `max_abs(ratio-1) < 1e-6`；
- Actor 和 Critic 的 update norm 大于 0；
- 奖励字段有限且逐步守恒残差不超过 `1e-6`。

validator PASS 不替代 checkpoint/manifest、CARLA actor 清理和人工轨迹检查。

## 7. 人工与 checkpoint 验收

- 人工观察至少 3 个 episode，记录 update、episode index 和终局。
- 主 Run 至少覆盖成功、碰撞、max_steps 中两类；未覆盖类型只能用预先指定固定场景做补充 smoke，不能修改主 Run 数据。
- 检查 `checkpoint_manifest.json`：latest=4、interrupted=3、periodic updates=1-4，路径和 SHA256 与 sidecar 一致。
- 检查 checkpoint payload 含 Actor、agent-conditioned Critic、全部 optimizer、update、课程、LR、RNG、RunID/Git、训练/指标协议和 `last_metrics`。
- update 1、2、3、4 的 Actor LR 预期分别为初始值的 100%、50%、5%、5%；resume 不得重置 LR horizon。
- 运行前、episode 间和进程退出后比较 CARLA 中 `vehicle.*` 与 `sensor.*` 数量；不得持续累积训练创建的 actor，也不得出现销毁失败警告。
- LiDAR/TTC、collision、success 等稀疏 0 必须能由 raw diagnostic 解释，不能把“未触发”写成“传感器已证明正确”。

## 8. 停止与结果处置

以下任一情况均判为 FAIL/HOLD：CARLA 连接或同步冲突、样本计数不守恒、失活 agent 继续产生 transition、任意 NaN/Inf、PPO 被跳过、ratio 预检失败、模型未更新、日志/episode/checkpoint 不连续、checksum 不一致、resume 配置漂移、actor/sensor 泄漏或 validator 非 PASS。

若只因随机轨迹没有覆盖两类终局，判为 INCONCLUSIVE，并执行预先限定的补充场景；不得增加训练 update 后把结果改写为 PASS。所有失败 Run 原样保留，使用新 RunID 重跑，不覆盖、不删除失败证据、不事后放宽门槛。
