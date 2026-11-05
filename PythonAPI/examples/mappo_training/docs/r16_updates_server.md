# 将本地 R16 长实验迁移到服务器：按更新轮次运行

本方案迁移本地 `adversarial_u1000_R16_long_20261105_v1` 的训练设置，保留远程监控分支的网页俯视监控、TensorBoard 和选模工具。
模型/PPO/R2环境共用原实现，不启动本地训练，也不合并本地开发分支。

## 已确认的实验口径

| 项目 | 设置 |
|---|---|
| 更新上限 | 1000 个完整 A+B round，每个团队各更新 1000 次 |
| 每团队每次采样 | 16 个完整 episode；两团队每 round 共 32 个训练 episode |
| PPO / minibatch | 2 epochs / 128 |
| 初始化与奖励 | 指定冻结红方 u1000，蓝方从同一基础 Actor 初始化；完整 R2 |
| 方法 | B0，双 MAPPO 团队，互学习 off，seed 42 |
| 初始 actor/critic LR | 默认各 0.0003，继承源配置 |
| LR 调度 | 按更新 round 线性衰减，下限为初值的 10% |
| 训练结束 | 按更新上限结束；ticks 仅统计成本，无 tick 截断 |
| 独立评估 | 每 100 个完整 round，selection 场景 A/B × C0/C1/C2 |
| 模型保留 | 历次最佳合格模型；不因评估结果自动早停 |
| 最终 test | 完成更新目标后，测试历次最佳合格模型；test 不参与选模 |

内部 gate 诊断仍保留源配置的节奏与成本统计。B0 不做互蒸馏。
门控/选模沿用此前确认的 `qualified_primary_event_credit`，安全违规率上限仍为 0.125。
这些评估口径与本地旧 `safe_critical_event` 排序不同；R2奖励和PPO超参数保持一致。

## 1. 上传与更新代码

本地本次基础模型是：

```text
D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/diagnostics/base_driving_vnext_20261020/long_s42_u600_v1/u001000.pt
SHA256: 0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a
```

上传该模型和 `u001000.pt.sha256` 到已有的服务器目录：

```text
/root/carla-project/PythonAPI/examples/mappo_training/artifacts/
```

此前服务器同名 u1000 可能是另一摘要，使用下述 expected SHA 会明确拒绝错误文件。
服务器执行：

```bash
cd /root/carla-project
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
```

## 2. 启动或复用 CARLA

已有可用 CARLA 在 2000 端口时直接复用，不要重复启动，也不要同时运行其他训练/评估。
尚未启动时，独立 screen 中执行（按实际安装路径调整）：

```bash
screen -S carla
cd /opt/carla/CARLA_0.9.16
mkdir -p /tmp/xdg-runtime-carla
chown carla:carla /tmp/xdg-runtime-carla
chmod 700 /tmp/xdg-runtime-carla
runuser -u carla -- env -u VK_ICD_FILENAMES \
  HOME=/home/carla XDG_RUNTIME_DIR=/tmp/xdg-runtime-carla \
  ./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
  -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port=2000 -graphicsadapter=0
```

按 Ctrl+A，再按 D 分离。训练流程不会自动启动或关闭 CARLA。

## 3. 离线准备服务器实验

另开服务器终端。后续 Python 命令均从 `mappo_training` 目录执行：

```bash
screen -S r16-u1000
source /root/miniconda3/etc/profile.d/conda.sh
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
set -o pipefail

EXP="experiments/generated/R16_u1000_s42_server_$(date +%Y%m%d_%H%M%S)"

python -m utils.prepare_critical_reward_server \
  --checkpoint artifacts/u001000.pt \
  --expected-sha256 0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a \
  --output-dir "$EXP" \
  --updates 1000 --episodes-per-team-update 16 \
  --eval-every-updates 100 \
  --device cuda --seed 42 --host 127.0.0.1 --port 2000

printf 'export EXP=%q\n' "$EXP" > "$EXP/run.env"
echo "恢复变量：source $EXP/run.env"
```

准备阶段不连接 CARLA、不启动训练。应看到 `target_updates=1000`、`episodes_per_team_update=16`、
`stop_rule=rounds` 和 `tick_policy=audit_only_no_tick_stopping_cap`。
输出 `config.json`、`experiment.json`、`preparation.json`。
`run.train_tick_budget` 为 null，`run.rounds`/`lr_horizon_rounds` 为 1000。

默认 R2 模板的奖励/PPO数值与本地 R16 一致。复用其他 R2 源配置可加 `--source-config <服务器配置路径>`；
准备器会重写模型、参考目录、设备与更新调度，保留其学习/奖励参数和安全阈值。
换模型只需更换 `--checkpoint` 和相应预期 SHA，使用新的输出目录。
本模式不要同时传 `--train-ticks` 或 `--lr-comparison`。

## 4. 准备参考并启动长任务

先生成 C1 参考。每一步成功后再执行下一步：

```bash
python -u -m utils.run_first_point_experiments references \
  --experiment "$EXP" --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

然后运行整个长任务：

```bash
python -u -m utils.run_first_point_experiments run-updates \
  --experiment "$EXP" --resume --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080 \
  --monitor-width 960 --monitor-fps 5 \
  2>&1 | tee -a "$EXP/controller.log"
```

该命令自动执行：更新至 100 → selection → 更新至 200 → selection → … → 1000 → selection → 最佳模型 test。
不会根据评估结果提前结束；发生训练/评估异常会停止，保留原证据并要求明确恢复。
首段从基础 Actor 初始化，此后的 `--resume` 在相同运行内恢复完整 A+B checkpoint。
`--allow-weak-red` 表示当前是脆弱红方诊断实验，不表示红方已获得正式驾驶资格。

## 5. 断线与恢复

SSH 断开不会使 screen 内任务退出。重新连接：

```bash
screen -r r16-u1000
```

若进程已退出，重新激活环境、进入同一目录，恢复先前打印的 run.env：

```bash
source experiments/generated/本次实验目录/run.env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u -m utils.run_first_point_experiments run-updates \
  --experiment "$EXP" --resume --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080 \
  2>&1 | tee -a "$EXP/controller.log"
```

不要重新 prepare，也不要修改冻结配置。恢复会验证源码/配置/checkpoint身份。
若恰好训练到评估整点，先补做或复用该点评估，再继续；已到 1000 时不再训练，会补完最终评估。
旧源码/旧预算下的本地31轮运行不能直接作为此新配置的 `--resume` 源。

## 6. 实时监控与结果

另一个激活环境、恢复 EXP 的服务器终端中：

```bash
tensorboard --logdir "$EXP/runs" --host 127.0.0.1 --port 6006
```

本机建立端口转发（替换 SSH 别名）：

```bash
ssh -N -L 18080:127.0.0.1:8080 -L 16006:127.0.0.1:6006 服务器SSH别名
```

- 俯视监控：`http://127.0.0.1:18080/`，显示当前更新/1000；训练与评估切换时网页服务可能短暂重启，刷新即可。
- TensorBoard：`http://127.0.0.1:16006/`。本更新模式横轴是完整A+B更新轮次；ticks作为cost/meta指标保留。
- 训练状态：`$EXP/runs/R2_B0_s42/session/status.json`，最终应为 completed_updates=1000、training_complete=true。
- 完整 checkpoint：`$EXP/runs/R2_B0_s42/checkpoint_manifest.json` 指向可信边界；`latest.pt` 是便捷副本。
- 各阶段变化：`$EXP/selection_history.csv`，包含团队、轮次、主事件率、安全违规率和合格标记。
- 最佳模型指针：`$EXP/best_selection/R2_B0_s42.json`。读取 selection_dir，里面有 selection.json；
  有合格候选才生成 frozen_blue_team.pt 及 sidecar。
- 最终 test：`$EXP/comparison.json/csv`，以及 `$EXP/evaluations/*/test/summary.json`、cases.jsonl、traces。

训练完成与策略合格是两个结论。`NONE_QUALIFIED` 不会触发自动早停，也不会导出不合格模型。
最终评估优先看合格 R2 主事件率，并结合违规率、严重TTC、碰撞、完成度与舒适性；不要只看红车成功率。
中途尚未test时，comparison的test指标为null、qualified_exports可能为0，应看最佳模型指针判断是否已选出候选。
