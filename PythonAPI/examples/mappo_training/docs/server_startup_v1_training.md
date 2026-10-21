# 服务器训练：startup_v1 + 持续俯视监控

此次只将本地基础驾驶 `startup_v1` 及其训练依赖同步到 `codex/server-remote-monitor`，未合并开发分支。未迁移本机模型、训练轨迹、实验输出或 MutMAPPO 算法。`utils/mut_mappo_session.py` 仅作为已有进程锁/文件检查工具复用。

`startup_v1` 基于 `instant_pace_v6`：无阻挡效率成本的起步宽限从 2 秒变为 0.8 秒，最大成本从 0.5/秒变为 1.5/秒。V6 超速、参考车道偏离、转向变化和安全/目标项保留。训练池为 48 例，验证池为 32 例；使用配对采样和跨 update 课程。新合同必须从零建立独立 run，不能将旧监控实验直接切换奖励后续训。

以下命令由用户在服务器执行。本次同步和离线验证未连接 CARLA、未启动服务器训练。CARLA 0.9.16 必须已在本机 2000 端口运行；预检需要 Town05，且端口上没有其他训练/评估时钟拥有者。不要重复启动已运行的 CARLA。

## 拉取

```bash
conda activate carla-mappo
cd /root/carla-project
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
cd PythonAPI/examples/mappo_training
```

若服务器 conda 环境名或仓库路径不同，使用实际值。`git switch` / `pull` 若报告本地修改，保留现场后处理，勿使用强制覆盖命令。

## 地图预检、建立独立计划

先执行这组命令，任一步失败就停止检查原因。真实几何报告与服务器地图及当前 checker 绑定，不复制本机 PASS 报告。

```bash
set -e
RUN_ID="BD_STARTUP_V1_s42_$(date +%Y%m%d_%H%M%S)"
CONTROL="diagnostics/server_startup_v1/$RUN_ID"
mkdir -p "$CONTROL/preflight"

python -m utils.preflight_base_driving_scenarios_v1 \
  --scenario-dir data/training_scenarios_town05_base_vnext_v1 \
  --output "$CONTROL/preflight/train.json" --live
python -m utils.preflight_base_driving_scenarios_v1 \
  --scenario-dir data/training_scenarios_town05_base_vnext_v1_validation \
  --output "$CONTROL/preflight/validation.json" --live

python -m utils.base_driving_session prepare \
  --run-dir "logs/$RUN_ID" \
  --control-dir "$CONTROL/session" \
  --contract configs/base_driving/base_driving_startup_v1_v1.json \
  --stop-updates 25 50 \
  --train-geometry "$CONTROL/preflight/train.json" \
  --validation-geometry "$CONTROL/preflight/validation.json" \
  --seed 42 \
  --monitor-mode web \
  --monitor-host 127.0.0.1 --monitor-port 8080 \
  --monitor-view overhead \
  --monitor-overhead-height 35 --monitor-smoothing-seconds 0.5 \
  --monitor-width 960 --monitor-fps 5

PLAN="$CONTROL/session/plan.json"
python -m utils.base_driving_session inspect --plan "$PLAN"
echo "PLAN=$PLAN"
```

该示例总预算为 50 updates，里程碑 25/50；每 update 16 回合。25 完成后串行冻结评估，再完整恢复至 50 并再次评估。600 仍是学习率/熵调度长度，不代表本计划会训练到 600。预算应在 prepare 前按实际意图设置；已建立计划不可修改预算、奖励或源码。

`prepare` 只创建计划，`inspect` 只读文件。上面两条 `--live` 预检连接 CARLA，但不 spawn/tick；下面 `run` 才真正启动训练。

## 启动、验证与恢复

```bash
# 后台启动，使本地 SSH 断开不直接挂断训练控制器。
nohup python -u -m utils.base_driving_session run --plan "$PLAN" \
  > "$CONTROL/controller.log" 2>&1 < /dev/null &
echo "controller PID=$!"

# 本机服务器健康与当前状态；各命令独立执行。
curl --connect-timeout 2 --max-time 5 http://127.0.0.1:8080/healthz
curl --connect-timeout 2 --max-time 5 http://127.0.0.1:8080/status.json
python -m utils.base_driving_session inspect --plan "$PLAN"
tail -n 50 "$CONTROL/controller.log"
```

`inspect` 的 `launch.log_path` 指向当前训练日志（也可直接查看 `logs/$RUN_ID/session/training_*.log`）。`safe_update` 只统计已完整保存的 update；PID 存活或 HTTP `ok` 不等于 PPO 已推进，也不等于驾驶资格通过。

网页监控选项保存在不可变计划中，每一训练阶段自动继承；网页镜头启用时不再额外启动本地 spectator 跟踪辅助进程。训练阶段实时俯视、平滑跟随、跨回合换场景、显示优化/保存状态，训练退出后保留最后图像及该阶段结果。**阶段间冻结评估不向这个训练页面推送视频**；此时页面保留上一训练阶段结果，下一阶段训练会自动重新接入同一服务。评估进度和最终选模以控制目录中的 `status.json`、`u000025_summary.json` / `u000050_summary.json` 及 `screen_u*/` 为准。

异常时控制器写 `FAILED_NO_RESTART`，不会自行重启。恢复 CARLA 后，确认旧 worker 已退出，在相同源码和同一计划下显式恢复：

```bash
# PLAN 和 CONTROL 使用启动时打印并保存的实际路径；不要重新生成 RUN_ID。
nohup python -u -m utils.base_driving_session run --plan "$PLAN" --resume-controller \
  >> "$CONTROL/controller.log" 2>&1 < /dev/null &
```

最终 `COMPLETED` 后不再恢复该计划。首次安全 checkpoint 之前失败时保留失败目录，再创建独立新计划；不要删除失败证据伪装成同一次续训。

## 本地浏览器

Windows PowerShell 建立隧道，保持窗口运行：

```powershell
ssh -NT -F "C:\Users\hy\.sshconfig" `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  -L 127.0.0.1:18080:127.0.0.1:8080 px2-jcy.matpool.com
```

打开 <http://127.0.0.1:18080/>；健康检查是 <http://127.0.0.1:18080/healthz>。本地 18080 映射服务器 8080，不要将服务器监控监听地址改成自己的 Windows 地址。


## 使用脚本启动（screen）

脚本 `scripts/run_server_startup_v1.sh` 适用于 Linux 服务器。在更新后的 `PythonAPI/examples/mappo_training` 目录执行：

```bash
# screen 未安装时先执行：apt-get update && apt-get install -y screen
conda activate carla-mappo
bash scripts/run_server_startup_v1.sh all
```

`all` 启动或复用 CARLA，并在独立 screen 中建立全新的 startup_v1 训练；默认总预算 50 updates，25/50 两个冻结评估里程碑。终端打印的 DISPATCHED 只代表已派发，真正启动和预检结果以日志为准。训练屏幕名 `mappo-startup-v1`，CARLA 屏幕名 `carla-server`。

```bash
# CARLA 已运行时，只派发训练；这会创建另一个全新 run。
bash scripts/run_server_startup_v1.sh train

# 查看训练 / CARLA；离开 screen 使用 Ctrl+A，松开再按 D。
screen -r mappo-startup-v1
screen -r carla-server

# 使用启动时打印的实际 RUN_ID；以下不会创建新实验。
bash scripts/run_server_startup_v1.sh status BD_STARTUP_V1_s42_实际时间

# 程序异常退出后：先修复/重启 CARLA，再显式恢复原计划。
# SSH 断开而 screen 仍在时，直接重新连接 screen，勿执行恢复。
bash scripts/run_server_startup_v1.sh resume BD_STARTUP_V1_s42_实际时间
```

如果环境名是 myconda 或 CARLA 安装目录不同，在首次启动时设置：

```bash
CONDA_ENV=myconda CARLA_ROOT=/opt/carla/CARLA_0.9.16 \
  bash scripts/run_server_startup_v1.sh all
```

Conda、Python 依赖和 screen 需提前安装；脚本不会自动安装软件。默认 CARLA 用户为 carla，默认以 root 通过 runuser 启动，也允许 carla 用户自行运行。CARLA 用 Low 画质离屏渲染。脚本不会杀死旧进程，端口已占用时不重复启动；训练入口等待 CARLA 就绪后，在共享时钟锁下检查版本和地图，必要时加载 Town05。同步世界或其他时钟拥有者会导致拒绝启动，须先核查现有任务。

启动日志为 `diagnostics/server_startup_v1/RUN_ID/launcher.log`，CARLA 日志为 `logs/server_launcher/carla.log`。首次运行的环境信息保存在该 run 的 `launcher.env`，status/resume 会复用；不支持通过改环境变量切换同一计划的奖励、预算或监控端口。训练源码仍受计划 SHA 检查，计划建立后不要拉取新代码再直接续训。

此脚本仅包装服务器上的进程和训练流程。本地 Windows 的 SSH 监控隧道仍按本文“本地浏览器”部分建立。阶段间冻结评估仍保留上一训练画面，不推送实时评估视频。screen 保护终端断连，不能保护服务器重启或模拟器崩溃。

脚本通过 Bash 语法与 9 项模拟系统命令的离线检查；这些检查未启动真实 screen、CARLA 或训练。真实服务器首次启动结果以 launcher.log 为准。
