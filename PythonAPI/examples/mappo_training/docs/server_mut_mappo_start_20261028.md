# 服务器 MutMAPPO 训练、实时监控与独立评估指南

适用于 2026-10-28 同步后的 `codex/server-remote-monitor`。代码由 Git 同步，模型单独上传；继续使用 `/root/carla-project` 与 Conda `carla-mappo`。

## 本版同步内容与实验选择

保留基础训练和 MutMAPPO 的常驻俯视网页，合入本地新增的团队效率奖励、C1 参考曲线、逐车驾驶统计、完整角色交换对采样、串行训练/选模/测试、四组合贡献评估，以及可选 192 例对称案例生成器。

以下主流程使用 `utils.run_first_point_experiments`：默认 full 场景为 24 train / 16 gate / 16 selection / 16 test，包含四方向滚动起步、并排/前后错位和完整 Actor 交换对。192 例 symmetric_v2 是另一套候选场景，不自动混入主流程。A/B 是两套蓝团队，红0/1冻结，蓝2/3学习；不是红蓝轮流更新。

主流程奖励为 `c1_progress_team_v1`，以同案例普通蓝交通 C1 的红方进度曲线为参考，结合蓝方驾驶与安全成本。B0关闭互学习、B1普通互学习、B2选择性互学习。所有组共用同一奖励/案例/红方和实际训练 tick 预算。

u370 尚未独立合格，以下带 `--allow-weak-red` 的步骤明确用于工程与弱基线诊断。正常退出、门控开启或导出蓝方均不等于算法有效。此次源码/奖励/场景变更须新建实验；旧 run 只能在其原源码与配置下严格续训，不修改 hash 绕过校验。

## 1. 安全退出旧任务，更新代码

旧训练/评估完成或安全退出后执行，不在训练期间覆盖共享源码。Git 若报告本地修改，先保留并处理，不强制覆盖。

```bash
conda activate carla-mappo
cd /root/carla-project
git status --short
git fetch origin
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
git log -1 --oneline
cd PythonAPI/examples/mappo_training
python -m visualization.monitor_http_server --state-dir logs/web_monitor_8080 --stop
mkdir -p artifacts
```

依赖沿用现有环境。检查 Python 客户端和 CUDA：

```bash
python -c 'import carla, torch; print("CARLA client:", carla.__file__); print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
```

## 2. 模型上传与登记

服务器已有 `artifacts/red_weak_u370.pt` 时无需重新上传。否则在本地 Windows PowerShell 执行。SSH 别名须指向平台当前服务器，不沿用已失效的机器地址：

```powershell
scp -F "C:\Users\hy\.sshconfig" `
  "D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training\logs\BD_vNext_V6_scratch_s42_u600_20261020_v1\periodic\ckpt_update_000370.pt" `
  px2-jcy.matpool.com:/root/carla-project/PythonAPI/examples/mappo_training/artifacts/red_weak_u370.pt
```

服务器检查 SHA256 应为：

```bash
sha256sum artifacts/red_weak_u370.pt
# f0aba30af9f3048278e31100f3892df79351a32401f6c16439fb3c055b293c60
```

如果之前已成功运行 `prepare_mut_mappo_server`，模型 checksum sidecar 已登记，可跳过下面命令。首次使用才执行，输出目录必须全新：

```bash
python -m utils.prepare_mut_mappo_server \
  --checkpoint artifacts/red_weak_u370.pt \
  --expected-sha256 f0aba30af9f3048278e31100f3892df79351a32401f6c16439fb3c055b293c60 \
  --output-dir "configs/mut_mappo/generated/u370_registration_$(date +%Y%m%d_%H%M%S)" \
  --device cuda --seed 42
```

此命令校验文件并登记 sidecar，输出的是旧任务的 B0/B1/B2 配置。下文用新的实验准备入口生成团队效率奖励配置，不直接使用 `u370_smoke_v1/b2.json` 冒充新奖励实验。

## 3. 单独在 screen 中启动 CARLA

先检查已有实例；已正常运行2000端口时不要再启动第二个：

```bash
pgrep -af 'CarlaUE4|CarlaUnreal'
ss -ltnp '( sport = :2000 or sport = :2001 or sport = :2002 )'
```

仅在没有运行实例时执行：

```bash
screen -S carla
cd /opt/carla/CARLA_0.9.16
id carla
mkdir -p /tmp/xdg-runtime-carla
chown carla:carla /tmp/xdg-runtime-carla
chmod 700 /tmp/xdg-runtime-carla

runuser -u carla -- env -u VK_ICD_FILENAMES \
  HOME=/home/carla XDG_RUNTIME_DIR=/tmp/xdg-runtime-carla \
  ./CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
  CarlaUE4 -RenderOffScreen -nosound -quality-level=Low \
  -carla-rpc-port=2000 -graphicsadapter=0
```

此指令沿用已配置好的 Linux CARLA 安装及非 root 用户 `carla`。若安装路径不同，替换路径。`Address already in use` 应先排查已有实例，不重复启动。保持离屏渲染，俯视 RGB 监控需要渲染。

按 Ctrl+A，松开后按 D 分离 CARLA screen。随后新建训练 screen：

```bash
screen -S mut-u370
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import carla
c = carla.Client('127.0.0.1', 2000)
c.set_timeout(120)
version = c.get_server_version()
assert version.startswith('0.9.16'), version
w = c.get_world()
if w.get_settings().synchronous_mode:
    raise RuntimeError('世界已处于同步模式，请检查其他训练或遗留任务，勿同时推进世界')
if w.get_map().name.rsplit('/', 1)[-1] != 'Town05':
    w = c.load_world('Town05')
print('CARLA:', version, 'Map:', w.get_map().name)
PY
```

上面的地图切换仅在没有其他训练/评估运行时执行。所有后续步骤串行，共用一个 CARLA，不要多进程同时训练。

## 4. 准备新的实验计划（离线）

第一次先用一个 seed、12000训练 tick 做工程检查。预算为每组 A+B 合计；参考采集、教师门控及独立评估另计，不代表全部运行时间。到预算后完成当前完整团队边界，可能有超量。

```bash
EXP="experiments/generated/MUT_U370_team_v1_$(date +%Y%m%d_%H%M%S)"

python -m utils.run_first_point_experiments prepare \
  --output "$EXP" \
  --red-checkpoint artifacts/red_weak_u370.pt \
  --suite core --seeds 42 \
  --train-ticks 12000 --device cuda --scene-stage full

printf 'export EXP=%q\n' "$EXP" > "$EXP/launch.env"
echo "恢复变量：source $EXP/launch.env"
```

`base_config.json`、`cases.json`、`configs/B0_s42.json` 等和 `experiment.json` 在该目录生成。它们包含服务器路径，已被 Git 忽略，不修改已准备的实验来切换模型或预算。

正式扩大预算时重新准备独立目录，例如 `--train-ticks 120000 --seeds 42 43 44`。这是候选先导规模，不承诺足够收敛或当天完成。旧 `prepare_mut_mappo_server` 默认2轮与这里的实际 tick 停止规则不同。

## 5. 实时监控与 SSH 隧道

在 Windows PowerShell 单独执行并保持打开；如果已有同端口隧道，不重复启动：

```powershell
ssh -NT -F "C:\Users\hy\.sshconfig" `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 `
  -L 127.0.0.1:18080:127.0.0.1:8080 `
  px2-jcy.matpool.com
```

浏览器打开 `http://127.0.0.1:18080/`。后面的实时步骤都显式传 `--monitor-mode web`。参考曲线、训练、教师门控、独立评估及贡献评估均有俯视画面；base-eval 基础资格入口使用其原有文件报告。

## 6. 先检查红方，再采集 C1 参考曲线

训练 screen 内执行，第一步会输出 C0无蓝/C1普通蓝的逐车驾驶与交互结果：

```bash
python -u -m utils.run_first_point_experiments red-eval \
  --experiment "$EXP" --monitor-mode web --monitor-port 8080
```

查看 `$EXP/red_readiness.json` 和 `$EXP/red_screen/summary.json`。不通过不等于程序错误；u370可能自身失败。正式正常红方实验还需独立基础驾驶资格，不以此滚动交互检查替代。

采集 train/gate/selection 案例中同一冻结红方和普通蓝方的 C1 进度曲线：

```bash
python -u -m utils.run_first_point_experiments references \
  --experiment "$EXP" --allow-weak-red \
  --monitor-mode web --monitor-port 8080
```

`--allow-weak-red` 允许已有效执行但驾驶未合格的参考曲线，保留诊断标记；环境异常或无效执行仍会停止。此阶段可能比两轮短训练长，因为要逐个采集多个参考案例。参考不齐不能启动新奖励训练。不要用占位曲线代替。

## 7. 先运行 B0 两轮检查，再运行 B2

```bash
python -u -m utils.run_first_point_experiments train \
  --experiment "$EXP" --groups B0 --seeds 42 --stop-round 2 \
  --monitor-mode web --monitor-port 8080

python -m utils.mut_mappo_session inspect --run-dir "$EXP/runs/B0_s42"
```

第一条成功结束后再执行检查。应有 `$EXP/runs/B0_s42/checkpoint_manifest.json`，`completed_round` 应为2（若先达到tick预算，可能更早结束）。监控应显示采样/更新/保存/完成，安全边界应与 manifest 一致。

B0流程无误后运行选择性互学习 B2 到已准备的 tick 预算：

```bash
python -u -m utils.run_first_point_experiments train \
  --experiment "$EXP" --groups B2 --seeds 42 \
  --monitor-mode web --monitor-port 8080
```

B2首次达到10000训练tick后的完整边界才刷新教师；12000预算仅用于流程测试，可能几乎没有教师启用后的学习。要比较算法需更充分且相同的实际预算。

如需完成同预算B0和B1：

```bash
python -u -m utils.run_first_point_experiments train \
  --experiment "$EXP" --groups B0 B1 --seeds 42 --resume \
  --monitor-mode web --monitor-port 8080
```

`--resume` 对已有完整边界的B0恢复，对尚未建立run的B1从初始化开始。不是只加载Actor；会恢复完整优化器、教师、随机状态和成本计数。已达到预算的任务被跳过。

## 8. 选模、冻结后测试、贡献评估和汇总

先以已训练的B2为例：

```bash
python -u -m utils.run_first_point_experiments evaluate \
  --experiment "$EXP" --groups B2 --seeds 42 \
  --allow-weak-red --with-credit \
  --monitor-mode web --monitor-port 8080

python -m utils.run_first_point_experiments summarize --experiment "$EXP"
```

评估先在 selection 上做配对 C0/C1/C2，并按蓝方资格和安全效率选模。只有选出蓝方后，才生成 test 的 C1参考、评估冻结蓝方和00/10/01/11四组合贡献；没有合格候选会保留结果并跳过测试，不强行导出。

- `red_screen/summary.json`：基础红方在普通交互中的表现。
- `runs/B2_s42/metrics.jsonl`、`actual_cost_audit.jsonl`：训练指标与真实成本。
- `evaluations/`、`results/`：配对评估、选模和冻结后测试/贡献结果。
- `comparison.csv`、`comparison.json`：按组/外部训练seed汇总。A/B不作独立种子。

重点看有效案例数、红方自身失败、蓝方安全资格、`safe_efficiency_utility`、`AttackGain_safe`，以及逐车完成度/控制/纵向安全统计。无效案例不能作驾驶失败，u370自然超时不能全算攻击收益。

## 9. 一条命令运行已准备的完整流程

熟悉分步流程后，可选用串行控制入口，替代手动第6～8节，不同时运行两套：

```bash
python -u -m utils.run_first_point_experiments run \
  --experiment "$EXP" --groups B0 B1 B2 --seeds 42 \
  --allow-weak-red --resume --with-credit \
  --monitor-mode web --monitor-port 8080
```

弱策略诊断路径显式使用 `--allow-weak-red`。不带该参数的完整run要求基础驾驶资格及C0/C1检查通过；可用现有 `utils.preflight_base_driving_scenarios_v1 --live` 生成服务器validation几何报告，再向 `base-eval`/`run` 传 `--geometry-report <报告.json>`。不要上传另一台机器的实时预检来冒充本机验证。

## 10. 防断、异常恢复与监控诊断

Ctrl+A后按D分离；`screen -r mut-u370`返回。screen保持进程不因SSH断开退出，不会修复CARLA或Python崩溃。

异常时先看终端、`runs/<组_seed>/session/status.json`与manifest，并检查CARLA状态；恢复干净CARLA后在同源码/模型/配置下运行相应train命令并加 `--resume`。不在旧实验内切换红方、奖励或案例，不自动重启未知失败。

浏览器打不开时，在服务器逐条执行：

```bash
curl --connect-timeout 2 --max-time 5 http://127.0.0.1:8080/healthz
curl --connect-timeout 2 --max-time 5 http://127.0.0.1:8080/status.json
ss -ltnp 'sport = :8080'
```

healthz成功说明HTTP服务可用，还需检查SSH隧道；`stale`表示生产者失联，不能解释为正在训练。相机错误与训练错误分别查看。更新/保存期间仿真暂停，保留最后一帧正常。原生`-11`崩溃原因仍需服务器栈与日志诊断，本次同步不宣称已修复。

本地回归使用合成模型/环境及离线地图，不能替代服务器真实四车出生、训练稳定性或算法增益验证。
