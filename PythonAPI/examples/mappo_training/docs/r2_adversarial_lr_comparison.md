# R2 对抗学习修正与 B0 学习率对照

目标：蓝车在保持驾驶硬约束合格的前提下，增加对冻结红车的危险性交互。
本次保留 R2 奖励及其系数，先检查基础 PPO 学习，再决定是否扩大到 B012。

## 本次口径和控制变量

- 门控 `mutual.quality_metric` 和选模 `evaluation.selection_metric` 统一为
  `qualified_primary_event_credit`：最终蓝方合格且触发 R2 主事件的有效回合数 / 全部有效回合数。
- 主事件沿用在线 R2 实现：合法向前行驶、蓝车在红车前半空间、
  `0.5 < TTC <= 1.5` 秒持续至少 `0.3` 秒。不额外增加“整回合从未出现严重 TTC”的定义。
- 教师逐场景家族违规率上限和选模违规率上限都保留 `0.125`；
  教师质量差门槛保持 `0.25`。辅助检查仍包括严重 TTC、碰撞、完成度、舒适性和 C1 相对影响。
- 两个 B0 配置均关闭互学习，均使用双 MAPPO 团队 A/B，同一红方 checkpoint、seed、场景和共享 C1 参考。
  每次每团队 4 回合，2 PPO epochs，初始 actor/critic LR 均由源配置继承，默认 `0.0003`。
- `B0_LINEAR`：按实际训练 ticks 线性衰减至初值的 10%；`B0_CONSTANT`：保持初始 LR。
  两组各最多 60000 训练 ticks；评估 ticks 单独计数。完整 A+B 边界可能略超目标。
- 每约 10000 ticks 手动暂停、复评固定 selection 场景，保留历次最优；不会自动提前终止或自动推进下一段。
  A/B 是同一运行内的团队，不能当成两个独立随机种子。

这是在原有训练、评估和选模入口上接入。新 checkpoint 绑定当前源代码身份；
旧代码生成的运行不能直接在此版本 `--resume`，也不要修改旧配置/身份校验强行续训。
新对照分段续训必须始终使用第一次生成的同一份配置。

## 1. 服务器更新与 CARLA

先将本次代码提交并推送至 `codex/server-remote-monitor` 后，在服务器项目根目录：

```bash
cd /root/carla-project
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
```

已有正常运行的 CARLA 就复用，不要重复启动占用 2000/2001/2002。
没有运行时，在独立 screen 中启动；路径可按服务器安装位置调整：

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

按 `Ctrl+A`，再按 `D` 分离。已有其他训练/评估正在控制此 CARLA 时，先等其结束。

## 2. 准备同源 B0 双臂

另开训练 screen。只需上传模型与同名校验文件；不要上传本机生成的带绝对路径配置。
checkpoint 可以更换，路径与预期 SHA256 均是参数。下面从本次上传模型的 sidecar 读取摘要；
若需要独立核对上传一致性，可将 `SHA256` 直接填为本机计算的摘要。

```bash
screen -S r2-lr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training

CKPT=artifacts/red_u1000.pt
SHA256=$(awk 'NR==1 {print $1}' "$CKPT.sha256")
EXP="experiments/generated/R2_B0_LR_s42_$(date +%Y%m%d_%H%M%S)"

python -m utils.prepare_critical_reward_server \
  --lr-comparison \
  --checkpoint "$CKPT" --expected-sha256 "$SHA256" \
  --output-dir "$EXP" --seed 42 --train-ticks 60000 \
  --device cuda --host 127.0.0.1 --port 2000

printf 'export EXP=%q\n' "$EXP" > "$EXP/run.env"
echo "恢复变量：source $EXP/run.env"
```

准备不会连接 CARLA，也不会启动训练。输出 `experiment.json`、共享 `config.json`、
`configs/B0_LINEAR_s42.json` 和 `configs/B0_CONSTANT_s42.json`。
如需复用自己的一套 R2 参数，增加 `--source-config <配置.json>`；
本对照要求源配置仍是 2 epochs、每次 4 回合，奖励和安全门槛从该配置保留。
场景、参考目录、C1 控制器也可分别使用 `--case-manifest`、`--reference-dir`、`--c1-controller` 配置。

生成共享 C1 参考一次。使用现有默认重复次数 1，后续命令也保持相同重复次数：

```bash
python -u -m evaluation.mut_mappo_reference \
  --config "$EXP/config.json" --splits train gate selection \
  --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

`--allow-weak-red` 明确这是针对脆弱红方的诊断实验，不代表红方通过基础驾驶资格。
不要混用不同红方、任务或参考身份；复用已有参考目录时校验不匹配会拒绝运行。

## 3. 每段训练后复评

第一次 `TARGET=10000`，之后由你看结果再依次改为 `20000`、`30000`、`40000`、`50000`、`60000`。
`TARGET` 是累计绝对训练 ticks，不是新增预算；暂停不重置学习率。
下面两组串行使用同一个 CARLA，不要同时运行：

```bash
TARGET=10000

python -u -m utils.run_first_point_experiments train \
  --experiment "$EXP" --groups B0_LINEAR B0_CONSTANT --seeds 42 \
  --resume --stop-train-ticks "$TARGET" \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080

python -u -m utils.run_first_point_experiments evaluate \
  --experiment "$EXP" --groups B0_LINEAR B0_CONSTANT --seeds 42 \
  --selection-only --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

`--resume` 在本入口中会自动判断 manifest：首段从零初始化，此后从最后完整 A+B 边界恢复。
断线重连 `screen -r r2-lr`；若进程已退出，恢复 `EXP` 后重新执行本段训练命令即可。
若进程异常退出而 CARLA 无响应，先恢复 CARLA，再续训；不要删除 checkpoint manifest。

复评默认是 A/B × C0/C1/C2 × selection 场景，时间开销会明显增加。
`--selection-only` 不生成 test 反馈。训练/参考/评估都会接入现有俯视网页监控。
两段命令之间进程退出时，网页服务也可能退出；下一段启动后再刷新。

每段完成后查看：

- `$EXP/selection_history.csv`：每个阶段、每个团队的主事件率、违规率、合格状态和辅助选模指标。
- `$EXP/best_selection/B0_LINEAR_s42.json` 和 `B0_CONSTANT_s42.json`：历次最优指针。
  指向目录下 `selection.json` 有全部候选；合格时才有 `frozen_blue_team.pt` 及 sidecar。
- `$EXP/results/*_r*.json`：各段 selection、候选、最佳轮次等记录。
- `$EXP/evaluations/*/selection/summary.json`、`cases.jsonl`、`traces/`：完整多维独立评估证据。
- `$EXP/runs/*/metrics.jsonl` 与 `tb/`：训练窗口学习信号、KL、loss、实际学习率等。

若 `NONE_QUALIFIED`，代表暂无候选通过安全筛选；不会把不合格模型包装成最佳策略。
若 `comparison.json` 中 test 指标为 null，这是尚未执行最终 test，不能解释为效果为零。

## 4. 最终冻结后才运行 test

看过 selection 趋势、决定停止或完成预算后，只运行一次最终 test：

```bash
python -u -m utils.run_first_point_experiments evaluate \
  --experiment "$EXP" --groups B0_LINEAR B0_CONSTANT --seeds 42 \
  --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

该命令复用已有 selection，以历次最佳模型准备 test 参考并评估，更新 `comparison.json/csv`。
不要根据 test 结果再选择早停点。单 seed 只能定位学习信号；若恒定 LR 在 selection 上稳定提升
且安全约束合格，再决定补多种子、以及 B0/B1/B2 贡献对比。

## 5. TensorBoard 与本地端口

另开服务器终端：

```bash
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training
source <本次EXP目录>/run.env
tensorboard --logdir "$EXP/runs" --host 127.0.0.1 --port 6006
```

本机终端建立隧道（替换为现有 SSH 主机别名）：

```bash
ssh -N -L 18080:127.0.0.1:8080 -L 16006:127.0.0.1:6006 <SSH主机别名>
```

浏览器打开 `http://127.0.0.1:18080/` 与 `http://127.0.0.1:16006/`。
重点对照 `training_window/.../qualified_primary_event_credit`、违规率、
优化器学习率、PPO KL/clip/loss 和 advantage，而不是只看总奖励或红方成功率。
独立对抗效果以 selection/test 数据为准，训练窗口曲线只解释学习过程。
