# 服务器 R2 奖励同步与试跑（2026-11-01）

目标是同步本地最新 normal_driving_critical_team_v2（R2）及依赖，保留远程俯视监控。
不合并本地开发分支，不上传本地实验运行历史，不修改红方模型。

## 修改前发现的问题与处理

- 原pilot.prepare固定读取本地v2 base_config；原prepare_v2固定读取v3完成状态、结果marker和授权文件。服务器新增独立prepare入口及可移植模板，不运行这些本机脚本。
- 模型路径为Windows绝对路径，references也绑定旧实验；新入口显式接收checkpoint/output/reference路径。
- seed/device/预算/RPC参数在本地先导固定；服务器入口接受参数，默认保持seed42、B0/off、24000ticks、cuda、127.0.0.1:2000。
- 原pipeline没有透传web监控；服务器直接调用带监控参数的已有reference、supervise、critical evaluator，不引入另一个写死路径的controller。
- 本地训练与参考入口缺少远程监控hooks；选择性三方合并，保留HTTP服务、俯视跟随及生命周期。
- 保留服务器现有的CARLA_ROOT/Linux地图寻找；v4生成器支持--source-manifest、--xodr、--output-dir。正常试跑使用冻结v4清单，无需重生成。
- 测试原本依赖本地私有红方模型；改成显式合成模型，不把真实训练效果当测试断言。

## 与本地对应的实验

R2 profile=normal_driving_critical_team_v2；四车、两个蓝方学习团队A/B，外层seed42，B0互学习关闭。
红方固定，蓝方Actor从红方基础Actor初始化，Critic/优化器/ValueNorm新建。是新奖励实验，不续训旧奖励run。
默认训练24000实际train world ticks（含setup，完整A+B边界soft cap）。
C1为lane_pid_v1普通交通控制，C2始终是蓝方Actor；不向C2加避碰屏蔽控制器。
冻结场景interaction_small_v4共96例，每split24，含前/后方及对角交互，完整镜像和四方向。
R2新增中等TTC带有界信用、一次持续事件奖励，强化已有正常驾驶成本；不是以碰撞或越界为攻击成果。
候选奖励沿用本地设置，尚无正式效果保证；正常驾驶资格与独立危险性应共同查看。

本地对应红方为u001000.pt，SHA256：
0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a

要复现实验，请上传该模型及同名.pt.sha256。换checkpoint可复用准备入口，但成为不同红方实验。
上传文件名可自定，CHECKPOINT参数与sidecar名称须一致。prepare验证现有sidecar，不伪造上传校验。

## 服务器操作

停止使用同一CARLA的旧任务后拉代码；不要让新旧训练同时操作一个世界：

```bash
cd /root/carla-project
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
screen -S critical-r2
```

在screen内，保持用户启动的CARLA 0.9.16运行在RPC端口2000，且无其他训练/评估占用。
以下仅为项目位置与参数示例，可改变量，不改变脚本源码：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
CHECKPOINT="artifacts/red_u1000.pt"
EXP="experiments/generated/R2_server_$(date +%Y%m%d_%H%M%S)"

python -m utils.prepare_critical_reward_server \
  --checkpoint "$CHECKPOINT" \
  --expected-sha256 0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a \
  --output-dir "$EXP" \
  --device cuda --seed 42 --train-ticks 24000 \
  --host 127.0.0.1 --port 2000

CONFIG="$EXP/config.json"
printf 'export EXP=%q\nexport CONFIG=%q\n' "$EXP" "$CONFIG" > "$EXP/run.env"
```

prepare只做离线模型/配置检查，PREPARED_NOT_TRAINED不代表训练已运行。
输出目录必须是新目录。默认不需要任何本地v3/v4实验历史。
本地源模型的gamma与R2有界信用约束不兼容时会明确拒绝，不静默修改奖励。

第一步，重新生成服务器真实参考（不能复用本机绝对路径或伪造参考身份）：

```bash
python -u -m evaluation.mut_mappo_reference \
  --config "$CONFIG" --splits train gate selection \
  --repeats 2 --allow-weak-red \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

--allow-weak-red明确选择与本地相同的弱红方工程诊断。失败C1不会被重新标为合格，
不要求此入口先满足95%，也不降低正式筛选门槛。72参考案例×2重复的成本单独保留。

第二步，在screen前台运行监督器（内部训练子进程日志位置会打印）：

```bash
python -u -m utils.mut_mappo_session supervise \
  --config "$CONFIG" --run-dir "$EXP/run" \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

先完成references再训练。不要在本轮训练中编辑源代码、task配置或参考曲线。
Ctrl+A再D脱离；screen -r critical-r2恢复。需要恢复shell变量时source "$EXP/run.env"。

读取状态，无需CARLA连接：

```bash
python -m utils.mut_mappo_session inspect --run-dir "$EXP/run"
```

只有同一代码/配置/参考身份、已保存完整边界的中断任务，才用相同supervise命令加--resume。
新奖励或换红方模型不能续训旧run。原生崩溃先检查日志和CARLA，不循环盲重启。

第三步，训练正常结束后冻结评估所有A/B候选：

```bash
python -u -m utils.evaluate_mut_mappo_critical live \
  --config "$CONFIG" --checkpoint "$EXP/run/latest.pt" \
  --split gate --output-dir "$EXP/gate_eval_$(date +%Y%m%d_%H%M%S)" \
  --reuse-idle-world \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

结果在该gate_eval目录的analysis/index.html、teams.csv、episodes.csv、events.csv和critical_report.json。
相同空闲地图复用用于减少反复加载地图造成的资源增长，仍检查动态actor残留；不增加隐式tick。
先用gate诊断，不根据test调奖励。结果必须同时看raw/qualified危险产出、红方变化和蓝方驾驶质量。

监控沿用SSH转发：服务器8080，本地通常18080。浏览器网页服务持续运行不代表训练仍在运行。
需要新模型、预算、seed、主机或场景时改prepare参数或--source-config，不修改已运行实验的配置。

配置模板中的模型为占位，只通过prepare传入实际checkpoint后使用。高级路径参数还包括--case-manifest、--reference-dir、--c1-controller、--source-config。
场景重生成默认读取v4目录内source_v3_matrix.json冻结输入，不依赖本机历史实验；该输入仅用于构建，运行清单为cases.json。
端点锁按传入host/port区分，默认localhost与127.0.0.1共用旧锁；全机进程审计仍保守拒绝其他训练，本次不引入并行多开。
验证：102项离线/合成更新链测试通过；真实u1000模型已离线准备并逐字段核对R2配置。未连接CARLA，未替用户启动服务器训练。
