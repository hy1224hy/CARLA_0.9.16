# 服务器初步实验与无窗口可视化（v2）

上传整个 `highwaytest` 文件夹即可，不需要 CARLA 引擎、原 mappo_training 或 Windows venv。忽略 `runs/` 和 `__pycache__/`；旧 v1 短测 checkpoint 不兼容 v2 的28维 critic。本版已在本机 Python3.10/CPU、无窗口模式下验证，尚未在你的 Linux 服务器实测。

## 1. 安装、预检

以下以 Linux、项目父目录 `~/highway_exp` 为例。先把文件夹放成 `~/highway_exp/highwaytest`，所有命令在父目录执行。

```bash
cd ~/highway_exp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r highwaytest/requirements-server.txt
export MPLBACKEND=Agg
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m highwaytest check --configs configs/server_pilot
python -m unittest highwaytest.tests.test_contracts -v
```

Python要求3.10或以上；建议使用支持上述固定包版本的3.10/3.11。已装有合适 PyTorch 时可保留，但不同依赖版本应作为不同运行环境记录，不宣称跨平台逐位复现。初步实验固定为 CPU 训练，每个run一个环境、一个PyTorch线程。代码不会自动使用GPU；用服务器多CPU核并行多个run更直接。先用 `--jobs 2`，根据CPU/RAM占用再调整。

`check` 会验证依赖、固定 Highway-env/Gymnasium 版本、配置合法性、重复group/seed、公平预算/超参、24维局部观测、28维全局状态，并真实执行环境一步与无窗口渲染。检查通过不代表算法已收敛。

## 2. 先检查正常驾驶能否学习

建议使用tmux保持SSH断开后的进程：

```bash
tmux new -s highway
source ~/highway_exp/.venv/bin/activate
cd ~/highway_exp
python -u -m highwaytest train --config configs/driving_pilot.json --output runs/driving_pilot_G0_s42
```

这个配置为正常20s任务、G0、seed42、A+B合计20,000训练决策步，`attack_weight=0`。结束后先用选模集检查行为，暂不反复读取最终测试：

```bash
python -m highwaytest evaluate --checkpoint runs/driving_pilot_G0_s42/selected.pt --split selection --cases-per-family 8 --gif
```

若没有出现正常通行，先看回合失败原因、成功率、动作饱和、critic解释方差及GIF，再决定是否延长所有组预算或调整共同任务。20,000步是学习趋势探针，不是收敛承诺。

## 3. 跑五组初步对照

`configs/server_pilot/` 为 G0–G4、相同seed42、每组20,000步、相同对抗奖励0.3的五个配置。它们都从头初始化，五组总计100,000训练决策步，验证消耗另记；这是单seed探索结果。

```bash
python -u -m highwaytest batch --configs configs/server_pilot --output runs/server_pilot --jobs 2
```

该命令只训练，不自动多次运行最终测试。为每组创建独立子目录、console日志、状态与checkpoint。重复运行相同命令会跳过完成组、从未完成组的 `latest.pt` 恢复；配置或源码不同则拒绝。失败不妨碍其余已排队组继续；最后返回非零并列明失败组。已有目录但没有任何完整checkpoint时，使用新批次目录，保留原失败现场。OS文件锁会阻止两个进程写同一个run/batch。

批次计划：`runs/server_pilot/batch_manifest.json`；执行结果：`batch_status.json`；实时日志：`G0_s42.console.log` 等。每个run的 `status.json` 表示最近提交状态，进程被强杀后该文件可能仍显示running，应结合console和进程状态判断。完整A/B边界每轮提交一次。

单组恢复：

```bash
python -m highwaytest train --resume runs/server_pilot/G4_s42/latest.pt
```

恢复要求配置和Python源码一致；源码快照采用规范换行哈希，避免仅Windows/Linux换行差异阻止恢复。依赖差异仍可能改变数值，服务器实验应从头启动并固定环境。原v1的本地短测不用于服务器续训。

正式三seed矩阵位于 `configs/adversarial/`，每组100,000步，总计1,500,000步，另加验证开销；初步结果可信且任务冻结后再运行。若使用正常驾驶预训练，各组必须采用相同预训练协议与同seed来源，不能把从头组与预训练组直接归因互学习/通信。

## 4. 浏览训练曲线：TensorBoard + SSH

服务器另开一个终端，进入项目父目录并激活同一venv：

```bash
tensorboard --logdir highwaytest/runs --host 127.0.0.1 --port 6006
```

本地电脑另开终端，把 `user@server` 替换成实际SSH地址：

```bash
ssh -N -L 6006:127.0.0.1:6006 user@server
```

在本地浏览器打开 [http://localhost:6006](http://localhost:6006)。不需要服务器桌面、X11或图形卡输出。若本地6006被占用，可用 `-L 16006:127.0.0.1:6006`，浏览器打开16006端口。

横轴统一为 **A+B合计训练决策步**，同一轮A/B标在相同预算位置。主要标签：

|标签|用途|
|---|---|
|`train/A/mean_step_reward`、`train/B/...`|训练每步回报，仅作学习过程诊断|
|`rollout/A/return`、`return_agent_0/1`|完整训练回合累计回报，不包含被采样边界主动丢弃的尾段|
|`rollout/A/success`、`collision`、`offroad`|完整回合任务与失败比例|
|`validation/selection/A/success_rate`、`quality`|固定、独立于训练轨迹的选模表现；优先看这些|
|`validation/selection/A/red_delay_m`、`blue_progress_m`|同时检查对手受损与己方正常行驶|
|`train/A/explained_variance`、`value_loss`|critic是否学到有用回报预测|
|`train/A/approx_kl`、`clip_fraction`、`ratio_error`|PPO更新与动作概率一致性|
|`train/A/mutual_kl`、`gate_fraction`、`gate/A/parallel`|互学习强度与门控是否实际开启|
|`train/A/message_mean_effect`、`link_fraction`|消息置零前后均值变化与链路覆盖，不单独证明通信收益|
|`train/A/action_saturation`、`policy_std`|控制饱和与探索幅度|
|`budget/validation_steps`|验证额外开销，不能漏计到耗时比较|

标签A/B均独立记录，回合指标另按场景族记录。没有完整回合的轮次不填伪造的零值。TensorBoard仅在完整checkpoint提交后写入；崩溃恢复时从JSONL重建已提交曲线，半行写入和未提交尾段会被清理。TensorBoard可能提示purging expired events，这是预期的去重重建，不是重新计步。

## 5. 看车辆行为：无窗口导出GIF

```bash
python -m highwaytest evaluate --checkpoint runs/server_pilot/G4_s42/selected.pt --split selection --cases-per-family 8 --gif
```

输出 `highwaytest/runs/server_pilot/G4_s42/evaluations/selection_normal_0.json` 和同名GIF。GIF展示第一个评估案例；它是简化俯视图，红色ID0/1为固定IDM，蓝色ID2/3为学习车辆，标有速度、车辆朝向、红车延迟、蓝车进度和失败状态。其余案例数值均在JSON中。训练不会弹出窗口，也不会每步生成大量图像。

本地下载示例：

```bash
scp user@server:highway_exp/highwaytest/runs/server_pilot/G4_s42/evaluations/selection_normal_0.gif .
```

已有报告不会覆盖。想另跑探针时用 `--output runs/server_pilot/G4_s42/evaluations/selection_probe2.json`。源码与checkpoint不同默认拒绝评估；显式 `--allow-source-change` 会记录变化，汇总程序拒绝混入这类结果。

## 6. 冻结方案后的最终测试与汇总

初步筛选先看selection，任务/奖励/配置冻结后，再对五个选定模型各做一次test。以下在服务器Bash运行：

```bash
for group in G0 G1 G2 G3 G4; do
  python -m highwaytest evaluate --checkpoint "runs/server_pilot/${group}_s42/selected.pt" --cases-per-family 24
done
python -m highwaytest report --runs runs/server_pilot --output runs/server_pilot/comparison
python -m highwaytest evaluate --checkpoint runs/server_pilot/G4_s42/selected.pt --intervention zero
python -m highwaytest evaluate --checkpoint runs/server_pilot/G4_s42/selected.pt --intervention stale
python -m highwaytest evaluate --checkpoint runs/server_pilot/G4_s42/selected.pt --intervention drop --drop-probability 0.3
```

汇总检查源版本、训练总预算、超参、奖励、场景和预训练方式，同seed各组还检查预训练checkpoint哈希。单seed不输出伪造标准差、不支持显著性结论。A/B不能算两次独立重复。

本版两类简单场景未必需要强通信，因此通信组无收益也是可能结果。优先判断普通MAPPO是否学会正常行驶，再判断互学习是否有实际启用，最后做通信干预和有/无通信训练对照；不要只比较组合G4与G0就宣称两个模块都有效。

## 7. 正式三seed流程

正常驾驶和五组探索确认任务可学习、指标/奖励冻结后，在新目录运行正式矩阵。不要把探索结果和正式结果放在同一个汇总根目录。

```bash
python -m highwaytest check --configs configs/adversarial
python -u -m highwaytest batch --configs configs/adversarial --output runs/formal --jobs 2
```

G0–G4 × seeds42/43/44，每组100,000训练决策步。训练结束且选模规则冻结后：

```bash
for seed in 42 43 44; do
  for group in G0 G1 G2 G3 G4; do
    python -m highwaytest evaluate --checkpoint "runs/formal/${group}_s${seed}/selected.pt" --split test --cases-per-family 24
  done
done
python -m highwaytest report --runs runs/formal --output runs/formal/comparison
```

有通信的G3/G4再分别做固定模型干预：

```bash
for seed in 42 43 44; do
  for group in G3 G4; do
    for mode in zero stale drop; do
      python -m highwaytest evaluate --checkpoint "runs/formal/${group}_s${seed}/selected.pt" --split test --cases-per-family 24 --intervention "$mode" --drop-probability 0.3
    done
  done
done
```

普通主表汇总只包含normal测试，干预报告另存各run的evaluations目录。所有评估均默认拒绝覆盖旧报告。需要重新执行整个流程时使用新的run/batch目录，恢复则使用原始源码/配置/依赖和latest checkpoint。

2026-11-07补充：汇总读取目录下JSON时已排除列表型批次状态/既有汇总，避免 `batch_status.json` 导致报错；包含该修复的上传包应在启动新实验前更新。已启动实验保持其原始源码副本。

## 8. 实时查看正在训练的车辆

新实验单组训练增加 `--preview`，会把真实采样中的画面写入该run的 `preview/`。这是实际随机探索动作的训练轨迹，不是另外启动的确定性评估。观察端只读PNG/JSON，不连接环境、不推进仿真、不修改策略随机流。

快速训练、定期查看最新画面：

```bash
python -u -m highwaytest train --config configs/driving_pilot.json --output runs/driving_live --preview
```

想看清车辆连续运动，建议先运行一个1000步观察实验，并用1倍仿真速度采样：

```bash
python -u -m highwaytest train --config configs/driving_pilot.json --steps 1000 --output runs/driving_live_slow --preview --preview-speed 1
```

`--preview-speed 0`（默认）不限制训练速度，画面会跳跃；`1`表示仿真时间大致对应墙钟时间；`2`表示两倍速度，`0.5`表示半速。慢速模式仅用于观察，1000步在1倍速下采样约200秒，另外还有PPO和验证耗时。100,000步若全程1倍速，单采样即约5.6小时。正式性能/耗时比较默认关闭预览。

等训练开始创建preview目录后，服务器另一个终端执行（激活同一个venv、仍在项目父目录）：

```bash
python -m highwaytest serve-preview --run runs/driving_live_slow --port 8000
```

本地电脑执行：

```bash
ssh -N -L 8000:127.0.0.1:8000 user@server
```

本地浏览器打开 [http://localhost:8000](http://localhost:8000)，页面自动刷新。画面显示当前A/B团队、轮次、训练步数、车辆状态、动作、当前步奖励与帧更新时间。PPO更新和独立验证时暂停训练画面，并显示对应phase；快速模式可能跳过部分中间画面。默认批量入口不开预览，可视化观察请使用单组train命令。

预览选项不改变环境、奖励或总交互预算，但渲染和限速增加墙钟开销，启用记录保存在 `preview/sessions.jsonl`。新增功能改变了源码哈希；新实验使用更新包，已有实验保留原源码副本恢复。

## 9. 恢复中断与延长已完成实验

`train --resume` 恢复原定预算内的中断，不改变配置。已经完成100,000步的run再执行resume会直接结束，不会自动增加预算。

新版 `extend` 将完整训练状态延续到一个新run，原实验保留：

```bash
python -u -m highwaytest extend \
  --checkpoint runs/driving_G0_100k_s42/latest.pt \
  --steps 200000 \
  --output runs/driving_G0_200k_s42
```

`--steps 200000` 是累计A+B总预算，从100,000继续到200,000，即额外100,000步。只允许增加预算，不改变奖励、学习率、任务、组别或种子。必须提供latest.pt，selected.pt没有完整训练状态。

继承A/B模型、Adam optimizer及其学习率/动量、随机状态、冻结教师、门控、轮次、历史验证步数、最佳模型与选模分数；复制已提交JSONL，新的TensorBoard曲线从原预算继续。父目录不被覆盖。新manifest记录父checkpoint哈希与训练分段，汇总会区分一次性与分段训练，避免将不同延长安排混为完全相同协议。

延长后的中断仍用普通resume：

```bash
python -m highwaytest train --resume runs/driving_G0_200k_s42/latest.pt
```

若父checkpoint来自更新前的源码，默认拒绝延长。审查确认为兼容v2修改后，可在extend命令显式添加 `--allow-source-change`，变化将记录在血缘信息中；不绕过依赖一致性检查和严格模型加载。保留旧代码副本，不在仍运行的实验目录覆盖源码。此功能也适用于从20,000延长到100,000。

续训不重置optimizer，与 `--initialize-from` 的模型暖启动不同。由于旧预算终点可能有短rollout和额外教师刷新，分段训练不普遍保证与从头一次性运行更大预算逐位一致；所有对照组应采用相同分段预算。正式方案冻结前决定延长安排，不根据最终测试反复延长某一个算法组。
