# v4：一红两蓝瓶颈服务器实验

入口：`python -m highwaytest.bottleneck`。独立新实验，旧v3/staged_v1结果保留，不能直接续入旧权重或混入主表。

## 当前实现

- 真实3辆车：红车ID0，蓝车ID1/2。道路上游2车道，x=120–220m为平滑收窄段，下游1车道，共同蓝车目标线x=300m。
- 两场景族lead_merge/rear_merge：蓝1在红车前18–24m，蓝2在蓝1前34–44m或后60–70m的另一车道。后车保留安全初始时距，仍在80m通信范围内。红车IDM固定控制器，案例目标速度16/20m/s，允许正常MOBIL换道；同案例的C0/C1/C2参数严格一致。
- lead_merge中蓝1可见红车、蓝2看不到红车，本地感知25m、通信80m。配对信息检查保持蓝2观测相同、隐藏红车速度不同；这证明信息互补结构存在，不证明学得通信有效或必要。
- 资格：25s固定时窗，两蓝均到下游目标线；全程无碰撞/出界/超速/逆向；每蓝纵向进度至少为同案例正常IDM C1的85%，每蓝低于1m/s累计不超过3s。短暂合法减速允许，不以停车刷高分。
- U：满足资格时，clip((C1红车进度-C2红车进度)/180m,0,1)，否则0。提前失败原始距离差为null，不外推轨迹。C0记录自由交通，C1扣除普通瓶颈交通本身的影响。
- 局部观测25维：自身7 + 两邻车各6 + 道路6；集中状态25维，动作仍为有符号加速度±5m/s²、转向±0.25rad。actor 34→128→128→2、消息25→128→128→8、critic25→128→128→2；独立角色actor与A/B学习器沿用。
- 门控仍是场景族级：校验A/B初始状态/红车控制器指纹，比较整段质量与安全；蒸馏KL在同一学生观测、角色和通信掩码上，教师用自己的编码器重算消息。没有实现/声称逐状态反事实门控。

日志A/B表示两套团队学习器，每套都控制两辆蓝车；blue_0=物理蓝1，blue_1=物理蓝2。校准script A/B表示仅让物理蓝1/蓝2低速运行，与学习器A/B名称不同。

## 上传与检查

将highwaytest_server_bottleneck_v4.zip上传到/root/。使用新父目录：

```bash
conda activate carla-mappo
mkdir -p /root/carla-project/PythonAPI/examples/highway_exp_bottleneck_v4
cd /root/carla-project/PythonAPI/examples/highway_exp_bottleneck_v4
python -m zipfile -e /root/highwaytest_server_bottleneck_v4.zip .
python -m pip install -r highwaytest/requirements-server.txt
python -m unittest highwaytest.tests.test_contracts highwaytest.tests.test_research_v3 highwaytest.tests.test_staged highwaytest.tests.test_bottleneck -v
```

所有命令从highwaytest父目录执行。不要覆盖仍在运行的旧服务器源码。

## seed42分步操作

```bash
python -m highwaytest.bottleneck prepare --output runs/bottleneck_v4_s42 --seed 42 --normal-steps 100000 --attack-steps 200000
python -u -m highwaytest.bottleneck calibrate --study runs/bottleneck_v4_s42 --gif
python -u -m highwaytest.bottleneck bootstrap --study runs/bottleneck_v4_s42
python -u -m highwaytest.bottleneck normal --study runs/bottleneck_v4_s42
python -u -m highwaytest.bottleneck export --study runs/bottleneck_v4_s42
python -u -m highwaytest.bottleneck attack --study runs/bottleneck_v4_s42 --jobs 2
python -m highwaytest.bottleneck status --study runs/bottleneck_v4_s42
```

校准包含普通IDM、仅蓝1对抗、仅蓝2对抗、双蓝对抗4种脚本以及信息互补检查。要求普通参考有效、正常C2合格率100%且U=0、双蓝至少有正合格U，再允许后续训练。报告dual_exceeds_best_single，仅作脚本诊断；即使双蓝更好也不能当作通信已有效。

校准覆盖至少与gate相同的每族8个案例，默认16例；改变校准数量、配置、源码时应新建study，不静默复用旧校准。失败立即停止，留出报告，不自动降低门槛。

BC重新采集瓶颈IDM驾驶示范，默认64回合（最多8000联合决策样本）、拟合40轮，并在gate/selection每族检查>=87.5%通行合格与<=6.25%违规。普通PPO继续100k步，使用合格selected导出同一个初始化给全部攻击组。

攻击权重0.1→0.3→1：每级最少10k步，A/B连续两次通过门槛才晋级，退化时降级。没有在权重1训练的selected是eligible=false诊断模型；预算完成不等于实验成功。脚本不会强迫门控打开或方法胜出。

## 一键后台运行

```bash
mkdir -p highwaytest/runs
nohup bash highwaytest/scripts/run_bottleneck_s42.sh > highwaytest/runs/bottleneck_s42_pipeline.log 2>&1 &
tail -f highwaytest/runs/bottleneck_s42_pipeline.log
```

已激活的环境会被后台进程继承。默认只跑seed42的8组，最多并发2组；JOBS=1可降低并发。不要对同一study同时启动多个流水线。

普通PPO/对抗阶段支持相同命令恢复，完成的攻击组跳过；BC初始化目前需一次完成，中断且没有driver.pt时保留目录、另建study。旧v3的G3中断不自动迁移；新PPO只为这类批量浮点误差增加受限逐行回查，真正策略不一致仍会中止，并记录误差。

总预算为一次共享BC + 100k普通PPO + 8×200k攻击PPO，即1.7m训练决策步，BC/验证另计。预算是预实验候选，不保证算法收敛或优势。

## 看效果

```bash
tensorboard --logdir highwaytest/runs/bottleneck_v4_s42 --host 127.0.0.1 --port 6006
```

主要看validation/selection/{A,B}/下的qualified_attack_utility、success_rate、bad_rate、blue_min_progress_retention、blue_min_progress_m和两车stopped_s；同时看curriculum/attack_weight以及train/{A,B}/gate_fraction、mutual_kl、message_mean_effect。

calibration.gif在study根目录；训练真实轨迹预览可在normal命令增加--preview，仍用现有serve-preview。选中模型GIF：

```bash
python -m highwaytest.bottleneck evaluate --checkpoint runs/bottleneck_v4_s42/normal/selected.pt --split selection --gif --output runs/bottleneck_v4_s42/diagnostics/normal.json
python -m highwaytest.bottleneck evaluate --checkpoint runs/bottleneck_v4_s42/attack/G4_s42/selected.pt --split selection --gif --output runs/bottleneck_v4_s42/attack/G4_s42/evaluations/selection.json
```

JSON与GIF同名同目录，输出已存在时拒绝覆盖。所有组统一检查后可汇总：

```bash
for group in G0 G1 G2 G3 G4 G5 G6_random G7_strength; do
  python -m highwaytest.bottleneck evaluate --checkpoint "runs/bottleneck_v4_s42/attack/${group}_s42/selected.pt" --split selection --cases-per-family 24 --output "runs/bottleneck_v4_s42/attack/${group}_s42/evaluations/selection_24.json" || break
done
python -m highwaytest.bottleneck report --runs runs/bottleneck_v4_s42/attack --output runs/bottleneck_v4_s42/selection_summary --split selection
```

主表只指向attack：normal是共享准备，预算/课程不同，不与攻击组混表。report按v4协议过滤，不混v3。冻结场景/参数后再对独立test统一评价；随后增加43/44等独立训练seed。通信可用--intervention zero/stale/drop作固定模型诊断，但仍需独立无通信组证明增益。

## 可迁移边界

奖励原则、角色契约、SI动作、网络/互学习/通信和训练器可供CARLA方法复验。当前v4 Highway使用走廊坐标及固定道路几何，真实CARLA需将世界坐标投影到对应路线并适配三车角色、闭环控制和同一参考评价。本次未启动真实CARLA，也不直接迁移Highway权重。
