# v3：验证对抗训练与算法优势的完整协议

本版目标是让优势能够被公平检验，不保证MUT-MAPPO胜出。正常、零收益和负收益都保留。Highway结果用于算法主实验；迁移到CARLA的是方法定义、网络结构和评价原则，性能结论仍需CARLA复验。既不把接口测试当性能证据，也不要求直接迁移策略权重。

## 1. 与旧实验区分

- 旧实验：task未指定research_protocol或为legacy_v2，局部24维/critic28维，原有奖励与结束条件保留。
- 新实验：`route_relative_paired_attack_v3`，局部25维/critic29维，包含时钟、路线相对坐标与新目标。需从头训练，旧checkpoint不能改标签后混入主表。
- 新入口：训练仍用train/batch，但配置必须是 `configs/research_v3/normal` 或 `configs/research_v3/attack`；配对评估与汇总使用 `python -m highwaytest.research ...`。
- 不要再用旧 `configs/server_pilot` 来验证新协议。旧运行目录/源码副本保留，不覆盖。

## 2. 三条件配对

|条件|红车|蓝车|用途|
|---|---|---|---|
|C0|冻结正常控制器|无|自由参考|
|C1|同一冻结正常控制器|冻结正常IDM或另行冻结的正常驾驶策略|普通交通对照|
|C2|同一冻结正常控制器|当前待评估策略|对抗效果|

同案例使用相同红车初始位置、速度、朝向；C1/C2蓝车初始状态也相同，程序会校验，误配会报错。默认红车IDM目标18m/s、允许MOBIL安全换道；正常蓝车独立固定为目标18m/s、允许安全换道。修改红车规则不会连带修改正常蓝车规则。

固定物理时窗T=20s。蓝车到达180m后继续行驶至T，不能通过提前结束获得不同的观察窗口。C2碰撞/出界等安全失败可提前结束，标记censored：原始C1–C2距离差为null，有效攻击效用为0；不伪造剩余时间的轨迹。

C0/C1普通交通必须完成校准，C1蓝车也须达到任务目标。若参考失败，记录baseline_invalid_count，主表汇总直接拒绝；不能看攻击结果后选择性移除难案例。

## 3. 安全资格与主指标

资格条件：完整时间窗、两蓝车各前进至少180m、全程无四车碰撞/驶离道路、蓝车无持续超速（默认>22m/s持续超过1s）、无逆向朝向或明显倒退。判定在物理子步累计，事件一旦出现不会因后续恢复而清除。

这里是明确的实验资格，尚不是城市道路全部交规：Highway采用车辆中心的on_road判定；CARLA需要Driving-lane地图及同步碰撞传感器。复杂路口规则/完整车身越界需要扩展后另行冻结协议。

案例主效用：

```
U = 资格合格 * clip((红车C1平均进度 - 红车C2平均进度) / 蓝车目标距离, 0, 1)
```

报告所有案例平均U，失败样本计0，不能只平均成功攻击。同步报告资格成功率、碰撞/出界/超速、有效额外距离损失和各场景族结果。

`C0-C1`记录自然交通损失；`C1-C2`记录额外攻击效应。TTC和明显制动事件作为诊断，不把简单纵向TTC自动称为高质量危险交互或因果避险。

## 4. 新训练目标

为与评价对应，训练也使用正常蓝车C1参考，而非只与无蓝车C0比较。定义：

```
Phi_drive_i = clip(本车路线进度 / 目标距离, 0, 1)
Phi_attack  = clip((C1红车进度-C2红车进度) / 目标距离, 0, 1)
r = (gamma*Phi_drive_next-Phi_drive_previous)
  + attack_weight*(gamma*Phi_attack_next-Phi_attack_previous)
  + 小速度奖励 - 居中/朝向/时间成本 - 安全失败成本
  + 合格完成奖励
```

安全失败时下一驾驶/攻击势函数都置0，返还先前取得的势函数收益；到T仍未完成蓝车任务时，攻击势函数置0。按gamma折扣的攻击增量严格望远镜相消，不能反复制造/解除阻挡刷正奖励。正常驾驶attack_weight=0，对抗组候选为1.0，所有算法组相同。

速度奖励整个episode上界0.1；居中/朝向/时间项按决策数归一化；合格完成+0.5；默认安全失败成本碰撞10、其他5。配置检查折扣后的最小安全失败成本必须大于总速度正奖励预算。不要将这套新奖励数值与旧版return直接比较。

这是新定义的有限时窗目标，不宣称保持旧奖励的最优策略不变。有限时窗作为真正terminal处理，不再当无限时域time-limit bootstrap。

## 5. 方法与控制组

|组|互学习|通信|
|---|---|---|
|G0|无|无|
|G1|普通互学习|无|
|G2|质量门控互学习|无|
|G3|无|有|
|G4|质量门控互学习|有|
|G5|普通互学习|有|
|G6_random|打乱本次证据门控样本|无|
|G7_strength|全样本使用本次门控平均权重|无|

G6/G7匹配的是各自run当前update的证据覆盖，不保证跨run的覆盖历史或实际KL完全相等。不能把这些控制夸大成严格等KL实验；报告实际gate_fraction、mutual_kl和完整曲线。

教师需在配对gate案例中quality更好、安全违规率不高于学生/阈值、正常任务合格率达到门槛。选模同样先检查违规率<=12.5%、资格率>=75%、普通参考有效，再按效用选择；没有合格模型时保存诊断候选，但selection.eligible=false，不能冒充合格主结果。

网络主干不依赖仿真器：

- 局部输入25维：自身速度/车道误差/路线朝向/进度/角色/时间 + 三邻车的自车坐标位置速度、可见性与角色。
- actor：34→128→128→2（25维自身+8维消息+1维可用性）；消息编码器25→128→128→8；critic29→128→128→2。
- 物理动作：有符号加速度±5m/s²、车轮转角±0.25rad。CARLA再用经标定的闭环控制转换为油门/刹车/归一化转向。
- 互学习保持同角色、同输入、student||stopgrad(teacher)的高斯KL；消息编码器端到端训练，教师使用自己的消息网络重算。
- 选中模型记录policy哈希及其历史非零互学习更新数，识别“组名不同但选中同一策略”或“胜出模型没有实际参与互学习”。

## 6. 服务器流程

将新包解压到新的项目副本，保留原训练源码。以下均在包含highwaytest目录的父目录、激活原虚拟环境后执行。依赖不变，仍使用现有requirements-server.txt。

```bash
python -m unittest highwaytest.tests.test_contracts highwaytest.tests.test_research_v3 -v
python -m highwaytest check --configs configs/research_v3/normal
python -m highwaytest check --configs configs/research_v3/attack
python -m highwaytest.research validate --config configs/research_v3/attack/G0_s42.json --output runs/research_v3/calibration.json
```

配置已附带：normal有3份G0配置，attack有8组×3seed=24份，每run100k，总计2.7m训练步，验证另计。候选参数不保证收敛；先跑一组短边界检查，再决定完整预算：

```bash
python -u -m highwaytest train --config configs/research_v3/normal/G0_s42.json --output runs/research_v3/normal/G0_s42 --max-rounds 5 --preview
# 检查TensorBoard/预览后继续；batch可以恢复上面未完成的G0。
python -u -m highwaytest batch --configs configs/research_v3/normal --output runs/research_v3/normal --jobs 2
python -u -m highwaytest batch --configs configs/research_v3/attack --output runs/research_v3/attack --jobs 2
```

主评价默认C1为固定IDM，普通驾驶训练组作为额外负对照。先只读selection检查方案。冻结方法/超参后，再做最终test：

```bash
for seed in 42 43 44; do
  python -m highwaytest.research evaluate --checkpoint "runs/research_v3/normal/G0_s${seed}/selected.pt" --split test --cases-per-family 24
  for group in G0 G1 G2 G3 G4 G5 G6_random G7_strength; do
    python -m highwaytest.research evaluate --checkpoint "runs/research_v3/attack/${group}_s${seed}/selected.pt" --split test --cases-per-family 24
  done
done
python -m highwaytest.research report --runs runs/research_v3 --output runs/research_v3_summary
```

输出总表CSV/JSON、逐seed逐场景族CSV，及配对训练seed差值和探索性95%t区间。主比较包含攻击G0 vs 正常G0、G2 vs G0/G1、通信边际效应、门控机制控制、2×2交互 `(G4-G2)-(G3-G0)`。区间以独立训练seed为单位，不以A/B或评估episode伪造样本量，未做多重比较校正，不自动宣布优越。

固定模型通信诊断示例（正式实验应对G3/G4各seed执行）：

```bash
python -m highwaytest.research evaluate --checkpoint runs/research_v3/attack/G4_s42/selected.pt --split test --intervention zero
python -m highwaytest.research evaluate --checkpoint runs/research_v3/attack/G4_s42/selected.pt --split test --intervention stale
python -m highwaytest.research evaluate --checkpoint runs/research_v3/attack/G4_s42/selected.pt --split test --intervention drop --drop-probability 0.3
```

加 `--gif` 可导出第一个案例，训练预览仍用 --preview 和serve-preview。TensorBoard增加qualified_attack_utility、baseline_valid_rate、c2_censored_rate等验证指标。

可选学习型C1：给所有组同一seed指定相同 `--normal-checkpoint runs/research_v3/normal/G0_s42/selected.pt`，必须先确认该策略合格。将这类评估用 `--output` 放在独立根目录，避免与IDM-C1结果混合。红车固定车道/换道敏感性使用 --red-lanes fixed/change，同样在不同评估根目录汇总。

若想生成其他预算的矩阵，用新目录：

```bash
python -m highwaytest.research plan --output configs/research_v3_custom --steps 100000 --seeds 42 43 44
```

## 7. 如何判断结果

先检查正常参考与候选策略资格。再看攻击组是否相对正常驾驶产生额外效应，最后比较MUT/普通互学习/MAPPO的跨seed差值。若G2没有超过G0，保留结论；若G4与G3选中policy哈希相同，不算组合增益；若只有更弱的KL改善，不能独占归因质量门控。

Highway上的优势最多支持该任务分布。CARLA复验方案与真实接入边界见 ALGORITHM_TRANSFER.md。当前只完成工程检查与短闭环，尚未产生任何新算法优势结论。
