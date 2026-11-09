# staged_v1：保持通行能力的对抗驾驶

保留v3物理、20s时窗、两蓝车各180m、安全资格、C0/C1/C2和U，改变初始化、逐维探索和训练课程。新实验不能与旧v3混表，不保证算法胜出。

## 实际流程

- 共享64个训练专用IDM演示回合（最多6400个联合决策样本），含横向/朝向/速度恢复例子，行为克隆40轮。actor只用25维局部观测。
- 初始pre-tanh标准差：加速度0.3、转向0.08；范围分别[0.1,0.6]、[0.03,0.2]；熵系数0.001。
- 普通G0再训练100k步。BC模型和普通selected都须通过gate/selection每场景族>=87.5%合格率、<=6.25%违规率且参考全部有效，才允许发布初始化。
- 同一个普通selected复制到A/B，所有攻击组使用同一文件hash、重置优化器；A/B随机流不同。消息输入权重预训练时为0，打开通信不会突然改变初始行为，随后可学习。
- 8组各200k步，训练攻击权重0.1->0.3->1。每阶段至少10k步，A/B均合格率>=87.5%、违规率<=6.25%、参考有效，连续2次gate检查才晋级；连续2次不满足则降一级。默认每5轮检查。
- 评价保持同一U定义。未到完整攻击权重1的模型只有诊断资格，eligible=false。checkpoint保存课程状态；降级不会抹去历史合格选中的完整攻击模型。

所有组规则相同，但实际权重历史可能不同。须同时报告课程历史、完整攻击阶段步数、教师参与、共享BC数据/更新数及普通PPO成本，不能宣称实际奖励历史完全相同。

## 安装

新包解压到新目录，避免覆盖旧训练源码：

```bash
conda activate carla-mappo
mkdir -p /root/carla-project/PythonAPI/examples/highway_exp_staged
cd /root/carla-project/PythonAPI/examples/highway_exp_staged
python -m zipfile -e /root/highwaytest_server_staged_v1.zip .
python -m pip install -r highwaytest/requirements-server.txt
python -m unittest highwaytest.tests.test_contracts highwaytest.tests.test_research_v3 highwaytest.tests.test_staged -v
```

后续均在highwaytest父目录执行。Python相对runs/configs路径解析到highwaytest内部；shell/TensorBoard路径显式加highwaytest。

## seed42分步指令

```bash
python -m highwaytest.staged prepare --output runs/staged_v1_s42 --seed 42 --normal-steps 100000 --attack-steps 200000
python -u -m highwaytest.staged bootstrap --study runs/staged_v1_s42
python -u -m highwaytest.staged normal --study runs/staged_v1_s42
python -u -m highwaytest.staged export --study runs/staged_v1_s42
python -u -m highwaytest.staged attack --study runs/staged_v1_s42 --jobs 2
python -m highwaytest.staged status --study runs/staged_v1_s42
```

每步成功再执行下一步。资格失败会停止并保存qualification.json，不自动降低门槛。普通阶段可加--preview或--max-rounds；重复normal/attack命令可恢复，攻击已完成组会跳过。BC是短初始化任务，目前没有BC优化器中断恢复；中途退出且无driver.pt时保留证据，另建新study重新进行。

不要修改study配置、Python源码、依赖后原地恢复。旧v3不合格normal selected不能用作本流程初始化。

## 一键后台运行

激活虚拟环境后，可直接运行包内脚本；它依次执行上述阶段，失败即停。已有study会检查后恢复，不覆盖：

```bash
mkdir -p highwaytest/runs
nohup bash highwaytest/scripts/run_staged_s42.sh > highwaytest/runs/staged_s42_pipeline.log 2>&1 &
tail -f highwaytest/runs/staged_s42_pipeline.log
```

同一study只开一个流水线，不同时执行分步命令。JOBS=1可降低并发。预算为共享BC + 100k普通PPO + 8×200k对抗PPO，共1.7m决策步，BC和验证另计；这是候选预实验预算，不保证收敛。

## TensorBoard与GIF

```bash
tensorboard --logdir highwaytest/runs/staged_v1_s42 --host 127.0.0.1 --port 6006
python -m highwaytest.research evaluate --checkpoint runs/staged_v1_s42/normal/selected.pt --split selection --gif --output runs/staged_v1_s42/diagnostics/normal_selection.json
```

看validation/selection/{A,B}/下的success_rate、bad_rate、qualified_attack_utility、blue_min_progress_m、blue_0/1_progress_m、blue_0/1_stopped_s；看train/{A,B}/的acceleration_std、steering_std、gate_fraction、mutual_kl、attack_weight，以及curriculum/attack_weight。summary.json保存晋降级历史，selected.eligible标识是否真正合格。

训练完成后，只读selection预实验（输出存在时拒绝覆盖）：

```bash
for group in G0 G1 G2 G3 G4 G5 G6_random G7_strength; do
  python -m highwaytest.research evaluate --checkpoint "runs/staged_v1_s42/attack/${group}_s42/selected.pt" --split selection --gif --output "runs/staged_v1_s42/attack/${group}_s42/evaluations/selection.json" || break
done
python -m highwaytest.research report --runs runs/staged_v1_s42/attack --output runs/staged_v1_s42/selection_summary --split selection
```

GIF与指定JSON同目录同名。汇总只指向attack，避免把共享normal准备阶段混入不同训练程序的攻击主表。冻结方法/参数后再做独立test，不能用test调参。单seed仅作预实验。

简单匀速脚本在旧selection已有100%合格、U约0.201，应保留为移动阻挡正对照，并在独立案例复验。学习策略不能只凭U非零宣称优于简单规则。

方法可以在CARLA复验；本包没有验证真实CARLA控制标定或场景加载，也不直接迁移Highway权重。
