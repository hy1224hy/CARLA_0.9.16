# 危险交互评估 v1：已有检查点补评估

只新增评估文件，不修改训练、奖励、task合同和旧检查点。无需重新训练。
旧快照无真实bounding_box，不能默认车辆长宽来补算精确TTC；优先用live补采集。
所有路径、分组、seed、协议和输出目录都从参数或实验manifest读取。

## 协议与物理定义

版本为critical_interactions_v1，完整配置：configs/evaluation/mut_mappo_critical_v1.json。
每次运行保存protocol.json，并在critical_report.json保存规范化配置及SHA256。
阈值通过诊断后冻结；本协议是诊断候选，不是安全认证或碰撞概率模型。

- 2D OBB TTC：当前实际位置、速度、yaw和bbox局部中心/extent/yaw，固定yaw、匀速预测5秒；用连续分离轴区间求首个接触时刻。不相交返回None，并由COMPLETE状态表示无预测冲突；数据缺失另标状态。
- 2D假设用于同一平面道路，不能直接处理立交跨层或大坡度车身投影；未实现转弯/加速度预测。
- TET：每个post-tick样本若最小红蓝TTC<=阈值，累计dt。TIT累计max(0,阈值-TTC)*dt。多车辆对同一时刻只用最小值，场景暴露不重复加倍。
- 阈值同时输出1.0/1.5/2.0/3.0秒，主要阈值1.5秒。dt=0.1秒必须与task吻合。
- predicted_conflict：同一红蓝对连续低于主要阈值至少0.3秒。相隔最多0.2秒的低TTC段可归为一个事件，但持续门槛取最长连续段，暴露仅计真实低TTC样本。
- geometric_contact：当前OBB接触/重叠，不受0.3秒门槛过滤。不是传感器碰撞或责任推断。
- 接触与预测冲突均计raw critical event，events.csv保留kind；每场景只计一次产出。
- 响应从实际post-tick速度差和control.brake读取。纵向减速度按当前车头方向投影；记录事件区间及结束后1秒内的峰值。3m/s²为项目诊断hard-brake候选阈值，并非文献统一标准。
- 保留蓝车既有加速度/jerk/方向变化等驾驶指标，未赋予未经标定的舒适性通过阈值。
- 删失按实际车辆对/红车观测终止标记，不能把剩余时长补零或延伸离场车辆。
- v1不实现PET、碰撞责任归因、可解性证明或因果制动归因。

严格资格G沿用blue_eligible（无collision/offroad/road_rule/task违规）。H是测得危险事件。
raw critical rate=mean(H)，qualified critical rate=mean(G*H)，分母为完整可测配对案例。
分别报告C1/C2的两种rate和差值、TET/TIT、观测时长、蓝车资格。
qualification_only_improvements：H两边都为真，仅G由false变true。该类不能解释为危险强度增强。
C1不安全时仍保留负增量，不把一次C1成功当成事先注册的可靠参考子集。
A/B、重复评估不是独立训练seed。不同red/case/stage/budget不能合并为显著性结论。

## 服务器使用（独占运行中的CARLA，screen内串行）

从mappo_training目录运行。先同步codex/server-remote-monitor分支。

```bash
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training

EXP="experiments/generated/你的实验目录"
OUT="$EXP/critical_eval_$(date +%Y%m%d_%H%M%S)"

python -m utils.evaluate_mut_mappo_critical live \
  --experiment "$EXP" --groups B0 B1 B2 --seeds 42 \
  --split gate \
  --protocol configs/evaluation/mut_mappo_critical_v1.json \
  --output-dir "$OUT" --validate-only

python -u -m utils.evaluate_mut_mappo_critical live \
  --experiment "$EXP" --groups B0 B1 B2 --seeds 42 \
  --split gate \
  --protocol configs/evaluation/mut_mappo_critical_v1.json \
  --output-dir "$OUT" \
  --monitor-mode web --monitor-host 127.0.0.1 --monitor-port 8080
```

组名/seed过滤可省略，默认评估实验全部entries。不是只评估入选团队，A/B都保留。
从每个run的checkpoint_manifest.json校验SHA并选择最新完整训练边界；原始红方和参考文件须存在。
validate-only不会创建输出或连接CARLA。live输出必须新目录，重复运行换OUT。
默认gate用于诊断，不反复用test调阈值。不改变旧selection、test和筛选规则。
本入口不要求C0/C1达到95%，不重新生成参考，不新增--allow-weak-red参数；弱基础策略需结合参考质量解释。
若只想先做最小检查，用--groups B0；通过采集检查后再运行全部组。
监控仍通过已有SSH转发在本地18080访问远程8080。

任意配置/历史检查点：

```bash
python -u -m utils.evaluate_mut_mappo_critical live \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --split gate --output-dir "$NEW_OUT" \
  --monitor-mode web --monitor-port 8080
```

模型通过原有校验加载，完整A/B边界和单团队冻结导出都支持。
多轮训练过程对比可对预先选定的历史边界分别运行，再用offline一次分析多个评估目录。
不会改变训练模型或引入新的训练阶段。

已采集真实bbox的新轨迹可离线重复分析（无需CARLA/PyTorch）：

```bash
python -m utils.evaluate_mut_mappo_critical offline \
  --evaluations "$EVAL_DIR_1" "$EVAL_DIR_2" \
  --protocol configs/evaluation/mut_mappo_critical_v1.json \
  --output-dir "$NEW_ANALYSIS"
```

EVAL_DIR是包含manifest.json/resolved_config.json/cases.jsonl/traces的目录，
例如live输出的evaluations/eval_000，不是旧summary.json文件。
缺bbox返回MISSING_GEOMETRY，缺帧/终止/身份不一致不视为零危险。
没有任何可测配对时仍保存诊断，但命令退出码为2，提示补采集。

## 输出与判读

live结果在OUT/analysis，offline直接在指定输出目录：

- index.html：C1/C2危险率、合格产出率和增量图表。
- teams.csv：所有候选统计、完整配对数、计划覆盖率、排除数、资格与危险增量。
- episodes.csv：逐回合TTC/TET/TIT、状态/原因/覆盖率、红车响应、蓝车驾驶指标。
- events.csv：红蓝车辆ID、事件kind、时刻、持续时间、最小TTC、红车响应窗与删失。
- pairs.csv：同案例、同team、同seed、同attempt的C1/C2对照。
- critical_report.json：全部结果、原始任务结果、协议hash和输入manifest。

raw rate/TET/TIT提升且蓝车资格没有明显恶化，才是危险生成增强的证据。
只有qualified rate提升而raw不变，可能仅驾驶质量改善。需结合blue驾驶指标判读。
红车成功率可保持或提高；成功应对危险仍有训练价值。TTC代理指标本身不证明因果或真实事故概率。

## 文献依据与项目选择

1. FHWA SSAM Final Report, Chapter 2 (2008): https://www.fhwa.dot.gov/publications/research/safety/08051/02.cfm
   原文：“SSAM utilizes a default TTC value of 1.5 seconds”。仅提供1.5秒参考，不是所有场景通用阈值。
2. Westhofen et al., Criticality Metrics for Automated Driving: https://arxiv.org/abs/2108.02403
   作者维护的定义页：
   TTC https://criticality-metrics.readthedocs.io/en/latest/time-scale/TTC.html
   TET https://criticality-metrics.readthedocs.io/en/latest/time-scale/TET.html
   TIT https://criticality-metrics.readthedocs.io/en/latest/time-scale/TIT.html
   TET为TTC低于阈值的时间积分，TIT为阈值与TTC差值的时间积分。本实现采用离散post-tick近似。
3. 5秒预测窗、0.3秒连续阈值、0.2秒合并间隔、1秒响应窗及3m/s²诊断阈值，均为项目候选配置，须在诊断数据上验证后冻结。
4. QCR及其原始/资格分解是本项目评估协议，不冒充文献中的统一评价标准。
