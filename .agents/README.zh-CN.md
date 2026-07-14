# CARLA MAPPO 论文实验子 Agent 使用说明

这份说明用于指导你在 `PythonAPI/examples/mappo_training` 项目中使用 6 个子 agent 推进大论文实验。

核心思想很简单：不要让一个 agent 同时负责“设计实验、审命令、跑 CARLA、分析结果、改奖励”。CARLA 训练成本高，论文实验又要求可复现，所以必须把职责拆开。

## 6 个子 Agent 分工

| 子 agent | 什么时候用 | 主要产出 | 默认权限 |
| --- | --- | --- | --- |
| `Experiment-Planner` | 准备一轮新实验前 | 实验矩阵、RunID、seed、依赖关系、命令草案 | 只读 |
| `Command-Auditor` | 真正跑训练/评估前 | 审核后的命令、参数风险、路径风险 | 只读 |
| `Scenario-Steward` | 设计训练/测试场景、held-out 场景时 | 场景清单、覆盖分析、几何风险 | 默认只读，可被授权改 JSON |
| `Reward-Metrics-Guardian` | 改奖励、改指标、写对抗结论前 | reward-metric 对齐表、hacking 风险检查 | 默认只读，可被授权改奖励/指标文件 |
| `Training-Runner` | 命令已经审核后 | 训练状态、日志摘要、ckpt 路径、TensorBoard 路径 | 可运行，不能改代码 |
| `Evaluation-Repro-Analyst` | 每次训练结束后 | 评估结果表、per-scenario 摘要、复现实验台账 | 可运行评估，默认不改训练代码 |

## 推荐工作流

### 1. 规划阶段

先启动 `Experiment-Planner`。

让它回答：

- 这轮实验要验证哪个假设？
- 最小实验矩阵是什么？
- 每个 run 的 RunID 是什么？
- 需要哪些 seed？
- 依赖哪个 checkpoint？
- 结果应该用哪些指标判断？

典型输入：

```text
请作为 Experiment-Planner，只读分析。
目标：设计 robust v2 合作基线 + 2红1蓝对抗快验的两周实验矩阵。
约束：先不做 PSRO；核心结论至少 3 seeds；需要 held-out 场景。
输出 RunID、脚本、关键参数、seed、save-dir、依赖和评估指标。
```

### 2. 命令审计阶段

把 `Experiment-Planner` 给出的命令交给 `Command-Auditor`。

它只负责判断命令能不能跑，不负责跑。

重点检查：

- 是否从 `PythonAPI/examples/mappo_training` 目录用 `python -m ...` 运行。
- 参数是否真的存在于脚本 parser 中。
- checkpoint 路径是否存在。
- `--save-dir` 是否会覆盖旧实验。
- 是否显式设置 seed。
- 是否需要 CARLA 已经启动。

### 3. 场景和奖励并行检查

这一步可以并行开两个子 agent：

- `Scenario-Steward`：检查训练场景、测试场景、held-out 划分。
- `Reward-Metrics-Guardian`：检查奖励项、指标项和 reward-hacking 风险。

只有当你明确说“可以改文件”时，它们才允许改：

- 场景 JSON 归 `Scenario-Steward`。
- `adversarial_reward.py`、`adversarial_metrics.py` 和相关测试归 `Reward-Metrics-Guardian`。

### 4. 训练执行阶段

只把已经通过审计的一条命令交给 `Training-Runner`。

一次只跑一条。不要让它临场改参数。

典型输入：

```text
你是 Training-Runner。可运行，但禁止改代码。

已审核命令：
python -m training.train_no_rllib_mappo_robust_v2 --updates 150 --seed 42 --save-dir logs/exp/A0_robust_v2_s42

cwd:
PythonAPI/examples/mappo_training

要求：
1. 先检查 save-dir 是否冲突。
2. 检查 CARLA 端口假设。
3. 只运行这一条命令。
4. 返回日志摘要、ckpt 路径、TB 路径和失败原因。
```

### 5. 评估与归档阶段

训练完成后，立刻交给 `Evaluation-Repro-Analyst`。

它要做两件事：

1. 用正式评估脚本跑结果，不把训练窗口指标当论文结果。
2. 写复现实验台账，记录命令、seed、checkpoint、场景集、结果路径和关键指标。

台账建议格式：

```text
date | RunID | script | seed | input_ckpt | scenario_set | output_path | key_metrics | conclusion | notes
```

## 串行与并行规则

可以并行：

- `Scenario-Steward` 和 `Reward-Metrics-Guardian`
- 多个只读审计任务
- 已完成 JSON/TB 结果的分析任务

必须串行：

- `Training-Runner`，除非你有多个 CARLA server/port 和明确资源隔离。
- 同一个 `--save-dir` 下的训练。
- 修改同一组文件的任务。

## 文件所有权

| 文件/目录 | 负责 agent |
| --- | --- |
| `data/training_scenarios/*.json` | `Scenario-Steward` |
| `data/test_scenarios/*.json` | `Scenario-Steward` |
| `training/adversarial_reward.py` | `Reward-Metrics-Guardian` |
| `utils/adversarial_metrics.py` | `Reward-Metrics-Guardian` |
| `logs/**` 中单次训练输出 | `Training-Runner` |
| `logs/eval/**` | `Evaluation-Repro-Analyst` |
| `docs/experiments/**` | `Evaluation-Repro-Analyst` |

一个文件同一时间只能有一个 owner。否则很容易出现两个 agent 同时改奖励、同时改场景、最后口径乱掉。

## 最小论文闭环建议

当前项目建议先走这个闭环：

```text
robust v2 合作基线
  -> held-out 泛化评估
  -> 2红1蓝对抗快验
  -> baseline vs attacked 正式评估
  -> 最小消融
```

暂时不要把 PSRO 放在主线第一阶段。PSRO 可以作为后续扩展，等下面三个条件满足后再上：

- robust v2 合作基线 3 seeds 稳定。
- held-out 场景已有正式评估。
- 2红1蓝对抗奖励不会通过 kamikaze、自碰撞或 threat-window farming 刷分。

## 硬性注意事项

- 只读 agent 不能启动 CARLA。
- 训练 agent 不能改代码。
- 不要复用旧的 `--save-dir` 跑新实验。
- 没有 JSON、日志、TensorBoard 导出或 checkpoint 支撑的数字不能写进结论。
- 不同 seed、不同场景集、不同 reward version 的结果不能直接比较。
- 不确定的信息写 `NEEDS_CONFIRMATION`，不要猜。

## 快速选择指南

如果你想问“下一轮实验怎么设计”，用 `Experiment-Planner`。

如果你已经有命令但怕跑错，用 `Command-Auditor`。

如果你要加测试场景或 held-out，用 `Scenario-Steward`。

如果你要改对抗奖励或解释对抗有效性，用 `Reward-Metrics-Guardian`。

如果你已经准备好跑一条训练命令，用 `Training-Runner`。

如果你有 checkpoint，想得到可写进论文的表格，用 `Evaluation-Repro-Analyst`。
