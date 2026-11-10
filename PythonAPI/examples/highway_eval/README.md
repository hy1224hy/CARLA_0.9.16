# 独立 Highway 模型评估

该工具与训练、选模解耦，位于 `highwaytest` 包之外，不改变正在训练模型的核心源码摘要。它先冻结所有输入检查点，再使用固定、配对的场景逐个评估。开发集 seed 从 4,000,000 开始，最终 test 从 3,000,000 开始；均与 selection 分离。

从 `D:\study\yan2\CARLA_0.9.16\PythonAPI\examples` 运行：

```powershell
# 训练期间：独立开发评估，检查当前入选模型
& ..\..\venv\Scripts\python.exe -m highway_eval `
  --runs highwaytest/runs/reward_repair_20261110/r3/train `
  --checkpoint selected --split development --cases 40 `
  --output highwaytest/runs/reward_repair_20261110/r3/audit_dev_01

# 检查最新而非入选模型时，明确选择副本 A 或 B，并使用新的输出目录
& ..\..\venv\Scripts\python.exe -m highway_eval `
  --runs highwaytest/runs/reward_repair_20261110/r3/train `
  --checkpoint latest --team B --split development --cases 40 `
  --output highwaytest/runs/reward_repair_20261110/r3/audit_latest_B_01

# 方案与检查点冻结后，再使用最终 test
& ..\..\venv\Scripts\python.exe -m highway_eval `
  --runs highwaytest/runs/reward_repair_20261110/r3/train `
  --checkpoint selected --split test --cases 100 `
  --output highwaytest/runs/reward_repair_20261110/r3/audit_final_01
```

`--runs` 也可以直接指定单个运行目录或检查点文件。输出目录必须是新目录。

输出 `report.md`、`summary.json`、固定 `cases.json`；每个模型另有冻结 `checkpoint.pt`、`metrics.json`、`episodes.csv/jsonl`、`trajectories.jsonl` 及各场景首次成功/未成功的轨迹、速度、THW 图。事件计数与奖励分项为整回合累计，不是仅取最后一步。

成功率沿用环境中的等级事件定义，失败与无效均留在分母。额外报告完整换道、边界停留、多次切入、条件事件时间与行为标签。边界等诊断以策略频率采样，应与精确事件区分。L1/L2 碰撞是失败，L3 碰撞是任务目标，不能统一解释为风险改善。

模型与当前核心环境源码不一致时拒绝运行，避免对旧模型悄悄使用新奖励/成功判据。每个结果记录模型、核心环境、评估程序的摘要及运行版本。训练期间反复使用开发集；最终 test 不用于调奖励或选择模型。
