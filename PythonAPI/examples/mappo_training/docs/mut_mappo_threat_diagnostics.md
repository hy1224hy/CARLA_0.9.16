# 通用蓝车威胁配对诊断

本文保留服务器分支的使用说明。本地命令见 [本地同步说明](local_adversarial_metrics_sync_20261029.md)。末尾服务器示例中的 `--allow-weak-red` 不是本地启动授权；当前仍应先完成审查和准备，不能据此绕过训练资格。

本入口只读取已完成的独立评估摘要，不连接 CARLA，不加载模型，不改变筛选或训练。
训练轮次、组名、模型路径均不写死。支持任意组名、检查点和 selection/test 报告。

从 mappo_training 目录运行：

```bash
python -m utils.analyze_mut_mappo_threat \
  --experiment "$EXP" \
  --output-dir "$EXP/analysis/threat_$(date +%Y%m%d_%H%M%S)"
```

自动发现 results 中每个 run 最新评估边界，读取记录指向的 selection/test 摘要。
未入选的 A/B 候选也保留。历史检查点比较可以通过 --reports 显式传入各检查点摘要。
实验整体下载或迁移后，会尝试按 evaluations/<结果文件名>/<阶段目录> 重定位。
其他目录结构请显式传入报告：

```bash
python -m utils.analyze_mut_mappo_threat \
  --reports /path/to/first_summary.json /path/to/second_summary.json \
  --labels baseline candidate \
  --output-dir /path/to/new_analysis
```

--labels 可省略，默认从元数据 config_path 文件名获取标记；不会访问该配置路径。
输入须包含 C1/C2 的逐案例配对数据；C0 缺失时，相关指标为 null。
输出目录必须是新目录，原始证据不会覆盖。

输出：

- index.html：可直接用浏览器打开的表格和正向/反向贡献计数条形图，无外部依赖。
- teams.csv：每份报告、每个团队一行；Excel 可打开。
- cases.csv：每个共同有效案例一行，可按场景、违规资格、结果变化筛选。
- threat_report.json：完整结果、输入 SHA256 和原始元数据，便于追溯。

指标：

- safe_new_failures：C1 成功、C2 失败，且 C1/C2 蓝方都合格的案例数。
- safe_recoveries：C1 失败、C2 成功，且双方都合格的案例数。
- safe_net_gain：(上述新增失败数 - 恢复成功数) / 全部共同有效案例数。
- unsafe_new_failures / unsafe_recoveries：至少一侧蓝方不合格的对应变化；不赋予安全攻击贡献。
- c0_failure_cases：C0 已失败的案例数。
- c1_incremental_failure_cases：C0 成功、C1 失败的案例数；不是碰撞责任归因。
- blue_bad_rate：C2 蓝方不合格率。与 blue_eligible_pair 的双侧资格不同。
- safe_arrival_delay_s：双方红车都成功、双方蓝车都合格的案例平均 C2-C1 到达延迟。
- safe_completion_loss：双方蓝车合格案例中，平均 C1-C2 红车完成进度差。
- safe_efficiency_utility：保留原报告指标；不合格归零规则不变，不是仅合格回合均值。

正的延迟/进度损失表示红车受阻。零净增益可能来自无变化，也可能来自正负抵消。
C0/C1 的一次成功结果不是预先注册的可靠参考子集，本工具不据此声明参考稳定。
不能从摘要恢复真实制动、最近距离、仅合格回合效用等缺失数据；需要另读原始轨迹。
不同阶段、红方、场景清单、预算不可直接合并；工具不聚合独立样本或给显著性结论。
A/B 是同次训练的相关候选，不能充当两个训练 seed。

## 是否需要重训

有现成 summary.json 时只需运行离线分析。现有训练检查点可以直接补跑评估。
本次新增独立模块不改变已有模型、训练身份或评估语义，不需要重训。
修改奖励、场景、参考控制或训练协议时，才需要另建实验并重新训练。

若实验还没有评估摘要，可先运行现有入口（需要独占运行中的 CARLA）：

```bash
python -u -m utils.run_first_point_experiments evaluate \
  --experiment "$EXP" --allow-weak-red \
  --monitor-mode web --monitor-port 8080
```

--allow-weak-red 仅用于明确选择弱参考诊断的实验。若已有摘要，优先离线分析。
现有 evaluate 会复用已完成结果；新增制动/距离采集或重复评估时，必须使用新的输出目录。
不要通过反复查看 test 调整训练配置。
