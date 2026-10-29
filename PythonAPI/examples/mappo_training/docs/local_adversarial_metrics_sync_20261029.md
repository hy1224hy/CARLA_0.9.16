# 本地对抗评测指标同步（2026-10-29）

本地分支：`codex/mappo-experiment-baseline`。同步前快照：`ab4060f96ba20e398e1ece27036eda3da461f442`。
来源：`origin/codex/server-remote-monitor` 的以下两次提交：

- `38f034a`：已完成独立评估摘要的配对威胁诊断。
- `813a076`：真实车辆包围盒采集、危险交互指标和配对报表。

源码、测试、协议和说明已移植到本地训练分支。服务器监控 UI 不在本次同步范围内；本地入口默认关闭 UI，显式要求不可用的 web UI 会在评测前报错。离线报表不依赖 CARLA、PyTorch 或服务器监控模块。原服务器分支安装了 UI 且评测 API 支持时，入口仍可使用它。

## 指标与入口

| 入口 | 输入 | 新增统计 |
| --- | --- | --- |
| `utils.analyze_mut_mappo_threat` | 已完成独立评估的 `summary.json`，或实验目录 | 安全新增失败、恢复成功、净增益、不合格蓝方贡献、C0 自然失败、C1 增量失败、配对延迟与进度损失、蓝方违规和资格 |
| `utils.evaluate_mut_mappo_critical offline` | 包含 manifest、resolved_config、cases 和 traces 的评估目录 | OBB-TTC、TET/TIT、危险事件、红车减速/制动、蓝车驾驶质量、原始/合格危险率及 C2−C1 增量 |
| `utils.evaluate_mut_mappo_critical live` | 已有 MutMAPPO 蓝方检查点及原配置，或实验训练边界 manifest | 串行执行 C0/C1/C2，读取实际 actor 包围盒，保存新轨迹，再生成上述危险交互报表 |

两个分析工具均输出 HTML、CSV 和带来源标识的 JSON。危险交互工具额外输出逐回合、逐事件、逐配对结果和测量覆盖率。两类报表互补：红车最终成功仍可能经历危险交互；红车失败也可能来自自然失败或不合格蓝方。

本次只新增评测工具，不把独立评测指标混入训练窗口统计。现有训练、奖励、场景、资格阈值、检查点加载器、筛选规则以及历史证据保持原样。普通旧评测入口没有自动补采包围盒；需要完整几何指标时，应使用新的 critical live 入口。

## 本地兼容修正

1. 本地没有 `visualization.mut_mappo_monitor`，且本地 `evaluate()` 不接受 `monitor_options`。新入口仅在支持时传入该参数，离线模式不导入监控模块。
2. 配对统计先按 case/team/seed 选择**最新 attempt**，再检查条件完整性、有效性和几何测量。最新重试无效或不完整时排除整组，不回退使用旧尝试；这与本地既有评测合同一致。原始尝试仍保留在逐回合报表及 JSON 中。
3. 缺少真实 bbox、缺帧、身份不符或生命周期不完整时，指标为未测量，不填零。不假设通用车长车宽。
4. 初始帧与首个决策帧也必须保持生命周期连续。初始存在的蓝车不能从后续 active_mask 中静默消失或提前变为 inactive；此类缺失轨迹排除，不算“无危险”。

## 本地命令

以下命令中的“替换为…”路径均需替换为实际文件。每次使用新的输出目录；不会覆盖已有结果。

```powershell
Set-Location -LiteralPath 'D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training'
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/Activate.ps1'

# 已有 summary 时可立即离线分析，无需启动 CARLA。
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m utils.analyze_mut_mappo_threat `
  --reports 'evaluations/替换为已完成评估/summary.json' `
  --output-dir 'evaluations/替换为新的威胁分析目录'

# 已有包含实际 bbox 的轨迹时可离线计算几何指标。
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m utils.evaluate_mut_mappo_critical offline `
  --evaluations 'evaluations/替换为已采集评估目录' `
  --protocol 'configs/evaluation/mut_mappo_critical_v1.json' `
  --output-dir 'evaluations/替换为新的几何分析目录'

# 准备阶段仅做离线校验：不创建输出，不连接 CARLA。
# 此处 checkpoint 必须是已有 MutMAPPO 蓝方训练边界/冻结导出，不能传基础红方 u1000。
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m utils.evaluate_mut_mappo_critical live `
  --config '替换为该蓝方模型原始配置.json' `
  --checkpoint '替换为该蓝方模型检查点.pt' `
  --split gate --monitor-mode off `
  --protocol 'configs/evaluation/mut_mappo_critical_v1.json' `
  --output-dir 'evaluations/替换为新的诊断评估目录' --validate-only
```

实际采集需在用户明确启动后，独占 CARLA，按已确定预算串行运行；届时去掉 `--validate-only`。实验模式可以使用 `--experiment`、`--groups B0`、`--seeds 42`，按各 run 的 checkpoint manifest 核验路径及 SHA。没有完成的蓝方检查点、参考文件或配套原配置时，不应尝试 C2。

旧摘要能立即计算威胁配对指标；缺 bbox 的旧轨迹不能恢复精确几何指标，需要补评估采集，无需因此重训。零可测配对时仍输出缺失原因，critical 命令退出码为 2。

## 解释边界与下一步

- `safe_net_gain` 的分母为全部共同有效配对；安全贡献要求 C1/C2 双侧蓝方合格。不合格带来的失败、延迟单列。
- 危险率同时比较 raw H 与 qualified G×H。仅蓝方资格变好而 H 不变，不能解释成危险生成增强。
- 现有 TTC 阈值等参数是版本化诊断协议，不是驾驶资格放行阈值；本次未调参。
- TTC 采用二维、固定朝向、匀速包围盒预测。红车响应仅表示时序关联；不推断事故概率、责任或因果攻击成功。
- A/B 分开报告，不能当两个独立训练 seed。不同红方、场景、预算、selection/test 不合并推断。
- 当前 u1000 红方仍是 `NOT_QUALIFIED`。这些工具不改变训练资格，不添加弱红方授权；后续弱红诊断与正式算法对比仍须区分。

本次验证使用合成轨迹、模拟运行器和本地地图，不连接 CARLA。真实四车采集、有效攻击信号和奖励适用性仍待用户明确启动后验证，不能把离线测试当作攻击实验结果。

## 离线验证记录

新增指标及本地兼容测试 44 项通过；连同原有配对评测、筛选导出、几何证据、第一研究点流程和新场景回归，共 **75 项通过**。验证时显式拦截 `carla.Client`，创建尝试为 **0**。另用阻断运行时模块导入的离线 CLI 检查，确认缺失轨迹输出未知值及退出码 2，且摘要的安全/不安全贡献分开统计。

原有原子保存测试在 Windows 沙箱临时目录遇到 `WinError 5`，在批准的沙箱外环境运行最终全部 75 项后通过，没有为此修改保存逻辑。代码审查发现的 reset 到首帧生命周期缺口已修复，并经独立复核。与同步前快照相比，训练、环境、奖励、场景、实验配置、历史诊断以及原有评测/筛选源码没有变更。

可从上述工作目录复跑：

```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -m unittest `
  tests.test_critical_interaction_metrics tests.test_critical_interaction_report `
  tests.test_analyze_mut_mappo_threat tests.test_mut_mappo_metrics `
  tests.test_mut_mappo_selection tests.test_mut_mappo_geometry_evidence `
  tests.test_first_point_experiment tests.test_mut_mappo_interaction_cases -v
```
