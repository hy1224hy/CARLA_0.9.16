# 本先导的模块命令（准备记录，不是启动授权）

所有路径属于新的 `experiments/adversarial_u1000_pilot_20261028_v1`。不运行旧示例，不连接或恢复基础 run。每段开始先明确工作目录与绝对Python路径。当前只执行过离线段；下方真实段须等用户明确启动相应阶段。

## 已可运行的离线检查

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\Activate.ps1'
$py = 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe'
& $py -m experiments.adversarial_u1000_pilot_20261028_v1.offline_audit
& $py -m training.train_mut_mappo --config experiments/adversarial_u1000_pilot_20261028_v1/configs/B0_s42.json --validate-only
& $py -m experiments.adversarial_u1000_pilot_20261028_v1.screen_train
```

重跑原有回归时，在同一工作目录执行：

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m experiments.adversarial_u1000_pilot_20261028_v1.offline_audit --tests
```

预期 `OFFLINE_COMPATIBLE_NOT_LIVE_VALIDATED`；81项CPU/合成环境/本地OpenDRIVE检查，非真实CARLA训练。审查脚本禁止创建carla.Client。Windows沙箱临时目录可能阻止原子rename；本轮获准在沙箱外重跑后81/81通过，不能把此临时权限错误解释成算法失败。

`--prepare`仅用于第一次生成配置，已有配置时拒绝覆盖；不要再次添加。配置生成复用现有`build_cases`、`experiment_config`及`validate_config`，只登记B0。

## 获准启动 L1 后：80例 C0/C1，串行

前提：用户启动CARLA，核验0.9.16、Town05、全机所有者、无其他tick/reset控制器。入口会执行端口锁和idle world检查；失败就停，不kill别的客户端，不自动切世界/重启服务器。

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
$py = 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe'
& $py -m experiments.adversarial_u1000_pilot_20261028_v1.screen_train --execute
if ($LASTEXITCODE -ne 0) { throw 'train-state C0/C1 failed; inspect evidence before continuing' }
& $py -m utils.run_first_point_experiments red-eval --experiment experiments/adversarial_u1000_pilot_20261028_v1 --groups B0 --seeds 42
```

先48例训练初态，再32例gate初态。若要重跑，保留失败目录和证据并登记新attempt；不能覆盖历史或只补表现差的一侧。完成后先解释C0自然失败、C1普通交通影响及蓝方资格，不自动进入学习。

## L2准备：仅在弱红诊断范围明确获准后

**当前不执行以下阶段。** u1000基础资格未通过；单独train入口不会替执行者重检这个条件。生成配置和通过离线检查都不授权绕过它。先按审查计划补齐reference成本callback和真实sensor/初态证据。

普通参考的现有严格入口如下。它按模板检查红到达和蓝事件资格，不能代替全局基础资格；失败时保留结果并停。只有用户明确选择弱红诊断，才可以显式讨论加入`--allow-weak-red`，并保留整个实验的弱红标签。本文件没有把该参数加入任何可执行命令。

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m evaluation.mut_mappo_reference --config experiments/adversarial_u1000_pilot_20261028_v1/base_config.json --splits train gate selection --repeats 2
```

本计划使用两次参考。不要使用wrapper的`references`/`run`/`evaluate`一键链：它们内部默认repeats=1，会与已登记的2次参考冲突；也不能用一键run反复重跑已知失败的基础资格。

## L2训练候选命令：首轮退出再恢复，总预算仍24k

以下只在前述授权与参考准备检查完成后使用。每轮A4+B4回合；首轮和后续恢复共同消耗同一24k实际训练tick预算。

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m utils.mut_mappo_session launch --config experiments/adversarial_u1000_pilot_20261028_v1/configs/B0_s42.json --run-dir experiments/adversarial_u1000_pilot_20261028_v1/runs/B0_s42 --rounds 1
```

launch返回只表示派发，必须等待对应supervisor的exit，验证完整边界manifest/SHA、A/B更新数、有限值、红冻结、尾项和退出状态，才执行下一段。长段启动后才建立本run专属20分钟纯文件监控。supervisor记录的是Popen.pid，在Windows上可能是venv launcher；必须另核验实际训练worker命令行/父子树并保存关联证据，不能仅凭派发PID或CLI返回宣称实际worker匹配退出。

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m utils.mut_mappo_session launch --config experiments/adversarial_u1000_pilot_20261028_v1/configs/B0_s42.json --run-dir experiments/adversarial_u1000_pilot_20261028_v1/runs/B0_s42 --resume
```

本段是预算内恢复，不能改config或提高train_tick_budget。异常停在已提交A+B边界，由用户/本对话检查后处理，不自动重启。session supervisor现有`launch`会在候选配置上显式传`--allow-candidate`；这个开发标签仍不能替代弱红授权。

## 最终冻结评估候选命令

只评估预登记的最终完整边界，selection用A/B各16例C1/C2。先从manifest定位权威checkpoint并核对SHA，不以方便副本latest.pt代替权威身份。

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
$py = 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe'
$experimentPath = 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training\experiments\adversarial_u1000_pilot_20261028_v1'
$runPath = Join-Path $experimentPath 'runs/B0_s42'
$boundary = Get-Content -LiteralPath (Join-Path $runPath 'checkpoint_manifest.json') -Raw | ConvertFrom-Json
$checkpointPath = Join-Path $runPath $boundary.checkpoint
if ($boundary.completed_round -lt 1) { throw 'No learned complete boundary' }
if ((Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $boundary.sha256) { throw 'Checkpoint SHA mismatch' }
& $py -m evaluation.evaluate_mut_mappo --config experiments/adversarial_u1000_pilot_20261028_v1/configs/B0_s42.json --checkpoint $checkpointPath --team both --split selection --conditions C1 C2 --select-best --output-dir experiments/adversarial_u1000_pilot_20261028_v1/evaluations/B0_final_selection
```

若NONE_QUALIFIED/INCOMPLETE_EVIDENCE就停止，不生成假合格模型。仅在已冻结selected export存在后，生成test参考并跑48例C0/C1/C2：

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
$py = 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath 'experiments/adversarial_u1000_pilot_20261028_v1/evaluations/B0_final_selection/frozen_blue_team.pt')) { throw 'No frozen selected blue team' }
& $py -m evaluation.mut_mappo_reference --config experiments/adversarial_u1000_pilot_20261028_v1/base_config.json --splits test --repeats 2
if ($LASTEXITCODE -ne 0) { throw 'Test reference failed; preserve evidence' }
& $py -m evaluation.evaluate_mut_mappo --config experiments/adversarial_u1000_pilot_20261028_v1/configs/B0_s42.json --checkpoint experiments/adversarial_u1000_pilot_20261028_v1/evaluations/B0_final_selection/frozen_blue_team.pt --team both --split test --conditions C0 C1 C2 --output-dir experiments/adversarial_u1000_pilot_20261028_v1/evaluations/B0_frozen_test
```

这些显式评估阶段输出各自summary/selection，未通过wrapper写`results/`索引，不能把wrapper空comparison表当本先导结果。结束后据完整summary和audit生成一个outer seed的一行诊断报告，无组间std、不把A/B算两次训练重复；仍保留红方NOT_QUALIFIED标签。

可选四组合不在首轮必做预算内；只有另行决定执行且补齐其成本callback后，使用现成模块（相同工作目录、配置和已冻结export），不重新选模。监控按计划在该长段完成后删除。
