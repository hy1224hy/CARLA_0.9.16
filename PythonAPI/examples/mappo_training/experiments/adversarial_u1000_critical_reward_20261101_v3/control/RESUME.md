# 接续说明

## 最新覆盖：已恢复并进入下一候选

R0重评已在2026-11-01 14:25（Asia/Shanghai）正常完成144/144，完整分析见analysis/reward_comparison_complete_v1/comparison.json与review.md。A/B事件资格危险R1为6/3，R0为3/8，各24例；蓝方事件资格R1为12/15，R0为14/17，没有一致优势。

下一候选已接入生产并通过76项离线检查，进入新实验experiments/adversarial_u1000_critical_reward_20261101_v4。后续请优先读取该目录control/RESUME.md及v1/control/status.json最新指针。旧v3生产源码已完整快照，历史报告/模型不改写；不要以新源码恢复或重跑旧流水线。五小时续做carla-2已在成功恢复并保存进度后删除，20分钟监控carla继续。

## 最新覆盖：R1结果及R0服务器重启后评估

R1已完成42轮真实蓝车训练和144回合固定末点gate。A/B各24例，满足事件资格的危险分别6/3，C1为6/24；蓝方事件资格分别12/15，仍存在速度、车道和实际转向缺口。第一候选尚未达到用户联合目标，不能称为奖励有效。详见R1_frozen_result_review.md。

R0已完成41轮真实训练。原gate在29/144后客户端原生异常0xC0000409；retry1中CARLA RPC持续超时，已核验实际worker11124并终止。两份未完成证据保留，不拼接、不计作学习失败。用户随后明确说“我重启了Carla”，只读preflight确认客户端/服务0.9.16、0车辆/0传感器、异步空闲。

现在使用 `utils.recover_critical_reward_evaluation --root experiments/adversarial_u1000_critical_reward_20261101_v3 --group R0 --attempt 2` 串行继续。实际当前PID和进度以pipeline_status.json为准；新输出为evaluations/R0_gate_retry2，不得重复启动。原pipeline工具不要重跑，它会拒绝部分旧评估目录。重评成功会写R0_gate_completed.json，analyze/inspect工具已通过该完成标记读取正确报告，并将前两次失败attempt成本下界单列。完整source冻结继续生效，R0重评结束前不得修改environment/training/evaluation/models。

下一候选已隔离于control/reward_v2_candidate.py，4项离线激励检查通过；尚未接入生产或真实训练。现有71项回归属于第一候选。接入步骤见reward_v2_design.md；应新建v4实验及R2组，固定同一红方/场景/seed/总训练预算、重新生成profile绑定参考。不要把下一候选离线检查当成真实学习证据。

20分钟纯文件监控carla和五小时继续任务carla-2均已登记。额度接近上限时已有恢复任务，不需要重复新建。五小时任务唤醒后先检查存活进程与最新指针；若评估仍在进行，只跟进；若完成，则做R1/R0比较并继续版本化奖励调整。

用户目标：训练蓝2/3，在自己正常驾驶的同时使冻结弱红0/1出现更多危险交互。红方固定u1000，绝不恢复基础训练。用户已明确澄清允许C1检查失败条件下的短程工程训练，随后再次要求继续。不得再把C1失败当成本轮工程训练硬停止条件；失败标记和原资格阈值保留。授权见同目录training_authorization.json。

当前真实执行：`utils.critical_reward_pipeline`，实际worker PID25240；主目录为本目录上级。它会串行完成R1参考、R1蓝车训练、R1冻结gate、R0参考、R0蓝车训练、R0冻结gate。每组外层seed42、B0、A+B合计24000实际train world ticks，按完整边界可能超量。评估和废弃采样另记。全程俯视。

先用项目venv在mappo_training工作目录运行 `-m utils.inspect_critical_reward_pilot`，这是纯文件检查，不连接CARLA。核验pipeline_status.json、stage_pid、实际子worker、训练manifest与exit文件；已有存活worker时绝不能重复启动控制器。源代码已冻结，pipeline_sources.json与source_snapshot保存精确内容。不要在流水线未完成时修改environment/training/evaluation/models源码，或两个pipeline工具源码。

流水线异常时先看对应stage_log和*_exit.json。训练正常退出必须匹配launch_id、supervisor/trainer启动器PID及实际worker证据。已有checkpoint时可在源码/配置身份未变且实际进程已退出后使用原流水线入口恢复；partial评估目录不能覆盖，须先审查后登记新attempt。

流水线完成后分析evaluations/R1_gate、R0_gate下analysis/critical_report.json与原始cases/traces。分别报告A/B，不能当成两个外层seed；C1/C2必须同案例同初态同seed。原始危险率、G资格率、合格危险率、仅资格改善、TET/TIT、红方任务与效率、蓝方完整驾驶质量同时检查。不能把C1自然失败或普通蓝车已有威胁计为新增攻击。C1参考71/72例不合格，需明确地板/天花板效应；完整原始案例保留。

已完成104回合真实基线/探针；71项离线检查通过（包含合成测试动态的完整PPO接口测试，不能当真实驾驶或学习效果证据）。新奖励已在真实C1轨迹触发正项，尚不能由此宣称学习有效。R1正在实际PPO更新；actor2/3梯度非零，早期奖励残差和preupdate ratio误差为0。

若R1不达目标，依据冻结评估与奖励分项提出并执行小预算、版本化的下一轮修改；保留所有失败和成本，不调低驾驶资格阈值，不引入通信/PSRO/新信用网络。正式B0/B1/B2结论仍需合格红方、多外层seed等证据；当前只是弱红工程先导。

20分钟纯文件监控automation id为carla，绑定本对话，主要状态入口仍为v1/control/status.json指向本流水线。整个调整目标完成后删除对应监控。额度接近上限时保存状态并创建5小时后在本对话续做的任务；若届时worker仍活着，仅跟进它，禁止重复tick/reset。不要消耗额度重置券。
