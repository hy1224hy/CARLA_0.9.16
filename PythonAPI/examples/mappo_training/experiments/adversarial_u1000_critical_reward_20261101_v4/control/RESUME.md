# 最新状态：2026-11-03 B2门控审查完成

R2 retry1的144回合已完成并正常退出，完整报告SHA已核验；没有继续运行旧评估。R2目标仍未达到，不能宣称正常驾驶危险行为训练成功。最新审查见 docs/B2门控实现审查_20261103.md 和 diagnostics/b2_gate_review_20261103_v1/audit.json。

本轮只做离线审查，未连接CARLA或启动新训练/评估，world ticks=0。41项相关检查通过。已修复case_keys缺失/重复/长度不符的校验漏洞与efficiency gap非法输入校验；原门槛、奖励和历史训练证据不变。当前源码与历史R2源码身份已不同，后续新实验需版本化登记，不能悄悄恢复旧run。

R2实际第19轮、10281 train ticks刷新一次，A/B全部family门关闭；最终B0模型离线模拟刷新仍全关。这不是B2真实学习结果。既有prepare入口会重置旧奖励/案例/quality_metric，不能直接启动R2三组。后续先准备新配置和预算，再按单控制器所有权核验串行执行。A/B不是独立seed，不放宽正式资格。

下面保留历史接续记录；与本段冲突的运行状态以本段及签名完成标记为准。
# R2奖励调整接续

## 2026-11-03接续：固定检查点重新评估已启动

用户已重新启动CARLA并明确要求继续。102项冻结源码SHA与红方/蓝方checkpoint SHA再次核验一致，R2无须重训。空闲只读检查通过：客户端/服务0.9.16，0车辆、0传感器，异步世界。

启动时整卡7431MiB，仅527MiB可用；用户选择保留当前服务器直接切Town05，未执行Low画质重启。实际评估首次载入Town05后复用地图，初期整卡显存约3034MiB、可用约4924MiB；不能由初期数据断言整个评估已稳定。

当前命令为 `-m utils.recover_critical_reward_evaluation --root experiments/adversarial_u1000_critical_reward_20261101_v4 --group R2 --attempt 1 --reuse-idle-world`，工具会串行完成新目录evaluations/R2_gate_retry1的144回合C0/C1/C2×A/B。UUID为ed6a128dadb040a29da66a2a9c29c8cd，实际PID以pipeline_status.json为准。registration保存源代码/模型SHA及空闲证据；operation.json记录地图复用，gpu_memory.jsonl每5秒记录整卡显存。旧121回合失败目录保留，不拼接效果。

20分钟纯文件监控carla已恢复ACTIVE。主代理在分析本轮，不得重复执行。先完成这144回合、核对正常退出/配对覆盖及显存走势，再与R0/R1做全指标比较；不得中途改奖励或改设备。R2仍为CPU训练的已完成模型，后续CUDA训练若实施需新配置登记。

## 最新覆盖：R2训练完成，评估渲染异常后待重启

训练正常完成43轮，训练边界24343 ticks，checkpoint SHA为4fcc1a7bce140871f14ce709fcbc4edd27302dafcc64562d2775e57a66c1108a。独立gate于121/144中断，退出0xC0000409，用户截图显示显存分配失败。实际CARLA与评估进程已退出，不重复启动训练。诊断、修复与恢复命令见control/render_failure_review/review.md。

实际危险评估入口已默认复用同名空闲地图，避免每回合重载；不改训练/奖励源码或已完成配置。新增显存采样及operational源码存证。43项相关检查与固定R2 checkpoint离线预检通过，尚未做修复后的真实评估。用户尚未在本次显存退出后明确说服务已重启；不要盲目重复连接或重试。

最新实际进度：8回合真实探针已完成，C0为4/4、C1红方0/4、C1蓝方事件资格2/4。所有回合有效、奖励残差0，持续事件奖金触发，违规回合50分成本保留；这些是普通蓝车轨迹，不能称为学习效果。探针耗608实际controller ticks（496 decision）。真实R2流水线已经启动，worker PID25208、UUID082679f10b464d59aa39c678db316ec8，最初stage为R2_references。后续以pipeline_status.json及UUID/实际子进程为准；已有worker时严禁重复启动。

用户授权固定弱红u1000，训练蓝2/3正常驾驶同时产生危险交互；失败C1允许工程先导，正式阈值不能降低。红0/1冻结，不恢复基础训练。当前已恢复主任务；已删除一次五小时续做carla-2，20分钟纯文件监控carla仍保留。若之后再次接近额度上限，再保存状态并按用户要求设置五小时续做。

前一实验v3全部完成并核验报告SHA。完整对照：v3/analysis/reward_comparison_complete_v1/comparison.json与review.md。R1 A/B事件资格危险6/3，R0为3/8，各24例；蓝方事件资格R1为12/15，R0为14/17。没有一致提升，速度/车道/转向仍有问题。A/B不是独立seed；C1已有23/24原始危险，不把普通交通或自然失败算成新增攻击。

本实验R2为normal_driving_critical_team_v2。保持同红方、场景SHA、seed42、B0、初始化与24000实际train ticks预算；新参考144回合，冻结gate144回合。仅修改奖励及显式登记的危险信用保护：曝光正预算3，一次持续危险事件5，既有超速总倍率10、操纵变化总倍率5、合法豁免后车道成本倍率10、pace系数.1；危险信用保护1秒，与原3秒stall/slowdown保护分开。测量事件协议、正式阈值和PPO参数均不改。

76项离线检查通过，含两profile的真实PPO更新接口测试，但这些使用明确合成动态，不能视为驾驶证据。先在screen_2做8回合真实C0/C1接口探针；它可能仍因弱红能力而失败，不作为再度阻止已授权工程训练的硬门槛。只要接口、几何、奖励分项和安全成本正确，可继续登记的短预算。

真实流水线命令：在mappo_training目录以项目venv运行 `-m utils.critical_reward_pipeline --root experiments/adversarial_u1000_critical_reward_20261101_v4 --groups R2`。首次启动前核对当前探针及机器实际进程，不能与探针并行。流水线创建pipeline_status.json、plan和完整source_snapshot，串行参考→蓝车训练→冻结评估；源代码从启动起冻结，不运行默认R1/R0组。

主监控入口仍为v1/control/status.json，active_run应指向当前探针或本实验pipeline_status。进程与UUID/正常退出要对应，所有旧失败和成本保留。出现原生退出时先读证据，不自动重启用户CARLA或无限循环评估。全程俯视。

完成后比较R2与同预算R0/R1；同时报告原始/事件资格危险、蓝方资格、资格单独改善、TET/TIT、红方任务与效率、蓝方速度/车道/操纵。不因奖励上升宣布成功，不把事件资格当完整正常驾驶。若不达标，保留结果后再小步调整，不能偷加预算或看test调参。

