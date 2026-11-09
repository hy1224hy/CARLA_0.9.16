# R5：红车结果驱动的单车对抗奖励

状态：代码候选；未生成参考、未训练、未声称行为有效。工程尺度尚未通过真实轨迹校准。
用户授权本次实现R5；R4训练评估完成后总结并关闭CARLA；本次不自动启动R5。

## 研究目标与能力边界

奖励红车相对冻结基础参考的额外通行损失和驾驶负担。仅接近、低TTC或小间距不提供正向收益。
红车正确的避让减速代表交互负担，不自动代表驾驶策略错误或不安全。结果称为结果退化代理。
同case参考和路径门控无法证明单车因果贡献；替换单车策略的评估另外报告。

## 实际公式

每辆蓝车分别接收：

    r_i = .03 * mission_i + .3 * remaining_v6_soft_i
          + collision_i + offroad_i + road_rule_i -20*first_task_violation_i
          + outcome_i -2*I(TTC<=.5)*dt

碰撞首版仍为-50，未启用-25消融。任务项包括进度、到达、任务超时与自由流速度亏损；
其余基础软项沿用v6注册分量。Actor、critic、动作、观测及PPO保持原配置。

    phi(x;s,e)=clip(sign(x)*max(abs(x)-e,0)/s,-1,1)
    delay_j = phi(P_reference_j-P_actual_j; .10, .01+2*std(P_reference_j))
    burden_j = phi(B_actual_j-B_reference_j; .20, .02+2*std(B_reference_j))
    score_i = sum_j share_ij*(.7*delay_j+.3*burden_j)
    share_ij = interaction_gate_ij / number_of_gated_blues_for_red_j / 2
    outcome_i = 1.0*score_i*dt, after 1s warmup

每辆车正收益累计封顶8；负结果分值有界但不占用、也不返还正收益预算。
实际红车表现改善可产生负奖励，避免正差值截断把所有波动都变成收益。
所有诊断积分与持续时长不受预算和奖励warmup截断。
0.7/0.3、0.10/0.20等是显式工程初值，不是文献结论或已完成校准。

## 红车负担与参考

原始B=.5*clip(deceleration/3)+.25*clip(abs(longitudinal_jerk)/10)+.25*clip(abs(steer_rate)/1)。
真实轨迹和参考轨迹使用相同tau=.3s EMA，参考协议强制匹配滤波tau。
目前没有车道偏离或横向加速度奖励，不能声称完整测量驾驶失稳。
P来自环境冻结路线轴投影，适用于当前Town05直线路段；不直接声称适用于任意曲线/路口。

每case至少3次冻结基础蓝车参考，种子case.seed+rep*10000，保留均值、标准差、有效样本数。
新参考不复用R4响应文件。缺失参考或有效样本不足时不发结果奖励，覆盖率明确报告。
参考接触后的红车状态不外推；已到达红车的参考进度保持1且负担保持0到固定时域。
参考未完成且提前中断的尾段不补零，不伪造数据。
真实红车到达后停止奖励，防止把到达后的零速度解释为受阻。
训练种子与校准参考种子不完全相同，时间/状态错位仍是混杂，不能宣称同seed因果归因。

## 交互门控与分配

同向航向差<=30度，蓝车位于红车前方30m且有横向车身投影重叠，或纵向重叠且侧向净距<=2m。
门控保持1秒短滞后，以涵盖控制响应。它不要求TTC危险，不按接近程度加分。
仅为道路空间代理：前方门控可能误收旁观者，仍须用策略替换评估验证。
当前不覆盖非同向穿越冲突；报告时限制适用范围，不能用其他车的压力代领穿越攻击收益。
同一红车的分配份额之和<=1，再按两个固定红车槽位平均，避免两辆蓝车各领取完整团队收益。
自身安全失败停止该蓝车结果计分；红车接触/几何重叠后永久停止关联的结果计分。
TTC、间距保留R4真实几何诊断，R5的risk_reward和response_reward字段固定为0。

## Rubric（描述性，不能替代数值指标）

各角色分开报告效果和参考覆盖：
- no_measured_effect：未测得正向结果差异。
- transient_proxy_effect：测得短时正向差异。
- sustained_proxy_effect：正向结果差异连续至少1秒。
- reference unavailable/limited/adequate：覆盖率无参考/<90%/>=90%。
1秒与90%仅是工程标记阈值，应做敏感性检查，不用作自动选优门槛。
碰撞和越界单独报告，碰撞型结果不是最高等级。还应报告到达延迟、停车、路线完成率和行为回放。

## 配对反事实评估

同初态、同case seed、冻结其余车辆策略，顺序运行none/blue2/blue3/both四种基础策略替换模式。
其余车辆策略相同，但动作可响应新观测改变。这是仿真策略干预，不是COMA动作级基线。
逐项报告两红车平均进度亏损积分与负担积分，不用临时混合总分：

    marginal_2=L(none)-L(blue2)
    marginal_3=L(none)-L(blue3)
    joint=L(none)-L(both)
    interaction=L(none)-L(blue2)-L(blue3)+L(both)

接触或无效配对组不生成贡献估计，但原始失败记录保留；需报告排除率及场景组成偏差。
非接触未完成进度保持最后观测值补足固定max_steps时域；未来负担不虚构补齐。
这一删失口径也应随结果报告，不能把提前结束当较低通行损失。

## 验证与后续

已实现纯函数奖励测试、参考聚合、替换对照计算及离线完整PPO更新测试。
尚需真实CARLA参考生成、尺度校准、零影响/正影响回放审计和多seed实验。
R5公平对照从同base_u0初始化，而不是从R4 u66继续；同预算、同场景清单、同PPO设置。
消融顺序：R4 vs R5 -> 去除延迟/去除负担 -> 碰撞-50 vs -25 -> 未参与调参测试。
不在选择集结果不利时修改场景或阈值；A/B不是独立训练种子。

## 后续手动运行入口（本次没有执行）

从PythonAPI/examples/mappo_training目录，用仓库venv Python运行：

    python -m training.train_mut_mappo --config experiments/reward_iteration_20261107/R5_red_outcome/config.json --validate-only
    python -m utils.prepare_r5_reference --config experiments/reward_iteration_20261107/R5_red_outcome/config.json --repeats 3
    python -m training.train_mut_mappo --config experiments/reward_iteration_20261107/R5_red_outcome/config.json --run-dir experiments/reward_iteration_20261107/R5_red_outcome/run --allow-candidate
    python -m evaluation.evaluate_r5_counterfactual --config experiments/reward_iteration_20261107/R5_red_outcome/config.json --checkpoint <R5_checkpoint> --output-dir <fresh_output_dir> --team A

validate-only不证明参考齐全或真实行为正确；新参考目录不存在时，实际训练会拒绝启动。
反事实评估需对A、B各自运行，不能把A/B均值当独立训练种子。普通通用summary中的legacy事件字段仅保留诊断。
