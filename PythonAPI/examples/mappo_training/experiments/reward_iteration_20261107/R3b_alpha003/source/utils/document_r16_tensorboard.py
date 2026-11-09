"""Enumerate every actual scalar tag and document its source semantics."""
import hashlib
import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/adversarial_u1000_R16_long_20261105_v1/analysis/tensorboard_replay_20261105_v1'

SUMMARY={
 'return/team_discounted_mean':('蓝方团队折扣回报：每回合sum(gamma^t × team_reward_t)，再取案例均值；gamma=.999。不是红蓝奖励之和。','奖励单位','training每点16回合；frozen每点24回合。越高表示当前奖励改善，不保证对抗目标达成。'),
 'safety/blue_bad_rate':('至少一辆蓝车发生碰撞、越界、道路违规或任务违规的回合比例。','比例0~1','越低越好；训练批次16回合。'),
 'safety/blue_event_eligible_rate':('两辆蓝车均没有已登记安全/任务违规事件的回合比例G。','比例0~1','不是完整速度、车道、转向驾驶质量。'),
 'quality/full_driving_rate':('事件资格G与蓝2/3所有已注册驾驶检查同时通过的案例比例D。','比例0~1','不额外要求到达；任一检查明确失败则该例失败，信息不足且无明确失败则未知。'),
 'quality/full_driving_and_resolved':('同一案例中D通过且至少一段红蓝险情被观测到局部无接触解除的联合比例。','比例0~1','逐案例求交集后计数，不是两个总体比例相乘，也不是以有险情案例为分母。'),
 'red/both_reached_rate':('红0和红1都到达目标的案例比例。','比例0~1','降低不必然是蓝方有效攻击；需结合C1、蓝方质量和红方自然失败。'),
 'critical/legacy_event_qualified_rate':('旧口径危险事件H与蓝方事件资格G同时成立的案例比例。','比例0~1','旧H含几何重叠/TTC=0或持续低TTC；不含完整驾驶质量要求，不等于新联合率。'),
 'critical/legacy_raw_event_rate':('旧口径危险事件H出现比例；H含任意红蓝几何重叠或TTC<=1.5s持续>=.3s。','比例0~1','不筛选蓝方资格，碰撞也可能计入，不能单独当安全攻击成绩。'),
 'value/explained_variance':('更新后Critic在本批GAE目标上的EV=1-Var(target-prediction)/(Var(target)+1e-8)。','无量纲','1接近完美，0不优于常数方差解释，负数更差；不是策略成功率。'),
 'value/critic_loss_logged_mean':('ValueNorm归一化空间的Critic Huber回归损失，取日志中epoch/小批次均值。','归一化损失','降低只支持价值拟合改善；不是原始回报MSE。'),
 'value/return_target_std':('本批两蓝角色GAE回归目标在原奖励尺度上的总体标准差。','奖励单位','目标=旧value+raw GAE；不是整回合return的标准差。结合MAE/EV看。'),
 'value/absolute_error_logged_mean':('更新后Critic预测与本批GAE目标的平均绝对误差，原奖励尺度。','奖励单位','越低通常拟合越好；目标分布改变会影响其可比性。'),
}
CRITICAL={
 'precontact_predicted_conflict_rate':('同一红蓝配对在首次物理/几何接触前，0<TTC<=1.5s连续>=.3s的事件出现比例。','比例0~1','只描述危险出现，不保证蓝方合格；不是碰撞率。'),
 'observed_resolved_event_rate':('至少一个上述危险事件随后被观测到无接触解除的案例比例：该配对TTC>1.5s或不预测碰撞连续>=.3s。','比例0~1','局部解除不代表整个回合没有其他事故；事件后失去观察不算解除。'),
 'censored_event_rate':('至少一个危险事件因回合结束、退场或缺少后续样本，无法判定接触/解除的案例比例。','比例0~1','观察截断不是安全解除，且可与同一案例其他已解除事件并存。'),
 'precontact_tet_s':('各案例接触前0<TTC<=1.5s的时间并集长度，随后取案例均值；同一步多配对不重复累计。','秒','包含未达到持续.3s事件门槛的短采样；越大表示暴露更久，不自动更好。'),
 'precontact_ttc_deficit_s2':('接触前低TTC缺口积分：sum_t(1.5-min_positive_TTC_t)*.1，再取案例均值。','秒²','是线性TTC缺口积分，不是1/TTC积分；同一步使用最危险有效配对。'),
}
DRIVING={
 'mean_speed_mps':('先计算各回合该蓝车活跃时实际平均速度，再取回合均值。','米/秒','不是越大越好，偏好速度2~4.1667m/s。'),
 'above_preferred_speed_fraction':('各回合活跃实测速度>4.1667m/s的比例，再取回合均值。','比例0~1','每例门槛<=.05；前方阻挡不豁免此安全项。'),
 'preferred_speed_fraction':('各回合有效巡航样本中2~4.1667m/s占比，再取有效回合均值。','比例0~1','巡航排除前2s、距目标<=5m和前方阻挡；每例>=.8且样本>=5。'),
 'reference_lateral_p95_m':('每回合相对道路参考车道横向距离绝对值P95，再取回合均值。','米','不是所有帧合并P95；此描述测量未重新排除合法变道豁免，资格看lane_p95失败率。'),
 'actual_steer_rate_p95_per_second':('每回合实际VehicleControl的abs(steer_t-steer_prev)/.1的P95，再取回合均值。','归一化steer/秒','不是方向盘角速度。每例门槛<=1.5；随机训练和确定性评估口径不同。'),
 'unblocked_low_speed_fraction':('各回合有效无阻挡巡航样本中速度<.2m/s的占比，再取有效回合均值。','比例0~1','每例<=.05；无有效测量不补0。'),
}
CHECKS={
 'peak_speed':('峰值速度<=8.3333m/s，约30km/h','失败率越低越好'),
 'overspeed':('活跃采样中超过4.1667m/s的比例<=5%','失败率越低越好'),
 'cruise':('有效巡航样本>=5，偏好速度范围内占比>=80%','样本不足为未知，不是失败'),
 'unblocked_low_speed':('有效无阻挡巡航采样中速度<.2m/s比例<=5%','无有效采样可为未知'),
 'unblocked_stall':('无阻挡低速连续时间<=1秒','失败率越低越好'),
 'actual_steering':('实际转向变化率P95<=1.5/秒','使用实际控制读取，不是策略均值动作'),
 'lane_p95':('道路参考车道偏离P95<=.6米','仅排除已记录合法变道过渡豁免'),
 'lane_peak':('道路参考车道偏离最大值<=1.2米','仅排除已记录合法变道过渡豁免'),
}
PPO={
 'approx_kl':('近似KL：均值[(ratio-1)-log(ratio)]，ratio=exp(new_logp-old_logp)。','无量纲','日志小批次/epoch均值；不是整批更新后精确KL。小不等于收敛。target_kl=.02。'),
 'clip_frac':('abs(ratio-1)>.15的样本占比，按日志均值导出。','比例0~1','表示概率比超出裁剪区间，不是梯度被截断比例；大/小均不能单独判成功。'),
 'post_tanh_entropy':('tanh压缩后的连续动作分布微分熵，当前策略重新采样估计。','nats','不是旧样本交叉熵；可以为负。大通常更随机，不一定更好。'),
 'grad_norm_actor':('Actor反向传播后、全局范数裁剪前的梯度L2范数。','梯度范数','clip_grad_norm_返回裁剪前值，所以超过.5不代表裁剪失效。'),
 'actor_loss':('本轮B0 Actor损失=surrogate_loss-.01*post_tanh_entropy；无互学习/anchor辅助项。','损失单位','不能当回报，正负都可能；不要求逐轮单调下降或趋0。'),
 'surrogate_loss':('PPO裁剪代理损失：-mean[min(ratio*A,clip(ratio,.85,1.15)*A)]，A已按角色批次标准化。','损失单位','不是负环境回报；接近0可能来自优势标准化，不能判学好。'),
 'entropy_bonus':('熵奖励项=.01*post_tanh_entropy；从Actor损失中减去。','损失单位','不是环境奖励，不包含在team_discounted_return中。'),
 'deterministic_steer_mean':('在更新小批次观测上，对确定性动作tanh(mu)的转向分量取均值。','[-1,1]','有正负抵消；均值近0不能证明转向平稳，且不是实际随机执行控制均值。'),
 'deterministic_longitudinal_mean':('在更新小批次观测上，对确定性动作tanh(mu)的纵向分量取均值。','[-1,1]','正值偏向驱动，负值偏向制动；经动作映射才成为油门/刹车，不是速度。'),
}
REWARD={
 'attack':'红0/1进度相对C1参考进度的损失项，权重2；更慢通常正、更快可负。使用事件资格和启动宽限门控。',
 'completed_task_tail':'红车均已到达时，把已知完成状态在剩余任务时间内的C1进度差折扣尾项折入本步；不虚构后续动作。',
 'driving':'蓝2/3路线进度的势函数差平均，gamma*p_t-p_prev，权重1；不是原双车基础奖励整体。',
 'critical':'合法前进且位于红车前半区域的红蓝TTC危险带收益；取配对最大值，累计加权暴露上限3。任一严重配对抑制正收益。',
 'primary_event_bonus':'同一红蓝配对.5<TTC<=1.5s连续>=.3s等条件成立后的单次+5；每回合最多一次。',
 'risk':'蓝车基础超速、射线近障和TTC成本之和的负值，两蓝平均。',
 'comfort':'蓝车基础纵向与转向变化成本的负值，两蓝平均。',
 'safety_event':'蓝方碰撞/越界/道路违规每类首次-50，任务违规首次-20；同类团队事件不反复扣，可跨类叠加。',
 'severe':'红蓝TTC<=.5s的严重险情连续惩罚，按2*dt*严重程度扣分，不作为正攻击收益。',
 'pace':'蓝车无前方阻挡、非近终点/终止等情形下，路线速度不足2.5m/s的连续惩罚；权重.1。',
 'normal_overspeed':'R2在基础超速成本之外追加的惩罚：额外9倍该基础超速成本，两蓝平均；不是超速率。',
 'normal_comfort':'R2在基础comfort之外追加的惩罚：额外4倍基础舒适性成本；与comfort合计为5倍。',
 'normal_lane':'R2追加参考车道偏离成本：10倍已有lane_deviation负项的幅度，两蓝平均；保留原合法豁免。',
}


def describe(tag):
    if tag in SUMMARY:return SUMMARY[tag]
    prefix,suffix=tag.split('/',1)
    if prefix in ('critical','reference_C1') and suffix in CRITICAL:
        d,u,n=CRITICAL[suffix]
        return (('普通冻结蓝车C1参考：' if prefix=='reference_C1' else '')+d,u,n)
    if prefix=='safety' and suffix.endswith('_rate'):
        event=suffix[:-5];names={'collision':'碰撞','offroad':'越界','road_rule':'道路违规','task_violation':'任务违规（如登记的无故停滞）'}
        return ('至少一辆蓝车发生'+names[event]+'的案例比例。','比例0~1','每例最多记1；各事件可重叠，不能相加当总失败率。')
    if prefix=='reward_component':return (REWARD[suffix],'回合累计奖励单位','先累计每回合分项，再对16回合取均值；分项一般未折扣，不能相加当折扣回报。')
    parts=tag.split('/')
    if prefix=='ppo':return PPO[parts[2].removesuffix('_logged_mean')]
    if prefix=='driving':
        if len(parts)==3:return DRIVING[parts[2].removesuffix('_episode_mean')]
        check=parts[3].removesuffix('_rate');condition,note=CHECKS[check]
        if parts[2]=='unknown_checks':return ('该项无法确定通过/失败的案例比例。条件为：'+condition+'。','比例0~1','缺失测量/无测量机会；不应当作合格，也不直接填作不合格。')
        return ('不满足以下条件的案例比例：'+condition+'。','比例0~1',note+'；各检查可重叠，不能相加当回合失败率。')
    raise ValueError('undocumented scalar '+tag)


def main():
    all_tags={};runs={}
    for directory in sorted((BASE/'events').glob('*/*')):
        run=directory.relative_to(BASE/'events').as_posix()
        acc=EventAccumulator(str(directory),size_guidance={'scalars':0}).Reload()
        tags=acc.Tags()['scalars'];runs[run]=len(tags)
        for tag in tags:
            points=acc.Scalars(tag);desc,unit,note=describe(tag)
            item=all_tags.setdefault(tag,dict(tag=tag,meaning=desc,unit=unit,note=note,runs={}))
            item['runs'][run]=dict(points=len(points),first_step=points[0].step,last_step=points[-1].step,
                first_value=points[0].value,last_value=points[-1].value)
    output=BASE/'metric_dictionary_20261105_v1';output.mkdir(exist_ok=False)
    metadata=dict(unique_tags=len(all_tags),run_tag_counts=runs,entries=[all_tags[t] for t in sorted(all_tags)],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'dictionary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    text=['# R16 TensorBoard完整指标字典','',f'实际事件文件：{len(all_tags)}个不同tag；各run标签数：'+json.dumps(runs,ensure_ascii=False),'',
        '这是历史JSONL补导。training/A和B是两套蓝方学习器，各31节点，每点16回合；frozen/A和B各4节点，每点24个固定案例。A/B不是红蓝双方，也不是独立外层seed。',
        '横轴Step是A+B合计训练world ticks，包含训练setup；不等于update、回合数或总含评估ticks。比例0.75=75%。',
        'training为随机动作采样，frozen为确定性策略；二者回报不能按同一采样分布解读。冻结wall time为导出时间，应使用Step比较。',
        'episode_mean是先算每例再平均，不是汇总全部帧后重算。logged_mean是对已有epoch/minibatch日志均值再次平均，不是所有样本等权重的最终更新后指标。',
        'P95是95%分位数。速度条件是研究偏好，不是法定限速。正式红方资格/独立seed评估不由这些单项自动授予。',
        '当前v1回放的failed_checks_overlapping和unknown_checks按原Counter稀疏写入：为0的键可能不写点。因此该类曲线缺点不等同于0或新的测量缺失，连线也不代表中间节点保持该值。请结合源summary.json；下表列出每个run实际点数。',
        '', '|标签|含义|单位|解读与限制|各run点数及最后实测值|','|---|---|---|---|---|']
    for tag,item in sorted(all_tags.items()):
        values='；'.join(f'{run}: n={v["points"]}, last={v["last_value"]:.6g}' for run,v in item['runs'].items())
        text.append('|`'+tag+'`|'+item['meaning']+'|'+item['unit']+'|'+item['note']+'|'+values+'|')
    (output/'全部指标解释.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(unique_tags=len(all_tags),run_tag_counts=runs,output=str(output)),ensure_ascii=False))


if __name__=='__main__':main()
