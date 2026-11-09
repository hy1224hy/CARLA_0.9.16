"""Render logged R2 curves and retrospectively export TensorBoard event files."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def numeric_mean(value):
    if isinstance(value, list):
        values = [numeric_mean(v) for v in value]
        return statistics.mean(v for v in values if v is not None) if any(v is not None for v in values) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def rolling(values, weights, width=12):
    return [sum(v*w for v,w in zip(values[max(0,i-width+1):i+1],weights[max(0,i-width+1):i+1])) /
            sum(weights[max(0,i-width+1):i+1]) for i in range(len(values))]


def export_tensorboard(output, training, frozen):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    validation = {}
    for kind, data in (('training', training), ('frozen', frozen)):
        for team in ('A', 'B'):
            directory = output/kind/team
            writer = EventFileWriter(str(directory))
            rows = [r for r in data if r['team'] == team]
            for row in rows:
                timestamp = datetime.fromisoformat(row['created_at']).timestamp() if row.get('created_at') else time.time()
                values = [Summary.Value(tag=tag, simple_value=float(value))
                          for tag,value in row['scalars'].items() if value is not None]
                writer.add_event(Event(wall_time=timestamp, step=row['train_ticks'],summary=Summary(value=values)))
            writer.flush()
            writer.close()
            accumulator = EventAccumulator(str(directory),size_guidance={'scalars':0}).Reload()
            primary = 'return/team_discounted_mean' if kind == 'training' else 'quality/full_driving_and_resolved'
            points = accumulator.Scalars(primary)
            if len(points) != len(rows) or [p.step for p in points] != [r['train_ticks'] for r in rows]:
                raise ValueError('TensorBoard replay does not match source steps/counts')
            validation[kind+'/'+team] = dict(points=len(points),first_step=points[0].step,last_step=points[-1].step)
    return validation


def build(root, output):
    paths = [root/'runs/R2/metrics.jsonl', root/'runs/R2/episodes.jsonl',
             root/'analysis/final_review_v1/summary.json', root/'control/frozen_return_and_completion.json',
             ROOT/'configs/evaluation/base_driving_vnext_v1.json']
    fingerprint = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes = fingerprint()
    metrics = [r for r in lines(paths[0]) if 'training_window' in r]
    episodes = defaultdict(list)
    for row in lines(paths[1]):
        if not row['valid']:
            raise ValueError('invalid training episode cannot enter the plotted mean')
        episodes[(row['round'], row['team'])].append(row)
    training = []
    for row in metrics:
        for team in ('A','B'):
            window, ppo = row['training_window'][team], row['teams'][team]
            batch = episodes[(row['completed_round'],team)]
            if len(batch) != window['n']:
                raise ValueError('episode count differs from the committed training boundary')
            scalars = {
                'return/team_discounted_mean': statistics.mean(r['team_discounted_return'] for r in batch),
                'safety/blue_bad_rate': window['b'],
                'critical/legacy_event_qualified_rate': window['safe_critical_event'],
                'red/both_reached_rate': window['red_success_rate'],
                'value/explained_variance': ppo['explained_variance'],
                'value/critic_loss_logged_mean': numeric_mean(ppo['critic_loss']),
                'ppo/approx_kl_role_epoch_mean': statistics.mean(numeric_mean(ppo['approx_kl_'+str(a)]) for a in (2,3)),
            }
            for aid in (2,3):
                scalars['speed/blue_'+str(aid)+'_episode_mean_mps'] = window['agents'][str(aid)]['mean_speed_mps']
                for key in ('approx_kl','clip_frac','post_tanh_entropy','grad_norm_actor','deterministic_steer_mean'):
                    scalars['ppo/'+key+'_'+str(aid)+'_logged_mean'] = numeric_mean(ppo.get(key+'_'+str(aid)))
            for key,value in window['reward_components_mean'].items():
                scalars['reward_component/'+key] = value
            training.append(dict(team=team,round=row['completed_round'],train_ticks=row['actual_counters']['train'],
                created_at=row['created_at'],n=len(batch),scalars=scalars))
    frozen = []
    returns = {(r['node'],r['team']):r['frozen_mean_discounted_return'] for r in read(paths[3])}
    for row in read(paths[2])['series']:
        quality, legacy = row['interaction_quality'], row['legacy']
        frozen.append(dict(team=row['team'],train_ticks=row['actual_train_ticks'],target=row['target_train_ticks'],
            scalars={
                'return/team_discounted_mean': returns[(row['target_train_ticks'],row['team'])],
                'safety/blue_event_eligible_rate': row['blue_event_eligible_count']/row['n'],
                'safety/blue_collision_rate': row['blue_violations_overlapping']['collision']/row['n'],
                'critical/legacy_event_qualified_rate': legacy['qualified_critical_rate_C2'],
                'critical/precontact_predicted_conflict_rate': quality['predicted_conflict_C2'],
                'critical/observed_resolved_event_rate': quality['observed_contact_free_resolved_C2'],
                'critical/censored_event_rate': quality['contact_free_censored_C2'],
                'quality/full_driving_rate': quality['driving_quality_rate_C2'],
                'quality/full_driving_and_resolved': quality['joint_normal_resolved_C2'],
                'red/both_reached_rate': row['red_both_reached_count']/row['n'],
            }))
    output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    colors = {'A':'#2474b7','B':'#df7a20'}
    fig, axes = plt.subplots(3,2,figsize=(14,12))
    panels = [('return/team_discounted_mean','团队折扣回报','回报',1),
              ('safety/blue_bad_rate','蓝方事件坏例率','百分比（%）',100),
              ('critical/legacy_event_qualified_rate','事件资格危险率 G×H（不含完整驾驶资格）','百分比（%）',100),
              (None,'蓝车回合均速（不能替代超速比例资格）','m/s',1),
              ('value/explained_variance','Critic explained variance','EV',1),
              ('ppo/approx_kl_role_epoch_mean','策略更新：日志 KL 的角色/epoch 均值','估计 KL（symlog）',1)]
    for ax,(tag,title,ylabel,scale) in zip(axes.flat,panels):
        for team in ('A','B'):
            subset = [r for r in training if r['team'] == team]
            x = [r['train_ticks']/1000 for r in subset]
            weights = [r['n'] for r in subset]
            tags = [(tag,team,'-')] if tag else [('speed/blue_2_episode_mean_mps',team+' 蓝2','-'),('speed/blue_3_episode_mean_mps',team+' 蓝3','--')]
            for key,label,style in tags:
                y = [r['scalars'][key]*scale for r in subset]
                ax.plot(x,y,color=colors[team],alpha=.16,lw=.7,ls=style)
                ax.plot(x,rolling(y,weights),color=colors[team],lw=2,ls=style,label=label)
        for point in (24,48,72):
            ax.axvline(point,color='#999',ls=':',lw=.8)
        if tag is None:
            thresholds = read(paths[4])['thresholds']
            ax.axhspan(thresholds['preferred_speed_min_mps'],thresholds['preferred_speed_max_mps'],color='#71ac72',alpha=.12)
        if scale == 100:
            ax.set_ylim(0,100)
        if tag and 'approx_kl' in tag:
            ax.set_yscale('symlog',linthresh=1e-5)
        ax.set(title=title,xlabel='A+B 实际训练 world ticks（千）',ylabel=ylabel)
        ax.legend(loc='best',fontsize=10,ncol=2 if tag is None else 1)
    fig.suptitle('R2 / B0：随机采样训练曲线（seed42，CPU）',fontsize=19,y=.99)
    fig.text(.5,.952,'浅色：单轮日志；深色：尾随12轮加权均值（通常每队48回合）。评估与训练成本分开记录。',ha='center',fontsize=11)
    fig.text(.5,.012,'A/B 是同一外层实验的两个学习器，不是独立实验 seed；Critic EV 上升不能单独证明行为改善或收敛。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.925))
    fig.savefig(output/'training_curves.png',dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(13,8.5))
    panels = [('safety/blue_event_eligible_rate','蓝方事件资格通过率（不含完整驾驶质量）'),
              ('critical/precontact_predicted_conflict_rate','接触前持续预测冲突出现率'),
              ('critical/observed_resolved_event_rate','局部无接触且观测到解除的事件回合率'),
              ('quality/full_driving_and_resolved','完整驾驶质量与已解除险情联合率')]
    for ax,(tag,title) in zip(axes.flat,panels):
        for team in ('A','B'):
            subset = [r for r in frozen if r['team']==team]
            ax.plot([r['train_ticks']/1000 for r in subset],[r['scalars'][tag]*100 for r in subset],
                color=colors[team],lw=2,marker='o' if team=='A' else '^',ls='-' if team=='A' else '--',label=team)
        ax.set(title=title,xlabel='A+B 实际训练 world ticks（千）',ylabel='百分比（%）',ylim=(-2,102),xticks=[0,24,48,72])
        ax.legend()
    fig.suptitle('固定案例冻结评估：四个模型节点',fontsize=19,y=.99)
    fig.text(.5,.94,'每个节点、每队24个相同配对案例；局部险情解除不代表整回合无其他碰撞。',ha='center',fontsize=11)
    fig.text(.5,.018,'本轮所有节点的完整驾驶质量与险情联合结果均为 0/24。单 seed、弱红工程诊断，不作算法优越性结论。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.915))
    fig.savefig(output/'frozen_evaluation.png',dpi=170)
    plt.close(fig)
    validation = export_tensorboard(output/'tensorboard_replay',training,frozen)
    if hashes != fingerprint():
        raise ValueError('source logs changed while rendering')
    metadata = dict(derived_from_logs=True,native_training_tensorboard=False,source_hashes=hashes,
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_points_per_team=128,frozen_points_per_team=4,rolling_rounds=12,
        tensorboard_validation=validation,wall_time_note='training timestamps are original committed-boundary times; frozen replay events are written now')
    (output/'curve_data.json').write_text(json.dumps(dict(training=training,frozen=frozen),indent=2)+'\n',encoding='utf-8')
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'README.md').write_text('# R2曲线与TensorBoard回放\n\n本轮没有原生TensorBoard事件文件。此目录从已完成的JSONL/冻结评估日志派生，原始数据未修改。训练和冻结评估使用独立run，横轴为A+B实际train ticks。\n\n图中深色线为尾随12轮均值，TensorBoard保存未平滑逐轮值。training每队128点、frozen每队4点，已重新读取事件文件核验。\n\n在mappo_training工作目录运行：\n\n```powershell\n& \'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe\' -m tensorboard.main --logdir \"'+str(output/'tensorboard_replay')+'\" --port 6006\n```\n',encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'experiments/adversarial_u1000_R2_budget_20261104_v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.root.resolve(),args.output.resolve()),ensure_ascii=False,indent=2))
