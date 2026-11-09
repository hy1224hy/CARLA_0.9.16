"""File-only final review and plots for R16 long experiment."""
import json
from pathlib import Path
import statistics


def build(root):
    from utils.analyze_r2_budget_results import analyze
    from utils.r2_budget_experiment import records,read
    from training.checkpoint_protocol import atomic_write_json as write
    output=root/'analysis/final_review_v1'
    result=analyze(root,output)
    returns=[]
    for node in read(root/'control/milestones.json'):
        report=read(root/'evaluations'/('frozen_'+str(node['target_train_ticks']).zfill(6))/'analysis/critical_report.json')
        for team in ('A','B'):
            cases=[c for c in report['reports'][0]['cases'] if c['condition']=='C2' and c['team']==team]
            returns.append(dict(node=node['target_train_ticks'],team=team,
                frozen_mean_discounted_return=statistics.mean(c['team_discounted_return'] for c in cases)))
    write(returns,root/'control/frozen_return_and_completion.json')
    last=records(root/'runs/R2/actual_cost_audit.jsonl')[-1]['actual_counters']
    frozen_ticks=sum(sum(c['tick_counts'].get('total',0) for c in records(p)) for p in (root/'evaluations').glob('*/evaluations/eval_000/cases.jsonl'))
    write(dict(actual_training_process_counters=last,frozen_evaluation_ticks=frozen_ticks,
        combined_explicit_ticks=last['total']+frozen_ticks),root/'control/cost_summary.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'axes.unicode_minus':False,'axes.grid':True,'grid.alpha':.2})
    metrics=[r for r in records(root/'runs/R2/metrics.jsonl') if 'training_window' in r]
    episodes=records(root/'runs/R2/episodes.jsonl')
    colors={'A':'#2474b7','B':'#df7a20'}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for team,color in colors.items():
        x=[r['actual_counters']['train']/1000 for r in metrics]
        reward=[statistics.mean(e['team_discounted_return'] for e in episodes if e['team']==team and e['round']==r['completed_round']) for r in metrics]
        def mean3(values):return [statistics.mean(values[max(0,i-2):i+1]) for i in range(len(values))]
        axes[0,0].plot(x,reward,color=color,alpha=.25);axes[0,0].plot(x,mean3(reward),color=color,label=team)
        bad=[100*r['training_window'][team]['b'] for r in metrics]
        axes[0,1].plot(x,bad,color=color,alpha=.25);axes[0,1].plot(x,mean3(bad),color=color,label=team)
        series=[r for r in result['series'] if r['team']==team]
        fx=[r['actual_train_ticks']/1000 for r in series]
        axes[1,0].plot(fx,[next(v['frozen_mean_discounted_return'] for v in returns if v['team']==team and v['node']==r['target_train_ticks']) for r in series],marker='o',color=color,label=team)
        axes[1,1].plot(fx,[100*r['interaction_quality']['joint_normal_resolved_C2'] for r in series],marker='o',color=color,label=team)
    for ax,title in zip(axes.flat,['训练折扣回报（深线最近3轮/48回合）','训练事件坏例率（%）','固定案例冻结回报','完整驾驶且无接触解除险情联合率（%）']):
        ax.set_title(title);ax.set_xlabel('A+B实际训练 world ticks（千）');ax.legend()
    axes[0,1].set_ylim(0,100);axes[1,1].set_ylim(-2,102)
    fig.suptitle('R16 / R2 72k弱红工程实验；A/B不是独立seed')
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=160);plt.close(fig)
    return result
