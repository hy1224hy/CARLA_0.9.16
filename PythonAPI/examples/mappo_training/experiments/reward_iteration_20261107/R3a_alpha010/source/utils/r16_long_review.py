"""File-only final review and plots for R16 long experiment."""
import json
from pathlib import Path
import statistics


def build(root, output=None):
    from utils.analyze_r2_budget_results import analyze
    from utils.r2_budget_experiment import records,read
    from training.checkpoint_protocol import atomic_write_json as write
    output=output or root/'analysis/final_review_v1'
    result=analyze(root,output)
    returns=[]
    for node in read(root/'control/milestones.json'):
        report=read(root/'evaluations'/('frozen_'+str(node['target_train_ticks']).zfill(6))/'analysis/critical_report.json')
        for team in ('A','B'):
            cases=[c for c in report['reports'][0]['cases'] if c['condition']=='C2' and c['team']==team]
            returns.append(dict(node=node['target_train_ticks'],team=team,
                frozen_mean_discounted_return=statistics.mean(c['team_discounted_return'] for c in cases)))
    write(returns,output/'frozen_return_and_completion.json')
    last=records(root/'runs/R2/actual_cost_audit.jsonl')[-1]['actual_counters']
    frozen_ticks=sum(sum(c['tick_counts'].get('total',0) for c in records(p)) for p in (root/'evaluations').glob('*/evaluations/eval_000/cases.jsonl'))
    write(dict(actual_training_process_counters=last,frozen_evaluation_ticks=frozen_ticks,
        combined_explicit_ticks=last['total']+frozen_ticks),output/'cost_summary.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'axes.unicode_minus':False,'axes.grid':True,'grid.alpha':.2})
    metrics=[r for r in records(root/'runs/R2/metrics.jsonl') if 'training_window' in r]
    episodes=records(root/'runs/R2/episodes.jsonl')
    colors={'A':'#2474b7','B':'#df7a20'}
    fig,axes=plt.subplots(3,2,figsize=(13,12))
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
        axes[1,1].plot(fx,[100*r['legacy']['blue_eligible_rate_C2'] for r in series],marker='o',color=color,label=team)
        axes[2,0].plot(fx,[100*r['interaction_quality']['predicted_conflict_C2'] for r in series],marker='o',color=color,label=team)
        axes[2,1].plot(fx,[100*r['interaction_quality']['observed_contact_free_resolved_C2'] for r in series],marker='o',color=color,label=team)
    for ax,title in zip(axes.flat,['训练折扣回报（深线最近3轮/48回合）','训练事件坏例率（%）','固定案例冻结回报','冻结：蓝方事件资格率（非完整驾驶资格，%）','冻结：接触前持续危险交互率（%）','冻结：局部无接触且观测到解除率（%）']):
        ax.set_title(title);ax.set_xlabel('A+B实际训练 world ticks（千）');ax.legend()
    for ax in (axes[0,1],axes[1,1],axes[2,0],axes[2,1]):ax.set_ylim(-2,102)
    fig.suptitle('R16 / R2 72k弱红工程实验；A/B不是独立seed')
    fig.text(.5,.012,'所有冻结节点的完整驾驶资格及合格险情联合率均为0/24；局部险情解除不代表全回合无其他碰撞。',ha='center')
    fig.tight_layout(rect=(0,.035,1,.97));fig.savefig(output/'curves.png',dpi=170);plt.close(fig)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=build(a.root.resolve(),a.output.resolve())
    print(json.dumps(dict(status='DERIVED_ANALYSIS_COMPLETE',rows=len(result['series']),output=str(a.output.resolve()))))
