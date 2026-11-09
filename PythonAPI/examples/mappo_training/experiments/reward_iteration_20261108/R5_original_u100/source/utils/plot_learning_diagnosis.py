"""Plot completed diagnosis logs without contacting the simulator."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def records(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def smooth(rows,field):
    result=[]
    for i in range(len(rows)):
        remaining,total,count=48,0.,0
        for row in reversed(rows[:i+1]):
            n=min(remaining,row['n']);total+=n*row[field];count+=n;remaining-=n
            if not remaining:break
        result.append(total/count)
    return result


def build():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    root=ROOT/'experiments/adversarial_u1000_learning_diagnosis_20261104_v3'
    out=root/'analysis/curves_20261104_v1';out.mkdir(exist_ok=False)
    inputs=[root/'analysis/summary.json',root/'analysis/case_review_20261104_v1/summary.json']
    for group in ('N4','R4','R16'):
        inputs.extend([root/'runs'/group/'metrics.jsonl',root/'runs'/group/'episodes.jsonl'])
    fingerprints=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    hashes=fingerprints()
    review=json.loads(inputs[0].read_text(encoding='utf-8'))
    cases=json.loads(inputs[1].read_text(encoding='utf-8'))['rows']
    training=[]
    for group in ('N4','R4','R16'):
        episodes=records(root/'runs'/group/'episodes.jsonl')
        for row in records(root/'runs'/group/'metrics.jsonl'):
            if 'training_window' not in row:continue
            for team in ('A','B'):
                batch=[e for e in episodes if e['round']==row['completed_round'] and e['team']==team]
                window=row['training_window'][team]
                assert len(batch)==window['n'] and all(e['valid'] for e in batch)
                training.append(dict(group=group,team=team,round=row['completed_round'],
                    ticks=row['actual_counters']['train'],n=len(batch),
                    reward=statistics.mean(e['team_discounted_return'] for e in batch),
                    bad=100*window['b'],ev=row['teams'][team]['explained_variance']))
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'axes.unicode_minus':False,'font.size':11,'axes.grid':True,'grid.alpha':.2,
        'axes.spines.top':False,'axes.spines.right':False})
    colors={'A':'#2474b7','B':'#df7a20'}
    labels={'N4':'N4：关闭对抗项，4回合/更新','R4':'R4：R2，4回合/更新','R16':'R16：R2，16回合/更新'}
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,sharey='row')
    for col,group in enumerate(labels):
        for ridx,(field,title) in enumerate([('reward','团队折扣回报'),('bad','蓝方事件坏例率（%）'),('ev','Critic explained variance')]):
            ax=axes[ridx,col]
            for team in colors:
                rows=[r for r in training if r['group']==group and r['team']==team]
                x=[r['ticks']/1000 for r in rows]
                ax.plot(x,[r[field] for r in rows],color=colors[team],alpha=.25,lw=.8,marker='.' if group=='R16' else None)
                ax.plot(x,smooth(rows,field),color=colors[team],lw=2,label=team)
            ax.set_title(labels[group]+'\n'+title if ridx==0 else title,fontsize=12)
            ax.set_xlim(0,25.5);ax.legend(fontsize=9)
            if field=='bad':ax.set_ylim(0,100)
            if ridx==2:ax.set_xlabel('A+B 实际训练 world ticks（千）')
    fig.suptitle('三组随机采样训练曲线：回报、失败与价值预测',fontsize=19,y=.99)
    fig.text(.5,.938,'浅线：每次更新的采样均值；深线：最近48回合均值（不足48时用已有回合）。R16每点采样更多。',ha='center',fontsize=11)
    fig.text(.5,.014,'单外层 seed42；A/B 不是独立 seed。N4 与 R2 的奖励定义不同，绝对回报不直接作组间排名；EV 上升不等于策略收敛。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.915));fig.savefig(out/'training_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,sharey='row')
    frozen=[]
    for col,group in enumerate(labels):
        for team in colors:
            qs=[r for r in review['frozen'] if r['group']==group and r['quality']['team']==team]
            for row in qs:
                case=next(r for r in cases if r['group']==group and r['team']==team and r['target']==row['node']['target'])
                frozen.append(dict(group=group,team=team,ticks=row['node']['actual'],
                    reward=case['frozen_mean_return'],eligible=100*row['legacy']['blue_eligible_rate_C2'],
                    conflict=100*row['quality']['predicted_conflict_C2']))
        for ridx,(field,title) in enumerate([('reward','冻结评估平均折扣回报'),('eligible','蓝方事件资格通过率（%）'),('conflict','接触前持续预测冲突率（%）')]):
            ax=axes[ridx,col]
            for team in colors:
                rows=[r for r in frozen if r['group']==group and r['team']==team]
                ax.plot([r['ticks']/1000 for r in rows],[r[field] for r in rows],color=colors[team],
                    marker='o' if team=='A' else '^',ls='-' if team=='A' else '--',label=team)
            ax.set_title(labels[group]+'\n'+title if ridx==0 else title,fontsize=12)
            ax.set_xlim(-.7,25.5);ax.set_xticks([0,12,24]);ax.legend(fontsize=9)
            if field!='reward':ax.set_ylim(-2,102)
            if ridx==2:ax.set_xlabel('A+B 实际训练 world ticks（千）')
    fig.suptitle('固定案例冻结评估：每组3个模型节点',fontsize=19,y=.99)
    fig.text(.5,.938,'每节点、每队24个相同案例；图中事件资格不包含完整速度、车道、转向要求。折线只连接实测节点。',ha='center',fontsize=11)
    fig.text(.5,.014,'所有节点的完整驾驶质量合格率及完整驾驶与已解除险情联合率均为0/24。仅3个节点，不能据此确认收敛。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.915));fig.savefig(out/'frozen_curves.png',dpi=170);plt.close(fig)
    assert hashes==fingerprints(),'source evidence changed'
    meta=dict(source_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_points_per_team={g:sum(r['group']==g and r['team']=='A' for r in training) for g in labels},
        smoothing='trailing48 episodes, weighted by episode count; no smoothing on frozen',frozen_points_per_team=3)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'curve_data.json').write_text(json.dumps(dict(training=training,frozen=frozen),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),**{k:v for k,v in meta.items() if k!='source_hashes'}),ensure_ascii=False))


if __name__=='__main__':build()
