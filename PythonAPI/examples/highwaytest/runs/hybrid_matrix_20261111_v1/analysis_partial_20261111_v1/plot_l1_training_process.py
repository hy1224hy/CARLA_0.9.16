"""Two complete L1 process figures, with all seeds and replicas retained."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'l1_training_process'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
METHODS=['independent','mutual']
LABELS=['PPO','Mut-PPO']
COLORS=['#4277b8','#e08b38']
curves={}
export=[]
for method in METHODS:
    seed_curves=[]
    train_curves=[]
    clocks=None
    for seed in range(42,47):
        folder=ROOT.parent/'train'/'main'/f'L1_{method}_s{seed}'
        vals=[json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
        ticks=np.array([v['train_steps'] for v in vals])
        if clocks is None:clocks=ticks
        assert np.array_equal(ticks,clocks)
        records=[]
        for val in vals:
            replicas=[val['reports'][team]['overall'] for team in ['A','B']]
            records.append([np.mean([r[key] for r in replicas])*100 for key in ['attack_success_rate','collision_rate','grade_violation_rate']])
        seed_curves.append(records)
        episodes=[json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
        bins=[]
        for lo in range(0,500000,20000):
            replica_means=[]
            for team in ['A','B']:
                rows=[e for e in episodes if e['team']==team and lo<e['train_steps']<=lo+20000]
                assert rows
                replica_means.append(np.mean([e['attack_success'] for e in rows])*100)
            bins.append(np.mean(replica_means))
        train_curves.append(bins)
    curves[method]=dict(steps=clocks,validation=np.array(seed_curves),train=np.array(train_curves))
    for i,step in enumerate(clocks):
        for k,metric in enumerate(['selection_asr','selection_collision','selection_grade_violation']):
            values=curves[method]['validation'][:,i,k]
            export.append(dict(method=method,steps=int(step),metric=metric,mean=float(values.mean()),sample_sd=float(values.std(ddof=1))))
    for i,step in enumerate(range(20000,500001,20000)):
        values=curves[method]['train'][:,i]
        export.append(dict(method=method,steps=step,metric='training_asr_20k_bin',mean=float(values.mean()),sample_sd=float(values.std(ddof=1))))
assert np.array_equal(curves[METHODS[0]]['steps'],curves[METHODS[1]]['steps'])

def draw(ax,mode,column=None):
    for method,label,color in zip(METHODS,LABELS,COLORS):
        item=curves[method]
        if mode=='train':
            x=np.arange(20000,500001,20000)/1000
            matrix=item['train']
        else:
            x=item['steps']/1000
            matrix=item['validation'][:,:,column]
        mean=matrix.mean(axis=0);sd=matrix.std(axis=0,ddof=1)
        ax.plot(x,mean,label=label,color=color,linewidth=2.3)
        ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=color,alpha=.15)
    ax.axvspan(0,20,color='#bbbbbb',alpha=.17)
    ax.axvspan(20,40,color='#bbbbbb',alpha=.09)
    ax.set_xlim(0,500);ax.set_ylim(0,105)
    ax.set_xlabel('合计训练步数（千步，A+B）')
    ax.set_ylabel('比例（%）')
    ax.grid(alpha=.2);ax.legend(loc='best')

fig,axes=plt.subplots(1,2,figsize=(13,4.8))
draw(axes[0],'train');draw(axes[1],'selection',0)
axes[0].set_title('训练回合成功率（随机执行）')
axes[1].set_title('固定selection场景成功率（确定性执行）')
fig.suptitle('L1训练过程：攻击能力的形成与保持',fontsize=17)
fig.text(.5,.015,'全部5个种子；先平均A/B，再跨种子取均值±样本SD。左：每20k步分箱；右：原始评估点。未额外平滑、未挑选种子。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.055,1,.94))
fig.savefig(OUT/'L1_training_success.png',dpi=180);fig.savefig(OUT/'L1_training_success.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,4.8))
draw(axes[0],'selection',1);draw(axes[1],'selection',2)
axes[0].set_title('固定selection场景碰撞率（越低越好）')
axes[1].set_title('固定selection场景越级率（越低越好）')
for ax in axes:
    ax.set_ylim(0,40)
    ax.axhline(12.5,color='#777777',linestyle='--',linewidth=1)
    ax.text(495,13.5,'选模单项门槛12.5%',ha='right',fontsize=9,color='#555555')
fig.suptitle('L1训练过程：攻击成功之外的约束代价',fontsize=17)
fig.text(.5,.015,'每次评估32个固定场景；全部5种子，先平均A/B，再取种子均值±样本SD。门槛用于各副本选模，组均值过线不等于所有模型合格。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.055,1,.94))
fig.savefig(OUT/'L1_training_constraints.png',dpi=180);fig.savefig(OUT/'L1_training_constraints.pdf');plt.close(fig)

with (OUT/'curves.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
summary={}
for method in METHODS:
    v=curves[method]['validation'][:,curves[method]['steps']>=375000,:]
    summary[method]={}
    for k,key in enumerate(['selection_asr','selection_collision','selection_grade_violation']):
        per_seed=v[:,:,k].mean(axis=1)
        summary[method][key]=dict(mean=float(per_seed.mean()),sample_sd=float(per_seed.std(ddof=1)))
(OUT/'late_window_statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# L1训练过程图

数据：main下全部L1，两方法各5个种子，每个种子A/B两副本。
成功率图左侧以20k步分箱，每副本先按完整训练回合统计，再平均副本、种子。
其余图使用原始selection记录，每次固定32个场景，确定性执行，先平均A/B再平均种子。
阴影为种子间样本标准差，非置信区间。未额外平滑、未使用A/B最优包络、未挑选种子。
横轴为A+B合计步数，灰色背景标示Mut-PPO前20k预热及20k至40k的beta爬升。
曲线描述训练过程，selection参与选模，不代替独立development/test结论。
碰撞、越级门槛针对单个副本；组平均不能用于判定每个模型的资格。
''',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(OUT)
