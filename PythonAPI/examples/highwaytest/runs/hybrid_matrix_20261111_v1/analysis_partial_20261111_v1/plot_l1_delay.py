"""Show raw progress deficit alongside existing constrained normalized utility."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
out=root/'l1_training_process'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
rows=[]
summary={}
for method,label,color in [('independent','PPO','#4277b8'),('mutual','Mut-PPO','#e08b38')]:
    curves=[]
    for seed in range(42,47):
        path=root.parent/'train'/'main'/f'L1_{method}_s{seed}'/'validation.jsonl'
        vals=[json.loads(s) for s in path.read_text().splitlines()]
        steps=np.array([v['train_steps'] for v in vals])
        curves.append([[np.mean([v['reports'][t]['overall'][key] for t in ['A','B']]) for key in ['red_delay_m','utility']] for v in vals])
    matrix=np.array(curves)
    summary[label]={}
    for i,key in enumerate(['red_delay_m','utility']):
        mean=matrix[:,:,i].mean(axis=0);sd=matrix[:,:,i].std(axis=0,ddof=1)
        axes[i].plot(steps/1000,mean,color=color,label=label,linewidth=2)
        axes[i].fill_between(steps/1000,mean-sd,mean+sd,color=color,alpha=.15)
        for step,m,s in zip(steps,mean,sd):rows.append(dict(method=label,steps=int(step),metric=key,mean=float(m),sample_sd=float(s)))
        late=matrix[:,steps>=375000,i].mean(axis=1)
        summary[label][key]=dict(late_mean=float(late.mean()),late_seed_sd=float(late.std(ddof=1)))
axes[0].set_title('原始纵向进度损失（含提前终止回合）')
axes[0].set_ylabel('参考目标车位置 − 实际目标车位置（米）')
axes[1].set_title('现有utility：完整时长且未碰撞、越级、无效的归一化进度损失')
axes[1].set_ylabel('utility（其余回合记0）')
for ax in axes:
    ax.set_xlabel('合计训练步数（千步，A+B）');ax.set_xlim(0,500);ax.grid(alpha=.2);ax.legend()
fig.suptitle('L1：攻击影响程度与约束表现',fontsize=17)
fig.text(.5,.015,'固定32个selection场景；全部5种子，先平均A/B，再跨种子取均值±样本SD。原始距离在各回合结束时测量，观测时长可能不同。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94))
fig.savefig(out/'L1_training_delay_utility.png',dpi=180);fig.savefig(out/'L1_training_delay_utility.pdf');plt.close(fig)
with (out/'delay_utility_curves.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(out/'delay_utility_statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
