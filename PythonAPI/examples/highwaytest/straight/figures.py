"""Matplotlib figures for the straight protocol; data assembly lives in reporting."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .behavior import CLASSES


def _save(fig,path):
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=160);fig.savefig(str(path)+'.pdf');plt.close(fig)


def matrix_heatmap(result,path):
    rows=result['rows']
    labels=[f"L{r['level']}-{r['method'][:4]}" for r in rows]
    data=np.array([[r['classes'][name] for name in CLASSES] for r in rows])
    fig,ax=plt.subplots(figsize=(7,.7*max(len(rows),1)+1.4))
    im=ax.imshow(data,vmin=0.,vmax=1.,cmap='viridis',aspect='auto')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if data[i,j]>.5 else 'black')
    ax.set_xticks(range(len(CLASSES)),list(CLASSES));ax.set_yticks(range(len(rows)),labels)
    ax.set_xlabel('actual behavior class');ax.set_ylabel('training intent x method')
    ax.set_title('Level-consistency matrix (selection split, row-normalized)')
    fig.colorbar(im,ax=ax,fraction=.046)
    _save(fig,path)


def _label(entry):
    _,method,beta,geometry,effect,risk,event=entry
    label=method+(f' b={beta:g}' if method=='mutual' else '')
    if (geometry,effect,risk,event)!=(0.,1.,1.,1.):
        label+=f' g/d/r/e={geometry:g}/{effect:g}/{risk:g}/{event:g}'
    return label


def final_metric_figure(rows,groups,metric,outdir):
    levels=sorted({r['level'] for r in rows})
    fig,axes=plt.subplots(1,len(levels),figsize=(5*len(levels),4),squeeze=False)
    for ax,level in zip(axes[0],levels):
        entries=[(key,items) for key,items in groups.items() if key[0]==level]
        labels=[]
        for j,(key,items) in enumerate(entries):
            vals=[r[metric] for r in items if isinstance(r.get(metric),(int,float)) and np.isfinite(r[metric])]
            labels.append(_label(key))
            if not vals:continue
            ax.errorbar(j,np.mean(vals),yerr=np.std(vals,ddof=1) if len(vals)>1 else 0.,fmt='o',capsize=4)
            ax.scatter(np.full(len(vals),j)+np.linspace(-.06,.06,len(vals)),vals,s=18)
        ax.set_xticks(range(len(entries)),labels,rotation=20);ax.set_title(f'L{level}');ax.set_ylabel(metric)
    _save(fig,outdir/metric)


def learning_curve_figure(series_by_run,outdir):
    """Per-round winner series (selection) mean over seeds with per-seed thin lines."""
    for level in sorted({key[0] for key in series_by_run}):
        fig,axes=plt.subplots(1,2,figsize=(11,4),squeeze=False)
        for ax,method in zip(axes[0],('independent','mutual')):
            runs=series_by_run.get((level,method),[])
            if not runs:
                ax.set_title(f'L{level} {method} (no data)');continue
            steps=sorted({s for _,series in runs for s,_ in series})
            means=[];stds=[]
            for s in steps:
                vals=[rate for _,series in runs for ss,rate in series if ss==s]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals,ddof=1)) if len(vals)>1 else 0.)
            for _,series in runs:
                ax.plot([s for s,_ in series],[r for _,r in series],lw=.6,alpha=.6)
            ax.errorbar(steps,means,yerr=stds,color='black',lw=2,capsize=3,label=f'mean over {len(runs)} seeds')
            ax.legend(fontsize=7);ax.set_title(f'L{level} {method}')
            ax.set_xlabel('A+B total environment steps');ax.set_ylabel('selection success rate (per-round winner)')
        _save(fig,outdir/f'learning_curve_L{level}')


def stability_figure(rows,outdir):
    for level in (1,2,3):
        subset=[r for r in rows if r.get('level')==level]
        if not subset:continue
        groups={}
        for r in subset:groups.setdefault(r['method'],[]).append(r)
        fig,axes=plt.subplots(1,3,figsize=(13,4),squeeze=False)
        for ax,metric in zip(axes[0],('attack_success_rate','collision_rate','drawdown_max_to_min')):
            for j,(method,items) in enumerate(groups.items()):
                vals=[r[metric] for r in items if isinstance(r.get(metric),(int,float)) and np.isfinite(r[metric])]
                if not vals:continue
                ax.bar(j,np.mean(vals),yerr=np.std(vals,ddof=1) if len(vals)>1 else 0.,capsize=4)
                ax.scatter(np.full(len(vals),j)+np.linspace(-.08,.08,len(vals)),vals,s=16)
            ax.set_xticks(range(len(groups)),list(groups));ax.set_title(metric)
        _save(fig,outdir/f'stability_L{level}')


def _is_beta_row(r):
    return (r.get('level')==2 and r.get('risk_weight',1.)==1. and r.get('event_weight',1.)==1.
            and r.get('effect_weight',1.)==1. and r.get('geometry_weight',0.)==0.)


def _is_ablation_row(r):
    return (r.get('level')==2 and r.get('method')=='mutual' and r.get('beta')==.02
            and r.get('effect_weight',1.)==1.)


def _panel_categories(rows,name_fn):
    categories=[]
    for r in rows:
        name=name_fn(r)
        if name not in categories:categories.append(name)
    order={'0':0,'0.005':1,'0.02':2,'0.08':3,'full':0,'no_risk':1,'no_event':2,'no_effect':3}
    categories.sort(key=lambda n:(order.get(n,9),str(n)))
    return categories


def _treatment_panel(rows,selector,outdir,title,name_fn):
    subset=[r for r in rows if selector(r)]
    if not subset:return
    categories=_panel_categories(subset,name_fn)
    fig,axes=plt.subplots(1,3,figsize=(13,4),squeeze=False)
    for ax,metric in zip(axes[0],('attack_success_rate','utility','drawdown_max_to_min')):
        for j,name in enumerate(categories):
            items=[r for r in subset if name_fn(r)==name]
            vals=[r[metric] for r in items if isinstance(r.get(metric),(int,float)) and np.isfinite(r[metric])]
            if not vals:continue
            ax.bar(j,np.mean(vals),yerr=np.std(vals,ddof=1) if len(vals)>1 else 0.,capsize=4)
            ax.scatter(np.full(len(vals),j)+np.linspace(-.08,.08,len(vals)),vals,s=16)
        ax.set_xticks(range(len(categories)),categories);ax.set_title(metric)
    fig.suptitle(title)
    _save(fig,outdir/title.replace(' ','_'))


def _beta_name(r):
    b=r['beta']
    return '0' if b==0 else f'{b:g}'


def beta_panel(rows,outdir):
    """beta=0 categorical axis; only full-reward L2 runs, one category per beta."""
    _treatment_panel(rows,_is_beta_row,outdir,'beta_sensitivity_L2',_beta_name)


def ablation_panel(rows,outdir):
    _treatment_panel(rows,_is_ablation_row,outdir,'reward_ablation_L2',lambda r:r.get('ablation') or 'full')


def trajectory_figure(trajectory_paths,outdir):
    """One figure per run: x-y, speeds, bumper gap for up to three representative cases."""
    for path in sorted(trajectory_paths):
        data=json.loads(path.read_text(encoding='utf-8'))
        cases=data.get('cases',[])
        if not cases:continue
        run_name=path.parent.parent.name
        fig,axes=plt.subplots(len(cases),3,figsize=(13,3.4*len(cases)),squeeze=False)
        for row,case in zip(axes,cases):
            t=[s['time'] for s in case['steps']]
            ax_,ay_=zip(*[s['attacker_position'] for s in case['steps']])
            tx,ty=zip(*[s['target_position'] for s in case['steps']])
            av=[s['attacker_speed'] for s in case['steps']];tv=[s['target_speed'] for s in case['steps']]
            gap=[max(0.,abs(s['attacker_position'][0]-s['target_position'][0])-5.) for s in case['steps']]
            row[0].plot(ax_,ay_,'b-',lw=1.5,label='attacker');row[0].plot(tx,ty,'r-',lw=1.5,label='target')
            row[0].set_title(f"case {case['case_seed']} ({case['family']})");row[0].set_xlabel('x (m)');row[0].set_ylabel('y (m)')
            row[0].legend(fontsize=7)
            row[1].plot(t,av,'b-',label='attacker');row[1].plot(t,tv,'r-',label='target')
            row[1].set_xlabel('t (s)');row[1].set_ylabel('speed (m/s)');row[1].legend(fontsize=7)
            row[2].plot(t,gap,'k-');row[2].set_xlabel('t (s)');row[2].set_ylabel('bumper gap (m)')
        fig.suptitle(f'Representative trajectories: {run_name}')
        _save(fig,outdir/f'trajectory_{run_name}')
