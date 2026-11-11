"""Audit both final replicas at exactly 500k; retain matched seed aggregation."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import torch

from highway_hybrid.runner import evaluate_suite

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'l2_latest_500k'


def main():
    OUT.mkdir(exist_ok=True)
    frozen=OUT/'checkpoints'
    frozen.mkdir(exist_ok=True)
    manifest=[]
    for method in ['independent','mutual']:
        for seed in range(42,47):
            name=f'L2_{method}_s{seed}'
            source=ROOT.parent/'train'/'main'/name/'latest.pt'
            target=frozen/name/'latest.pt'
            if not target.exists():
                payload=source.read_bytes()
                state=torch.load(source,map_location='cpu',weights_only=False)
                assert state['train_steps']==500000, (name,state['train_steps'])
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(payload)
            state=torch.load(target,map_location='cpu',weights_only=False)
            assert state['train_steps']==500000
            manifest.append(dict(run=name,steps=state['train_steps'],sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summaries={}
    for team in ['A','B']:
        folder=OUT/f'audit_{team}'
        if not (folder/'summary.json').exists():
            print('Evaluating final replica',team,flush=True)
            summaries[team]=evaluate_suite(frozen,folder,'development',40,'latest',team)
        else:
            summaries[team]=json.loads((folder/'summary.json').read_text())
        print('Completed replica',team,flush=True)
    by={(team,row['run']):row for team,rows in summaries.items() for row in rows}
    metrics=['attack_success_rate','collision_rate','grade_violation_rate','blocking_time','completed_lane_changes','return','cut_count','repeated_cut_rate','red_delay_m','utility']
    rows=[]
    for method in ['independent','mutual']:
        for seed in range(42,47):
            name=f'L2_{method}_s{seed}'
            for team in ['A','B']:
                report=by[team,name]
                row=dict(method=method,seed=seed,team=team,steps=500000)
                row.update({key:report['overall'][key] for key in metrics})
                row.update({fam+'_asr':report['families'][fam]['attack_success_rate'] for fam in ['follow','adjacent']})
                rows.append(row)
    fields=metrics+['follow_asr','adjacent_asr']
    grouped={}
    for method in ['independent','mutual']:
        grouped[method]={}
        for team in ['A','B','A_B_mean']:
            seed_rows=[]
            for seed in range(42,47):
                chosen=[r for r in rows if r['method']==method and r['seed']==seed and (team=='A_B_mean' or r['team']==team)]
                seed_rows.append({key:statistics.mean(r[key] for r in chosen) for key in fields})
            grouped[method][team]={key:dict(mean=statistics.mean(r[key] for r in seed_rows),sample_sd=statistics.stdev(r[key] for r in seed_rows)) for key in fields}
    paired={}
    for key in fields:
        diffs=[]
        for seed in range(42,47):
            means={m:statistics.mean(r[key] for r in rows if r['method']==m and r['seed']==seed) for m in ['independent','mutual']}
            diffs.append(means['mutual']-means['independent'])
        delta=statistics.mean(diffs);half=2.7764451051977987*statistics.stdev(diffs)/(5**.5)
        paired[key]=dict(differences=diffs,mean=delta,exploratory_t95=[delta-half,delta+half])
    stats=dict(groups=grouped,paired=paired)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    with (OUT/'results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# L2最终500k模型独立开发评估','','每方法5个训练种子，每种子评估A/B两个最终副本，各40个固定开发场景。确定性执行。',
        '主统计先在每个种子内平均A/B，再计算5个种子的均值和样本SD。每方法400回合，独立训练重复数仍为5。',
        '未按本开发评估成绩挑选副本；selected历史最优结果另列，不能与这里混为同一口径。','',
        '| 方法/副本 | 成功率 | 碰撞率 | 越级率 | 阻碍时间(s) | 换道次数 | 同车道成功率 | 邻道成功率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in grouped:
        for team in grouped[method]:
            obj=grouped[method][team]
            vals=[]
            for key in ['attack_success_rate','collision_rate','grade_violation_rate','blocking_time','completed_lane_changes','follow_asr','adjacent_asr']:
                factor=100 if 'rate' in key or 'asr' in key else 1
                vals.append(f'{obj[key]["mean"]*factor:.2f} ± {obj[key]["sample_sd"]*factor:.2f}'+('%' if factor==100 else ''))
            lines.append('| '+method+'/'+team+' | '+' | '.join(vals)+' |')
    lines+=['','## 配对差（Mut-PPO − PPO）','',f'成功率差：{paired["attack_success_rate"]["mean"]*100:.2f}个百分点；探索性配对t 95%区间：'+str([round(v*100,2) for v in paired['attack_success_rate']['exploratory_t95']])+'个百分点。n=5，区间依赖分布假设。',
        '','A/B均值对应不事后挑选副本的算法比较；部署单个策略时，可另报告预先固定A副本或使用selection选副本的协议，不能用本开发集挑选。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(grouped)
    print('\n'.join(lines),flush=True)


def plot(grouped):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,(key,title,factor) in zip(axes.flat,[('attack_success_rate','L2成功率（%）',100),('blocking_time','平均阻碍时间（秒）',1),('cut_count','平均有效切入次数',1),('collision_rate','碰撞率（%）',100),('follow_asr','同车道起始成功率（%）',100),('adjacent_asr','邻道起始成功率（%）',100)]):
        for i,method in enumerate(['independent','mutual']):
            s=grouped[method]['A_B_mean'][key]
            mean,sd=s['mean']*factor,s['sample_sd']*factor
            ax.bar(i,mean,yerr=sd,capsize=5,color=['#4277b8','#e08b38'][i],alpha=.8)
            label=f'{mean:.3f} ± {sd:.3f}' if key=='cut_count' else f'{mean:.2f} ± {sd:.2f}'
            ax.annotate(label,(i,mean+sd),xytext=(0,6),textcoords='offset points',ha='center')
        upper=max((grouped[m]['A_B_mean'][key]['mean']+grouped[m]['A_B_mean'][key]['sample_sd'])*factor for m in ['independent','mutual'])
        ax.set_xticks([0,1],['PPO','Mut-PPO']);ax.set_title(title);ax.set_ylim(0,max(upper*1.22,.01));ax.grid(axis='y',alpha=.2)
    fig.suptitle('L2：最终500k步模型的独立评估（A/B均值）',fontsize=17)
    fig.text(.5,.015,'每方法5种子 × 2副本 × 相同40场景；先平均A/B，再跨种子统计。误差线：种子间样本SD。确定性执行。',ha='center')
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'L2_latest_500k.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
