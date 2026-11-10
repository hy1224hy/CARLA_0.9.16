"""Frozen-model behavioral evaluation, separate from training and model selection."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from highwaytest.straight.environment import StraightEnv
from highwaytest.straight.runner import configuration, model_from_state
from highwaytest.training.checkpoint import source_digest, runtime_versions


def case_contract(split, cases):
    if split not in ('development', 'test') or cases < 2 or cases % 2:
        raise ValueError('Use development/test and an even case count >= 2')
    offset = 4000000 if split == 'development' else 3000000
    return [dict(seed=offset+i, family=('follow', 'adjacent')[i % 2]) for i in range(cases)]


def summarize(rows):
    if not rows:
        raise ValueError('Empty evaluation')
    result = dict(cases=len(rows))
    for key in ('attack_success','collision','target_collision','invalid','offroad',
                'grade_violation','baseline_valid','stop_qualified','full_horizon'):
        result[key+'_rate'] = sum(float(r.get(key, 0)) for r in rows)/len(rows)
    for key in ('return','utility','red_delay_m','cut_count','blocking_time','time',
                'max_target_brake','completed_lane_changes','boundary_fraction'):
        result[key] = sum(float(r.get(key, 0) or 0) for r in rows)/len(rows)
    result['repeated_cut_rate'] = sum(r.get('cut_count',0)>=2 for r in rows)/len(rows)
    for key in ('cut_first_thw','first_effective_cut_time','contact_time'):
        values=[r[key] for r in rows if r.get(key) is not None]
        result[key+'_conditional_mean']=float(np.mean(values)) if values else None
        result[key+'_observed_cases']=len(values)
    diag=Counter();parts=Counter();labels=Counter()
    for row in rows:
        diag.update(row.get('cut_diag',{}))
        parts.update(row.get('reward_components',{}))
        labels[row.get('behavior_label','unknown')]+=1
    result['cut_diagnostics_total']=dict(diag)
    result['mean_reward_components']={key:value/len(rows) for key,value in parts.items()}
    result['behavior_counts']=dict(labels)
    return result


def rollout(model, cfg, case):
    env=StraightEnv(cfg)
    obs,_=env.reset(seed=case['seed'], options={'family':case['family']})
    trace=[];total=0.;parts=Counter();boundary_time=0.;changes=0
    stable_lane=env._lane(env.attacker);first_cut=None;last_time=0.
    try:
        with torch.no_grad():
            while True:
                action=model.distribution(torch.as_tensor(obs,dtype=torch.float32)).loc.tanh().numpy()
                obs,reward,term,trunc,info=env.step(action)
                total+=reward;parts.update(info['reward_components'])
                a,v=env.attacker,env.target
                dt=info['time']-last_time;last_time=info['time']
                lane=env._lane(a);center_error=abs(float(a.position[1])-4*lane)
                if center_error>1.5:boundary_time+=dt
                if lane!=stable_lane and center_error<.5 and abs(a.heading)<np.deg2rad(10):
                    changes+=1;stable_lane=lane
                if first_cut is None and info['cut_count']:
                    first_cut=info['time']
                ref=env.reference.vehicles[1];gap=float(a.position[0]-v.position[0]-5.)
                trace.append(dict(time=info['time'],attacker_x=float(a.position[0]),
                    attacker_y=float(a.position[1]),target_x=float(v.position[0]),target_y=float(v.position[1]),
                    attacker_speed=float(a.speed),target_speed=float(v.speed),reference_speed=float(ref.speed),
                    attacker_heading=float(a.heading),gap=gap,
                    thw=gap/max(v.speed,2.) if gap>0 else 100.,
                    relative_speed_loss=float((ref.velocity[0]-v.velocity[0])/max(ref.velocity[0],5.)),
                    cut_count=info['cut_count'],action=action.tolist(),reward=reward))
                if term or trunc:break
        label=('invalid' if info['invalid'] else 'contact' if info['collision'] else
               'effective_cut' if info['cut_count'] else 'grade_violation' if info['grade_violation'] else
               'stop' if info['stop_qualified'] else 'mild_block' if info['blocking_time']>=1.5 else
               'no_qualified_attack')
        row=dict(info,case_seed=case['seed'],family=case['family'],**{'return':total},
                 reward_components=dict(parts),behavior_label=label,
                 completed_lane_changes=changes,boundary_fraction=boundary_time/max(info['time'],1e-9),
                 first_effective_cut_time=first_cut,contact_time=info['time'] if info['collision'] else None)
        return row,trace
    finally:
        env.close()


def plot_trace(trace, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    t=[r['time'] for r in trace]
    axes[0].plot([r['attacker_x'] for r in trace],[r['attacker_y'] for r in trace],label='attacker')
    axes[0].plot([r['target_x'] for r in trace],[r['target_y'] for r in trace],label='target')
    axes[0].axhline(2,color='gray',ls='--');axes[0].set(xlabel='x (m)',ylabel='y (m)')
    for key,label in (('attacker_speed','attacker'),('target_speed','target'),('reference_speed','reference')):
        axes[1].plot(t,[r[key] for r in trace],label=label)
    axes[1].set(xlabel='time (s)',ylabel='speed (m/s)')
    axes[2].plot(t,[min(r['thw'],5) for r in trace],label='THW (display capped at 5s)')
    axes[2].axhspan(.3,1.2,alpha=.15,color='green')
    axes[2].set(xlabel='time (s)',ylabel='THW (s)')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def evaluate_suite(runs, output, split='development', cases=100, checkpoint='selected', team='A'):
    contract=case_contract(split,cases)
    runs,output=Path(runs),Path(output)
    if output.exists():raise FileExistsError(output)
    if checkpoint not in ('selected','latest') or team not in ('A','B'):
        raise ValueError('checkpoint=selected/latest; team=A/B')
    sources=([runs] if runs.is_file() else [runs/(checkpoint+'.pt')]
             if (runs/(checkpoint+'.pt')).is_file() else sorted(runs.glob('*/'+checkpoint+'.pt')))
    if not sources:raise FileNotFoundError('No matching checkpoints under '+str(runs))
    first=torch.load(io.BytesIO(sources[0].read_bytes()),map_location='cpu',weights_only=False)
    if first.get('protocol')=='highway_hybrid_twin_v1':
        from highway_hybrid.runner import evaluate_suite as hybrid_evaluate
        return hybrid_evaluate(runs,output,split,cases,checkpoint,team)
    current_source=source_digest();versions=runtime_versions();frozen=[]
    # Freeze every input BEFORE the first rollout so live training cannot replace it.
    for path in sources:
        payload=path.read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        if state.get('protocol')!='straight_twin_v1':raise ValueError('Unsupported checkpoint '+str(path))
        if state['source_sha256']!=current_source:raise ValueError('Core source differs from checkpoint '+str(path))
        if state.get('packages') is not None and state['packages']!=versions:
            raise ValueError('Runtime differs from checkpoint '+str(path))
        if 'models' in state:
            state=dict(state,model=state['models'][team])
            selected_team=team;steps=state['train_steps']
        else:
            selected_team=(state.get('selection') or {}).get('team')
            steps=(state.get('selection') or {}).get('train_steps')
        frozen.append((path,payload,state,selected_team,steps))
    output.mkdir(parents=True)
    (output/'cases.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    tool_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summaries=[];torch.set_num_threads(1)
    for path,payload,state,selected_team,steps in frozen:
        name=path.parent.name
        if len({p.parent.name for p,_,_,_,_ in frozen})!=len(frozen):
            raise ValueError('Run names must be distinct')
        directory=output/name;directory.mkdir()
        (directory/'checkpoint.pt').write_bytes(payload)
        model=model_from_state(state);model.eval();cfg=configuration(state['config'])
        rows=[];represented=set();started=time.perf_counter()
        with (directory/'episodes.jsonl').open('w',encoding='utf-8') as episodes, \
             (directory/'trajectories.jsonl').open('w',encoding='utf-8') as trajectories:
            for index,case in enumerate(contract):
                row,trace=rollout(model,cfg,case);rows.append(row)
                episodes.write(json.dumps(row)+'\n');episodes.flush()
                trajectories.write(json.dumps(dict(case_seed=case['seed'],family=case['family'],steps=trace))+'\n')
                representative=(case['family'],bool(row['attack_success']))
                if representative not in represented:
                    represented.add(representative)
                    outcome='success' if row['attack_success'] else 'unsuccessful'
                    plot_trace(trace,directory/f'{case["family"]}_{outcome}_{case["seed"]}.png',
                               f'{name} / {case["family"]} / {outcome} / seed {case["seed"]}')
                if (index+1)%10==0:print(f'{name}: {index+1}/{cases}',flush=True)
        result=dict(run=name,split=split,config=cfg.to_dict(),team=selected_team,checkpoint_train_steps=steps,
            input_checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
            core_source_sha256=current_source,evaluator_sha256=tool_hash,packages=versions,
            evaluated_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
            overall=summarize(rows),families={f:summarize([r for r in rows if r['family']==f])
                                           for f in ('follow','adjacent')})
        (directory/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        with (directory/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['case_seed','family','attack_success','collision','invalid','grade_violation',
                    'cut_count','blocking_time','time','boundary_fraction','behavior_label','return']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        summaries.append(result)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    lines=['# 独立模型评估报告','',f'评估集：{split}；每模型 {cases} 个固定场景，两类各半。',
           '开发集用于诊断；test 用于模型冻结后的最终报告。成功率分母包含所有失败与无效回合。','',
           '| 模型 | 总成功率 | follow | adjacent | 碰撞率 | 无效率 | 平均有效切入 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        m=row['overall'];f=row['families']
        lines.append(f'| {row["run"]} | {m["attack_success_rate"]:.1%} | '
            f'{f["follow"]["attack_success_rate"]:.1%} | {f["adjacent"]["attack_success_rate"]:.1%} | '
            f'{m["collision_rate"]:.1%} | {m["invalid_rate"]:.1%} | {m["cut_count"]:.3f} |')
    lines+=['','每个模型目录包含冻结检查点、指标、逐回合 CSV/JSONL、全部轨迹和按场景自动选取的首次成功/未成功轨迹图。',
            '事件定义复用冻结训练环境，以保持任务口径一致；边界停留和完整换道为 5 Hz 轨迹诊断量，非物理子步精确事件。',
            '条件事件时间只对发生事件的回合取均值，另列发生回合数；跨等级回报不直接比较。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries
