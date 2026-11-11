"""Independent twin training, transactional checkpoints, and frozen-model audits."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from highwaytest.training.checkpoint import save,runtime_versions
from highwaytest.training.locking import run_lock
from highway_eval.evaluate import case_contract,summarize as base_summarize,plot_trace
from .config import HybridConfig,PROTOCOL
from .environment import HybridEnv
from .learning import collect,update,beta_at


def summarize(rows):
    report=base_summarize(rows)
    if rows:
        for key in ('lane_changes_started','dangerous_cut_count','behind_positive_impact','blocking_time_10s'):
            report[key]=float(np.mean([r.get(key,0.) for r in rows]))
        report['early_success_rate']=float(np.mean([r.get('early_success',False) for r in rows]))
        for relation in ('behind','ahead'):
            selected=[r for r in rows if r.get('initial_relation')==relation]
            report['initial_'+relation+'_cases']=len(selected)
            if selected:report['initial_'+relation+'_asr']=float(np.mean([r['attack_success'] for r in selected]))
        report['causal_cut_diagnostics_total']={key:sum(r.get('causal_cut_diag',{}).get(key,0) for r in rows)
            for key in {k for r in rows for k in r.get('causal_cut_diag',{})}}
    return report
from .model import HybridActorCritic


def source_files():
    examples=Path(__file__).resolve().parents[1]
    files=list(Path(__file__).parent.glob('*.py'))+list((Path(__file__).parent/'straight').glob('*.py'))
    files.extend(examples/path for path in (
        'highwaytest/training/ppo.py',
        'highwaytest/training/checkpoint.py','highwaytest/training/locking.py',
        'highwaytest/config.py','highway_eval/evaluate.py'))
    return sorted(files,key=lambda p:p.relative_to(examples).as_posix())


def source_digest():
    examples=Path(__file__).resolve().parents[1];digest=hashlib.sha256()
    for path in source_files():
        digest.update(path.relative_to(examples).as_posix().encode())
        digest.update(path.read_text(encoding='utf-8-sig').replace('\r\n','\n').encode())
    return digest.hexdigest()


def load_checkpoint(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('protocol')!=PROTOCOL:raise ValueError('Not a hybrid-action checkpoint')
    return state


def append(path,row):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def evaluate_model(model,cfg,split='selection',cases=32,capture=False):
    if split=='selection':
        if cases<1:raise ValueError('positive case count required')
        contract=[dict(seed=2000000+i,family=('follow','adjacent')[i%2]) for i in range(cases)]
    else:contract=case_contract(split,cases)
    env=HybridEnv(cfg);episodes=[];traces=[];model.eval()
    try:
        with torch.no_grad():
            for case in contract:
                obs,_=env.reset(seed=case['seed'],options={'family':case['family']})
                total=0.;parts={};trace=[];changes=0;boundary_time=0.;last_time=0.
                stable=env._lane(env.attacker);first_cut=None
                while True:
                    acceleration,lane=model.deterministic(torch.as_tensor(obs['state']),torch.as_tensor(obs['lane_mask']))
                    obs,reward,term,trunc,info=env.step(dict(acceleration=acceleration.numpy(),lane=int(lane)))
                    total+=reward
                    for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
                    a,v=env.attacker,env.target
                    actual=env._lane(a);error=abs(a.position[1]-4*actual)
                    if actual!=stable and error<.5 and abs(a.heading)<np.deg2rad(10):
                        changes+=1;stable=actual
                    if error>1.5:boundary_time+=info['time']-last_time
                    last_time=info['time']
                    if first_cut is None and info['cut_count']:first_cut=info['time']
                    if capture:
                        gap=float(a.position[0]-v.position[0]-5.)
                        trace.append(dict(time=info['time'],attacker_x=float(a.position[0]),attacker_y=float(a.position[1]),
                            target_x=float(v.position[0]),target_y=float(v.position[1]),attacker_speed=float(a.speed),
                            target_speed=float(v.speed),reference_speed=float(env.reference.vehicles[1].speed),
                            thw=gap/max(v.speed,2.) if gap>0 else 100.,cut_count=info['cut_count'],
                            acceleration=float(acceleration[0]),lane_action=int(lane),
                            lane_target=info['lane_target'],lane_busy=info['lane_busy']))
                    if term or trunc:break
                label=('invalid' if info['invalid'] else 'contact' if info['collision'] else
                       'effective_cut' if info['cut_count'] else 'grade_violation' if info['grade_violation'] else
                       'stop' if info['stop_qualified'] else 'mild_block' if info['blocking_time']>=1.5 else
                       'no_qualified_attack')
                episodes.append(dict(info,case_seed=case['seed'],**{'return':total},reward_components=parts,
                    completed_lane_changes=changes,boundary_fraction=boundary_time/max(info['time'],1e-9),
                    first_effective_cut_time=first_cut,contact_time=info['time'] if info['collision'] else None,
                    behavior_label=label))
                if capture:traces.append(dict(case_seed=case['seed'],family=case['family'],success=info['success'],steps=trace))
    finally:
        env.close();model.train()
    overall=summarize(episodes)
    overall['success_rate']=sum(e['success'] for e in episodes)/len(episodes)
    return dict(split=split,overall=overall,
                families={family:summarize([e for e in episodes if e['family']==family])
                          for family in ('follow','adjacent') if any(e['family']==family for e in episodes)},
                episodes=episodes,trajectories=traces)


def rank(report,level):
    invalid=report['invalid_rate'];collision=report['collision_rate'] if level!=3 else 0.
    violation=report['grade_violation_rate']
    eligible=report['baseline_valid_rate']==1. and max(invalid,collision,violation)<=.125
    primary=report['success_rate' if level==0 else 'attack_success_rate'] if eligible else -invalid-collision-violation
    secondary=report['utility']+(.05*report['cut_count'] if level==2 else 0.)
    return (int(eligible),primary,secondary)


def train(cfg,output,resume=None,max_rounds=None):
    cfg.validate()
    if max_rounds is not None and max_rounds<1:raise ValueError('positive max_rounds required')
    with run_lock(Path(output).resolve()):return _train(cfg,Path(output).resolve(),resume,max_rounds)


def _train(cfg,output,resume,max_rounds):
    torch.set_num_threads(cfg.threads);torch.manual_seed(cfg.seed)
    initial=HybridActorCritic(cfg.hidden,cfg.state_dim)
    models={name:copy.deepcopy(initial) for name in ('A','B')}
    teachers={name:copy.deepcopy(model).eval().requires_grad_(False) for name,model in models.items()}
    optimizers={name:torch.optim.Adam(model.parameters(),lr=cfg.lr,eps=1e-5) for name,model in models.items()}
    rngs={name:np.random.default_rng(cfg.seed+10007*i) for i,name in enumerate(models)}
    torch_states={}
    for i,name in enumerate(models):torch.manual_seed(cfg.seed+10007*i);torch_states[name]=torch.get_rng_state()
    identity=source_digest();packages=runtime_versions();steps=round_id=validation_steps=0
    best=(-1,-float('inf'),0.);selection=None;best_model=None;inherited_wall=0.
    if resume:
        if Path(resume).resolve()!=output/'latest.pt':raise ValueError('Resume this run latest.pt only')
        state=load_checkpoint(resume)
        if state['config']!=cfg.to_dict() or state['source_sha256']!=identity or state['packages']!=packages:
            raise ValueError('Hybrid resume config/source/runtime mismatch')
        for name in models:
            models[name].load_state_dict(state['models'][name]);teachers[name].load_state_dict(state['teachers'][name])
            optimizers[name].load_state_dict(state['optimizers'][name]);rngs[name].bit_generator.state=state['numpy_rng'][name]
        torch_states=state['torch_rng'];steps=state['train_steps'];round_id=state['round']
        validation_steps=state['validation_steps'];best=tuple(state['best_rank'])
        selection=state['selection'];best_model=state['best_model'];inherited_wall=state['wall_seconds']
        for filename in ('metrics.jsonl','episodes.jsonl','validation.jsonl'):
            path=output/filename;committed=[]
            for line in path.read_text().splitlines() if path.exists() else []:
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if row['round']<=round_id:committed.append(row)
            path.write_text(''.join(json.dumps(row)+'\n' for row in committed),encoding='utf-8')
    else:
        output.mkdir(parents=True,exist_ok=False)
        (output/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
        (output/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,source_sha256=identity,packages=packages,
            initialization='same random neural weights for A/B and across levels for the same seed; native lateral controller',
            budget='A+B sampled decisions; validation counted separately'),indent=2),encoding='utf-8')
        examples=Path(__file__).resolve().parents[1]
        for path in source_files():
            target=output/'source'/path.relative_to(examples);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes())
    def selected_payload():
        return dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=identity,packages=packages,
                    model=best_model,selection=selection)
    if best_model is not None:save(output/'selected.pt',selected_payload())
    envs={name:HybridEnv(cfg) for name in models}
    writer=SummaryWriter(str(output/'tensorboard'),purge_step=steps if resume else None)
    writer.add_text('configuration',json.dumps(cfg.to_dict()),steps)
    started=time.perf_counter();start_round=round_id
    def log(prefix,values):
        for key,value in values.items():
            if isinstance(value,(int,float)) and np.isfinite(value):writer.add_scalar(prefix+'/'+key,value,steps)
    try:
        while steps<cfg.total_steps and (max_rounds is None or round_id-start_round<max_rounds):
            clock=steps;n=min(cfg.rollout_steps,(cfg.total_steps-steps)//2);round_id+=1
            beta_acceleration,beta_lane=beta_at(clock,cfg);round_rows=[]
            for name in models:
                torch.set_rng_state(torch_states[name]);sample_started=time.perf_counter()
                batch,episodes=collect(models[name],envs[name],n,rngs[name])
                stats=update(models[name],optimizers[name],batch,cfg,teachers['B' if name=='A' else 'A'],beta_acceleration,beta_lane)
                torch_states[name]=torch.get_rng_state()
                stats.update(mean_step_reward=float(batch['reward'].mean()),decisions_per_second=n/(time.perf_counter()-sample_started))
                row=dict(stats,round=round_id,team=name,train_steps=clock+2*n);round_rows.append(row)
                append(output/'metrics.jsonl',row)
                for episode in episodes:append(output/'episodes.jsonl',dict(episode,round=round_id,team=name,train_steps=clock+2*n))
                steps=clock+2*n;log('train/'+name,stats)
                if episodes:
                    log('episode/'+name,summarize(episodes))
                    log('cut_diagnostics/'+name,summarize(episodes)['causal_cut_diagnostics_total'])
                    for key in episodes[0]['reward_components']:
                        writer.add_scalar('reward/'+name+'/'+key,np.mean([e['reward_components'].get(key,0.) for e in episodes]),steps)
            if round_id%cfg.refresh_rounds==0:
                teachers={name:copy.deepcopy(model).eval().requires_grad_(False) for name,model in models.items()}
            if selection is None or steps==cfg.total_steps or steps//cfg.eval_interval!=clock//cfg.eval_interval:
                reports={name:evaluate_model(model,cfg,'selection',cfg.eval_cases) for name,model in models.items()}
                validation_steps+=sum(e['episode_steps'] for report in reports.values() for e in report['episodes'])
                append(output/'validation.jsonl',dict(round=round_id,train_steps=steps,reports=reports))
                for name,report in reports.items():log('validation/'+name,report['overall'])
                winner=max(reports,key=lambda name:rank(reports[name]['overall'],cfg.level))
                candidate=rank(reports[winner]['overall'],cfg.level)
                if candidate>=best:
                    best=candidate;best_model=copy.deepcopy(models[winner].state_dict())
                    selection=dict(team=winner,train_steps=steps,eligible=bool(best[0]),
                        attack_capable=bool(best[0] and reports[winner]['overall']['attack_success_rate']>0),
                        overall=reports[winner]['overall'])
                    save(output/'selected.pt',selected_payload())
            save(output/'latest.pt',dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=identity,packages=packages,
                models={name:model.state_dict() for name,model in models.items()},
                teachers={name:model.state_dict() for name,model in teachers.items()},
                optimizers={name:optimizer.state_dict() for name,optimizer in optimizers.items()},
                numpy_rng={name:rng.bit_generator.state for name,rng in rngs.items()},torch_rng=torch_states,
                train_steps=steps,round=round_id,validation_steps=validation_steps,best_rank=best,
                selection=selection,best_model=best_model,wall_seconds=inherited_wall+time.perf_counter()-started))
            writer.flush()
            print(f'hybrid L{cfg.level} {cfg.method} A+B={steps}/{cfg.total_steps} beta=({beta_acceleration:.4f},{beta_lane:.4f})',flush=True)
    finally:
        writer.close()
        for env in envs.values():env.close()
    summary=dict(status='complete' if steps==cfg.total_steps else 'paused',train_steps=steps,
                 validation_steps=validation_steps,selection=selection,wall_seconds=inherited_wall+time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def evaluate_suite(runs,output,split='development',cases=100,checkpoint='selected',team='A'):
    contract=case_contract(split,cases);runs=Path(runs);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if checkpoint not in ('selected','latest') or team not in ('A','B'):raise ValueError('Invalid checkpoint choice')
    paths=([runs] if runs.is_file() else [runs/(checkpoint+'.pt')]
           if (runs/(checkpoint+'.pt')).is_file() else sorted(runs.glob('*/'+checkpoint+'.pt')))
    if not paths:raise FileNotFoundError('No hybrid checkpoints')
    if len({p.parent.name for p in paths})!=len(paths):raise ValueError('Distinct run names required')
    identity=source_digest();packages=runtime_versions();frozen=[]
    for path in paths:
        payload=path.read_bytes();state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
        if state.get('protocol')!=PROTOCOL:raise ValueError('Not a hybrid-action checkpoint')
        if state['source_sha256']!=identity or state['packages']!=packages:raise ValueError('Hybrid evaluation source/runtime mismatch')
        frozen.append((path,payload,state))
    output.mkdir(parents=True);(output/'cases.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    summaries=[];torch.set_num_threads(1)
    for path,payload,state in frozen:
        cfg=HybridConfig.from_dict(state['config']);name=path.parent.name
        selected_team=team if 'models' in state else state['selection']['team']
        weights=state['models'][team] if 'models' in state else state['model']
        steps=state['train_steps'] if 'models' in state else state['selection']['train_steps']
        model=HybridActorCritic(cfg.hidden,cfg.state_dim);model.load_state_dict(weights)
        directory=output/name;directory.mkdir();(directory/'checkpoint.pt').write_bytes(payload)
        report=evaluate_model(model,cfg,split,cases,capture=True)
        for filename,rows in (('episodes.jsonl',report['episodes']),('trajectories.jsonl',report['trajectories'])):
            (directory/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        with (directory/'episodes.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['case_seed','family','attack_success','collision','invalid','grade_violation','cut_count','blocking_time','time','return']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(report['episodes'])
        represented=set()
        for trajectory in report['trajectories']:
            key=(trajectory['family'],trajectory['success'])
            if key not in represented:
                represented.add(key)
                plot_trace(trajectory['steps'],directory/f'{key[0]}_{"success" if key[1] else "unsuccessful"}_{trajectory["case_seed"]}.png',name+' / '+str(key))
        result=dict(run=name,protocol=PROTOCOL,split=split,team=selected_team,checkpoint_train_steps=steps,
            config=dict(cfg.to_dict(),state_dim=cfg.state_dim),source_sha256=identity,packages=packages,
            checkpoint_sha256=hashlib.sha256(payload).hexdigest(),overall=report['overall'],families=report['families'])
        (directory/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');summaries.append(result)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    lines=['# 混合动作独立评估','',f'固定 {split} 场景，每模型 {cases} 回合。','',
           '| 模型 | 成功率 | follow | adjacent | 碰撞率 | 无效率 | 平均有效切入 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        m=row['overall'];f=row['families']
        lines.append(f'| {row["run"]} | {m["attack_success_rate"]:.1%} | {f["follow"]["attack_success_rate"]:.1%} | '
            f'{f["adjacent"]["attack_success_rate"]:.1%} | {m["collision_rate"]:.1%} | {m["invalid_rate"]:.1%} | {m["cut_count"]:.3f} |')
    lines+=['','低层控制器执行车道跟踪，网络决定加速度和换道；该协议与旧连续转向协议分别统计。',
            '碰撞对 L1/L2 是失败，对 L3 是任务目标；过程指标与最终成功率分别解释。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summaries
