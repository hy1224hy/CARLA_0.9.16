"""Independent same-task twin training; no communication or quality gates."""
import copy
import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .model import ActorCritic
from .learning import collect,update,beta_at
from ..training.checkpoint import source_digest,source_files,runtime_versions,save
from ..training.locking import run_lock

PROTOCOL='straight_twin_v1'
OFFSETS={'train':0,'gate':1000000,'selection':2000000,'test':3000000}


def load_state(path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('protocol')!=PROTOCOL:raise ValueError('Not a straight twin checkpoint')
    return state


def model_from_state(state):
    from .sac import SACActorCritic
    raw=state['config']
    cls=SACActorCritic if state.get('learner')=='sac' else ActorCritic
    model=cls(raw['hidden'],raw.get('obs_dim',24))
    model.load_state_dict(state['model'])
    return model


def configuration(raw):
    from .config import StraightConfig
    return StraightConfig(**raw).validate()


def append(path,row):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


def rank_report(report,level):
    valid=report.get('baseline_valid_rate',1.)>=1.
    invalid=report.get('invalid_rate',0.)
    collision=report.get('collision_rate',0.) if level!=3 else 0.
    violation=report.get('grade_violation_rate',0.)
    eligible=valid and invalid<=.125 and collision<=.125 and violation<=.125
    primary=report['success_rate' if level==0 else 'attack_success_rate'] if eligible else -invalid-collision-violation
    secondary=report.get('utility',0.)
    if level==2:secondary=secondary+0.05*report.get('cut_count',0.)
    return (int(eligible),primary,secondary)


def evaluate_model(model,cfg,split,cases,demonstration=False,capture=False):
    from .environment import StraightEnv
    if split not in OFFSETS or cases<1:raise ValueError('Invalid split/cases')
    env=StraightEnv(cfg);rows=[];frames=[];trajectory=[];representatives={'first':None,'success':None,'failure':None}
    try:
        with torch.no_grad():
            for i in range(cases):
                obs,_=env.reset(seed=OFFSETS[split]+i,options={'family':('follow','adjacent')[i%2],'demonstration':demonstration})
                total=0.;initial_lane=env._lane(env.attacker);completed_change=False
                record=(capture and (i==0 or representatives['success'] is None or representatives['failure'] is None))
                if record:steps=[]
                while True:
                    action=model.distribution(torch.as_tensor(obs,dtype=torch.float32)).loc.tanh().numpy()
                    obs,r,term,trunc,info=env.step(action);total+=r
                    lane=env._lane(env.attacker)
                    completed_change|=lane!=initial_lane and abs(env.attacker.position[1]-4*lane)<.5 and abs(env.attacker.heading)<np.pi/18
                    if record:
                        steps.append(dict(time=info.get('time'),attacker_position=env.attacker.position.tolist(),
                            target_position=env.target.position.tolist(),attacker_speed=float(env.attacker.speed),target_speed=float(env.target.speed)))
                        if i==0:frames.append(env.render())
                    if term or trunc:break
                if record:
                    won=bool(info.get('attack_success',False)) or (cfg.level==0 and bool(info.get('success',False)))
                    failed=bool(info.get('collision',False) or info.get('invalid',False) or info.get('grade_violation',False))
                    entry=dict(case_seed=OFFSETS[split]+i,family=env.family,steps=steps)
                    if i==0:trajectory.append(entry);representatives['first']=entry['case_seed']
                    elif won and representatives['success'] is None:trajectory.append(entry);representatives['success']=entry['case_seed']
                    elif failed and representatives['failure'] is None:trajectory.append(entry);representatives['failure']=entry['case_seed']
                rows.append(dict(info,**{'return':total,'case_seed':OFFSETS[split]+i,'lane_change_completed':bool(completed_change and not info['invalid'] and not info['collision'])}))
    finally:env.close()
    overall={}
    rate_keys={'success','attack_success','collision','target_collision','offroad','invalid','grade_violation','baseline_valid','cut_in','stop_qualified'}
    for key in ('success','attack_success','collision','target_collision','offroad','invalid','grade_violation','baseline_valid','cut_in','stop_qualified','return','utility','red_delay_m','episode_steps','blocking_time','stop_time','excessive_loss_time','max_target_brake','avg_speed_ratio','cut_count','cut_first_thw'):
        values=[r[key] for r in rows if isinstance(r.get(key),(int,float,bool)) and np.isfinite(r[key])]
        if values:
            overall[key+('_rate' if key in rate_keys else '')]=float(np.mean(values))
    orientations=[r.get('contact_orientation') for r in rows]
    overall['contact_lateral_rate']=float(np.mean([o=='lateral' for o in orientations]))
    overall['contact_longitudinal_rate']=float(np.mean([o=='longitudinal' for o in orientations]))
    overall['lane_change_rate']=float(np.mean([r['lane_change_completed'] for r in rows]))
    result={'split':split,'cases':cases,'level':cfg.level,'overall':overall,'episodes':rows,'demonstration_scenarios':demonstration}
    if capture:result.update(_frames=frames,trajectory=trajectory,representatives=representatives)
    return result


def evaluate(checkpoint,split='test',cases=None,output=None,allow_source_change=False,**kwargs):
    if cases is None:cases=100 if split=='test' else 32
    if cases<1:raise ValueError('cases must be positive')
    capture=bool(kwargs.get('gif',False))
    if capture and output is None:raise ValueError('GIF requires explicit output directory')
    state=load_state(checkpoint)
    changed=state['source_sha256']!=source_digest()
    if changed and not allow_source_change:raise ValueError('Source changed; explicitly audit before evaluation')
    cfg=configuration(state['config']);torch.set_num_threads(cfg.threads)
    model=model_from_state(state)
    result=evaluate_model(model,cfg,split,cases,capture=capture)
    frames=result.pop('_frames',[])
    sibling=Path(checkpoint).parent/'latest.pt'
    if sibling.is_file():
        try:train_steps_total=load_state(sibling).get('train_steps')
        except ValueError:train_steps_total=None
    else:train_steps_total=None
    if train_steps_total is None:train_steps_total=(state.get('selection') or {}).get('train_steps')
    result.update(config=cfg.to_dict(),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),source_changed=changed,
        source_sha256=state['source_sha256'],evaluation_source_sha256=source_digest(),packages=runtime_versions(),
        train_steps_total=train_steps_total)
    if output:
        path=Path(output)
        if path.suffix!='.json':path=path/'evaluation.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        trajectory_data=result.pop('trajectory',[]) if capture else None
        with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
        if capture:
            (path.parent/'trajectory.json').write_text(json.dumps(dict(cases=trajectory_data,
                representatives=result.get('representatives',{})),indent=2),encoding='utf-8')
        if frames:
            from PIL import Image
            images=[Image.fromarray(frame) for frame in frames]
            images[0].save(path.with_suffix('.gif'),save_all=True,append_images=images[1:],duration=200,loop=0)
    return result


def train(config,output,initialize_from=None,resume=None,max_rounds=None):
    with run_lock(Path(output).resolve()):
        if config.learner=='sac':
            from .sac import train_sac
            return train_sac(config,output,initialize_from,resume,max_rounds)
        return _train(config,output,initialize_from,resume,max_rounds)


def _train(config,output,initialize_from=None,resume=None,max_rounds=None):
    from .environment import StraightEnv
    cfg=config.validate();output=Path(output).resolve()
    if max_rounds is not None and max_rounds<1:raise ValueError('max_rounds must be positive')
    if resume and initialize_from:raise ValueError('Choose resume or initialization')
    torch.set_num_threads(cfg.threads)
    models={};optimizers={};rngs={};torch_states={}
    for j,name in enumerate(('A','B')):
        torch.manual_seed(cfg.seed+10007*j)
        models[name]=ActorCritic(cfg.hidden,cfg.obs_dim)
        optimizers[name]=torch.optim.Adam(models[name].parameters(),lr=cfg.lr,eps=1e-5)
        rngs[name]=np.random.default_rng(cfg.seed+10007*j)
        torch_states[name]=torch.get_rng_state()
    teachers={k:copy.deepcopy(v).eval().requires_grad_(False) for k,v in models.items()}
    steps=round_id=validation_steps=0;best=(-1,-float('inf'),0.);selection=None;best_model=None
    identity=source_digest();versions=runtime_versions()
    if resume:
        state=load_state(resume)
        if Path(resume).resolve()!=output/'latest.pt':raise ValueError('Resume this run latest.pt only')
        if state['config']!=cfg.to_dict() or state['source_sha256']!=identity or state['packages']!=versions:raise ValueError('Resume config/source/runtime mismatch')
        for k in models:
            models[k].load_state_dict(state['models'][k]);optimizers[k].load_state_dict(state['optimizers'][k])
            teachers[k].load_state_dict(state['teachers'][k]);rngs[k].bit_generator.state=state['numpy_rng'][k];torch_states[k]=state['torch_rng'][k]
        steps=state['train_steps'];round_id=state['round'];validation_steps=state['validation_steps'];best=tuple(state['best_rank']);selection=state['selection'];best_model=state['best_model']
        if best_model is not None:
            save(output/'selected.pt',dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=identity,packages=versions,model=best_model,selection=selection))
        for filename in ('metrics.jsonl','episodes.jsonl','validation.jsonl'):
            path=output/filename
            if path.exists():
                committed=[]
                lines=path.read_text(encoding='utf-8').splitlines()
                for index,line in enumerate(lines):
                    try:row=json.loads(line)
                    except json.JSONDecodeError:
                        if index!=len(lines)-1:raise
                        break
                    if row['round']<=round_id:committed.append(row)
                path.write_text(''.join(json.dumps(r)+'\n' for r in committed),encoding='utf-8')
    else:
        output.mkdir(parents=True,exist_ok=False)
        if initialize_from:
            initial=load_state(initialize_from)
            compatible=((initial['config']['level']==0 and initial.get('attack_level') is None)
                        or initial.get('attack_level')==cfg.level)
            if not compatible or initial['config']['hidden']!=cfg.hidden:raise ValueError('Initialization must be a compatible normal driver or matching-level attack driver')
            if not initial.get('qualified') or initial['source_sha256']!=identity or initial['packages']!=versions:raise ValueError('Driver qualification/source/runtime mismatch')
            for key in ('duration','policy_frequency','simulation_frequency','target_speed','target_time_gap'):
                if initial['config'][key]!=cfg.to_dict()[key]:raise ValueError('Driver physical contract mismatch: '+key)
            for model in models.values():model.actor.load_state_dict(initial['actor'])
            teachers={k:copy.deepcopy(v).eval().requires_grad_(False) for k,v in models.items()}
        (output/'config.json').write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
        (output/'manifest.json').write_text(json.dumps(dict(protocol=PROTOCOL,source_sha256=identity,packages=versions,initialization={'path':str(initialize_from),'sha256':hashlib.sha256(Path(initialize_from).read_bytes()).hexdigest()} if initialize_from else None,budget='A+B environment decisions'),indent=2),encoding='utf-8')
        source=output/'source'
        source.mkdir()
        for file in source_files():
            snapshot=source/file.relative_to(Path(__file__).resolve().parents[1])
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes(file.read_bytes())
    writer=SummaryWriter(str(output/'tensorboard'),purge_step=0)
    writer.add_text('configuration',json.dumps(cfg.to_dict(),indent=2),0)
    def log(row,prefix):
        allowed=({'policy_loss','value_loss','entropy','mutual_kl','mutual_loss','approx_kl','clip_fraction','gradient_norm','effective_beta','ratio_error','acceleration_std','steering_std','mean_step_reward','decisions_per_second'} if prefix.startswith('train/') else {'success_rate','attack_success_rate','collision_rate','target_collision_rate','offroad_rate','invalid_rate','grade_violation_rate','baseline_valid_rate','utility','red_delay_m','return','max_target_brake','avg_speed_ratio','cut_count','cut_first_thw','stop_qualified_rate','contact_lateral_rate','contact_longitudinal_rate','blocking_time','stop_time','excessive_loss_time'})
        for key,value in row.items():
            if key in allowed and isinstance(value,(int,float,bool)) and np.isfinite(value):writer.add_scalar(prefix+'/'+key,value,row['train_steps'])
    def log_episodes(episodes,name,step):
        for key in ('return','attack_success','collision','offroad','grade_violation','blocking_time','cut_count','stop_time'):
            values=[e[key] for e in episodes if key in e]
            if values:writer.add_scalar('rollout/'+name+'/'+key,float(np.mean(values)),step)
        for key in sorted({k for e in episodes for k in e.get('reward_components',{})}):
            values=[e['reward_components'][key] for e in episodes if key in e.get('reward_components',{})]
            if values:writer.add_scalar('reward/'+name+'/'+key,float(np.mean(values)),step)
    if resume:
        for filename,prefix in (('metrics.jsonl','train'),('validation.jsonl','validation')):
            path=output/filename
            if path.exists():
                for line in path.read_text().splitlines():
                    row=json.loads(line)
                    if prefix=='train':log(row,'train/'+row['team'])
                    else:
                        for name,report in row['reports'].items():log(dict(report['overall'],train_steps=row['train_steps']),'validation/'+name)
                        if 'validation_steps' in row:writer.add_scalar('budget/validation_steps',row['validation_steps'],row['train_steps'])
        path=output/'episodes.jsonl'
        if path.exists():
            groups={}
            for line in path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line);groups.setdefault((row['round'],row['team'],row['train_steps']),[]).append(row)
            for (_,name,step),episodes in groups.items():log_episodes(episodes,name,step)
    envs={k:StraightEnv(cfg) for k in models};start_round=round_id
    started_at=time.perf_counter();inherited_wall=state.get('wall_seconds',0.) if resume else 0.
    try:
        while steps<cfg.total_steps and (max_rounds is None or round_id-start_round<max_rounds):
            n=min(cfg.rollout_steps,(cfg.total_steps-steps)//2)
            clock=steps;round_id+=1;rows=[]
            beta=beta_at(clock,cfg.beta,cfg.warmup_steps,cfg.ramp_steps) if cfg.method=='mutual' else 0.
            for name in models:
                torch.set_rng_state(torch_states[name]);started=time.perf_counter()
                batch,episodes=collect(models[name],envs[name],n,rngs[name])
                stats=update(models[name],optimizers[name],batch,cfg,teachers['B' if name=='A' else 'A'],beta)
                torch_states[name]=torch.get_rng_state();steps+=n
                row=dict(stats,team=name,round=round_id,train_steps=clock+2*n,mean_step_reward=float(batch['reward'].mean()),decisions_per_second=n/(time.perf_counter()-started))
                rows.append(row);append(output/'metrics.jsonl',row);log(row,'train/'+name)
                for episode in episodes:append(output/'episodes.jsonl',dict(episode,team=name,round=round_id,train_steps=clock+2*n))
                log_episodes(episodes,name,clock+2*n)
            if round_id%cfg.refresh_rounds==0:teachers={k:copy.deepcopy(v).eval().requires_grad_(False) for k,v in models.items()}
            if steps==cfg.total_steps or steps//cfg.eval_interval!=clock//cfg.eval_interval or selection is None:
                reports={k:evaluate_model(v,cfg,'selection',cfg.eval_cases) for k,v in models.items()}
                validation_steps+=sum(e['episode_steps'] for report in reports.values() for e in report['episodes'])
                append(output/'validation.jsonl',dict(round=round_id,train_steps=steps,validation_steps=validation_steps,reports=reports))
                for k,report in reports.items():log(dict(report['overall'],train_steps=steps),'validation/'+k)
                winner=max(reports,key=lambda k:rank_report(reports[k]['overall'],cfg.level))
                rank=rank_report(reports[winner]['overall'],cfg.level)
                if rank>best:
                    best=rank;selection=dict(team=winner,train_steps=steps,eligible=bool(rank[0]),overall=reports[winner]['overall']);best_model=copy.deepcopy(models[winner].state_dict())
                    save(output/'selected.pt',dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=identity,packages=versions,model=best_model,selection=selection))
            writer.add_scalar('budget/validation_steps',validation_steps,steps);writer.flush()
            save(output/'latest.pt',dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=identity,packages=versions,
                models={k:v.state_dict() for k,v in models.items()},optimizers={k:v.state_dict() for k,v in optimizers.items()},teachers={k:v.state_dict() for k,v in teachers.items()},numpy_rng={k:v.bit_generator.state for k,v in rngs.items()},torch_rng=torch_states,train_steps=steps,round=round_id,validation_steps=validation_steps,best_rank=best,selection=selection,best_model=best_model,wall_seconds=inherited_wall+(time.perf_counter()-started_at)))
            print(f'level={cfg.level} {cfg.method} A+B={steps}/{cfg.total_steps} beta={beta:.5f}',flush=True)
    finally:
        writer.close()
        for env in envs.values():env.close()
    wall_seconds=inherited_wall+(time.perf_counter()-started_at)
    summary=dict(status='complete' if steps==cfg.total_steps else 'paused',train_steps=steps,validation_steps=validation_steps,selection=selection,wall_seconds=wall_seconds)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def bootstrap(config,output,episodes=128,epochs=40,dagger_rounds=2,dagger_episodes=16,attack_level=None):
    from .environment import StraightEnv
    if episodes<1 or epochs<1:raise ValueError('Positive BC budgets required')
    cfg=replace(config,level=0).validate();torch.set_num_threads(cfg.threads);torch.manual_seed(cfg.seed)
    output=Path(output)
    if attack_level is None:output.mkdir(parents=True,exist_ok=False)
    elif not output.is_dir():raise ValueError('attack bootstrap requires an existing driver output directory')
    env=StraightEnv(cfg);observations=[];actions=[];lane_changes=0
    def scenario(i):
        demonstration=i%5==0
        family='follow' if demonstration else ('follow','adjacent')[i%2]
        recovery=(not demonstration) and (i%4==3)
        return family,demonstration,recovery
    def fit(model,x,y,n_epochs):
        optimizer=torch.optim.Adam(model.actor.parameters(),lr=1e-3)
        for _ in range(n_epochs):
            for idx in torch.randperm(len(x)).split(256):
                loss=((model.actor(x[idx]).tanh()-y[idx]).square()*torch.tensor([1.,10.])).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
    try:
        rng=np.random.default_rng(cfg.seed)
        for i in range(episodes):
            family,demonstration,recovery=scenario(i)
            obs,_=env.reset(seed=int(rng.integers(900000)),options={'family':family,'demonstration':demonstration,'recovery':recovery})
            previous_lane=env._lane(env.attacker)
            while True:
                action=env.normal_action()
                observations.append(obs);actions.append(action)
                obs,_,term,trunc,_=env.step(action)
                current_lane=env._lane(env.attacker)
                if current_lane!=previous_lane:lane_changes+=1
                previous_lane=current_lane
                if term or trunc:break
    finally:env.close()
    x=torch.as_tensor(np.asarray(observations),dtype=torch.float32);y=torch.as_tensor(np.asarray(actions),dtype=torch.float32)
    model=ActorCritic(cfg.hidden,cfg.obs_dim)
    fit(model,x,y,epochs)
    # DAgger: relabel the model's own rollouts with the teacher's corrective actions,
    # so the controller surface is learned in exactly the states the model visits.
    env=StraightEnv(cfg)
    try:
        rng=np.random.default_rng(cfg.seed+1)
        for round_ in range(dagger_rounds):
            new_obs=[];new_act=[]
            for i in range(dagger_episodes):
                family,demonstration,recovery=scenario(round_*dagger_episodes+i+episodes)
                obs,_=env.reset(seed=int(rng.integers(900000)),options={'family':family,'demonstration':demonstration,'recovery':recovery})
                while True:
                    with torch.no_grad():
                        action=model.distribution(torch.as_tensor(obs,dtype=torch.float32)).loc.tanh().numpy()
                    new_obs.append(obs);new_act.append(env.normal_action())
                    obs,_,term,trunc,_=env.step(action)
                    if term or trunc:break
            if not new_obs:break
            x=torch.cat([x,torch.as_tensor(np.asarray(new_obs),dtype=torch.float32)])
            y=torch.cat([y,torch.as_tensor(np.asarray(new_act),dtype=torch.float32)])
            fit(model,x,y,max(4,epochs//4))
    finally:env.close()
    np.savez_compressed(output/'demonstrations.npz',obs=x.numpy(),actions=y.numpy())
    qualification=evaluate_model(model,cfg,'selection',cfg.eval_cases)
    lane_qualification=evaluate_model(model,cfg,'selection',cfg.eval_cases,demonstration=True)
    qualification['lane_change_validation']=lane_qualification
    qualified=qualification['overall']['success_rate']>=.9 and qualification['overall']['offroad_rate']<=.05 and lane_changes>0 and lane_qualification['overall']['lane_change_rate']>=.5 and lane_qualification['overall']['success_rate']>=.9
    state=dict(protocol=PROTOCOL,config=cfg.to_dict(),source_sha256=source_digest(),packages=runtime_versions(),model=model.state_dict(),actor=model.actor.state_dict(),qualified=qualified,demonstration_steps=len(x),demonstration_lane_changes=lane_changes,qualification=qualification)
    save(output/('driver.pt' if qualified else 'diagnostic_driver.pt'),state)
    (output/'qualification.json').write_text(json.dumps(qualification,indent=2),encoding='utf-8')
    if attack_level is None:
        return dict(qualified=qualified,demonstration_steps=len(x),demonstration_lane_changes=lane_changes,checkpoint=str(output/('driver.pt' if qualified else 'diagnostic_driver.pt')))
    # Second stage: fine-tune the qualified driver on scripted attack demonstrations
    # (doc-sanctioned co-driving initialization; level-0 termination semantics so
    # script demos run their full course; L1/L2 discard colliding/invalid demos).
    from .environment import scripted_action
    if not qualified:raise ValueError('Attack initialization requires a qualified driver.pt first')
    if attack_level not in (1,2,3):raise ValueError('attack_level must be 1/2/3')
    script={1:'block',2:'cut',3:'collision'}[attack_level]
    env=StraightEnv(cfg)
    script_obs=[];script_act=[];kept=0
    try:
        rng=np.random.default_rng(cfg.seed+7)
        for i in range(max(32,episodes)):
            family='adjacent' if attack_level==2 else ('follow','adjacent')[i%2]
            obs,_=env.reset(seed=int(rng.integers(900000)),options={'family':family})
            buffer_obs=[];buffer_act=[];clean=True
            while True:
                action=scripted_action(env,script)
                buffer_obs.append(obs);buffer_act.append(action)
                obs,_,term,trunc,info=env.step(action)
                if info.get('invalid'):clean=False
                if attack_level!=3 and info.get('collision'):clean=False
                if term or trunc:break
            if clean:
                script_obs.extend(buffer_obs);script_act.extend(buffer_act);kept+=1
    finally:env.close()
    if not script_obs:raise ValueError('no clean script demonstrations collected')
    sx=torch.as_tensor(np.asarray(script_obs),dtype=torch.float32);sy=torch.as_tensor(np.asarray(script_act),dtype=torch.float32)
    attack=ActorCritic(cfg.hidden,cfg.obs_dim);attack.actor.load_state_dict(model.actor.state_dict())
    fit(attack,sx,sy,max(12,epochs//2))
    if dagger_rounds:
        # DAgger on the attack policy: relabel the model's own staging rollouts with
        # script actions so the attack maneuver is learned in the states it visits.
        env2=StraightEnv(cfg)
        try:
            rng=np.random.default_rng(cfg.seed+17)
            for round_ in range(dagger_rounds):
                new_obs=[];new_act=[]
                for i in range(16):
                    family='adjacent' if attack_level==2 else ('follow','adjacent')[i%2]
                    obs,_=env2.reset(seed=int(rng.integers(900000)),options={'family':family})
                    buffer_obs=[];buffer_act=[];clean=True
                    while True:
                        with torch.no_grad():
                            action=attack.distribution(torch.as_tensor(obs,dtype=torch.float32)).loc.tanh().numpy()
                        buffer_obs.append(obs);buffer_act.append(scripted_action(env2,script))
                        obs,_,term,trunc,info=env2.step(action)
                        if info.get('invalid'):clean=False
                        if attack_level!=3 and info.get('collision'):clean=False
                        if term or trunc:break
                    if clean:
                        new_obs.extend(buffer_obs);new_act.extend(buffer_act)
                if not new_obs:break
                sx=torch.cat([sx,torch.as_tensor(np.asarray(new_obs),dtype=torch.float32)])
                sy=torch.cat([sy,torch.as_tensor(np.asarray(new_act),dtype=torch.float32)])
                fit(attack,sx,sy,max(6,epochs//6))
        finally:env2.close()
    attack_cfg=replace(config,level=attack_level)
    attack_qualification=evaluate_model(attack,attack_cfg,'selection',cfg.eval_cases)
    overall=attack_qualification['overall']
    attack_qualified={1:(overall.get('blocking_time',0.)>=.5 and overall.get('collision_rate',1.)<=.3),
                      2:(overall.get('cut_in_rate',0.)>0 and overall.get('collision_rate',1.)<=.3),
                      3:(overall.get('collision_rate',0.)>=.5)}[attack_level]
    state=dict(protocol=PROTOCOL,config=cfg.to_dict(),attack_level=attack_level,source_sha256=source_digest(),packages=runtime_versions(),model=attack.state_dict(),actor=attack.actor.state_dict(),qualified=attack_qualified,demonstration_steps=len(x)+len(sx),script_episodes_kept=kept,qualification=attack_qualification)
    save(output/f'attack_L{attack_level}.pt',state)
    return dict(qualified=attack_qualified,attack_level=attack_level,checkpoint=str(output/f'attack_L{attack_level}.pt'),script_episodes_kept=kept,qualification=overall)


def report(runs,output):
    from .reporting import report as aggregate
    return aggregate(runs,output)
