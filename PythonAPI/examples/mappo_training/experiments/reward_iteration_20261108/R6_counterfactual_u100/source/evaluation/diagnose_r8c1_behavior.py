"""Read-only policy diagnostics, separate from frozen C0 and training contracts."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import traceback

import numpy as np
import torch

from models.actor_critic import Actor
from training.action_protocol import action_tanh_to_env
from training.checkpoint_protocol import load_checkpoint_file, validate_checkpoint_header, sha256_file, atomic_write_json, utc_now_iso
from evaluation.r8c1_diagnostic_protocol import actor_state_hashes
from evaluation.evaluate_base_policy_c0 import _preflight_idle_world, _cleanup_environment, _restore_failed_initialization, _set_seed
from environment.carla_multi_agent_env import CarlaMultiAgentEnv

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224'
CONFIG=ROOT/'configs/evaluation/r8c1_behavior_u208_v1.json'


def native(x):
    if isinstance(x,dict): return {str(k):native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [native(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    return x


def make_actors(checkpoint):
    cfg=checkpoint['training_config']
    result={}
    for aid in (0,1):
        actor=Actor(obs_dim=20,act_dim=2,hidden=checkpoint['model_spec']['hidden_dim'],
                    init_log_std=cfg['init_log_std'],min_log_std=cfg['min_log_std'],max_log_std=cfg['max_log_std'])
        actor.load_state_dict(checkpoint['actors'][aid],strict=True)
        actor.eval().requires_grad_(False)
        result[aid]=actor
    return result


def variants(short):
    a=copy.deepcopy(short)
    a['goal_zone']['y']+=30
    for goal in a['agent_goal_zones'].values(): goal['y']+=30
    b=copy.deepcopy(a); b['episode_max_steps']=500
    c=copy.deepcopy(b); c['agents']['agent1']['initial_speed_mps']=4.0
    return {'north_length45':a,'north_length45_horizon500':b,'north_length45_horizon500_a1speed4':c}


def prepare():
    if CONFIG.exists(): raise FileExistsError('diagnostic config already frozen')
    extension_path=ROOT/'configs/base_mappo/p0_r8c1_continue_u600_v1.json'
    extension=json.loads(extension_path.read_text())
    source=dict(extension['source_sha256'])
    for name in ['evaluation/diagnose_r8c1_behavior.py','evaluation/evaluate_base_policy_c0.py','evaluation/base_c0_protocol.py','evaluation/r8c1_diagnostic_protocol.py']:
        source[name]=sha256_file(ROOT/name)
    for name,digest in extension['source_sha256'].items():
        if sha256_file(ROOT/name)!=digest: raise ValueError(f'training source drift: {name}')
    checkpoints={}
    manifest=json.loads((RUN/'checkpoint_manifest.json').read_text())
    for update in (100,185,208):
        path=RUN/('latest.pt' if update==208 else f'periodic/ckpt_update_{update:06d}.pt')
        c,h=load_checkpoint_file(path,map_location='cpu'); validate_checkpoint_header(c)
        assert c['update']==update and c['model_spec']['obs_dim']==20 and c['model_spec']['action_dim']==2
        assert c['training_config']['reward_version']=='v6'
        assert c['run_metadata']['experiment_contract_sha256']==extension['parent_contract_sha256']
        records=[manifest['latest'],*manifest['periodic']]
        assert any(r['update']==update and r['sha256']==h for r in records)
        if update>100: assert c['run_metadata']['continuation']['sha256']==sha256_file(extension_path)
        checkpoints[str(update)]={'path':path.relative_to(ROOT).as_posix(),'sha256':h,
            'source_metadata':c['run_metadata'],'actor_hashes':actor_state_hashes(make_actors(c))}
    scenarios={}
    for n in (1,4,5,6):
        path=next((ROOT/'data/training_scenarios_town05_r8a_v1').glob(f'scenario_{n:02d}_*.json'))
        scenarios[f'scene{n:02d}']={'config':json.loads(path.read_text()),'path':path.relative_to(ROOT).as_posix(),'sha256':sha256_file(path)}
    for name,scene in variants(scenarios['scene01']['config']).items(): scenarios[name]={'config':scene,'derived_from':'scene01'}
    cases=[]
    for u in (100,185,208):
        for scene in ('scene01','scene04','scene05','scene06'):
            for mode in ('deterministic','stochastic'):
                for seed in (62001,62002): cases.append({'checkpoint':str(u),'scenario':scene,'mode':mode,'seed':seed})
    for scene in list(scenarios)[4:]:
        for seed in (62001,62002): cases.append({'checkpoint':'208','scenario':scene,'mode':'deterministic','seed':seed})
    for i,c in enumerate(cases): c['id']=f"case_{i+1:03d}_u{c['checkpoint']}_{c['scenario']}_{c['mode']}_s{c['seed']}"
    config={'version':'r8c1_behavior_v1','created_at':utc_now_iso(),'diagnostic_only':True,
        'formal_eligibility':False,'source_sha256':source,'checkpoints':checkpoints,'scenarios':scenarios,'cases':cases,
        'thresholds':{'low_speed_mps':0.2,'startup_speed_mps':0.5,'minimum_low_speed_run_steps':10},
        'environment':{'town':'Town05','num_agents':2,'straight_road_length':50,'timestep':0.1,'max_steps':500,
            'spawn_vehicles':False,'adversarial_agent_ids':[],'reward_version':'v6','reward_gamma':0.999,
            'lidar_range_m':20,'ttc_max_range_m':20,'disable_tensorboard':True,'follow_spectator':True}}
    atomic_write_json(config,CONFIG)
    print('FROZEN',len(cases),sha256_file(CONFIG))


def preflight(config):
    assert config['version']=='r8c1_behavior_v1' and len(config['cases'])==54
    for name,h in config['source_sha256'].items():
        if sha256_file(ROOT/name)!=h: raise ValueError(f'diagnostic source drift: {name}')
    loaded={}
    for u,record in config['checkpoints'].items():
        c,h=load_checkpoint_file(ROOT/record['path'],map_location='cpu'); validate_checkpoint_header(c)
        assert h==record['sha256'] and c['update']==int(u)
        actors=make_actors(c)
        assert actor_state_hashes(actors)==record['actor_hashes']
        loaded[u]=actors
    for scene in config['scenarios'].values():
        if 'path' in scene: assert sha256_file(ROOT/scene['path'])==scene['sha256']
    return loaded


def kinematics(env):
    result={}
    for aid,v in enumerate(env.vehicles):
        if v is None or not v.is_alive: result[str(aid)]=None; continue
        t=v.get_transform(); vel=v.get_velocity(); ctrl=v.get_control()
        result[str(aid)]={'x':t.location.x,'y':t.location.y,'z':t.location.z,'yaw':t.rotation.yaw,
            'vx':vel.x,'vy':vel.y,'speed':math.sqrt(vel.x**2+vel.y**2+vel.z**2),
            'control':[ctrl.throttle,ctrl.steer,ctrl.brake]}
    return result


def camera(env):
    import carla
    ps=[v.get_location() for v in env.vehicles if v is not None and v.is_alive]
    if ps:
        x=sum(p.x for p in ps)/len(ps); y=sum(p.y for p in ps)/len(ps)
        radius=max(math.hypot(p.x-x,p.y-y) for p in ps)
        env.world.get_spectator().set_transform(carla.Transform(carla.Location(x=x,y=y,z=max(22,radius*1.8+12)),carla.Rotation(pitch=-90)))


def run_case(env,actors,case,config,path):
    scene=copy.deepcopy(config['scenarios'][case['scenario']]['config'])
    _set_seed(case['seed'],env)
    env.apply_scenario_config(scene)
    obs,info=env.reset()
    assert env.max_steps==scene['episode_max_steps']
    before=actor_state_hashes(actors)
    totals={str(i):{'return':0.,'steps':0,'speed_sum':0.,'brake_steps':0,'low_steps':0,'first_motion_step':None,'longest_low_run':0,'low_run':0} for i in (0,1)}
    with path.open('x',encoding='utf-8') as out:
        def emit(row): out.write(json.dumps(native(row),allow_nan=False)+'\n'); out.flush()
        emit({'type':'reset','case':case,'frame':env.world.get_snapshot().frame,'observation':obs,'info':info,'kinematics':kinematics(env)})
        for step in range(1,env.max_steps+1):
            active={i:bool(info['active_mask'][i]) for i in (0,1)}
            pre=kinematics(env); policy={}; actions={}; means={}; stds={}
            with torch.no_grad():
                for i in (0,1):
                    if not active[i]: continue
                    x=np.asarray(obs[i],dtype=np.float32)
                    assert x.shape==(20,) and np.isfinite(x).all()
                    tensor=torch.as_tensor(x).unsqueeze(0)
                    mu,logstd=actors[i](tensor)
                    means[i]=torch.tanh(mu)[0].numpy(); stds[i]=logstd[0].numpy()
                    action=actors[i].get_deterministic_action(tensor) if case['mode']=='deterministic' else actors[i].sample_action_z(tensor)[0]
                    policy[i]=action[0].numpy(); actions[i]=action_tanh_to_env(policy[i])
                    assert np.isfinite(policy[i]).all()
            next_obs,rewards,dones,next_info=env.step(actions)
            post=kinematics(env)
            for i in (0,1):
                if not active[i]: continue
                a=totals[str(i)]; d=next_info['reward_diagnostics'][i]
                residual=float(rewards[i])-sum(float(v) for v in next_info['reward_details'][i].values())
                assert abs(residual)<=1e-6
                speed=float(d['speed_mps']); a['steps']+=1; a['speed_sum']+=speed; a['return']+=float(rewards[i])
                a['brake_steps']+=int(actions[i][2]>0)
                low=speed<config['thresholds']['low_speed_mps']; a['low_steps']+=int(low)
                a['low_run']=a['low_run']+1 if low else 0; a['longest_low_run']=max(a['longest_low_run'],a['low_run'])
                if a['first_motion_step'] is None and speed>=config['thresholds']['startup_speed_mps']: a['first_motion_step']=step
                a['final_completion']=d['route_completion']; a['final_diagnostics']=d
            emit({'type':'step','step':step,'frame':env.world.get_snapshot().frame,'pre':pre,'post':post,
                'observation':obs,'next_observation':next_obs,'policy_action':policy,'deterministic_action':means,
                'effective_log_std':stds,'environment_action':actions,'reward':rewards,'done':dones,'info':next_info})
            camera(env); obs,info=next_obs,next_info
            if not any(info['active_mask'].values()): break
        else: raise RuntimeError('environment failed to terminate within its horizon')
    after=actor_state_hashes(actors); assert before==after
    for a in totals.values():
        a['mean_speed']=a['speed_sum']/max(a['steps'],1); a['brake_rate']=a['brake_steps']/max(a['steps'],1)
        a['low_speed_rate']=a['low_steps']/max(a['steps'],1)
    return {'case':case,'steps':step,'reason':info['episode_end_reason'],'agents':native(totals),
        'actor_hashes_before':before,'actor_hashes_after':after,'trace_sha256':sha256_file(path)}


def run(output):
    config=json.loads(CONFIG.read_text()); actors=preflight(config)
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(config,output/'resolved_config.json')
    manifest={'status':'RUNNING','started_at':utc_now_iso(),'config_sha256':sha256_file(CONFIG),'completed_cases':0,'diagnostic_only':True,'formal_eligibility':False}
    atomic_write_json(manifest,output/'manifest.json')
    env=None; idle=None
    try:
        idle=_preflight_idle_world(host='127.0.0.1',port=2000,expected_version='0.9.16')
        manifest['preflight']=idle
        env=CarlaMultiAgentEnv(host='127.0.0.1',port=2000,seed=62001,**config['environment'])
        env._original_synchronous_mode=idle['synchronous_mode']; env._original_fixed_delta_seconds=idle['fixed_delta_seconds']
        (output/'traces').mkdir()
        with (output/'episodes.jsonl').open('x',encoding='utf-8') as episodes:
            for case in config['cases']:
                result=run_case(env,actors[case['checkpoint']],case,config,output/'traces'/f"{case['id']}.jsonl")
                episodes.write(json.dumps(result,allow_nan=False)+'\n'); episodes.flush()
                manifest['completed_cases']+=1; manifest['last_case']=case['id']
                atomic_write_json(manifest,output/'manifest.json')
                print('CASE_DONE',case['id'],result['reason'],result['steps'],flush=True)
        preflight(config)
        manifest['status']='COMPLETED'
    except BaseException:
        manifest['status']='FAILED'; manifest['error']=traceback.format_exc(); raise
    finally:
        if env is not None:
            audit,error=_cleanup_environment(env); manifest['cleanup']=audit
            if error: manifest['cleanup_error']=error; manifest['status']='FAILED'
        elif idle:
            audit,error=_restore_failed_initialization(host='127.0.0.1',port=2000,preflight=idle)
            manifest['initialization_cleanup']=audit; manifest['cleanup_error']=error
        manifest['ended_at']=utc_now_iso(); atomic_write_json(manifest,output/'manifest.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--preflight',action='store_true'); parser.add_argument('--output')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.prepare: prepare()
    elif args.preflight: preflight(json.loads(CONFIG.read_text())); print('OFFLINE_PREFLIGHT_PASS')
    elif args.output: run(Path(args.output).resolve())
    else: parser.error('choose --prepare, --preflight or --output')
