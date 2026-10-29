"""Explicit interruption recovery at u943; retain serial frozen stops1000 and1200."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
sys.path.insert(0,str(ROOT))
from utils import base_driving_session as base
from training.checkpoint_protocol import atomic_write_json,atomic_copy_checkpoint,sha256_file
from utils.mut_mappo_session import RunSessionOwner,CarlaOwner,inspect_files

CONTROL=ROOT/'diagnostics/base_driving_vnext_20261020/long_s42_u600_v1'
PLAN=CONTROL/'plan.json'
EXTENSION=CONTROL/'extension_u1200_recovery_v1.json'
SOURCE=CONTROL/'u000943_recovery_20261027_v1.pt'
SOURCE_SHA='85ba7641e6c5fff147e99545770b36cc7bb9f7fb189f8782530313c0e80cd630'
PREVIOUS_EXTENSION=CONTROL/'extension_u1000_v1.json'
PREVIOUS_EXTENSION_SHA='8ec4f02ba2c282d3ef9f737e0e13a662351c97303eb29fafc18098fc5cb050e4'
STOPS=[50,100,200,400,600,800,1000,1200]
COMPLETED=[50,100,200,400,600,800]
VERSION='base_driving_explicit_u1200_recovery_v1'

def remaining_actions(stops,safe_update,completed_screens):
    if (not stops or any(type(v) is not int or not 1<=v<=1200 for v in stops)
            or stops!=sorted(set(stops))):raise ValueError('Invalid explicit1200 stops')
    if type(safe_update) is not int or not 0<=safe_update<=stops[-1]:
        raise ValueError('Safe boundary outside authorized budget')
    if completed_screens!=sorted(set(completed_screens)) or any(type(v) is not int or v not in stops or v>safe_update for v in completed_screens):
        raise ValueError('Invalid completed frozen stages')
    if any(v<safe_update and v not in completed_screens for v in stops):
        raise ValueError('Missing earlier frozen milestone')
    actions=[];current=safe_update
    for stop in stops:
        if stop in completed_screens:continue
        if stop>current:actions.append(('train',current,stop));current=stop
        actions.append(('screen',stop,stop))
    return actions

def effective_record(require_extension=True):
    record=base.load_plan(PLAN)
    if record['stop_updates']!=[50,100,200,400,600] or record['total_budget_updates']!=600:
        raise ValueError('Original plan binding changed')
    if sha256_file(PREVIOUS_EXTENSION)!=PREVIOUS_EXTENSION_SHA:
        raise ValueError('Prior u1000 authorization changed')
    old=base.read(PREVIOUS_EXTENSION)
    if sha256_file(old['controller_source'])!=old['controller_source_sha256']:
        raise ValueError('Prior controller evidence changed')
    if require_extension:
        e=base.read(EXTENSION)
        if (e['version']!=VERSION or e['source_checkpoint_sha256']!=SOURCE_SHA
                or e['source_update']!=943 or e['total_budget_updates']!=1200 or e['total_budget_episodes']!=19200
                or e['original_plan_sha256']!=sha256_file(PLAN)
                or e['previous_extension_sha256']!=PREVIOUS_EXTENSION_SHA
                or e['controller_source_sha256']!=sha256_file(__file__)
                or e['initialization_sha256']!=sha256_file(Path(record['run_dir'])/'initialization.json')
                or sha256_file(SOURCE)!=SOURCE_SHA):raise ValueError('Explicit recovery authorization/source binding changed')
    return dict(record,stop_updates=STOPS,total_budget_updates=1200,total_budget_episodes=19200)

def checkpoint_summary(record):
    safe,checkpoint,digest=base.safe_boundary(record)
    def steps(opt):return sorted({int(v['step']) for v in opt['state'].values() if 'step' in v})
    return {'safe_update':safe,'sha256':digest,'episodes':checkpoint['total_episodes'],
            'actor_adam_steps':{str(k):steps(v) for k,v in checkpoint['actor_optims'].items()},
            'critic_adam_steps':steps(checkpoint['critic_optim']),
            'lr_schedule':checkpoint['lr_schedule'],'rng_components':list(checkpoint['rng_state']),
            'source_fingerprint':checkpoint['run_metadata']['code_fingerprint'],
            'base_driving_state_keys':list(checkpoint['base_driving_state']),
            'value_normalizer_update_count':checkpoint['last_metrics']['scalars']['critic/value_normalizer_update_count']}

def preflight(record):
    base._assert_worker_idle(record,PLAN)
    base.require_geometry(record)
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    with CarlaOwner(run_dir=record['run_dir'],purpose='explicit u943 interruption recovery preflight'):
        return _preflight_idle_world(host='127.0.0.1',port=2000,expected_version='0.9.16')

def prepare():
    if EXTENSION.exists():raise FileExistsError('Recovery already prepared; no implicit retry')
    record=effective_record(False);run=Path(record['run_dir'])
    with ExitStack() as owners:
        for d in [run,CONTROL]:owners.enter_context(RunSessionOwner(d))
        world=preflight(record)
        info=checkpoint_summary(record)
        if (info['safe_update']!=943 or info['sha256']!=SOURCE_SHA or info['episodes']!=15088
                or info['actor_adam_steps']!={'0':[4715],'1':[4715]} or info['critic_adam_steps']!=[4715]
                or info['value_normalizer_update_count']!=943
                or set(info['rng_components'])!={'python','numpy','torch_cpu','torch_cuda'}
                or info['lr_schedule']['decay_horizon']!=600):raise ValueError('Complete recovery source does not match u943')
        if base._completed_screens(record)!=COMPLETED:raise ValueError('Prior frozen evidence changed')
        previous=base.read(CONTROL/'status.json');launch=base.read(run/'session/launch.json')
        if previous['version']!='base_driving_explicit_u1000_extension_v1' or previous['target_update']!=1000:
            raise ValueError('Not the explicitly interrupted u1000 extension')
        if launch['launch_id']!='30c38e9c0dfa4934909b946d3192b1a5' or launch['actual_worker_pid']!=16824:
            raise ValueError('Interrupted worker identity changed')
        if SOURCE.exists():
            if sha256_file(SOURCE)!=SOURCE_SHA:raise ValueError('Recovery source collision')
        else:atomic_copy_checkpoint(run/'latest.pt',SOURCE)
        tails=[r for r in base.rows(run/'episodes.jsonl') if r['update']>943]
        archive=CONTROL/'interrupted_controller_u1000_before_u1200.json'
        if archive.exists():raise FileExistsError('Prior controller archive already exists')
        atomic_write_json(previous,archive)
        payload={'version':VERSION,'authorized_at':base.now(),
            'authorization':'Human reported accidental interruption and explicitly requested continuing the same run to u1200 on2026-10-27.',
            'original_plan':str(PLAN),'original_plan_sha256':sha256_file(PLAN),
            'previous_extension':str(PREVIOUS_EXTENSION),'previous_extension_sha256':PREVIOUS_EXTENSION_SHA,
            'source_checkpoint':str(SOURCE),'source_checkpoint_sha256':SOURCE_SHA,'source_update':943,'source_episodes':15088,
            'total_budget_updates':1200,'total_budget_episodes':19200,'remaining_updates':257,'remaining_episodes':4112,
            'new_frozen_stops':[1000,1200],'completed_stages':COMPLETED,
            'controller_source':str(Path(__file__).resolve()),'controller_source_sha256':sha256_file(__file__),
            'initialization_sha256':sha256_file(run/'initialization.json'),'complete_source_state':info,
            'original_controller_archive':str(archive),'original_status_sha256':sha256_file(CONTROL/'status.json'),
            'interrupted_launch_sha256':sha256_file(run/'session/launch.json'),
            'interrupted_exit_present':(run/'session/exit.json').exists(),'uncommitted_episode_count':len(tails),
            'live_preflight':world,'schedule_policy':'Preserve original cumulative600 LR/entropy clock and terminal values; no reset/stretch.',
            'tail_policy':'Archive raw interrupted session first. Native full-state recorder reconciles episodes to safe943, then redo944.',
            'automatic_restart':False}
        atomic_write_json(payload,EXTENSION)
        print(json.dumps({'status':'PREPARED_NOT_STARTED','extension':str(EXTENSION),'source':info,'preflight':world,'uncommitted_tail':len(tails)}),flush=True)

def snapshot(record):
    run=Path(record['run_dir']);data=inspect_files(run)
    data.update(protocol=VERSION,inspection_mode='FILES_ONLY',automatic_restart=False,
                authorized_extension=str(EXTENSION),current_control=base.read(CONTROL/'status.json'),
                safe_update=base.read(run/'checkpoint_manifest.json')['latest']['update'])
    atomic_write_json(data,run/'session/monitor_latest.json')
    with (run/'session/monitor_history.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(data,allow_nan=False)+'\n')

def supervise(record,start,stop):
    run=Path(record['run_dir']);session=run/'session';stamp=uuid.uuid4().hex
    if (session/'launch.json').exists() or (session/'exit.json').exists():raise ValueError('Prior session must be archived')
    log=session/f'training_{stamp}.log'
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker','--from-update',str(start),'--to-update',str(stop),'--launch-id',stamp]
    launch={'launch_id':stamp,'spawned_pid':None,'actual_worker_pid':None,'trainer_pid':None,'supervisor_pid':os.getpid(),
            'command':command,'log_path':str(log),'started_at':base.now(),'resume_update':start,'stop_update':stop,
            'status':'DISPATCHED_NOT_TRAINING_SUCCESS','automatic_restart':False,'extension_record':str(EXTENSION)}
    atomic_write_json(launch,session/'launch.json')
    camera=None;env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8');flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with log.open('x',encoding='utf-8') as out,(session/f'camera_{stamp}.log').open('x',encoding='utf-8') as camout:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=flags)
        launch['spawned_pid']=child.pid;atomic_write_json(launch,session/'launch.json');deadline=0
        while child.poll() is None:
            actual=base.read(session/'launch.json').get('actual_worker_pid')
            if actual and camera is None:
                camera=subprocess.Popen([sys.executable,'-B','-u','-m','utils.follow_reward_fork','--trainer-pid',str(actual),'--run-dir',str(run)],
                    cwd=ROOT,stdout=camout,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=flags)
            if time.monotonic()>=deadline:
                try:snapshot(record)
                except Exception as e:atomic_write_json({'at':base.now(),'error':f'{type(e).__name__}: {e}'},session/'monitor_error.json')
                deadline=time.monotonic()+1200
            time.sleep(2)
        if camera is not None:
            try:camera.wait(timeout=8)
            except subprocess.TimeoutExpired:pass
        actual=base.read(session/'launch.json').get('actual_worker_pid')
        atomic_write_json({'launch_id':stamp,'spawned_pid':child.pid,'trainer_pid':actual,'supervisor_pid':os.getpid(),
            'returncode':child.returncode,'ended_at':base.now(),'camera_pid':camera.pid if camera else None,
            'camera_returncode':camera.poll() if camera else None,'automatic_restart':False},session/'exit.json')
    if child.returncode or not actual or not base._matching_exit(run,stop=stop):raise RuntimeError('Training failed/missing matching actual worker exit; no automatic retry')
    base.safe_boundary(record,expected=stop)

def controller():
    record=effective_record();run=Path(record['run_dir']);authorization=base.read(EXTENSION)
    with ExitStack() as owners:
        for d in [run,CONTROL]:owners.enter_context(RunSessionOwner(d))
        preflight(record);base.safe_boundary(record,expected=943)
        if sha256_file(CONTROL/'status.json')!=authorization['original_status_sha256'] or sha256_file(run/'session/launch.json')!=authorization['interrupted_launch_sha256']:
            raise ValueError('Old session changed since explicit preparation')
        if (run/'session/STOP_REQUESTED').exists():raise ValueError('STOP_REQUESTED requires explicit investigation')
        completed=base._completed_screens(record)
        actions=remaining_actions(STOPS,943,completed)
        state={'version':VERSION,'status':'STARTING','pid':os.getpid(),'started_at':base.now(),'initial_safe_update':943,
               'total_budget_updates':1200,'total_budget_episodes':19200,'extension_sha256':sha256_file(EXTENSION),
               'completed_stages':completed,'automatic_restart':False,'explicit_interruption_recovery':True}
        atomic_write_json(state,CONTROL/'status.json')
        try:
            for action,start,stop in actions:
                effective_record()
                if (run/'session/STOP_REQUESTED').exists():raise ValueError('STOP_REQUESTED exists')
                state.update(status='TRAINING' if action=='train' else 'FROZEN_EVALUATION',target_update=stop,changed_at=base.now())
                atomic_write_json(state,CONTROL/'status.json')
                if action=='train':
                    base.safe_boundary(record,expected=start)
                    archive=base.archive_session(record,reason='interrupted' if start==943 else 'completed')
                    if start==943:
                        state['interrupted_session_archive']=archive;atomic_write_json(state,CONTROL/'status.json')
                    supervise(record,start,stop)
                else:
                    result=base.frozen_screen(record,stop)
                    state['completed_stages'].append(stop);state['last_stage_selection']=result['selection']
                    atomic_write_json(state,CONTROL/'status.json')
            state.update(status='COMPLETED',ended_at=base.now());atomic_write_json(state,CONTROL/'status.json');snapshot(record)
        except BaseException as e:
            state.update(status='FAILED_NO_RESTART',error=f'{type(e).__name__}: {e}',ended_at=base.now())
            atomic_write_json(state,CONTROL/'status.json');raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','prepare','run','worker'])
    parser.add_argument('--from-update',type=int);parser.add_argument('--to-update',type=int);parser.add_argument('--launch-id')
    args=parser.parse_args()
    if args.mode=='audit':print(json.dumps(checkpoint_summary(effective_record(False))),flush=True)
    elif args.mode=='prepare':prepare()
    elif args.mode=='run':controller()
    else:
        if (args.from_update,args.to_update) not in [(943,1000),(1000,1200)] or not args.launch_id:
            raise ValueError('Worker cannot skip authorized frozen stages')
        record=effective_record()
        with RunSessionOwner(Path(record['run_dir'])/'worker_guard'):
            base._worker_owned(PLAN,record,args.from_update,args.to_update,args.launch_id)

if __name__=='__main__':main()
