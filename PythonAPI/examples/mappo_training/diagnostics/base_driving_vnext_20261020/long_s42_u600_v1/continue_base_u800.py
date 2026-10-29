"""Explicit u600 -> u800 extension; preserve the completed original plan.

The trainer, reward, curriculum, source fingerprints and schedule are unchanged.
Only this serial orchestration adapter owns the newly authorized budget.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training')
sys.path.insert(0, str(ROOT))
from utils import base_driving_session as base
from training.checkpoint_protocol import atomic_write_json, sha256_file
from utils.mut_mappo_session import RunSessionOwner, inspect_files

CONTROL = ROOT / 'diagnostics/base_driving_vnext_20261020/long_s42_u600_v1'
PLAN = CONTROL / 'plan.json'
EXTENSION = CONTROL / 'extension_u800_v1.json'
SOURCE_SHA = '4fd6193e34cdf29ca111d537f99f94c7bae425d6b14279939b14aef6db525498'
VERSION = 'base_driving_explicit_u800_extension_v1'

def effective_record(require_extension=True):
    record = base.load_plan(PLAN)
    if record['total_budget_updates'] != 600 or record['stop_updates'] != [50,100,200,400,600]:
        raise ValueError('Original completed u600 plan changed')
    summary = base.read(CONTROL / 'u000600_summary.json')
    source = CONTROL / 'u000600.pt'
    if summary['status'] != 'COMPLETED' or not summary['normal_training_matching_exit_0']:
        raise ValueError('u600 must have a complete matching-exit frozen stage')
    if summary['checkpoint_sha256'] != SOURCE_SHA or sha256_file(source) != SOURCE_SHA:
        raise ValueError('Immutable u600 source mismatch')
    if require_extension:
        ext = base.read(EXTENSION)
        if (ext['version'] != VERSION or ext['original_plan_sha256'] != sha256_file(PLAN)
                or ext['source_checkpoint_sha256'] != SOURCE_SHA
                or ext['total_budget_updates'] != 800 or ext['total_budget_episodes'] != 12800
                or ext['controller_source_sha256'] != sha256_file(__file__)
                or ext['initialization_sha256'] != sha256_file(Path(record['run_dir'])/'initialization.json')):
            raise ValueError('Authorized extension binding changed')
    record = dict(record)
    record.update(stop_updates=[50,100,200,400,600,800], total_budget_updates=800, total_budget_episodes=12800)
    return record

def checkpoint_summary(record):
    safe, checkpoint, digest = base.safe_boundary(record)
    def steps(optimizer):
        return sorted({int(value['step']) for value in optimizer['state'].values() if 'step' in value})
    return {'safe_update':safe,'sha256':digest,'episodes':checkpoint['total_episodes'],
            'actor_adam_steps':{str(k):steps(v) for k,v in checkpoint['actor_optims'].items()},
            'critic_adam_steps':steps(checkpoint['critic_optim']),
            'lr_schedule':checkpoint['lr_schedule'],
            'rng_components':list(checkpoint['rng_state']),
            'base_driving_state_keys':list(checkpoint['base_driving_state']),
            'source_fingerprint':checkpoint['run_metadata']['code_fingerprint']}

def prepare():
    if EXTENSION.exists():
        raise FileExistsError('Extension already exists; inspect it explicitly')
    record = effective_record(False)
    state = base.read(CONTROL/'status.json')
    if state.get('status') != 'COMPLETED' or state.get('completed_stages') != [50,100,200,400,600]:
        raise ValueError('Original controller has not completed all five frozen nodes')
    info = checkpoint_summary(record)
    if info['safe_update'] != 600 or info['sha256'] != SOURCE_SHA or info['episodes'] != 9600:
        raise ValueError('Extension must start at the exact complete u600 boundary')
    if not base._matching_exit(Path(record['run_dir']),stop=600):
        raise ValueError('u600 launch UUID / actual worker exit mismatch')
    if base._completed_screens(record) != [50,100,200,400,600]:
        raise ValueError('Frozen evidence is incomplete')
    payload={'version':VERSION,'authorized_at':base.now(),
             'authorization':'Human explicitly requested continuing the same training from u600 to u800 on 2026-10-26.',
             'original_plan':str(PLAN),'original_plan_sha256':sha256_file(PLAN),
             'source_checkpoint':str(CONTROL/'u000600.pt'),'source_checkpoint_sha256':SOURCE_SHA,
             'source_update':600,'source_episodes':9600,'total_budget_updates':800,'total_budget_episodes':12800,
             'additional_updates':200,'additional_episodes':3200,'new_frozen_stops':[800],
             'schedule_policy':'Preserve cumulative LR/entropy clock and existing 600 horizon; retain terminal schedule values after u600.',
             'automatic_restart':False,'original_status_archive':str(CONTROL/'completed_controller_u600_before_u800.json'),
             'initialization_sha256':sha256_file(Path(record['run_dir'])/'initialization.json'),
             'controller_source':str(Path(__file__).resolve()),'controller_source_sha256':sha256_file(__file__),
             'complete_source_state':info}
    atomic_write_json(state,CONTROL/'completed_controller_u600_before_u800.json')
    atomic_write_json(payload,EXTENSION)
    print(json.dumps({'status':'PREPARED_NOT_STARTED','extension':str(EXTENSION),'source':info},ensure_ascii=True),flush=True)

def snapshot(record):
    run=Path(record['run_dir']);data=inspect_files(run)
    data.update(protocol=VERSION,inspection_mode='FILES_ONLY',automatic_restart=False,
                authorized_extension=str(EXTENSION),current_control=base.read(CONTROL/'status.json'),
                safe_update=base.read(run/'checkpoint_manifest.json')['latest']['update'])
    atomic_write_json(data,run/'session/monitor_latest.json')
    with (run/'session/monitor_history.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(data,allow_nan=False)+'\n')

def supervise(record,start):
    run=Path(record['run_dir']);session=run/'session';stamp=uuid.uuid4().hex
    if (session/'launch.json').exists() or (session/'exit.json').exists():
        raise ValueError('Prior launch must be archived')
    log=session/f'training_{stamp}.log'
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker',
             '--from-update',str(start),'--launch-id',stamp]
    launch={'launch_id':stamp,'spawned_pid':None,'actual_worker_pid':None,'trainer_pid':None,
            'supervisor_pid':os.getpid(),'command':command,'log_path':str(log),'started_at':base.now(),
            'resume_update':start,'stop_update':800,'status':'DISPATCHED_NOT_TRAINING_SUCCESS',
            'automatic_restart':False,'extension_record':str(EXTENSION)}
    atomic_write_json(launch,session/'launch.json')
    camera=None;env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with log.open('x',encoding='utf-8') as out,(session/f'camera_{stamp}.log').open('x',encoding='utf-8') as camout:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=flags)
        launch['spawned_pid']=child.pid;atomic_write_json(launch,session/'launch.json');deadline=0
        while child.poll() is None:
            actual=base.read(session/'launch.json').get('actual_worker_pid')
            if actual and camera is None:
                camera=subprocess.Popen([sys.executable,'-B','-u','-m','utils.follow_reward_fork',
                    '--trainer-pid',str(actual),'--run-dir',str(run)],cwd=ROOT,stdout=camout,
                    stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,creationflags=flags)
            if time.monotonic()>=deadline:
                try:
                    snapshot(record)
                except Exception as e:
                    atomic_write_json({'at':base.now(),'error':f'{type(e).__name__}: {e}'},session/'monitor_error.json')
                deadline=time.monotonic()+1200
            time.sleep(2)
        if camera is not None:
            try:camera.wait(timeout=8)
            except subprocess.TimeoutExpired:pass
        actual=base.read(session/'launch.json').get('actual_worker_pid')
        atomic_write_json({'launch_id':stamp,'spawned_pid':child.pid,'trainer_pid':actual,
            'supervisor_pid':os.getpid(),'returncode':child.returncode,'ended_at':base.now(),
            'camera_pid':camera.pid if camera else None,'camera_returncode':camera.poll() if camera else None,
            'automatic_restart':False},session/'exit.json')
    if child.returncode or not actual:
        raise RuntimeError(f'Extension training failed, exit={child.returncode}, actual_worker={actual}')
    base.safe_boundary(record,expected=800)

def controller():
    record=effective_record();run=Path(record['run_dir'])
    with ExitStack() as owners:
        for directory in [run,CONTROL]:owners.enter_context(RunSessionOwner(directory))
        previous=base.read(CONTROL/'status.json')
        if previous.get('status')!='COMPLETED' or previous.get('target_update')!=600:
            raise ValueError('Initial extension requires original terminal u600 controller; no automatic recovery')
        base._assert_worker_idle(record,PLAN)
        base.require_geometry(record)
        base.safe_boundary(record,expected=600)
        if (run/'session/STOP_REQUESTED').exists():
            raise ValueError('STOP_REQUESTED exists; no implicit override')
        state={'version':VERSION,'status':'STARTING','pid':os.getpid(),'started_at':base.now(),
               'original_plan_sha256':sha256_file(PLAN),'extension_sha256':sha256_file(EXTENSION),
               'completed_stages':[50,100,200,400,600],'initial_safe_update':600,'target_update':800,
               'total_budget_updates':800,'total_budget_episodes':12800,'automatic_restart':False}
        atomic_write_json(state,CONTROL/'status.json')
        try:
            base.archive_session(record,reason='completed')
            state.update(status='TRAINING',changed_at=base.now());atomic_write_json(state,CONTROL/'status.json')
            supervise(record,600)
            state.update(status='FROZEN_EVALUATION',changed_at=base.now());atomic_write_json(state,CONTROL/'status.json')
            result=base.frozen_screen(record,800)
            state.update(status='COMPLETED',ended_at=base.now(),completed_stages=[50,100,200,400,600,800],last_stage_selection=result['selection'])
            atomic_write_json(state,CONTROL/'status.json');snapshot(record)
        except BaseException as e:
            state.update(status='FAILED_NO_RESTART',error=f'{type(e).__name__}: {e}',ended_at=base.now())
            atomic_write_json(state,CONTROL/'status.json');raise

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','prepare','run','worker'])
    p.add_argument('--from-update',type=int);p.add_argument('--launch-id');a=p.parse_args()
    if a.mode=='audit':print(json.dumps(checkpoint_summary(effective_record(False)),ensure_ascii=True))
    elif a.mode=='prepare':prepare()
    elif a.mode=='run':controller()
    else:
        if a.from_update!=600 or not a.launch_id:raise ValueError('Worker must resume exactly u600')
        record=effective_record()
        with RunSessionOwner(Path(record['run_dir'])/'worker_guard'):
            base._worker_owned(PLAN,record,600,800,a.launch_id)

if __name__=='__main__':main()
