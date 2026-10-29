"""Explicit u800 -> u1000 extension after new u400/u600 reevaluation; preserve the completed original plan.

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
EXTENSION = CONTROL / 'extension_u1000_v1.json'
SOURCE_SHA = '4d71b51e35e9b0bb8b32dea7bc931d02848783cc86035a3c9cb9900a75e70564'
VERSION = 'base_driving_explicit_u1000_extension_v1'

def effective_record(require_extension=True):
    record = base.load_plan(PLAN)
    if record['total_budget_updates'] != 600 or record['stop_updates'] != [50,100,200,400,600]:
        raise ValueError('Original completed u600 plan changed')
    summary = base.read(CONTROL / 'u000800_summary.json')
    source = CONTROL / 'u000800.pt'
    if summary['status'] != 'COMPLETED' or not summary['normal_training_matching_exit_0']:
        raise ValueError('u800 must have a complete matching-exit frozen stage')
    if summary['checkpoint_sha256'] != SOURCE_SHA or sha256_file(source) != SOURCE_SHA:
        raise ValueError('Immutable u800 source mismatch')
    if require_extension:
        ext = base.read(EXTENSION)
        if (ext['version'] != VERSION or ext['original_plan_sha256'] != sha256_file(PLAN)
                or ext['source_checkpoint_sha256'] != SOURCE_SHA
                or ext['total_budget_updates'] != 1000 or ext['total_budget_episodes'] != 16000
                or ext['controller_source_sha256'] != sha256_file(__file__)
                or ext['pretraining_reevaluation_sha256'] != sha256_file(REEVAL/'summary.json')
                or ext['initialization_sha256'] != sha256_file(Path(record['run_dir'])/'initialization.json')):
            raise ValueError('Authorized extension binding changed')
    record = dict(record)
    record.update(stop_updates=[50,100,200,400,600,800,1000], total_budget_updates=1000, total_budget_episodes=16000)
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
    validate_reevaluation()
    state = base.read(CONTROL/'status.json')
    if state.get('status') != 'COMPLETED' or state.get('completed_stages') != [50,100,200,400,600,800]:
        raise ValueError('u800 controller has not completed all six frozen nodes')
    info = checkpoint_summary(record)
    if info['safe_update'] != 800 or info['sha256'] != SOURCE_SHA or info['episodes'] != 12800:
        raise ValueError('Extension must start at the exact complete u800 boundary')
    if not base._matching_exit(Path(record['run_dir']),stop=800):
        raise ValueError('u800 launch UUID / actual worker exit mismatch')
    if base._completed_screens(record) != [50,100,200,400,600,800]:
        raise ValueError('Frozen evidence is incomplete')
    payload={'version':VERSION,'authorized_at':base.now(),
             'authorization':'Human explicitly requested a fresh full u400/u600 evaluation followed by 200 more updates from u800 to u1000 on 2026-10-27.',
             'original_plan':str(PLAN),'original_plan_sha256':sha256_file(PLAN),
             'source_checkpoint':str(CONTROL/'u000800.pt'),'source_checkpoint_sha256':SOURCE_SHA,
             'source_update':800,'source_episodes':12800,'total_budget_updates':1000,'total_budget_episodes':16000,
             'additional_updates':200,'additional_episodes':3200,'new_frozen_stops':[1000],
             'schedule_policy':'Preserve cumulative LR/entropy clock and existing 600 horizon; retain terminal schedule values after u600.',
             'automatic_restart':False,'original_status_archive':str(CONTROL/'completed_controller_u800_before_u1000.json'),
             'initialization_sha256':sha256_file(Path(record['run_dir'])/'initialization.json'),
             'controller_source':str(Path(__file__).resolve()),'controller_source_sha256':sha256_file(__file__),
             'complete_source_state':info,'pretraining_reevaluation':str(REEVAL/'summary.json'),
             'pretraining_reevaluation_sha256':sha256_file(REEVAL/'summary.json')}
    atomic_write_json(state,CONTROL/'completed_controller_u800_before_u1000.json')
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
            'resume_update':start,'stop_update':1000,'status':'DISPATCHED_NOT_TRAINING_SUCCESS',
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
    base.safe_boundary(record,expected=1000)

def controller():
    record=effective_record();validate_reevaluation();run=Path(record['run_dir'])
    with ExitStack() as owners:
        for directory in [run,CONTROL]:owners.enter_context(RunSessionOwner(directory))
        previous=base.read(CONTROL/'status.json')
        if previous.get('status')!='COMPLETED' or previous.get('target_update')!=800:
            raise ValueError('Initial extension requires terminal u800 controller; no automatic recovery')
        base._assert_worker_idle(record,PLAN)
        base.require_geometry(record)
        base.safe_boundary(record,expected=800)
        if (run/'session/STOP_REQUESTED').exists():
            raise ValueError('STOP_REQUESTED exists; no implicit override')
        state={'version':VERSION,'status':'STARTING','pid':os.getpid(),'started_at':base.now(),
               'original_plan_sha256':sha256_file(PLAN),'extension_sha256':sha256_file(EXTENSION),
               'completed_stages':[50,100,200,400,600,800],'initial_safe_update':800,'target_update':1000,
               'total_budget_updates':1000,'total_budget_episodes':16000,'automatic_restart':False}
        atomic_write_json(state,CONTROL/'status.json')
        try:
            base.archive_session(record,reason='completed')
            state.update(status='TRAINING',changed_at=base.now());atomic_write_json(state,CONTROL/'status.json')
            supervise(record,800)
            state.update(status='FROZEN_EVALUATION',changed_at=base.now());atomic_write_json(state,CONTROL/'status.json')
            result=base.frozen_screen(record,1000)
            state.update(status='COMPLETED',ended_at=base.now(),completed_stages=[50,100,200,400,600,800,1000],last_stage_selection=result['selection'])
            atomic_write_json(state,CONTROL/'status.json');snapshot(record)
        except BaseException as e:
            state.update(status='FAILED_NO_RESTART',error=f'{type(e).__name__}: {e}',ended_at=base.now())
            atomic_write_json(state,CONTROL/'status.json');raise

REEVAL = CONTROL/'reevaluation_u400_u600_20261027_v1'

def validate_reevaluation():
    result=base.read(REEVAL/'summary.json')
    if result['status']!='COMPLETED' or result['evaluated_updates']!=[400,600]:
        raise ValueError('New full reevaluation must complete before training')
    from evaluation.base_driving_qualification import analyze_evaluation, select_candidate
    contract=ROOT/base.load_contract(effective_record(False)['contract_path'],ROOT)['payload']['qualification_contract']
    for stage in result['stages']:
        if sha256_file(stage['checkpoint'])!=stage['checkpoint_sha256']:
            raise ValueError('Reevaluation checkpoint changed')
        reports={}
        for suite,required in [('parallel',32),('validation',64)]:
            directory=REEVAL/f"screen_u{stage['update']:06d}"/suite
            if sha256_file(directory/'qualification.json')!=stage[suite+'_qualification_sha256']:
                raise ValueError('Reevaluation report changed')
            report=analyze_evaluation(directory,contract)
            if len(report['reports'])!=1:raise ValueError('Expected exactly one checkpoint per suite')
            item=report['reports'][0]
            if (item['case_count']!=required or item['evidence_status']!='COMPLETE_VALID_EVIDENCE'
                    or item['errors'] or item['checkpoint_sha256']!=stage['checkpoint_sha256']):
                raise ValueError('Incomplete/invalid evaluation evidence')
            reports[suite]=report['reports']
        if select_candidate(reports['validation'],regression_reports=reports['parallel'])!=stage['selection']:
            raise ValueError('Reevaluation selection changed')
    return result

def reevaluate():
    record=effective_record(False);run=Path(record['run_dir'])
    with ExitStack() as owners:
        for directory in [run,CONTROL]:owners.enter_context(RunSessionOwner(directory))
        base._assert_worker_idle(record,PLAN)
        base.safe_boundary(record,expected=800)
        if not base._matching_exit(run,stop=800):raise ValueError('u800 matching exit required')
        if REEVAL.exists():raise FileExistsError('Do not overwrite reevaluation evidence')
        base.require_geometry(record);REEVAL.mkdir()
        from evaluation.evaluate_base_driving_vnext import evaluate
        from evaluation.base_driving_qualification import select_candidate
        contract=ROOT/base.load_contract(record['contract_path'],ROOT)['payload']['qualification_contract']
        state={'status':'RUNNING','started_at':base.now(),'pid':os.getpid(),'stages':[],
               'evaluated_updates':[400,600],'automatic_restart':False,
               'source_training_safe_update':800,'training_resume_from_update':800}
        atomic_write_json(state,REEVAL/'summary.json')
        try:
            for u in [400,600]:
                checkpoint=CONTROL/f'u{u:06d}.pt';digest=sha256_file(checkpoint)
                old=base.read(CONTROL/f'u{u:06d}_summary.json')
                if digest!=old['checkpoint_sha256'] or not old['normal_training_matching_exit_0']:
                    raise ValueError('Historical fixed checkpoint identity invalid')
                suites={}
                for suite in ['parallel','validation']:
                    print('START_SUITE',u,suite,flush=True)
                    state.update(current_update=u,current_suite=suite,changed_at=base.now())
                    atomic_write_json(state,REEVAL/'summary.json')
                    split='train' if suite=='parallel' else 'validation'
                    suites[suite]=evaluate([checkpoint],REEVAL/f'screen_u{u:06d}'/suite,
                        Path(record[split+'_scenario_dir'])/'protocol/manifest.json',contract,
                        geometry_report=record['geometry_reports'][split],parallel_regression=(suite=='parallel'))
                if sha256_file(checkpoint)!=digest:raise ValueError('Fixed checkpoint changed during evaluation')
                stage={'update':u,'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
                       'selection':select_candidate(suites['validation']['reports'],regression_reports=suites['parallel']['reports'])}
                for suite in ['parallel','validation']:
                    stage[suite+'_qualification_sha256']=sha256_file(REEVAL/f'screen_u{u:06d}'/suite/'qualification.json')
                atomic_write_json(stage,REEVAL/f'u{u:06d}_summary.json')
                state['stages'].append(stage);atomic_write_json(state,REEVAL/'summary.json')
                print('REEVALUATION_STAGE_DONE',json.dumps(stage),flush=True)
            base.safe_boundary(record,expected=800)
            state.update(status='COMPLETED',ended_at=base.now());atomic_write_json(state,REEVAL/'summary.json')
            validate_reevaluation()
        except BaseException as e:
            state.update(status='FAILED_NO_RESTART',error=f'{type(e).__name__}: {e}',ended_at=base.now())
            atomic_write_json(state,REEVAL/'summary.json');raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','prepare','run','worker','reevaluate'])
    p.add_argument('--from-update',type=int);p.add_argument('--launch-id');a=p.parse_args()
    if a.mode=='audit':print(json.dumps(checkpoint_summary(effective_record(False)),ensure_ascii=True))
    elif a.mode=='reevaluate':reevaluate()
    elif a.mode=='prepare':prepare()
    elif a.mode=='run':controller()
    else:
        if a.from_update!=800 or not a.launch_id:raise ValueError('Worker must resume exactly u800')
        record=effective_record()
        with RunSessionOwner(Path(record['run_dir'])/'worker_guard'):
            base._worker_owned(PLAN,record,800,1000,a.launch_id)

if __name__=='__main__':main()
