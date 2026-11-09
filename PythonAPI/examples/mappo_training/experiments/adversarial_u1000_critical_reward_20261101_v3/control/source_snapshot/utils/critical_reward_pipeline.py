"""Serial, recoverable engineering reward pilot; each child owns CARLA separately."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]


def make_plan(root):
    stages=[]
    for group in ('R1','R0'):
        config=root/'configs'/f'{group}_B0_s42.json'
        run=root/'runs'/group
        evaluation=root/'evaluations'/f'{group}_gate'
        stages.extend([
            dict(id=f'{group}_references',kind='references',group=group,command=[
                'evaluation.mut_mappo_reference','--config',str(config),
                '--splits','train','gate','selection','--repeats','2','--allow-weak-red']),
            dict(id=f'{group}_train',kind='train',group=group,command=[
                'utils.mut_mappo_session','supervise','--config',str(config),'--run-dir',str(run)]),
            dict(id=f'{group}_gate',kind='evaluation',group=group,command=[
                'utils.evaluate_mut_mappo_critical','live','--config',str(config),
                '--checkpoint',str(run/'latest.pt'),'--split','gate',
                '--output-dir',str(evaluation),'--monitor-mode','off'])])
    return dict(schema='critical_reward_serial_pipeline_v1',stages=stages,
        training_budget_per_group=24000,outer_seed=42,teams_are_independent_seeds=False,
        scope='EXPLICIT_WEAK_RED_ENGINEERING_ONLY_FAILED_C1_SCREEN_DISCLOSED',
        reference_repeats=2,reference_episode_budget=288,
        reference_identity_note='R0/R1 reward profiles bind different reference identities; actual C1 runs are separately budgeted',
        frozen_evaluation_episodes_per_group=144,test_used=False)


def work(root):
    from training.checkpoint_protocol import atomic_write_json, sha256_file
    from training.mut_mappo_config import validate_config,load_config
    from utils.mut_mappo_session import RunSessionOwner, now
    control=root/'control'; control.mkdir(parents=True,exist_ok=True)
    authorization=json.loads((control/'training_authorization.json').read_text(encoding='utf-8'))
    if not authorization.get('allow_failed_C1_engineering_training') or not authorization.get('allow_weak_red_reference'):
        raise RuntimeError('failed-C1 engineering training requires recorded explicit authorization')
    plan=make_plan(root)
    for group in ('R0','R1'):
        cfg=validate_config(load_config(root/'configs'/f'{group}_B0_s42.json'))
        if cfg['method']!='off' or cfg['run']['train_tick_budget']!=24000 or cfg['seed']!=42:
            raise ValueError('pipeline only covers the registered single-seed B0 reward pilot')
    path=control/'pipeline_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('pipeline plan changed; preserve old evidence')
    atomic_write_json(plan,path)
    sources={str(p.relative_to(ROOT)):sha256_file(p) for folder in ('environment','training','evaluation','models')
             for p in (ROOT/folder).glob('*.py')}
    sources['utils/critical_reward_pipeline.py']=sha256_file(Path(__file__))
    sources['utils/mut_mappo_session.py']=sha256_file(ROOT/'utils/mut_mappo_session.py')
    source_path=control/'pipeline_sources.json'
    if source_path.exists() and json.loads(source_path.read_text(encoding='utf-8'))!=sources:
        raise ValueError('pipeline sources changed; inspect before resuming trained evidence')
    atomic_write_json(sources,source_path)
    state={'status':'RUNNING','scope':plan['scope'],'worker_pid':os.getpid(),
           'pipeline_uuid':uuid.uuid4().hex,'started_at':now(),'completed_stages':[]}
    status=control/'pipeline_status.json'
    with RunSessionOwner(root/'pipeline'):
        try:
            for stage in plan['stages']:
                marker=control/(stage['id']+'_completed.json')
                if marker.exists():
                    done=json.loads(marker.read_text(encoding='utf-8'))
                    if done['returncode']!=0: raise ValueError('stage completion marker is not successful')
                    for name,digest in done['artifact_hashes'].items():
                        if sha256_file(root/name)!=digest:
                            raise ValueError('completed stage evidence changed: '+name)
                    state['completed_stages'].append(stage['id']); continue
                command=[sys.executable,'-X','utf8','-u','-m',*stage['command']]
                if stage['kind']=='train' and (root/'runs'/stage['group']/'checkpoint_manifest.json').exists():
                    command.append('--resume')
                if stage['kind']=='evaluation':
                    out=Path(command[command.index('--output-dir')+1])
                    if out.exists():
                        raise FileExistsError('partial evaluation retained; inspect and register a fresh output directory')
                attempt=uuid.uuid4().hex[:10]
                log=control/f"{stage['id']}_{attempt}.log"
                with log.open('x',encoding='utf-8') as stream:
                    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),**kwargs)
                    state.update(stage=stage['id'],stage_pid=process.pid,stage_command=command,
                                 stage_log=str(log),at=now())
                    atomic_write_json(state,status)
                    code=process.wait()
                exit_record=dict(stage=stage['id'],returncode=code,stage_pid=process.pid,
                    worker_pid=os.getpid(),pipeline_uuid=state['pipeline_uuid'],log=str(log),ended_at=now())
                atomic_write_json(exit_record,control/f"{stage['id']}_{attempt}_exit.json")
                if code: raise RuntimeError(f"{stage['id']} exited {code}; inspect {log}")
                if stage['kind']=='train':
                    session=root/'runs'/stage['group']/'session'
                    launch=json.loads((session/'launch.json').read_text(encoding='utf-8'))
                    exited=json.loads((session/'exit.json').read_text(encoding='utf-8'))
                    result=json.loads((session/'status.json').read_text(encoding='utf-8'))
                    if any(launch[k]!=exited[k] for k in ('trainer_pid','supervisor_pid','launch_id')) or exited['returncode']!=0:
                        raise ValueError('training exit identity mismatch')
                    if not result.get('training_budget_reached'):
                        raise ValueError('training ended before its registered budget')
                    exit_record['training_exit']=exited
                artifact=(root/'references'/stage['group']/'manifest.json' if stage['kind']=='references' else
                          root/'runs'/stage['group']/'checkpoint_manifest.json' if stage['kind']=='train' else
                          root/'evaluations'/f"{stage['group']}_gate"/'analysis/critical_report.json')
                exit_record['artifact_hashes']={str(artifact.relative_to(root)):sha256_file(artifact)}
                atomic_write_json(exit_record,marker)
                state['completed_stages'].append(stage['id'])
            state.update(status='COMPLETED_FOR_ANALYSIS',ended_at=now(),stage_pid=None)
            atomic_write_json(state,status)
            return state
        except BaseException as error:
            state.update(status='FAILED_REVIEW_REQUIRED',error=f'{type(error).__name__}: {error}',
                         traceback=traceback.format_exc(),at=now())
            atomic_write_json(state,status)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root if args.root.is_absolute() else ROOT/args.root
    print(json.dumps(work(root.resolve()),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
