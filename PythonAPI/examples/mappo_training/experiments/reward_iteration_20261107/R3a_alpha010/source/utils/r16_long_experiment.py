"""Registered R16 72k run with serial milestone evaluation and owned-server close."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'experiments/adversarial_u1000_R16_long_20261105_v1'
PS='C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def powershell(code):
    result=subprocess.run([PS,'-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True,
        encoding='utf-8',timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


def server_identity():
    return powershell("$ErrorActionPreference='Stop'; $all=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'CarlaUE4*'} | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId;parent=$_.ParentProcessId;path=$_.ExecutablePath;created=$_.CreationDate.ToUniversalTime().ToString('o')} }); $listeners=@(Get-NetTCPConnection -LocalPort 2000 -State Listen); if($listeners.Count -ne 1){throw 'Expected exactly one registered CARLA listener'}; [pscustomobject]@{processes=$all;engine_pid=$listeners[0].OwningProcess} | ConvertTo-Json -Depth 4")


def prepare(root):
    from training.checkpoint_protocol import atomic_write_json as write,sha256_file
    from training.mut_mappo_config import load_config,portable_config,validate_config
    from environment.mut_mappo_efficiency_reward import ReferenceStore,reference_identity
    old=ROOT/'experiments/adversarial_u1000_learning_diagnosis_20261104_v3'
    cfg=portable_config(load_config(old/'configs/R16.json'))
    cfg['run']['train_tick_budget']=72000
    cfg['task']['team_reward']['reference_dir']=str(root/'references/R2')
    cfg['diagnostic_authorization']['source']='2026-11-05 user authorized R16 72k, periodic monitoring, five-hour follow-up and close owned CARLA after completed training/evaluation'
    validated=validate_config(cfg)
    refs=ReferenceStore(old/'references/R2',reference_identity(validated))
    for key in read(old/'references/R2/manifest.json')['cases']:refs.curve(key)
    server=server_identity()
    engines=[p for p in server['processes'] if p['pid']==server['engine_pid']]
    expected=ROOT.parents[2]/'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'
    if len(engines)!=1 or Path(engines[0]['path']).resolve()!=expected.resolve():
        raise ValueError('listener is not the expected workspace CARLA engine')
    root.mkdir(parents=True,exist_ok=False);(root/'control').mkdir()
    shutil.copytree(old/'references/R2',root/'references/R2')
    config=root/'configs/R2_B0_s42.json';write(cfg,config)
    write(server,root/'control/server_identity.json')
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha256_file(p)
        for folder in ('environment','training','evaluation','models') for p in (ROOT/folder).glob('*.py')}
    for name in ('utils/r16_long_experiment.py','utils/r2_budget_experiment.py','utils/mut_mappo_session.py',
                 'utils/evaluate_mut_mappo_critical.py','utils/recover_critical_reward_evaluation.py',
                 'utils/analyze_r2_budget_results.py','utils/r16_long_review.py','configs/evaluation/base_driving_vnext_v1.json'):
        sources[name]=sha256_file(ROOT/name)
    for name in sources:
        target=root/'control/source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    write(sources,root/'control/source_hashes.json')
    plan=dict(config=str(config),config_sha256=sha256_file(config),train_tick_budget=72000,
        milestone_ticks=[0,24000,48000,72000],frozen_episode_budget=576,frozen_episodes_per_node=144,
        scope='one outer seed42; A/B are two learners, weak-red engineering only',
        initialization='fixed u1000 Actors; fresh four-car Critic, optimizer, ValueNorm',
        red_sha256=cfg['red_checkpoint']['sha256'],episodes_per_team_update=16,
        lr_horizon_train_ticks=72000,case_manifest_sha256=validated['_root_validation']['case_manifest_sha256'],
        environment=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        early_stop='At48k, stop if BOTH teams have frozen mean return below initial AND24k, and event eligibility below24k. Engineering regression guard, no threshold changes.',
        close_server='after all requested or early-stop frozen stages exited0, evidence checked, shared lease acquired and world idle; exact PID/path/creation identity only',
        user_authorization='2026-11-05 current chat',estimated_scope='training ticks include setup and cleanup; validation/frozen/failed sampling separate')
    write(plan,root/'control/experiment_plan.json')
    write(dict(status='PREPARED',at=time.time()),root/'control/pipeline_status.json')
    write(dict(source=str(old/'references/R2'),manifest_sha256=sha256_file(old/'references/R2/manifest.json'),
        new_reference_ticks=0),root/'control/reference_reuse.json')
    (root/'control/RESUME.md').write_text('User authorized R16 72k on 2026-11-05. Read experiment_plan.json, pipeline_status.json, automations.json, server_identity.json and session exit evidence. Follow actual worker and UUID; never duplicate tick/reset. No reward/qualification changes. Runs/R2 has segment24/48/72 continuation of the SAME new experiment, never resume base driving. All stages serial. Healthy progress quiet. Five-hour heartbeat restores only monitoring when needed; no blind training restart. Completed pipeline automatically writes review, then closes exact registered CARLA identity. Never stop other processes or a replacement server. Delete both automations when complete.\n',encoding='utf-8')
    return plan


def checkpoint_node(root,target):
    from utils.r2_budget_experiment import records
    from training.checkpoint_protocol import sha256_file
    rows=records(root/'runs/R2/metrics.jsonl')
    row=next(r for r in rows if r['completed_round']==0) if target==0 else next(r for r in rows if r.get('actual_counters',{}).get('train',0)>=target)
    paths=list((root/'runs/R2/boundaries').glob('round_'+format(row['completed_round'],'08d')+'_*.pt'))
    if len(paths)!=1:raise ValueError('ambiguous checkpoint node')
    return dict(target_train_ticks=target,completed_round=row['completed_round'],actual_train_ticks=row.get('actual_counters',{}).get('train',0),checkpoint=str(paths[0]),sha256=sha256_file(paths[0]))


def frozen_stats(root,target):
    report=read(root/'evaluations'/('frozen_'+str(target).zfill(6))/'analysis/critical_report.json')
    cases=report['reports'][0]['cases'];result={}
    for team in ('A','B'):
        cohort=[c for c in cases if c['condition']=='C2' and c['team']==team]
        result[team]=dict(mean_return=sum(c['team_discounted_return'] for c in cohort)/len(cohort),
            eligible=sum(c['blue_eligible'] for c in cohort)/len(cohort))
    return result


def close_server(root):
    from training.checkpoint_protocol import atomic_write_json as write,sha256_file
    from utils.mut_mappo_session import CarlaOwner,now
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    state=read(root/'control/pipeline_status.json')
    if state['status'] not in ('EVIDENCE_COMPLETE','COMPLETED','STOPPED_FOR_REGRESSION'):
        raise ValueError('cannot close server before verified stage completion')
    for stage in state['completed_stages']:
        done=read(root/'control'/(stage+'_completed.json'))
        if done['returncode']!=0:raise ValueError('stage failed')
        for name,digest in done['artifact_hashes'].items():
            if sha256_file(root/name)!=digest:raise ValueError('completed artifact changed')
    identity=read(root/'control/server_identity.json')
    engine=next(p for p in identity['processes'] if p['pid']==identity['engine_pid'])
    parent=next((p for p in identity['processes'] if p['pid']==engine['parent']),None)
    def ps_quote(value):return "'"+str(value).replace("'","''")+"'"
    code="$ErrorActionPreference='Stop'; $items=@("+','.join("[pscustomobject]@{pid="+str(p['pid'])+";path="+ps_quote(p['path'])+";created="+ps_quote(p['created'])+"}" for p in [engine,parent] if p)+"); $out=@(); foreach($item in $items){ $proc=Get-CimInstance Win32_Process -Filter ('ProcessId='+$item.pid); if($null -eq $proc){$out += [pscustomobject]@{pid=$item.pid;status='already_exited'};continue}; if($proc.ExecutablePath -ne $item.path -or $proc.CreationDate.ToUniversalTime().ToString('o') -ne $item.created){throw 'PID identity changed; refuse close'}; Stop-Process -Id $item.pid -ErrorAction Stop; $out += [pscustomobject]@{pid=$item.pid;status='closed_registered_process'} }; ConvertTo-Json -InputObject $out -Depth 3"
    with CarlaOwner(purpose='close registered CARLA after completed R16 experiment'):
        current=server_identity()
        if current['engine_pid']!=engine['pid']:raise ValueError('replacement server; refuse shutdown')
        idle=_preflight_idle_world(host='127.0.0.1',port=2000,expected_version='0.9.16')
        result=powershell(code)
    write(dict(status='CLOSED',at=now(),processes=result,pre_close_idle=idle),root/'control/server_shutdown.json')
    return result


def run(root):
    from training.checkpoint_protocol import atomic_write_json as write,sha256_file
    from utils.mut_mappo_session import RunSessionOwner,now
    from utils.r2_budget_experiment import live_files
    from utils.recover_critical_reward_evaluation import preflight
    plan=read(root/'control/experiment_plan.json')
    state=dict(status='RUNNING',pipeline_uuid=uuid.uuid4().hex,worker_pid=os.getpid(),started_at=now(),completed_stages=[])
    def verify():
        if sha256_file(plan['config'])!=plan['config_sha256']:raise ValueError('configuration changed')
        for path,digest in read(root/'control/source_hashes.json').items():
            if sha256_file(ROOT/path)!=digest:raise ValueError('registered source changed '+path)
    def stage(name,args,out=None):
        verify();log=root/'control'/(name+'.log')
        with log.open('x',encoding='utf-8') as stream:
            proc=subprocess.Popen([sys.executable,'-X','utf8','-u','-m',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',**plan['environment']),creationflags=subprocess.CREATE_NO_WINDOW)
            state.update(stage=name,stage_pid=proc.pid,stage_log=str(log),at=now(),evaluation_output=str(out) if out else None)
            write(state,root/'control/pipeline_status.json')
            while proc.poll() is None:live_files(root,state);time.sleep(10)
        done=dict(stage=name,returncode=proc.returncode,stage_pid=proc.pid,pipeline_uuid=state['pipeline_uuid'],ended_at=now(),log=str(log))
        write(done,root/'control'/(name+'_exit.json'))
        if proc.returncode:raise RuntimeError(name+' failed; retain evidence')
        if name.startswith('train_'):
            session=root/'runs/R2/session';launch,end=read(session/'launch.json'),read(session/'exit.json')
            if any(launch[k]!=end[k] for k in ('launch_id','trainer_pid','supervisor_pid')) or end['returncode']:
                raise ValueError('trainer exit identity mismatch')
            done['training_exit']=end
            write(dict(launch=launch,exit=end,status=read(session/'status.json')),root/'control'/(name+'_session.json'))
            if not read(session/'status.json')['segment_tick_cap_reached']:raise ValueError('segment cap not reached')
            # Each segment signs immutable latest-boundary checkpoints, not the mutable manifest.
            node=checkpoint_node(root,int(name.split('_')[1]));artifact=Path(node['checkpoint'])
        else:
            manifest=read(out/'evaluations/eval_000/manifest.json');report=read(out/'analysis/critical_report.json')
            if manifest['status']!='COMPLETED' or manifest['completed_attempted_episodes']!=144 or any(r['measurement_coverage']!=1 for r in report['team_rows']):
                raise ValueError('incomplete frozen evaluation')
            artifact=out/'analysis/critical_report.json'
        done['artifact_hashes']={str(artifact.relative_to(root)):sha256_file(artifact)}
        write(done,root/'control'/(name+'_completed.json'));state['completed_stages'].append(name);state['stage_pid']=None
        write(state,root/'control/pipeline_status.json')
    from utils.evaluate_mut_mappo_critical import gpu_memory_capture
    with RunSessionOwner(root/'pipeline'),gpu_memory_capture(root/'control/gpu_memory.jsonl'):
        try:
            verify();write(preflight(),root/'control/server_readiness.json')
            # Refuse a replacement simulator, even if its map is idle.
            if server_identity()!=read(root/'control/server_identity.json'):raise ValueError('registered server identity changed')
            nodes=[];early=False
            for target in (24000,48000,72000):
                args=['utils.mut_mappo_session','supervise','--config',plan['config'],'--run-dir',str(root/'runs/R2'),'--stop-train-ticks',str(target)]
                if target>24000:args.append('--resume')
                stage('train_'+str(target).zfill(6),args)
                for node_target in ((0,24000) if target==24000 else (target,)):
                    node=checkpoint_node(root,node_target);nodes.append(node);write(nodes,root/'control/milestones.json')
                    name='frozen_'+str(node_target).zfill(6);out=root/'evaluations'/name
                    stage(name,['utils.evaluate_mut_mappo_critical','live','--config',plan['config'],'--checkpoint',node['checkpoint'],
                        '--split','gate','--output-dir',str(out),'--monitor-mode','off','--reuse-idle-world'],out)
                if target==48000:
                    stats={str(t):frozen_stats(root,t) for t in (0,24000,48000)}
                    early=all(stats['48000'][t]['mean_return']<min(stats['0'][t]['mean_return'],stats['24000'][t]['mean_return'])
                        and stats['48000'][t]['eligible']<stats['24000'][t]['eligible'] for t in ('A','B'))
                    write(dict(stop=early,statistics=stats,rule=plan['early_stop']),root/'control/regression_guard.json')
                    if early:break
            state.update(status='EVIDENCE_COMPLETE',stage_pid=None,ended_at=now(),early_stopped=early)
            write(state,root/'control/pipeline_status.json');live_files(root,state)
            from utils.r16_long_review import build
            try:build(root)
            except Exception as analysis_error:
                state['analysis_error']=str(analysis_error)
                write(dict(error=str(analysis_error),traceback=traceback.format_exc()),root/'control/analysis_error.json')
            close_server(root)
            state.update(status='STOPPED_FOR_REGRESSION' if early else 'COMPLETED',server_closed=True,ended_at=now())
        except BaseException as error:
            state.update(status='FAILED_REVIEW_REQUIRED',at=now(),error=str(error),traceback=traceback.format_exc())
            write(state,root/'control/pipeline_status.json');live_files(root,state);raise
        write(state,root/'control/pipeline_status.json');live_files(root,state)
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','close'))
    p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();root=a.root.resolve()
    result=prepare(root) if a.command=='prepare' else close_server(root) if a.command=='close' else run(root)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
