"""Versioned three-step real-CARLA PPO and reward diagnosis; serial controllers."""
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
DEFAULT=ROOT/'experiments/adversarial_u1000_learning_diagnosis_20261104_v1'
GROUPS=('N4','R4','R16')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare(root):
    from training.checkpoint_protocol import atomic_write_json as write,sha256_file
    from training.mut_mappo_config import load_config,portable_config,validate_config
    from environment.mut_mappo_efficiency_reward import ReferenceStore,reference_identity
    old=ROOT/'experiments/adversarial_u1000_R2_budget_20261104_v1'
    base=portable_config(load_config(old/'configs/R2_B0_s42.json'))
    validated=validate_config(base)
    refs=ReferenceStore(old/'references/R2',reference_identity(validated))
    for key in read(old/'references/R2/manifest.json')['cases']:refs.curve(key)
    root.mkdir(parents=True,exist_ok=False)
    (root/'control').mkdir()
    shutil.copytree(old/'references/R2',root/'references/R2')
    configs={}
    for group in GROUPS:
        cfg=json.loads(json.dumps(base))
        cfg['run'].update(train_tick_budget=24000,episodes_per_team_update=16 if group=='R16' else 4)
        # Preserve the previous 72k LR trajectory for comparability to its first 24k.
        # The wrapper registers this independently from the shorter stopping budget.
        cfg['run']['lr_horizon_train_ticks']=72000
        cfg['task']['team_reward']['reference_dir']=str(root/'references/R2')
        cfg['diagnostic_authorization']['source']='2026-11-04 user authorized three-step real trajectory/PPO, normal-only and R2 batch diagnosis'
        if group=='N4':
            cfg['task']['team_reward'].update(attack_weight=0.,critical_weight=0.,primary_event_reward=0.,severe_weight=0.)
        validate_config(cfg)
        path=root/'configs'/(group+'.json');write(cfg,path)
        configs[group]=dict(path=str(path),sha256=sha256_file(path))
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha256_file(p)
        for folder in ('environment','training','evaluation','models') for p in (ROOT/folder).glob('*.py')}
    for name in ('utils/r2_learning_diagnosis.py','utils/ppo_real_rollout_audit.py','utils/evaluate_mut_mappo_critical.py',
                 'utils/mut_mappo_session.py','configs/evaluation/base_driving_vnext_v1.json'):
        sources[name]=sha256_file(ROOT/name)
    for name in sources:
        target=root/'control/source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    write(sources,root/'control/source_hashes.json')
    plan=dict(configs=configs,groups=list(GROUPS),train_ticks_per_group=24000,total_train_ticks=72000,
        soft_cap='finish current complete A+B update; record overshoot, never discard samples',
        lr_tick_horizon=72000,milestones=[0,12000,24000],
        frozen_episodes_per_node=144,maximum_frozen_episodes=1296,
        reward_ablation='N4 zeros attack/critical/primary/severe components; all existing normal driving costs unchanged',
        validation='same gate24 cases, paired C0/C1/C2; each team24 cases; selection/test untouched',
        formal_scope='weak-red engineering single outer seed42; A/B not independent seeds',
        success='optimization diagnostics plus fixed-case reward/driving improvement; adversarial success also needs precontact conflict, local resolution and full-driving joint improvement over initial/C1',
        no_guarantee='a 24-case diagnostic is insufficient for formal algorithm superiority',
        environment=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    write(plan,root/'control/plan.json')
    write(dict(status='PREPARED',groups=list(GROUPS)),root/'control/pipeline_status.json')
    (root/'control/RESUME.md').write_text('User authorized three-step diagnosis. Read plan.json, pipeline_status.json, source_hashes.json, updates.jsonl. Follow existing worker; never duplicate CARLA controller. All groups 24k soft train tick cap, same72k LR horizon. N4 then R4 then R16, frozen0/12k/24k. Failure is retained, not automatically restarted. Do not change thresholds/red. Delete monitor after completion.\n',encoding='utf-8')
    return plan


def train_group(root,group):
    from training import train_mut_mappo as runner
    from training.mut_mappo_config import validate_config,load_config
    from utils.mut_mappo_session import CarlaOwner,RunSessionOwner
    from utils.ppo_real_rollout_audit import instrument
    cfg=validate_config(load_config(root/'configs'/(group+'.json')))
    with RunSessionOwner(root/'runs'/group),CarlaOwner(purpose='registered learning diagnosis '+group),instrument(root/'audits'/group):
        return runner.train(cfg,root/'runs'/group)


def run(root):
    from training.checkpoint_protocol import atomic_write_json as write,sha256_file
    from utils.mut_mappo_session import RunSessionOwner,now
    from utils.r2_budget_experiment import records
    from utils.recover_critical_reward_evaluation import preflight
    plan=read(root/'control/plan.json')
    state=dict(status='RUNNING',uuid=uuid.uuid4().hex,worker_pid=os.getpid(),started_at=now(),completed=[])
    def verify():
        for path,digest in read(root/'control/source_hashes.json').items():
            if sha256_file(ROOT/path)!=digest:raise ValueError('registered source changed '+path)
        for item in plan['configs'].values():
            if sha256_file(item['path'])!=item['sha256']:raise ValueError('config changed')
    def stage(name,args):
        verify()
        log=root/'control'/(name+'.log')
        if log.exists():raise FileExistsError('preserve prior attempt '+name)
        with log.open('x',encoding='utf-8') as stream:
            p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',**plan['environment']),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            state.update(stage=name,stage_pid=p.pid,stage_log=str(log),at=now());write(state,root/'control/pipeline_status.json')
            while p.poll() is None:
                write(dict(at=now(),stage=name,stage_pid=p.pid,
                    metrics={g:(records(root/'runs'/g/'metrics.jsonl') or [{}])[-1] for g in GROUPS}),root/'control/live_metrics.json')
                time.sleep(10)
        write(dict(stage=name,pid=p.pid,returncode=p.returncode,ended_at=now(),log=str(log)),root/'control'/(name+'_exit.json'))
        if p.returncode:raise RuntimeError(name+' failed, inspect '+str(log))
        state['completed'].append(name);state['stage_pid']=None
        write(state,root/'control/pipeline_status.json')
    with RunSessionOwner(root/'pipeline'):
        try:
            verify();write(preflight(),root/'control/server_readiness.json')
            for group in GROUPS:
                stage(group+'_train',['utils.r2_learning_diagnosis','train','--root',str(root),'--group',group])
                rows=records(root/'runs'/group/'metrics.jsonl');nodes=[]
                for target in plan['milestones']:
                    row=next(r for r in rows if r['completed_round']==0) if target==0 else next(r for r in rows if r.get('actual_counters',{}).get('train',0)>=target)
                    paths=list((root/'runs'/group/'boundaries').glob('round_'+format(row['completed_round'],'08d')+'_*.pt'))
                    if len(paths)!=1:raise ValueError('ambiguous boundary')
                    nodes.append(dict(target=target,round=row['completed_round'],actual=row.get('actual_counters',{}).get('train',0),checkpoint=str(paths[0]),sha256=sha256_file(paths[0])))
                write(nodes,root/'control'/(group+'_nodes.json'))
                for node in nodes:
                    name=group+'_frozen_'+str(node['target']).zfill(6)
                    stage(name,['utils.evaluate_mut_mappo_critical','live','--config',plan['configs'][group]['path'],
                        '--checkpoint',node['checkpoint'],'--split','gate','--output-dir',str(root/'evaluations'/name),
                        '--monitor-mode','off','--reuse-idle-world'])
            state.update(status='COMPLETED_FOR_ANALYSIS',ended_at=now(),stage_pid=None)
        except BaseException as error:
            state.update(status='FAILED_REVIEW_REQUIRED',at=now(),error=str(error),traceback=traceback.format_exc())
            write(state,root/'control/pipeline_status.json');raise
        write(state,root/'control/pipeline_status.json')
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','train'))
    p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--group',choices=GROUPS)
    a=p.parse_args();root=a.root.resolve()
    if a.command=='prepare':result=prepare(root)
    elif a.command=='run':result=run(root)
    else:result=train_group(root,a.group)
    if a.command!='train':print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
