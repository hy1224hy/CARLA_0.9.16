"""Run exactly one agent-chosen reward experiment; never choose a parameter automatically."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(r'D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training/experiments/reward_iteration_20261108')
PROJECT=ROOT.parent.parent
sys.path.insert(0,str(PROJECT))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--version',required=True)
    p.add_argument('--from-version',required=True)
    p.add_argument('--parameter',required=True,choices=['outcome_weight','control_weight','progress_weight','collision_scale'])
    p.add_argument('--value',required=True,type=float)
    p.add_argument('--reason-file',required=True)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    for name in (args.version,args.from_version):
        if not re.fullmatch('[A-Za-z0-9_]+',name):raise ValueError('version must be one local directory name')
    if not math.isfinite(args.value):raise ValueError('finite reward parameter required')
    ranges={'outcome_weight':(.01,20.),'control_weight':(0.,.3),'progress_weight':(0.,.03),'collision_scale':(.5,1.)}
    low,high=ranges[args.parameter]
    if not low<=args.value<=high:raise ValueError('parameter outside registered experimental range')
    reason=Path(args.reason_file).read_text(encoding='utf-8-sig')
    if len(reason)<100:raise ValueError('an evidence-based agent rationale is required')
    spec=importlib.util.spec_from_file_location('r5_iteration_helper',ROOT/'control/nightly_r5.py')
    n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
    prior=ROOT/args.from_version;directory=ROOT/args.version
    previous=n.read(prior/'analysis.json')
    state=n.read(ROOT/'control/state.json')
    if state.get('version')!=args.from_version or state.get('status')!='AWAITING_AGENT_REWARD_ANALYSIS':
        raise RuntimeError('prior experiment is not ready for agent review')
    if n.read(ROOT/'control/worker.json').get('active') is not False:raise RuntimeError('simulator worker still active')
    if not all(x.get('complete') for x in previous['counterfactual'].values()):raise RuntimeError('previous policy replacement evidence incomplete')
    if args.launch:
        # This launches only the explicitly specified next experiment, after its rationale exists.
        command=[sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!='--launch']]
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log=ROOT/'control'/f'{args.version}_{stamp}_controller.log'
        with log.open('x',encoding='utf-8') as output:
            child=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP,
                env={**os.environ,'PYTHONFAULTHANDLER':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        n.write(ROOT/'control'/f'{args.version}_dispatch.json',dict(pid=child.pid,command=command,log=str(log),at=n.now()))
        print(json.dumps(dict(pid=child.pid,command=command,log=str(log))))
        return
    from utils.mut_mappo_session import RunSessionOwner
    with RunSessionOwner(ROOT/'control/controller_lock'):
        directory.mkdir(exist_ok=False)
        cfg=n.read(prior/'config.json')
        old=cfg['task']['team_reward'][args.parameter]
        if old==args.value:raise ValueError('the proposed experiment does not change the reward')
        cfg['task']['team_reward'][args.parameter]=args.value
        assert cfg['run']['rounds']==100 and cfg['run']['lr_horizon_rounds']==100 and cfg['method']=='off' and cfg['seed']==42
        n.write(directory/'config.json',cfg)
        (directory/'REASON.md').write_text(reason+f'\n\n唯一参数变更：{args.parameter}: {old} -> {args.value}。从相同未对抗训练基础策略重新初始化，B0 seed42、100updates、LR horizon100。该参数由智能体审阅前版结果后指定，脚本不决定下一组参数。\n',encoding='utf-8')
        (prior/('AGENT_REVIEW_FOR_'+args.version+'.md')).write_text(reason,encoding='utf-8')
        n.snapshot(directory)
        shutil.copyfile(__file__,directory/'experiment_driver.py')
        hashes=n.read(directory/'source_hashes.json')
        hashes['experiment_driver.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        n.write(directory/'source_hashes.json',hashes)
        n.write(directory/'AGENT_DECISION.json',dict(at=n.now(),previous_version=args.from_version,parameter=args.parameter,
            before=old,after=args.value,previous_analysis_sha256=hashlib.sha256((prior/'analysis.json').read_bytes()).hexdigest(),automatic_parameter_selection=False))
        config=directory/'config.json'
        try:
            n.execute(directory,'training_u100','training.train_mut_mappo',['--config',config,'--run-dir',directory/'run','--allow-candidate'])
            if n.read(directory/'run/session/status.json').get('completed_round')!=100:raise RuntimeError('100 complete updates required')
            initial=n.initial_checkpoint(directory)
            base=n.evaluate(directory,initial,'evaluation_base_u0')
            checkpoint=directory/'run/latest.pt'
            candidate=n.evaluate(directory,checkpoint,'evaluation_candidate_u100')
            effects=n.cf(directory,checkpoint)
            n.write(directory/'analysis.json',dict(at=n.now(),baseline=base,candidate=candidate,counterfactual=effects,
                selection_screen=any(r['engineering_screen_passed'] for r in effects.values()),
                purpose='Evidence for the agent to decide the next iteration; no automated reward changes.'))
            n.write(ROOT/'control/state.json',dict(at=n.now(),status='AWAITING_AGENT_REWARD_ANALYSIS',version=args.version,phase='review',
                completed_updates=100,automatic_parameter_changes=False))
        except BaseException as error:
            n.write(ROOT/'control/state.json',dict(at=n.now(),status='FAILED_REVIEW_REQUIRED',version=args.version,error=repr(error),automatic_restart=False))
            raise

if __name__=='__main__':main()
