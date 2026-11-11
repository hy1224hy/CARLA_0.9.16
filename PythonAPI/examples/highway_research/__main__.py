"""python -m highway_research --help (cwd: PythonAPI/examples)."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--root',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44,45,46])
    p.add_argument('--steps',type=int,default=500000);p.add_argument('--optional',action='store_true');p.add_argument('--tuning',action='store_true');p.add_argument('--tuning-selection')
    p=sub.add_parser('select-tuning');p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('smoke');p.add_argument('--root',required=True)
    p=sub.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--max-rounds',type=int)
    p=sub.add_parser('batch');p.add_argument('--root',required=True);p.add_argument('--arms',nargs='+',default=['main']);p.add_argument('--jobs',type=int,default=4);p.add_argument('--retries',type=int,default=1);p.add_argument('--dry-run',action='store_true')
    p=sub.add_parser('audit');p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='latest');p.add_argument('--team',choices=['A','B'],default='A')
    evaluation_args(p)
    p=sub.add_parser('audit-study');p.add_argument('--root',required=True);p.add_argument('--arms',nargs='+',default=['main'])
    p.add_argument('--checkpoint',default='latest');p.add_argument('--teams',nargs='+',choices=['A','B'],default=['A','B'])
    p.add_argument('--labels',nargs='+')
    p.add_argument('--conditions',nargs='+',default=['nominal']);p.add_argument('--action-seeds',nargs='+',type=int,default=[0])
    evaluation_args(p)
    p=sub.add_parser('report');p.add_argument('--root',required=True);p.add_argument('--output');p.add_argument('--allow-partial',action='store_true')
    p=sub.add_parser('replay');p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',default='latest')
    p.add_argument('--team',choices=['A','B'],default='A');p.add_argument('--case-seed',type=int,default=4000003);p.add_argument('--family',choices=['follow','adjacent'],default='adjacent')
    args=parser.parse_args()
    if args.command=='plan':
        from .matrix import prepare
        result=prepare(args.root,args.seeds,args.steps,args.optional,args.tuning,args.tuning_selection)
        result={k:result[k] for k in ['run_count','training_decisions','source_sha256']}
    elif args.command=='smoke':
        from .smoke import smoke
        result=smoke(args.root)
    elif args.command=='select-tuning':
        from .matrix import recommend_tuning
        result=recommend_tuning(args.root,args.output)
    elif args.command=='train':
        from .config import StudyConfig
        from .runtime import train
        result=train(StudyConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8-sig'))),args.output,args.resume,args.max_rounds)
    elif args.command=='batch':
        from .batch import run_batch
        result=run_batch(args.root,args.arms,args.jobs,args.retries,args.dry_run)
    elif args.command=='audit':
        from .evaluate import audit
        result=audit(args.run,args.output,args.checkpoint,args.team,args.split,args.cases,args.condition,args.execution,args.action_seed)
    elif args.command=='audit-study':result=audit_study(args)
    elif args.command=='replay':
        from .visualize import replay
        result=replay(args.run,args.output,args.checkpoint,args.team,args.case_seed,args.family)
    else:
        from .report import build_report
        result=build_report(args.root,args.output,args.allow_partial)
    print(json.dumps(result,indent=2))


def evaluation_args(parser):
    parser.add_argument('--split',choices=['development','test'],default='development')
    parser.add_argument('--cases',type=int,default=40)
    parser.add_argument('--condition',choices=['nominal','idm_short_gap','idm_long_gap','cautious_lane','position_shift'],default='nominal')
    parser.add_argument('--execution',choices=['deterministic','stochastic'],default='deterministic')
    parser.add_argument('--action-seed',type=int,default=0)


def audit_study(args):
    from .evaluate import audit
    from .runtime import digest
    root=Path(args.root).resolve();plan=json.loads((root/'study.json').read_text())
    if plan['source_sha256']!=digest():raise ValueError('Study source changed')
    conditions=set(args.conditions)
    valid={'nominal','idm_short_gap','idm_long_gap','cautious_lane','position_shift'}
    if not conditions<=valid:raise ValueError('unknown condition')
    if args.execution=='deterministic' and args.action_seeds!=[0]:raise ValueError('Multiple action seeds only meaningful for stochastic execution')
    completed=[];missing=[]
    for row in plan['runs']:
        if row['arm'] not in args.arms:continue
        if args.labels and row['label'] not in args.labels:continue
        if args.checkpoint=='init' and row['label']!='PPO':continue
        run=Path(row['output']);summary=run/'summary.json'
        if not summary.exists() or json.loads(summary.read_text())['status']!='complete':missing.append(row['run_id']);continue
        for condition in args.conditions:
            for action_seed in args.action_seeds:
                teams=['A'] if args.checkpoint in ('selected','init') else args.teams
                for team in teams:
                    output=root/'audits'/args.split/args.checkpoint/condition/(args.execution+f'_a{action_seed}')/row['run_id']/team
                    if (output/'metrics.json').exists():completed.append(str(output));continue
                    if output.exists():
                        quarantine=output.with_name(output.name+'.incomplete_'+str(time.time_ns()))
                        assert output.resolve().is_relative_to(root)
                        output.rename(quarantine)
                    audit(run,output,args.checkpoint,team,args.split,args.cases,condition,args.execution,action_seed)
                    completed.append(str(output));print('AUDITED',row['run_id'],team,condition,flush=True)
    result=dict(audits=len(completed),missing_completed_training=missing)
    (root/'audit_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':main()
