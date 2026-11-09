"""Independent v4 workflow: python -m highwaytest.bottleneck ..."""
import argparse
import json
from pathlib import Path
from .. import staged
from ..config import load
from ..training.checkpoint import source_digest
from .contract import task,PROTOCOL
from .calibration import calibrate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',default='runs/bottleneck_v4_s42')
    a.add_argument('--seed',type=int,default=42);a.add_argument('--normal-steps',type=int,default=100000)
    a.add_argument('--attack-steps',type=int,default=200000)
    a=sub.add_parser('calibrate');a.add_argument('--study',default='runs/bottleneck_v4_s42')
    a.add_argument('--cases-per-family',type=int,default=8);a.add_argument('--gif',action='store_true')
    for command in ('bootstrap','normal','export','attack','status'):
        a=sub.add_parser(command);a.add_argument('--study',default='runs/bottleneck_v4_s42')
        if command=='bootstrap':
            a.add_argument('--episodes',type=int,default=64);a.add_argument('--epochs',type=int,default=40)
        if command=='normal':
            a.add_argument('--max-rounds',type=int);a.add_argument('--preview',action='store_true')
        if command=='attack':a.add_argument('--jobs',type=int,default=2)
    a=sub.add_parser('evaluate');a.add_argument('--checkpoint',required=True)
    a.add_argument('--output',required=True);a.add_argument('--split',choices=('selection','test'),default='selection')
    a.add_argument('--cases-per-family',type=int,default=24);a.add_argument('--gif',action='store_true')
    a.add_argument('--intervention',choices=('normal','zero','stale','drop'),default='normal')
    a.add_argument('--drop-probability',type=float,default=.3)
    a=sub.add_parser('report');a.add_argument('--runs',required=True);a.add_argument('--output',required=True)
    a.add_argument('--split',choices=('selection','test'),default='test')
    args=p.parse_args()
    if args.command=='normal' and args.max_rounds is not None and args.max_rounds<1:
        raise ValueError('max-rounds must be positive')
    if args.command=='prepare':
        print(json.dumps(staged.prepare(staged.path(args.output),args.seed,args.normal_steps,args.attack_steps,task=task()),indent=2));return
    if args.command=='evaluate':
        from ..research import evaluate_checkpoint,load_policy
        config,_,_,_=load_policy(staged.path(args.checkpoint))
        if config.task.research_protocol!=PROTOCOL:raise ValueError('not a v4 checkpoint')
        report=evaluate_checkpoint(staged.path(args.checkpoint),staged.path(args.output),split=args.split,
            per_family=args.cases_per_family,gif=args.gif,intervention=args.intervention,
            drop_probability=args.drop_probability)
        print(json.dumps(report['overall'],indent=2));return
    if args.command=='report':
        from ..research import report_study
        report_study(staged.path(args.runs),staged.path(args.output),args.split,protocol=PROTOCOL);return
    root=staged.path(args.study);spec,cfg=staged.study_config(root)
    if cfg.task.research_protocol!=PROTOCOL:raise ValueError('not a bottleneck study')
    receipt=root/'calibration.json'
    if args.command=='calibrate':
        from ..training.locking import run_lock
        with run_lock(root/'study_controller'):
            if receipt.exists():
                result=json.loads(receipt.read_text())
                if result['source_sha256']!=source_digest() or result['config_sha256']!=cfg.digest():
                    raise ValueError('calibration identity changed')
                if result.get('cases_per_family')!=args.cases_per_family:
                    raise ValueError('existing calibration case count differs; use a fresh study')
            else:
                result,frames=calibrate(cfg,args.cases_per_family,args.gif)
                staged.write(receipt,result)
                if frames:
                    from PIL import Image
                    images=[Image.fromarray(f) for f in frames]
                    images[0].save(root/'calibration.gif',save_all=True,append_images=images[1:],duration=200,loop=0)
            print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
            if not result['passed']:raise RuntimeError('calibration failed; inspect reports before training')
        return
    if args.command!='status':
        if not receipt.exists():raise ValueError('run calibrate first')
        r=json.loads(receipt.read_text())
        if not r['passed'] or r['source_sha256']!=source_digest() or r['config_sha256']!=cfg.digest():
            raise ValueError('valid same-source bottleneck calibration required')
        if r.get('cases_per_family',0)<cfg.gate_cases_per_family:
            raise ValueError('calibration must cover at least the declared gate case count')
    forward=[args.command,'--study',str(root)]
    if args.command=='bootstrap':forward+=['--episodes',str(args.episodes),'--epochs',str(args.epochs)]
    if args.command=='normal':
        if args.max_rounds:forward+=['--max-rounds',str(args.max_rounds)]
        if args.preview:forward+=['--preview']
    if args.command=='attack':forward+=['--jobs',str(args.jobs)]
    staged.main(forward)

if __name__=='__main__':main()
