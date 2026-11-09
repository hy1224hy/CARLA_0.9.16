"""Server entry: shared driving bootstrap -> normal PPO -> qualified attack curriculum."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from .config import ROOT, Config, Task, GROUPS, load
from .training.checkpoint import read, source_digest
from .training.driving import bootstrap, export_driver, file_hash
from .training.locking import run_lock
from .training.runner import run
from .experiments import batch

EXPLORATION={'initial':[.3,.08],'minimum':[.1,.03],'maximum':[.6,.2]}
COURSE={'weights':[.1,.3,1.],'min_steps':10000,'qualified':.875,'max_bad':.0625,'patience':2}

def path(value):
    p=Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()

def write(target, data):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)

def prepare(output, seed=42, normal_steps=100000, attack_steps=200000, *, task=None):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'runs'):
        raise ValueError('study must be under highwaytest/runs')
    if output.exists():
        raise FileExistsError('use a new study directory; existing studies are immutable')
    base=Config(seed=seed,total_train_steps=normal_steps,entropy_coef=.001,
        exploration=EXPLORATION,task=task or Task(research_protocol='route_relative_paired_attack_v3',attack_weight=0)).validate()
    attack=[replace(base,group=g,total_train_steps=attack_steps,
              task=replace(base.task,attack_weight=1),curriculum=COURSE).validate() for g in GROUPS]
    write(output/'configs/normal.json',base.to_dict())
    for cfg in attack:
        write(output/'configs/attack'/f'{cfg.group}_s{seed}.json',cfg.to_dict())
    study={'protocol':'staged_driving_attack_v1','seed':seed,'source_sha256':source_digest(),
        'normal_steps':normal_steps,'attack_steps_per_group':attack_steps,'groups':list(GROUPS),
        'curriculum':COURSE,'exploration':EXPLORATION,
        'budget_note':'shared BC data/updates + shared normal PPO + per-group attack A+B decisions; validation separate',
        'config_sha256':{'normal':base.digest(),**{c.group:c.digest() for c in attack}}}
    if base.task.research_protocol=='bottleneck_paired_attack_v4':
        study['protocol']='bottleneck_staged_v4'
    write(output/'study.json',study)
    return study

def study_config(root):
    spec=json.loads((root/'study.json').read_text(encoding='utf-8'))
    if spec['source_sha256'] != source_digest():
        raise ValueError('study source changed; restore the source or prepare a fresh study')
    for key,p in [('normal',root/'configs/normal.json'),
                  *((g,root/'configs/attack'/f'{g}_s{spec["seed"]}.json') for g in spec['groups'])]:
        if load(p).digest()!=spec['config_sha256'][key]:
            raise ValueError('study configuration changed: '+key)
    return spec,load(root/'configs/normal.json')

def run_normal(root, cfg, max_rounds=None, preview=False):
    output=root/'normal'
    checkpoint=output/'latest.pt'
    initial=root/'bootstrap/driver.pt'
    if not initial.exists():
        raise ValueError('run bootstrap first; only qualified BC models are published')
    manifest=output/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())['initialization']['sha256']
        if previous!=file_hash(initial):
            raise ValueError('normal-driving initialization changed')
    return run(cfg,output,resume=checkpoint if checkpoint.exists() else None,
               initialize_from=None if checkpoint.exists() else initial,max_rounds=max_rounds,preview=preview)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--output',default='runs/staged_v1_s42')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--normal-steps',type=int,default=100000)
    p.add_argument('--attack-steps',type=int,default=200000)
    for name in ('bootstrap','normal','export','attack','status'):
        p=sub.add_parser(name); p.add_argument('--study',default='runs/staged_v1_s42')
        if name=='bootstrap':
            p.add_argument('--episodes',type=int,default=64); p.add_argument('--epochs',type=int,default=40)
        if name=='normal':
            p.add_argument('--max-rounds',type=int); p.add_argument('--preview',action='store_true')
        if name=='attack':
            p.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args(argv)
    if args.command=='prepare':
        print(json.dumps(prepare(path(args.output),args.seed,args.normal_steps,args.attack_steps),indent=2)); return
    root=path(args.study)
    with run_lock(root/'study_controller'):
        spec,cfg=study_config(root)
        if args.command=='bootstrap':
            target=root/'bootstrap/driver.pt'
            if target.exists():
                saved=read(target)
                prep=saved['driving_preparation']
                if (saved['source_sha256']!=source_digest() or saved['config']!=cfg.to_dict()
                        or prep['demonstration_episodes']!=args.episodes or prep['epochs']!=args.epochs):
                    raise ValueError('existing bootstrap identity differs')
                result={'status':'already_qualified','driver':str(target)}
            else:
                result=bootstrap(cfg,root/'bootstrap',episodes=args.episodes,epochs=args.epochs)
        elif args.command=='normal':
            result=run_normal(root,cfg,args.max_rounds,args.preview)
        elif args.command=='export':
            target=root/'attack_initial.pt'
            if target.exists():
                saved=read(target)
                if (saved['source_sha256']!=source_digest() or saved['driving_preparation']['normal_checkpoint_sha256']
                        !=file_hash(root/'normal/selected.pt')):
                    raise ValueError('exported driving initialization changed')
                result={'status':'already_exported','sha256':file_hash(target)}
            else:
                result=export_driver(root/'normal',target)
        elif args.command=='attack':
            target=root/'attack_initial.pt'
            if not target.exists():
                raise ValueError('export a qualified normal driver first')
            saved=read(target)
            if saved['source_sha256']!=source_digest() or saved['config']!=cfg.to_dict():
                raise ValueError('attack initialization identity mismatch')
            result=batch(root/'configs/attack',root/'attack',args.jobs,initialize_from=target)
        else:
            result={'study':spec,'bootstrap_qualified':(root/'bootstrap/driver.pt').exists(),
                    'attack_initial_ready':(root/'attack_initial.pt').exists(),
                    'runs':{p.parent.relative_to(root).as_posix():json.loads(p.read_text())
                            for p in root.glob('**/summary.json')}}
        print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()
