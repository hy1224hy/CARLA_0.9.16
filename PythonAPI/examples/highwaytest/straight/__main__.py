"""Independent one-attacker experiment; run from PythonAPI/examples."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def write_plan(output, steps=200000, seeds=(42,43,44)):
    from .config import StraightConfig
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not seeds or len(set(seeds))!=len(seeds):
        raise ValueError('provide distinct training seeds')
    rows=[]
    for level in (1,2,3):
        for method in ('independent','mutual'):
            for seed in seeds:
                config=StraightConfig(level=level,method=method,seed=seed,total_steps=steps,
                    warmup_steps=min(20000,steps//10),ramp_steps=min(20000,steps//10))
                config.validate()
                rows.append((f'L{level}_{method}_s{seed}.json',config.to_dict()))
    output.mkdir(parents=True)
    paths=[]
    for name,config in rows:
        path=output/name
        path.write_text(json.dumps(config,indent=2),encoding='utf-8')
        paths.append(path)
    return paths


def load_config(path):
    from .config import StraightConfig
    return StraightConfig(**json.loads(Path(path).read_text(encoding='utf-8-sig'))).validate()


def check_matrix(directory,steps=None,seeds=None):
    paths=sorted(Path(directory).glob('*.json'))
    if not paths:raise ValueError('no configuration files')
    configs=[load_config(p) for p in paths]
    if len({c.total_steps for c in configs})!=1:raise ValueError('unmatched total budgets')
    if any(c.eval_cases!=32 for c in configs):raise ValueError('frozen selection contract requires eval_cases=32; regenerate configs')
    if steps is not None and any(c.total_steps!=steps for c in configs):
        raise ValueError('existing matrix disagrees with requested steps; use a new study')
    if seeds is not None:
        expected={(level,method,seed) for level in (1,2,3) for method in ('independent','mutual') for seed in seeds}
        actual=[(c.level,c.method,c.seed) for c in configs]
        if set(actual)!=expected or len(actual)!=len(expected):
            raise ValueError('existing matrix disagrees with requested seeds/groups')
    return {'configs':len(configs),'steps':configs[0].total_steps,'seeds':sorted({c.seed for c in configs})}


def write_study_plan(output,study,steps,seeds):
    from .config import StraightConfig
    if study=='main':return write_plan(output,steps,seeds)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('distinct seeds required')
    configs=[]
    for seed in seeds:
        base=StraightConfig(level=2,seed=seed,total_steps=steps,
            warmup_steps=min(20000,steps//10),ramp_steps=min(20000,steps//10))
        if study=='beta':
            for beta in (.005,.08):
                configs.append((f'L2_beta{beta:g}_s{seed}',replace(base,beta=beta,method='mutual')))
        elif study=='reward':
            for name,changes in (('no_risk',{'risk_weight':0.}),('no_event',{'event_weight':0.})):
                configs.append((f'L2_{name}_s{seed}',replace(base,**changes)))
        else:raise ValueError('unknown study')
    for _,cfg in configs:cfg.validate()
    output.mkdir(parents=True)
    paths=[]
    for name,cfg in configs:
        path=output/(name+'.json');path.write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8');paths.append(path)
    return paths


def parser():
    p=argparse.ArgumentParser(description='One attacker vs frozen IDM+MOBIL; graded twin PPO')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bootstrap',help='ordinary-driving BC initialization; no attack training')
    b.add_argument('--output',required=True); b.add_argument('--config')
    b.add_argument('--seed',type=int,default=42)
    b.add_argument('--episodes',type=int,default=128); b.add_argument('--epochs',type=int,default=40)
    b.add_argument('--attack-level',type=int,choices=(1,2,3),help='second-stage scripted attack BC: attack_L<level>.pt in the same output dir')
    b.add_argument('--repertoire',action='store_true',help='shared multi-behavior repertoire BC: repertoire.pt in the same output dir')
    t=sub.add_parser('train')
    t.add_argument('--config',required=True); t.add_argument('--output',required=True)
    t.add_argument('--initialize-from'); t.add_argument('--resume'); t.add_argument('--max-rounds',type=int)
    e=sub.add_parser('evaluate')
    e.add_argument('--checkpoint',required=True); e.add_argument('--split',choices=('selection','test'),default='test')
    e.add_argument('--cases',type=int,default=None); e.add_argument('--output')
    e.add_argument('--gif',action='store_true',help='first-case actual trajectory and GIF')
    m=sub.add_parser('plan',help='generate balanced configs only')
    m.add_argument('--output',default='highwaytest/configs/straight_v1')
    m.add_argument('--steps',type=int,default=200000); m.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    m.add_argument('--study',choices=('main','beta','reward'),default='main')
    check=sub.add_parser('check',help='validate matrix before bootstrap or launch')
    check.add_argument('--configs',required=True); check.add_argument('--steps',type=int)
    check.add_argument('--seeds',type=int,nargs='+')
    b=sub.add_parser('batch',help='bounded subprocess workers; automatically resumes latest.pt')
    b.add_argument('--configs',required=True); b.add_argument('--output',required=True)
    b.add_argument('--jobs',type=int,default=1); b.add_argument('--initialize-from-dir')
    r=sub.add_parser('report')
    r.add_argument('--runs',required=True,nargs='+'); r.add_argument('--output',required=True)
    pool=sub.add_parser('pool',help='archive qualified representatives using selection only')
    pool.add_argument('--runs',required=True); pool.add_argument('--output',required=True)
    mx=sub.add_parser('matrix',help='training-intent x actual-behavior consistency matrix (main runs only)')
    mx.add_argument('--runs',required=True); mx.add_argument('--output',required=True)
    c=sub.add_parser('calibrate',help='scripted development probes; does not demonstrate learning')
    c.add_argument('--output',required=True); c.add_argument('--cases',type=int,default=8)
    return p


def batch(args):
    if not 1<=args.jobs<=8:
        raise ValueError('jobs must be between 1 and 8')
    paths=sorted(Path(args.configs).glob('*.json'))
    if not paths:
        raise ValueError('no configuration files')
    configs=[load_config(p) for p in paths]
    if len({c.total_steps for c in configs})!=1:
        raise ValueError('batch requires matched A+B training budgets')
    output=Path(args.output).resolve()
    initializers={}
    if args.initialize_from_dir:
        for c in configs:
            # Shared normal-driving initialization for all levels and algorithms:
            # levels differ only through the reward (frozen design decision).
            path=Path(args.initialize_from_dir).resolve()/f'base_s{c.seed}'/'driver.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
            initializers[(c.seed,c.level)]=path
    output.mkdir(parents=True,exist_ok=True)
    logdir=output/'batch_logs'; logdir.mkdir(exist_ok=True)
    def job(item):
        path,c=item; run=output/path.stem
        cmd=[sys.executable,'-m','highwaytest.straight','train','--config',str(path.resolve()),'--output',str(run)]
        latest=run/'latest.pt'
        if latest.exists(): cmd+=['--resume',str(latest)]
        elif (c.seed,c.level) in initializers: cmd+=['--initialize-from',str(initializers[(c.seed,c.level)])]
        with (logdir/(path.stem+'.log')).open('a',encoding='utf-8') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=False)
        return {'run':path.stem,'exit_code':result.returncode}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results=list(pool.map(job,zip(paths,configs)))
    (output/'batch_status.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(x['exit_code'] for x in results):
        raise RuntimeError('batch failures; inspect batch_logs and batch_status.json')
    return results


def calibrate(output,cases):
    from .config import StraightConfig
    from .environment import StraightEnv,scripted_action
    if cases<1: raise ValueError('cases must be positive')
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    rows=[]
    for level in (1,2,3):
        for kind in ('normal','block','cut','collision'):
            for case in range(cases):
                env=StraightEnv(StraightConfig(level=level))
                try:
                    env.reset(seed=4000000+case)
                    total=0.
                    while True:
                        _,reward,term,trunc,info=env.step(scripted_action(env,kind))
                        total+=float(reward)
                        if term or trunc: break
                    rows.append(dict(level=level,script=kind,case=case,return_=total,
                        **{k:info[k] for k in ('success','collision','invalid','grade_violation','baseline_valid') if k in info}))
                finally: env.close()
    output.parent.mkdir(parents=True,exist_ok=True)
    report={'purpose':'development scripts, not learned policies or algorithm advantage','episodes':rows}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    return report


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='plan': result=[str(p) for p in write_study_plan(args.output,args.study,args.steps,args.seeds)]
    elif args.command=='check': result=check_matrix(args.configs,args.steps,args.seeds)
    elif args.command=='batch': result=batch(args)
    elif args.command=='calibrate': result=calibrate(args.output,args.cases)
    elif args.command in ('report','pool','matrix'):
        from .reporting import report,build_pool,matrix as run_matrix
        result={'report':report,'pool':build_pool,'matrix':run_matrix}[args.command](args.runs,args.output)
    else:
        from . import runner
        if args.command=='train':
            result=runner.train(load_config(args.config),args.output,initialize_from=args.initialize_from,
                resume=args.resume,max_rounds=args.max_rounds)
        elif args.command=='bootstrap':
            from .config import StraightConfig
            cfg=replace(load_config(args.config),level=0,seed=args.seed) if args.config else StraightConfig(level=0,seed=args.seed)
            result=runner.bootstrap(cfg,args.output,episodes=args.episodes,epochs=args.epochs,attack_level=args.attack_level,repertoire=args.repertoire)
        elif args.command=='evaluate':
            result=runner.evaluate(args.checkpoint,split=args.split,cases=args.cases,output=args.output,gif=args.gif)
        else: result=runner.report(args.runs,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    if args.command=='bootstrap' and not result.get('qualified'):
        raise SystemExit('Driver qualification failed; diagnostic checkpoint is not a valid attack initializer.')


if __name__=='__main__': main()
