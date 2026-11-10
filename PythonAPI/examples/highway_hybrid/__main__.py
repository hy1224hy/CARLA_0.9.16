"""Run from PythonAPI/examples: python -m highway_hybrid --help."""
import argparse
import json
from pathlib import Path


def write_plan(output,steps,seeds,levels,methods,beta):
    from .config import HybridConfig
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Distinct seeds required')
    rows=[]
    for level in levels:
        for method in methods:
            for seed in seeds:
                cfg=HybridConfig(level=level,method=method,seed=seed,total_steps=steps,beta=beta,
                                 warmup_steps=min(20000,steps//10),ramp_steps=min(20000,steps//10)).validate()
                rows.append((f'L{level}_{method}_s{seed}.json',cfg))
    if len({name for name,_ in rows})!=len(rows):raise ValueError('Duplicate plan groups')
    output.mkdir(parents=True)
    for name,cfg in rows:(output/name).write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
    return [str(output/name) for name,_ in rows]


def probe(level,cases,seed,output):
    import numpy as np
    import torch
    from .config import HybridConfig
    from .model import HybridActorCritic
    from .environment import HybridEnv
    from highway_eval.evaluate import case_contract,summarize
    from .runner import source_digest
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);torch.manual_seed(seed)
    cfg=HybridConfig(level=level,seed=seed).validate();model=HybridActorCritic(cfg.hidden,cfg.state_dim)
    env=HybridEnv(cfg);rows=[]
    try:
        with torch.no_grad():
            for case in case_contract('development',cases):
                obs,_=env.reset(seed=case['seed'],options={'family':case['family']})
                torch.manual_seed(seed+case['seed']);parts={};total=0.;changes=0;stable=env._lane(env.attacker)
                boundary_time=0.;last_time=0.;first_cut=None
                while True:
                    acceleration,lane,_,_=model.sample(torch.as_tensor(obs['state']),torch.as_tensor(obs['lane_mask']))
                    obs,reward,term,trunc,info=env.step(dict(acceleration=acceleration.numpy(),lane=int(lane)))
                    total+=reward
                    for key,value in info['reward_components'].items():parts[key]=parts.get(key,0.)+value
                    current=env._lane(env.attacker)
                    if abs(env.attacker.position[1]-4*current)>1.5:boundary_time+=info['time']-last_time
                    last_time=info['time']
                    if first_cut is None and info['cut_count']:first_cut=info['time']
                    if current!=stable and abs(env.attacker.position[1]-4*current)<.5 and abs(env.attacker.heading)<np.pi/18:
                        changes+=1;stable=current
                    if term or trunc:break
                label=('invalid' if info['invalid'] else 'contact' if info['collision'] else
                       'effective_cut' if info['cut_count'] else 'grade_violation' if info['grade_violation'] else
                       'stop' if info['stop_qualified'] else 'mild_block' if info['blocking_time']>=1.5 else
                       'no_qualified_attack')
                rows.append(dict(info,case_seed=case['seed'],completed_lane_changes=changes,
                    boundary_fraction=boundary_time/max(info['time'],1e-9),first_effective_cut_time=first_cut,
                    contact_time=info['time'] if info['collision'] else None,behavior_label=label,
                    reward_components=parts,**{'return':total}))
    finally:env.close()
    result=dict(note='Untrained stochastic policy coverage; not learned-policy performance',
                protocol='highway_hybrid_twin_v1',config=cfg.to_dict(),source_sha256=source_digest(),
                overall=summarize(rows),episodes=rows)
    output.mkdir(parents=True);(output/'probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return dict(note=result['note'],overall=result['overall'])


def parser():
    p=argparse.ArgumentParser(description='Hybrid acceleration/lane PPO with same-task twin mutual learning')
    sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan')
    plan.add_argument('--output',required=True);plan.add_argument('--steps',type=int,default=200000)
    plan.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    plan.add_argument('--levels',type=int,nargs='+',choices=(1,2,3),default=[1,2,3])
    plan.add_argument('--methods',nargs='+',choices=('independent','mutual'),default=['independent','mutual'])
    plan.add_argument('--beta',type=float,default=.02)
    train=sub.add_parser('train');train.add_argument('--config',required=True);train.add_argument('--output',required=True)
    train.add_argument('--resume');train.add_argument('--max-rounds',type=int)
    audit=sub.add_parser('evaluate')
    audit.add_argument('--runs',required=True);audit.add_argument('--output',required=True)
    audit.add_argument('--split',choices=('development','test'),default='development')
    audit.add_argument('--cases',type=int,default=100)
    audit.add_argument('--checkpoint',choices=('selected','latest'),default='selected')
    audit.add_argument('--team',choices=('A','B'),default='A')
    smoke=sub.add_parser('smoke');smoke.add_argument('--output',required=True)
    smoke.add_argument('--level',type=int,choices=(1,2,3),default=2);smoke.add_argument('--steps',type=int,default=512)
    smoke.add_argument('--method',choices=('independent','mutual'),default='mutual')
    coverage=sub.add_parser('probe');coverage.add_argument('--output',required=True)
    coverage.add_argument('--level',type=int,choices=(1,2,3),default=2)
    coverage.add_argument('--cases',type=int,default=12);coverage.add_argument('--seed',type=int,default=42)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='plan':
        result=write_plan(args.output,args.steps,args.seeds,args.levels,args.methods,args.beta)
    elif args.command=='probe':result=probe(args.level,args.cases,args.seed,args.output)
    else:
        from .config import HybridConfig
        from .runner import train,evaluate_suite
        if args.command=='train':
            cfg=HybridConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
            result=train(cfg,args.output,args.resume,args.max_rounds)
        elif args.command=='evaluate':
            result=evaluate_suite(args.runs,args.output,args.split,args.cases,args.checkpoint,args.team)
            result=[dict(run=r['run'],team=r['team'],steps=r['checkpoint_train_steps'],overall=r['overall']) for r in result]
        else:
            root=Path(args.output)
            if root.exists():raise FileExistsError(root)
            cfg=HybridConfig(level=args.level,method=args.method,total_steps=args.steps,rollout_steps=128,
                minibatch=64,epochs=2,hidden=64,duration=4.,eval_cases=4,eval_interval=256,warmup_steps=0,ramp_steps=0).validate()
            summary=train(cfg,root/'train')
            audit=evaluate_suite(root/'train',root/'evaluation','development',4,'latest','B')
            result=dict(note='Pipeline smoke only; not effectiveness evidence',training=summary,evaluation=audit[0]['overall'])
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
