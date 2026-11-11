"""Five-seed matrix with references reused rather than duplicate runs."""
import hashlib
import json
from pathlib import Path
from .config import StudyConfig


def definitions(seeds=(42,43,44,45,46),steps=500000,optional=False):
    rows=[]
    def make(level,seed,tag,arm,**changes):
        cfg=StudyConfig.default(level,seed,arm=arm,run_id=f'L{level}_{tag}_s{seed}',**changes)
        cfg.task.update(total_steps=steps,warmup_steps=min(20000,steps//10),ramp_steps=min(20000,steps//10),eval_cases=32)
        rows.append(cfg);return cfg
    for level in [1,2,3]:
        for seed in seeds:
            make(level,seed,'PPO','main')
            a=make(level,seed,'A2C','main',algorithm='a2c');a.task.update(lr=7e-4,rollout_steps=128,epochs=1,gae_lambda=1.)
            t=make(level,seed,'TRPO','main',algorithm='trpo');t.task.update(entropy_coef=0.,lane_entropy_coef=0.)
            make(level,seed,'MUT','main',mutual='both')
            for beta in [.005,.08]:make(level,seed,'beta_'+str(beta).replace('.','p'),'beta',mutual='both',beta=beta)
            if optional:make(level,seed,'SELFKL','optional',self_kl=.02)
            if level<3:
                make(level,seed,'ONEWAY','mechanism',mutual='one_way')
                make(level,seed,'ACC_ONLY','mechanism',mutual='both',lane_multiplier=0.)
                make(level,seed,'LANE_ONLY','mechanism',mutual='both',acc_multiplier=0.)
            if optional and level==2:
                cfg=make(level,seed,'NO_WARMUP','optional',mutual='both');cfg.task.update(warmup_steps=0,ramp_steps=0)
                cfg=make(level,seed,'REFRESH5','optional',mutual='both');cfg.task['refresh_rounds']=5
            components=['style','impact','geometry','lane_change'] if level==1 else ['risk','event_and_terminal','geometry'] if level==2 else []
            for component in components:
                cfg=make(level,seed,'no_'+component,'reward',mutual='both',drop_components=[component] if component not in ('risk','event_and_terminal') else [])
                if component=='risk':cfg.task['risk_weight']=0.
                if component=='event_and_terminal':cfg.task['event_weight']=0.
    for cfg in rows:cfg.validate()
    return rows


def tuning_definitions(seeds=(32,33),steps=100000):
    rows=[]
    for level in [1,2,3]:
        for seed in seeds:
            for algorithm,mutual,tag in [('ppo','off','PPO'),('a2c','off','A2C'),('trpo','off','TRPO'),('ppo','both','MUT')]:
                for value in ([.005,.01,.02] if algorithm=='trpo' else [1e-4,3e-4,7e-4]):
                    cfg=StudyConfig.default(level,seed,algorithm=algorithm,mutual=mutual,arm='tuning',
                        run_id=f'L{level}_{tag}_tune{value}_s{seed}')
                    cfg.task.update(total_steps=steps,warmup_steps=10000,ramp_steps=10000)
                    if algorithm=='trpo':cfg.trpo_delta=value
                    else:cfg.task['lr']=value
                    if algorithm=='a2c':cfg.task.update(rollout_steps=128,epochs=1,gae_lambda=1.)
                    if algorithm=='trpo':cfg.task.update(entropy_coef=0.,lane_entropy_coef=0.)
                    cfg.validate();rows.append(cfg)
    return rows


def prepare(root,seeds=(42,43,44,45,46),steps=500000,optional=False,tuning=False,tuning_selection=None):
    from .runtime import digest,write_json
    if len(set(seeds))!=len(seeds) or not seeds or any(s<0 for s in seeds):raise ValueError('distinct nonnegative seeds required')
    if steps<2 or steps%2:raise ValueError('positive even A+B budget required')
    root=Path(root).resolve()
    if root.exists():raise FileExistsError('Use a new study root; frozen configs are never overwritten')
    root.mkdir(parents=True);rows=tuning_definitions(seeds,steps) if tuning else definitions(seeds,steps,optional);runs=[]
    package=Path(__file__).resolve().parent
    for path in package.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md','.ps1'):
            destination=root/'control'/'tooling_source'/'highway_research'/path.relative_to(package)
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
    selected=json.loads(Path(tuning_selection).read_text())['selected'] if tuning_selection else None
    for cfg in rows:
        if selected is not None:
            tag='A2C' if cfg.algorithm=='a2c' else 'TRPO' if cfg.algorithm=='trpo' else 'MUT' if cfg.mutual!='off' else 'PPO'
            choice=selected[f'L{cfg.level}_{tag}']
            if tag=='TRPO':cfg.trpo_delta=choice['trpo_delta']
            else:cfg.task['lr']=choice['lr']
            cfg.validate()
        path=root/'configs'/cfg.arm/(cfg.run_id+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        write_json(path,cfg.to_dict())
        runs.append(dict(run_id=cfg.run_id,arm=cfg.arm,level=cfg.level,seed=cfg.seed,label=cfg.label,
            config=str(path),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),output=str(root/'train'/cfg.arm/cfg.run_id)))
    plan=dict(schema='research1_matrix_v1',seeds=list(seeds),steps=steps,max_jobs=4,source_sha256=digest(),
        run_count=len(runs),training_decisions=len(runs)*steps,runs=runs,
        reuse=dict(beta_zero='main/PPO',beta_002='main/MUT',full_reward='main/MUT',full_mutual='main/MUT'),
        execution='deterministic primary; stochastic audits separate with fixed action seeds',
        policy_statistics='average A/B per training seed; sample SD over five seeds',
        stage_order=['tuning'] if tuning else ['main','beta','mechanism','reward','optional'],
        stage0='verify L1 learnability, L2 event validity, L3 physical collision before full study',
        tuning='freeze hyperparameters on separate dev seeds before main; use equal tuning budgets per algorithm')
    if tuning_selection:plan['tuning_selection_sha256']=hashlib.sha256(Path(tuning_selection).read_bytes()).hexdigest()
    write_json(root/'study.json',plan)
    return plan


def recommend_tuning(root,output):
    """Choose on development only, then export explicit settings for a new plan."""
    import statistics
    root=Path(root);plan=json.loads((root/'study.json').read_text());candidates={}
    if plan['stage_order']!=['tuning']:raise ValueError('not a tuning study')
    for run in plan['runs']:
        cfg=json.loads(Path(run['config']).read_text());group=run['run_id'].rsplit('_s',1)[0]
        values=[]
        for team in ['A','B']:
            path=root/'audits'/'development'/'latest'/'nominal'/'deterministic_a0'/run['run_id']/team/'metrics.json'
            if not path.exists():raise FileNotFoundError('Missing tuning audit '+str(path))
            values.append(json.loads(path.read_text())['overall'])
        candidates.setdefault(group,[]).append((cfg,{k:statistics.mean(v[k] for v in values) for k in ['attack_success_rate','collision_rate','grade_violation_rate','invalid_rate']}))
    best={}
    for group,rows in candidates.items():
        if len(rows)!=len(plan['seeds']):raise ValueError('incomplete tuning seeds')
        cfg=rows[0][0];level=cfg['task']['level']
        tag='TRPO' if cfg['algorithm']=='trpo' else 'A2C' if cfg['algorithm']=='a2c' else 'MUT' if cfg['mutual']!='off' else 'PPO'
        scores={k:statistics.mean(r[1][k] for r in rows) for k in rows[0][1]}
        collision=scores['collision_rate'] if level<3 else 0.
        eligible=max(collision,scores['grade_violation_rate'],scores['invalid_rate'])<=.125
        rank=(int(eligible),scores['attack_success_rate'] if eligible else -collision-scores['grade_violation_rate']-scores['invalid_rate'],scores['attack_success_rate'])
        key=f'L{level}_{tag}'
        setting=dict(trpo_delta=cfg['trpo_delta']) if tag=='TRPO' else dict(lr=cfg['task']['lr'])
        if key not in best or rank>best[key][0]:best[key]=(rank,dict(setting,chosen_group=group,development_scores=scores))
    if len(best)!=12:raise ValueError('Expected all 12 level/algorithm settings')
    result=dict(selected={k:v[1] for k,v in best.items()},note='development-only tuning; n=2 normally, not evidence of superiority')
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8');return result
