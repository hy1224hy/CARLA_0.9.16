"""Shared IDM demonstrations and audited, qualified normal-driving initialization."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ..config import Config, Task
from ..common.contract import inputs
from ..common.outcomes import OutcomeTracker
from ..environment.conditions import HighwayConditionBackend
from ..models.team import Team
from ..evaluation.paired import evaluate_paired
from .checkpoint import PROTOCOL, read, save, source_digest

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_driver(config, reports, threshold=.875):
    if config.task.attack_weight != 0 or config.communication or config.curriculum:
        raise ValueError('driver must be normal-driving G0 without an attack curriculum')
    for split in ('gate','selection'):
        for family, metrics in reports[split]['families'].items():
            if (metrics['success_rate'] < threshold or metrics['bad_rate'] > .0625
                    or metrics['baseline_invalid_count']):
                raise ValueError(f'driver not qualified: {split}/{family}; inspect driving qualification report')

def qualify(model, config, cases=8):
    from ..evaluation.runner import evaluate
    return {split:evaluate(model,config.task,split,cases,gamma=config.gamma)[0]
            for split in ('gate','selection')}

def demonstrations(config, episodes):
    if type(episodes) is not int or episodes<1:
        raise ValueError('demonstration episodes must be positive')
    if config.task.research_protocol=='bottleneck_paired_attack_v4':
        from ..bottleneck.driving import demonstrations as bottleneck_demonstrations
        return bottleneck_demonstrations(config,episodes)
    rng=np.random.default_rng(config.seed+52013)
    rows, actions, seeds, families = [], [], [], []
    for episode in range(episodes):
        family=('parallel','closing')[episode%2]
        seed=int(rng.integers(0,900000)); seeds.append(seed); families.append(family)
        backend=HighwayConditionBackend(config.task,'C1',family,seed,gamma=config.gamma)
        try:
            # Training-only recovery examples: no gate/selection/test initial states.
            for vehicle in backend.road.vehicles[2:]:
                vehicle.position[1]+=float(rng.uniform(-.4,.4))
                vehicle.heading+=float(rng.uniform(-.05,.05))
                vehicle.speed=float(rng.uniform(8,20))
            backend.frame=backend.env.snapshot(backend.road,time=0)
            backend.tracker=OutcomeTracker(config.task); backend.tracker.update(backend.frame)
            while backend.frame.time < config.task.duration-1e-6 and not backend.tracker.bad:
                obs,_,_=inputs(backend.frame,config.task)
                backend.road.act()
                command=np.asarray([[v.action['acceleration']/5,v.action['steering']/.25]
                                    for v in backend.road.vehicles[2:]],np.float32)
                rows.append(obs); actions.append(np.clip(command,-1,1))
                backend.step()
        finally:
            backend.close()
    return {'obs':np.asarray(rows,np.float32),'actions':np.asarray(actions,np.float32),
            'seeds':np.asarray(seeds,dtype=np.int64),'families':np.asarray(families)}

def bootstrap(config, output, *, episodes=64, epochs=40):
    if epochs<1 or type(epochs) is not int:
        raise ValueError('BC epochs must be positive')
    config.validate()
    if config.communication or config.task.attack_weight or config.curriculum:
        raise ValueError('bootstrap requires normal no-communication G0')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.threads); torch.manual_seed(config.seed)
    data=demonstrations(config,episodes)
    np.savez_compressed(output/'demonstrations.npz',**data)
    obs=torch.as_tensor(data['obs']); target=torch.as_tensor(data['actions'])
    model=Team(config)
    # Turning communication on later must not perturb the pretrained action mean.
    with torch.no_grad():
        for actor in model.actors:
            actor[0].weight[:,model.obs_dim:]=0
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    links=torch.zeros((len(obs),2,1))
    history=[]
    for epoch in range(epochs):
        losses=[]
        for idx in torch.randperm(len(obs)).split(256):
            prediction=model.distribution(obs[idx],links[idx]).loc.tanh()
            loss=((prediction-target[idx]).square()*torch.tensor([1.,20.])).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
        if (epoch+1)%10==0:
            print(f'BC epoch={epoch+1}/{epochs} loss={history[-1]["loss"]:.6f}',flush=True)
    model.eval()
    reports=qualify(model,config)
    (output/'qualification.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    preparation={'method':'IDM behavior cloning with training-only recovery perturbations',
        'demonstration_decisions':len(obs),'demonstration_agent_samples':2*len(obs),
        'demonstration_episodes':episodes,'dataset_sha256':file_hash(output/'demonstrations.npz'),
        'epochs':epochs,'bc_updates':epochs*((len(obs)+255)//256),'history':history,
        'source_sha256':source_digest(),'config':config.to_dict()}
    (output/'preparation.json').write_text(json.dumps(preparation,indent=2),encoding='utf-8')
    validate_driver(config,reports)
    state={'protocol':PROTOCOL,'config':config.to_dict(),'source_sha256':source_digest(),
           'models':{t:model.state_dict() for t in ('A','B')},'train_steps':0,
           'driving_preparation':preparation}
    save(output/'driver.pt',state)
    return preparation | {'qualified':True,'driver':str(output/'driver.pt')}

def export_driver(normal_run, destination):
    normal_run=Path(normal_run); destination=Path(destination)
    state=read(normal_run/'selected.pt')
    raw=state['config']; cfg=Config(**(raw | {'task':Task(**raw['task'])})).validate()
    if state['source_sha256'] != source_digest() or not state['selection']['eligible']:
        raise ValueError('normal selected model is not eligible or source changed')
    model=Team(cfg).eval(); model.load_state_dict(state['model'])
    reports=qualify(model,cfg)
    destination.with_suffix('.qualification.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    validate_driver(cfg,reports)
    summary=json.loads((normal_run/'summary.json').read_text())
    if summary['status']!='complete':
        raise ValueError('finish the declared normal-driving budget before exporting')
    manifest=json.loads((normal_run/'manifest.json').read_text())
    prep={'normal_checkpoint_sha256':file_hash(normal_run/'selected.pt'),
          'normal_training_steps':summary['train_steps'],'selected_at_steps':state['selection']['train_steps'],
          'bootstrap':(manifest.get('initialization') or {}).get('driving_preparation'),
          'qualification':{s:{f:m['success_rate'] for f,m in r['families'].items()} for s,r in reports.items()}}
    save(destination,{'protocol':PROTOCOL,'config':cfg.to_dict(),'source_sha256':source_digest(),
        'models':{t:state['model'] for t in ('A','B')},'train_steps':summary['train_steps'],
        'driving_preparation':prep})
    return prep
