"""Small real end-to-end check; never starts the formal 500k matrix."""
import hashlib
from pathlib import Path
from .matrix import definitions
from .runtime import digest,write_json
from .batch import run_batch
from .evaluate import audit
from .report import build_report
from .visualize import replay


def smoke(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);runs=[]
    for cfg in definitions(seeds=[0],steps=64):
        if cfg.arm!='main':continue
        cfg.task.update(hidden=16,rollout_steps=32,minibatch=32,epochs=1,eval_cases=2,eval_interval=64,warmup_steps=0,ramp_steps=0)
        cfg.critic_epochs=1;cfg.validate()
        path=root/'configs'/(cfg.run_id+'.json');path.parent.mkdir(exist_ok=True);write_json(path,cfg.to_dict())
        runs.append(dict(run_id=cfg.run_id,config=str(path),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            output=str(root/'train'/cfg.run_id),arm='main',level=cfg.level,seed=0,label=cfg.label))
    write_json(root/'study.json',dict(seeds=[0],steps=64,source_sha256=digest(),runs=runs,stage_order=['main'],
        note='SMOKE ONLY: 12 runs x64 decisions; not research results'))
    run_batch(root,jobs=2,retries=0)
    for row in runs:
        for team in ['A','B']:
            audit(row['output'],root/'audits'/'development'/'latest'/'nominal'/'deterministic_a0'/row['run_id']/team,cases=2,team=team)
    result=build_report(root,allow_partial=True)
    result['smoke_training_decisions']=len(runs)*64
    replay(runs[0]['output'],root/'smoke_replay.gif')
    return result
