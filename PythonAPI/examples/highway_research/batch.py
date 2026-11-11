"""Bounded workers, resume, status and auditable subprocess failures."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .runtime import digest,load,check,write_json
from highwaytest.training.locking import run_lock


def recover_temp(run):
    """A failed atomic replace may leave a complete newer temporary checkpoint."""
    run=Path(run);temp=run/'latest.tmp';target=run/'latest.pt'
    if not temp.exists():return
    with run_lock(run):
        state=load(temp);check(state)
        if target.exists() and state['train_steps']<=load(target)['train_steps']:return
        if state['config']!=json.loads((run/'config.json').read_text()):raise ValueError('temp config mismatch')
        backup=run/f'recovery_{state["train_steps"]}.pt'
        if not backup.exists():backup.write_bytes(temp.read_bytes())
        os.replace(temp,target)


def command(row):
    args=[sys.executable,'-X','utf8','-u','-m','highway_research','train','--config',row['config'],'--output',row['output']]
    if (Path(row['output'])/'latest.pt').exists():args.append('--resume')
    return args


def run_batch(root,arms=('main',),jobs=4,retries=1,dry_run=False):
    root=Path(root).resolve();plan=json.loads((root/'study.json').read_text())
    if not 1<=jobs<=4:raise ValueError('jobs must be 1..4')
    if plan['source_sha256']!=digest():raise ValueError('Frozen study source changed; prepare a new version')
    unknown=set(arms)-set(plan['stage_order'])
    if unknown:raise ValueError('unknown arms '+str(unknown))
    for r in plan['runs']:
        if hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()!=r['config_sha256']:raise ValueError('config changed '+r['run_id'])
    rows=[r for arm in plan['stage_order'] if arm in arms for r in plan['runs'] if r['arm']==arm]
    if dry_run:return [command(r) for r in rows]
    with run_lock(root/'control'/'batch'):
        results=[];status_path=root/'control'/'batch_status.json';status_path.parent.mkdir(parents=True,exist_ok=True)
        def execute(row):
            run=Path(row['output']);summary=run/'summary.json'
            if summary.exists() and json.loads(summary.read_text())['status']=='complete':
                state=load(run/'latest.pt');check(state)
                if state['config']!=json.loads(Path(row['config']).read_text()):raise ValueError('completed config mismatch')
                return dict(run=row['run_id'],status='complete',skipped=True)
            log=root/'logs'/row['arm']/(row['run_id']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            for attempt in range(retries+1):
                if run.exists():recover_temp(run)
                args=command(row)
                if run.exists() and not (run/'latest.pt').exists():raise RuntimeError('partial run without checkpoint: '+str(run))
                with log.open('a',encoding='utf-8') as stream:
                    stream.write('\nATTEMPT '+str(attempt+1)+' '+json.dumps(args)+'\n');stream.flush()
                    process=subprocess.run(args,cwd=Path(__file__).resolve().parents[1],stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if process.returncode==0 and summary.exists() and json.loads(summary.read_text())['status']=='complete':
                    return dict(run=row['run_id'],status='complete',attempts=attempt+1)
                if 'another process holds this run' in log.read_text(encoding='utf-8'):break
            return dict(run=row['run_id'],status='failed',log=str(log))
        for arm in plan['stage_order']:
            selected=[r for r in rows if r['arm']==arm]
            if not selected:continue
            write_json(status_path,dict(stage=arm,total=len(rows),finished=len(results),results=results))
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                futures={pool.submit(execute,r):r for r in selected}
                for future in as_completed(futures):
                    try:result=future.result()
                    except Exception as error:result=dict(run=futures[future]['run_id'],status='failed',error=str(error))
                    results.append(result);write_json(status_path,dict(stage=arm,total=len(rows),finished=len(results),results=results))
            if any(r['status']=='failed' for r in results):raise RuntimeError('Stage failed; see '+str(status_path))
        write_json(status_path,dict(stage='complete',results=results));return results
