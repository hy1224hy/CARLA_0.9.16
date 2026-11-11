"""Freeze available evidence without altering the running matrix."""
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import statistics
import time

output=Path(__file__).resolve().parent
study=output.parent


def read_rows(path):
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows


def drawdown(values):
    if not values:return 0.
    peak=values[0];drop=0.
    for value in values:
        peak=max(peak,value);drop=max(drop,peak-value)
    return drop


inventory=[];complete=[];sources=Counter()
for arm in ('main','beta','reward'):
    for config in sorted((study/'configs'/arm).glob('*.json')):
        cfg=json.loads(config.read_text(encoding='utf-8'))
        run=study/'train'/arm/config.stem
        summaries=run/'summary.json'
        summary=json.loads(summaries.read_text(encoding='utf-8')) if summaries.exists() else {}
        metrics=read_rows(run/'metrics.jsonl')
        entry=dict(arm=arm,run=config.stem,config=cfg,
            status=summary.get('status','updating' if metrics else 'not_started'),
            steps=summary.get('train_steps',max((r['train_steps'] for r in metrics),default=0)))
        if metrics:entry['seconds_since_metric_update']=time.time()-(run/'metrics.jsonl').stat().st_mtime
        inventory.append(entry)
        if entry['status']!='complete':continue
        manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
        sources[manifest['source_sha256']]+=1
        payload=(run/'selected.pt').read_bytes()
        target=output/'checkpoints'/config.stem/'selected.pt'
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        selection=summary.get('selection') or {}
        validation=read_rows(run/'validation.jsonl')
        episodes=read_rows(run/'episodes.jsonl')
        selected_team=selection.get('team')
        selected_metrics=[r for r in metrics if r['train_steps']==selection.get('train_steps') and r['team']==selected_team]
        late=[r['reports'][selected_team]['overall']['attack_success_rate'] for r in validation
              if r['train_steps']>=.75*cfg['total_steps']]
        diag=Counter()
        for episode in episodes:diag.update(episode.get('cut_diag',{}))
        complete.append(dict(arm=arm,run=config.stem,config=cfg,summary=summary,
            checkpoint_sha256=hashlib.sha256(payload).hexdigest(),source_sha256=manifest['source_sha256'],
            selected_optimizer=selected_metrics[-1] if selected_metrics else None,
            final_validation=validation[-1] if validation else None,
            late_selected_lineage=dict(values=late,mean=statistics.mean(late) if late else None,
                sd=statistics.stdev(late) if len(late)>1 else None,drawdown=drawdown(late)),
            training_episodes=dict(cases=len(episodes),diagnostics=dict(diag),
                **{key:sum(float(e.get(key,0) or 0) for e in episodes)/max(1,len(episodes))
                   for key in ('attack_success','collision','invalid','grade_violation','cut_count','blocking_time')})))
stamp=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
record=dict(snapshot_time_beijing=stamp,inventory=inventory,completed=complete,source_groups=dict(sources),
            note='Partial snapshot; selection metrics are not independent final evaluations')
(output/'evidence.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(dict(time=stamp,statuses=dict(Counter(r['status'] for r in inventory)),
                     completed=len(complete),sources=dict(sources))),flush=True)
