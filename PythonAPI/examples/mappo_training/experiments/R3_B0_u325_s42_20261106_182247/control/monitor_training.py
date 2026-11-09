"""Read-only observer for the active local run; no CARLA client or training mutation."""
import datetime,json,math,pathlib,statistics,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
RUN=ROOT/'run'
OUT=ROOT/'control/monitor'
OUT.mkdir(exist_ok=True)
INTERVAL=1200

def rows(path):
    result=[]
    if not path.exists():return result
    for line in path.read_text(encoding='utf-8').splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass
    return result

def avg(values):
    values=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    return statistics.mean(values) if values else None

def snapshot():
    records=[r for r in rows(RUN/'metrics.jsonl') if 'teams' in r]
    status=json.loads((RUN/'session/status.json').read_text())
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    report=dict(at=now.isoformat(),completed_updates=records[-1]['completed_round'] if records else 0,status=status,teams={})
    episodes=rows(RUN/'episodes.jsonl')
    for team in ('A','B'):
        windows={}
        for label,group in [('previous10',records[-20:-10]),('latest10',records[-10:])]:
            windows[label]={'updates':[r['completed_round'] for r in group]}
            for key in ('b','safe_critical_event','critical_exposure_s','critical_tet_s','critical_tit_s','red_success_rate','max_abs_reward_residual'):
                windows[label][key]=avg([r['training_window'][team].get(key) for r in group])
            for key in ('explained_variance','value_error_abs_mean','mutual_beta'):
                windows[label][key]=avg([r['teams'][team].get(key) for r in group])
            windows[label]['rewards']={key:avg([r['training_window'][team].get('reward_components_mean',{}).get(key) for r in group]) for key in ('base_soft','base_hard','individual_adversarial','task_violation')}
        latest=report['completed_updates']
        subset=[r for r in episodes if r.get('team')==team and max(1,latest-9)<=r.get('round',0)<=latest]
        windows['individual']={}
        for aid in ('2','3'):
            values=[r.get('individual_adversarial_metrics',{}).get(aid,{}) for r in subset]
            windows['individual'][aid]={key:avg([v.get(key) for v in values]) for key in ('qualified_primary_event_credit','critical_exposure_s','safe_critical_event')}
        report['teams'][team]=windows
    report['metrics_age_seconds']=time.time()-(RUN/'metrics.jsonl').stat().st_mtime
    text=json.dumps(report,ensure_ascii=False,indent=2)
    (OUT/'latest.json').write_text(text,encoding='utf-8')
    with (OUT/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report,ensure_ascii=False)+'\n')
    lines=['# R3 B0 training observation',report['at'],f"Completed: {report['completed_updates']}/325; status: {status.get('status')}",'','|Team|Window|Bad rate|Safe event|Exposure s|EV|Base soft|Base hard|Adversarial|','|---|---|---|---|---|---|---|---|---|']
    def fmt(v):return 'NA' if v is None else f'{v:.4f}'
    for team,data in report['teams'].items():
        for label in ('previous10','latest10'):
            d=data[label];v=[d.get(k) for k in ('b','safe_critical_event','critical_exposure_s','explained_variance')]+[d['rewards'][k] for k in ('base_soft','base_hard','individual_adversarial')]
            lines.append('|'+team+'|'+label+'|'+'|'.join(map(fmt,v))+'|')
    if report['metrics_age_seconds']>2400:lines+=['','Attention: metrics unchanged for over 40 minutes; inspect worker before taking action.']
    lines+=['','Read-only observation. No restarts, parameter changes, server ticks or automated training interventions. Individual event qualification is in latest.json.']
    (OUT/'latest.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report

while True:
    try:
        r=snapshot()
        terminal=r['status'].get('status','')
        if r['completed_updates']>=325 or terminal in ('SEGMENT_COMPLETED','FAILED','INTERRUPTED_SAFE_BOUNDARY_RETAINED'):
            break
        deadline=time.monotonic()+INTERVAL
        while time.monotonic()<deadline:time.sleep(min(30,max(0,deadline-time.monotonic())))
    except Exception as exc:
        with (OUT/'observer_errors.log').open('a',encoding='utf-8') as f:f.write(str(datetime.datetime.now())+' '+repr(exc)+'\n')
        time.sleep(30)
