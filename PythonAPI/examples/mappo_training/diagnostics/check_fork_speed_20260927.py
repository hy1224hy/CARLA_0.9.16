import json
from pathlib import Path
from statistics import mean
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
old = root/'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224'
new = root/'logs/R8C1_RoutePaceV1_u356_to406_s42_20260927'
boundary = json.loads((new/'checkpoint_manifest.json').read_text())['latest']['update']

def read(path):
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: continue  # a pending final episode is excluded by safe update
    return result

def summarize(run, lo, hi):
    episodes = [e for e in read(run/'episodes.jsonl') if lo <= e['update'] <= hi]
    metrics = [m for m in read(run/'metrics.jsonl') if lo <= m['update'] <= hi]
    assert len(episodes) == 16*(hi-lo+1) and len(metrics) == hi-lo+1
    scenes = sorted({e['scenario_name'] for e in episodes})
    outcome = ['success','collision','offroad','road_violation','timeout']
    groups = {name:[e for e in episodes if e['scenario_name']==name] for name in scenes}
    rates = {name:{'n':len(group), **{k:sum(e[k] for e in group) for k in outcome}} for name,group in groups.items()}
    scalars = [m['scalars'] for m in metrics]
    return {'window':[lo,hi], 'n':len(episodes),
        'counts':{k:sum(e[k] for e in episodes) for k in outcome},
        'scenario_equal_weight_rates':{k:mean(mean(e[k] for e in group) for group in groups.values()) for k in outcome},
        'agents':{str(a):{
            'mean_speed_kmh':3.6*mean(s[f'curriculum/agent_{a}/mean_speed_mps'] for s in scalars),
            'max_speed_kmh':3.6*max(s[f'diagnostic/agent_{a}/speed_mps/maximum'] for s in scalars),
            'overspeed_fraction_update_mean':mean(s[f'diagnostic/agent_{a}/overspeed_risk/nonzero_rate'] for s in scalars),
            'mean_overspeed_cost_per_step':mean(s[f'reward/agent_{a}/overspeed/mean_per_active_step'] for s in scalars),
            'mean_pace_cost_per_step':mean(s[f'reward/agent_{a}/free_flow_pace_deficit/mean_per_active_step'] for s in scalars),
        } for a in [0,1]},'by_scene':rates}

report = {'at':datetime.now(timezone.utc).isoformat(), 'safe_update':boundary,
          'old_last20':summarize(old,337,356),
          'new_last20':summarize(new,max(357,boundary-19),boundary),
          'new_first10':summarize(new,357,366),
          'new_last10':summarize(new,boundary-9,boundary)}
path=new/'session/monitoring'/('speed_safety_u'+str(boundary)+'.json')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for label, value in report.items():
    if isinstance(value,dict):
        print(label,json.dumps({k:v for k,v in value.items() if k!='by_scene'},ensure_ascii=False))
print('evidence',path)
