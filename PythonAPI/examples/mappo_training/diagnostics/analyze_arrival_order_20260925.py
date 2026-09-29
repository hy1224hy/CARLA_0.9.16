"""Read-only arrival-order analysis; historical traces cannot prove u356 timing."""
import json
import statistics as st
from pathlib import Path

root=Path(__file__).resolve().parents[1]
run=root/'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224'
boundary=json.loads((run/'checkpoint_manifest.json').read_text())['latest']['update']
episodes=[json.loads(l) for l in (run/'episodes.jsonl').read_text().splitlines()]
result={'safe_update':boundary,'training':[],'historical_traces':[]}
for lo,hi in [(187,208),(337,356)]:
    rows=[r for r in episodes if lo<=r['update']<=min(hi,boundary)]
    for name in sorted({r['scenario_name'] for r in rows}):
        rr=[r for r in rows if r['scenario_name']==name]
        ss=[r for r in rr if r['success']]
        a=[r['agents']['0']['active_decision_steps'] for r in ss]
        b=[r['agents']['1']['active_decision_steps'] for r in ss]
        result['training'].append({'window':[lo,hi],'scene':name,'episodes':len(rr),'success':len(ss),
            'a0_first_success':sum(x<y for x,y in zip(a,b)),
            'a1_first_success':sum(y<x for x,y in zip(a,b)),
            'median_a0_arrival_s':st.median(a)*.1 if a else None,
            'median_a1_arrival_s':st.median(b)*.1 if b else None,
            'median_signed_gap_s':st.median([y-x for x,y in zip(a,b)])*.1 if a else None})
for path in sorted((root/'diagnostics/r8c1_behavior/20260919_v1/traces').glob('*.jsonl')):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    case=rows[0]['case']; ss=rows[1:]
    if case['scenario'] not in ['scene01','scene04','scene06']: continue
    arrival={a:next((s['step'] for s in ss if s['info']['reward_diagnostics'].get(a,{}).get('success_event')),None) for a in ['0','1']}
    motion={a:next((s['step'] for s in ss if s['info']['reward_diagnostics'].get(a,{}).get('speed_mps',0)>=.5),None) for a in ['0','1']}
    t=arrival['0']
    before=[s for s in ss if t and s['step']<=t and '1' in s['policy_action']]
    last=before[-1] if before else None
    result['historical_traces'].append({'case':case,'arrival_step':arrival,'first_motion_step':motion,
        'a1_moving_fraction_before_a0_arrival':sum(s['info']['reward_diagnostics']['1']['speed_mps']>=.5 for s in before)/len(before) if before else None,
        'a1_completion_at_a0_arrival':last['info']['reward_diagnostics']['1']['route_completion'] if last else None})
out=root/'diagnostics/r8c1_continuation/arrival_order_20260925.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
