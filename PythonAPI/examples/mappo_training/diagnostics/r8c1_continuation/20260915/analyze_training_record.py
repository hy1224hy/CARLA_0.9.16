import json
from pathlib import Path
from collections import Counter
import statistics

root=Path.cwd()
run=root/'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224'
boundary=json.loads((run/'checkpoint_manifest.json').read_text())['latest']['update']
rows=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
episodes=[json.loads(l) for l in (run/'episodes.jsonl').read_text().splitlines()]
complete=[e for e in episodes if e['update']<=boundary]
def summarize(lo,hi,names=None):
    es=[e for e in complete if lo<=e['update']<=hi and (names is None or int(e['scenario_name'][9:11]) in names)]
    selected=[r for r in rows if lo<=r['update']<=hi]
    result={'n':len(es),'outcomes':{k:sum(e[k] for e in es) for k in ['success','collision','offroad','road_violation','timeout']}}
    result['agents']={}
    for aid in ['0','1']:
        a={}
        for field in ['agent_mean_speed_mps','agent_brake_mode_rate','agent_mean_completion']:
            total=weight=0
            for r in selected:
                for name,w in r['audits']['r8a_scenario_mastery']['scenario_windows'].items():
                    if names is not None and int(name[9:11]) not in names: continue
                    matched=[e for e in es if e['update']==r['update'] and e['scenario_name']==name]
                    n=len(matched) if field=='agent_mean_completion' else sum(e['agents'][aid]['active_decision_steps'] for e in matched)
                    total+=w[field][aid]*n
                    weight+=n
            a[field]=total/weight if weight else None
        a['offroad_terminated']=sum(e['agents'][aid]['offroad_terminated'] for e in es)
        a['road_rule_terminated']=sum(e['agents'][aid]['road_rule_terminated'] for e in es)
        a['truncated']=sum(e['agents'][aid]['truncated'] for e in es)
        result['agents'][aid]=a
    result['episode_length_median']=statistics.median(e['episode_length'] for e in es) if es else None
    return result

report={'boundary':boundary,'complete_episodes':len(complete),'incomplete_prefix_count':len(episodes)-len(complete)}
report['windows']={label:summarize(lo,hi,names) for label,lo,hi,names in [
 ('S0A_late',136,155,None),('S0B_all',156,186,None),('S1_all',187,208,None),
 ('old_short_before',167,186,[1,2,3,4]),('old_short_after',187,208,[1,2,3,4]),
 ('S1_early',187,197,None),('S1_late',198,208,None)]}
report['S1_scenarios']={str(n):summarize(187,208,[n]) for n in range(1,9)}
report['policy_windows']={}
for label,lo,hi in [('early',1,20),('before_long',167,186),('long_early',187,197),('long_late',198,208)]:
    rs=[r for r in rows if lo<=r['update']<=hi]
    keys=[k for k in rs[-1]['scalars'] if any(s in k for s in ['deterministic_longitudinal_mean','stall_reportable/nonzero_rate','pace_deficit_eligible/nonzero_rate','explained_variance','post_tanh_entropy','approx_kl'])]
    report['policy_windows'][label]={k:statistics.mean(r['scalars'][k] for r in rs) for k in keys}
report['health']={'skipped':sum(r['scalars']['train/update_skipped'] for r in rows if r['update']<=boundary),
 'max_ratio_error':max(v for r in rows if r['update']<=boundary for k,v in r['scalars'].items() if 'preupdate_max_abs_ratio_error' in k),
 'max_reward_residual':max(r['scalars']['reward_audit/max_abs_step_residual'] for r in rows if r['update']<=boundary)}
out=root/'diagnostics/r8c1_continuation/20260915/training_record_analysis.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
