"""File-only driving and actor-presence review of the completed diagnosis."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def build():
    root=ROOT/'experiments/adversarial_u1000_learning_diagnosis_20261104_v3'
    output=root/'analysis/case_review_20261104_v1'
    output.mkdir(exist_ok=False)
    rows=[];presence=Counter();training=Counter();hashes={}
    for group in ('N4','R4','R16'):
        episodes=[json.loads(x) for x in (root/'runs'/group/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        for c in episodes:
            states=c['case_geometry']['realized_initial_states']
            training['episodes']+=1
            training['initial_both_blue_present']+=all(str(i) in states for i in (2,3))
            training['invalid']+=not c['valid']
        initial={}
        for target in (0,12000,24000):
            path=root/'evaluations'/(group+'_frozen_'+str(target).zfill(6))/'analysis/critical_report.json'
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            report=read(path)['reports'][0]
            for c in report['cases']:
                states=c['case_geometry']['realized_initial_states']
                condition=c['condition'];presence[condition+'_episodes']+=1
                presence[condition+'_initial_both_blue_present']+=all(str(i) in states for i in (2,3))
                presence[condition+'_invalid']+=not c['valid']
            for team in ('A','B'):
                cases=[c for c in report['cases'] if c['team']==team and c['condition']=='C2']
                if target==0:initial[team]={c['case_key']:c for c in cases}
                roles={}
                for aid in ('2','3'):
                    failed=Counter();unknown=Counter()
                    for c in cases:
                        for k,v in c['interaction_quality']['driving_quality']['per_blue'][aid]['checks'].items():
                            if v is False:failed[k]+=1
                            if v is None:unknown[k]+=1
                    roles[aid]=dict(failed_checks_overlapping=dict(failed),unknown=dict(unknown),
                        outcomes=dict(Counter(c['agent_results'][aid]['end_reason'] for c in cases)))
                    for field in ('mean_speed_mps','above_preferred_speed_fraction','reference_lateral_p95_m','actual_steer_rate_p95_per_second'):
                        values=[c['driving_metrics'][aid].get(field) for c in cases]
                        roles[aid][field]=statistics.mean(v for v in values if v is not None)
                transitions={}
                for field in ('predicted_conflict','observed_contact_free_resolved'):
                    pairs=[(initial[team][c['case_key']]['interaction_quality'][field],c['interaction_quality'][field]) for c in cases]
                    transitions[field]=dict(added=sum(a is False and b is True for a,b in pairs),removed=sum(a is True and b is False for a,b in pairs))
                rows.append(dict(group=group,target=target,team=team,n=len(cases),roles=roles,
                    frozen_mean_return=statistics.mean(c['team_discounted_return'] for c in cases),
                    red_both_reached=sum(c['red_success'] for c in cases),
                    paired_to_initial=transitions,
                    blue_event_eligible=sum(c['blue_eligible'] for c in cases),
                    safety_overlapping={k:sum(c['safety'].get(k) is True for c in cases) for k in ('collision','offroad','road_rule','task_violation')}))
    result=dict(rows=rows,evaluation_presence=dict(presence),training_presence=dict(training),
        source_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
