"""Descriptive C1/C2 paired criticality reporting; no training or physics imports."""
from __future__ import annotations
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path
from evaluation.critical_interaction_metrics import analyze_trace, validate_protocol


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def average(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def paired_diagnostics(records, required_conditions=('C0','C1','C2'), primary_ttc_s=1.5, planned_teams=()):
    if not {'C1','C2'}.issubset(required_conditions):
        raise ValueError('Paired criticality requires C1 and C2')
    groups, teams = defaultdict(dict), set(planned_teams)
    for row in records:
        teams.add(row['team'])
        key=(row['team'],row['case_key'],row['seed'],row['paired_attempt'])
        if row['condition'] in groups[key]:
            raise ValueError(f'duplicate condition for paired identity {key}')
        groups[key][row['condition']]=row
    latest, excluded = {}, []
    for key, group in groups.items():
        if any(c not in group or group[c].get('valid') is not True for c in required_conditions):
            excluded.append(dict(team=key[0], case_key=key[1], seed=key[2], attempt=key[3],reason='incomplete_or_invalid_attempt'))
            continue
        identity=key[:3]
        if identity not in latest or key[3]>latest[identity][0]:
            latest[identity]=(key[3],group)
    result={}
    threshold=str(float(primary_ttc_s))
    for team in sorted(teams):
        pairs=[]
        for identity, (attempt, group) in latest.items():
            if identity[0]!=team: continue
            first, second=group['C1'],group['C2']
            metrics=[r.get('critical_interaction',{}) for r in (first,second)]
            if (any(m.get('status')!='COMPLETE' or type(m.get('critical_event')) is not bool for m in metrics)
                    or any(type(r.get('blue_eligible')) is not bool for r in (first,second))):
                excluded.append(dict(team=team,case_key=identity[1],seed=identity[2],attempt=attempt,reason='unmeasured_criticality_or_qualification'))
                continue
            h1,h2=[m['critical_event'] for m in metrics]
            g1,g2=first['blue_eligible'],second['blue_eligible']
            pairs.append(dict(case_key=identity[1], seed=identity[2], paired_attempt=attempt,
                raw_C1=int(h1),raw_C2=int(h2), qualified_C1=int(h1 and g1), qualified_C2=int(h2 and g2),
                blue_eligible_C1=g1,blue_eligible_C2=g2,
                qualification_only_improvement=h1==h2 and h2 and not g1 and g2,
                tet_C1=metrics[0].get('tet_s',{}).get(threshold),tet_C2=metrics[1].get('tet_s',{}).get(threshold),
                tit_C1=metrics[0].get('tit_s',{}).get(threshold),tit_C2=metrics[1].get('tit_s',{}).get(threshold),
                duration_C1=metrics[0].get('duration_s'),duration_C2=metrics[1].get('duration_s'),
                red_success_C1=first.get('red_success'),red_success_C2=second.get('red_success')))
        n=len(pairs)
        row=dict(n_measured_pairs=n,n_excluded_groups=sum(r['team']==team for r in excluded),
            raw_critical_rate_C1=average([r['raw_C1'] for r in pairs]),
            raw_critical_rate_C2=average([r['raw_C2'] for r in pairs]),
            qualified_critical_rate_C1=average([r['qualified_C1'] for r in pairs]),
            qualified_critical_rate_C2=average([r['qualified_C2'] for r in pairs]),
            delta_raw_critical_rate=average([r['raw_C2']-r['raw_C1'] for r in pairs]),
            delta_qualified_critical_rate=average([r['qualified_C2']-r['qualified_C1'] for r in pairs]),
            blue_eligible_rate_C1=average([int(r['blue_eligible_C1']) for r in pairs]),
            blue_eligible_rate_C2=average([int(r['blue_eligible_C2']) for r in pairs]),
            added_critical_cases=sum(r['raw_C2']>r['raw_C1'] for r in pairs),
            removed_critical_cases=sum(r['raw_C2']<r['raw_C1'] for r in pairs),
            qualification_only_improvements=sum(r['qualification_only_improvement'] for r in pairs),
            pairs=pairs)
        for name in ('tet','tit','duration'):
            row['mean_'+name+'_C1']=average([r[name+'_C1'] for r in pairs])
            row['mean_'+name+'_C2']=average([r[name+'_C2'] for r in pairs])
            row['mean_delta_'+name]=average([r[name+'_C2']-r[name+'_C1'] for r in pairs
                                            if r[name+'_C1'] is not None and r[name+'_C2'] is not None])
        result[team]=row
    return dict(teams=result,excluded=excluded, denominator='same case/team/seed/attempt, whole valid group, fully measured C1/C2; missing groups excluded and counted')


def write_csv(path, rows):
    keys=sorted({k for r in rows for k in r})
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def generate_report(evaluations, output_dir, protocol):
    protocol=validate_protocol(protocol)
    output_dir=Path(output_dir).resolve()
    reports,flat,episodes,event_rows=[],[],[],[]
    seen=set()
    for directory in evaluations:
        directory=Path(directory).resolve()
        if directory in seen: raise ValueError('duplicate evaluation input')
        seen.add(directory)
        manifest=read_json(directory/'manifest.json')
        if manifest.get('status')!='COMPLETED': raise ValueError(f'{directory}: evaluation incomplete')
        task=read_json(directory/'resolved_config.json')['task']
        if abs(float(task.get('timestep',.1))-protocol['dt'])>1e-9:
            raise ValueError('protocol dt differs from evaluation task timestep')
        rows=[json.loads(line) for line in (directory/'cases.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        label=Path(str(manifest.get('config_path',directory.name)).replace('\\','/')).stem
        if 'completed_attempted_episodes' in manifest and len(rows)!=manifest['completed_attempted_episodes']:
            raise ValueError('manifest/records attempted episode count mismatch')
        cohorts=defaultdict(set)
        for record in rows:
            cohorts[(record['team'],record['case_key'])].add(record['seed'])
        if any(len(seeds)>1 for seeds in cohorts.values()):
            raise ValueError('multiple evaluation seeds for one team/case; analyze each cohort separately')
        planned_keys=manifest.get('case_keys')
        if planned_keys is not None and any(r['case_key'] not in planned_keys for r in rows):
            raise ValueError('recorded case is not declared in manifest')
        analyzed=[]
        for original in rows:
            r=dict(original)
            if r.get('evidence')!='independent_evaluation' or r.get('execution_mode')=='stochastic_training':
                raise ValueError('only independent evaluation records are supported')
            if r.get('valid') is not True:
                r['critical_interaction']=dict(status='INVALID_EPISODE',critical_event=None)
            else:
                trace=(directory / r['trace_path']).resolve()
                if not trace.is_relative_to(directory): raise ValueError('trace escapes evaluation directory')
                if trace.is_file():
                    samples=[json.loads(line) for line in trace.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                    r['critical_interaction']=analyze_trace(samples,protocol)
                    identities = ('case_key','condition','team')
                    mismatch = any(any(k in sample and sample[k] != r[k] for k in identities) for sample in samples)
                    if samples and 'seed' in samples[0] and samples[0]['seed'] != r['seed']:
                        mismatch = True
                    if mismatch:
                        r['critical_interaction']=dict(status='TRACE_IDENTITY_MISMATCH',critical_event=None)
                    decisions=sum(sample.get('step',0)>0 for sample in samples)
                    if r.get('steps') is not None and decisions!=r['steps']:
                        r['critical_interaction']=dict(status='INCOMPLETE_TRACE',reason='case/trace step count mismatch',critical_event=None)
                else:
                    r['critical_interaction']=dict(status='MISSING_TRACE',reason=str(trace),critical_event=None)
            analyzed.append(r)
            m=r['critical_interaction']
            episode=dict(report=label,split=manifest.get('split'),team=r['team'],condition=r['condition'],
                case_key=r['case_key'],seed=r['seed'],paired_attempt=r['paired_attempt'],
                valid=r.get('valid'),blue_eligible=r.get('blue_eligible'),red_success=r.get('red_success'),
                status=m['status'],measurement_reason=m.get('reason'),
                pair_coverage_json=json.dumps(m.get('pair_coverage')),critical_event=m.get('critical_event'), min_ttc_s=m.get('min_ttc_s'),
                observed_steps=m.get('observed_steps'),duration_s=m.get('duration_s'),
                trace_path=r.get('trace_path'),
                red_end_reasons_json=json.dumps(r.get('red_end_reasons',r.get('agent_outcomes')),ensure_ascii=False),
                red_response_json=json.dumps(m.get('red_response'),ensure_ascii=False),
                blue_driving_json=json.dumps({str(i):r.get('driving_metrics',{}).get(str(i)) for i in protocol['blue_ids']},ensure_ascii=False),
                safety_json=json.dumps(r.get('safety'),ensure_ascii=False))
            for metric in ('tet_s','tit_s'):
                for threshold,value in m.get(metric,{}).items(): episode[metric+'_'+threshold]=value
            for aid in protocol['blue_ids']:
                driving=r.get('driving_metrics',{}).get(str(aid),{})
                for key in ('mean_speed_mps','abs_acceleration_p95_mps2','abs_jerk_p95_mps3',
                            'actual_steer_rate_p95_per_second','reference_lateral_p95_m','measured_state_fraction'):
                    episode[f'blue_{aid}_{key}']=driving.get(key)
            for aid,value in m.get('red_response',{}).items():
                for key in ('max_deceleration_mps2','hard_brake_duration_s','max_brake','acceleration_samples'):
                    episode[f'red_{aid}_{key}']=value.get(key)
            episodes.append(episode)
            for event in m.get('events',[]):
                event_rows.append(dict(report=label,split=manifest.get('split'),team=r['team'],condition=r['condition'],
                    case_key=r['case_key'],seed=r['seed'],paired_attempt=r['paired_attempt'],
                    **{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in event.items()}))
        paired=paired_diagnostics(analyzed,tuple(manifest.get('conditions',('C0','C1','C2'))),protocol['primary_ttc_s'],manifest.get('teams',()))
        for team, row in paired['teams'].items():
            if planned_keys is not None:
                present={r['case_key'] for r in rows if r['team']==team}
                missing=sorted(set(planned_keys)-present)
                row.update(n_planned_cases=len(planned_keys),n_missing_planned_cases=len(missing),
                    missing_planned_cases=missing,
                    measurement_coverage=row['n_measured_pairs']/len(planned_keys) if planned_keys else None)
                if row['n_measured_pairs']>len(planned_keys):
                    raise ValueError('multiple seed cohorts per planned case; do not pool repeated evaluations')
            flat.append(dict(report=label,split=manifest.get('split'),team=team,
                completed_round=manifest.get('checkpoint_completed_round'),**{k:v for k,v in row.items() if k not in ('pairs','missing_planned_cases')}))
        reports.append(dict(source=str(directory),manifest=manifest,paired=paired,
            cases_sha256=hashlib.sha256((directory/'cases.jsonl').read_bytes()).hexdigest(),
            cases=analyzed))
    payload=dict(schema_version='mut_mappo_critical_interactions_report_v1',protocol=protocol,
        protocol_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        reports=reports,team_rows=flat,
        limitations=[
            'TTC is a constant-velocity fixed-yaw 2D OBB prediction, not collision probability.',
            'Qualification is the existing strict collision/offroad/rule/task eligibility; not fault attribution or calibrated comfort.',
            'A higher qualified yield alone can be due only to improved blue qualification; compare raw yield and TET/TIT too.',
            'C1 may be an unsafe baseline. No stable-reference subset is inferred from C2 outcomes.',
            'Exposure durations stop at observed termination; no padding. Compare duration and termination with TET/TIT.',
            'Missing measurements are excluded and counted, never scored as zero; failed attempts are retained.',
            'A/B and repeated evaluation are dependent; no significance claim. Never pool different reds/cases/stages.',
            'Recorded response is temporally associated, not causally attributed. No PET or collision-fault model in v1.',
        ])
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'critical_report.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_csv(output_dir/'teams.csv',flat); write_csv(output_dir/'episodes.csv',episodes)
    write_csv(output_dir/'events.csv',event_rows)
    pairs=[dict(report=report['source'],team=team,**pair) for report in reports for team,row in report['paired']['teams'].items() for pair in row['pairs']]
    write_csv(output_dir/'pairs.csv',pairs)
    columns=['report','split','team','completed_round','n_measured_pairs','n_excluded_groups','n_missing_planned_cases','measurement_coverage',
             'raw_critical_rate_C1','raw_critical_rate_C2','qualified_critical_rate_C1','qualified_critical_rate_C2',
             'delta_raw_critical_rate','delta_qualified_critical_rate','qualification_only_improvements',
             'mean_delta_tet','mean_delta_tit']
    esc=lambda x: html.escape('' if x is None else str(round(x,5) if isinstance(x,float) else x))
    page='<!doctype html><meta charset="utf-8"><title>Critical interactions</title><style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}section{max-width:900px}</style><h1>Critical interaction diagnostics</h1>'
    page+='<p>Protocol SHA256: '+payload['protocol_sha256']+'</p><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in payload['limitations'])+'</ul>'
    page+='<table><tr>'+''.join('<th>'+c+'</th>' for c in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(r.get(c))+'</td>' for c in columns)+'</tr>' for r in flat)+'</table><section>'
    for row in flat:
        page+='<h3>'+esc(row['report'])+' / '+esc(row['team'])+' / '+esc(row['split'])+'</h3>'
        for metric,color in [('raw_critical_rate_C1','#999'),('raw_critical_rate_C2','#d66'),('qualified_critical_rate_C2','#69c')]:
            value=row[metric]
            page+='<p>'+metric+': '+esc(value)+'</p><div style="height:12px;background:'+color+';width:'+str(100*value if value is not None else 0)+'%"></div>'
    page+='</section>'
    (output_dir/'index.html').write_text(page,encoding='utf-8')
    return payload
