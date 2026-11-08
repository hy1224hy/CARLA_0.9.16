"""Offline R4 diagnostics on registered evaluation traces; never opens CARLA."""
import argparse
import json
from pathlib import Path
from evaluation.r4_risk_metrics import RiskTracker, response_measures


def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def response_reference(rows):
    previous={};acc={};result={}
    for row in rows:
        step=row.get('step',0)
        if not step:continue
        result[str(step)]={}
        for rid in (0,1):
            snap=row.get('kinematics',{}).get(str(rid))
            if snap is not None:
                if rid in previous:
                    m=response_measures(previous[rid],snap,.1,acc.get(rid))
                    result[str(step)][str(rid)]=m;acc[rid]=m['acceleration_mps2']
                previous[rid]=snap
    return result


def analyze(rows,reference=None):
    tracker=RiskTracker()
    for row in rows:
        step=row.get('step',0)
        if not step:continue
        active={int(k):v for k,v in row['active_mask'].items()}
        snapshots={int(k):v for k,v in row['kinematics'].items()}
        safety={int(k):v for k,v in row.get('mut_metrics',{}).get('safety',{}).get('per_agent',{}).items()}
        tracker.step(step,snapshots,active,safety,reference)
    return tracker.metrics()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();base=args.root/'base_u0/evaluation'
    baseline={}
    for row in records(base/'cases.jsonl'):
        trace=records(base/row['trace_path'])
        baseline[(row['team'],row['case_key'])]=response_reference(trace)
    result=[]
    for directory in [base,*sorted(args.root.glob('R3*/evaluation'))]:
        path=directory/'cases.jsonl'
        if not path.exists():continue
        for row in records(path):
            # Preserve invalid evidence; never quietly turn missing measurements into zero.
            item=dict(version=directory.parent.name,team=row['team'],case_key=row['case_key'],valid=row['valid'])
            if row['valid']:
                try:item['risk']=analyze(records(directory/row['trace_path']),baseline.get((row['team'],row['case_key'])))
                except (ValueError,KeyError) as e:item.update(measurement_status='UNAVAILABLE',error=str(e))
            result.append(item)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol='r4_risk_v1',records=result,
        caveats=['Projected OBB clearances; not exact polygon distance.','Response residuals are time-aligned associations, not causal effects.',
        'Engineering gap thresholds: front 2+v+closing^2/6; side .75+.05v. Not empirically calibrated.',
        'Lane-relative deviation not measured; raw derivative response proxies can be noisy.']),indent=2),encoding='utf-8')
    print('R4 offline records:',len(result))


if __name__=='__main__':main()
