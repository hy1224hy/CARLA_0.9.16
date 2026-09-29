"""Offline, fixed-boundary comparison of the completed reward experiment."""
import json
from pathlib import Path
from statistics import mean
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'logs/R8C1_RoutePaceV1_u356_to406_s42_20260927'
OLD = ROOT / 'logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224'

def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def window(run, lo, hi):
    es = [x for x in rows(run/'episodes.jsonl') if lo <= x['update'] <= hi]
    ms = [x['scalars'] for x in rows(run/'metrics.jsonl') if lo <= x['update'] <= hi]
    assert len(es) == 16*(hi-lo+1) and len(ms) == hi-lo+1
    scenes = {}
    for name in sorted({x['scenario_name'] for x in es}):
        group = [x for x in es if x['scenario_name'] == name]
        ok = [x for x in group if x['success']]
        scenes[name] = {'n':len(group), 'success':len(ok),
            'collision':sum(x['collision'] for x in group),
            'offroad':sum(x['offroad'] for x in group),
            'arrival_success_only_s':[mean(x['agents'][str(a)]['active_decision_steps']*.1 for x in ok) if ok else None for a in (0,1)]}
    return {'window':[lo,hi], 'n':len(es),
        'counts':{k:sum(x[k] for x in es) for k in ('success','collision','offroad','road_violation','timeout')},
        'speed_kmh':[3.6*mean(x[f'curriculum/agent_{a}/mean_speed_mps'] for x in ms) for a in (0,1)],
        'pace_episode':[mean(x[f'reward/agent_{a}/free_flow_pace_deficit/sum_per_episode_mean'] for x in ms) for a in (0,1)],
        'critic_ev':mean(x['critic/explained_variance'] for x in ms),
        'coverage':dict(Counter(x['scenario_name'] for x in es)), 'scenes':scenes}

def main():
    report = {'baseline':window(OLD,337,356), 'all_new':window(NEW,357,456),
        'blocks':[window(NEW,lo,lo+19) for lo in range(357,457,20)]}
    output = NEW/'session/monitoring/comparison_u456.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for label, block in [('baseline',report['baseline'])]+[(str(b['window']),b) for b in report['blocks']]:
        print(label,json.dumps({k:v for k,v in block.items() if k not in ('scenes','coverage')},ensure_ascii=False))
    print('coverage',report['all_new']['coverage'])
    print('SCENES old -> final')
    for name, before in report['baseline']['scenes'].items():
        print(name,json.dumps({'old':before,'final':report['blocks'][-1]['scenes'][name]},ensure_ascii=False))
    print('evidence',output)

if __name__ == '__main__':
    main()
