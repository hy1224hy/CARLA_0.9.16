"""Restore the tested red corridor; retain all symmetric interaction templates."""
import argparse
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build_cases(source=None):
    source=Path(source) if source is not None else ROOT/'data/mut_mappo/interaction_small_v4/source_v3_matrix.json'
    result=copy.deepcopy(json.loads(source.read_text(encoding='utf-8')))
    result['case_set_version']='interaction_small_v4'
    result['geometry_status']='offline_map_checked_live_pending'
    result['scope']='weak-red diagnostic; red original corridor, no filtering of directions or mirrors'
    result['revision_reason']='v3 moved both red start and goal 8m; real C0/C1 screen exposed natural failures'
    result['revision_protocol']={'red_geometry':'original base vnext rolling geometry',
        'rear_gap_m':8.,'front_gaps_m':[10.,11.,12.,13.],
        'goal_acceptance':'unchanged 3m x 1.6m half extents','qualification_thresholds':'unchanged',
        'topologies':'unchanged sequential and crossed, four directions, complete lateral mirror'}
    for case in result['cases']:
        side='a0l' if case['red0_left'] else 'a0r'
        case['base_scenario']='data/training_scenarios_town05_base_vnext_v1/scenario_bd1_{}_rolling_z00_{}.json'.format(case['direction'],side)
        case['geometry_status']='candidate_not_live_validated'
        for spec in case['blue_spawn_offsets'].values():
            if spec['forward_m']<0: spec['forward_m']=-8.
    return result


def main():
    import carla
    from environment.mut_mappo_env import MutMAPPOEnv
    from utils.build_mut_mappo_symmetric_cases_v2 import DEFAULT_XODR
    from training.checkpoint_protocol import atomic_write_json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--xodr',type=Path,default=DEFAULT_XODR)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/mut_mappo/interaction_small_v4')
    args=parser.parse_args()
    result=build_cases(args.source_manifest)
    env=MutMAPPOEnv.__new__(MutMAPPOEnv)
    env.map=carla.Map('Town05',args.xodr.read_text(encoding='utf-8'))
    env.red_ids=(0,1); env.blue_ids=(2,3); env.num_agents=4; env.default_max_steps=500
    for case in result['cases']: env._resolve_case(case)
    target=args.output_dir/'cases.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=result:
        raise FileExistsError('preserve prior case manifest')
    atomic_write_json(result,target)
    print(json.dumps({'status':'OFFLINE_MAP_CHECKED_NOT_LIVE_VALIDATED','cases':len(result['cases']),
        'splits':{k:len(v) for k,v in result['splits'].items()},'carla_client_created':False,'output':str(target)}))


if __name__=='__main__': main()
