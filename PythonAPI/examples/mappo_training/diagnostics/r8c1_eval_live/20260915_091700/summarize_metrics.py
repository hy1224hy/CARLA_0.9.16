"""Read-only observation of durable episode evidence; never connects to CARLA."""
import datetime
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
state = json.loads((ROOT / 'matrix_status.json').read_text(encoding='utf-8'))
run_summaries = []
episode_rows = []
for run in state['runs']:
    path = Path(run['output_dir']) / 'episodes.jsonl'
    records = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # A writer may currently be completing its next line.
    counts = Counter(record['outcome'] for record in records)
    run_summaries.append({key: run[key] for key in ('eval_id', 'profile', 'update', 'repeat', 'status')} |
                         {'episodes': len(records), 'outcomes': dict(counts)})
    for record in records:
        row = {key: record[key] for key in ('case_id', 'scenario_name', 'scenario_group', 'direction', 'outcome', 'episode_length')}
        row.update(eval_id=run['eval_id'], update=run['update'], repeat=run['repeat'], profile=run['profile'])
        row['agents'] = {}
        for agent_id, agent in record['agents'].items():
            d, a = agent['diagnostics'], agent['actions']
            row['agents'][agent_id] = {
                'reached': agent['reached_target'], 'active_steps': agent['active_decision_steps'],
                'final_completion': d['route_completion']['last'], 'speed_mps': d['speed_mps']['mean'],
                'episode_return': agent['episode_return'], 'throttle_mean': a['throttle_mean'],
                'brake_mean': a['brake_mean'], 'brake_mode_rate': a['brake_mode_rate'],
                'coast_mode_rate': a['coast_mode_rate'], 'steer_mean': a['steer_mean'],
                'pace_eligible_rate': d['pace_deficit_eligible']['mean'],
                'pace_reward_sum': agent['reward_component_sums']['free_flow_pace_deficit'],
                'reward_residual_max_abs': agent['reward_residual_max_abs'],
                'road_reason': agent['road_legality']['violation_reason'],
                'offroad': agent['offroad_terminated'],
            }
        episode_rows.append(row)

groups = []
for update, group, repeat in sorted({(r['update'], r['scenario_group'], r['repeat']) for r in episode_rows}):
    records = [r for r in episode_rows if (r['update'], r['scenario_group'], r['repeat']) == (update, group, repeat)]
    agents = {}
    for agent_id in ('0', '1'):
        values = [r['agents'][agent_id] for r in records]
        steps = sum(v['active_steps'] for v in values)
        agents[agent_id] = {
            'reached_count': sum(v['reached'] for v in values),
            'mean_final_completion': sum(v['final_completion'] for v in values) / len(values),
            **{name: sum(v[name] * v['active_steps'] for v in values) / steps
               for name in ('speed_mps', 'brake_mode_rate', 'coast_mode_rate', 'pace_eligible_rate')},
        }
    groups.append({'update': update, 'scenario_group': group, 'repeat': repeat,
                   'case_count': len(records), 'outcomes': dict(Counter(r['outcome'] for r in records)), 'agents': agents})
report = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'batch_status': state['status'],
          'evaluation_purpose': 'diagnostic_only', 'selection_allowed': False, 'performance_decision': 'NOT_EVALUATED',
          'completed_probe_episodes': len(episode_rows), 'smoke_episodes': 2,
          'runs': run_summaries, 'groups_separated_by_repeat': groups, 'episodes': episode_rows,
          'repeatability': state['comparisons']}
temporary = ROOT / 'metric_snapshot.tmp'
temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
temporary.replace(ROOT / 'metric_snapshot.json')
print(json.dumps({'status': state['status'], 'completed_probe_episodes': len(episode_rows), 'runs': run_summaries}, ensure_ascii=False))
