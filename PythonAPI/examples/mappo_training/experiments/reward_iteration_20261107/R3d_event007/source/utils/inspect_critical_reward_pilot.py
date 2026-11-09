"""File-only pilot progress. No simulator, PyTorch, ticks, or process mutation."""
import argparse
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def records(path):
    if not path.exists():return []
    # Journals are written concurrently. Only inspect committed newline records.
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def inspect(root):
    state=read(root/'control/pipeline_status.json')
    out={'scope':'FILE_ONLY_TRAINING_WINDOW_NOT_INDEPENDENT_EFFECTIVENESS',
         'pipeline':{k:state.get(k) for k in ('status','stage','stage_pid','worker_pid','error')} if state else None,
         'groups':{}}
    plan=read(root/'control/pipeline_plan.json')
    groups=list(dict.fromkeys(s['group'] for s in plan['stages'])) if plan else ['R1','R0']
    for group in groups:
        refs=read(root/'references'/group/'manifest.json')
        item={'reference_cases':len(refs['cases']) if refs else 0,
              'weak_reference_cases':sum(not c['qualified'] for c in refs['cases'].values()) if refs else 0}
        run=root/'runs'/group
        status=read(run/'session/status.json');manifest=read(run/'checkpoint_manifest.json')
        item['committed_round']=manifest.get('completed_round') if manifest else None
        if status:item['training_status']={k:status.get(k) for k in ('status','actual_counters','at','training_budget_reached')}
        rows=[r for r in records(run/'metrics.jsonl') if r.get('completed_round',0)>0][-5:]
        if rows:
            item['recent_training_rounds']=[r['completed_round'] for r in rows]
            item['teams']={}
            for team in ('A','B'):
                windows=[r['training_window'][team] for r in rows]
                values={key:statistics.mean(float(w[key]) for w in windows) for key in ('b','red_success_rate')}
                for key in ('safe_critical_event','safe_efficiency_utility'):
                    if all(key in w for w in windows):values[key]=statistics.mean(float(w[key]) for w in windows)
                values['max_reward_residual']=max(w['max_abs_reward_residual'] for w in windows)
                values['max_preupdate_ratio_error']=max(abs(float(v)) for r in rows
                    for v in r['teams'][team].get('preupdate_ratio_errors',{}).values())
                item['teams'][team]=values
        completion=read(root/'control'/f'{group}_gate_completed.json')
        report_path=root/'evaluations'/f'{group}_gate'/'analysis/critical_report.json'
        if completion:
            candidates=[p for p in completion['artifact_hashes'] if p.replace('\\','/').endswith('analysis/critical_report.json')]
            if len(candidates)==1:report_path=root/candidates[0]
        report=read(report_path) if completion else None
        if report:item['independent_evaluation']=report['team_rows']
        if state and state.get('stage','').startswith(group+'_gate') and '--output-dir' in state.get('stage_command',[]):
            command=state['stage_command']
            progress=read(Path(command[command.index('--output-dir')+1])/'evaluations/eval_000/manifest.json')
            if progress:item['evaluation_progress']={k:progress.get(k) for k in ('status','completed_attempted_episodes','planned_episodes')}
        out['groups'][group]=item
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/adversarial_u1000_critical_reward_20261101_v3'))
    args=parser.parse_args();root=args.root if args.root.is_absolute() else ROOT/args.root
    print(json.dumps(inspect(root),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
