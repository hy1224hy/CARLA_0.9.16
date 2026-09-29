"""Read durable artifacts only; never connect to or advance CARLA."""
import collections
from datetime import datetime
import json
import math

from utils.r8c1_training_session import RUN, OPS, ROOT, pid_alive, now, write_json


def inspect():
    manifest = json.loads((RUN / "checkpoint_manifest.json").read_text())
    boundary = manifest["latest"]["update"]
    issues = []
    def rows(name):
        result = []
        for i,line in enumerate((RUN / name).read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.append(f"{name} incomplete/invalid line {i+1}; excluded")
                continue
            if row["update"] <= boundary:
                result.append(row)
        return result
    metrics, episodes = rows("metrics.jsonl"), rows("episodes.jsonl")
    keys = [(r["update"],r["episode_index"]) for r in episodes]
    if len(keys) != len(set(keys)):
        issues.append("duplicate completed episode keys")
    if len(episodes) != boundary*16:
        issues.append(f"completed episode count {len(episodes)} != {boundary*16}")
    if [r["update"] for r in metrics] != list(range(1,boundary+1)):
        issues.append("metrics/checkpoint boundary mismatch or duplicate/gap; retry if writer is committing")
    window=metrics[-20:]
    last=window[-1]
    s=last["scalars"]
    for r in window:
        for k,v in r["scalars"].items():
            if isinstance(v,(int,float)) and not math.isfinite(v):
                issues.append(f"u{r['update']} nonfinite {k}")
            if ("preupdate_max_abs_ratio_error" in k or "max_abs_step_residual" in k) and abs(v)>1e-6:
                issues.append(f"u{r['update']} {k}={v}")
            if k == "train/update_skipped" and v:
                issues.append(f"u{r['update']} skipped update")
    rates={reason:sum(r['scalars'][f'episode/{reason}_rate'] for r in window)/len(window)
           for reason in ('success','collision','offroad','road_violation','timeout')}
    all_scenarios=sorted(p.name for p in (ROOT/'data/training_scenarios_town05_r8a_v1').glob('*.json'))
    counts=collections.Counter(r['scenario_name'] for r in episodes)
    coverage={name:counts[name] for name in all_scenarios}
    session=json.loads((OPS/'session.json').read_text())
    alive=pid_alive(session['trainer_pid'])
    previous_path=OPS/'monitoring'/'inspection_latest.json'
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
    snapshot={"at":now(),"safe_update":boundary,"completed_episodes":len(episodes),
        "trainer_alive":alive,"session":session,"numeric_or_integrity_issues":issues,
        "last_20_update_range":[window[0]['update'],last['update']],"last_20_team_rates":rates,
        "stage_index":s['curriculum/stage_index'],"mastery":last['audits'].get('r8a_scenario_mastery'),
        "scenario_counts":coverage,"unseen_scenarios":[k for k,v in coverage.items() if not v],
        "latest_agent_and_optimizer_metrics":{k:v for k,v in s.items() if k.startswith('curriculum/agent_') or
            any(t in k for t in ('approx_kl','post_tanh_entropy','entropy_coefficient','explained_variance','preupdate_max_abs_ratio_error','max_abs_step_residual','learning_rate'))},
        "updates_since_previous_inspection":boundary-previous['safe_update'] if previous else None,
        "previous_inspection_at":previous['at'] if previous else None,
        "interpretation":"Training diagnostics only; prior evaluation repeatability HOLD remains."}
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    write_json(OPS/'monitoring'/f'inspection_{stamp}.json',snapshot)
    write_json(previous_path,snapshot)
    print(json.dumps(snapshot,ensure_ascii=False,indent=2))
    return snapshot


if __name__=='__main__': inspect()
