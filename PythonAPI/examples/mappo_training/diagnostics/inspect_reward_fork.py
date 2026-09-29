"""File-only monitoring: no CARLA imports or simulator calls."""
import json
import math
from pathlib import Path
from collections import Counter
from statistics import mean
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/R8C1_RoutePaceV1_u356_to406_s42_20260927"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(episodes):
    successes = [e for e in episodes if e["success"]]
    return {"n": len(episodes),
        **{k: sum(bool(e[k]) for e in episodes) for k in ("success", "collision", "offroad", "road_violation", "timeout")},
        "agent0_offroad": sum(bool(e["agents"]["0"]["offroad_terminated"]) for e in episodes),
        "success_only_arrival_seconds": {
            aid: mean(e["agents"][aid]["active_decision_steps"] * .1 for e in successes) if successes else None
            for aid in ("0", "1")}}


def main():
    manifest = json.loads((RUN / "checkpoint_manifest.json").read_text())
    boundary = manifest["latest"]["update"]
    metrics = [m for m in rows(RUN / "metrics.jsonl") if 356 < m["update"] <= boundary][-20:]
    episodes = [e for e in rows(RUN / "episodes.jsonl") if metrics[0]["update"] <= e["update"] <= boundary]
    if json.loads((RUN / "checkpoint_manifest.json").read_text())["latest"] != manifest["latest"]:
        raise RuntimeError("checkpoint advanced during snapshot; rerun")
    if metrics[-1]["update"] != boundary or len(episodes) != len(metrics)*16:
        raise RuntimeError("commit window incomplete; reread after commit")
    scalars = [m["scalars"] for m in metrics]
    keys = [k for k in scalars[0] if k.startswith("curriculum/agent_") or
            any(x in k for x in ("approx_kl", "post_tanh_entropy", "explained_variance")) or
            (k.startswith("reward/agent_") and k.endswith("free_flow_pace_deficit/sum_per_episode_mean"))]
    integrity = {}
    for key in scalars[0]:
        if any(x in key for x in ("nonfinite", "update_skipped", "max_abs_step_residual", "preupdate_max_abs_ratio_error")):
            integrity[key] = max(abs(s[key]) for s in scalars)
    split = metrics[len(metrics)//2]["update"]
    by_scene = {}
    for name in sorted({e["scenario_name"] for e in episodes}):
        group = [e for e in episodes if e["scenario_name"] == name]
        by_scene[name] = {"all": summary(group),
            "early": summary([e for e in group if e["update"] < split]),
            "late": summary([e for e in group if e["update"] >= split])}
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "safe_update": boundary,
        "latest_checkpoint": manifest["latest"], "window": [metrics[0]["update"], boundary],
        "comparison_split": split, "outcomes": summary(episodes),
        "metric_means": {k: mean(s[k] for s in scalars) for k in keys},
        "metric_latest": {k: scalars[-1][k] for k in keys}, "integrity_maxima": integrity,
        "nonfinite_values": sum(not math.isfinite(v) for s in scalars for v in s.values() if isinstance(v,(int,float))),
        "stages": sorted({s["curriculum/stage_index"] for s in scalars}),
        "coverage": dict(Counter(e["scenario_name"] for e in episodes)), "by_scene": by_scene,
        "camera": json.loads((RUN / "session/camera.json").read_text()),
        "log_times": {p.name: datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()
                      for p in (RUN / "session").glob("train_*.log")}}
    target = RUN / "session/monitoring"
    target.mkdir(exist_ok=True)
    path = target / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json")
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"evidence":str(path), **report},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
