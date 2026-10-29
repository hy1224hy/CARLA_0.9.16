"""Read existing files only; never import CARLA or alter experiment outputs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONTROL = ROOT / "diagnostics/base_driving_vnext_20261020/long_s42_u600_v1"
RUN = ROOT / "logs/BD_vNext_V6_scratch_s42_u600_20261020_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def report_summary(path):
    summaries = []
    for report in read(path).get("reports", []):
        reasons = Counter()
        for case in report.get("cases", []):
            for agent in case.get("agents", {}).values():
                reasons.update(agent.get("failed_checks", []))
        summaries.append({key: report.get(key) for key in (
            "case_count", "both_reached_fraction", "qualified",
            "qualification_decision", "evidence_status", "failed_checks",
            "checkpoint_sha256", "scope", "suite_role",
        )} | {"vehicle_episode_failed_check_counts": dict(reasons)})
    return {"source": identity(path), "reports": summaries}


def main():
    latest = read(RUN / "checkpoint_manifest.json")["latest"]
    last_metric = None
    with (RUN / "metrics.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                last_metric = json.loads(line)
    scalars = (last_metric or {}).get("scalars", {})
    snapshot = {
        "captured_at_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "Existing file evidence only; no simulator connection or new experiment",
        "latest_checkpoint": latest,
        "latest_metric_update": (last_metric or {}).get("update"),
        "latest_metric_selected_scalars": {key: value for key, value in scalars.items()
            if key in ("curriculum/stage_index", "curriculum/available_scenario_count",
                       "curriculum/base_review_required", "train/update_skipped",
                       "reward_audit/nonfinite_count", "reward_audit/max_abs_step_residual")},
        "stages": [],
    }
    for update in (50, 100, 200, 400, 600, 800):
        path = CONTROL / f"u{update:06d}_summary.json"
        stage = {"update": update, "stage_summary_exists": path.exists()}
        if path.exists():
            summary = read(path)
            stage.update({"summary_source": identity(path),
                          "training_integrity": summary.get("training_integrity"),
                          "selection": summary.get("selection"),
                          "stage_status": summary.get("status")})
        for scope in ("parallel", "validation"):
            qualification = CONTROL / f"screen_u{update:06d}" / scope / "qualification.json"
            stage[scope] = (report_summary(qualification) if qualification.exists() else
                           {"status": "NO_COMPLETE_QUALIFICATION_FILE_AT_CAPTURE"})
        snapshot["stages"].append(stage)
    for label, relative in (
        ("default_mut_config", "configs/mut_mappo/local_candidate_v1.json"),
        ("default_mut_cases", "data/mut_mappo/cases_v1.json"),
        ("symmetric_v2_cases", "data/mut_mappo/symmetric_v2/cases.json"),
    ):
        path = ROOT / relative
        value = read(path)
        snapshot[label] = {"source": identity(path)}
        if "cases" in value:
            snapshot[label].update({"case_count": len(value["cases"]),
                "split_counts": {key: len(items) for key, items in value["splits"].items()},
                "geometry_status": value.get("geometry_status")})
        else:
            snapshot[label].update({"communication": value.get("communication"),
                "model_spec": value.get("model_spec"),
                "red_capability": value.get("red_checkpoint", {}).get("capability_status")})
    destination = OUT / "事实核验快照.json"
    destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "latest_safe_update": latest["update"],
        "stage_report_updates": [stage["update"] for stage in snapshot["stages"]
            if stage["stage_summary_exists"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
