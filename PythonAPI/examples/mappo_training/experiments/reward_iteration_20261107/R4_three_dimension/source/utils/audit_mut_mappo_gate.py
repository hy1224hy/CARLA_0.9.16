"""Replay B2 rules on saved B0 evidence. File-only; never imports a CARLA client."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from training.checkpoint_protocol import load_checkpoint_file, sha256_file, utc_now_iso
from training.mut_mappo_checkpoint import _validate_payload
from training.mut_mappo_learning import GateState, actor_states_hash

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def explain(gate):
    cfg = gate.config
    minimum = cfg.get("gate_min_cases", 8)
    bad_max = cfg.get("gate_max_bad_rate", .125)
    margin = cfg.get("gate_min_q_gap", .25)
    result = []
    for student, teacher in (("A", "B"), ("B", "A")):
        for family, own in gate.statistics[student].items():
            peer = gate.statistics[teacher][family]
            ns, nt = own["n_all"], peer["n_all"]
            checks = dict(enough_samples=min(ns, nt) >= minimum,
                teacher_bad_within_limit=peer["b_count"] / nt <= bad_max,
                teacher_no_worse_bad_rate=peer["b_count"] / nt <= own["b_count"] / ns,
                teacher_quality_gap=peer["q_count"] / nt - own["q_count"] / ns >= margin)
            assert gate.gates[student][family] == int(all(checks.values()))
            result.append(dict(student=student, teacher=teacher, family=family,
                n=nt, student_q=own["q_count"], student_bad=own["b_count"],
                teacher_q=peer["q_count"], teacher_bad=peer["b_count"],
                maximum_teacher_bad_count=math.floor(nt * bad_max),
                required_quality_count_gap=math.ceil(nt * margin),
                gate=gate.gates[student][family], checks=checks))
    return result


def audit(root, output):
    config_path = root / "configs/R2_B0_s42.json"
    config = read(config_path)
    if config["method"] != "off" or config["mutual"]["quality_metric"] != "critical":
        raise ValueError("this audit expects the saved R2 B0 critical-quality cohort")
    metrics_path = root / "runs/R2/metrics.jsonl"
    rows = [row for row in records(metrics_path) if "gate" in row]
    payload, checkpoint_sha = load_checkpoint_file(root / "runs/R2/latest.pt", map_location="cpu")
    _validate_payload(payload)
    gate = GateState(config["mutual"])
    gate.load_state_dict(payload["gate"])
    if rows[-1]["gate"] != gate.state_dict():
        raise ValueError("latest metrics and checkpoint gate differ")
    marker_path = root / "control/R2_gate_completed.json"
    marker = read(marker_path)
    if marker["returncode"] != 0:
        raise ValueError("frozen evaluation did not exit normally")
    paths = [(name, sha) for name, sha in marker["artifact_hashes"].items()
             if name.replace("\\", "/").endswith("analysis/critical_report.json")]
    if len(paths) != 1:
        raise ValueError("expected a single signed frozen report")
    name, digest = paths[0]
    report_path = root / name.replace("\\", "/")
    if sha256_file(report_path) != digest:
        raise ValueError("signed frozen report SHA differs")
    report = read(report_path)
    if len(report["reports"]) != 1:
        raise ValueError("do not pool evaluation cohorts")
    cohort = report["reports"][0]
    manifest = read(ROOT / config["case_manifest"])
    if (cohort["manifest"]["checkpoint_sha256"] != checkpoint_sha or
            cohort["manifest"]["case_manifest_sha256"] != sha256_file(ROOT / config["case_manifest"]) or
            cohort["manifest"]["config_sha256"] != sha256_file(config_path)):
        raise ValueError("frozen evaluation config/model/cases differ")
    version = gate.version + 1
    hashes = {team: actor_states_hash(state["actors"]) for team, state in payload["teams"].items()}
    stats, normal, differences = {"A": {}, "B": {}}, {}, []
    lookup = {case["case_key"]: case for case in manifest["cases"]}
    for team in stats:
        cases = [case for case in cohort["cases"] if case["team"] == team and case["condition"] == "C2"]
        if len(cases) != len(manifest["splits"]["gate"]) or {c["case_key"] for c in cases} != set(manifest["splits"]["gate"]):
            raise ValueError("incomplete or duplicate C2 gate cases")
        for case in cases:
            if not case["valid"] or case["family"] != lookup[case["case_key"]]["family"]:
                raise ValueError("invalid frozen gate case")
            eligible_h = int(case["blue_eligible"] and case["critical_interaction"]["critical_event"])
            if case["safe_critical_event"] != eligible_h:
                differences.append([team, case["case_key"]])
            record = stats[team].setdefault(case["family"], dict(n_all=0, q_count=0, b_count=0,
                case_keys=[], teacher_hash=hashes[team], version=version))
            record["n_all"] += 1
            record["q_count"] += int(case["safe_critical_event"])
            record["b_count"] += int(not case["blue_eligible"])
            record["case_keys"].append(case["case_key"])
        qualified = [c for c in cases if c["safe_critical_event"]]
        normal[team] = dict(qualified_danger=len(qualified), both_blue_above_speed_fraction_at_most_005=sum(
            all(c["driving_metrics"][str(aid)]["above_preferred_speed_fraction"] <= .05 for aid in (2, 3))
            for c in qualified))
    final = GateState(config["mutual"])
    final.refresh(hashes, stats, version=version, completed_round=payload["completed_round"])
    refreshes = [r for i, r in enumerate(rows) if r["gate"]["version"] and
                 (i == 0 or r["gate"]["version"] != rows[i - 1]["gate"]["version"])]
    result = dict(schema="b2_gate_audit_v1", created_at=utc_now_iso(),
        scope="file-only B0 evidence replay, not B2 training performance", live_world_ticks=0,
        artifacts={str(path.relative_to(ROOT)): sha256_file(path) for path in
            (config_path, metrics_path, marker_path, report_path, root / "runs/R2/latest.pt",
             ROOT / "training/mut_mappo_learning.py", ROOT / "training/train_mut_mappo.py",
             ROOT / "training/mutual_distillation.py", ROOT / "utils/run_first_point_experiments.py",
             Path(__file__))},
        actual_refresh_schedule=[dict(round=r["completed_round"], train_ticks=r["actual_counters"]["train"],
            version=r["gate"]["version"]) for r in refreshes],
        actual_saved_gate=dict(state=gate.state_dict(), decisions=explain(gate)),
        counterfactual_final_gate=dict(state=final.state_dict(), decisions=explain(final)),
        online_offline_quality_disagreements=differences,
        qualified_danger_speed_diagnostic=normal,
        limitations=["A/B are two learners of one outer seed; no independent-seed inference",
            "Final replay uses B0 policies and cannot predict a trained B2 trajectory",
            "Gate quality counts event eligibility, not full normal driving or C1-relative causal attack gain",
            "Gate split reused for development; test split not accessed"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiments/adversarial_u1000_critical_reward_20261101_v4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in ("scope", "actual_refresh_schedule",
        "qualified_danger_speed_diagnostic", "online_offline_quality_disagreements")}, ensure_ascii=False))
    for key in ("actual_saved_gate", "counterfactual_final_gate"):
        print(key, json.dumps(result[key]["decisions"], ensure_ascii=False))


if __name__ == "__main__":
    main()
