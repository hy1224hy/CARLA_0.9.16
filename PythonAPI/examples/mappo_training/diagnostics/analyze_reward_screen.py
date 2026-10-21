"""Pure-file analysis of matched reward-fork behavioral engineering screens.

No simulator, policy, or training imports are used. Active transitions include
the terminating transition and exclude all later, possibly stale diagnostics.
This screen describes known scenarios, not population-level generalization.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


AGENTS = ("0", "1")
EXEMPTIONS = (
    "pace_grace_exempt", "pace_front_blocker_exempt",
    "pace_near_goal_exempt", "pace_terminal_exempt",
    "pace_complex_interaction_exempt", "pace_deficit_eligible",
)


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant)


def _read(path):
    return _loads(path.read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _distribution(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p10": _quantile(values, 0.1),
        "p50": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "peak": max(values) if values else None,
    }


def _duration(flags, dt):
    count = sum(flags)
    run = longest = 0
    episodes = 0
    for flag in flags:
        if flag:
            episodes += int(run == 0)
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return {"steps": count, "fraction": count / len(flags) if flags else None,
            "seconds": count * dt, "longest_continuous_seconds": longest * dt,
            "interval_count": episodes}


def _analyze_trace(path, episode, dt, thresholds):
    records = {aid: {"speeds": [], "closing": [], "rows": [],
                     "exemptions": {key: [] for key in EXEMPTIONS},
                     "reward_terms": {}, "return": 0.0,
                     "first_motion_step": None, "arrival_step": None,
                     "termination_step": None, "final_diagnostics": None,
                     "terminal_events": Counter()} for aid in AGENTS}
    errors = []
    first_arrival = None
    previous_info = None
    previous_frame = None
    step_count = 0
    reset_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = _loads(line)
            if row.get("type") == "reset":
                reset_count += 1
                if line_number != 1:
                    raise ValueError("reset must be the first and only reset row")
                if row["case"] != episode["case"]:
                    errors.append("trace reset case differs from episode case")
                previous_info = row["info"]
                previous_frame = row["frame"]
                continue
            if row.get("type") != "step" or previous_info is None:
                raise ValueError(f"unexpected trace row type at line {line_number}")
            step_count += 1
            if row["step"] != step_count:
                errors.append(f"non-contiguous step at line {line_number}")
            if row["frame"] != previous_frame + 1:
                errors.append(f"world frame advanced by more than one at step {step_count}")
            info = row["info"]
            active_before = previous_info["active_mask"]
            active_after = info["active_mask"]
            new_arrival = False
            for aid in AGENTS:
                if not isinstance(active_before[aid], bool) or not isinstance(active_after[aid], bool):
                    raise ValueError("active_mask must contain booleans")
                if not active_before[aid]:
                    if active_after[aid]:
                        errors.append(f"agent {aid} reactivated at step {step_count}")
                    continue
                rec = records[aid]
                diagnostics = info["reward_diagnostics"][aid]
                details = info["reward_details"][aid]
                speed = _number(diagnostics["speed_mps"], "speed_mps")
                if speed < 0:
                    raise ValueError("speed_mps must be nonnegative")
                rec["speeds"].append(speed * 3.6)
                rec["closing"].append(_number(diagnostics["closing_speed_mps"], "closing_speed_mps"))
                rec["rows"].append(row["step"])
                rec["final_diagnostics"] = diagnostics
                if rec["first_motion_step"] is None and speed >= thresholds["startup_speed_mps"]:
                    rec["first_motion_step"] = row["step"]
                if diagnostics.get("success_event") and rec["arrival_step"] is None:
                    rec["arrival_step"] = row["step"]
                    new_arrival = True
                for key in ("success_event", "collision_event", "offroad_terminated",
                            "road_violation_terminated", "mission_timeout_event"):
                    rec["terminal_events"][key] += int(bool(diagnostics.get(key)))
                for key in EXEMPTIONS:
                    if key not in diagnostics:
                        errors.append(f"missing diagnostic {key}, agent {aid}, step {step_count}")
                    rec["exemptions"][key].append(bool(diagnostics.get(key)))
                reward = _number(row["reward"][aid], "reward")
                rec["return"] += reward
                summed = 0.0
                for term, value in details.items():
                    value = _number(value, term)
                    summed += value
                    rec["reward_terms"][term] = rec["reward_terms"].get(term, 0.0) + value
                if abs(reward - summed) > 1e-6:
                    errors.append(f"reward decomposition mismatch, agent {aid}, step {step_count}")
                if not active_after[aid]:
                    rec["termination_step"] = row["step"]
                    if not (info["terminated"][aid] or info["truncated"][aid]):
                        errors.append(f"agent {aid} inactive without termination/truncation")
            if new_arrival and first_arrival is None:
                first_arrival = {
                    "step": row["step"], "seconds": row["step"] * dt,
                    "agents": {aid: {
                        "active_before": active_before[aid], "active_after": active_after[aid],
                        "arrived": records[aid]["arrival_step"] is not None,
                        "route_completion": info["reward_diagnostics"][aid].get("route_completion"),
                        "speed_kmh": _number(info["reward_diagnostics"][aid]["speed_mps"], "speed_mps") * 3.6,
                    } for aid in AGENTS},
                }
            previous_info, previous_frame = info, row["frame"]
    if reset_count != 1 or step_count == 0:
        errors.append("missing reset or step rows")
    if step_count != episode.get("steps"):
        errors.append("trace step count differs from episode")
    if previous_info is not None:
        if any(previous_info["active_mask"].values()):
            errors.append("trace ends with active agents; incomplete evidence")
        if previous_info.get("episode_end_reason") != episode.get("reason"):
            errors.append("trace/episode terminal reason mismatch")
    output_agents = {}
    for aid, rec in records.items():
        source_agent = episode.get("agents", {}).get(aid, {})
        if len(rec["speeds"]) != source_agent.get("steps"):
            errors.append(f"agent {aid} active-step count differs from episode")
        if abs(rec["return"] - source_agent.get("return", float("inf"))) > 1e-6:
            errors.append(f"agent {aid} cumulative return differs from episode")
        if rec["first_motion_step"] != source_agent.get("first_motion_step"):
            errors.append(f"agent {aid} first-motion step differs from episode")
        output_agents[aid] = {
            "active_steps": len(rec["speeds"]), "active_seconds": len(rec["speeds"]) * dt,
            "first_motion_step": rec["first_motion_step"],
            "first_motion_seconds": rec["first_motion_step"] * dt if rec["first_motion_step"] is not None else None,
            "arrival_step": rec["arrival_step"],
            "arrival_seconds": rec["arrival_step"] * dt if rec["arrival_step"] is not None else None,
            "termination_step": rec["termination_step"],
            "reached_goal": rec["arrival_step"] is not None,
            "final_completion": (rec["final_diagnostics"] or {}).get("route_completion"),
            "speed_kmh_active_transition": _distribution(rec["speeds"]),
            "route_closing_speed_mps_active_transition": _distribution(rec["closing"]),
            "above_15_kmh": _duration([v > 15.0 for v in rec["speeds"]], dt),
            "above_30_kmh": _duration([v > 30.0 for v in rec["speeds"]], dt),
            "below_low_speed": _duration([v / 3.6 < thresholds["low_speed_mps"] for v in rec["speeds"]], dt),
            "pace_exemptions": {key: _duration(flags, dt) for key, flags in rec["exemptions"].items()},
            "return_recorded_reward": rec["return"],
            "reward_terms_sum": rec["reward_terms"],
            "recorded_pace_reward_sum": rec["reward_terms"].get("free_flow_pace_deficit"),
            "recorded_overspeed_reward_sum": rec["reward_terms"].get("overspeed"),
            "terminal_events": dict(rec["terminal_events"]),
        }
    arrivals = [rec["arrival_step"] for rec in records.values()]
    return {
        "agents": output_agents,
        "first_arrival": first_arrival,
        "arrival_gap_seconds": abs(arrivals[0] - arrivals[1]) * dt if all(v is not None for v in arrivals) else None,
        "errors": list(dict.fromkeys(errors)),
    }, records


def analyze(evaluation_dir):
    directory = Path(evaluation_dir).resolve()
    errors = []
    manifest_path = directory / "manifest.json"
    config_path = directory / "resolved_config.json"
    episodes_path = directory / "episodes.jsonl"
    manifest = _read(manifest_path)
    config = _read(config_path)
    checkpoints = {item["sha256"]: item for item in manifest.get("checkpoints", [])}
    thresholds = config["thresholds"]
    episodes = [_loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_count = len(config["scenarios"]) * len(config["seeds"]) * len(checkpoints)
    if manifest.get("status") != "COMPLETED":
        errors.append("evaluation manifest is not COMPLETED")
    if manifest.get("completed_cases") != len(episodes) or len(episodes) != expected_count:
        errors.append("completed, recorded, or expected case counts differ")
    summaries = []
    pooled = {aid: {"speeds": [], "closing": []} for aid in AGENTS}
    seen_ids = set()
    seen_conditions = set()
    for episode in episodes:
        case = episode["case"]
        case_errors = []
        case_id = case["id"]
        if case_id in seen_ids:
            case_errors.append("duplicate case ID")
        seen_ids.add(case_id)
        digest = episode.get("checkpoint_sha256")
        condition = (digest, case.get("scenario"), case.get("seed"), case.get("mode"))
        if condition in seen_conditions:
            case_errors.append("duplicate checkpoint/scenario/seed/mode condition")
        seen_conditions.add(condition)
        checkpoint = checkpoints.get(digest)
        dt = checkpoint.get("environment", {}).get("timestep") if checkpoint else None
        if dt is None:
            case_errors.append("checkpoint or fixed timestep absent from manifest")
        else:
            dt = _number(dt, "timestep")
            if dt <= 0:
                case_errors.append("fixed timestep must be positive")
        if case.get("scenario") not in config["scenarios"] or case.get("seed") not in config["seeds"]:
            case_errors.append("case scenario or seed absent from resolved config")
        if case.get("mode") != config.get("mode"):
            case_errors.append("case mode differs from resolved config")
        before, after = episode.get("actor_hashes_before"), episode.get("actor_hashes_after")
        if not before or before != after:
            case_errors.append("frozen actor hashes missing or changed")
        path = directory / "traces" / (case_id + ".jsonl")
        if path.resolve().parent != (directory / "traces").resolve():
            case_errors.append("case ID escapes trace directory")
        actual_sha = _sha(path) if path.is_file() else None
        if actual_sha is None:
            case_errors.append("trace file missing")
        elif actual_sha != episode.get("trace_sha256"):
            case_errors.append("trace SHA differs from recorded episode")
        result = {}
        records = None
        if path.is_file() and dt is not None and dt > 0 and path.resolve().parent == (directory / "traces").resolve():
            try:
                result, records = _analyze_trace(path, episode, dt, thresholds)
                case_errors.extend(result.pop("errors"))
            except (KeyError, TypeError, ValueError) as exc:
                case_errors.append(f"trace analysis failed: {exc}")
        valid = not case_errors
        if valid and records:
            for aid in AGENTS:
                pooled[aid]["speeds"].extend(records[aid]["speeds"])
                pooled[aid]["closing"].extend(records[aid]["closing"])
        summaries.append({"case": case, "checkpoint_sha256": digest,
                          "pace_profile": episode.get("pace_profile"), "timestep_seconds": dt,
                          "episode_end_reason": episode.get("reason"),
                          "trace_sha256": actual_sha,
                          "evidence_status": "VALID" if valid else "INVALID_OR_INCOMPLETE",
                          **result, "errors": list(dict.fromkeys(case_errors))})
    expected_conditions = {(digest, scenario, seed, config.get("mode"))
                           for digest in checkpoints for scenario in config["scenarios"]
                           for seed in config["seeds"]}
    if seen_conditions != expected_conditions:
        errors.append("checkpoint/scenario/seed/mode coverage differs from resolved plan")
    all_valid = not errors and all(case["evidence_status"] == "VALID" for case in summaries)
    aggregate = {"valid_case_count": sum(case["evidence_status"] == "VALID" for case in summaries),
                 "recorded_case_count": len(summaries), "expected_case_count": expected_count,
                 "episode_reasons": dict(Counter(case["episode_end_reason"] for case in summaries)),
                 "both_reached_count_valid_cases": sum(case["evidence_status"] == "VALID" and all(
                     agent["reached_goal"] for agent in case["agents"].values()) for case in summaries),
                 "agents": {}}
    for aid in AGENTS:
        valid_agents = [case["agents"][aid] for case in summaries if case["evidence_status"] == "VALID"]
        aggregate["agents"][aid] = {
            "reached_goal_count": sum(agent["reached_goal"] for agent in valid_agents),
            "speed_kmh_pooled_active_transitions": _distribution(pooled[aid]["speeds"]),
            "route_closing_speed_mps_pooled_active_transitions": _distribution(pooled[aid]["closing"]),
            "case_mean_speed_kmh": _distribution([agent["speed_kmh_active_transition"]["mean"] for agent in valid_agents]),
            "case_first_motion_seconds": _distribution([agent["first_motion_seconds"] for agent in valid_agents if agent["first_motion_seconds"] is not None]),
            "case_arrival_seconds_reached_only": _distribution([agent["arrival_seconds"] for agent in valid_agents if agent["arrival_seconds"] is not None]),
            "above_15_seconds": sum(agent["above_15_kmh"]["seconds"] for agent in valid_agents),
            "above_30_seconds": sum(agent["above_30_kmh"]["seconds"] for agent in valid_agents),
            "above_15_fraction_pooled": sum(v > 15 for v in pooled[aid]["speeds"]) / len(pooled[aid]["speeds"]) if pooled[aid]["speeds"] else None,
            "above_30_fraction_pooled": sum(v > 30 for v in pooled[aid]["speeds"]) / len(pooled[aid]["speeds"]) if pooled[aid]["speeds"] else None,
            "recorded_pace_reward_sum": sum(agent["recorded_pace_reward_sum"] for agent in valid_agents if agent["recorded_pace_reward_sum"] is not None),
            "recorded_overspeed_reward_sum": sum(agent["recorded_overspeed_reward_sum"] for agent in valid_agents if agent["recorded_overspeed_reward_sum"] is not None),
        }
    return {
        "schema": "reward_screen_file_analysis_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_dir": str(directory),
        "analysis_evidence_status": "COMPLETE_VALID_EVIDENCE" if all_valid else "INVALID_OR_INCOMPLETE_EVIDENCE",
        "qualification_decision": "NOT_ASSIGNED_ENGINEERING_DESCRIPTION_ONLY",
        "scope": "Known-scene behavioral engineering screen; no population confidence or held-out generalization claim.",
        "definitions": {
            "active_transition": "Agent active in reset/previous info.active_mask; includes own terminating step, excludes all later rows.",
            "speed": "Post-step reward_diagnostics.speed_mps, converted to km/h; active-transition weighted pooled summaries.",
            "duration": "Count of qualifying active transitions times fixed_delta timestep; discrete sampling, not interpolated crossing duration.",
            "arrival": "First success_event on an active transition; completion may be below 1 because rectangular goal zones terminate early.",
            "first_motion": "First post-step speed at or above resolved-config startup_speed_mps; includes rolling starts.",
            "pace_reward": "Sum of recorded reward component only; no counterfactual rescoring of the trace.",
            "physical_jerk": "Not computed; command changes are not physical jerk.",
            "first_arrival": "Observed partner completion/speed at first success; does not infer waiting intentions.",
        },
        "input_sha256": {name: _sha(directory / name) for name in ("manifest.json", "resolved_config.json", "episodes.jsonl")},
        "analysis_source_sha256": _sha(Path(__file__)),
        "manifest_status": manifest.get("status"),
        "manifest_formal_eligibility": manifest.get("formal_eligibility"),
        "manifest_source_sha256": manifest.get("source_sha256"),
        "checkpoint_identities": manifest.get("checkpoints"),
        "aggregate": aggregate, "cases": summaries, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.evaluation_dir)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": report["analysis_evidence_status"],
                      "manifest_status": report["manifest_status"], "aggregate": report["aggregate"]},
                     ensure_ascii=False, allow_nan=False))
    return 0 if report["analysis_evidence_status"] == "COMPLETE_VALID_EVIDENCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
