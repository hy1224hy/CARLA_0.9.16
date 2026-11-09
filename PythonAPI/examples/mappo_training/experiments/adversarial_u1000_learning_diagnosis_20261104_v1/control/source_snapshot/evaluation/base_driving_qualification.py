"""Pure-file, per-vehicle qualification for the versioned base-driving task.

This module has no simulator, Torch, trainer or evaluator imports. It accepts the
existing reward-screen trace format, but old screens cannot silently become new
protocol qualification evidence. Arrival time differences are descriptive only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping

SCHEMA = "base_driving_qualification_vnext_v1"
PROTOCOL = "base_driving_vnext_v1"
AGENTS = ("0", "1")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/evaluation/base_driving_vnext_v1.json"
EVENTS = ("collision_event", "offroad_terminated", "road_violation_terminated", "mission_timeout_event")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _flag(value, name):
    if value not in (0, 1, False, True):
        raise ValueError(f"{name} must be an explicit boolean or zero/one")
    return bool(value)


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _mean(values):
    return statistics.fmean(values) if values else None


def _longest(flags, dt):
    run = longest = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest * dt


def load_contract(path=DEFAULT_CONTRACT):
    contract = read_json(path)
    validate_contract(contract)
    return contract


def verify_evaluation_sources(sources, root=ROOT):
    """Refuse to qualify traces after recorded execution/analysis source drift."""
    required = {"evaluation/base_driving_qualification.py", "evaluation/evaluate_base_driving_vnext.py",
                "evaluation/diagnose_r8c1_behavior.py", "environment/base_driving_env.py"}
    if not isinstance(sources, dict) or not required.issubset({str(name).replace("\\", "/") for name in sources}):
        raise ValueError("evaluation implementation identity is absent or incomplete")
    root = Path(root).resolve()
    for name, digest in sources.items():
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError("evaluation source paths must be relative to project ROOT")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"evaluation source path escapes ROOT or is missing: {name}")
        if sha256_file(path) != digest:
            raise ValueError(f"evaluation source SHA drift: {name}; a new analysis protocol/version is required")


def validate_contract(contract):
    if contract.get("schema") != SCHEMA or contract.get("protocol") != PROTOCOL:
        raise ValueError("unsupported base driving qualification contract")
    limits = contract["thresholds"]
    for key, value in limits.items():
        if _number(value, key) < 0:
            raise ValueError(f"{key} must be nonnegative")
        if "fraction" in key and value > 1:
            raise ValueError(f"{key} must be a proportion")
    if limits["preferred_speed_min_mps"] >= limits["preferred_speed_max_mps"]:
        raise ValueError("invalid preferred task speed band")
    if limits["low_speed_mps"] >= limits["startup_speed_mps"]:
        raise ValueError("low-speed threshold must be below first-motion threshold")
    if contract["evaluation"]["mode"] != "deterministic":
        raise ValueError("qualification requires deterministic policy execution")
    return contract


def _agent_item(mapping, aid):
    return mapping[aid] if aid in mapping else mapping[int(aid)]


def _metadata(scenario, aid):
    role = scenario["agent_roles"].get(aid, scenario["agent_roles"].get("agent" + aid))
    if not isinstance(role, str) or not role:
        raise ValueError(f"missing explicit role for agent {aid}")
    startup = scenario["startup_mode"]
    if startup not in {"stationary", "rolling"}:
        raise ValueError("startup_mode must be stationary or rolling")
    direction = scenario["direction"]
    if direction not in {"northbound", "southbound", "eastbound", "westbound"}:
        raise ValueError("direction must be an explicitly registered cardinal road direction")
    for field in ("task_family", "pair_id", "physical_template_id", "symmetry_partner"):
        if not isinstance(scenario[field], str) or not scenario[field]:
            raise ValueError(f"missing explicit scenario {field}")
    return {"task_family": scenario["task_family"], "direction": direction,
            "startup_mode": startup, "role": role,
            "pair_id": scenario["pair_id"], "physical_template_id": scenario["physical_template_id"]}


def _failed_case(episode, errors):
    return {"case": episode.get("case", {}), "checkpoint_sha256": episode.get("checkpoint_sha256"),
            "evidence_status": "INVALID_OR_INCOMPLETE", "qualified": False,
            "both_reached": None, "agents": {}, "errors": list(dict.fromkeys(errors))}


def qualify_episode(episode: Mapping, trace_rows: Iterable[Mapping], scenario: Mapping,
                    contract: Mapping, evidence_context: Mapping):
    """Score one existing-format episode with explicit scenario/protocol context.

    Context keys: protocol, timestep_seconds, checkpoint_eligible. Only the new
    evaluator may establish checkpoint_eligible after exact checkpoint checks.
    A caller may set False for an explicitly labelled historical reference.
    """
    validate_contract(contract)
    errors = []
    warnings = []
    if evidence_context.get("protocol") != PROTOCOL:
        return _failed_case(episode, ["trace was not collected under the registered new evaluation protocol"])
    if episode.get("case", {}).get("mode") != contract["evaluation"]["mode"]:
        return _failed_case(episode, ["policy execution mode does not match qualification contract"])
    try:
        dt = _number(evidence_context["timestep_seconds"], "timestep_seconds")
        if dt <= 0:
            raise ValueError("timestep_seconds must be positive")
        metadata = {aid: _metadata(scenario, aid) for aid in AGENTS}
        rows = list(trace_rows)
        if not rows or rows[0].get("type") != "reset":
            raise ValueError("missing initial reset evidence")
        if rows[0].get("case") != episode.get("case"):
            raise ValueError("reset case differs from episode")
        before = rows[0]["info"]
        frame = rows[0]["frame"]
        limits = contract["thresholds"]
        initial_speeds = {}
        for aid in AGENTS:
            requested = _number(scenario["agents"]["agent" + aid]["initial_speed_mps"], "requested initial speed")
            measured = _number(_agent_item(rows[0]["kinematics"], aid)["speed"], "measured initial speed")
            if requested < 0 or measured < 0 or abs(requested - measured) > 0.5:
                raise ValueError("initial physical speed does not match registered scenario request")
            stationary = metadata[aid]["startup_mode"] == "stationary"
            if (stationary and measured >= limits["startup_speed_mps"]) or (not stationary and measured < limits["startup_speed_mps"]):
                raise ValueError("startup mode differs from measured initial physical speed")
            initial_speeds[aid] = measured
        records = {aid: {"speed": [], "lateral": [], "cruise": [], "low": [],
                         "steer_rates": [], "actual_controls": 0, "missing_terminal_control": 0,
                         "events": Counter(), "first_motion_seconds": None,
                         "first_motion_unblocked_seconds": None, "unblocked_elapsed": 0.0,
                         "arrival_seconds": None, "last_steer": None, "last_control_step": None,
                         "steps": 0, "return": 0.0} for aid in AGENTS}
        steps = 0
        for row in rows[1:]:
            steps += 1
            if row.get("type") != "step" or row.get("step") != steps or row["frame"] != frame + 1:
                raise ValueError("trace step/frame sequence is incomplete or non-contiguous")
            info = row["info"]
            for aid in AGENTS:
                active_before = _flag(_agent_item(before["active_mask"], aid), "active_mask_before")
                active_after = _flag(_agent_item(info["active_mask"], aid), "active_mask_after")
                if not active_before:
                    if active_after:
                        raise ValueError("inactive agent reactivated")
                    continue
                data = records[aid]
                d = _agent_item(info["reward_diagnostics"], aid)
                speed = _number(d["speed_mps"], "speed_mps")
                lateral = abs(_number(d["lateral_distance_m"], "lateral_distance_m"))
                distance = _number(d["distance_to_goal_m"], "distance_to_goal_m")
                blocked = _flag(d["pace_front_blocker_exempt"], "pace_front_blocker_exempt")
                success = _flag(d["success_event"], "success_event")
                if speed < 0 or distance < 0:
                    raise ValueError("negative speed or own goal distance")
                data["steps"] += 1
                data["speed"].append(speed)
                data["lateral"].append(lateral)
                if data["first_motion_seconds"] is None:
                    if not blocked:
                        data["unblocked_elapsed"] += dt
                    if speed >= limits["startup_speed_mps"]:
                        data["first_motion_seconds"] = steps * dt
                        data["first_motion_unblocked_seconds"] = data["unblocked_elapsed"]
                if success:
                    if data["arrival_seconds"] is not None:
                        raise ValueError("duplicate own success event")
                    data["arrival_seconds"] = steps * dt
                for event in EVENTS:
                    data["events"][event] += int(_flag(d[event], event))
                reward = _number(_agent_item(row["reward"], aid), "reward")
                details = _agent_item(info["reward_details"], aid)
                if abs(sum(_number(v, name) for name, v in details.items()) - reward) > 1e-6:
                    raise ValueError("recorded reward decomposition differs from reward")
                data["return"] += reward
                efficiency = (steps * dt > limits["startup_exclusion_seconds"] and
                              distance > limits["near_goal_exclusion_m"] and not blocked)
                data["cruise"].append(speed if efficiency else None)
                # False at every excluded transition prevents joining two low-speed intervals.
                data["low"].append(efficiency and speed < limits["low_speed_mps"])
                actual = _agent_item(row["post"], aid)
                if actual is None:
                    if active_after:
                        raise ValueError(f"agent {aid} actual control missing before termination")
                    data["missing_terminal_control"] += 1
                    data["last_steer"] = None
                else:
                    control = actual["control"]
                    if len(control) != 3:
                        raise ValueError("actual VehicleControl readback needs throttle/steer/brake")
                    throttle, steer, brake = [_number(v, "actual control") for v in control]
                    if not (0 <= throttle <= 1 and -1 <= steer <= 1 and 0 <= brake <= 1):
                        raise ValueError("actual VehicleControl readback is out of bounds")
                    data["actual_controls"] += 1
                    if data["last_steer"] is not None and data["last_control_step"] == steps - 1:
                        data["steer_rates"].append(abs(steer - data["last_steer"]) / dt)
                    data["last_steer"], data["last_control_step"] = steer, steps
                if not active_after and not (_flag(_agent_item(info["terminated"], aid), "terminated") or
                                             _flag(_agent_item(info["truncated"], aid), "truncated")):
                    raise ValueError("inactive without terminated/truncated evidence")
            before, frame = info, row["frame"]
        if steps != episode["steps"] or any(_agent_item(before["active_mask"], aid) for aid in AGENTS):
            raise ValueError("episode/trace incomplete active boundary")
        if before.get("episode_end_reason") != episode.get("reason"):
            raise ValueError("episode and trace terminal reason differ")
        if not episode.get("actor_hashes_before") or episode["actor_hashes_before"] != episode.get("actor_hashes_after"):
            raise ValueError("frozen actor hashes absent or changed")
        output = {}
        for aid, data in records.items():
            if data["steps"] <= 0:
                raise ValueError("agent has no active transition evidence")
            recorded = _agent_item(episode["agents"], aid)
            if data["steps"] != recorded["steps"] or abs(data["return"] - recorded["return"]) > 1e-6:
                raise ValueError("episode summary and active trace totals differ")
            cruise = [v for v in data["cruise"] if v is not None]
            n = len(cruise)
            low = sum(data["low"]) / n if n else None
            band = sum(limits["preferred_speed_min_mps"] <= v <= limits["preferred_speed_max_mps"] for v in cruise) / n if n else None
            above = sum(v > limits["preferred_speed_max_mps"] for v in data["speed"]) / data["steps"]
            stationary = metadata[aid]["startup_mode"] == "stationary"
            metrics = {"active_steps": data["steps"], "active_seconds": data["steps"] * dt,
                       "measured_initial_speed_mps": initial_speeds[aid],
                       "reached_goal": data["arrival_seconds"] is not None,
                       "first_motion_seconds": data["first_motion_seconds"],
                       "unblocked_first_motion_seconds": data["first_motion_unblocked_seconds"] if stationary else None,
                       "startup_applicable": stationary, "arrival_seconds": data["arrival_seconds"],
                       "cruise_samples": n, "preferred_speed_fraction": band,
                       "unblocked_low_speed_fraction": low,
                       "unblocked_low_speed_longest_seconds": _longest(data["low"], dt),
                       "above_preferred_speed_fraction": above,
                       "peak_speed_mps": max(data["speed"]), "mean_speed_mps": _mean(data["speed"]),
                       "reference_lateral_p95_m": _quantile(data["lateral"], .95),
                       "reference_lateral_peak_m": max(data["lateral"]),
                       "actual_control_fraction": data["actual_controls"] / data["steps"],
                       "missing_terminal_actual_control": data["missing_terminal_control"],
                       "actual_steer_rate_p95_per_second": _quantile(data["steer_rates"], .95),
                       "terminal_events": dict(data["events"])}
            checks = {"own_goal_reached": metrics["reached_goal"],
                      "safe": not any(data["events"].values()),
                      "stationary_unblocked_startup": (not stationary or
                          (metrics["unblocked_first_motion_seconds"] is not None and
                           metrics["unblocked_first_motion_seconds"] <= limits["stationary_start_max_unblocked_first_motion_seconds"])),
                      "sufficient_cruise_evidence": n >= limits["minimum_cruise_samples"],
                      "preferred_cruise_speed": band is not None and band >= limits["minimum_preferred_speed_fraction"],
                      "unblocked_low_speed": low is not None and low <= limits["max_unblocked_low_speed_fraction"] and
                          metrics["unblocked_low_speed_longest_seconds"] <= limits["max_unblocked_low_speed_continuous_seconds"],
                      "upper_task_speed": above <= limits["max_above_preferred_speed_fraction"] and
                          metrics["peak_speed_mps"] <= limits["max_peak_speed_mps"],
                      "reference_lane_stability": metrics["reference_lateral_p95_m"] <= limits["max_reference_lateral_p95_m"] and
                          metrics["reference_lateral_peak_m"] <= limits["max_reference_lateral_peak_m"],
                      "actual_control_coverage": metrics["actual_control_fraction"] >= limits["minimum_actual_control_fraction"],
                      "actual_steering_stability": metrics["actual_steer_rate_p95_per_second"] is not None and
                          metrics["actual_steer_rate_p95_per_second"] <= limits["max_actual_steer_rate_p95_per_second"]}
            if data["missing_terminal_control"]:
                warnings.append(f"agent {aid}: terminating actual control unavailable, excluded and counted")
            output[aid] = {"stratum": metadata[aid], "metrics": metrics, "checks": checks,
                           "qualified": all(checks.values()), "failed_checks": [k for k, passed in checks.items() if not passed]}
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(str(exc))
        return _failed_case(episode, errors)
    arrivals = [output[aid]["metrics"]["arrival_seconds"] for aid in AGENTS]
    return {"case": episode["case"], "checkpoint_sha256": episode.get("checkpoint_sha256"),
            "evidence_status": "VALID", "checkpoint_eligible": evidence_context.get("checkpoint_eligible") is True,
            "qualified": evidence_context.get("checkpoint_eligible") is True and all(v["qualified"] for v in output.values()),
            "both_reached": all(v["metrics"]["reached_goal"] for v in output.values()),
            "arrival_gap_seconds_descriptive_only": abs(arrivals[0] - arrivals[1]) if all(v is not None for v in arrivals) else None,
            "agents": output, "warnings": warnings, "errors": []}


def qualify_evaluation(cases, contract, expected_conditions=None):
    """Require both agents and each family/direction/startup/role stratum to pass."""
    validate_contract(contract)
    cfg = contract["evaluation"]
    cases = list(cases)
    errors = []
    conditions = [(c.get("checkpoint_sha256"), c.get("case", {}).get("scenario"),
                   c.get("case", {}).get("seed"), c.get("case", {}).get("mode")) for c in cases]
    if len(set(conditions)) != len(conditions):
        errors.append("duplicate checkpoint/scenario/seed/mode condition")
    if expected_conditions is not None and set(conditions) != set(tuple(v) for v in expected_conditions):
        errors.append("evaluated coverage differs from registered expected conditions")
    if len({c.get("checkpoint_sha256") for c in cases}) != 1:
        errors.append("qualification must contain exactly one checkpoint")
    valid = [c for c in cases if c.get("evidence_status") == "VALID"]
    if len(valid) != len(cases) or not cases:
        errors.append("missing or invalid episode evidence")
    agent_reports = {}
    any_safety_event = False
    for aid in AGENTS:
        groups = defaultdict(list)
        for case in valid:
            record = case["agents"][aid]
            stratum = record["stratum"]
            groups[tuple(stratum[name] for name in cfg["stratum_fields"])].append(record)
            any_safety_event |= bool(any(record["metrics"]["terminal_events"].values()))
        strata = []
        for labels, records in sorted(groups.items()):
            fraction = sum(r["qualified"] for r in records) / len(records)
            strata.append({"labels": dict(zip(cfg["stratum_fields"], labels)), "cases": len(records),
                           "qualified_fraction": fraction,
                           "failed_checks": dict(Counter(check for r in records for check in r["failed_checks"])),
                           "passed": len(records) >= cfg["minimum_cases_per_stratum"] and
                               fraction >= cfg["minimum_qualified_case_fraction"]})
        directions = {r["stratum"]["direction"] for c in valid for r in [c["agents"][aid]]}
        startup_modes = {r["stratum"]["startup_mode"] for c in valid for r in [c["agents"][aid]]}
        coverage = (set(cfg["required_directions"]).issubset(directions) and
                    set(cfg["required_startup_modes"]).issubset(startup_modes))
        agent_reports[aid] = {"qualified": coverage and bool(strata) and all(s["passed"] for s in strata),
                              "required_axis_coverage": coverage, "strata": strata}
    both = sum(c["both_reached"] is True for c in valid) / len(cases) if cases else None
    checks = {"complete_valid_evidence": not errors,
              "new_protocol_checkpoint": bool(cases) and all(c.get("checkpoint_eligible") is True for c in cases),
              "minimum_cases": len(cases) >= cfg["minimum_cases"],
              "minimum_seeds": len({c.get("case", {}).get("seed") for c in cases}) >= cfg["minimum_seeds"],
              "registered_seeds": {c.get("case", {}).get("seed") for c in cases} == set(cfg["seeds"]),
              "no_safety_or_timeout_events": not any_safety_event and bool(valid),
              "both_reached_fraction": both is not None and both >= cfg["minimum_both_reached_fraction"],
              "each_vehicle_and_stratum": all(r["qualified"] for r in agent_reports.values())}
    passed = all(checks.values())
    return {"schema": SCHEMA, "protocol": PROTOCOL,
            "evidence_status": "COMPLETE_VALID_EVIDENCE" if not errors else "INVALID_OR_INCOMPLETE_EVIDENCE",
            "qualification_decision": "QUALIFIED_FOR_REGISTERED_SUITE_ONLY" if passed else "NOT_QUALIFIED",
            "qualified": passed, "checks": checks, "failed_checks": [k for k, v in checks.items() if not v],
            "case_count": len(cases), "both_reached_fraction": both, "agents": agent_reports,
            "cases": cases, "errors": errors,
            "scope": "Registered engineering/validation suite only. No held-out or population generalization claim; arrival gap is never a qualification condition."}


def _qualified_report(report):
    return (report.get("schema") == SCHEMA and report.get("protocol") == PROTOCOL and
            report.get("qualified") is True and report.get("evidence_status") == "COMPLETE_VALID_EVIDENCE" and
            report.get("checks", {}).get("new_protocol_checkpoint") is True)


def select_candidate(reports, regression_reports=None):
    """Select only a qualified new-protocol checkpoint under identical contracts.

    Reward return is deliberately absent from the ranking. This function never
    promotes an unsafe fastest policy when no candidate passes qualification.
    """
    reports = list(reports)
    if not reports:
        return {"selected_checkpoint_sha256": None, "decision": "NO_REPORTS"}
    identity = {(r.get("contract_sha256"), r.get("scenario_suite_sha256"), r.get("suite_role")) for r in reports}
    if len(identity) != 1 or any(not v for row in identity for v in row):
        raise ValueError("candidate selection requires identical explicit contract, suite, and split role")
    if reports[0]["suite_role"] != "validation":
        raise ValueError("candidate selection requires a validation suite, not training screen or held-out test")
    eligible = [r for r in reports if _qualified_report(r)]
    if not eligible:
        return {"selected_checkpoint_sha256": None, "decision": "NO_QUALIFIED_CANDIDATE",
                "reason": "Continue development; joint arrival alone cannot qualify either vehicle."}
    regressions = defaultdict(list)
    for report in regression_reports or []:
        regressions[report.get("checkpoint_sha256")].append(report)
    cleared, pending, failed = [], [], []
    for candidate in eligible:
        digest = candidate["checkpoint_sha256"]
        matches = regressions[digest]
        if not matches:
            pending.append(digest)
            continue
        if len(matches) != 1:
            raise ValueError("each candidate needs exactly one explicit parallel regression report")
        regression = matches[0]
        definition = regression.get("scenario_suite_definition", {})
        checks = {"same_qualification_contract": regression.get("contract_sha256") == candidate.get("contract_sha256"),
                  "training_screen_only": regression.get("suite_role") == "training_screen",
                  "explicit_parallel_subset": regression.get("parallel_regression") is True and
                      definition.get("kind") == "parallel_stage0_regression" and
                      definition.get("source_scenario_count") == 48 and len(definition.get("scenario_names", [])) == 16,
                  "same_training_pool": bool(candidate.get("training_scenario_pool_sha256")) and
                      candidate.get("training_scenario_pool_sha256") == definition.get("pool_manifest_sha256"),
                  "valid_qualified_regression": _qualified_report(regression)}
        if all(checks.values()):
            cleared.append(candidate)
        else:
            failed.append({"checkpoint_sha256": digest, "failed_checks": [key for key, passed in checks.items() if not passed]})
    if not cleared:
        return {"selected_checkpoint_sha256": None,
                "decision": "PENDING_PARALLEL_REGRESSION" if pending else "PARALLEL_REGRESSION_FAILED",
                "pending_checkpoint_sha256": pending, "failed_regressions": failed,
                "reason": "Validation qualification is retained, but selection also requires the same checkpoint to preserve registered parallel driving capability. Training regression is not held-out evidence."}
    def score(report):
        records = [a["metrics"] for c in report["cases"] for a in c["agents"].values()]
        return (_mean([v["above_preferred_speed_fraction"] for v in records]),
                _mean([v["reference_lateral_p95_m"] for v in records]),
                _mean([v["arrival_seconds"] for v in records]), str(report.get("checkpoint_sha256")))
    selected = min(cleared, key=score)
    return {"selected_checkpoint_sha256": selected["checkpoint_sha256"], "decision": "SELECTED_QUALIFIED_VALIDATION_CANDIDATE",
            "parallel_regression_required": True, "pending_checkpoint_sha256": pending, "failed_regressions": failed,
            "ranking": "Validation and same-checkpoint parallel retention gates first, then lower upper-speed fraction, lateral p95 and mean own arrival time. No synchronized-arrival objective; no held-out claim."}


def analyze_evaluation(evaluation_dir, contract_path=DEFAULT_CONTRACT):
    """Verify hashes and qualify a new evaluator directory using files only."""
    directory = Path(evaluation_dir).resolve()
    contract = load_contract(contract_path)
    contract_sha = sha256_file(contract_path)
    manifest = read_json(directory / "manifest.json")
    config = read_json(directory / "resolved_config.json")
    if manifest.get("protocol") != PROTOCOL or config.get("protocol") != PROTOCOL:
        raise ValueError("historical reward/C0 screens are reference evidence, not new-protocol qualification")
    if manifest.get("contract_sha256") != contract_sha or config.get("contract_sha256") != contract_sha:
        raise ValueError("evaluation qualification contract SHA mismatch")
    if manifest.get("status") != "COMPLETED":
        raise ValueError("evaluation is not complete")
    if manifest.get("resolved_config_sha256") != sha256_file(directory / "resolved_config.json"):
        raise ValueError("resolved evaluation configuration SHA mismatch")
    verify_evaluation_sources(manifest.get("source_sha256"))
    if manifest.get("geometry_preflight", {}).get("status") != "PASS_GEOMETRY_ONLY" or config.get("geometry_preflight") != manifest.get("geometry_preflight"):
        raise ValueError("qualification requires a passed frozen geometry preflight")
    if sha256_file(directory / "geometry_preflight.json") != manifest["geometry_preflight"].get("report_sha256"):
        raise ValueError("preserved geometry preflight report SHA mismatch")
    from utils.preflight_base_driving_scenarios_v1 import require_geometry_preflight
    geometry = require_geometry_preflight(Path(config["scenario_manifest_path"]).parent.parent,
                                         directory / "geometry_preflight.json")
    from evaluation.evaluate_base_driving_vnext import load_scenarios, select_evaluation_scenarios
    pool, all_scenarios, pool_sha = load_scenarios(config["scenario_manifest_path"])
    parallel_regression = config.get("parallel_regression")
    selected_scenarios, suite_sha, definition = select_evaluation_scenarios(
        pool, all_scenarios, pool_sha, parallel_regression=parallel_regression)
    if (geometry["manifest_sha256"] != pool_sha or manifest.get("scenario_pool_manifest_sha256") != pool_sha or
            config.get("scenario_pool_manifest_sha256") != pool_sha or
            config.get("scenario_suite_sha256") != suite_sha or manifest.get("scenario_suite_sha256") != suite_sha or
            config.get("scenario_suite_definition") != definition or manifest.get("scenario_suite_definition") != definition or
            config.get("scenarios") != selected_scenarios or manifest.get("parallel_regression") != parallel_regression):
        raise ValueError("evaluation suite differs from its geometry preflight")
    cleanup = manifest.get("cleanup", [])
    if not cleanup or any(c.get("error") or not c.get("audit", {}).get("world_timing_restored") or
                          c.get("audit", {}).get("vehicle_actor_count") != 0 or
                          c.get("audit", {}).get("sensor_actor_count") != 0 for c in cleanup):
        raise ValueError("evaluation cleanup evidence is missing or failed")
    results = []
    episodes_path = directory / "episodes.jsonl"
    episodes = [json.loads(line, parse_constant=_reject_constant) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoints = {c["sha256"]: c for c in manifest["checkpoints"]}
    for checkpoint in checkpoints.values():
        if checkpoint.get("qualification_eligible") is True:
            binding = checkpoint.get("base_driving_contract") or {}
            if binding.get("protocol") != PROTOCOL or binding.get("qualification_contract_sha256") != contract_sha:
                raise ValueError("eligible checkpoint does not carry the registered qualification binding")
    if manifest["completed_cases"] != len(episodes):
        raise ValueError("manifest/episode case count mismatch")
    for episode in episodes:
        case = episode["case"]
        path = (directory / "traces" / (case["id"] + ".jsonl")).resolve()
        if path.parent != (directory / "traces").resolve() or sha256_file(path) != episode["trace_sha256"]:
            raise ValueError("invalid trace path or SHA")
        checkpoint = checkpoints[episode["checkpoint_sha256"]]
        scenario = config["scenarios"][case["scenario"]]["config"]
        traces = [json.loads(line, parse_constant=_reject_constant) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        results.append(qualify_episode(episode, traces, scenario, contract,
                       {"protocol": PROTOCOL, "timestep_seconds": checkpoint["environment"]["timestep"],
                        "checkpoint_eligible": checkpoint.get("qualification_eligible") is True}))
    expected = [(digest, scenario, seed, config["mode"]) for digest in checkpoints for scenario in config["scenarios"] for seed in config["seeds"]]
    reports = []
    for digest in checkpoints:
        report = qualify_evaluation([c for c in results if c["checkpoint_sha256"] == digest], contract,
                                   [c for c in expected if c[0] == digest])
        report.update(checkpoint_sha256=digest, contract_sha256=contract_sha,
                      scenario_suite_sha256=manifest["scenario_suite_sha256"], suite_role=config["suite_role"],
                      parallel_regression=parallel_regression, scenario_suite_definition=definition,
                      training_scenario_pool_sha256=(checkpoints[digest].get("base_driving_contract") or {}).get("scenario_pool_sha256"),
                      source_sha256=sha256_file(__file__),
                      input_sha256={name: sha256_file(directory / name) for name in ("manifest.json", "resolved_config.json", "episodes.jsonl")})
        reports.append(report)
    return {"schema": SCHEMA, "reports": reports,
            "selection": select_candidate(reports) if config["suite_role"] == "validation" else
                {"selected_checkpoint_sha256": None, "decision": "SCREEN_OR_TEST_NOT_USED_FOR_SELECTION"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_evaluation(args.evaluation_dir, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "selection": result["selection"],
                      "decisions": [r["qualification_decision"] for r in result["reports"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
