"""File-only V5/V6 cost comparison on fixed, completed V5 trace recordings.

This does not simulate a different policy or predict its behavior.  Only
stdlib and CARLA-independent reward functions are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from environment import instant_pace_reward_v5 as v5
from environment import instant_pace_reward_v6 as v6


CHANGED = ("overspeed", "lane_deviation", "steer_rate")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def steer_cost_evidence(row, aid, previous, timestep):
    """Return old cost and V6 bounds, using applied controls when recorded."""
    post = row["post"].get(aid)
    if post is not None:
        current = post["control"][1]
        old = (0.0 if previous is None else
               -.001 * min(1., ((current-previous)/(2*timestep))**2))
        new, _ = v6.steer_rate_penalty(current, previous, timestep=timestep)
        return old, new, new, "applied_readback"
    if row["info"]["active_mask"][aid]:
        raise ValueError("missing applied control for a still-active vehicle")
    # Arrival despawns the actor before the trace's post snapshot.  A capped
    # old risk cannot recover the original command delta.  Keep that ambiguity.
    risk = float(row["info"]["reward_diagnostics"][aid]["steer_rate_normalized"])
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        raise ValueError("invalid recorded V5 steering risk")
    old = -.001 * risk
    upper = -v6.STEER_RATE_COST_PER_SECOND * timestep * risk
    if risk < 1:
        return old, upper, upper, "terminal_unsaturated_risk"
    max_risk = (2/(v6.STEER_REFERENCE_RATE_PER_SECOND*timestep))**2
    lower = -v6.STEER_RATE_COST_PER_SECOND * timestep * max_risk
    return old, lower, upper, "terminal_clipped_risk_bounds"


def audit(evaluation_dir):
    directory = Path(evaluation_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "COMPLETED" or len(manifest["checkpoints"]) != 1:
        raise ValueError("requires a completed, single-checkpoint V5 evaluation")
    environment = manifest["checkpoints"][0]["environment"]
    if environment["pace_profile"] != "instant_pace_v5":
        raise ValueError("recorded source must be instant_pace_v5")
    dt = environment["timestep"]
    paths = sorted((directory / "traces").glob("*.jsonl"))
    if len(paths) != manifest["completed_cases"]:
        raise ValueError("trace count differs from completed cases")
    totals = {a: {"active_rows": 0, "recorded_total": 0.0, "rescored_total_upper": 0.0,
                  "steer_uncertainty_lower_adjustment": 0.0, "steer_evidence": {},
                  "requested_applied_steer_differing_rows": 0,
                  "recorded_components": {k: 0.0 for k in CHANGED},
                  "rescored_components_upper": {k: 0.0 for k in CHANGED}}
              for a in ("0", "1")}
    errors = {name: 0.0 for name in (*CHANGED, "recorded_decomposition", "replacement")}
    for path in paths:
        active = {a: True for a in totals}
        previous = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["type"] != "step":
                continue
            for aid, values in totals.items():
                # Use the mask before this transition: include own terminal
                # step, exclude all later steps while the teammate continues.
                if not active[aid]:
                    continue
                diagnostics = row["info"]["reward_diagnostics"][aid]
                details = row["info"]["reward_details"][aid]
                lane_kwargs = dict(
                    timestep=dt, exempt=bool(diagnostics["lane_deviation_exempted"]),
                    road_violation=bool(diagnostics["road_violation_raw"]),
                    opposing_lane=diagnostics["road_current_to_reference_cosine"] <= -.5,
                    in_junction=bool(diagnostics["road_in_junction"]),
                )
                speed = diagnostics["speed_mps"]
                distance = diagnostics["lateral_distance_m"]
                prior = previous.get(aid)
                old_steer, turn_lower, turn_upper, evidence = steer_cost_evidence(row, aid, prior, dt)
                evidence_counts = values["steer_evidence"]
                evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1
                values["steer_uncertainty_lower_adjustment"] += turn_lower-turn_upper
                post = row["post"].get(aid)
                if post is not None:
                    steer = post["control"][1]
                    values["requested_applied_steer_differing_rows"] += int(
                        abs(steer-row["environment_action"][aid][1]) > 1e-6)
                    previous[aid] = steer
                old = dict(
                    overspeed=v5.continuous_overspeed_penalty(speed, timestep=dt)[0],
                    lane_deviation=v5.continuous_lane_deviation_penalty(distance, **lane_kwargs)[0],
                    steer_rate=old_steer,
                )
                new = dict(
                    overspeed=v6.continuous_overspeed_penalty(speed, timestep=dt)[0],
                    lane_deviation=v6.continuous_lane_deviation_penalty(distance, **lane_kwargs)[0],
                    steer_rate=turn_upper,
                )
                recorded = math.fsum(details.values())
                rescored = math.fsum({**details, **new}.values())
                replacement = recorded + math.fsum(new[k]-details[k] for k in CHANGED)
                errors["recorded_decomposition"] = max(
                    errors["recorded_decomposition"], abs(recorded-row["reward"][aid]))
                errors["replacement"] = max(errors["replacement"], abs(rescored-replacement))
                values["active_rows"] += 1
                values["recorded_total"] += recorded
                values["rescored_total_upper"] += rescored
                for name in CHANGED:
                    errors[name] = max(errors[name], abs(old[name]-details[name]))
                    values["recorded_components"][name] += details[name]
                    values["rescored_components_upper"][name] += new[name]
            active = {a: bool(row["info"]["active_mask"][a]) for a in totals}
    if not all(math.isfinite(value) and value <= 1e-9 for value in errors.values()):
        raise ValueError(f"recorded V5 formulas or replacement accounting disagree: {errors}")
    for values in totals.values():
        values["case_mean_recorded_return"] = values["recorded_total"] / len(paths)
        values["case_mean_rescored_return_bounds"] = [
            (values["rescored_total_upper"] + values["steer_uncertainty_lower_adjustment"])/len(paths),
            values["rescored_total_upper"]/len(paths),
        ]
        values["case_mean_recorded_components"] = {
            k: v/len(paths) for k, v in values["recorded_components"].items()}
        values["case_mean_rescored_components_upper"] = {
            k: v/len(paths) for k, v in values["rescored_components_upper"].items()}
    return {
        "evidence_status": "COUNTERFACTUAL_SCORING_ONLY",
        "scope": "fixed V5 trajectories; no new driving, policy update or efficacy/generalization claim",
        "evaluation_dir": str(directory),
        "checkpoint": manifest["checkpoints"][0],
        "case_count": len(paths), "active_boundary": "pre-transition active; include own terminal",
        "steer_evidence": "use applied post control; missing terminal post with saturated old risk yields physical command bounds, never an invented exact cost",
        "input_sha256": {str(p): digest(p) for p in [directory/"manifest.json", *paths]},
        "recorded_source_sha256": manifest["source_sha256"],
        "cost_source_sha256": {str(Path(m.__file__).resolve()): digest(m.__file__) for m in (v5, v6)},
        "reward_spec": v6.instant_pace_spec(), "max_abs_accounting_error": errors,
        "agents": totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [audit(directory) for directory in args.evaluation_dir]
    # Complete the audit before opening output, and never overwrite evidence.
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump({"evidence_status": "COUNTERFACTUAL_SCORING_ONLY", "evaluations": reports},
                  stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "COUNTERFACTUAL_AUDIT_COMPLETE", "output": args.output,
                      "evaluations": len(reports)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
