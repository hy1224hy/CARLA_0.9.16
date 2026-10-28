"""Four frozen policy interventions: keep four cars and measure team contribution."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.driving_policy_metrics import average
from training.checkpoint_protocol import atomic_write_json, sha256_file
from training.mut_mappo_config import load_config, validate_config, resolve_path, cases_for, make_actors


def credit_summary(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = row["team"], row["case_key"]
        combination = row["credit_combination"]
        if combination in groups[key]:
            raise ValueError("duplicate credit intervention")
        groups[key][combination] = row
    complete, incomplete = [], []
    for (team, case), members in groups.items():
        if set(members) != {"00", "10", "01", "11"} or not all(r["valid"] for r in members.values()):
            incomplete.append({"team": team, "case_key": case})
            continue
        utility = {key: float(row["team_utility_total"]) for key, row in members.items()}
        u00, u10, u01, u11 = (utility[key] for key in ("00", "10", "01", "11"))
        complete.append({"team": team, "case_key": case, "family": members["11"]["family"],
            "phi_blue2": .5 * ((u10 - u00) + (u11 - u01)),
            "phi_blue3": .5 * ((u01 - u00) + (u11 - u10)),
            "interaction": u11 - u10 - u01 + u00, "joint_gain": u11 - u00,
            "utility": utility, "all_four_blue_safe": all(r["blue_eligible"] for r in members.values())})
    safe = [row for row in complete if row["all_four_blue_safe"]]
    return {"schema": "mut_mappo_credit_v1", "status": "COMPLETE" if not incomplete else "INCOMPLETE",
        "definition": "paired frozen-policy interventions, not exact state/action counterfactuals; negative contributions retained",
        "n_complete": len(complete), "incomplete": incomplete, "cases": complete,
        "means": {key: average([r[key] for r in complete]) for key in ("phi_blue2", "phi_blue3", "interaction", "joint_gain")},
        "all_four_safe_count": len(safe),
        "all_four_safe_means": {key: average([r[key] for r in safe]) for key in ("phi_blue2", "phi_blue3", "interaction", "joint_gain")},
        "teams_are_independent_seeds": False}


def evaluate(config_path, checkpoint_path, output_dir, *, split="test", team="both", monitor_options=None):
    from training.train_mut_mappo import create_runtime, run_case, validate_checkpoint_for_config, append_record
    from utils.mut_mappo_session import CarlaOwner
    config = validate_config(load_config(config_path))
    payload = validate_checkpoint_for_config(config, checkpoint_path)
    teams = sorted(payload["teams"]) if "teams" in payload else [payload["team"]]
    if team != "both":
        if team not in teams:
            raise ValueError("requested team absent from checkpoint")
        teams = [team]
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = config["_red_payload"]
    ordinary = make_actors({2: source["actors"][0], 3: source["actors"][1]}, source["training_config"], (2, 3), config["device"])
    rows = []
    from visualization.mut_mappo_monitor import monitored_evaluation
    with monitored_evaluation(config, output, monitor_options, operation="credit") as monitor, \
         CarlaOwner(config["host"], config["port"], run_dir=output, purpose="four-policy credit evaluation"):
        for name in teams:
            runtime = None
            try:
                runtime = create_runtime(config, payload, condition="C2", team=name)
                runtime.monitor = monitor
                attack = dict(runtime.blue_actors)
                for case in cases_for(config, split):
                    for combination in ("00", "10", "01", "11"):
                        monitor.context(credit_combination=combination)
                        runtime.blue_actors = {aid: attack[aid] if combination[aid - 2] == "1" else ordinary[aid]
                                               for aid in (2, 3)}
                        row = run_case(runtime, case, seed=case["seed"])
                        row.update(credit_combination=combination, split=split)
                        append_record(output / "cases.jsonl", row)
                        rows.append(row)
                        if not row["valid"]:
                            raise RuntimeError("invalid intervention; do not score incomplete four-case comparisons")
                        if "team_utility_total" not in row:
                            raise ValueError("credit evaluation requires the shared team reward profile")
                        print("CREDIT_DONE", name, case["case_key"], combination, flush=True)
            finally:
                if runtime is not None:
                    runtime.close()
        report = credit_summary(rows)
        report["checkpoint_sha256"] = sha256_file(resolve_path(checkpoint_path))
        report["identity"] = config["_root_validation"]
        report["source_sha256"] = sha256_file(__file__)
        report["split"] = split
        atomic_write_json(report, output / "summary.json")
        return report


def main():
    from visualization.mut_mappo_monitor import add_monitor_arguments, options_from_args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("selection", "test"), default="test")
    parser.add_argument("--team", choices=("A", "B", "both"), default="both")
    add_monitor_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config, args.checkpoint, args.output_dir, split=args.split, team=args.team,
                              monitor_options=options_from_args(args)), indent=2))


if __name__ == "__main__":
    main()
