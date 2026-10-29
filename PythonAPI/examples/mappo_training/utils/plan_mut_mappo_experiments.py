"""Write a serial experiment queue and result contract; never launch a run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training.mut_mappo_config import load_config, portable_config, validate_config
from training.checkpoint_protocol import atomic_write_json
from training.mut_mappo_experiments import GROUPS, SUITES, experiment_config


def plan(config, *, phase="pilot", seeds=(42, 43, 44), suite="core", formal_budget_ticks=None):
    if phase not in {"pilot", "formal"}:
        raise ValueError("unknown experiment phase")
    if phase == "formal":
        if config.get("research_status") != "frozen":
            raise ValueError("formal queue requires a frozen research configuration; candidate configs are pilot-only")
        if isinstance(formal_budget_ticks, bool) or not isinstance(formal_budget_ticks, int) or formal_budget_ticks <= 0:
            raise ValueError("formal queue requires an explicit positive --train-tick-budget after pilot calibration")
        config = validate_config(config)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("outer training seeds must be nonempty and unique")
    selected = tuple(seeds[:1]) if phase == "pilot" else tuple(seeds)
    if suite not in SUITES:
        raise ValueError("unknown experiment suite")
    entries = []
    for seed in selected:
        for group in SUITES[suite]:
            run_id = f"MutMAPPO_{phase}_{group}_s{seed}"
            variant = experiment_config(portable_config(config), group)
            variant.update(seed=int(seed))
            mode = variant["method"]
            if phase == "pilot":
                # A candidate estimate only; align the three runs by the same
                # actual tick cap rather than claiming 25 updates are enough.
                variant["run"].update(rounds=25, train_tick_budget=120000)
            else:
                variant["run"]["train_tick_budget"] = formal_budget_ticks
            entries.append({"run_id": run_id, "group": group, "outer_seed": seed,
                "method": mode, "status": "PLANNED_NOT_RUN",
                "algorithm": variant["algorithm"], "team_count": variant["team_count"],
                "question": GROUPS[group]["question"], "suite": suite,
                "config": variant, "run_dir": "logs/mut_mappo/" + run_id,
                "training_budget": "A+B actual train world.tick including reset; report all discarded work",
                "required_evidence": ["checkpoint_manifest.json", "actual_cost_audit.jsonl", "metrics.jsonl",
                    "evaluations/selection/summary.json", "evaluations/test/summary.json"],
                "launch_module": "training.train_mut_mappo", "requires_user_started_carla": True})
    return {"schema_version": "mut_mappo_serial_queue_v2", "phase": phase, "suite": suite,
            "status": "CANDIDATE_QUEUE_NOT_SCIENTIFIC_RESULT", "max_concurrent_tick_owners": 1,
            "budget_policy": "complete_boundary_soft_cap_report_overshoot_and_all_discarded_work",
            "control_scope": "random/strength match own update's weights, not guaranteed identical cross-run realized KL",
            "entries": entries, "execution_order": "serial_shared_port_2000",
            "formal_readiness": "requires frozen research contract, qualified red and live geometry evidence",
            "later_deliverables": {"communication": "independent 2x2 training then fixed-policy packet-loss evaluation",
                                   "styles": "normal red style probes and admission evidence",
                                   "psro": "shared external payoff and one blue response; no equilibrium claim"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mut_mappo/local_candidate_v1.json")
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--suite", choices=tuple(SUITES), default="core")
    parser.add_argument("--train-tick-budget", type=int, help="required for formal after qualification and pilot calibration")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("queue/config evidence exists; use a new output directory")
    queue = plan(load_config(args.config), phase=args.phase, seeds=args.seeds, suite=args.suite,
                 formal_budget_ticks=args.train_tick_budget)
    output.mkdir(parents=True)
    for entry in queue["entries"]:
        config_path = output / (entry["run_id"] + ".json")
        atomic_write_json(entry.pop("config"), config_path)
        entry["config_path"] = str(config_path)
    atomic_write_json(queue, output / "queue.json")
    print(json.dumps({"status": queue["status"], "runs": len(queue["entries"]), "queue": str(output / "queue.json")}, indent=2))


if __name__ == "__main__":
    main()
