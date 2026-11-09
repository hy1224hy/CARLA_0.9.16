"""Prepare normal-traffic C1 curves, serially, before first-point training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from environment.mut_mappo_efficiency_reward import reference_identity
from training.checkpoint_protocol import atomic_write_json, sha256_file
from training.mut_mappo_config import load_config, validate_config, resolve_path, cases_for


def prepare(config_path, splits=("train", "gate", "selection"), *, allow_weak_red=False, repeats=1):
    from training.train_mut_mappo import create_runtime, run_case, append_record, now
    from utils.mut_mappo_session import CarlaOwner
    config = validate_config(load_config(config_path))
    config["_building_reference"] = True
    directory = resolve_path(config["task"]["team_reward"]["reference_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    identity = reference_identity(config)
    path = directory / "manifest.json"
    header = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "mut_mappo_C1_reference_v1", "identity": identity, "cases": {},
        "diagnostic_weak_red_override": bool(allow_weak_red),
        "normal_blue_controller": config['task'].get('c1_controller','frozen_actor')}
    if header["identity"] != identity:
        raise ValueError("reference directory is already bound to another experiment")
    if repeats < 1:
        raise ValueError("reference repeats must be positive")
    atomic_write_json(header, path)
    runtime = None
    context={'phase':'construction'}
    def audit(event):
        append_record(directory/'world_tick_audit.jsonl',dict(event,**context,at=now()))
    with CarlaOwner(config["host"], config["port"], run_dir=directory, purpose="C1 reference"):
        try:
            for split in splits:
                for case in cases_for(config, split):
                    filename = hashlib.sha256(case["case_key"].encode()).hexdigest() + ".json"
                    target = directory / filename
                    previous = header["cases"].get(case["case_key"])
                    if previous and target.exists() and sha256_file(target) == previous["sha256"]:
                        if previous.get("repeats", 1) != repeats:
                            raise ValueError("reference repeat count changed; preserve this experiment and prepare a new version")
                        if not previous["qualified"] and not allow_weak_red:
                            raise RuntimeError("saved C1 case failed normal-driving screen; diagnostic override is explicit")
                        continue
                    if runtime is None:
                        runtime = create_runtime(config, condition=('C2' if config['task'].get('reward_profile') == 'individual_three_dimension_risk_v4' else 'C1'), audit_callback=audit)
                    curves, results = [], []
                    for repetition in range(repeats):
                        context.update(phase='reference',case_key=case['case_key'],seed=case['seed']+repetition*10000)
                        curve = {"0": [0.0], "1": [0.0]}
                        response_states={};previous_red={};previous_acc={}
                        from evaluation.r4_risk_metrics import response_measures
                        def collect(info):
                            p = info["mut_metrics"]["route_progress"]
                            step=len(curve['0'])
                            snapshots=info.get('post_tick_kinematics',{})
                            response_states[str(step)]={}
                            for rid in (0,1):
                                snap=snapshots.get(rid,snapshots.get(str(rid)))
                                if snap is not None:
                                    if rid in previous_red:
                                        measures=response_measures(previous_red[rid],snap,float(config['task']['timestep']),previous_acc.get(rid))
                                        response_states[str(step)][str(rid)]=measures
                                        previous_acc[rid]=measures['acceleration_mps2']
                                    previous_red[rid]=snap
                            for rid in ("0", "1"):
                                curve[rid].append(float(p.get(rid, p.get(int(rid)))))
                        result = run_case(runtime, case, seed=case["seed"] + repetition * 10000, step_callback=collect)
                        if not result["valid"]:
                            raise RuntimeError("invalid C1 execution cannot define a reward reference")
                        curves.append(curve)
                        results.append(result)
                    qualified = all(r["red_success"] and r["blue_eligible"] for r in results)
                    if not qualified and not allow_weak_red:
                        raise RuntimeError(f"C1 is not normal/safe in {case['case_key']}; inspect red policy first, or use --allow-weak-red for diagnostics")
                    length = max(len(c["0"]) for c in curves)
                    progress = {rid: [sum(c[rid][min(i, len(c[rid]) - 1)] for c in curves) / repeats
                                      for i in range(length)] for rid in ("0", "1")}
                    record = {"case_key": case["case_key"], "split": split, "qualified": qualified,
                        "identity": identity, "progress": progress, "reference_runs": results,
                        "response_states": response_states if repeats == 1 else {},
                        "response_reference_mode": "frozen_base_blue" if config["task"].get("reward_profile") == "individual_three_dimension_risk_v4" else "normal_C1",
                        "scope": "normal traffic reference; qualified here means C1 reached and blue-safe, not full base-driving qualification"}
                    atomic_write_json(record, target)
                    header["cases"][case["case_key"]] = {"file": filename, "sha256": sha256_file(target),
                                                          "qualified": qualified, "split": split, "repeats": repeats}
                    atomic_write_json(header, path)
                    print("REFERENCE_DONE", case["case_key"], "NORMAL" if qualified else "DIAGNOSTIC_WEAK", flush=True)
        finally:
            if runtime is not None:
                context['phase']='cleanup'
                runtime.close()
    return {"directory": str(directory), "cases": len(header["cases"]),
            "diagnostic_weak_cases": sum(not row["qualified"] for row in header["cases"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "gate", "selection", "test"), default=["train", "gate", "selection"])
    parser.add_argument("--allow-weak-red", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.splits, allow_weak_red=args.allow_weak_red, repeats=args.repeats), indent=2))


if __name__ == "__main__":
    main()
