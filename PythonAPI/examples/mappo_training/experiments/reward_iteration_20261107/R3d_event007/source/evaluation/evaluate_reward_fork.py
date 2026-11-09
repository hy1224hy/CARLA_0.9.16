"""Matched 12-scenario behavioral evaluation for frozen-v6 and route-pace forks.

Uses the existing per-step trace engine without changing historical C0 contracts.
This is a reward screening evaluation, not a claim of unseen-scene generalization.
"""
import argparse
import json
from pathlib import Path
import traceback

from training.checkpoint_protocol import (
    load_checkpoint_file, validate_checkpoint_header, atomic_write_json,
    sha256_file, utc_now_iso,
)
from training.train_no_rllib_mappo_robust_v2 import checkpoint_protocol_versions
from evaluation.diagnose_r8c1_behavior import make_actors, run_case
from evaluation.evaluate_base_policy_c0 import (
    _preflight_idle_world, _cleanup_environment, _restore_failed_initialization,
)
from environment.carla_multi_agent_env import CarlaMultiAgentEnv

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_environment(checkpoint):
    validate_checkpoint_header(checkpoint)
    cfg = checkpoint["training_config"]
    profile = cfg.get("pace_profile", "frozen_v6")
    if cfg["reward_version"] != "v6" or profile not in {"frozen_v6", "route_pace_v1", "instant_pace_v2", "instant_pace_v3", "instant_pace_v4", "instant_pace_v5", "instant_pace_v6"}:
        raise ValueError("this evaluator supports only registered v6 reward profiles")
    expected = checkpoint_protocol_versions("v6", cfg["lidar_range_m"], cfg["curriculum_protocol"])
    if profile != "frozen_v6":
        expected["pace_reward"] = profile
    if checkpoint["protocol_versions"] != expected:
        raise ValueError("evaluation/checkpoint protocol mismatch")
    if checkpoint["model_spec"]["obs_dim"] != 20 or checkpoint["model_spec"]["action_dim"] != 2:
        raise ValueError("evaluation requires the unchanged 20D observation and 2D policy")
    return {
        **{key: cfg[key] for key in ("town", "straight_road_length", "timestep", "max_steps",
                                     "lidar_range_m", "ttc_max_range_m", "reward_version")},
        "reward_gamma": cfg["gamma"], "pace_profile": profile,
        "num_agents": 2, "spawn_vehicles": False, "disable_tensorboard": True,
        "follow_spectator": True,
    }


def summarize_trace(path, timestep):
    arrivals = {"0": None, "1": None}
    at_first_arrival = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["type"] != "step":
                continue
            diagnostics = row["info"]["reward_diagnostics"]
            for aid in ("0", "1"):
                if diagnostics[aid].get("success_event") and arrivals[aid] is None:
                    arrivals[aid] = row["step"] * timestep
            if at_first_arrival is None and any(v is not None for v in arrivals.values()):
                at_first_arrival = {aid: diagnostics[aid].get("route_completion") for aid in ("0", "1")}
    return {"arrival_seconds": arrivals,
            "completion_at_first_arrival": at_first_arrival,
            "arrival_gap_seconds": abs(arrivals["0"]-arrivals["1"]) if all(v is not None for v in arrivals.values()) else None}


def evaluate(paths, output, seeds=(62001, 62002), mode="deterministic", validate_only=False):
    loaded = []
    for path in paths:
        checkpoint, digest = load_checkpoint_file(path, map_location="cpu")
        environment = checkpoint_environment(checkpoint)
        actors = make_actors(checkpoint)
        loaded.append((str(Path(path).resolve()), digest, checkpoint, environment, actors))
    scenarios = {p.stem: {"path": str(p), "sha256": sha256_file(p),
                         "config": json.loads(p.read_text(encoding="utf-8"))}
                 for p in sorted((ROOT / "data/training_scenarios_town05_r8a_v1").glob("scenario_*.json"))}
    if len(scenarios) != 12:
        raise ValueError("expected 12 fixed screening scenarios")
    if validate_only:
        print(json.dumps({"status": "OFFLINE_VALIDATED", "checkpoints": len(loaded),
                          "cases": len(loaded)*len(scenarios)*len(seeds),
                          "profiles": [item[3]["pace_profile"] for item in loaded]}, indent=2))
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    config = {"version": "reward_fork_behavior_v1", "scenarios": scenarios,
              "seeds": list(seeds), "mode": mode,
              "thresholds": {"low_speed_mps": 0.2, "startup_speed_mps": 0.5}}
    manifest = {"status": "RUNNING", "started_at": utc_now_iso(),
                "formal_eligibility": False, "completed_cases": 0,
                "checkpoints": [{"path": p, "sha256": h, "update": c["update"], "environment": e}
                                for p,h,c,e,_ in loaded],
                "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                    Path(__file__), ROOT/"evaluation/diagnose_r8c1_behavior.py",
                    ROOT/"environment/carla_multi_agent_env.py", ROOT/"environment/route_pace_reward.py",
                    ROOT/"environment/instant_pace_reward.py",
                    ROOT/"environment/instant_pace_reward_v3.py",
                    ROOT/"environment/instant_pace_reward_v4.py",
                    ROOT/"environment/instant_pace_reward_v5.py",
                    ROOT/"environment/instant_pace_reward_v6.py")}}
    atomic_write_json(config, output / "resolved_config.json")
    atomic_write_json(manifest, output / "manifest.json")
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as episodes:
            for index, (source, digest, checkpoint, environment, actors) in enumerate(loaded):
                env = None
                idle = None
                try:
                    idle = _preflight_idle_world(host="127.0.0.1", port=2000, expected_version="0.9.16")
                    env = CarlaMultiAgentEnv(host="127.0.0.1", port=2000, seed=seeds[0], **environment)
                    env._original_synchronous_mode = idle["synchronous_mode"]
                    env._original_fixed_delta_seconds = idle["fixed_delta_seconds"]
                    for scenario in scenarios:
                        for seed in seeds:
                            case = {"id": f"c{index}_u{checkpoint['update']}_{scenario}_s{seed}_{mode}",
                                    "checkpoint": source, "scenario": scenario, "seed": seed, "mode": mode}
                            trace = output / "traces" / (case["id"] + ".jsonl")
                            result = run_case(env, actors, case, config, trace)
                            result.update(summarize_trace(trace, environment["timestep"]))
                            result["checkpoint_sha256"] = digest
                            result["pace_profile"] = environment["pace_profile"]
                            episodes.write(json.dumps(result, allow_nan=False) + "\n")
                            episodes.flush()
                            manifest["completed_cases"] += 1
                            atomic_write_json(manifest, output / "manifest.json")
                            print("CASE_DONE", case["id"], result["reason"], flush=True)
                finally:
                    error = ""
                    if env is not None:
                        audit, error = _cleanup_environment(env)
                    elif idle is not None:
                        audit, error = _restore_failed_initialization(host="127.0.0.1", port=2000, preflight=idle)
                    else:
                        audit = {}
                    manifest.setdefault("cleanup", []).append({"audit": audit, "error": error})
                    if error:
                        raise RuntimeError(error)
                if sha256_file(source) != digest:
                    raise RuntimeError("checkpoint changed during evaluation")
        manifest["status"] = "COMPLETED"
    except BaseException:
        manifest["status"] = "FAILED"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["ended_at"] = utc_now_iso()
        atomic_write_json(manifest, output / "manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seeds", nargs="+", type=int, default=[62001, 62002])
    parser.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.output and not args.validate_only:
        parser.error("--output is required for live evaluation")
    evaluate(args.checkpoints, args.output, args.seeds, args.mode, args.validate_only)
