"""Serial, frozen evaluation for the explicitly versioned base-driving task.

The old reward-fork evaluator and its fixed twelve-scene contract are untouched.
--validate-only loads files/weights but never imports or connects to CARLA.
Live execution holds the shared endpoint lease and reuses the established
active-transition trace engine. Historical checkpoints require an explicit
reference flag and can never receive qualification through this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

from evaluation.base_driving_qualification import (
    DEFAULT_CONTRACT, PROTOCOL, ROOT, _metadata, analyze_evaluation,
    load_contract, read_json, sha256_file,
)

POOL_PROTOCOL = "town05_base_driving_matrix_v1"


def load_scenarios(manifest_path):
    """Load an immutable pool plus complete physical actor-swap pairs."""
    manifest_path = Path(manifest_path).resolve()
    pool = read_json(manifest_path)
    if pool.get("pool_protocol") != POOL_PROTOCOL or pool.get("split") not in {"train", "validation"}:
        raise ValueError("unsupported scenario pool protocol or split")
    directory = manifest_path.parent.parent
    expected = pool["scenario_sha256"]
    actual_names = {p.name for p in directory.glob("scenario_*.json")}
    if set(expected) != actual_names or len(expected) != pool["scenario_count"]:
        raise ValueError("pool files differ from frozen manifest")
    scenarios = {}
    groups = {}
    for name, digest in sorted(expected.items()):
        path = (directory / name).resolve()
        if path.parent != directory or sha256_file(path) != digest:
            raise ValueError("scenario file path or SHA mismatch")
        config = read_json(path)
        if config.get("goal_mode") != "per_agent_explicit" or config.get("split") != pool["split"]:
            raise ValueError("new pool requires explicit per-agent goals and the registered split")
        for aid in ("0", "1"):
            _metadata(config, aid)
        groups.setdefault(config["pair_id"], []).append(name)
        scenarios[path.stem] = {"path": str(path), "sha256": digest, "config": config}
    for pair, names in groups.items():
        if len(names) != 2:
            raise ValueError(f"incomplete numbered mirror pair: {pair}")
        for name in names:
            scene = scenarios[Path(name).stem]["config"]
            if scene["symmetry_partner"] not in names or scene["symmetry_partner"] == name:
                raise ValueError("numbered mirror mapping is not reciprocal")
            partner = scenarios[Path(scene["symmetry_partner"]).stem]["config"]
            if partner["symmetry_partner"] != name or scene["physical_template_id"] != partner["physical_template_id"]:
                raise ValueError("mirror pair changed physical template or reciprocal mapping")
            for left, right in (("agent0", "agent1"), ("agent1", "agent0")):
                for field in ("agents", "agent_goal_zones", "agent_roles"):
                    if scene[field][left] != partner[field][right]:
                        raise ValueError("mirror must swap whole physical state, own goal and role")
    return pool, scenarios, sha256_file(manifest_path)


def select_evaluation_scenarios(pool, scenarios, pool_manifest_sha256, *, parallel_regression=False):
    """Bind a complete suite or the registered parallel subset to its source pool."""
    if not isinstance(parallel_regression, bool):
        raise ValueError("parallel_regression must be an explicit boolean")
    selected = scenarios
    kind = "complete_registered_pool"
    if parallel_regression:
        if pool.get("split") != "train" or pool.get("scenario_count") != 48 or len(scenarios) != 48:
            raise ValueError("parallel regression requires the complete registered 48-scene training pool")
        selected = {name: row for name, row in scenarios.items() if row["config"]["curriculum_stage"] == 0}
        if len(selected) != 16 or any(row["config"]["delta_s_m"] != 0 for row in selected.values()):
            raise ValueError("parallel regression must contain the sixteen stage-zero parallel scenes")
        names = {Path(row["path"]).name for row in selected.values()}
        if any(row["config"]["symmetry_partner"] not in names for row in selected.values()):
            raise ValueError("parallel regression cannot split an actor-swap pair")
        kind = "parallel_stage0_regression"
    definition = {"kind": kind, "pool_manifest_sha256": pool_manifest_sha256,
                  "source_scenario_count": len(scenarios), "scenario_names": sorted(selected)}
    digest = (hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
              if parallel_regression else pool_manifest_sha256)
    return selected, digest, definition


def checkpoint_environment(checkpoint, contract, contract_sha256, *, allow_legacy_reference=False):
    """Validate an actor checkpoint without importing the simulator or trainer."""
    from training.checkpoint_protocol import (
        R6_TIME_AWARE_OBSERVATION_PROTOCOL_VERSION, validate_checkpoint_header,
    )
    from training.action_protocol import ACTION_PROTOCOL_VERSION

    validate_checkpoint_header(checkpoint)
    cfg = checkpoint["training_config"]
    spec = checkpoint["model_spec"]
    expected_spec = contract["checkpoint"]["required_model_spec"]
    if any(spec.get(k) != value for k, value in expected_spec.items()) or spec.get("num_agents") != 2:
        raise ValueError("base-driving evaluator requires registered 20D/2D two-agent actors")
    versions = checkpoint["protocol_versions"]
    if (cfg.get("reward_version") != "v6" or cfg.get("pace_profile") != "instant_pace_v6" or
            float(cfg.get("lidar_range_m", -1)) != 20.0 or versions.get("reward") != "v6" or
            versions.get("action") != ACTION_PROTOCOL_VERSION or
            versions.get("observation") != R6_TIME_AWARE_OBSERVATION_PROTOCOL_VERSION or
            versions.get("pace_reward") != "instant_pace_v6"):
        raise ValueError("checkpoint action/observation/reward protocol differs from base driving evaluator")
    binding = checkpoint.get("base_driving_contract")
    if binding is None:
        if not allow_legacy_reference:
            raise ValueError("historical checkpoint requires --allow-legacy-reference; it can never qualify")
        eligible = False
        profile = "v6"
    else:
        if (binding.get("protocol") != PROTOCOL or
                binding.get("qualification_contract_sha256") != contract_sha256 or
                cfg.get("base_driving_protocol") != PROTOCOL or versions.get("base_driving") != PROTOCOL):
            raise ValueError("new checkpoint base-driving contract/protocol binding differs from evaluation")
        digest = binding.get("scenario_pool_sha256", "")
        if len(digest) != 64 or any(v not in "0123456789abcdef" for v in digest):
            raise ValueError("checkpoint training scenario pool identity is absent")
        profile = cfg.get("base_driving_reward_profile")
        if profile not in contract["checkpoint"]["accepted_base_driving_reward_profiles"]:
            raise ValueError("unknown new base-driving reward profile")
        if binding.get("reward_profile") != profile:
            raise ValueError("checkpoint reward profile differs from its own registered contract")
        eligible = True
    environment = {key: cfg[key] for key in ("town", "straight_road_length", "timestep", "max_steps",
                                            "lidar_range_m", "ttc_max_range_m", "reward_version")}
    environment.update(reward_gamma=cfg["gamma"], pace_profile="instant_pace_v6",
                       base_driving_reward_profile=profile, num_agents=2, spawn_vehicles=False,
                       disable_tensorboard=True, follow_spectator=True)
    return environment, eligible


def evaluate(paths, output, scenario_manifest, contract_path=DEFAULT_CONTRACT, *,
             seeds=None, validate_only=False, allow_legacy_reference=False,
             geometry_report=None, parallel_regression=False):
    """Run one serial suite; default seeds come from the frozen contract."""
    from training.checkpoint_protocol import load_checkpoint_file, atomic_write_json, utc_now_iso

    contract = load_contract(contract_path)
    contract_sha = sha256_file(contract_path)
    pool, all_scenarios, pool_sha = load_scenarios(scenario_manifest)
    geometry = {"status": "NOT_VERIFIED"}
    if geometry_report is not None:
        from utils.preflight_base_driving_scenarios_v1 import require_geometry_preflight
        require_geometry_preflight(Path(scenario_manifest).resolve().parent.parent, geometry_report)
        geometry = {"status": "PASS_GEOMETRY_ONLY", "report_path": str(Path(geometry_report).resolve()),
                    "report_sha256": sha256_file(geometry_report)}
    elif not validate_only:
        raise ValueError("live evaluation requires --geometry-report for this exact frozen scenario pool")
    # Validate the full manifest and geometry before choosing the stage-zero
    # subset. A partial preflight cannot license this regression screen.
    scenarios, suite_sha, suite_definition = select_evaluation_scenarios(
        pool, all_scenarios, pool_sha, parallel_regression=parallel_regression)
    seeds = list(contract["evaluation"]["seeds"] if seeds is None else seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(isinstance(v, bool) or not isinstance(v, int) for v in seeds):
        raise ValueError("evaluation seeds must be a nonempty unique integer sequence")
    # A small smoke run remains useful but is marked incomplete qualification
    # coverage. It cannot obtain a different implicit acceptance threshold.
    loaded = []
    for path in paths:
        checkpoint, digest = load_checkpoint_file(path, map_location="cpu")
        environment, eligible = checkpoint_environment(checkpoint, contract, contract_sha,
                                                       allow_legacy_reference=allow_legacy_reference)
        if parallel_regression and eligible and checkpoint["base_driving_contract"]["scenario_pool_sha256"] != pool_sha:
            raise ValueError("parallel regression source pool differs from the checkpoint training pool")
        loaded.append((str(Path(path).resolve()), digest, checkpoint, environment, eligible))
    if not loaded or len({item[1] for item in loaded}) != len(loaded):
        raise ValueError("a nonempty set of distinct checkpoint identities is required")
    if validate_only:
        result = {"status": "OFFLINE_VALIDATED_NOT_CARLA_RUN", "protocol": PROTOCOL,
                  "checkpoints": len(loaded), "cases": len(loaded) * len(scenarios) * len(seeds),
                  "contract_sha256": contract_sha, "scenario_suite_sha256": suite_sha,
                  "scenario_pool_manifest_sha256": pool_sha, "scenario_suite_definition": suite_definition,
                  "parallel_regression": parallel_regression,
                  "qualification_eligible": [item[4] for item in loaded],
                  "scope": pool["evaluation_scope"], "live_geometry": geometry}
        print(json.dumps(result, indent=2))
        return result

    # Live-only imports: these modules expose CARLA classes. Keeping them here
    # makes file validation, qualification and unit tests simulator-independent.
    from evaluation.diagnose_r8c1_behavior import make_actors, run_case
    from evaluation.evaluate_base_policy_c0 import (
        _preflight_idle_world, _cleanup_environment, _restore_failed_initialization,
    )
    from environment.base_driving_env import BaseDrivingEnv
    from utils.mut_mappo_session import CarlaOwner

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse evaluation directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    # Preserve the exact preflight bytes as evidence alongside the suite.
    (output / "geometry_preflight.json").write_bytes(Path(geometry_report).read_bytes())
    config = {"version": SCHEMA_EVALUATION, "protocol": PROTOCOL, "scenarios": scenarios,
              "seeds": seeds, "mode": contract["evaluation"]["mode"],
              "suite_role": "validation" if pool["split"] == "validation" else "training_screen",
              "contract": contract, "contract_sha256": contract_sha,
              "scenario_suite_sha256": suite_sha, "scenario_pool_manifest": pool,
              "scenario_pool_manifest_sha256": pool_sha, "scenario_suite_definition": suite_definition,
              "parallel_regression": parallel_regression,
              "scenario_manifest_path": str(Path(scenario_manifest).resolve()),
              "geometry_preflight": geometry,
              "thresholds": {name: contract["thresholds"][name] for name in ("low_speed_mps", "startup_speed_mps")}}
    atomic_write_json(config, output / "resolved_config.json")
    source_paths = [Path(__file__), ROOT / "evaluation/base_driving_qualification.py",
                    ROOT / "evaluation/diagnose_r8c1_behavior.py", ROOT / "environment/base_driving_env.py",
                    ROOT / "environment/carla_multi_agent_env.py", ROOT / "environment/instant_pace_reward.py",
                    ROOT / "environment/instant_pace_reward_v4.py", ROOT / "environment/instant_pace_reward_v5.py",
                    ROOT / "environment/instant_pace_reward_v6.py", ROOT / "environment/instant_pace_reward_v3.py",
                    ROOT / "environment/base_reward_v4.py", ROOT / "environment/base_reward_v5.py",
                    ROOT / "environment/base_reward_v6.py", ROOT / "training/action_protocol.py",
                    ROOT / "models/actor_critic.py", ROOT / "utils/preflight_base_driving_scenarios_v1.py"]
    extra_reward = ROOT / "environment/base_driving_reward.py"
    if extra_reward.exists():
        source_paths.append(extra_reward)
    manifest = {"protocol": PROTOCOL, "status": "RUNNING", "started_at": utc_now_iso(),
                "formal_eligibility": False, "contract_sha256": contract_sha,
                "scenario_suite_sha256": suite_sha, "completed_cases": 0,
                "scenario_pool_manifest_sha256": pool_sha, "scenario_suite_definition": suite_definition,
                "parallel_regression": parallel_regression,
                "geometry_preflight": geometry,
                "resolved_config_sha256": sha256_file(output / "resolved_config.json"),
                "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
                "checkpoints": [{"path": p, "sha256": digest, "update": c["update"],
                                 "environment": env, "qualification_eligible": eligible,
                                 "base_driving_contract": c.get("base_driving_contract"),
                                 "reference_only": not eligible}
                                for p, digest, c, env, eligible in loaded]}
    atomic_write_json(manifest, output / "manifest.json")
    try:
        with CarlaOwner(run_dir=output, purpose="frozen versioned base driving qualification"):
            with (output / "episodes.jsonl").open("x", encoding="utf-8") as episodes:
                for index, (source, digest, checkpoint, environment, eligible) in enumerate(loaded):
                    actors = make_actors(checkpoint)
                    env = idle = None
                    try:
                        idle = _preflight_idle_world(host="127.0.0.1", port=2000, expected_version="0.9.16")
                        env = BaseDrivingEnv(host="127.0.0.1", port=2000, seed=seeds[0], **environment)
                        env._original_synchronous_mode = idle["synchronous_mode"]
                        env._original_fixed_delta_seconds = idle["fixed_delta_seconds"]
                        for scenario in scenarios:
                            for seed in seeds:
                                case = {"id": f"c{index}_u{checkpoint['update']}_{scenario}_s{seed}_deterministic",
                                        "checkpoint": source, "scenario": scenario, "seed": seed,
                                        "mode": config["mode"]}
                                trace = output / "traces" / (case["id"] + ".jsonl")
                                result = run_case(env, actors, case, config, trace)
                                result.update(checkpoint_sha256=digest, pace_profile=environment["pace_profile"],
                                              base_driving_reward_profile=environment["base_driving_reward_profile"])
                                episodes.write(json.dumps(result, allow_nan=False) + "\n")
                                episodes.flush()
                                manifest["completed_cases"] += 1
                                atomic_write_json(manifest, output / "manifest.json")
                                print("CASE_DONE", case["id"], result["reason"], flush=True)
                    finally:
                        if env is not None:
                            audit, error = _cleanup_environment(env)
                        elif idle is not None:
                            audit, error = _restore_failed_initialization(host="127.0.0.1", port=2000, preflight=idle)
                        else:
                            audit, error = {}, "initial preflight did not complete"
                        manifest.setdefault("cleanup", []).append({"audit": audit, "error": error})
                        if error:
                            raise RuntimeError(error)
                    if sha256_file(source) != digest:
                        raise RuntimeError("frozen checkpoint changed during evaluation")
        manifest["status"] = "COMPLETED"
    except BaseException:
        manifest["status"] = "FAILED"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["ended_at"] = utc_now_iso()
        atomic_write_json(manifest, output / "manifest.json")
    report = analyze_evaluation(output, contract_path)
    atomic_write_json(report, output / "qualification.json")
    return report


SCHEMA_EVALUATION = "base_driving_frozen_evaluation_vnext_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-legacy-reference", action="store_true")
    parser.add_argument("--geometry-report", type=Path)
    parser.add_argument("--parallel-regression", action="store_true",
                        help="Training pool only: evaluate all sixteen stage-zero parallel scenes as a retention screen")
    args = parser.parse_args()
    if not args.validate_only and not args.output:
        parser.error("--output is required for live evaluation")
    if not args.validate_only and not args.geometry_report:
        parser.error("--geometry-report is required for live evaluation")
    evaluate(args.checkpoints, args.output, args.scenario_manifest, args.contract,
             seeds=args.seeds, validate_only=args.validate_only,
             allow_legacy_reference=args.allow_legacy_reference, geometry_report=args.geometry_report,
             parallel_regression=args.parallel_regression)


if __name__ == "__main__":
    main()
