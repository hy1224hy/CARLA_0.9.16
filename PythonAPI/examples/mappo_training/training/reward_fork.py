"""Offline creation of a reward-only fork, followed by ordinary strict resume."""
import argparse
import copy
import json
from pathlib import Path

from training.checkpoint_protocol import (
    CheckpointProtocolError, atomic_write_json, files_fingerprint, get_git_metadata,
    load_checkpoint_file, save_checkpoint_file, utc_now_iso,
    validate_checkpoint_header, validate_checkpoint_payload,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/P0_BASE_MAPPO_Town05R8C1PaceU100_s42_20260903_180224/latest.pt"
SOURCE_SHA = "eea971a82ef6293357161fad40bcf58fa91663a9282c60dedbcfc78193dfe371"


def fork_payload(source, *, source_path, digest, run_id, profile, fingerprint, git_sha):
    validate_checkpoint_header(source)
    if source["training_config"]["reward_version"] != "v6":
        raise CheckpointProtocolError("reward fork requires v6 source")
    if profile not in {"route_pace_v1", "frozen_v6"}:
        raise CheckpointProtocolError("unsupported fork reward profile")
    if source["run_metadata"]["run_id"] == run_id:
        raise CheckpointProtocolError("fork requires a different RunID")
    result = copy.deepcopy(source)
    cfg = result["training_config"]
    if profile == "frozen_v6":
        cfg.pop("pace_profile", None)
        result["protocol_versions"].pop("pace_reward", None)
    else:
        cfg["pace_profile"] = profile
        result["protocol_versions"]["pace_reward"] = profile
    # No old-reward episode or last_metrics is attributed to the new experiment.
    result["last_metrics"] = None
    result["checkpoint_kind"] = "reward_fork_seed"
    fork = {
        "version": "reward_fork_v1", "source_checkpoint": str(source_path),
        "source_sha256": digest, "source_run_id": source["run_metadata"]["run_id"],
        "source_update": int(source["update"]), "profile": profile,
        "source_total_episodes": int(source["total_episodes"]),
        "state_transfer": "actors/critic/optimizers/ValueNorm/RNG/curriculum/sampler/LR+entropy phase retained",
        "counter_semantics": "inherited lifetime counters; new-run metrics begin after source_update",
        "source_metadata": copy.deepcopy(source["run_metadata"]),
        "scope": "reward screening with unchanged 20D observation; not a fresh-training result",
    }
    result["run_metadata"] = {
        "run_id": run_id, "seed": source["run_metadata"]["seed"],
        "created_at": utc_now_iso(), "origin_git_sha": git_sha, "last_git_sha": git_sha,
        "origin_code_fingerprint": fingerprint, "code_fingerprint": fingerprint,
        "source_type": "reward_fork", "source_checkpoint": str(source_path),
        "source_sha256": digest, "reward_fork": fork, "resume_count": 0,
        "experiment_contract_id": "", "experiment_contract_sha256": "",
        "config_path": "", "config_sha256": "",
    }
    result["args"] = {**result["args"], "run_id": run_id, "pace_profile": profile,
                      "experiment_contract": "", "pretrained_checkpoint": ""}
    validate_checkpoint_payload(result,
        expected_protocol_versions=result["protocol_versions"],
        expected_action_spec=source["action_spec"], expected_model_spec=source["model_spec"],
        expected_training_config=cfg, expected_critic_spec=source["critic_spec"])
    return result


def prepare(destination, profile="route_pace_v1", source_path=SOURCE, expected_sha=SOURCE_SHA):
    from training.train_no_rllib_mappo_robust_v2 import training_source_paths
    source_path, destination = Path(source_path).resolve(), Path(destination).resolve()
    source, digest = load_checkpoint_file(source_path, map_location="cpu")
    if digest != expected_sha or source["update"] != 356:
        raise CheckpointProtocolError("expected the registered u356 source; refuse a moving source")
    if source["curriculum_state"]["mastery_state"]["stage_index"] != 3:
        raise CheckpointProtocolError("u356 source must already be at S2")
    if destination.exists():
        raise FileExistsError(f"fork destination already exists: {destination}")
    git = get_git_metadata(ROOT.parents[2])
    fingerprint = files_fingerprint(training_source_paths(ROOT))
    checkpoint = fork_payload(source, source_path=source_path, digest=digest,
        run_id=destination.name, profile=profile, fingerprint=fingerprint, git_sha=git["commit"])
    checkpoint["args"]["save_dir"] = str(destination)
    destination.mkdir(parents=True, exist_ok=False)
    saved_sha = save_checkpoint_file(checkpoint, destination / "latest.pt", immutable=True)
    record = {"path": "latest.pt", "update": 356, "sha256": saved_sha, "recorded_at": utc_now_iso()}
    atomic_write_json({
        "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
        "training_protocol_version": checkpoint["training_protocol_version"],
        "run_id": destination.name, "created_at": utc_now_iso(),
        "latest": record, "interrupted": None, "periodic": [],
    }, destination / "checkpoint_manifest.json")
    atomic_write_json(checkpoint["run_metadata"]["reward_fork"], destination / "reward_fork.json")
    print(json.dumps({"status": "PREPARED_NOT_STARTED", "directory": str(destination),
                      "source_update": 356, "next_update": 357, "stop_update": 406,
                      "profile": profile, "sha256": saved_sha}, indent=2))


def resume_args(directory, stop_update=406):
    from training.train_no_rllib_mappo_robust_v2 import (
        build_parser, apply_resume_configuration, checkpoint_training_config,
        training_source_paths,
    )
    directory = Path(directory).resolve()
    checkpoint, digest = load_checkpoint_file(directory / "latest.pt", map_location="cpu")
    validate_checkpoint_header(checkpoint)
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if manifest["latest"]["sha256"] != digest or manifest["latest"]["update"] != checkpoint["update"]:
        raise CheckpointProtocolError("fork manifest/checkpoint mismatch")
    fork = checkpoint["run_metadata"].get("reward_fork", {})
    if fork.get("source_sha256") != SOURCE_SHA or fork.get("source_update") != 356:
        raise CheckpointProtocolError("unregistered reward fork source")
    if not checkpoint["update"] < stop_update <= 600:
        raise CheckpointProtocolError("stop update must exceed safe update and stay within LR horizon")
    if files_fingerprint(training_source_paths(ROOT)) != checkpoint["run_metadata"]["code_fingerprint"]:
        raise CheckpointProtocolError("fork source code changed; do not silently resume")
    args = build_parser().parse_args([])
    # Restore configuration before train() performs its startup validation.
    args.reward_version = "v6"
    args.scenario_dir = str(ROOT / "data/training_scenarios_town05_r8a_v1")
    args.save_dir, args.resume, args.updates = str(directory), "latest", int(stop_update)
    args.follow_spectator = True
    args._explicit_cli_options = set()
    apply_resume_configuration(args, checkpoint)
    from training.train_no_rllib_mappo_robust_v2 import scenario_pool_fingerprint
    if scenario_pool_fingerprint(args.scenario_dir) != checkpoint["curriculum_state"]["scenario_pool_fingerprint"]:
        raise CheckpointProtocolError("fork scenario pool differs from source")
    validate_checkpoint_payload(checkpoint,
        expected_protocol_versions=checkpoint["protocol_versions"],
        expected_action_spec=checkpoint["action_spec"], expected_model_spec=checkpoint["model_spec"],
        expected_training_config=checkpoint_training_config(args), expected_critic_spec=checkpoint["critic_spec"])
    return args, checkpoint


def run_prepared(directory, stop_update=406, validate_only=False):
    args, checkpoint = resume_args(directory, stop_update)
    print(json.dumps({"next_update": checkpoint["update"] + 1, "stop_update": stop_update,
        "stage": checkpoint["curriculum_state"]["mastery_state"]["stage_index"],
        "pace_profile": args.pace_profile, "lr_horizon": args.lr_decay_horizon,
        "source_update": 356, "follow_spectator": args.follow_spectator,
        "validate_only": validate_only}, indent=2), flush=True)
    if validate_only:
        return
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    from training.train_no_rllib_mappo_robust_v2 import train
    _preflight_idle_world(host=args.host, port=args.port, expected_version="0.9.16")
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--run-dir")
    parser.add_argument("--profile", choices=["route_pace_v1", "frozen_v6"], default="route_pace_v1")
    parser.add_argument("--stop-update", type=int, default=406)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output:
        if args.validate_only:
            parser.error("--validate-only requires --run-dir")
        prepare(args.output, args.profile)
    else:
        run_prepared(args.run_dir, args.stop_update, args.validate_only)
