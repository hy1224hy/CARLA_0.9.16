"""Explicit complete-state u456 reward screening forks, never historical resume.

The source critic, ValueNorm and critic probe remain tied to the source reward
at the seed boundary.  New rollouts adapt the value model and replace the probe;
this tool does not claim that old value estimates already fit the new reward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.checkpoint_protocol import (
    CheckpointProtocolError, atomic_copy_checkpoint, atomic_write_json,
    files_fingerprint, get_git_metadata, load_checkpoint_file,
    save_checkpoint_file, utc_now_iso, validate_checkpoint_header,
    validate_checkpoint_payload, verify_checkpoint_file,
)
from training.reward_fork import fork_payload as _legacy_fork_payload


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/R8C1_RoutePaceV1_u356_to406_s42_20260927/latest.pt"
SOURCE_SHA = "8b9cb173cd70aacc201f0cb68183fdfeeb593cc8f29863c096a5f59076954545"
SOURCE_UPDATE = 456
SOURCE_PROFILE = "route_pace_v1"
FORK_VERSION = "reward_fork_v2_u456"
PROFILES = ("instant_pace_v2", "route_pace_v1", "instant_pace_v3", "instant_pace_v4", "instant_pace_v5", "instant_pace_v6")
DEFAULT_STOP_UPDATE = 486
SCENARIO_DIR = ROOT / "data/training_scenarios_town05_r8a_v1"


def _validate_source_shape(checkpoint):
    validate_checkpoint_header(checkpoint)
    cfg, model = checkpoint["training_config"], checkpoint["model_spec"]
    if cfg.get("reward_version") != "v6":
        raise CheckpointProtocolError("u456 reward fork requires reward v6")
    expected = {"obs_dim": 20, "action_dim": 2, "state_dim": 40,
                "num_agents": 2, "agent_ids": [0, 1], "hidden_dim": 128,
                "critic_type": "AgentConditionedCritic"}
    if any(model.get(key) != value for key, value in expected.items()):
        raise CheckpointProtocolError("u456 fork requires the registered two-agent 20D/2D model")
    if checkpoint["curriculum_state"]["mastery_state"].get("stage_index") != 3:
        raise CheckpointProtocolError("u456 fork must retain source S2 curriculum")
    if checkpoint["lr_schedule"].get("decay_horizon") != 600:
        raise CheckpointProtocolError("u456 fork must retain the registered LR horizon 600")
    validate_checkpoint_payload(
        checkpoint, expected_protocol_versions=checkpoint["protocol_versions"],
        expected_action_spec=checkpoint["action_spec"], expected_model_spec=model,
        expected_training_config=cfg, expected_critic_spec=checkpoint["critic_spec"],
    )


def validate_registered_source(source, digest):
    """Reject moving or merely similar source policies before deriving a run."""
    if digest != SOURCE_SHA or source.get("update") != SOURCE_UPDATE:
        raise CheckpointProtocolError("expected the registered u456 SHA256/update source")
    _validate_source_shape(source)
    if source["training_config"].get("pace_profile") != SOURCE_PROFILE:
        raise CheckpointProtocolError("registered u456 source requires route_pace_v1")


def validate_inherited_contract(checkpoint, source):
    """Only the registered reward profile can change in this fork family."""
    expected_config = {**source["training_config"],
                       "pace_profile": checkpoint["training_config"]["pace_profile"]}
    expected_protocols = {**source["protocol_versions"],
                          "pace_reward": checkpoint["training_config"]["pace_profile"]}
    if checkpoint["training_config"] != expected_config:
        raise CheckpointProtocolError("u456 reward-only fork changed inherited training configuration")
    if checkpoint["protocol_versions"] != expected_protocols:
        raise CheckpointProtocolError("u456 reward-only fork changed inherited non-pace protocols")
    for key in ("action_spec", "model_spec", "critic_spec"):
        if checkpoint[key] != source[key]:
            raise CheckpointProtocolError(f"u456 reward-only fork changed inherited {key}")


def fork_payload(source, *, source_path, digest, run_id, profile,
                 fingerprint, git_sha):
    """Pure complete-state derivation; runtime profile registration is separate."""
    validate_registered_source(source, digest)
    if profile not in PROFILES:
        raise CheckpointProtocolError("unregistered u456 reward fork profile")
    if not run_id or run_id == source["run_metadata"]["run_id"]:
        raise CheckpointProtocolError("u456 fork requires a distinct nonempty RunID")
    result = _legacy_fork_payload(
        source, source_path=source_path, digest=digest, run_id=run_id,
        profile=SOURCE_PROFILE, fingerprint=fingerprint, git_sha=git_sha,
    )
    result["training_config"]["pace_profile"] = profile
    result["protocol_versions"]["pace_reward"] = profile
    result["args"]["pace_profile"] = profile
    fork = result["run_metadata"]["reward_fork"]
    fork.update({
        "version": FORK_VERSION, "profile": profile,
        "source_reward_version": "v6", "source_pace_profile": SOURCE_PROFILE,
        "source_critic_and_probe_sha256": digest,
        "source_critic_and_probe_update": SOURCE_UPDATE,
        "critic_transfer": "source critic/optimizer/ValueNorm retained; no claim of adaptation to a changed reward",
        "critic_probe_scope": "source-reward evidence until first valid fork rollout replaces probe",
        "counter_semantics": "inherited lifetime counters; fork-only metrics start at update 457",
        "scope": "known-scene reward screening; not fresh training or independent generalization",
    })
    validate_checkpoint_payload(
        result, expected_protocol_versions=result["protocol_versions"],
        expected_action_spec=source["action_spec"], expected_model_spec=source["model_spec"],
        expected_training_config=result["training_config"],
        expected_critic_spec=source["critic_spec"],
    )
    validate_inherited_contract(result, source)
    return result


def _trainer_api():
    from training import train_no_rllib_mappo_robust_v2 as trainer
    return trainer


def _runtime_fingerprint(trainer, profile):
    parser = trainer.build_parser()
    action = next(action for action in parser._actions if action.dest == "pace_profile")
    if profile not in action.choices:
        raise CheckpointProtocolError("reward profile is not integrated in the trainer; prepare refused")
    paths = [Path(path).resolve() for path in trainer.training_source_paths(ROOT)]
    if Path(__file__).resolve() not in paths:
        raise CheckpointProtocolError("reward_fork_v2.py must be included in trainer source fingerprints")
    profile_file = {
        "instant_pace_v2": ROOT / "environment" / "instant_pace_reward.py",
        "instant_pace_v3": ROOT / "environment" / "instant_pace_reward_v3.py",
        "instant_pace_v4": ROOT / "environment" / "instant_pace_reward_v4.py",
        "instant_pace_v5": ROOT / "environment" / "instant_pace_reward_v5.py",
        "instant_pace_v6": ROOT / "environment" / "instant_pace_reward_v6.py",
    }.get(profile)
    if profile_file is not None and profile_file.resolve() not in paths:
        raise CheckpointProtocolError(f"{profile_file.name} must be included in trainer source fingerprints")
    return files_fingerprint(paths)


def prepare(destination, profile="instant_pace_v2", source_path=SOURCE):
    trainer = _trainer_api()
    fingerprint = _runtime_fingerprint(trainer, profile)
    source_path, destination = Path(source_path).resolve(), Path(destination).resolve()
    source, digest = load_checkpoint_file(source_path, map_location="cpu")
    validate_registered_source(source, digest)
    if destination.exists():
        raise FileExistsError(f"fork destination already exists: {destination}")
    if trainer.scenario_pool_fingerprint(SCENARIO_DIR) != source["curriculum_state"]["scenario_pool_fingerprint"]:
        raise CheckpointProtocolError("registered source scenario pool changed")
    git = get_git_metadata(ROOT.parents[2])
    checkpoint = fork_payload(
        source, source_path=source_path, digest=digest, run_id=destination.name,
        profile=profile, fingerprint=fingerprint, git_sha=git["commit"],
    )
    checkpoint["args"]["save_dir"] = str(destination)
    destination.mkdir(parents=True, exist_ok=False)
    saved_sha = save_checkpoint_file(checkpoint, destination / "seed.pt", immutable=True)
    latest_sha = atomic_copy_checkpoint(destination / "seed.pt", destination / "latest.pt")
    if latest_sha != saved_sha:
        raise CheckpointProtocolError("seed/latest identity mismatch")
    recorded_at = utc_now_iso()
    seed_record = {"path": "seed.pt", "update": SOURCE_UPDATE,
                   "sha256": saved_sha, "recorded_at": recorded_at}
    atomic_write_json(checkpoint["run_metadata"]["reward_fork"], destination / "reward_fork.json")
    # Publishing the manifest is the seed commit, after both verified files.
    atomic_write_json({
        "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
        "training_protocol_version": checkpoint["training_protocol_version"],
        "run_id": destination.name, "created_at": recorded_at, "seed": seed_record,
        "latest": {**seed_record, "path": "latest.pt"},
        "interrupted": None, "periodic": [],
    }, destination / "checkpoint_manifest.json")
    record = {"status": "PREPARED_NOT_STARTED", "directory": str(destination),
              "source_update": SOURCE_UPDATE, "next_update": SOURCE_UPDATE + 1,
              "profile": profile, "sha256": saved_sha,
              "suggested_stop_update": DEFAULT_STOP_UPDATE}
    print(json.dumps(record, indent=2), flush=True)
    return record


def validate_fork_identity(checkpoint, manifest, digest, fingerprint):
    """Pure checks for the identity of a seed or later complete fork boundary."""
    _validate_source_shape(checkpoint)
    fork = checkpoint["run_metadata"].get("reward_fork", {})
    if (fork.get("version") != FORK_VERSION or fork.get("source_sha256") != SOURCE_SHA
            or fork.get("source_update") != SOURCE_UPDATE
            or fork.get("source_pace_profile") != SOURCE_PROFILE):
        raise CheckpointProtocolError("unregistered u456 reward fork origin")
    profile = checkpoint["training_config"].get("pace_profile")
    if profile not in PROFILES or fork.get("profile") != profile:
        raise CheckpointProtocolError("fork profile identity mismatch")
    if checkpoint["protocol_versions"].get("pace_reward") != profile:
        raise CheckpointProtocolError("fork pace protocol mismatch")
    latest = manifest.get("latest") or {}
    if (latest.get("path") != "latest.pt" or latest.get("sha256") != digest
            or latest.get("update") != checkpoint["update"]):
        raise CheckpointProtocolError("fork manifest/checkpoint safe boundary mismatch")
    if manifest.get("run_id") != checkpoint["run_metadata"]["run_id"]:
        raise CheckpointProtocolError("fork manifest/checkpoint RunID mismatch")
    if any(manifest.get(key) != checkpoint[key] for key in (
            "checkpoint_schema_version", "training_protocol_version")):
        raise CheckpointProtocolError("fork manifest/checkpoint protocol mismatch")
    if not SOURCE_UPDATE <= checkpoint["update"] <= 600:
        raise CheckpointProtocolError("fork boundary falls outside retained update horizon")
    if any(checkpoint["run_metadata"].get(key) != fingerprint for key in (
            "code_fingerprint", "origin_code_fingerprint")):
        raise CheckpointProtocolError("fork source code changed; explicit new experiment required")


def resume_args(directory, stop_update=DEFAULT_STOP_UPDATE):
    trainer = _trainer_api()
    directory = Path(directory).resolve()
    checkpoint, digest = load_checkpoint_file(directory / "latest.pt", map_location="cpu")
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text(encoding="utf-8-sig"))
    profile = checkpoint["training_config"].get("pace_profile")
    fingerprint = _runtime_fingerprint(trainer, profile)
    validate_fork_identity(checkpoint, manifest, digest, fingerprint)
    seed = manifest.get("seed") or {}
    if (seed.get("path") != "seed.pt" or seed.get("update") != SOURCE_UPDATE
            or verify_checkpoint_file(directory / "seed.pt") != seed.get("sha256")):
        raise CheckpointProtocolError("immutable reward fork seed identity mismatch")
    source_path = checkpoint["run_metadata"]["reward_fork"]["source_checkpoint"]
    source, source_digest = load_checkpoint_file(source_path, map_location="cpu")
    validate_registered_source(source, source_digest)
    validate_inherited_contract(checkpoint, source)
    if (isinstance(stop_update, bool) or not isinstance(stop_update, int)
            or not checkpoint["update"] < stop_update <= 600):
        raise CheckpointProtocolError("absolute stop update must exceed safe boundary and stay at most 600")
    args = trainer.build_parser().parse_args([])
    args.reward_version, args.scenario_dir = "v6", str(SCENARIO_DIR)
    args.save_dir, args.resume, args.updates = str(directory), "latest", stop_update
    args.follow_spectator, args._explicit_cli_options = True, set()
    trainer.apply_resume_configuration(args, checkpoint)
    if trainer.scenario_pool_fingerprint(args.scenario_dir) != checkpoint["curriculum_state"]["scenario_pool_fingerprint"]:
        raise CheckpointProtocolError("fork scenario pool differs from checkpoint")
    expected_protocols = trainer.checkpoint_protocol_versions(
        "v6", args.lidar_range_m, args.curriculum_protocol)
    expected_protocols["pace_reward"] = profile
    validate_checkpoint_payload(
        checkpoint, expected_protocol_versions=expected_protocols,
        expected_action_spec=checkpoint["action_spec"], expected_model_spec=checkpoint["model_spec"],
        expected_training_config=trainer.checkpoint_training_config(args),
        expected_critic_spec=checkpoint["critic_spec"],
    )
    return args, checkpoint


def run_prepared(directory, stop_update=DEFAULT_STOP_UPDATE, validate_only=False):
    args, checkpoint = resume_args(directory, stop_update)
    args.stop_request_file = str(Path(directory).resolve() / "session" / "STOP_REQUESTED")
    print(json.dumps({"next_update": checkpoint["update"] + 1,
        "stop_update": stop_update, "fork_updates_planned": stop_update - SOURCE_UPDATE,
        "stage": checkpoint["curriculum_state"]["mastery_state"]["stage_index"],
        "pace_profile": args.pace_profile, "lr_horizon": args.lr_decay_horizon,
        "source_update": SOURCE_UPDATE, "follow_spectator": args.follow_spectator,
        "validate_only": validate_only}, indent=2), flush=True)
    if validate_only:
        return
    if Path(args.stop_request_file).exists():
        raise CheckpointProtocolError("STOP_REQUESTED exists; inspect and archive it before explicit resume")
    from utils.mut_mappo_session import CarlaOwner
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    with CarlaOwner(host=args.host, port=args.port, run_dir=directory,
                    purpose="u456 reward fork training"):
        _preflight_idle_world(host=args.host, port=args.port, expected_version="0.9.16")
        _trainer_api().train(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output")
    mode.add_argument("--run-dir")
    parser.add_argument("--source-checkpoint", default=str(SOURCE))
    parser.add_argument("--profile", choices=PROFILES, default="instant_pace_v2")
    parser.add_argument("--stop-update", type=int, default=DEFAULT_STOP_UPDATE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.output:
        if args.validate_only:
            parser.error("--validate-only requires --run-dir")
        prepare(args.output, args.profile, args.source_checkpoint)
    else:
        run_prepared(args.run_dir, args.stop_update, args.validate_only)


if __name__ == "__main__":
    main()
