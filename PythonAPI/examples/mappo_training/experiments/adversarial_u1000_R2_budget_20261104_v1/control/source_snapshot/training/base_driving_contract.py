"""Pure-file configuration for the explicitly versioned base-driving experiment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

PROTOCOL = "base_driving_vnext_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(path, project_dir=None):
    root = Path(project_dir or Path(__file__).resolve().parents[1])
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("unregistered base-driving contract")
    if payload.get("reward_profile") not in {"v6", "startup_v1"}:
        raise ValueError("unknown base-driving reward profile")
    scene_dir = (root / payload["scenario_dir"]).resolve()
    from training.base_driving_scenarios_v1 import load_base_driving_manifest
    load_base_driving_manifest(scene_dir)
    qualification = (root / payload["qualification_contract"]).resolve()
    from evaluation.base_driving_qualification import load_contract as load_qualification
    load_qualification(qualification)
    overrides = payload.get("training_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("training_overrides must be explicit")
    required = {"reward_version": "v6", "pace_profile": "instant_pace_v6",
                "curriculum_protocol": PROTOCOL, "scenario_sampling_mode": "base_driving_paired_v1",
                "curriculum_mode": "metadata", "rollout_episodes": 16}
    if any(overrides.get(key) != value for key, value in required.items()):
        raise ValueError("base driving requires V6 parent, 20/2 and explicit paired curriculum")
    return {"path": str(path), "payload": payload, "scenario_dir": str(scene_dir),
            "record": {"protocol": PROTOCOL, "contract_sha256": sha(path),
                       "qualification_contract_sha256": sha(qualification),
                       "scenario_pool_sha256": sha(scene_dir / "protocol/manifest.json"),
                       "reward_profile": payload["reward_profile"]}}


def configure(args, project_dir):
    path = getattr(args, "base_driving_contract", "")
    if not path:
        if getattr(args, "curriculum_protocol", "") == PROTOCOL or getattr(args, "scenario_sampling_mode", "") == "base_driving_paired_v1":
            raise ValueError("base-driving protocol requires --base-driving-contract")
        return None
    if args.experiment_contract or args.continuation_contract or args.allow_resume_code_drift:
        raise ValueError("base driving cannot bypass frozen contracts or code hashes")
    metadata = load_contract(path, project_dir)
    explicit = set(getattr(args, "_explicit_cli_options", ()))
    for key, value in metadata["payload"]["training_overrides"].items():
        if not hasattr(args, key) or key.startswith("_"):
            raise ValueError(f"unknown base-driving training option: {key}")
        if f"--{key.replace('_', '-')}" in explicit and getattr(args, key) != value:
            raise ValueError(f"explicit option conflicts with base-driving contract: {key}")
        if key == "lr_decay" and "--no-lr-decay" in explicit and value:
            raise ValueError("explicit no-lr-decay conflicts with the base-driving schedule")
        setattr(args, key, value)
    if "--scenario-dir" in explicit and Path(args.scenario_dir).resolve() != Path(metadata["scenario_dir"]):
        raise ValueError("scenario directory conflicts with base-driving contract")
    args.scenario_dir = metadata["scenario_dir"]
    args.base_driving_contract = metadata["path"]
    args._base_driving_metadata = metadata
    return metadata


def validate_resume_contract(args, checkpoint):
    current = getattr(args, "_base_driving_metadata", None)
    saved = checkpoint.get("base_driving_contract")
    declared = (checkpoint.get("training_config", {}).get("base_driving_protocol")
                or checkpoint.get("protocol_versions", {}).get("base_driving"))
    if declared and not saved:
        raise ValueError("base-driving checkpoint is missing its explicit contract binding")
    if bool(current) != bool(saved):
        raise ValueError("legacy and base-driving runs cannot be silently resumed into each other")
    if current and current["record"] != saved:
        raise ValueError("base-driving contract, reward, pool or qualification changed on resume")


def source_paths(root, metadata):
    return [root / name for name in (
        "training/base_driving_contract.py", "training/base_driving_curriculum.py",
        "training/base_driving_runtime.py", "training/base_driving_scenarios_v1.py",
        "environment/base_driving_env.py", "environment/base_driving_reward.py",
        "evaluation/base_driving_qualification.py",
    )] + [Path(metadata["path"]),
          root / metadata["payload"]["qualification_contract"],
          Path(metadata["scenario_dir"]) / "protocol/manifest.json"]
