"""Frozen R8-C1 diagnostic identities and source evidence, without CARLA.

This module supplements the existing C0 runner/validator.  It never migrates a
checkpoint, changes a training contract, or grants model-selection eligibility.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from training.checkpoint_protocol import files_fingerprint, get_git_metadata, sha256_file
from training.critic_diagnostics import validate_critic_probe_batch


R8C1_PROTOCOL_VERSION = "p0_base_c0_diag_town05_r8c1_v1"
R8C1_EPISODE_RECORD_VERSION = "p0_base_c0_town05_r8c1_diag_episode_v1"
R8C1_SUMMARY_VERSION = "p0_base_c0_town05_r8c1_diag_summary_v1"
R8C1_MANIFEST_VERSION = "p0_base_c0_town05_r8c1_diag_manifest_v1"
R8C1_RESOLVED_CONFIG_VERSION = "p0_base_c0_town05_r8c1_diag_resolved_v1"
R8C1_SOURCE_PROVENANCE_VERSION = "r8c1_diagnostic_source_provenance_v1"
R8C1_REPEATABILITY_MODE = "r8c1_road_outcome_strict_tick_tolerant_v1"
R8C1_PROFILES = ("smoke", "short_probe", "long_probe")
TOWN05_R8C1_C0_PROTOCOL_VERSION = R8C1_PROTOCOL_VERSION
TOWN05_R8C1_C0_EPISODE_RECORD_VERSION = R8C1_EPISODE_RECORD_VERSION
R8C1_C0_MANIFEST_VERSION = R8C1_MANIFEST_VERSION
R8C1_C0_RESOLVED_CONFIG_VERSION = R8C1_RESOLVED_CONFIG_VERSION
R8C1_CONTRACT_ID = "P0_BASE_C0_TOWN05_R8C1_PROBE_V1"
R8C1_TRAINING_CONTRACT_ID = "P0_BASE_MAPPO_TOWN05_R8C1_PACE_V1"
R8C1_TRAINING_PROTOCOL_VERSION = "mappo_robust_v2_r8c1_pace_p0_t05_v8"
R8C1_PROFILE_CHECKPOINT_UPDATES = {
    "smoke": [100], "short_probe": [20, 60, 100], "long_probe": [100],
}
R8C1_SCENARIO_DIR = "data/training_scenarios_town05_r8a_v1"
R8C1_SCENARIO_NAMES = (
    "scenario_01_north_short_rolling_a0_left.json",
    "scenario_02_south_short_rolling_a1_left.json",
    "scenario_03_north_short_stationary_a0_left.json",
    "scenario_04_south_short_stationary_a1_left.json",
    "scenario_05_northbound_free_flow.json",
    "scenario_06_southbound_free_flow.json",
    "scenario_07_eastbound_free_flow.json",
    "scenario_08_westbound_free_flow.json",
    "scenario_09_north_parallel_stationary_a0_left.json",
    "scenario_10_south_parallel_stationary_a1_left.json",
    "scenario_11_east_parallel_stationary_a1_left.json",
    "scenario_12_west_parallel_stationary_a0_left.json",
)
R8C1_EXPECTED_PROTOCOL_VERSIONS = {
    "observation": "carla_obs_20_lidar20m_time_v3",
    "action": "tanh_gaussian_signed_longitudinal_2d_pre_tanh_z_v2",
    "lifecycle": "agent_centric_offroad_terminated_truncated_v2",
    "offroad": "driving_footprint_offroad_v2_r6",
    "reward": "v6",
    "metrics": "mappo_metrics_r8c1_pace_p0_t05_v9",
    "value_normalization": "mappo_valuenorm_v1",
    "value_loss": "normalized_huber_delta_10_v1",
    "policy_distribution": "tanh_squashed_gaussian_entropy_v3_r7",
    "curriculum": "town05_r8a_stratified_mastery_v1",
    "stall": "stall_diagnostic_v1_record_only",
    "road_legality": "town05_road_legality_r7_v1",
    "critic_diagnostics": "agent_outcome_scenario_critic_r7_v1",
    "learning_rate": "separate_actor_critic_lr_r7_v1",
    "scenario_sampling": "town05_r8a_exact_balanced_v1",
}
R8C1_MODEL_SPEC = {
    "obs_dim": 20, "action_dim": 2, "state_dim": 40, "num_agents": 2,
    "agent_ids": [0, 1], "hidden_dim": 128,
    "critic_type": "AgentConditionedCritic", "critic_output_semantics": "normalized_value",
}
DIAGNOSTIC_POLICY_FIELDS = frozenset({
    "evaluation_purpose", "selection_allowed", "performance_decision",
    "registration_timing", "primary_checkpoint_update", "profile_checkpoint_updates",
    "total_execution_episodes", "training_run_id", "training_run_path", "training_git_sha",
    "training_contract_path", "training_contract_sha256", "training_code_fingerprint",
    "training_fingerprint_paths", "training_source_sha256", "checkpoint_manifest_path",
    "checkpoint_manifest_sha256", "checkpoints", "scenario_sha256", "evaluator_source_paths",
})


class R8C1DiagnosticError(ValueError):
    """A diagnostic input or source identity differs from its frozen contract."""


def is_r8c1_protocol(value: Any) -> bool:
    if isinstance(value, Mapping):
        value = value.get("protocol_version")
    return value == R8C1_PROTOCOL_VERSION


def diagnostic_usage() -> dict[str, Any]:
    return {
        "evaluation_purpose": "diagnostic_only",
        "selection_allowed": False,
        "performance_decision": "NOT_EVALUATED",
    }


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise R8C1DiagnosticError(f"{label} mismatch: {actual!r} != {expected!r}")


def _inside(project_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise R8C1DiagnosticError(f"source path must be a nonempty relative POSIX path: {relative!r}")
    path = (project_dir / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(project_dir.resolve()):
        raise R8C1DiagnosticError(f"source path escapes project: {relative!r}")
    return path


def _sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise R8C1DiagnosticError(f"{label} must be a SHA256")


def _hash_map(paths: list[str], project_dir: Path) -> dict[str, str]:
    return {relative: sha256_file(_inside(project_dir, relative)) for relative in sorted(paths)}


def _map_fingerprint(values: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_r8c1_contract_identity(contract: Mapping[str, Any], *, project_dir: Path) -> None:
    """Validate added diagnostic fields; C0 owns common environment/record validation."""
    _equal(contract.get("protocol_version"), R8C1_PROTOCOL_VERSION, "R8C1 protocol")
    _equal(contract.get("contract_id"), R8C1_CONTRACT_ID, "R8C1 contract id")
    policy = contract.get("diagnostic_policy")
    if not isinstance(policy, Mapping) or set(policy) != DIAGNOSTIC_POLICY_FIELDS:
        raise R8C1DiagnosticError("diagnostic_policy schema mismatch")
    for name, value in diagnostic_usage().items():
        _equal(policy[name], value, f"diagnostic_policy.{name}")
    if policy["selection_allowed"] is not False:
        raise R8C1DiagnosticError("diagnostic policy forbids selection")
    _equal(policy["registration_timing"], "post_training_diagnostic", "registration timing")
    _equal(policy["primary_checkpoint_update"], 100, "primary checkpoint")
    _equal(policy["profile_checkpoint_updates"], R8C1_PROFILE_CHECKPOINT_UPDATES, "profile checkpoints")
    _equal(policy["total_execution_episodes"], 42, "execution count")
    _equal(contract["repeatability"]["required_repeats"], 2, "required repeats")
    expected_profiles = {
        "smoke": (R8C1_SCENARIO_NAMES[:1], [61002]),
        "short_probe": (R8C1_SCENARIO_NAMES[:4], [61001]),
        "long_probe": (R8C1_SCENARIO_NAMES[4:], [61001]),
    }
    _equal(set(contract["profiles"]), set(expected_profiles), "diagnostic profiles")
    execution_count = 0
    for name, (scenario_names, seeds) in expected_profiles.items():
        profile = contract["profiles"][name]
        _equal(profile["scenario_dir"], R8C1_SCENARIO_DIR, f"{name}.scenario_dir")
        _equal(profile["scenario_names"], list(scenario_names), f"{name}.scenario_names")
        _equal(profile["episode_seeds"], seeds, f"{name}.episode_seeds")
        _equal(profile["formal_use"], "diagnostic_only", f"{name}.formal_use")
        execution_count += len(scenario_names) * len(seeds) * len(R8C1_PROFILE_CHECKPOINT_UPDATES[name]) * 2
    _equal(execution_count, policy["total_execution_episodes"], "expanded execution count")
    _equal(set(policy["checkpoints"]), {"20", "60", "100"}, "checkpoint updates")
    _equal(contract["checkpoint"]["required_protocol_versions"], R8C1_EXPECTED_PROTOCOL_VERSIONS, "checkpoint protocols")
    _equal(contract["checkpoint"]["required_training_contract_id"], R8C1_TRAINING_CONTRACT_ID, "training contract id")
    _equal(contract["checkpoint"]["required_training_contract_sha256"], policy["training_contract_sha256"], "training contract hash")
    if not isinstance(policy["training_git_sha"], str) or re.fullmatch(r"[0-9a-f]{40}", policy["training_git_sha"]) is None:
        raise R8C1DiagnosticError("training_git_sha must be a Git SHA")
    for name in ("training_contract_sha256", "training_code_fingerprint", "checkpoint_manifest_sha256"):
        _sha(policy[name], name)
    for name in ("training_run_path", "training_contract_path", "checkpoint_manifest_path"):
        _inside(project_dir, policy[name])
    run_dir = _inside(project_dir, policy["training_run_path"])
    _equal(run_dir.name, policy["training_run_id"], "training run directory")
    for update, item in policy["checkpoints"].items():
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise R8C1DiagnosticError("registered checkpoint schema mismatch")
        _sha(item["sha256"], f"checkpoint {update}")
        _equal(_inside(project_dir, item["path"]), run_dir / "periodic" / f"ckpt_update_{int(update):06d}.pt", "periodic path")
    _equal(_inside(project_dir, policy["checkpoint_manifest_path"]), run_dir / "checkpoint_manifest.json", "manifest path")
    sources = policy["training_source_sha256"]
    if not isinstance(sources, Mapping) or not sources:
        raise R8C1DiagnosticError("training source hash map is empty")
    for name in ("training_fingerprint_paths", "evaluator_source_paths"):
        paths = policy[name]
        if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
            raise R8C1DiagnosticError(f"{name} must contain unique source paths")
        for relative in paths:
            _inside(project_dir, relative)
    if not set(policy["training_fingerprint_paths"]).issubset(sources):
        raise R8C1DiagnosticError("training fingerprint paths missing from source map")
    if set(policy["evaluator_source_paths"]) & set(sources):
        raise R8C1DiagnosticError("evaluation sources overlap frozen training sources")
    for relative, digest in sources.items():
        _inside(project_dir, relative)
        _sha(digest, f"source {relative}")
    scenario_hashes = policy["scenario_sha256"]
    if not isinstance(scenario_hashes, Mapping) or len(scenario_hashes) != 12:
        raise R8C1DiagnosticError("exactly twelve scenario hashes are required")
    _equal(set(scenario_hashes), set(R8C1_SCENARIO_NAMES), "registered scenario names")
    for relative, digest in scenario_hashes.items():
        if Path(relative).name != relative:
            raise R8C1DiagnosticError("scenario hash keys must be filenames")
        _sha(digest, f"scenario {relative}")
        source_path = f"{R8C1_SCENARIO_DIR}/{relative}"
        _equal(sha256_file(_inside(project_dir, source_path)), digest, f"scenario SHA256 {relative}")
        _equal(sources.get(source_path), digest, f"training scenario source SHA256 {relative}")


def validate_r8c1_checkpoint_identity(
    checkpoint: Mapping[str, Any], *, path: str | Path, sha256: str,
    contract: Mapping[str, Any], project_dir: Path,
) -> None:
    """Bind an already parsed checkpoint to its exact file, sidecar, and manifest."""
    policy = contract["diagnostic_policy"]
    # A contract can be inspected without private run artifacts (for example in
    # CI). File identity is mandatory at checkpoint preflight, before rollout.
    _equal(sha256_file(_inside(project_dir, policy["training_contract_path"])), policy["training_contract_sha256"], "training contract file")
    update = checkpoint.get("update")
    if isinstance(update, bool) or not isinstance(update, int) or str(update) not in policy["checkpoints"]:
        raise R8C1DiagnosticError("unregistered R8C1 checkpoint update")
    registered = policy["checkpoints"][str(update)]
    resolved = Path(path).resolve()
    _equal(resolved, _inside(project_dir, registered["path"]), "checkpoint path")
    _equal(sha256, registered["sha256"], "checkpoint registered SHA256")
    _equal(sha256_file(resolved), sha256, "checkpoint current SHA256")
    sidecar = resolved.with_name(resolved.name + ".sha256").read_text(encoding="utf-8").split()
    if len(sidecar) != 2:
        raise R8C1DiagnosticError("checkpoint sidecar must contain SHA256 and filename")
    _equal(sidecar[0], sha256, "checkpoint sidecar SHA256")
    _equal(sidecar[1], resolved.name, "checkpoint sidecar filename")
    _equal(checkpoint.get("checkpoint_schema_version"), 8, "checkpoint schema")
    _equal(checkpoint.get("checkpoint_kind"), "periodic", "checkpoint kind")
    _equal(checkpoint.get("training_protocol_version"), R8C1_TRAINING_PROTOCOL_VERSION, "checkpoint training protocol")
    _equal(checkpoint.get("protocol_versions"), R8C1_EXPECTED_PROTOCOL_VERSIONS, "checkpoint protocols")
    _equal(checkpoint.get("model_spec"), R8C1_MODEL_SPEC, "checkpoint model spec")
    metadata = checkpoint["run_metadata"]
    for name, expected in {
        "run_id": policy["training_run_id"], "seed": 42,
        "origin_git_sha": policy["training_git_sha"], "last_git_sha": policy["training_git_sha"],
        "code_fingerprint": policy["training_code_fingerprint"],
        "origin_code_fingerprint": policy["training_code_fingerprint"],
        "experiment_contract_id": R8C1_TRAINING_CONTRACT_ID,
        "experiment_contract_sha256": policy["training_contract_sha256"],
        "config_sha256": policy["training_contract_sha256"],
        "source_type": "fresh", "code_drift_override": False,
    }.items():
        _equal(metadata.get(name), expected, f"checkpoint metadata.{name}")
    env = contract["environment"]
    for name, expected in {
        "reward_version": "v6", "gamma": env["reward_gamma"], "town": env["town"],
        "timestep": env["timestep"], "max_steps": env["max_steps"],
        "lidar_range_m": env["lidar_range_m"], "ttc_max_range_m": env["ttc_max_range_m"],
        "curriculum_protocol": "r8a_mastery_v1", "scenario_sampling_mode": "r8a_balanced",
    }.items():
        _equal(checkpoint["training_config"].get(name), expected, f"checkpoint training_config.{name}")
    runtime = metadata["runtime"]
    _equal(runtime.get("environment_reward_gamma"), env["reward_gamma"], "runtime reward gamma")
    for name in ("offroad_protocol", "road_legality_protocol"):
        _equal(runtime.get(name), env[name], f"runtime {name}")
    validate_critic_probe_batch(checkpoint.get("critic_probe_batch"), expected_state_dim=40)
    manifest_path = _inside(project_dir, policy["checkpoint_manifest_path"])
    _equal(sha256_file(manifest_path), policy["checkpoint_manifest_sha256"], "checkpoint manifest SHA256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _equal(manifest.get("run_id"), policy["training_run_id"], "manifest run id")
    _equal(manifest.get("checkpoint_schema_version"), 8, "manifest schema")
    _equal(manifest.get("training_protocol_version"), R8C1_TRAINING_PROTOCOL_VERSION, "manifest training protocol")
    entries = [entry for entry in manifest.get("periodic", []) if entry.get("update") == update]
    if len(entries) != 1:
        raise R8C1DiagnosticError("checkpoint manifest must register exactly one matching update")
    _equal(entries[0].get("sha256"), sha256, "manifest checkpoint SHA256")
    _equal(str(entries[0].get("path", "")).replace("\\", "/"), f"periodic/ckpt_update_{update:06d}.pt", "manifest checkpoint path")


def build_source_provenance(
    *, contract: Mapping[str, Any], project_dir: Path, repository_root: Path,
    checkpoint_record: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject shared-source drift while independently identifying the new evaluator."""
    if checkpoint is None:
        if checkpoint_record is None:
            raise R8C1DiagnosticError("checkpoint or checkpoint_record is required")
        checkpoint = checkpoint_record["checkpoint"]
    policy = contract["diagnostic_policy"]
    metadata = checkpoint["run_metadata"]
    _equal(metadata["code_fingerprint"], policy["training_code_fingerprint"], "training source identity")
    actual_sources = _hash_map(list(policy["training_source_sha256"]), project_dir)
    differences = [name for name, digest in actual_sources.items() if digest != policy["training_source_sha256"][name]]
    if differences:
        raise R8C1DiagnosticError("shared/training source drift: " + ", ".join(differences))
    current_fingerprint = files_fingerprint(_inside(project_dir, name) for name in policy["training_fingerprint_paths"])
    _equal(current_fingerprint, policy["training_code_fingerprint"], "training source fingerprint")
    evaluator_sources = _hash_map(policy["evaluator_source_paths"], project_dir)
    return {
        "source_provenance_version": R8C1_SOURCE_PROVENANCE_VERSION,
        "training": {
            "run_id": metadata["run_id"], "git_sha": metadata["last_git_sha"],
            "code_fingerprint": metadata["code_fingerprint"],
            "contract_sha256": metadata["experiment_contract_sha256"],
            "source_sha256": actual_sources,
        },
        "evaluation": {
            "git_sha": get_git_metadata(repository_root).get("commit", "unknown"),
            "source_sha256": evaluator_sources,
            "source_fingerprint": _map_fingerprint(evaluator_sources),
        },
        "shared_source_identity": "MATCH",
    }


def validate_source_provenance(
    saved: Mapping[str, Any], *, contract: Mapping[str, Any], project_dir: Path,
    repository_root: Path, checkpoint_record: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None, check_current_evaluator: bool = True,
) -> dict[str, Any]:
    """Recompute hashes; Git-only changes never mask or manufacture source drift."""
    current = build_source_provenance(
        contract=contract, checkpoint_record=checkpoint_record, checkpoint=checkpoint,
        project_dir=project_dir, repository_root=repository_root,
    )
    if not isinstance(saved, Mapping) or set(saved) != set(current):
        raise R8C1DiagnosticError("source provenance schema mismatch")
    for name in ("source_provenance_version", "training", "shared_source_identity"):
        _equal(saved[name], current[name], f"source provenance {name}")
    evaluation = saved["evaluation"]
    if not isinstance(evaluation, Mapping) or set(evaluation) != set(current["evaluation"]):
        raise R8C1DiagnosticError("evaluation source provenance schema mismatch")
    _equal(_map_fingerprint(evaluation["source_sha256"]), evaluation["source_fingerprint"], "saved evaluator source fingerprint")
    if check_current_evaluator:
        for name in ("source_sha256", "source_fingerprint"):
            _equal(evaluation[name], current["evaluation"][name], f"evaluator {name}")
    return current


def actor_state_hashes(actors_or_states: Mapping[Any, Any]) -> dict[str, str]:
    """Hash tensor names, dtypes, shapes and bytes, independent of torch.save format."""
    result = {}
    for agent_id, actor_or_state in sorted(actors_or_states.items(), key=lambda pair: int(pair[0])):
        key = str(int(agent_id))
        if key in result:
            raise R8C1DiagnosticError("duplicate actor id")
        state = actor_or_state.state_dict() if hasattr(actor_or_state, "state_dict") else actor_or_state
        if not isinstance(state, Mapping) or not state:
            raise R8C1DiagnosticError(f"actor {key} has no state")
        digest = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(json.dumps([name, str(array.dtype), list(array.shape)], separators=(",", ":")).encode())
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
            digest.update(b"\n")
        result[key] = digest.hexdigest()
    if set(result) != {"0", "1"}:
        raise R8C1DiagnosticError("R8C1 actor hashes require exactly agents 0 and 1")
    return result


def actor_state_fingerprint(actors_or_states: Mapping[Any, Any]) -> str:
    return _map_fingerprint(actor_state_hashes(actors_or_states))
