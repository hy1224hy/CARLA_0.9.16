"""Versioned, atomic checkpoint protocol for the current robust_v2 trainer.

This module deliberately has no CARLA dependency so save/resume invariants can
be exercised in fast unit tests before an end-to-end simulator smoke run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch

from training.action_protocol import (
    ACTION_PROTOCOL_VERSION,
    LEGACY_ACTION_PROTOCOL_VERSION,
)

LEGACY_CHECKPOINT_SCHEMA_VERSION = 2
ACTION_CHECKPOINT_SCHEMA_VERSION = 3
VALUENORM_CHECKPOINT_SCHEMA_VERSION = 4
CHECKPOINT_SCHEMA_VERSION = 5
LEGACY_TRAINING_PROTOCOL_VERSION = "mappo_robust_v2_p0_00g_v2"
ACTION_TRAINING_PROTOCOL_VERSION = "mappo_robust_v2_action_v2_p0_00g_v3"
VALUENORM_TRAINING_PROTOCOL_VERSION = "mappo_robust_v2_valuenorm_p0_00g_v4"
TRAINING_PROTOCOL_VERSION = "mappo_robust_v2_offroad_p0_00g_v5"
OBSERVATION_PROTOCOL_VERSION = "carla_obs_19_v1"
TOWN05_BASE_OBSERVATION_PROTOCOL_VERSION = "carla_obs_19_lidar20m_v2"
PRE_OFFROAD_LIFECYCLE_PROTOCOL_VERSION = "agent_centric_terminated_truncated_v1"
LIFECYCLE_PROTOCOL_VERSION = "agent_centric_offroad_terminated_truncated_v2"

LATEST_CHECKPOINT_NAME = "latest.pt"
INTERRUPTED_CHECKPOINT_NAME = "interrupted.pt"
PERIODIC_DIR_NAME = "periodic"
MANIFEST_NAME = "checkpoint_manifest.json"


class CheckpointProtocolError(RuntimeError):
    """Raised when a checkpoint cannot safely resume the current run."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace_bytes(data: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as file_obj:
            file_obj.write(data)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(payload: Mapping[str, Any], path: os.PathLike[str] | str) -> None:
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    _atomic_replace_bytes(serialized, Path(path))


def checksum_path(checkpoint_path: os.PathLike[str] | str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(checkpoint_path.name + ".sha256")


def _write_checksum(checkpoint_path: Path, digest: str) -> None:
    line = f"{digest}  {checkpoint_path.name}\n".encode("ascii")
    _atomic_replace_bytes(line, checksum_path(checkpoint_path))


def save_checkpoint_file(
    payload: Mapping[str, Any],
    path: os.PathLike[str] | str,
    *,
    immutable: bool = False,
) -> str:
    """Atomically serialize a checkpoint and its SHA256 sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if immutable and (destination.exists() or checksum_path(destination).exists()):
        raise FileExistsError(f"immutable checkpoint already exists: {destination}")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        with open(temporary_path, "rb+") as file_obj:
            os.fsync(file_obj.fileno())
        digest = sha256_file(temporary_path)
        os.replace(temporary_path, destination)
        _write_checksum(destination, digest)
        return digest
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_checkpoint_file(path: os.PathLike[str] | str) -> str:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    sidecar = checksum_path(checkpoint_path)
    if not sidecar.is_file():
        raise CheckpointProtocolError(
            f"checkpoint checksum sidecar not found: {sidecar}"
        )
    tokens = sidecar.read_text(encoding="ascii").strip().split()
    if not tokens or len(tokens[0]) != 64:
        raise CheckpointProtocolError(f"invalid checksum sidecar: {sidecar}")
    expected = tokens[0].lower()
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise CheckpointProtocolError(
            f"checkpoint SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def load_checkpoint_file(
    path: os.PathLike[str] | str,
    *,
    map_location: Any = "cpu",
) -> Tuple[Dict[str, Any], str]:
    digest = verify_checkpoint_file(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise CheckpointProtocolError("checkpoint root must be a dictionary")
    return checkpoint, digest


def atomic_copy_checkpoint(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
) -> str:
    """Copy a verified safe checkpoint to another atomically replaced name."""
    source_path = Path(source)
    verify_checkpoint_file(source_path)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(source_path, temporary_path)
        with open(temporary_path, "rb+") as file_obj:
            os.fsync(file_obj.fileno())
        digest = sha256_file(temporary_path)
        os.replace(temporary_path, destination_path)
        _write_checksum(destination_path, digest)
        return digest
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def resolve_resume_path(resume: str, save_dir: os.PathLike[str] | str) -> Path:
    if not resume:
        raise ValueError("resume selector must not be empty")
    if resume == "latest":
        path = Path(save_dir).resolve() / LATEST_CHECKPOINT_NAME
    else:
        path = Path(resume).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"resume checkpoint not found: {path}; refusing to start from zero"
        )
    return path


def infer_run_root(checkpoint_path: os.PathLike[str] | str) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    if checkpoint_path.parent.name == PERIODIC_DIR_NAME:
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def ensure_fresh_run_directory(save_dir: os.PathLike[str] | str) -> Path:
    run_dir = Path(save_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    protected = [
        run_dir / LATEST_CHECKPOINT_NAME,
        run_dir / INTERRUPTED_CHECKPOINT_NAME,
        run_dir / PERIODIC_DIR_NAME,
        run_dir / MANIFEST_NAME,
        run_dir / "metrics_schema.json",
        run_dir / "metrics.jsonl",
        run_dir / "episodes.jsonl",
        run_dir / "tb",
    ]
    old_checkpoints = list(run_dir.glob("*.pt"))
    old_checkpoints.extend(run_dir.glob("*.pt.sha256"))
    conflicts = [str(path) for path in protected if path.exists()]
    conflicts.extend(
        str(path) for path in old_checkpoints if str(path) not in conflicts
    )
    if conflicts:
        raise FileExistsError(
            "fresh run directory already contains protected run artifacts: "
            + ", ".join(conflicts)
        )
    return run_dir


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> Tuple[str, ...]:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required - set(state)
    if missing:
        raise CheckpointProtocolError(
            f"RNG state missing fields: {sorted(missing)}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    warnings = []
    cuda_states = list(state["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available():
            warnings.append(
                "checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "continuation is not bitwise reproducible"
            )
        elif len(cuda_states) != torch.cuda.device_count():
            warnings.append(
                "CUDA device count differs from checkpoint; continuation is not "
                "bitwise reproducible"
            )
        else:
            torch.cuda.set_rng_state_all(cuda_states)
    return tuple(warnings)


def scenario_pool_fingerprint(scenario_dir: os.PathLike[str] | str) -> str:
    directory = Path(scenario_dir)
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def files_fingerprint(paths: Iterable[os.PathLike[str] | str]) -> str:
    """Hash selected protocol-critical source files independent of git noise."""
    normalized = sorted(Path(path).resolve() for path in paths)
    digest = hashlib.sha256()
    for path in normalized:
        if not path.is_file():
            raise FileNotFoundError(f"fingerprint source not found: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def get_git_metadata(workdir: os.PathLike[str] | str) -> Dict[str, Any]:
    def _run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout.strip()

    try:
        commit = _run("rev-parse", "HEAD")
        status = _run("status", "--porcelain", "--untracked-files=no")
        return {
            "commit": commit,
            "dirty": bool(status),
            "dirty_paths": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "commit": "unknown",
            "dirty": True,
            "dirty_paths": [f"git metadata unavailable: {error}"],
        }


def runtime_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def validate_checkpoint_payload(
    checkpoint: Mapping[str, Any],
    *,
    expected_protocol_versions: Mapping[str, Any],
    expected_action_spec: Mapping[str, Any],
    expected_model_spec: Mapping[str, Any],
    expected_training_config: Mapping[str, Any],
    expected_critic_spec: Optional[Mapping[str, Any]] = None,
) -> None:
    validate_checkpoint_header(checkpoint)
    required = {
        "protocol_versions",
        "checkpoint_kind",
        "update",
        "actors",
        "critic",
        "value_normalizer",
        "critic_spec",
        "actor_optims",
        "critic_optim",
        "args",
        "action_spec",
        "model_spec",
        "training_config",
        "lr_schedule",
        "run_metadata",
        "curriculum_state",
        "rng_state",
        "last_metrics",
        "total_episodes",
        "total_successes",
        "total_collisions",
        "total_timeouts",
        "total_offroads",
    }
    missing = required - set(checkpoint)
    if missing:
        raise CheckpointProtocolError(
            f"checkpoint missing required fields: {sorted(missing)}"
        )
    comparisons = (
        ("protocol_versions", checkpoint["protocol_versions"], expected_protocol_versions),
        ("action_spec", checkpoint["action_spec"], expected_action_spec),
        ("model_spec", checkpoint["model_spec"], expected_model_spec),
        ("training_config", checkpoint["training_config"], expected_training_config),
        *(
            (("critic_spec", checkpoint["critic_spec"], expected_critic_spec),)
            if expected_critic_spec is not None
            else ()
        ),
    )
    mismatches = []
    for group_name, saved, expected in comparisons:
        if not isinstance(saved, Mapping):
            mismatches.append(f"{group_name}: saved value is not a mapping")
            continue
        for key, expected_value in expected.items():
            saved_value = saved.get(key, "<missing>")
            if saved_value != expected_value:
                mismatches.append(
                    f"{group_name}.{key}: saved={saved_value!r}, "
                    f"current={expected_value!r}"
                )
    if mismatches:
        raise CheckpointProtocolError(
            "resume compatibility check failed:\n  - " + "\n  - ".join(mismatches)
        )

    expected_agent_ids = set(expected_model_spec["agent_ids"])
    for key in ("actors", "actor_optims"):
        if not isinstance(checkpoint[key], Mapping):
            raise CheckpointProtocolError(f"{key} must be a mapping")
        actual_ids = {int(agent_id) for agent_id in checkpoint[key]}
        if actual_ids != expected_agent_ids:
            raise CheckpointProtocolError(
                f"{key} agent ids mismatch: saved={sorted(actual_ids)}, "
                f"expected={sorted(expected_agent_ids)}"
            )
    if not isinstance(checkpoint["curriculum_state"], Mapping):
        raise CheckpointProtocolError("curriculum_state must be a mapping")
    if not isinstance(checkpoint["value_normalizer"], Mapping):
        raise CheckpointProtocolError("value_normalizer must be a state mapping")
    if not isinstance(checkpoint["critic_spec"], Mapping):
        raise CheckpointProtocolError("critic_spec must be a mapping")


def validate_checkpoint_header(
    checkpoint: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> None:
    required = {
        "checkpoint_schema_version",
        "training_protocol_version",
        "run_metadata",
        "training_config",
        "lr_schedule",
        "rng_state",
        "protocol_versions",
        "last_metrics",
        "update",
    }
    missing = required - set(checkpoint)
    if missing:
        raise CheckpointProtocolError(
            f"checkpoint header missing required fields: {sorted(missing)}"
        )
    protocol_versions = checkpoint["protocol_versions"]
    if not isinstance(protocol_versions, Mapping):
        raise CheckpointProtocolError("protocol_versions must be a mapping")
    is_legacy = (
        checkpoint["checkpoint_schema_version"]
        == LEGACY_CHECKPOINT_SCHEMA_VERSION
        and checkpoint["training_protocol_version"]
        == LEGACY_TRAINING_PROTOCOL_VERSION
        and protocol_versions.get("action") == LEGACY_ACTION_PROTOCOL_VERSION
    )
    is_action_historical = (
        checkpoint["checkpoint_schema_version"] == ACTION_CHECKPOINT_SCHEMA_VERSION
        and checkpoint["training_protocol_version"]
        == ACTION_TRAINING_PROTOCOL_VERSION
        and protocol_versions.get("action") == ACTION_PROTOCOL_VERSION
    )
    is_valuenorm_historical = (
        checkpoint["checkpoint_schema_version"]
        == VALUENORM_CHECKPOINT_SCHEMA_VERSION
        and checkpoint["training_protocol_version"]
        == VALUENORM_TRAINING_PROTOCOL_VERSION
        and protocol_versions.get("action") == ACTION_PROTOCOL_VERSION
    )
    if allow_legacy and is_legacy:
        expected_schema = LEGACY_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = LEGACY_TRAINING_PROTOCOL_VERSION
    elif allow_legacy and is_action_historical:
        expected_schema = ACTION_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = ACTION_TRAINING_PROTOCOL_VERSION
    elif allow_legacy and is_valuenorm_historical:
        expected_schema = VALUENORM_CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = VALUENORM_TRAINING_PROTOCOL_VERSION
    else:
        expected_schema = CHECKPOINT_SCHEMA_VERSION
        expected_training_protocol = TRAINING_PROTOCOL_VERSION
    if checkpoint["checkpoint_schema_version"] != expected_schema:
        raise CheckpointProtocolError(
            "checkpoint schema mismatch: "
            f"saved={checkpoint['checkpoint_schema_version']}, "
            f"expected={expected_schema}"
        )
    if checkpoint["training_protocol_version"] != expected_training_protocol:
        raise CheckpointProtocolError(
            "training protocol mismatch: "
            f"saved={checkpoint['training_protocol_version']}, "
            f"expected={expected_training_protocol}"
        )
    if not is_legacy and "action_spec" not in checkpoint:
        raise CheckpointProtocolError(
            "current checkpoint header missing required field: action_spec"
        )
    if not is_legacy and not is_action_historical:
        for field in ("value_normalizer", "critic_spec"):
            if field not in checkpoint:
                raise CheckpointProtocolError(
                    f"current checkpoint header missing required field: {field}"
                )
    run_metadata = checkpoint["run_metadata"]
    if not isinstance(run_metadata, Mapping):
        raise CheckpointProtocolError("run_metadata must be a mapping")
    for key in (
        "run_id",
        "seed",
        "origin_git_sha",
        "last_git_sha",
        "code_fingerprint",
    ):
        if key not in run_metadata:
            raise CheckpointProtocolError(
                f"run_metadata missing required field: {key}"
            )
    if not str(run_metadata["run_id"]).strip():
        raise CheckpointProtocolError("run_metadata.run_id cannot be empty")
    try:
        update = int(checkpoint["update"])
    except (TypeError, ValueError) as error:
        raise CheckpointProtocolError("checkpoint update must be an integer") from error
    if update < 0:
        raise CheckpointProtocolError("checkpoint update cannot be negative")

    metrics_protocol = str(protocol_versions.get("metrics", ""))
    if not metrics_protocol:
        raise CheckpointProtocolError(
            "protocol_versions.metrics cannot be empty"
        )
    last_metrics = checkpoint["last_metrics"]
    if update == 0:
        if last_metrics not in (None, {}):
            raise CheckpointProtocolError(
                "update-0 checkpoint must not contain last_metrics"
            )
    else:
        if not isinstance(last_metrics, Mapping):
            raise CheckpointProtocolError(
                "completed checkpoint must contain last_metrics mapping"
            )
        try:
            metrics_update = int(last_metrics.get("update", -1))
        except (TypeError, ValueError) as error:
            raise CheckpointProtocolError(
                "last_metrics update must be an integer"
            ) from error
        if metrics_update != update:
            raise CheckpointProtocolError(
                "last_metrics update does not match checkpoint update"
            )
        if last_metrics.get("metrics_protocol_version") != metrics_protocol:
            raise CheckpointProtocolError("last_metrics protocol mismatch")
        if str(last_metrics.get("run_id", "")) != str(run_metadata["run_id"]):
            raise CheckpointProtocolError("last_metrics run_id mismatch")
        scalars = last_metrics.get("scalars")
        if not isinstance(scalars, Mapping) or not scalars:
            raise CheckpointProtocolError(
                "last_metrics scalars must be a non-empty mapping"
            )
        try:
            scalar_values = np.asarray(
                [float(value) for value in scalars.values()],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise CheckpointProtocolError(
                "last_metrics scalars must be numeric"
            ) from error
        if not np.all(np.isfinite(scalar_values)):
            raise CheckpointProtocolError(
                "last_metrics scalars contain NaN/Inf"
            )

    if not isinstance(checkpoint["training_config"], Mapping):
        raise CheckpointProtocolError("training_config must be a mapping")
    lr_schedule = checkpoint["lr_schedule"]
    if not isinstance(lr_schedule, Mapping) or "decay_horizon" not in lr_schedule:
        raise CheckpointProtocolError(
            "lr_schedule must contain decay_horizon"
        )
    try:
        decay_horizon = int(lr_schedule["decay_horizon"])
    except (TypeError, ValueError) as error:
        raise CheckpointProtocolError(
            "lr_schedule.decay_horizon must be an integer"
        ) from error
    if decay_horizon <= 0:
        raise CheckpointProtocolError(
            "lr_schedule.decay_horizon must be positive"
        )

    rng_state = checkpoint["rng_state"]
    if not isinstance(rng_state, Mapping):
        raise CheckpointProtocolError("rng_state must be a mapping")
    missing_rng = {"python", "numpy", "torch_cpu", "torch_cuda"} - set(
        rng_state
    )
    if missing_rng:
        raise CheckpointProtocolError(
            f"RNG state missing fields: {sorted(missing_rng)}"
        )


def load_manifest(path: os.PathLike[str] | str) -> Dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CheckpointProtocolError("checkpoint manifest root must be a dictionary")
    return payload


def manifest_checkpoint_record(
    checkpoint_path: os.PathLike[str] | str,
    *,
    update: int,
    digest: str,
) -> Dict[str, Any]:
    return {
        "path": str(Path(checkpoint_path).name),
        "update": int(update),
        "sha256": digest,
        "recorded_at": utc_now_iso(),
    }


def normalize_command_line(argv: Optional[Iterable[str]] = None) -> str:
    values = list(sys.argv if argv is None else argv)
    return subprocess.list2cmdline(values)
