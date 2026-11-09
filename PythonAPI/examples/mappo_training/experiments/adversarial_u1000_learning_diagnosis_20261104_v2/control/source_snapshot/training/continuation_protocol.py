"""Explicit budget-only continuation; the parent scientific contract stays frozen."""
from __future__ import annotations
import json
from pathlib import Path
from training.checkpoint_protocol import CheckpointProtocolError, sha256_file


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise CheckpointProtocolError(f"continuation path escapes project: {name}")
    return path


def load_continuation(args, parent, project_dir):
    """Validate all pinned sources before accepting the larger stop budget."""
    path = _inside(project_dir, args.continuation_contract)
    p = json.loads(path.read_text(encoding="utf-8"))
    if p.get("version") != "r8c1_budget_continuation_v1":
        raise CheckpointProtocolError("unknown continuation protocol")
    if not args.resume or args.pretrained_checkpoint or args.allow_resume_code_drift:
        raise CheckpointProtocolError("continuation requires strict full-state resume")
    if p["parent_contract_sha256"] != parent["sha256"] or p["run_id"] != args.run_id:
        raise CheckpointProtocolError("continuation parent identity mismatch")
    if (p["parent_update"] != parent["payload"]["planned_total_updates"]
            or p["planned_total_updates"] != 600
            or parent["payload"]["trainer_args"]["lr_decay_horizon"] != 600
            or parent["payload"]["trainer_args"]["entropy_anneal_updates"] != 600):
        raise CheckpointProtocolError("continuation budget/schedule mismatch")
    if not p["parent_update"] < args.updates <= p["planned_total_updates"]:
        raise CheckpointProtocolError("continuation stop outside approved budget")
    pins = p["source_sha256"]
    required = set()
    for name in parent["payload"]["critical_source_paths"]:
        source = _inside(project_dir, name)
        paths = source.rglob("*") if source.is_dir() else [source]
        required.update(f.relative_to(project_dir).as_posix() for f in paths
                        if f.is_file() and "__pycache__" not in f.parts)
    required.add("training/continuation_protocol.py")
    if not required <= pins.keys():
        raise CheckpointProtocolError("continuation omits critical source pins")
    for name, digest in pins.items():
        if sha256_file(_inside(project_dir, name)) != digest:
            raise CheckpointProtocolError(f"continuation source changed: {name}")
    old = p["parent_source_sha256"]
    changed = sorted(name for name in old if old[name] != pins.get(name))
    if changed != sorted(p["allowed_changed_sources"]):
        raise CheckpointProtocolError("continuation source change allowlist mismatch")
    if set(changed) - {"training/train_no_rllib_mappo_robust_v2.py", "training/experiment_contract.py"}:
        raise CheckpointProtocolError("budget extension changed scientific source")
    return {"path": str(path), "sha256": sha256_file(path), "payload": p}


def validate_continuation_checkpoint(record, checkpoint, digest, current_fingerprint,
                                     current_git_sha, fingerprint_paths, project_dir):
    """Only the exact parent can cross the audited source transition once."""
    p, meta = record["payload"], checkpoint["run_metadata"]
    for path in fingerprint_paths:
        name = Path(path).resolve().relative_to(project_dir).as_posix()
        if p["source_sha256"].get(name) != sha256_file(path):
            raise CheckpointProtocolError(f"continuation source unpinned: {name}")
    saved = meta.get("continuation")
    if saved is None:
        if (checkpoint["update"] != p["parent_update"]
                or digest != p["parent_checkpoint_sha256"]
                or meta["code_fingerprint"] != p["parent_code_fingerprint"]
                or meta["last_git_sha"] != p["parent_git_sha"]):
            raise CheckpointProtocolError("not the registered continuation parent")
    elif (saved["sha256"] != record["sha256"]
          or meta["code_fingerprint"] != current_fingerprint
          or meta["last_git_sha"] != current_git_sha):
        raise CheckpointProtocolError("continuation identity changed on resume")
    if checkpoint["update"] >= p["planned_total_updates"]:
        raise CheckpointProtocolError("continuation budget already completed")
    return {"id": p["id"], "sha256": record["sha256"], "path": record["path"],
            "parent_update": p["parent_update"],
            "parent_checkpoint_sha256": p["parent_checkpoint_sha256"],
            "planned_total_updates": p["planned_total_updates"],
            "compatibility": "audited_budget_only_full_state_resume"}
