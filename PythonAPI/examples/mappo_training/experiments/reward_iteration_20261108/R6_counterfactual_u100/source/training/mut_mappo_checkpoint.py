"""Atomic complete-A+B checkpoints and append-only actual-cost audit.

An immutable checkpoint plus SHA is staged first. A single atomic manifest
replacement commits the complete boundary; metrics can then be replayed after
a crash. Partial A-only work and audit costs never masquerade as safe rounds.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

from training.checkpoint_protocol import (
    CheckpointProtocolError, atomic_copy_checkpoint, atomic_write_json,
    load_checkpoint_file, save_checkpoint_file, utc_now_iso,
)
from training.mut_mappo_learning import actor_states_hash


SCHEMA_VERSION = "mut_mappo_complete_team_boundary_v2"
REQUIRED_PAYLOAD = {"teams", "teachers", "gate", "completed_round", "next_team",
                    "counters", "last_metrics"}


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_payload(payload: Mapping) -> None:
    if not isinstance(payload, Mapping) or not REQUIRED_PAYLOAD.issubset(payload):
        missing = REQUIRED_PAYLOAD - set(payload) if isinstance(payload, Mapping) else REQUIRED_PAYLOAD
        raise CheckpointProtocolError(f"A/B checkpoint missing fields: {sorted(missing)}")
    completed = payload["completed_round"]
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise CheckpointProtocolError("completed_round must be a nonnegative integer")
    if payload["next_team"] not in {"A", "B"}:
        raise CheckpointProtocolError("next_team must be A or B at a complete boundary")
    team_count = payload.get("team_count", 2)
    if isinstance(team_count, bool) or team_count not in (1, 2):
        raise CheckpointProtocolError("declared team_count must be 1 or 2")
    expected_teams = {"A"} if team_count == 1 else {"A", "B"}
    if set(payload["teams"]) != expected_teams or set(payload["teachers"]) != expected_teams:
        raise CheckpointProtocolError("complete boundary must contain every declared team and teacher")
    if payload["next_team"] not in expected_teams:
        raise CheckpointProtocolError("next_team is not present in this training topology")
    team_fields = {"actors", "critic", "actor_optims", "critic_optim", "value_normalizer",
                   "rng", "scenario_rng", "update_count", "cost_state", "schedule_state",
                   "peer_snapshot", "peer_gates", "anchor", "model_spec", "config"}
    for name, team in payload["teams"].items():
        if not team_fields.issubset(team):
            raise CheckpointProtocolError(f"team {name} incomplete recovery state")
        if set(team["actors"]) != {2, 3} or set(team["actor_optims"]) != {2, 3}:
            raise CheckpointProtocolError(f"team {name} does not contain both blue roles")
        spec = team["model_spec"]
        team_only = spec.get("critic_input_scope") == "own_blue_team_observations_plus_local_id"
        expected_id_dim = 2 if team_only else 4
        if any(spec.get(key) != value for key, value in
               (("obs_dim", 20), ("action_dim", 2), ("global_state_dim", 80), ("critic_id_dim", expected_id_dim))):
            raise CheckpointProtocolError(f"team {name} model protocol differs from declared critic scope")
        if team_only:
            if (spec.get("algorithm") != "mappo" or spec.get("critic_value_state_dim") != 40
                    or team["config"].get("critic_scope") != "blue_team"):
                raise CheckpointProtocolError("two-blue critic requires MAPPO, 40 observations and matching scope")
            first_weight = team["critic"].get("net.0.weight")
            if first_weight is None or len(first_weight.shape) != 2 or first_weight.shape[1] != 42:
                raise CheckpointProtocolError("two-blue critic weight input must be 40+2")
        if spec.get("algorithm", "mappo") == "ippo":
            if spec.get("critic_input_scope") != "own_physical_slot_local_observation_only":
                raise CheckpointProtocolError("IPPO critic must not consume another physical role's observation")
            if any(set(team[key]) != {2, 3} for key in ("critic", "critic_optim", "value_normalizer")):
                raise CheckpointProtocolError("IPPO needs each blue role's independent critic/optimizer/ValueNorm")
        if team["update_count"] != completed or team["schedule_state"].get("update") != completed:
            raise CheckpointProtocolError("checkpoint is not a complete A+B update boundary")
        if set(team["rng"]) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
            raise CheckpointProtocolError(f"team {name} missing RNG streams")
    for name, snapshot in payload["teachers"].items():
        # None is allowed before the first peer-refresh boundary.
        if snapshot is None:
            continue
        if not {"actors", "hash", "version", "model_spec"}.issubset(snapshot):
            raise CheckpointProtocolError(f"teacher {name} snapshot incomplete")
        if snapshot["hash"] != actor_states_hash(snapshot["actors"]):
            raise CheckpointProtocolError(f"teacher {name} content hash mismatch")
    for student, peer in (("A", "B"), ("B", "A")):
        if student not in expected_teams:
            continue
        stored_peer = payload["teams"][student]["peer_snapshot"]
        authoritative = payload["teachers"].get(peer)
        if stored_peer is not None and (authoritative is None or
                stored_peer.get("hash") != authoritative.get("hash") or
                stored_peer.get("version") != authoritative.get("version")):
            raise CheckpointProtocolError("team peer differs from committed frozen teacher")
    if not isinstance(payload["gate"], Mapping) or not isinstance(payload["counters"], Mapping):
        raise CheckpointProtocolError("gate/counters must be explicit recovery dictionaries")
    if not isinstance(payload["last_metrics"], Mapping):
        raise CheckpointProtocolError("last_metrics must be replayable JSON metadata")
    if payload["last_metrics"].get("completed_round", completed) != completed:
        raise CheckpointProtocolError("last_metrics belongs to a different completed boundary")
    try:
        json.dumps(_json_safe(payload["last_metrics"]), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CheckpointProtocolError("last_metrics contains non-JSON/NaN values") from error


class CheckpointStore:
    def __init__(self, run_dir, identity: Mapping):
        self.run_dir = Path(run_dir).resolve()
        self.identity = copy.deepcopy(dict(identity))
        # Ensure identity is stable JSON and does not hide numpy equivalence.
        json.dumps(self.identity, sort_keys=True, allow_nan=False)
        self.manifest_path = self.run_dir / "checkpoint_manifest.json"
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.audit_path = self.run_dir / "actual_cost_audit.jsonl"

    def _read_manifest(self):
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("identity") != self.identity:
            raise CheckpointProtocolError("resume manifest schema or run identity differs")
        filename = manifest.get("checkpoint", "")
        candidate = (self.run_dir / filename).resolve()
        if candidate.parent != (self.run_dir / "boundaries").resolve():
            raise CheckpointProtocolError("manifest checkpoint escapes the boundaries directory")
        return manifest, candidate

    def save(self, payload: Mapping, metrics: Mapping | None = None) -> dict:
        complete = copy.deepcopy(dict(payload))
        if metrics is not None:
            complete["last_metrics"] = copy.deepcopy(dict(metrics))
        _validate_payload(complete)
        if self.manifest_path.exists():
            previous, _ = self._read_manifest()
            if complete["completed_round"] <= previous["completed_round"]:
                raise CheckpointProtocolError("safe checkpoint rounds must increase")
        complete["schema_version"] = SCHEMA_VERSION
        complete["identity"] = copy.deepcopy(self.identity)
        filename = f"round_{complete['completed_round']:08d}_{uuid.uuid4().hex[:12]}.pt"
        path = self.run_dir / "boundaries" / filename
        digest = save_checkpoint_file(complete, path, immutable=True)
        manifest = {"schema_version": SCHEMA_VERSION, "identity": self.identity,
                    "checkpoint": "boundaries/" + filename, "sha256": digest,
                    "completed_round": complete["completed_round"], "saved_at": utc_now_iso()}
        atomic_write_json(manifest, self.manifest_path)
        # Convenience copy is not the recovery authority; the manifest is.
        atomic_copy_checkpoint(path, self.run_dir / "latest.pt")
        self._replay_metrics(complete)
        return manifest

    commit = save

    def load(self, *, map_location="cpu") -> dict:
        manifest, path = self._read_manifest()
        payload, digest = load_checkpoint_file(path, map_location=map_location)
        if digest != manifest["sha256"] or payload.get("identity") != self.identity:
            raise CheckpointProtocolError("manifest/payload SHA or identity differs")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointProtocolError("payload checkpoint schema differs")
        _validate_payload(payload)
        if payload["completed_round"] != manifest["completed_round"]:
            raise CheckpointProtocolError("manifest/payload completed boundary differs")
        self._replay_metrics(payload)
        return payload

    def _append(self, path: Path, record: Mapping, *, ascii_only=False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        unterminated_tail = False
        if path.exists() and path.stat().st_size:
            with path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                unterminated_tail = existing.read(1) != b"\n"
        record = dict(record)
        if unterminated_tail:
            # Retain the interrupted bytes as evidence, but delimit them so
            # the first resumed cost record remains independently parseable.
            record["previous_unterminated_tail"] = True
        line = json.dumps(_json_safe(record), ensure_ascii=ascii_only, sort_keys=True, allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8") as stream:
            if unterminated_tail:
                stream.write("\n")
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def _replay_metrics(self, payload) -> None:
        record = _json_safe(dict(payload["last_metrics"]))
        record["completed_round"] = payload["completed_round"]
        rows = []
        torn = False
        if self.metrics_path.exists():
            raw = self.metrics_path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn last append is recoverable from last_metrics.
                    if line != raw.splitlines()[-1]:
                        raise CheckpointProtocolError("metrics has corrupt interior record")
                    torn = True
        matching = [row for row in rows if row.get("completed_round") == payload["completed_round"]]
        if matching:
            if len(matching) != 1 or matching[0] != record:
                raise CheckpointProtocolError("committed metrics conflict with checkpoint replay")
            if torn or any(row.get("completed_round", -1) > payload["completed_round"] for row in rows):
                from training.checkpoint_protocol import _atomic_replace_bytes
                clean = [row for row in rows if row.get("completed_round", -1) <= payload["completed_round"]]
                data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in clean)
                _atomic_replace_bytes(data.encode("utf-8"), self.metrics_path)
            return
        # Rewrite a clean, deduplicated prefix atomically before append. A
        # previous interrupted A-only metric must not appear as a full round.
        prefix = [row for row in rows if row.get("completed_round", -1) < payload["completed_round"]]
        if torn or rows != prefix or (self.metrics_path.exists() and not raw.endswith("\n")):
            from training.checkpoint_protocol import _atomic_replace_bytes
            data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prefix)
            _atomic_replace_bytes(data.encode("utf-8"), self.metrics_path)
        self._append(self.metrics_path, record)

    def append_audit(self, record: Mapping) -> None:
        """Actual ticks survive rollback of a partial A/B round."""
        # ASCII escaping also makes a mid-write prefix valid UTF-8, even if an
        # interruption splits an originally non-ASCII diagnostic string.
        self._append(self.audit_path, dict(record, audited_at=utc_now_iso()), ascii_only=True)

    def export(self, team: str, path) -> str:
        payload = self.load()
        if team not in payload["teams"]:
            raise ValueError("export team is absent from this training topology")
        state = payload["teams"][team]
        return save_checkpoint_file({"schema_version": "mut_mappo_blue_export_v1",
            "team": team, "actors": state["actors"], "model_spec": state["model_spec"],
            "actor_hash": actor_states_hash(state["actors"]), "identity": self.identity,
            "completed_round": payload["completed_round"]}, path)
