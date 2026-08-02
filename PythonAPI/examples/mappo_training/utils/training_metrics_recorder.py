"""Centralized, versioned training-metric aggregation and persistence.

Environment and optimizer code provide plain data to this module.  The module
validates reward conservation, keeps agent/lifecycle ownership intact, writes
the same update-level scalar payload to TensorBoard and JSONL, and preserves a
per-episode lifecycle audit trail for real-CARLA smoke validation.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from training.metrics_protocol import (
    DISABLED_REWARD_COMPONENTS,
    EPISODE_RECORD_VERSION,
    METRICS_PROTOCOL_VERSION,
    REWARD_AUDIT_ATOL,
    SPARSE_REWARD_COMPONENTS,
    MetricsProtocolError,
    reward_components,
    reward_diagnostics,
    reward_schema_document,
    validate_active_reward_payload,
    validate_episode_record,
)


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    nonzero_count: int = 0
    nonzero_total: float = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise MetricsProtocolError(f"running statistic received {value!r}")
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if value != 0.0:
            self.nonzero_count += 1
            self.nonzero_total += value

    def merge(self, other: "RunningStats") -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        self.nonzero_count += other.nonzero_count
        self.nonzero_total += other.nonzero_total

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        if not self.count:
            return 0.0
        variance = max(0.0, self.total_sq / self.count - self.mean ** 2)
        return math.sqrt(variance)

    @property
    def nonzero_rate(self) -> float:
        return self.nonzero_count / self.count if self.count else 0.0

    @property
    def nonzero_mean(self) -> float:
        return (
            self.nonzero_total / self.nonzero_count
            if self.nonzero_count
            else 0.0
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
    return name[:80] or "unnamed"


def _atomic_write_json(payload: Mapping[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
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


def _atomic_write_jsonl(
    payloads: Sequence[Mapping[str, object]], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        for payload in payloads
    ).encode("utf-8")
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


def _finite_scalar(value: object, *, tag: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise MetricsProtocolError(
            f"metric {tag!r} must be a numeric scalar, got {value!r}"
        ) from error
    if not math.isfinite(scalar):
        raise MetricsProtocolError(f"metric {tag!r} contains NaN/Inf: {scalar!r}")
    return scalar


def _module_parameter_snapshot(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _parameter_norm(module: torch.nn.Module) -> float:
    total_sq = 0.0
    for parameter in module.parameters():
        values = parameter.detach()
        total_sq += float(torch.sum(values * values).item())
    return math.sqrt(total_sq)


def _parameter_delta_norm(
    module: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]
) -> float:
    current = dict(module.named_parameters())
    if set(current) != set(snapshot):
        raise MetricsProtocolError("model parameter names changed during one update")
    total_sq = 0.0
    for name, parameter in current.items():
        delta = parameter.detach().cpu() - snapshot[name]
        total_sq += float(torch.sum(delta * delta).item())
    return math.sqrt(total_sq)


class TrainingMetricsRecorder:
    """Aggregate one complete update and persist it after checkpoint commit."""

    def __init__(
        self,
        *,
        run_dir: os.PathLike[str] | str,
        run_id: str,
        agent_ids: Iterable[int],
        reward_version: str,
        start_update: int,
        completed_update: int,
        resume_metrics_payload: Optional[Mapping[str, object]] = None,
        histogram_interval: int = 10,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir = self.run_dir / "tb"
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.episode_jsonl_path = self.run_dir / "episodes.jsonl"
        self.schema_path = self.run_dir / "metrics_schema.json"
        self.run_id = str(run_id)
        self.agent_ids = tuple(int(agent_id) for agent_id in agent_ids)
        if not self.run_id.strip():
            raise MetricsProtocolError("metrics run_id cannot be empty")
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise MetricsProtocolError("metrics agent_ids must be unique and non-empty")
        if int(start_update) != int(completed_update) + 1:
            raise MetricsProtocolError(
                "metrics start_update must equal completed_update + 1"
            )
        if histogram_interval < 0:
            raise MetricsProtocolError("histogram_interval cannot be negative")

        self.reward_version = str(reward_version)
        self.component_names = reward_components(self.reward_version)
        self.diagnostic_names = reward_diagnostics(self.reward_version)
        self.histogram_interval = int(histogram_interval)
        self._closed = False
        self._pending_payload: Optional[Dict[str, object]] = None
        self._pending_histogram_modules = None

        schema = reward_schema_document(self.reward_version)
        schema.update(
            {
                "run_id": self.run_id,
                "agent_ids": list(self.agent_ids),
                "tensorboard_directory": "tb",
                "jsonl_file": self.jsonl_path.name,
                "episode_records": {
                    "version": EPISODE_RECORD_VERSION,
                    "jsonl_file": self.episode_jsonl_path.name,
                    "commit_semantics": (
                        "fsync each episode before checkpoint; discard records "
                        "ahead of the latest safe update on resume"
                    ),
                },
            }
        )
        if self.schema_path.is_file():
            existing_schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
            if existing_schema != schema:
                raise MetricsProtocolError(
                    "existing metrics_schema.json differs from current protocol"
                )
        else:
            _atomic_write_json(schema, self.schema_path)

        self._episode_record_counts_by_update = self._reconcile_episode_jsonl(
            completed_update=int(completed_update)
        )
        existing_updates = self._read_jsonl_updates()
        if existing_updates and existing_updates[-1] > completed_update:
            raise MetricsProtocolError(
                "metrics.jsonl is ahead of the safe checkpoint update"
            )
        self._last_committed_update = existing_updates[-1] if existing_updates else 0
        self.writer = SummaryWriter(
            str(self.tb_dir),
            purge_step=start_update if completed_update > 0 else None,
        )
        try:
            self.writer.add_text(
                "run/metrics_schema",
                "```json\n"
                + json.dumps(schema, ensure_ascii=False, indent=2)
                + "\n```",
                global_step=max(0, completed_update),
            )
            self._reconcile_checkpoint_metrics(
                completed_update=completed_update,
                resume_metrics_payload=resume_metrics_payload,
            )
            self._validate_completed_episode_coverage(
                completed_update=int(completed_update)
            )
            self._reset_window()
        except Exception:
            self.writer.close()
            self._closed = True
            raise

    def _reconcile_episode_jsonl(self, *, completed_update: int) -> Dict[int, int]:
        if not self.episode_jsonl_path.is_file():
            _atomic_write_jsonl([], self.episode_jsonl_path)
            return {}

        retained_records = []
        counts: Dict[int, int] = {}
        last_update = 0
        expected_episode_index = 0
        pruned = False
        for line_number, line in enumerate(
            self.episode_jsonl_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise MetricsProtocolError(
                    f"episodes.jsonl line {line_number} is corrupt"
                ) from error
            normalized = validate_episode_record(
                payload,
                run_id=self.run_id,
                agent_ids=self.agent_ids,
                context=f"episodes.jsonl line {line_number}",
            )
            update = int(normalized["update"])
            episode_index = int(normalized["episode_index"])
            if update < last_update:
                raise MetricsProtocolError(
                    "episodes.jsonl updates must be non-decreasing"
                )
            if update != last_update:
                expected_episode_index = 0
                last_update = update
            if episode_index != expected_episode_index:
                raise MetricsProtocolError(
                    f"episodes.jsonl update {update} expected episode_index "
                    f"{expected_episode_index}, got {episode_index}"
                )
            expected_episode_index += 1
            if update > completed_update:
                pruned = True
                continue
            retained_records.append(normalized)
            counts[update] = counts.get(update, 0) + 1

        if pruned:
            _atomic_write_jsonl(retained_records, self.episode_jsonl_path)
        return counts

    def _validate_completed_episode_coverage(self, *, completed_update: int) -> None:
        metric_updates = set(self._metrics_payloads_by_update)
        episode_updates = set(self._episode_record_counts_by_update)
        if episode_updates != metric_updates:
            raise MetricsProtocolError(
                "episodes.jsonl completed updates differ from metrics.jsonl; "
                f"episodes={sorted(episode_updates)}, metrics={sorted(metric_updates)}"
            )
        if metric_updates and max(metric_updates) != completed_update:
            raise MetricsProtocolError(
                "episode/metric audit does not reach the safe checkpoint update"
            )
        for update, payload in self._metrics_payloads_by_update.items():
            expected = int(payload["scalars"]["episode/count"])
            actual = self._episode_record_counts_by_update.get(update, 0)
            if actual != expected:
                raise MetricsProtocolError(
                    f"episodes.jsonl update {update} count={actual}, "
                    f"metrics episode/count={expected}"
                )

    def _read_jsonl_updates(self) -> Sequence[int]:
        self._last_jsonl_payload: Optional[Dict[str, object]] = None
        self._metrics_payloads_by_update: Dict[int, Dict[str, object]] = {}
        if not self.jsonl_path.is_file():
            return []
        updates = []
        for line_number, line in enumerate(
            self.jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise MetricsProtocolError(
                    f"metrics.jsonl line {line_number} is corrupt"
                ) from error
            normalized = self._validate_payload(payload)
            update = int(normalized["update"])
            if update <= 0 or (updates and update <= updates[-1]):
                raise MetricsProtocolError(
                    "metrics.jsonl updates must be positive and strictly increasing"
                )
            updates.append(update)
            self._last_jsonl_payload = normalized
            self._metrics_payloads_by_update[update] = normalized
        return updates

    def _validate_payload(self, payload: Mapping[str, object]) -> Dict[str, object]:
        if payload.get("metrics_protocol_version") != METRICS_PROTOCOL_VERSION:
            raise MetricsProtocolError("metrics payload protocol mismatch")
        if str(payload.get("run_id", "")) != self.run_id:
            raise MetricsProtocolError("metrics payload run_id mismatch")
        update = int(payload.get("update", -1))
        if update <= 0:
            raise MetricsProtocolError("metrics payload update must be positive")
        scalars = payload.get("scalars")
        if not isinstance(scalars, Mapping) or not scalars:
            raise MetricsProtocolError("metrics payload scalars must be non-empty")
        normalized_scalars = {
            str(tag): _finite_scalar(value, tag=str(tag))
            for tag, value in scalars.items()
        }
        return {
            "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "update": update,
            "created_at": str(payload.get("created_at", _utc_now_iso())),
            "scalars": normalized_scalars,
        }

    def _reconcile_checkpoint_metrics(
        self,
        *,
        completed_update: int,
        resume_metrics_payload: Optional[Mapping[str, object]],
    ) -> None:
        if completed_update == 0:
            if resume_metrics_payload not in (None, {}):
                raise MetricsProtocolError(
                    "update-0 checkpoint must not contain committed metrics"
                )
            return
        if resume_metrics_payload is None:
            raise MetricsProtocolError(
                "safe checkpoint is missing its last committed metrics payload"
            )
        payload = self._validate_payload(resume_metrics_payload)
        if int(payload["update"]) != int(completed_update):
            raise MetricsProtocolError(
                "checkpoint last_metrics update does not match checkpoint update"
            )
        if self._last_committed_update == completed_update:
            assert self._last_jsonl_payload is not None
            comparable_keys = (
                "metrics_protocol_version",
                "run_id",
                "update",
                "scalars",
            )
            if any(
                self._last_jsonl_payload[key] != payload[key]
                for key in comparable_keys
            ):
                raise MetricsProtocolError(
                    "metrics.jsonl last payload differs from safe checkpoint"
                )
            return
        if self._last_committed_update != completed_update - 1:
            raise MetricsProtocolError(
                "metrics.jsonl has a gap before the safe checkpoint update"
            )
        # The checkpoint is authoritative.  A crash may have happened after the
        # atomic checkpoint save but before TensorBoard/JSONL flush; replay the
        # exact payload stored in that checkpoint.
        self._emit_payload(payload)

    def _reset_window(self) -> None:
        self.component_stats = {
            aid: {name: RunningStats() for name in self.component_names}
            for aid in self.agent_ids
        }
        self.episode_component_stats = {
            aid: {name: RunningStats() for name in self.component_names}
            for aid in self.agent_ids
        }
        self.current_episode_component_sums = {
            aid: {name: 0.0 for name in self.component_names}
            for aid in self.agent_ids
        }
        self.diagnostic_stats = {
            aid: {name: RunningStats() for name in self.diagnostic_names}
            for aid in self.agent_ids
        }
        self.action_stats = {
            aid: {
                name: RunningStats()
                for name in (
                    "throttle",
                    "steer",
                    "brake",
                    "throttle_boundary",
                    "steer_boundary",
                    "brake_boundary",
                )
            }
            for aid in self.agent_ids
        }
        self.active_transition_counts = {aid: 0 for aid in self.agent_ids}
        self.current_episode_active_counts = {aid: 0 for aid in self.agent_ids}
        self.episode_return_stats = {aid: RunningStats() for aid in self.agent_ids}
        self.episode_length_stats = RunningStats()
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.end_reason_counts: Dict[str, int] = {}
        self.scenario_stats: Dict[str, Dict[str, RunningStats]] = {}
        self.max_step_reward_residual = 0.0
        self.max_episode_reward_residual = 0.0
        self.external_scalars: Dict[str, float] = {}
        self._model_snapshots = None

    def record_env_step(
        self,
        *,
        active_agent_ids: Iterable[int],
        rewards: Mapping[int, object],
        reward_details: Mapping[int, Mapping[str, object]],
        reward_diagnostics_by_agent: Mapping[int, Mapping[str, object]],
        actions: Mapping[int, object],
        context: str,
    ) -> None:
        active_ids = tuple(int(agent_id) for agent_id in active_agent_ids)
        unknown_ids = set(active_ids) - set(self.agent_ids)
        if unknown_ids:
            raise MetricsProtocolError(
                f"{context}: unknown active agents {sorted(unknown_ids)}"
            )
        normalized = validate_active_reward_payload(
            active_agent_ids=active_ids,
            rewards=rewards,
            reward_details_by_agent=reward_details,
            reward_diagnostics_by_agent=reward_diagnostics_by_agent,
            reward_version=self.reward_version,
            context=context,
        )
        _, details_by_agent, diagnostics_by_agent, residuals = normalized

        normalized_actions: Dict[int, np.ndarray] = {}
        for agent_id in active_ids:
            if agent_id not in actions:
                raise MetricsProtocolError(
                    f"{context} agent={agent_id}: active action is missing"
                )
            action = np.asarray(actions[agent_id], dtype=np.float64)
            if action.shape != (3,) or not np.all(np.isfinite(action)):
                raise MetricsProtocolError(
                    f"{context} agent={agent_id}: action must be finite shape (3,)"
                )
            normalized_actions[agent_id] = action

        for agent_id in active_ids:
            self.active_transition_counts[agent_id] += 1
            self.current_episode_active_counts[agent_id] += 1
            self.max_step_reward_residual = max(
                self.max_step_reward_residual, abs(residuals[agent_id])
            )
            for name, value in details_by_agent[agent_id].items():
                self.component_stats[agent_id][name].add(value)
                self.current_episode_component_sums[agent_id][name] += value
            for name, value in diagnostics_by_agent[agent_id].items():
                self.diagnostic_stats[agent_id][name].add(value)

            throttle, steer, brake = normalized_actions[agent_id]
            action_values = {
                "throttle": throttle,
                "steer": steer,
                "brake": brake,
                "throttle_boundary": float(throttle <= 0.01 or throttle >= 0.99),
                "steer_boundary": float(abs(steer) >= 0.99),
                "brake_boundary": float(brake <= 0.01 or brake >= 0.99),
            }
            for name, value in action_values.items():
                self.action_stats[agent_id][name].add(value)

    def finish_episode(
        self,
        *,
        update: int,
        episode_index: int,
        episode_returns: Mapping[int, object],
        episode_length: int,
        active_decision_steps: Mapping[int, int],
        transition_counts: Mapping[int, int],
        dones: Mapping[int, bool],
        terminated: Mapping[int, bool],
        truncated: Mapping[int, bool],
        bootstrap_enabled: Mapping[int, bool],
        success: bool,
        collision: bool,
        timeout: bool,
        end_reason: str,
        scenario_name: str,
        context: str,
    ) -> None:
        if episode_length <= 0:
            raise MetricsProtocolError(f"{context}: episode_length must be positive")
        expected_agent_ids = set(self.agent_ids)
        lifecycle_mappings = {
            "episode_returns": episode_returns,
            "active_decision_steps": active_decision_steps,
            "transition_counts": transition_counts,
            "dones": dones,
            "terminated": terminated,
            "truncated": truncated,
            "bootstrap_enabled": bootstrap_enabled,
        }
        for name, mapping in lifecycle_mappings.items():
            if set(mapping) != expected_agent_ids:
                raise MetricsProtocolError(
                    f"{context}: {name} agent ids differ; "
                    f"expected={sorted(expected_agent_ids)}, actual={sorted(mapping)}"
                )

        normalized_returns: Dict[int, float] = {}
        episode_residuals: Dict[int, float] = {}
        for agent_id in self.agent_ids:
            if int(active_decision_steps[agent_id]) != int(transition_counts[agent_id]):
                raise MetricsProtocolError(
                    f"{context} agent={agent_id}: active decisions do not match transitions"
                )
            if int(active_decision_steps[agent_id]) != self.current_episode_active_counts[agent_id]:
                raise MetricsProtocolError(
                    f"{context} agent={agent_id}: recorder active-step count mismatch"
                )
            episode_return = _finite_scalar(
                episode_returns[agent_id], tag=f"episode_return_agent_{agent_id}"
            )
            normalized_returns[agent_id] = episode_return
            component_sum = math.fsum(
                self.current_episode_component_sums[agent_id].values()
            )
            residual = episode_return - component_sum
            episode_residuals[agent_id] = residual
            episode_atol = REWARD_AUDIT_ATOL * max(
                1, self.current_episode_active_counts[agent_id]
            )
            if abs(residual) > episode_atol:
                raise MetricsProtocolError(
                    f"{context} agent={agent_id}: episode return conservation failed; "
                    f"return={episode_return}, component_sum={component_sum}, "
                    f"residual={residual}, atol={episode_atol}"
                )

        episode_record = validate_episode_record(
            {
                "episode_record_version": EPISODE_RECORD_VERSION,
                "run_id": self.run_id,
                "update": int(update),
                "episode_index": int(episode_index),
                "created_at": _utc_now_iso(),
                "scenario_name": str(scenario_name),
                "episode_length": int(episode_length),
                "episode_end_reason": str(end_reason),
                "success": bool(success),
                "collision": bool(collision),
                "timeout": bool(timeout),
                "agents": {
                    str(agent_id): {
                        "episode_return": normalized_returns[agent_id],
                        "active_decision_steps": int(
                            active_decision_steps[agent_id]
                        ),
                        "transition_count": int(transition_counts[agent_id]),
                        "done": bool(dones[agent_id]),
                        "terminated": bool(terminated[agent_id]),
                        "truncated": bool(truncated[agent_id]),
                        "bootstrap_enabled": bool(
                            bootstrap_enabled[agent_id]
                        ),
                    }
                    for agent_id in self.agent_ids
                },
            },
            run_id=self.run_id,
            agent_ids=self.agent_ids,
            context=context,
        )
        self._append_episode_record(episode_record)

        for agent_id in self.agent_ids:
            residual = episode_residuals[agent_id]
            self.max_episode_reward_residual = max(
                self.max_episode_reward_residual, abs(residual)
            )
            self.episode_return_stats[agent_id].add(normalized_returns[agent_id])
            for name in self.component_names:
                self.episode_component_stats[agent_id][name].add(
                    self.current_episode_component_sums[agent_id][name]
                )
                self.current_episode_component_sums[agent_id][name] = 0.0
            self.current_episode_active_counts[agent_id] = 0

        self.episode_count += 1
        self.episode_length_stats.add(float(episode_length))
        self.success_count += int(bool(success))
        self.collision_count += int(bool(collision))
        self.timeout_count += int(bool(timeout))
        reason = _safe_name(end_reason)
        self.end_reason_counts[reason] = self.end_reason_counts.get(reason, 0) + 1

        scenario = _safe_name(scenario_name)
        if scenario not in self.scenario_stats:
            self.scenario_stats[scenario] = {
                "success": RunningStats(),
                "collision": RunningStats(),
                "timeout": RunningStats(),
                "length": RunningStats(),
            }
        self.scenario_stats[scenario]["success"].add(float(bool(success)))
        self.scenario_stats[scenario]["collision"].add(float(bool(collision)))
        self.scenario_stats[scenario]["timeout"].add(float(bool(timeout)))
        self.scenario_stats[scenario]["length"].add(float(episode_length))

    def _append_episode_record(self, payload: Mapping[str, object]) -> None:
        update = int(payload["update"])
        episode_index = int(payload["episode_index"])
        if update != self._last_committed_update + 1:
            raise MetricsProtocolError(
                f"episode record update must be next safe update: "
                f"last={self._last_committed_update}, new={update}"
            )
        expected_index = self._episode_record_counts_by_update.get(update, 0)
        if episode_index != expected_index:
            raise MetricsProtocolError(
                f"episode record update {update} expected episode_index "
                f"{expected_index}, got {episode_index}"
            )
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with open(
            self.episode_jsonl_path, "a", encoding="utf-8", newline="\n"
        ) as file_obj:
            file_obj.write(line)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        self._episode_record_counts_by_update[update] = expected_index + 1

    def record_scalar(self, tag: str, value: object) -> None:
        tag = str(tag).strip("/")
        if not tag:
            raise MetricsProtocolError("metric tag cannot be empty")
        scalar = _finite_scalar(value, tag=tag)
        if tag in self.external_scalars and self.external_scalars[tag] != scalar:
            raise MetricsProtocolError(f"metric tag {tag!r} was recorded twice")
        self.external_scalars[tag] = scalar

    def record_scalars(self, scalars: Mapping[str, object]) -> None:
        for tag, value in scalars.items():
            self.record_scalar(str(tag), value)

    def capture_model_snapshot(
        self,
        actors: Mapping[int, torch.nn.Module],
        critic: torch.nn.Module,
    ) -> None:
        if self._model_snapshots is not None:
            raise MetricsProtocolError("model snapshot already captured for this update")
        self._model_snapshots = {
            "actors": {
                int(agent_id): _module_parameter_snapshot(module)
                for agent_id, module in actors.items()
            },
            "critic": _module_parameter_snapshot(critic),
        }

    def record_model_update(
        self,
        actors: Mapping[int, torch.nn.Module],
        critic: torch.nn.Module,
    ) -> None:
        if self._model_snapshots is None:
            raise MetricsProtocolError("model snapshot was not captured")
        for agent_id in self.agent_ids:
            actor = actors[agent_id]
            parameter_norm = _parameter_norm(actor)
            delta_norm = _parameter_delta_norm(
                actor, self._model_snapshots["actors"][agent_id]
            )
            self.record_scalar(
                f"model/actor_agent_{agent_id}/parameter_norm", parameter_norm
            )
            self.record_scalar(
                f"model/actor_agent_{agent_id}/update_norm", delta_norm
            )
            self.record_scalar(
                f"model/actor_agent_{agent_id}/relative_update_norm",
                delta_norm / max(parameter_norm, 1e-12),
            )
            if hasattr(actor, "log_std"):
                log_std = actor.log_std.detach()
                self.record_scalar(
                    f"policy/agent_{agent_id}/log_std_mean", log_std.mean().item()
                )
                self.record_scalar(
                    f"policy/agent_{agent_id}/log_std_min", log_std.min().item()
                )
                self.record_scalar(
                    f"policy/agent_{agent_id}/log_std_max", log_std.max().item()
                )

        critic_parameter_norm = _parameter_norm(critic)
        critic_delta_norm = _parameter_delta_norm(
            critic, self._model_snapshots["critic"]
        )
        self.record_scalar("model/critic/parameter_norm", critic_parameter_norm)
        self.record_scalar("model/critic/update_norm", critic_delta_norm)
        self.record_scalar(
            "model/critic/relative_update_norm",
            critic_delta_norm / max(critic_parameter_norm, 1e-12),
        )
        self._pending_histogram_modules = (dict(actors), critic)

    @staticmethod
    def _mean_metric(metrics: Mapping[str, object], key: str) -> Optional[float]:
        values = metrics.get(key)
        if values is None:
            return None
        if isinstance(values, (list, tuple, np.ndarray)):
            if len(values) == 0:
                return None
            return float(np.mean(values))
        return float(values)

    def record_ppo_update(
        self,
        *,
        ppo_metrics: Optional[Mapping[str, object]],
        advantage_stats: Mapping[int, Mapping[str, object]],
        preupdate_ratio_errors: Mapping[int, object],
        agent_stop_epoch: Mapping[int, int],
        transition_counts: Mapping[int, int],
        actor_lrs: Mapping[int, object],
        critic_lr: object,
        skipped: bool,
        max_grad_norm: float,
    ) -> None:
        self.record_scalar("train/update_skipped", float(bool(skipped)))
        self.record_scalar(
            "rollout/transition_count_total", sum(int(v) for v in transition_counts.values())
        )
        self.record_scalar("optimizer/critic/lr", critic_lr)
        for agent_id in self.agent_ids:
            self.record_scalar(
                f"rollout/agent_{agent_id}/transition_count",
                transition_counts[agent_id],
            )
            self.record_scalar(
                f"optimizer/actor_agent_{agent_id}/lr", actor_lrs[agent_id]
            )

        if skipped:
            return
        if ppo_metrics is None:
            raise MetricsProtocolError("non-skipped PPO update has no metrics")

        critic_metric_names = (
            "critic_loss",
            "grad_norm_critic",
            "explained_variance",
            "value_prediction_mean",
            "value_prediction_std",
            "return_target_mean",
            "return_target_std",
            "value_error_abs_mean",
        )
        for name in critic_metric_names:
            value = self._mean_metric(ppo_metrics, name)
            if value is not None:
                self.record_scalar(f"critic/{name}", value)
        self.record_scalar(
            "critic/samples_seen", ppo_metrics.get("critic_samples_seen", 0)
        )
        critic_grad_values = ppo_metrics.get("grad_norm_critic", [])
        if critic_grad_values:
            self.record_scalar(
                "critic/grad_clip_activation_rate",
                float(np.mean(np.asarray(critic_grad_values) > max_grad_norm)),
            )

        actor_metric_names = (
            "actor_loss",
            "surrogate_loss",
            "entropy",
            "entropy_bonus",
            "approx_kl",
            "clip_frac",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "grad_norm_actor",
        )
        for agent_id in self.agent_ids:
            stats = advantage_stats.get(agent_id)
            if stats is not None:
                for name, value in stats.items():
                    self.record_scalar(
                        f"advantage/agent_{agent_id}/{name}", value
                    )
            if agent_id in preupdate_ratio_errors:
                self.record_scalar(
                    f"policy/agent_{agent_id}/preupdate_max_abs_ratio_error",
                    preupdate_ratio_errors[agent_id],
                )
            self.record_scalar(
                f"policy/agent_{agent_id}/stop_epoch",
                agent_stop_epoch.get(agent_id, 0),
            )
            self.record_scalar(
                f"policy/agent_{agent_id}/samples_seen",
                ppo_metrics.get(f"actor_samples_seen_{agent_id}", 0),
            )
            for public_name in actor_metric_names:
                metric_key = f"{public_name}_{agent_id}"
                metric_values = ppo_metrics.get(metric_key, [])
                if public_name == "ratio_min" and metric_values:
                    value = float(np.min(metric_values))
                elif public_name == "ratio_max" and metric_values:
                    value = float(np.max(metric_values))
                else:
                    value = self._mean_metric(ppo_metrics, metric_key)
                if value is not None:
                    self.record_scalar(
                        f"policy/agent_{agent_id}/{public_name}", value
                    )
            grad_values = ppo_metrics.get(f"grad_norm_actor_{agent_id}", [])
            if grad_values:
                self.record_scalar(
                    f"policy/agent_{agent_id}/grad_clip_activation_rate",
                    float(np.mean(np.asarray(grad_values) > max_grad_norm)),
                )

    def reward_component_mean(
        self, component: str, *, agent_id: Optional[int] = None
    ) -> float:
        if component not in self.component_names:
            raise MetricsProtocolError(f"unknown reward component: {component}")
        if agent_id is not None:
            return self.component_stats[int(agent_id)][component].mean
        pooled = RunningStats()
        for current_agent_id in self.agent_ids:
            pooled.merge(self.component_stats[current_agent_id][component])
        return pooled.mean

    def finalize_update(self, update: int) -> Dict[str, object]:
        if self._pending_payload is not None:
            raise MetricsProtocolError("previous metrics payload was not committed")
        if self.episode_count <= 0:
            raise MetricsProtocolError("cannot finalize an update without episodes")
        if any(self.current_episode_active_counts.values()):
            raise MetricsProtocolError("cannot finalize with an unfinished episode")
        recorded_episode_count = self._episode_record_counts_by_update.get(
            int(update), 0
        )
        if recorded_episode_count != self.episode_count:
            raise MetricsProtocolError(
                f"episodes.jsonl update {update} count={recorded_episode_count}, "
                f"recorder episode_count={self.episode_count}"
            )

        scalars: Dict[str, float] = {}

        def put(tag: str, value: object) -> None:
            if tag in scalars:
                raise MetricsProtocolError(f"duplicate finalized metric tag: {tag}")
            scalars[tag] = _finite_scalar(value, tag=tag)

        put("episode/count", self.episode_count)
        put("episode/length_mean", self.episode_length_stats.mean)
        put("episode/length_std", self.episode_length_stats.std)
        put("episode/success_rate", self.success_count / self.episode_count)
        put("episode/collision_rate", self.collision_count / self.episode_count)
        put("episode/timeout_rate", self.timeout_count / self.episode_count)
        for reason, count in sorted(self.end_reason_counts.items()):
            put(f"episode/end_reason/{reason}_count", count)

        put("reward_audit/max_abs_step_residual", self.max_step_reward_residual)
        put("reward_audit/max_abs_episode_residual", self.max_episode_reward_residual)
        put("reward_audit/missing_field_count", 0.0)
        put("reward_audit/nonfinite_count", 0.0)

        disabled = DISABLED_REWARD_COMPONENTS[self.reward_version]
        sparse = SPARSE_REWARD_COMPONENTS[self.reward_version]
        for component in self.component_names:
            put(f"reward_status/{component}/enabled", float(component not in disabled))
            put(f"reward_status/{component}/sparse", float(component in sparse))
            pooled = RunningStats()
            team_episode_means = []
            for agent_id in self.agent_ids:
                step_stats = self.component_stats[agent_id][component]
                episode_stats = self.episode_component_stats[agent_id][component]
                pooled.merge(step_stats)
                team_episode_means.append(episode_stats.mean)
                prefix = f"reward/agent_{agent_id}/{component}"
                put(f"{prefix}/mean_per_active_step", step_stats.mean)
                put(f"{prefix}/sum_per_episode_mean", episode_stats.mean)
                put(f"{prefix}/nonzero_rate", step_stats.nonzero_rate)
                put(f"{prefix}/nonzero_mean", step_stats.nonzero_mean)
                put(f"{prefix}/minimum", step_stats.minimum if step_stats.count else 0.0)
                put(f"{prefix}/maximum", step_stats.maximum if step_stats.count else 0.0)
                if component in sparse:
                    put(
                        f"events/agent_{agent_id}/{component}/nonzero_count_per_update",
                        step_stats.nonzero_count,
                    )
                    put(
                        f"events/agent_{agent_id}/{component}/nonzero_rate_per_active_step",
                        step_stats.nonzero_rate,
                    )
            put(
                f"reward/pooled/{component}/mean_per_active_step", pooled.mean
            )
            put(
                f"reward/team/{component}/mean_agent_episode_sum",
                float(np.mean(team_episode_means)),
            )

        for agent_id in self.agent_ids:
            put(
                f"rollout/agent_{agent_id}/active_transition_count",
                self.active_transition_counts[agent_id],
            )
            return_stats = self.episode_return_stats[agent_id]
            for suffix, value in (
                ("mean", return_stats.mean),
                ("std", return_stats.std),
                ("minimum", return_stats.minimum if return_stats.count else 0.0),
                ("maximum", return_stats.maximum if return_stats.count else 0.0),
            ):
                put(f"episode/agent_{agent_id}/return_{suffix}", value)
            for name, stats in self.diagnostic_stats[agent_id].items():
                prefix = f"diagnostic/agent_{agent_id}/{name}"
                put(f"{prefix}/mean", stats.mean)
                put(f"{prefix}/minimum", stats.minimum if stats.count else 0.0)
                put(f"{prefix}/maximum", stats.maximum if stats.count else 0.0)
                put(f"{prefix}/nonzero_rate", stats.nonzero_rate)
            for name, stats in self.action_stats[agent_id].items():
                prefix = f"action/agent_{agent_id}/{name}"
                if name.endswith("_boundary"):
                    put(f"{prefix}_rate", stats.mean)
                else:
                    put(f"{prefix}/mean", stats.mean)
                    put(f"{prefix}/std", stats.std)
                    put(f"{prefix}/minimum", stats.minimum if stats.count else 0.0)
                    put(f"{prefix}/maximum", stats.maximum if stats.count else 0.0)

        for scenario, stats in sorted(self.scenario_stats.items()):
            put(f"scenario/{scenario}/episode_count", stats["success"].count)
            put(f"scenario/{scenario}/success_rate", stats["success"].mean)
            put(f"scenario/{scenario}/collision_rate", stats["collision"].mean)
            put(f"scenario/{scenario}/timeout_rate", stats["timeout"].mean)
            put(f"scenario/{scenario}/episode_length_mean", stats["length"].mean)

        for tag, value in self.external_scalars.items():
            put(tag, value)

        payload: Dict[str, object] = {
            "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "update": int(update),
            "created_at": _utc_now_iso(),
            "scalars": dict(sorted(scalars.items())),
        }
        self._pending_payload = payload
        return payload

    def _append_jsonl(self, payload: Mapping[str, object]) -> None:
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with open(self.jsonl_path, "a", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(line)
            file_obj.flush()
            os.fsync(file_obj.fileno())

    def _emit_payload(self, payload: Mapping[str, object]) -> None:
        normalized = self._validate_payload(payload)
        update = int(normalized["update"])
        if update != self._last_committed_update + 1:
            raise MetricsProtocolError(
                f"metrics update must be sequential: last={self._last_committed_update}, "
                f"new={update}"
            )
        for tag, value in normalized["scalars"].items():
            self.writer.add_scalar(tag, value, update)
        self.writer.flush()
        self._append_jsonl(normalized)
        self._last_committed_update = update
        self._metrics_payloads_by_update[update] = normalized

    def commit_payload(self, payload: Mapping[str, object]) -> None:
        if self._pending_payload is not payload:
            raise MetricsProtocolError(
                "only the payload finalized by this recorder can be committed"
            )
        self._emit_payload(payload)
        update = int(payload["update"])
        if (
            self.histogram_interval > 0
            and update % self.histogram_interval == 0
            and self._pending_histogram_modules is not None
        ):
            actors, critic = self._pending_histogram_modules
            for agent_id, actor in actors.items():
                for name, parameter in actor.named_parameters():
                    self.writer.add_histogram(
                        f"histogram/actor_agent_{agent_id}/{name}",
                        parameter.detach().cpu(),
                        update,
                    )
            for name, parameter in critic.named_parameters():
                self.writer.add_histogram(
                    f"histogram/critic/{name}",
                    parameter.detach().cpu(),
                    update,
                )
            self.writer.flush()
        self._pending_payload = None
        self._pending_histogram_modules = None
        self._reset_window()

    def close(self) -> None:
        if self._closed:
            return
        self.writer.flush()
        self.writer.close()
        self._closed = True
