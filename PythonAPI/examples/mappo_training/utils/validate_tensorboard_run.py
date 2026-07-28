"""Validate one current-protocol MAPPO metrics run without starting CARLA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training.metrics_protocol import METRICS_PROTOCOL_VERSION, MetricsProtocolError


def _required_tags(schema: Mapping[str, object]) -> List[str]:
    agent_ids = [int(agent_id) for agent_id in schema["agent_ids"]]
    components = list(schema["reward_components"])
    tags = [
        "episode/count",
        "episode/length_mean",
        "episode/success_rate",
        "episode/collision_rate",
        "episode/timeout_rate",
        "reward_audit/max_abs_step_residual",
        "reward_audit/max_abs_episode_residual",
        "reward_audit/missing_field_count",
        "reward_audit/nonfinite_count",
        "train/update_skipped",
        "rollout/transition_count_total",
    ]
    for component in components:
        tags.extend(
            [
                f"reward_status/{component}/enabled",
                f"reward/pooled/{component}/mean_per_active_step",
                f"reward/team/{component}/mean_agent_episode_sum",
            ]
        )
        for agent_id in agent_ids:
            prefix = f"reward/agent_{agent_id}/{component}"
            tags.extend(
                [
                    f"{prefix}/mean_per_active_step",
                    f"{prefix}/sum_per_episode_mean",
                    f"{prefix}/nonzero_rate",
                    f"{prefix}/nonzero_mean",
                ]
            )
    for agent_id in agent_ids:
        tags.extend(
            [
                f"rollout/agent_{agent_id}/active_transition_count",
                f"rollout/agent_{agent_id}/transition_count",
                f"episode/agent_{agent_id}/return_mean",
                f"optimizer/actor_agent_{agent_id}/lr",
                f"model/actor_agent_{agent_id}/parameter_norm",
            ]
        )
    return sorted(set(tags))


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"metrics JSONL not found: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise MetricsProtocolError(
                f"metrics.jsonl line {line_number} is corrupt"
            ) from error
        if record.get("metrics_protocol_version") != METRICS_PROTOCOL_VERSION:
            raise MetricsProtocolError(
                f"metrics.jsonl line {line_number} protocol mismatch"
            )
        if not isinstance(record.get("scalars"), dict):
            raise MetricsProtocolError(
                f"metrics.jsonl line {line_number} has no scalar mapping"
            )
        records.append(record)
    if not records:
        raise MetricsProtocolError("metrics.jsonl contains no completed update")
    updates = [int(record["update"]) for record in records]
    if any(current <= previous for previous, current in zip(updates, updates[1:])):
        raise MetricsProtocolError("metrics.jsonl updates are not strictly increasing")
    return records


def validate_metrics_run(run_dir: Path | str) -> Dict[str, object]:
    run_dir = Path(run_dir).resolve()
    schema_path = run_dir / "metrics_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"metrics schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("metrics_protocol_version") != METRICS_PROTOCOL_VERSION:
        raise MetricsProtocolError("metrics schema protocol mismatch")
    records = _load_jsonl(run_dir / "metrics.jsonl")
    required_tags = _required_tags(schema)

    for record in records:
        scalars = record["scalars"]
        missing = sorted(set(required_tags) - set(scalars))
        if missing:
            raise MetricsProtocolError(
                f"update {record['update']} is missing required tags: {missing}"
            )
        for tag, value in scalars.items():
            if not math.isfinite(float(value)):
                raise MetricsProtocolError(
                    f"update {record['update']} tag {tag!r} contains NaN/Inf"
                )

    tb_dir = run_dir / str(schema.get("tensorboard_directory", "tb"))
    if not list(tb_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"TensorBoard event file not found: {tb_dir}")
    accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tensorboard_tags = set(accumulator.Tags().get("scalars", []))
    missing_tb = sorted(set(required_tags) - tensorboard_tags)
    if missing_tb:
        raise MetricsProtocolError(
            f"TensorBoard is missing required scalar tags: {missing_tb}"
        )

    for tag in required_tags:
        values_by_step = {
            int(event.step): float(event.value)
            for event in accumulator.Scalars(tag)
        }
        for record in records:
            update = int(record["update"])
            if update not in values_by_step:
                raise MetricsProtocolError(
                    f"TensorBoard tag {tag!r} is missing update {update}"
                )
            expected = float(record["scalars"][tag])
            actual = values_by_step[update]
            if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6):
                raise MetricsProtocolError(
                    f"TensorBoard/JSONL mismatch for {tag!r} update={update}: "
                    f"tensorboard={actual}, jsonl={expected}"
                )

    last_scalars = records[-1]["scalars"]
    if last_scalars["reward_audit/missing_field_count"] != 0.0:
        raise MetricsProtocolError("reward metric fields were missing")
    if last_scalars["reward_audit/nonfinite_count"] != 0.0:
        raise MetricsProtocolError("reward metrics contained NaN/Inf")
    if last_scalars["reward_audit/max_abs_step_residual"] > float(
        schema["reward_audit_atol"]
    ):
        raise MetricsProtocolError("reward conservation tolerance was exceeded")

    return {
        "run_dir": str(run_dir),
        "run_id": schema["run_id"],
        "updates": [int(record["update"]) for record in records],
        "required_tag_count": len(required_tags),
        "tensorboard_scalar_tag_count": len(tensorboard_tags),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate current MAPPO TensorBoard and metrics.jsonl artifacts"
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_metrics_run(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
