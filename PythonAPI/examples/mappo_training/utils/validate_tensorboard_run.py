"""Validate one current-protocol MAPPO metrics run without starting CARLA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training.metrics_protocol import (
    ACTION_RATE_NAMES,
    ACTION_METRICS_PROTOCOL_VERSION,
    EPISODE_RECORD_VERSION,
    LEGACY_METRICS_PROTOCOL_VERSION,
    METRICS_PROTOCOL_VERSION,
    POLICY_ACTION_RATE_NAMES,
    MetricsProtocolError,
    validate_episode_record,
)


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
        tags.extend(
            f"action/agent_{agent_id}/{name}/mean"
            for name in ("throttle", "steer", "brake")
        )
        tags.extend(
            f"action/agent_{agent_id}/{name}_rate"
            for name in ACTION_RATE_NAMES
        )
        if schema.get("metrics_protocol_version") in {
            ACTION_METRICS_PROTOCOL_VERSION,
            METRICS_PROTOCOL_VERSION,
        }:
            tags.extend(
                f"action/agent_{agent_id}/{name}/mean"
                for name in ("policy_signed_longitudinal", "policy_steer")
            )
            tags.extend(
                f"action/agent_{agent_id}/{name}_rate"
                for name in POLICY_ACTION_RATE_NAMES
            )
    return sorted(set(tags))


def _ppo_required_tags(schema: Mapping[str, object]) -> List[str]:
    tags = [
        "critic/critic_loss",
        "critic/grad_norm_critic",
        "model/critic/update_norm",
    ]
    if schema.get("metrics_protocol_version") == METRICS_PROTOCOL_VERSION:
        tags.extend(
            [
                "critic/explained_variance",
                "critic/normalized_return_target_std",
                "critic/value_normalizer_mean",
                "critic/value_normalizer_std",
                "critic/value_normalizer_update_count",
            ]
        )
    for agent_id in [int(value) for value in schema["agent_ids"]]:
        prefix = f"policy/agent_{agent_id}"
        tags.extend(
            [
                f"{prefix}/preupdate_max_abs_ratio_error",
                f"{prefix}/actor_loss",
                f"{prefix}/entropy",
                f"{prefix}/approx_kl",
                f"{prefix}/clip_frac",
                f"{prefix}/ratio_mean",
                f"{prefix}/ratio_min",
                f"{prefix}/ratio_max",
                f"{prefix}/grad_norm_actor",
                f"model/actor_agent_{agent_id}/update_norm",
            ]
        )
    return sorted(set(tags))


def _load_jsonl(
    path: Path,
    *,
    expected_protocol_version: str,
) -> List[Dict[str, object]]:
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
        if record.get("metrics_protocol_version") != expected_protocol_version:
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


def _load_episode_jsonl(
    path: Path, *, run_id: str, agent_ids: Sequence[int]
) -> List[Dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"episode JSONL not found: {path}")
    records = []
    last_update = 0
    expected_episode_index = 0
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise MetricsProtocolError(
                f"episodes.jsonl line {line_number} is corrupt"
            ) from error
        record = validate_episode_record(
            payload,
            run_id=run_id,
            agent_ids=agent_ids,
            context=f"episodes.jsonl line {line_number}",
        )
        update = int(record["update"])
        episode_index = int(record["episode_index"])
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
        records.append(record)
    if not records:
        raise MetricsProtocolError("episodes.jsonl contains no completed episode")
    return records


def validate_metrics_run(
    run_dir: Path | str,
    *,
    expected_updates: Optional[Sequence[int]] = None,
    expected_episodes_per_update: Optional[int] = None,
    require_ppo_updates: bool = False,
    max_preupdate_ratio_error: Optional[float] = None,
) -> Dict[str, object]:
    run_dir = Path(run_dir).resolve()
    schema_path = run_dir / "metrics_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"metrics schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_protocol = str(schema.get("metrics_protocol_version", ""))
    if schema_protocol not in {
        LEGACY_METRICS_PROTOCOL_VERSION,
        ACTION_METRICS_PROTOCOL_VERSION,
        METRICS_PROTOCOL_VERSION,
    }:
        raise MetricsProtocolError("metrics schema protocol mismatch")
    records = _load_jsonl(
        run_dir / "metrics.jsonl",
        expected_protocol_version=schema_protocol,
    )
    if any(str(record.get("run_id", "")) != str(schema["run_id"]) for record in records):
        raise MetricsProtocolError("metrics.jsonl run_id differs from schema")
    actual_updates = [int(record["update"]) for record in records]
    if expected_updates is not None:
        normalized_expected_updates = [int(update) for update in expected_updates]
        if actual_updates != normalized_expected_updates:
            raise MetricsProtocolError(
                f"completed updates differ: expected={normalized_expected_updates}, "
                f"actual={actual_updates}"
            )
    if expected_episodes_per_update is not None and expected_episodes_per_update <= 0:
        raise ValueError("expected_episodes_per_update must be positive")
    if max_preupdate_ratio_error is not None and max_preupdate_ratio_error <= 0:
        raise ValueError("max_preupdate_ratio_error must be positive")

    required_tags = _required_tags(schema)
    if require_ppo_updates:
        required_tags = sorted(set(required_tags) | set(_ppo_required_tags(schema)))

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
        for agent_id in [int(value) for value in schema["agent_ids"]]:
            def action_rate(name: str) -> float:
                value = float(scalars[f"action/agent_{agent_id}/{name}_rate"])
                if not 0.0 <= value <= 1.0:
                    raise MetricsProtocolError(
                        f"update {record['update']} agent {agent_id} "
                        f"action rate {name!r} is outside [0,1]"
                    )
                return value

            for pedal in ("throttle", "brake"):
                if not math.isclose(
                    action_rate(f"{pedal}_boundary"),
                    action_rate(f"{pedal}_zero")
                    + action_rate(f"{pedal}_saturation"),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise MetricsProtocolError(
                        f"update {record['update']} agent {agent_id} "
                        f"{pedal} boundary decomposition is invalid"
                    )
            if not math.isclose(
                action_rate("steer_boundary"),
                action_rate("steer_saturation"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise MetricsProtocolError(
                    f"update {record['update']} agent {agent_id} "
                    "steer boundary decomposition is invalid"
                )
            mode_total = sum(
                action_rate(name)
                for name in (
                    "throttle_mode",
                    "brake_mode",
                    "coast_mode",
                    "longitudinal_conflict",
                )
            )
            if not math.isclose(mode_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise MetricsProtocolError(
                    f"update {record['update']} agent {agent_id} "
                    "longitudinal action modes do not conserve"
                )
            if action_rate("longitudinal_conflict") != 0.0:
                raise MetricsProtocolError(
                    f"update {record['update']} agent {agent_id} violates "
                    "mutually exclusive longitudinal control"
                )
            if schema_protocol in {
                ACTION_METRICS_PROTOCOL_VERSION,
                METRICS_PROTOCOL_VERSION,
            }:
                for name in POLICY_ACTION_RATE_NAMES:
                    action_rate(name)
        if expected_episodes_per_update is not None and int(
            scalars["episode/count"]
        ) != int(expected_episodes_per_update):
            raise MetricsProtocolError(
                f"update {record['update']} episode/count="
                f"{int(scalars['episode/count'])}, expected="
                f"{int(expected_episodes_per_update)}"
            )
        if require_ppo_updates and float(scalars["train/update_skipped"]) != 0.0:
            raise MetricsProtocolError(
                f"update {record['update']} skipped PPO and cannot pass P0-00G"
            )

    episode_schema = schema.get("episode_records")
    if not isinstance(episode_schema, Mapping):
        raise MetricsProtocolError("metrics schema has no episode_records protocol")
    if episode_schema.get("version") != EPISODE_RECORD_VERSION:
        raise MetricsProtocolError("episode record schema protocol mismatch")
    agent_ids = [int(agent_id) for agent_id in schema["agent_ids"]]
    episode_records = _load_episode_jsonl(
        run_dir / str(episode_schema.get("jsonl_file", "episodes.jsonl")),
        run_id=str(schema["run_id"]),
        agent_ids=agent_ids,
    )
    episodes_by_update: Dict[int, List[Dict[str, object]]] = {}
    for episode in episode_records:
        episodes_by_update.setdefault(int(episode["update"]), []).append(episode)
    if sorted(episodes_by_update) != actual_updates:
        raise MetricsProtocolError(
            "episode record updates differ from metrics updates; "
            f"episodes={sorted(episodes_by_update)}, metrics={actual_updates}"
        )

    for record in records:
        update = int(record["update"])
        scalars = record["scalars"]
        episodes = episodes_by_update[update]
        expected_count = int(scalars["episode/count"])
        if len(episodes) != expected_count:
            raise MetricsProtocolError(
                f"update {update} episodes.jsonl count={len(episodes)}, "
                f"metrics episode/count={expected_count}"
            )
        lengths = [int(episode["episode_length"]) for episode in episodes]
        if not math.isclose(
            float(sum(lengths) / len(lengths)),
            float(scalars["episode/length_mean"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise MetricsProtocolError(
                f"update {update} episode length aggregate differs from records"
            )
        outcome_fields = {
            "success": "episode/success_rate",
            "collision": "episode/collision_rate",
            "timeout": "episode/timeout_rate",
        }
        for field, tag in outcome_fields.items():
            actual_rate = sum(bool(episode[field]) for episode in episodes) / len(
                episodes
            )
            if not math.isclose(
                actual_rate, float(scalars[tag]), rel_tol=1e-9, abs_tol=1e-9
            ):
                raise MetricsProtocolError(
                    f"update {update} {field} aggregate differs from records"
                )
        end_reason_counts: Dict[str, int] = {}
        for episode in episodes:
            reason = str(episode["episode_end_reason"])
            end_reason_counts[reason] = end_reason_counts.get(reason, 0) + 1
        for reason, count in end_reason_counts.items():
            tag = f"episode/end_reason/{reason}_count"
            if tag not in scalars or int(scalars[tag]) != count:
                raise MetricsProtocolError(
                    f"update {update} end reason {reason!r} differs from records"
                )
        for agent_id in agent_ids:
            key = str(agent_id)
            active_total = sum(
                int(episode["agents"][key]["active_decision_steps"])
                for episode in episodes
            )
            transition_total = sum(
                int(episode["agents"][key]["transition_count"])
                for episode in episodes
            )
            if active_total != transition_total:
                raise MetricsProtocolError(
                    f"update {update} agent {agent_id} episode active/transition totals differ"
                )
            for tag in (
                f"rollout/agent_{agent_id}/active_transition_count",
                f"rollout/agent_{agent_id}/transition_count",
            ):
                if int(scalars[tag]) != active_total:
                    raise MetricsProtocolError(
                        f"update {update} {tag} differs from episode records"
                    )
            return_mean = sum(
                float(episode["agents"][key]["episode_return"])
                for episode in episodes
            ) / len(episodes)
            if not math.isclose(
                return_mean,
                float(scalars[f"episode/agent_{agent_id}/return_mean"]),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise MetricsProtocolError(
                    f"update {update} agent {agent_id} return mean differs from records"
                )

    if require_ppo_updates:
        ratio_limit = (
            1e-6
            if max_preupdate_ratio_error is None
            else float(max_preupdate_ratio_error)
        )
        for record in records:
            update = int(record["update"])
            scalars = record["scalars"]
            for agent_id in agent_ids:
                ratio_error = float(
                    scalars[
                        f"policy/agent_{agent_id}/preupdate_max_abs_ratio_error"
                    ]
                )
                if ratio_error >= ratio_limit:
                    raise MetricsProtocolError(
                        f"update {update} agent {agent_id} pre-update ratio error "
                        f"{ratio_error} >= {ratio_limit}"
                    )
                if float(scalars[f"model/actor_agent_{agent_id}/update_norm"]) <= 0:
                    raise MetricsProtocolError(
                        f"update {update} agent {agent_id} actor did not update"
                    )
            if float(scalars["model/critic/update_norm"]) <= 0:
                raise MetricsProtocolError(
                    f"update {update} critic did not update"
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
        "updates": actual_updates,
        "episode_record_version": EPISODE_RECORD_VERSION,
        "episode_record_count": len(episode_records),
        "required_tag_count": len(required_tags),
        "tensorboard_scalar_tag_count": len(tensorboard_tags),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate current MAPPO TensorBoard and metrics.jsonl artifacts"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-updates", type=int, nargs="+")
    parser.add_argument("--expected-episodes-per-update", type=int)
    parser.add_argument("--require-ppo-updates", action="store_true")
    parser.add_argument("--max-preupdate-ratio-error", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_metrics_run(
                args.run_dir,
                expected_updates=args.expected_updates,
                expected_episodes_per_update=args.expected_episodes_per_update,
                require_ppo_updates=args.require_ppo_updates,
                max_preupdate_ratio_error=args.max_preupdate_ratio_error,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
