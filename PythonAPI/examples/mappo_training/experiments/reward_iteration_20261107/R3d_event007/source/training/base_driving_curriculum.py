"""Cross-update training curriculum for the versioned symmetric base pool.

This module is deliberately independent of CARLA, reward and PPO code.  Its
evidence is on-policy training data, so an unlock is never a frozen driving
qualification.  Windows are disjoint: the same easy episodes cannot provide
multiple consecutive passes.  Unlocked scenarios remain available for review.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence


BASE_DRIVING_CURRICULUM_PROTOCOL_VERSION = "town05_base_driving_cross_update_v2"


@dataclass(frozen=True)
class BaseDrivingCurriculumConfig:
    minimum_window_updates: int = 20
    maximum_window_updates: int = 80
    minimum_window_episodes: int = 320
    minimum_stratum_episodes: int = 20
    consecutive_passing_windows: int = 2
    consecutive_failing_windows_for_review: int = 2
    minimum_agent_mean_speed_mps: float = 1.0
    minimum_agent_mean_completion: float = 0.50
    maximum_agent_brake_mode_rate: float = 0.80
    minimum_team_success_rate: float = 0.60
    maximum_team_timeout_rate: float = 0.20
    maximum_team_collision_rate: float = 0.10
    maximum_team_offroad_rate: float = 0.10
    maximum_team_road_violation_rate: float = 0.10


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: object, name: str, *, ratio: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (ratio and result > 1):
        raise ValueError(f"{name} must be {'in [0,1]' if ratio else 'finite and >= 0'}")
    return result


def validate_base_driving_curriculum_config(config: BaseDrivingCurriculumConfig) -> None:
    """Keep physical m/s thresholds separate from dimensionless proportions."""
    if not isinstance(config, BaseDrivingCurriculumConfig):
        raise ValueError("config must be BaseDrivingCurriculumConfig")
    for name, value in asdict(config).items():
        if name in (
            "minimum_window_updates", "maximum_window_updates",
            "minimum_window_episodes", "minimum_stratum_episodes",
            "consecutive_passing_windows", "consecutive_failing_windows_for_review",
        ):
            _integer(value, name, 1)
        else:
            _number(value, name, ratio=not name.endswith("_mps"))
    if config.maximum_window_updates < config.minimum_window_updates:
        raise ValueError("maximum_window_updates must cover minimum_window_updates")


def _agent_value(values: Mapping, aid: int, name: str):
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be an agent mapping")
    matches = [values[key] for key in (aid, str(aid), f"agent{aid}") if key in values]
    if len(matches) != 1:
        raise ValueError(f"{name} must contain agent {aid} exactly once")
    return matches[0]


def _label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _key(metadata: Mapping, aid: int) -> str:
    # JSON tuple encoding avoids delimiter collisions in user supplied labels.
    return json.dumps([
        metadata["task_family"], metadata["direction"],
        metadata["startup_mode"], aid, metadata["agent_roles"][str(aid)],
    ], separators=(",", ":"))


def _empty_bucket() -> Dict[str, object]:
    return {
        "episode_count": 0, "active_steps": 0, "speed_sum_mps": 0.0,
        "completion_sum": 0.0, "brake_sum": 0.0, "brake_observation_steps": 0,
        "success": 0, "timeout": 0, "collision": 0,
        "offroad": 0, "road_violation": 0,
    }


class BaseDrivingCurriculum:
    """Accumulate complete updates and gate cumulative scenario support.

    ``scenario_metadata`` maps scenario filenames to metadata with
    ``curriculum_stage``, ``task_family``, ``direction``, ``startup_mode`` and
    ``agent_roles``.  Stages must start at zero and be contiguous.  The stage
    metadata is an immutable protocol identity, included in every saved state.

    ``observe_update`` consumes complete episodes with ``scenario_id`` (or
    ``scenario_name``), team outcome booleans and ``agent_metrics``.  Each agent
    metric requires ``mean_speed_mps``, ``completion`` and ``brake_mode_rate``;
    optional ``active_decision_steps`` gives proper step-weighted aggregation.
    ``brake_observation_steps`` separately counts actual control readbacks. It
    defaults to active steps for adapters with complete readback. Missing actual
    controls are never filled with the policy's sent command or a zero sample.
    Pass an entire update once, after rollout completion.  The caller includes
    this state in the same atomic checkpoint as its actors/optimizers/RNG.
    """

    def __init__(
        self,
        scenario_metadata: Mapping[str, Mapping],
        *,
        agent_ids: Sequence[int] = (0, 1),
        config: BaseDrivingCurriculumConfig = BaseDrivingCurriculumConfig(),
        source_protocol_version: str = "base_driving_vnext_v1",
        completed_update: int = 0,
        stage_names: Optional[Sequence[str]] = None,
    ):
        validate_base_driving_curriculum_config(config)
        self.config = config
        self.agent_ids = tuple(_integer(aid, "agent_id") for aid in agent_ids)
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must be non-empty and unique")
        self.source_protocol_version = _label(source_protocol_version, "source_protocol_version")
        self.origin_completed_update = _integer(completed_update, "completed_update")
        if not isinstance(scenario_metadata, Mapping) or not scenario_metadata:
            raise ValueError("scenario_metadata must be a non-empty mapping")
        self.scenario_metadata = {}
        for scenario, raw in sorted(scenario_metadata.items()):
            _label(scenario, "scenario_id")
            if not isinstance(raw, Mapping):
                raise ValueError(f"metadata for {scenario} must be a mapping")
            self.scenario_metadata[scenario] = {
                "curriculum_stage": _integer(raw.get("curriculum_stage"), "curriculum_stage"),
                **{name: _label(raw.get(name), name) for name in (
                    "task_family", "direction", "startup_mode",
                )},
                "agent_roles": {
                    str(aid): _label(_agent_value(raw.get("agent_roles"), aid, "agent_roles"), "role")
                    for aid in self.agent_ids
                },
            }
        stages = sorted({row["curriculum_stage"] for row in self.scenario_metadata.values()})
        if stages != list(range(len(stages))):
            raise ValueError("curriculum stages must be contiguous and start at zero")
        self.stage_count = len(stages)
        self.stage_names = tuple(stage_names or [f"B{index}" for index in stages])
        if len(self.stage_names) != self.stage_count or len(set(self.stage_names)) != self.stage_count:
            raise ValueError("stage_names must uniquely name each stage")
        for name in self.stage_names:
            _label(name, "stage_name")
        canonical = json.dumps(self.scenario_metadata, sort_keys=True, separators=(",", ":"))
        self.scenario_metadata_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.stage_index = 0
        self.last_update = self.origin_completed_update
        self.consecutive_passes = 0
        self.consecutive_failures = 0
        self.review_required = False
        self.closed_window_count = 0
        self.last_evidence: Dict[str, object] = {}
        self._clear_window()

    def _clear_window(self) -> None:
        self._window_updates = []
        self._window_episode_count = 0
        self._scenario_exposures = {}
        self._strata = {}

    def available_scenarios(self) -> list[str]:
        """Cumulative support: passing a stage never removes earlier tasks."""
        return [scenario for scenario, metadata in self.scenario_metadata.items()
                if metadata["curriculum_stage"] <= self.stage_index]

    def _required_strata(self) -> set[str]:
        return {_key(self.scenario_metadata[name], aid)
                for name in self.available_scenarios() for aid in self.agent_ids}

    def _normalize_episode(self, episode: Mapping) -> Dict[str, object]:
        if not isinstance(episode, Mapping):
            raise ValueError("episode must be a mapping")
        scenario = episode.get("scenario_id", episode.get("scenario_name"))
        if scenario not in self.available_scenarios():
            raise ValueError(f"episode scenario is unknown or still locked: {scenario!r}")
        outcomes = {}
        for name in ("success", "timeout", "collision", "offroad", "road_violation"):
            value = episode.get(name)
            if not isinstance(value, bool):
                raise ValueError(f"episode {name} must be a boolean")
            outcomes[name] = value
        if outcomes["success"] and any(value for name, value in outcomes.items() if name != "success"):
            raise ValueError("successful episodes cannot have failure outcomes")
        agent_metrics = {}
        for aid in self.agent_ids:
            raw = _agent_value(episode.get("agent_metrics"), aid, "agent_metrics")
            if not isinstance(raw, Mapping):
                raise ValueError("agent metrics must be mappings")
            active_steps = _integer(raw.get("active_decision_steps", 1), "active_decision_steps", 1)
            brake_steps = _integer(raw.get("brake_observation_steps", active_steps), "brake_observation_steps")
            if brake_steps > active_steps:
                raise ValueError("brake_observation_steps cannot exceed active_decision_steps")
            agent_metrics[str(aid)] = {
                "mean_speed_mps": _number(raw.get("mean_speed_mps"), "mean_speed_mps"),
                "completion": _number(raw.get("completion"), "completion", ratio=True),
                "brake_mode_rate": _number(raw.get("brake_mode_rate"), "brake_mode_rate", ratio=True),
                "active_decision_steps": active_steps,
                "brake_observation_steps": brake_steps,
            }
        return {"scenario_id": scenario, "agent_metrics": agent_metrics, **outcomes}

    def observe_update(self, update: int, episodes: Sequence[Mapping]) -> Dict[str, object]:
        update = _integer(update, "update", 1)
        if update != self.last_update + 1:
            raise ValueError("complete updates must be contiguous and recorded once")
        if not episodes:
            raise ValueError("a complete update must contain episodes")
        # Validate the whole batch before mutating pending resume state.
        normalized = [self._normalize_episode(episode) for episode in episodes]
        self._window_updates.append(update)
        self._window_episode_count += len(normalized)
        self.last_update = update
        for episode in normalized:
            name = episode["scenario_id"]
            self._scenario_exposures[name] = self._scenario_exposures.get(name, 0) + 1
            for aid in self.agent_ids:
                key = _key(self.scenario_metadata[name], aid)
                bucket = self._strata.setdefault(key, _empty_bucket())
                metrics = episode["agent_metrics"][str(aid)]
                steps = metrics["active_decision_steps"]
                bucket["episode_count"] += 1
                bucket["active_steps"] += steps
                bucket["speed_sum_mps"] += metrics["mean_speed_mps"] * steps
                bucket["brake_observation_steps"] += metrics["brake_observation_steps"]
                bucket["brake_sum"] += metrics["brake_mode_rate"] * metrics["brake_observation_steps"]
                bucket["completion_sum"] += metrics["completion"]
                for outcome in ("success", "timeout", "collision", "offroad", "road_violation"):
                    bucket[outcome] += int(episode[outcome])
        return self._evaluate_pending()

    def _evaluate_pending(self) -> Dict[str, object]:
        required = self._required_strata()
        counts = {key: int(self._strata.get(key, {}).get("episode_count", 0)) for key in sorted(required)}
        missing = {key: count for key, count in counts.items()
                   if count < self.config.minimum_stratum_episodes}
        sufficient = (len(self._window_updates) >= self.config.minimum_window_updates
                      and self._window_episode_count >= self.config.minimum_window_episodes
                      and not missing)
        expired = len(self._window_updates) >= self.config.maximum_window_updates
        evidence = {
            "curriculum_protocol_version": BASE_DRIVING_CURRICULUM_PROTOCOL_VERSION,
            "evidence_source": "on_policy_training_not_frozen_qualification",
            "update": self.last_update, "previous_stage": self.stage_index,
            "stage_index": self.stage_index, "stage_name": self.stage_names[self.stage_index],
            "window_start_update": self._window_updates[0],
            "window_update_count": len(self._window_updates),
            "window_episode_count": self._window_episode_count,
            "window_closed": bool(sufficient or expired),
            "coverage_sufficient": sufficient, "missing_exposure": missing,
            "scenario_exposures": dict(self._scenario_exposures),
            "stratum_episode_counts": counts, "passed": False, "promoted": False,
            "failures": [], "strata": {},
        }
        if sufficient or expired:
            failures = [] if sufficient else ["coverage.insufficient_before_window_expiry"]
            summaries = {}
            if sufficient:
                for key in sorted(required):
                    bucket = self._strata[key]
                    count, steps = bucket["episode_count"], bucket["active_steps"]
                    brake_steps = bucket["brake_observation_steps"]
                    brake_rate = bucket["brake_sum"] / brake_steps if brake_steps else None
                    summary = {
                        "episode_count": count,
                        "mean_speed_mps": bucket["speed_sum_mps"] / steps,
                        "mean_completion": bucket["completion_sum"] / count,
                        "brake_mode_rate": brake_rate,
                        "brake_observation_steps": brake_steps,
                        "missing_control_steps": steps - brake_steps,
                        **{f"{outcome}_rate": bucket[outcome] / count for outcome in (
                            "success", "timeout", "collision", "offroad", "road_violation",
                        )},
                    }
                    summaries[key] = summary
                    checks = {
                        "mean_speed_mps": summary["mean_speed_mps"] >= self.config.minimum_agent_mean_speed_mps,
                        "mean_completion": summary["mean_completion"] >= self.config.minimum_agent_mean_completion,
                        "success_rate": summary["success_rate"] >= self.config.minimum_team_success_rate,
                        "timeout_rate": summary["timeout_rate"] <= self.config.maximum_team_timeout_rate,
                        "collision_rate": summary["collision_rate"] <= self.config.maximum_team_collision_rate,
                        "offroad_rate": summary["offroad_rate"] <= self.config.maximum_team_offroad_rate,
                        "road_violation_rate": summary["road_violation_rate"] <= self.config.maximum_team_road_violation_rate,
                    }
                    if brake_steps:
                        checks["brake_mode_rate"] = brake_rate <= self.config.maximum_agent_brake_mode_rate
                    else:
                        checks["missing_control"] = False
                    failures.extend(f"{key}.{name}" for name, passed in checks.items() if not passed)
            passed = not failures
            self.consecutive_passes = min(self.consecutive_passes + 1, self.config.consecutive_passing_windows) if passed else 0
            self.consecutive_failures = 0 if passed else min(
                self.consecutive_failures + 1, self.config.consecutive_failing_windows_for_review,
            )
            if self.consecutive_failures >= self.config.consecutive_failing_windows_for_review:
                self.review_required = True
            promoted = False
            if self.consecutive_passes >= self.config.consecutive_passing_windows:
                self.review_required = False
                if self.stage_index < self.stage_count - 1:
                    self.stage_index += 1
                    self.consecutive_passes = 0
                    promoted = True
            self.closed_window_count += 1
            evidence.update({
                "passed": passed, "promoted": promoted, "failures": failures,
                "strata": summaries, "stage_index": self.stage_index,
                "stage_name": self.stage_names[self.stage_index],
            })
            self._clear_window()
        evidence.update({
            "consecutive_passes": self.consecutive_passes,
            "consecutive_failures": self.consecutive_failures,
            "review_required": self.review_required,
            "closed_window_count": self.closed_window_count,
        })
        self.last_evidence = deepcopy(evidence)
        return deepcopy(evidence)

    def protocol_spec(self) -> Dict[str, object]:
        return {
            "curriculum_protocol_version": BASE_DRIVING_CURRICULUM_PROTOCOL_VERSION,
            "source_protocol_version": self.source_protocol_version,
            "purpose": "training_unlock_only_not_frozen_driving_qualification",
            "config": asdict(self.config), "agent_ids": list(self.agent_ids),
            "stage_names": list(self.stage_names),
            "scenario_metadata_sha256": self.scenario_metadata_sha256,
            "scenario_metadata": deepcopy(self.scenario_metadata),
            "window_rule": "disjoint_cross_update_strata_with_minimum_exposure",
            "retention_rule": "all_previously_unlocked_scenarios_remain_available",
            "failure_hysteresis": "review_flag_without_relocking_or_automatic_stop",
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            **self.protocol_spec(), "origin_completed_update": self.origin_completed_update,
            "stage_index": self.stage_index, "last_update": self.last_update,
            "consecutive_passes": self.consecutive_passes,
            "consecutive_failures": self.consecutive_failures,
            "review_required": self.review_required,
            "closed_window_count": self.closed_window_count,
            "pending_window_updates": list(self._window_updates),
            "pending_window_episode_count": self._window_episode_count,
            "pending_scenario_exposures": dict(self._scenario_exposures),
            "pending_strata": deepcopy(self._strata),
            "last_evidence": deepcopy(self.last_evidence),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Reject incompatible source protocols and restore every pending sum."""
        if not isinstance(state, Mapping):
            raise ValueError("saved curriculum must be a mapping")
        for name, expected in self.protocol_spec().items():
            if state.get(name) != expected:
                raise ValueError(f"base driving curriculum state {name} mismatch")
        if state.get("origin_completed_update") != self.origin_completed_update:
            raise ValueError("curriculum origin_completed_update mismatch")
        stage = _integer(state.get("stage_index"), "stage_index")
        if stage >= self.stage_count:
            raise ValueError("saved stage_index is invalid")
        last_update = _integer(state.get("last_update"), "last_update", self.origin_completed_update)
        passes = _integer(state.get("consecutive_passes"), "consecutive_passes")
        failures = _integer(state.get("consecutive_failures"), "consecutive_failures")
        if passes > self.config.consecutive_passing_windows or failures > self.config.consecutive_failing_windows_for_review:
            raise ValueError("saved curriculum streak exceeds configured cap")
        if passes and failures:
            raise ValueError("saved curriculum cannot have simultaneous pass and failure streaks")
        review = state.get("review_required")
        if not isinstance(review, bool):
            raise ValueError("review_required must be boolean")
        closed = _integer(state.get("closed_window_count"), "closed_window_count")
        updates = state.get("pending_window_updates")
        if not isinstance(updates, list):
            raise ValueError("pending_window_updates must be a list")
        updates = [_integer(value, "pending update", self.origin_completed_update + 1) for value in updates]
        if updates and (updates != list(range(updates[0], last_update + 1)) or len(updates) >= self.config.maximum_window_updates):
            raise ValueError("pending window updates must be a contiguous unfinished suffix")
        episode_count = _integer(state.get("pending_window_episode_count"), "pending_window_episode_count")
        exposures = state.get("pending_scenario_exposures")
        strata = state.get("pending_strata")
        if not isinstance(exposures, Mapping) or not isinstance(strata, Mapping):
            raise ValueError("pending exposures and strata must be mappings")
        available = {name for name, row in self.scenario_metadata.items() if row["curriculum_stage"] <= stage}
        exposures = {name: _integer(count, "scenario exposure", 1) for name, count in exposures.items()}
        if set(exposures) - available or sum(exposures.values()) != episode_count:
            raise ValueError("pending scenario exposures do not match episode count or stage")
        if bool(updates) != bool(episode_count):
            raise ValueError("pending episode count and window updates disagree")
        expected_counts = {}
        for name, count in exposures.items():
            for aid in self.agent_ids:
                key = _key(self.scenario_metadata[name], aid)
                expected_counts[key] = expected_counts.get(key, 0) + count
        if set(strata) != set(expected_counts):
            raise ValueError("pending strata do not match exposed scenarios")
        normalized = {}
        for key, raw in strata.items():
            if not isinstance(raw, Mapping) or set(raw) != set(_empty_bucket()):
                raise ValueError("invalid pending stratum schema")
            bucket = {name: (_number(value, name) if name in (
                "speed_sum_mps", "completion_sum", "brake_sum",
            ) else _integer(value, name)) for name, value in raw.items()}
            if bucket["episode_count"] != expected_counts[key] or bucket["active_steps"] < bucket["episode_count"]:
                raise ValueError("pending stratum counts are inconsistent")
            if bucket["brake_observation_steps"] > bucket["active_steps"]:
                raise ValueError("pending brake observation steps exceed active steps")
            if bucket["completion_sum"] > bucket["episode_count"] or bucket["brake_sum"] > bucket["brake_observation_steps"]:
                raise ValueError("pending proportion sums exceed their denominator")
            if any(bucket[name] > bucket["episode_count"] for name in (
                "success", "timeout", "collision", "offroad", "road_violation",
            )):
                raise ValueError("pending outcome counts exceed episode count")
            normalized[key] = bucket
        last_evidence = state.get("last_evidence")
        if not isinstance(last_evidence, Mapping):
            raise ValueError("last_evidence must be a mapping")
        self.stage_index, self.last_update = stage, last_update
        self.consecutive_passes, self.consecutive_failures = passes, failures
        self.review_required, self.closed_window_count = review, closed
        self._window_updates, self._window_episode_count = updates, episode_count
        self._scenario_exposures, self._strata = dict(exposures), normalized
        self.last_evidence = deepcopy(dict(last_evidence))
