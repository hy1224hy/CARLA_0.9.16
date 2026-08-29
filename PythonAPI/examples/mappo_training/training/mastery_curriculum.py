"""Capability-gated curriculum state machine for R6 and R7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, Mapping, Optional, Sequence

from training.r8a_curriculum import (
    R8A_MASTERY_PROTOCOL_VERSION,
    R8A_STAGE_NAMES,
    r8a_required_mastery_scenarios,
)


CURRICULUM_PROTOCOL_VERSION = "town05_mastery_curriculum_v2"
R7_CURRICULUM_PROTOCOL_VERSION = "town05_mastery_curriculum_road_rule_v3"


@dataclass(frozen=True)
class MasteryConfig:
    minimum_agent_mean_speed_mps: float = 0.3
    minimum_agent_mean_completion: float = 0.15
    maximum_agent_brake_mode_rate: float = 0.80
    minimum_team_success_rate: float = 0.10
    maximum_team_timeout_rate: float = 0.90
    maximum_team_collision_rate: float = 0.10
    maximum_team_offroad_rate: float = 0.10
    consecutive_passing_windows: int = 2


DEFAULT_MASTERY_CONFIG = MasteryConfig()


@dataclass(frozen=True)
class MasteryWindow:
    update: int
    episode_count: int
    agent_mean_speed_mps: Mapping[int, float]
    agent_mean_completion: Mapping[int, float]
    agent_brake_mode_rate: Mapping[int, float]
    team_success_rate: float
    team_timeout_rate: float
    team_collision_rate: float
    team_offroad_rate: float
    team_road_violation_rate: float = 0.0


class MasteryGate:
    def __init__(
        self,
        *,
        stage_count: int,
        agent_ids: Sequence[int],
        config: MasteryConfig = DEFAULT_MASTERY_CONFIG,
        maximum_team_road_violation_rate: Optional[float] = None,
    ):
        _validate_config(config)
        if isinstance(stage_count, bool) or int(stage_count) <= 0:
            raise ValueError("stage_count must be positive")
        normalized_ids = tuple(int(agent_id) for agent_id in agent_ids)
        if not normalized_ids or len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("agent_ids must be unique and non-empty")
        self.stage_count = int(stage_count)
        self.agent_ids = normalized_ids
        self.config = config
        if maximum_team_road_violation_rate is not None:
            maximum_team_road_violation_rate = float(
                maximum_team_road_violation_rate
            )
            if not math.isfinite(maximum_team_road_violation_rate) or not (
                0.0 <= maximum_team_road_violation_rate <= 1.0
            ):
                raise ValueError(
                    "maximum_team_road_violation_rate must be in [0,1]"
                )
        self.maximum_team_road_violation_rate = (
            maximum_team_road_violation_rate
        )
        self.stage_index = 0
        self.consecutive_passes = 0
        self.last_update = 0
        self.last_evidence: Dict[str, object] = {}

    def evaluate(self, window: MasteryWindow) -> Dict[str, object]:
        self._validate_window(window)
        if window.update <= self.last_update:
            raise ValueError("mastery windows must have strictly increasing updates")
        failures = _mastery_failures(
            window,
            agent_ids=self.agent_ids,
            config=self.config,
            maximum_team_road_violation_rate=(
                self.maximum_team_road_violation_rate
            ),
        )
        passed = not failures
        self.consecutive_passes = (
            min(
                self.consecutive_passes + 1,
                self.config.consecutive_passing_windows,
            )
            if passed
            else 0
        )
        previous_stage = self.stage_index
        promoted = False
        if (
            self.consecutive_passes >= self.config.consecutive_passing_windows
            and self.stage_index < self.stage_count - 1
        ):
            self.stage_index += 1
            self.consecutive_passes = 0
            promoted = True
        self.last_update = window.update
        evidence = {
            "update": window.update,
            "passed": passed,
            "failures": failures,
            "previous_stage": previous_stage,
            "stage_index": self.stage_index,
            "promoted": promoted,
            "consecutive_passes": self.consecutive_passes,
            "window": _window_as_dict(
                window,
                self.agent_ids,
                include_road_violation=(
                    self.maximum_team_road_violation_rate is not None
                ),
            ),
        }
        self.last_evidence = evidence
        return dict(evidence)

    def state_dict(self) -> Dict[str, object]:
        r7 = self.maximum_team_road_violation_rate is not None
        config = asdict(self.config)
        if r7:
            config["maximum_team_road_violation_rate"] = (
                self.maximum_team_road_violation_rate
            )
        return {
            "curriculum_protocol_version": (
                R7_CURRICULUM_PROTOCOL_VERSION if r7
                else CURRICULUM_PROTOCOL_VERSION
            ),
            "stage_count": self.stage_count,
            "agent_ids": list(self.agent_ids),
            "config": config,
            "stage_index": self.stage_index,
            "consecutive_passes": self.consecutive_passes,
            "last_update": self.last_update,
            "last_evidence": dict(self.last_evidence),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = self.state_dict()
        for name in (
            "curriculum_protocol_version",
            "stage_count",
            "agent_ids",
            "config",
        ):
            if state.get(name) != expected[name]:
                raise ValueError(f"mastery state {name} mismatch")
        stage_index = int(state.get("stage_index", -1))
        consecutive = int(state.get("consecutive_passes", -1))
        last_update = int(state.get("last_update", -1))
        if not 0 <= stage_index < self.stage_count:
            raise ValueError("saved mastery stage is invalid")
        if not 0 <= consecutive <= self.config.consecutive_passing_windows:
            raise ValueError("saved mastery pass count is invalid")
        if last_update < 0:
            raise ValueError("saved mastery update is invalid")
        self.stage_index = stage_index
        self.consecutive_passes = consecutive
        self.last_update = last_update
        self.last_evidence = dict(state.get("last_evidence", {}))

    def _validate_window(self, window: MasteryWindow) -> None:
        _validate_mastery_window(window, self.agent_ids)


class R8AMasteryGate:
    """R7 thresholds plus independent evidence for the short mirror pair."""

    def __init__(
        self,
        *,
        stage_count: int,
        agent_ids: Sequence[int],
        config: MasteryConfig = DEFAULT_MASTERY_CONFIG,
        maximum_team_road_violation_rate: float = 0.10,
    ):
        if int(stage_count) != len(R8A_STAGE_NAMES):
            raise ValueError(
                "R8-A mastery stage_count must match the frozen four stages"
            )
        validator = MasteryGate(
            stage_count=stage_count,
            agent_ids=agent_ids,
            config=config,
            maximum_team_road_violation_rate=maximum_team_road_violation_rate,
        )
        self.stage_count = validator.stage_count
        self.agent_ids = validator.agent_ids
        self.config = validator.config
        self.maximum_team_road_violation_rate = (
            validator.maximum_team_road_violation_rate
        )
        self.stage_index = 0
        self.consecutive_passes = 0
        self.last_update = 0
        self.last_evidence: Dict[str, object] = {}

    def evaluate(
        self,
        window: MasteryWindow,
        *,
        scenario_windows: Mapping[str, MasteryWindow],
    ) -> Dict[str, object]:
        _validate_mastery_window(window, self.agent_ids)
        if window.update <= self.last_update:
            raise ValueError("mastery windows must have strictly increasing updates")
        if not isinstance(scenario_windows, Mapping):
            raise ValueError("R8-A scenario_windows must be a mapping")

        failures = [
            f"pooled.{failure}"
            for failure in _mastery_failures(
                window,
                agent_ids=self.agent_ids,
                config=self.config,
                maximum_team_road_violation_rate=(
                    self.maximum_team_road_violation_rate
                ),
            )
        ]
        required_scenarios = r8a_required_mastery_scenarios(self.stage_index)
        normalized_scenario_windows: Dict[str, Dict[str, object]] = {}
        for scenario_name, scenario_window in scenario_windows.items():
            _validate_mastery_window(scenario_window, self.agent_ids)
            if scenario_window.update != window.update:
                raise ValueError(
                    "R8-A scenario mastery update differs from pooled update"
                )
            normalized_scenario_windows[str(scenario_name)] = _window_as_dict(
                scenario_window,
                self.agent_ids,
                include_road_violation=True,
            )
        for scenario_name in required_scenarios:
            scenario_window = scenario_windows.get(scenario_name)
            if scenario_window is None:
                failures.append(f"scenario.{scenario_name}.missing")
                continue
            failures.extend(
                f"scenario.{scenario_name}.{failure}"
                for failure in _mastery_failures(
                    scenario_window,
                    agent_ids=self.agent_ids,
                    config=self.config,
                    maximum_team_road_violation_rate=(
                        self.maximum_team_road_violation_rate
                    ),
                )
            )

        passed = not failures
        self.consecutive_passes = (
            min(
                self.consecutive_passes + 1,
                self.config.consecutive_passing_windows,
            )
            if passed
            else 0
        )
        previous_stage = self.stage_index
        promoted = False
        if (
            self.consecutive_passes >= self.config.consecutive_passing_windows
            and self.stage_index < self.stage_count - 1
        ):
            self.stage_index += 1
            self.consecutive_passes = 0
            promoted = True
        self.last_update = window.update
        evidence = {
            "update": window.update,
            "protocol_version": R8A_MASTERY_PROTOCOL_VERSION,
            "passed": passed,
            "failures": failures,
            "required_scenarios": list(required_scenarios),
            "previous_stage": previous_stage,
            "stage_index": self.stage_index,
            "promoted": promoted,
            "consecutive_passes": self.consecutive_passes,
            "window": _window_as_dict(
                window, self.agent_ids, include_road_violation=True
            ),
            "scenario_windows": normalized_scenario_windows,
        }
        self.last_evidence = evidence
        return dict(evidence)

    def state_dict(self) -> Dict[str, object]:
        config = asdict(self.config)
        config["maximum_team_road_violation_rate"] = (
            self.maximum_team_road_violation_rate
        )
        return {
            "curriculum_protocol_version": R8A_MASTERY_PROTOCOL_VERSION,
            "stage_count": self.stage_count,
            "agent_ids": list(self.agent_ids),
            "config": config,
            "stage_names": list(R8A_STAGE_NAMES),
            "stage_index": self.stage_index,
            "consecutive_passes": self.consecutive_passes,
            "last_update": self.last_update,
            "last_evidence": dict(self.last_evidence),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = self.state_dict()
        for name in (
            "curriculum_protocol_version",
            "stage_count",
            "agent_ids",
            "config",
            "stage_names",
        ):
            if state.get(name) != expected[name]:
                raise ValueError(f"R8-A mastery state {name} mismatch")
        stage_index = int(state.get("stage_index", -1))
        consecutive = int(state.get("consecutive_passes", -1))
        last_update = int(state.get("last_update", -1))
        if not 0 <= stage_index < self.stage_count:
            raise ValueError("saved R8-A mastery stage is invalid")
        if not 0 <= consecutive <= self.config.consecutive_passing_windows:
            raise ValueError("saved R8-A mastery pass count is invalid")
        if last_update < 0:
            raise ValueError("saved R8-A mastery update is invalid")
        self.stage_index = stage_index
        self.consecutive_passes = consecutive
        self.last_update = last_update
        self.last_evidence = dict(state.get("last_evidence", {}))


def _validate_mastery_window(
    window: MasteryWindow, agent_ids: Sequence[int]
) -> None:
    if not isinstance(window, MasteryWindow):
        raise ValueError("window must be MasteryWindow")
    if window.update <= 0 or window.episode_count <= 0:
        raise ValueError("mastery update and episode_count must be positive")
    expected = {int(agent_id) for agent_id in agent_ids}
    mappings = (
        window.agent_mean_speed_mps,
        window.agent_mean_completion,
        window.agent_brake_mode_rate,
    )
    if any(set(mapping) != expected for mapping in mappings):
        raise ValueError("mastery window must contain every registered agent")
    for mapping in mappings:
        for value in mapping.values():
            if not math.isfinite(float(value)):
                raise ValueError("mastery window contains NaN/Inf")
    for value in (
        window.team_success_rate,
        window.team_timeout_rate,
        window.team_collision_rate,
        window.team_offroad_rate,
        window.team_road_violation_rate,
        *window.agent_brake_mode_rate.values(),
        *window.agent_mean_completion.values(),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("mastery rates/completions must be in [0,1]")
    if any(float(value) < 0.0 for value in window.agent_mean_speed_mps.values()):
        raise ValueError("mastery speeds must be non-negative")


def _mastery_failures(
    window: MasteryWindow,
    *,
    agent_ids: Sequence[int],
    config: MasteryConfig,
    maximum_team_road_violation_rate: Optional[float],
) -> list[str]:
    failures: list[str] = []
    for agent_id in agent_ids:
        if (
            window.agent_mean_speed_mps[agent_id]
            < config.minimum_agent_mean_speed_mps
        ):
            failures.append(f"agent{agent_id}.mean_speed")
        if (
            window.agent_mean_completion[agent_id]
            < config.minimum_agent_mean_completion
        ):
            failures.append(f"agent{agent_id}.mean_completion")
        if (
            window.agent_brake_mode_rate[agent_id]
            > config.maximum_agent_brake_mode_rate
        ):
            failures.append(f"agent{agent_id}.brake_mode")
    team_checks = [
        ("team.success_rate", window.team_success_rate >= config.minimum_team_success_rate),
        ("team.timeout_rate", window.team_timeout_rate <= config.maximum_team_timeout_rate),
        ("team.collision_rate", window.team_collision_rate <= config.maximum_team_collision_rate),
        ("team.offroad_rate", window.team_offroad_rate <= config.maximum_team_offroad_rate),
    ]
    if maximum_team_road_violation_rate is not None:
        team_checks.append((
            "team.road_violation_rate",
            window.team_road_violation_rate
            <= maximum_team_road_violation_rate,
        ))
    failures.extend(name for name, passed in team_checks if not passed)
    return failures


def mastery_protocol_spec(
    config: MasteryConfig = DEFAULT_MASTERY_CONFIG,
    *,
    maximum_team_road_violation_rate: Optional[float] = None,
) -> Dict[str, object]:
    _validate_config(config)
    if maximum_team_road_violation_rate is not None:
        maximum_team_road_violation_rate = float(
            maximum_team_road_violation_rate
        )
        if not math.isfinite(maximum_team_road_violation_rate) or not (
            0.0 <= maximum_team_road_violation_rate <= 1.0
        ):
            raise ValueError("maximum road-violation rate must be in [0,1]")
    parameters = asdict(config)
    if maximum_team_road_violation_rate is not None:
        parameters["maximum_team_road_violation_rate"] = (
            maximum_team_road_violation_rate
        )
    return {
        "curriculum_protocol_version": (
            R7_CURRICULUM_PROTOCOL_VERSION
            if maximum_team_road_violation_rate is not None
            else CURRICULUM_PROTOCOL_VERSION
        ),
        "parameters": parameters,
        "promotion_scope": "all_agents_and_team_outcomes",
        "promotion_rule": "consecutive_frozen_windows",
        "automatic_update_schedule": False,
    }


def r8a_mastery_protocol_spec(
    config: MasteryConfig = DEFAULT_MASTERY_CONFIG,
    *,
    maximum_team_road_violation_rate: float = 0.10,
) -> Dict[str, object]:
    base = mastery_protocol_spec(
        config,
        maximum_team_road_violation_rate=maximum_team_road_violation_rate,
    )
    return {
        **base,
        "curriculum_protocol_version": R8A_MASTERY_PROTOCOL_VERSION,
        "promotion_scope": (
            "all_agents_team_outcomes_and_each_current_short_mirror"
        ),
        "promotion_rule": "consecutive_frozen_stratified_windows",
        "stratified_stage_names": list(R8A_STAGE_NAMES[:2]),
    }


def _window_as_dict(
    window: MasteryWindow,
    agent_ids: Sequence[int],
    *,
    include_road_violation: bool = False,
) -> Dict[str, object]:
    result = {
        "update": window.update,
        "episode_count": window.episode_count,
        "agent_mean_speed_mps": {
            str(agent_id): float(window.agent_mean_speed_mps[agent_id])
            for agent_id in agent_ids
        },
        "agent_mean_completion": {
            str(agent_id): float(window.agent_mean_completion[agent_id])
            for agent_id in agent_ids
        },
        "agent_brake_mode_rate": {
            str(agent_id): float(window.agent_brake_mode_rate[agent_id])
            for agent_id in agent_ids
        },
        "team_success_rate": float(window.team_success_rate),
        "team_timeout_rate": float(window.team_timeout_rate),
        "team_collision_rate": float(window.team_collision_rate),
        "team_offroad_rate": float(window.team_offroad_rate),
    }
    if include_road_violation:
        result["team_road_violation_rate"] = float(
            window.team_road_violation_rate
        )
    return result


def _validate_config(config: MasteryConfig) -> None:
    if not isinstance(config, MasteryConfig):
        raise ValueError("config must be MasteryConfig")
    for name, value in asdict(config).items():
        if name == "consecutive_passing_windows":
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError("consecutive windows must be positive")
            continue
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid mastery threshold: {name}")


_validate_config(DEFAULT_MASTERY_CONFIG)
