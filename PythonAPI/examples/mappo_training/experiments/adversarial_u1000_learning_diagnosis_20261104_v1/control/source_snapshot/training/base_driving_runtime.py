"""Small explicit integration adapter for the symmetric base-driving protocol.

Importing this module performs no CARLA calls.  The original trainer selects
this adapter only when the new contract is explicitly requested; legacy
CurriculumLearning and frozen R8-A behavior remain independent.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from training.base_driving_curriculum import (
    BaseDrivingCurriculum,
    BaseDrivingCurriculumConfig,
    _integer,
)
from training.base_driving_scenarios_v1 import (
    BASE_DRIVING_STAGE_NAMES,
    base_driving_sampler_state,
    load_base_driving_manifest,
    sample_base_driving_balanced,
    validate_base_driving_sampler_state,
)


BASE_DRIVING_RUNTIME_PROTOCOL_VERSION = "town05_base_driving_runtime_v1"


class BaseDrivingCurriculumAdapter:
    """Trainer-facing facade with explicit gate/sampler checkpoint state.

    ``record_scenario_result`` preserves the trainer's history counters only;
    ``observe_update`` separately records one complete rollout in the gate.
    The two calls intentionally do not double-count each other's evidence.
    """

    def __init__(
        self,
        scenario_dir: str | Path,
        *,
        agent_ids: Sequence[int] = (0, 1),
        config: BaseDrivingCurriculumConfig = BaseDrivingCurriculumConfig(),
        source_protocol_version: str = "base_driving_vnext_v1",
        completed_update: int = 0,
    ):
        root = Path(scenario_dir)
        self.manifest = load_base_driving_manifest(root)
        if self.manifest.get("split") != "train":
            raise ValueError("base-driving runtime requires the training split")
        self.scenario_dir = str(root)
        self.scenario_metadata = {
            name: json.loads((root / name).read_text(encoding="utf-8"))
            for name in sorted(self.manifest["scenario_sha256"])
        }
        canonical_manifest = json.dumps(self.manifest, sort_keys=True, separators=(",", ":"))
        self.pool_manifest_sha256 = hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()
        self._gate_kwargs = {
            "agent_ids": tuple(agent_ids), "config": config,
            "source_protocol_version": source_protocol_version,
            "completed_update": completed_update,
            "stage_names": BASE_DRIVING_STAGE_NAMES,
        }
        self.base_gate = BaseDrivingCurriculum(self.scenario_metadata, **self._gate_kwargs)
        self.curriculum_stages = [
            [name for name, metadata in self.scenario_metadata.items()
             if metadata["curriculum_stage"] <= stage]
            for stage in range(self.base_gate.stage_count)
        ]
        # Explicitly bypass the old per-update gate; no implicit legacy fallback.
        self.mastery_gate = None
        self.mastery_enabled = False
        self.r8a_enabled = False
        self.mode = "base_driving_v1"
        self.scenario_families = {
            name: metadata["task_family"] for name, metadata in self.scenario_metadata.items()
        }
        self.scenario_success_history = {name: [] for name in self.scenario_metadata}
        self.scenario_total_sample_counts = {name: 0 for name in self.scenario_metadata}

    def _get_stage_idx(self, current_update: int) -> int:
        _integer(current_update, "current_update")
        return self.base_gate.stage_index

    def _compute_transition_alpha(self, current_update: int) -> float:
        self._get_stage_idx(current_update)
        return 1.0

    def get_available_scenarios(self, current_update: int) -> list[str]:
        self._get_stage_idx(current_update)
        return self.base_gate.available_scenarios()

    def get_scenario_coverage_groups(self, scenarios: Sequence[str]) -> dict[str, str]:
        self._validate_scenarios(scenarios)
        return {name: self.scenario_metadata[name]["pair_id"] for name in scenarios}

    def get_scenario_sampling_weights(self, current_update: int) -> dict[str, float]:
        """Diagnostic weights only; sampling must use complete-pair scheduling."""
        return {name: 1.0 for name in self.get_available_scenarios(current_update)}

    def sample_scenarios(self, n_episodes: int, *, phase: Optional[int] = None) -> list[str]:
        completed = self.base_gate.last_update if phase is None else _integer(phase, "phase")
        if completed != self.base_gate.last_update:
            raise ValueError("sampler phase must equal the safely completed curriculum update")
        return sample_base_driving_balanced(
            self.base_gate.available_scenarios(), n_episodes, phase=completed,
        )

    def _validate_scenarios(self, scenarios: Sequence[str]) -> None:
        unknown = set(scenarios) - set(self.scenario_metadata)
        if unknown:
            raise ValueError(f"unregistered base-driving scenarios: {sorted(unknown)}")

    def record_scenario_result(self, scenario_name: str, success: bool) -> None:
        self._validate_scenarios([scenario_name])
        if scenario_name not in self.base_gate.available_scenarios():
            raise ValueError("cannot record a scenario that is still curriculum-locked")
        if not isinstance(success, bool):
            raise ValueError("scenario success must be boolean")
        self.scenario_success_history[scenario_name].append(float(success))
        self.scenario_success_history[scenario_name] = self.scenario_success_history[scenario_name][-20:]
        self.scenario_total_sample_counts[scenario_name] += 1

    def get_scenario_success_rate(self, scenario_name: str) -> float:
        self._validate_scenarios([scenario_name])
        history = self.scenario_success_history[scenario_name]
        return sum(history) / len(history) if history else 0.5

    def get_scenario_sample_count(self, scenario_name: str) -> int:
        self._validate_scenarios([scenario_name])
        return len(self.scenario_success_history[scenario_name])

    def get_scenario_total_sample_count(self, scenario_name: str) -> int:
        self._validate_scenarios([scenario_name])
        return self.scenario_total_sample_counts[scenario_name]

    @property
    def total_count(self) -> int:
        return sum(self.scenario_total_sample_counts.values())

    def observe_update(self, update: int, episodes: Sequence[Mapping]) -> dict[str, object]:
        return self.base_gate.observe_update(update, episodes)

    def sampler_state_dict(self, *, completed_update: Optional[int] = None) -> dict[str, object]:
        if completed_update is not None and _integer(completed_update, "completed_update") != self.base_gate.last_update:
            raise ValueError("checkpoint update does not match safely completed curriculum update")
        return base_driving_sampler_state(
            self.base_gate.last_update, self.base_gate.available_scenarios(),
        )

    def state_dict(self, *, completed_update: Optional[int] = None) -> dict[str, object]:
        return {
            "runtime_protocol_version": BASE_DRIVING_RUNTIME_PROTOCOL_VERSION,
            "pool_manifest_sha256": self.pool_manifest_sha256,
            "base_gate": self.base_gate.state_dict(),
            "sampler_state": self.sampler_state_dict(completed_update=completed_update),
            "scenario_success_history": deepcopy(self.scenario_success_history),
            "scenario_total_sample_counts": dict(self.scenario_total_sample_counts),
        }

    def load_state_dict(self, state: Mapping[str, object], *, expected_completed_update: Optional[int] = None) -> None:
        """Validate all components before replacing any live adapter state."""
        if not isinstance(state, Mapping):
            raise ValueError("base-driving runtime state must be a mapping")
        if state.get("runtime_protocol_version") != BASE_DRIVING_RUNTIME_PROTOCOL_VERSION:
            raise ValueError("base-driving runtime protocol mismatch")
        if state.get("pool_manifest_sha256") != self.pool_manifest_sha256:
            raise ValueError("base-driving pool manifest mismatch")
        restored_gate = BaseDrivingCurriculum(self.scenario_metadata, **self._gate_kwargs)
        restored_gate.load_state_dict(state.get("base_gate"))
        if expected_completed_update is not None and _integer(
            expected_completed_update, "expected_completed_update",
        ) != restored_gate.last_update:
            raise ValueError("restored curriculum update differs from checkpoint update")
        validate_base_driving_sampler_state(
            state.get("sampler_state"),
            expected_completed_update=restored_gate.last_update,
            available_scenarios=restored_gate.available_scenarios(),
        )
        histories = state.get("scenario_success_history")
        counts = state.get("scenario_total_sample_counts")
        registered = set(self.scenario_metadata)
        if not isinstance(histories, Mapping) or set(histories) != registered:
            raise ValueError("saved scenario histories differ from pool")
        if not isinstance(counts, Mapping) or set(counts) != registered:
            raise ValueError("saved scenario counts differ from pool")
        normalized_histories, normalized_counts = {}, {}
        for name in self.scenario_metadata:
            history = histories[name]
            if not isinstance(history, list) or len(history) > 20:
                raise ValueError("saved scenario history exceeds the 20-episode diagnostic horizon")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0.0, 1.0)
                   for value in history):
                raise ValueError("saved scenario history must contain 0/1 outcomes")
            count = _integer(counts[name], "scenario total sample count")
            if len(history) != min(20, count):
                raise ValueError("saved scenario history length disagrees with total count")
            if count and name not in restored_gate.available_scenarios():
                raise ValueError("saved scenario count includes a locked task")
            normalized_histories[name] = [float(value) for value in history]
            normalized_counts[name] = count
        self.base_gate = restored_gate
        self.scenario_success_history = normalized_histories
        self.scenario_total_sample_counts = normalized_counts

    def validate_checkpoint_histories(self, saved_curriculum: Mapping) -> None:
        """Check the trainer's duplicate legacy fields instead of guessing."""
        if not isinstance(saved_curriculum, Mapping):
            raise ValueError("checkpoint curriculum fields must be a mapping")
        if saved_curriculum.get("success_history") != self.scenario_success_history:
            raise ValueError("checkpoint base-driving success histories disagree")
        if saved_curriculum.get("total_sample_counts") != self.scenario_total_sample_counts:
            raise ValueError("checkpoint base-driving sample counts disagree")


def load_adapter(
    scenario_dir: str | Path,
    *,
    project_dir: Optional[str | Path] = None,
    config: Optional[BaseDrivingCurriculumConfig | Mapping] = None,
    agent_ids: Sequence[int] = (0, 1),
    source_protocol_version: str = "base_driving_vnext_v1",
    completed_update: int = 0,
) -> BaseDrivingCurriculumAdapter:
    """Construct the explicit adapter after the caller validates its contract.

    Relative paths are resolved against ``project_dir`` when supplied. This
    helper does not select a reward, change arguments, infer a resume origin,
    or load old mastery states. A warm-start origin must be supplied explicitly.
    """
    path = Path(scenario_dir)
    if not path.is_absolute() and project_dir is not None:
        path = Path(project_dir) / path
    if config is None:
        config = BaseDrivingCurriculumConfig()
    elif isinstance(config, Mapping):
        try:
            config = BaseDrivingCurriculumConfig(**dict(config))
        except TypeError as error:
            raise ValueError(f"invalid base-driving curriculum configuration: {error}") from error
    return BaseDrivingCurriculumAdapter(
        path, agent_ids=agent_ids, config=config,
        source_protocol_version=source_protocol_version,
        completed_update=completed_update,
    )
