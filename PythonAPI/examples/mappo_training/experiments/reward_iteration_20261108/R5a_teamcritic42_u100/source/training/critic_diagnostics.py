"""Frozen critic diagnostics and replayable probe payloads for R7."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


CRITIC_DIAGNOSTICS_PROTOCOL_VERSION = "agent_outcome_scenario_critic_r7_v1"
CRITIC_PROBE_PROTOCOL_VERSION = "critic_probe_batch_r7_v1"


class CriticDiagnosticsError(ValueError):
    """Raised when critic evidence is missing, non-finite, or misaligned."""


def _finite_vector(values: Iterable[object], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CriticDiagnosticsError(f"{name} must be numeric") from error
    if array.ndim != 1 or array.size == 0:
        raise CriticDiagnosticsError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise CriticDiagnosticsError(f"{name} contains NaN/Inf")
    return array


def critic_fit_statistics(
    targets: Iterable[object], predictions: Iterable[object]
) -> Dict[str, float]:
    target = _finite_vector(targets, name="critic targets")
    prediction = _finite_vector(predictions, name="critic predictions")
    if target.shape != prediction.shape:
        raise CriticDiagnosticsError("critic target/prediction shapes differ")
    residual = target - prediction
    target_variance = float(np.var(target))
    prediction_std = float(np.std(prediction))
    target_std = float(np.std(target))
    explained_variance = 1.0 - float(np.var(residual)) / (target_variance + 1e-8)
    correlation = 0.0
    if target_std > 1e-12 and prediction_std > 1e-12:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    result = {
        "count": float(target.size),
        "explained_variance": explained_variance,
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": prediction_std,
        "prediction_target_std_ratio": prediction_std / max(target_std, 1e-8),
        "signed_bias": float(np.mean(prediction - target)),
        "absolute_error_mean": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise CriticDiagnosticsError("critic statistics contain NaN/Inf")
    return result


def grouped_critic_fit_statistics(
    targets: Sequence[object],
    predictions: Sequence[object],
    labels: Sequence[object],
) -> Dict[str, Dict[str, float]]:
    target = _finite_vector(targets, name="critic targets")
    prediction = _finite_vector(predictions, name="critic predictions")
    label_values = [str(value) for value in labels]
    if target.shape != prediction.shape or len(label_values) != target.size:
        raise CriticDiagnosticsError("grouped critic arrays are misaligned")
    if any(not label.strip() for label in label_values):
        raise CriticDiagnosticsError("critic labels cannot be empty")
    result = {}
    for label in sorted(set(label_values)):
        mask = np.asarray([value == label for value in label_values], dtype=bool)
        result[label] = critic_fit_statistics(target[mask], prediction[mask])
    return result


def build_critic_probe_batch(
    *,
    states: Sequence[Sequence[object]],
    agent_ids: Sequence[object],
    targets: Sequence[object],
    maximum_rows: int,
) -> Dict[str, object]:
    """Build a bounded deterministic sample that future checkpoints can replay."""

    try:
        state_array = np.asarray(states, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.float32)
        agent_array = np.asarray(agent_ids, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CriticDiagnosticsError("critic probe arrays must be numeric") from error
    if state_array.ndim != 2 or state_array.shape[0] == 0:
        raise CriticDiagnosticsError("critic probe states must be a non-empty matrix")
    if target_array.shape != (state_array.shape[0],):
        raise CriticDiagnosticsError("critic probe targets are misaligned")
    if agent_array.shape != (state_array.shape[0],):
        raise CriticDiagnosticsError("critic probe agent ids are misaligned")
    if not np.all(np.isfinite(state_array)) or not np.all(np.isfinite(target_array)):
        raise CriticDiagnosticsError("critic probe contains NaN/Inf")
    if isinstance(maximum_rows, bool) or int(maximum_rows) <= 0:
        raise CriticDiagnosticsError("maximum_rows must be positive")
    row_count = min(int(maximum_rows), state_array.shape[0])
    # Evenly spaced deterministic selection preserves the whole rollout span
    # and avoids coupling checkpoint evidence to NumPy/PyTorch RNG state.
    indices = np.linspace(0, state_array.shape[0] - 1, row_count, dtype=np.int64)
    return {
        "critic_probe_protocol_version": CRITIC_PROBE_PROTOCOL_VERSION,
        "state_dim": int(state_array.shape[1]),
        "row_count": int(row_count),
        "source_row_count": int(state_array.shape[0]),
        "selection": "evenly_spaced_deterministic",
        "states": state_array[indices].tolist(),
        "agent_ids": agent_array[indices].tolist(),
        "targets": target_array[indices].tolist(),
    }


def validate_critic_probe_batch(
    payload: Mapping[str, object], *, expected_state_dim: int
) -> Dict[str, object]:
    required = {
        "critic_probe_protocol_version",
        "state_dim",
        "row_count",
        "source_row_count",
        "selection",
        "states",
        "agent_ids",
        "targets",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise CriticDiagnosticsError("critic probe fields differ from R7 schema")
    if payload["critic_probe_protocol_version"] != CRITIC_PROBE_PROTOCOL_VERSION:
        raise CriticDiagnosticsError("critic probe protocol mismatch")
    if int(payload["state_dim"]) != int(expected_state_dim):
        raise CriticDiagnosticsError("critic probe state_dim mismatch")
    normalized = build_critic_probe_batch(
        states=payload["states"],
        agent_ids=payload["agent_ids"],
        targets=payload["targets"],
        maximum_rows=int(payload["row_count"]),
    )
    if int(payload["row_count"]) != normalized["row_count"]:
        raise CriticDiagnosticsError("critic probe row_count mismatch")
    if int(payload["source_row_count"]) < int(payload["row_count"]):
        raise CriticDiagnosticsError("critic probe source_row_count is invalid")
    return {
        **dict(payload),
        "states": normalized["states"],
        "agent_ids": normalized["agent_ids"],
        "targets": normalized["targets"],
    }


def critic_diagnostics_protocol_spec() -> Dict[str, object]:
    return {
        "critic_diagnostics_protocol_version": CRITIC_DIAGNOSTICS_PROTOCOL_VERSION,
        "probe_protocol_version": CRITIC_PROBE_PROTOCOL_VERSION,
        "phases": ["pre_update", "post_update"],
        "required_groups": ["pooled", "per_agent", "per_outcome", "per_scenario"],
        "statistics": [
            "count",
            "explained_variance",
            "target_mean",
            "target_std",
            "prediction_mean",
            "prediction_std",
            "prediction_target_std_ratio",
            "signed_bias",
            "absolute_error_mean",
            "correlation",
        ],
        "historical_r6_probe_available": False,
        "r7_probe_selection": "bounded_evenly_spaced_deterministic",
    }
