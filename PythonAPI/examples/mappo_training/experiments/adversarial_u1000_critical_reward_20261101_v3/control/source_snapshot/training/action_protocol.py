"""Versioned policy-to-CARLA action adapters for the current base trainer.

The current policy samples a two-dimensional bounded latent action
``[signed_longitudinal, steer]``.  CARLA still consumes the physical
three-channel command ``[throttle, steer, brake]``.  Keeping this conversion
in a CARLA-independent module gives training, evaluation, metrics and tests a
single protocol source of truth.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


LEGACY_ACTION_PROTOCOL_VERSION = "tanh_gaussian_pre_tanh_z_v1"
ACTION_PROTOCOL_VERSION = (
    "tanh_gaussian_signed_longitudinal_2d_pre_tanh_z_v2"
)
LEGACY_ACTION_MAPPING_VERSION = "winner_take_all_v1"
ACTION_MAPPING_VERSION = "signed_longitudinal_deadzone_v2"

POLICY_ACTION_DIM = 2
ENV_ACTION_DIM = 3
POLICY_ACTION_CHANNELS = ("signed_longitudinal", "steer")
ENV_ACTION_CHANNELS = ("throttle", "steer", "brake")
LONGITUDINAL_DEADZONE = 0.05
ACTION_BOUND_TOLERANCE = 1e-6
ACTION_SATURATION_THRESHOLD = 0.99


class ActionProtocolError(ValueError):
    """Raised when an action or action specification violates the protocol."""


def action_protocol_spec() -> Dict[str, object]:
    """Return the immutable current per-policy action specification."""
    return {
        "action_protocol_version": ACTION_PROTOCOL_VERSION,
        "mapping_version": ACTION_MAPPING_VERSION,
        "policy_action_dim": POLICY_ACTION_DIM,
        "policy_action_channels": list(POLICY_ACTION_CHANNELS),
        "environment_action_dim": ENV_ACTION_DIM,
        "environment_action_channels": list(ENV_ACTION_CHANNELS),
        "longitudinal_deadzone": LONGITUDINAL_DEADZONE,
    }


def legacy_action_protocol_spec() -> Dict[str, object]:
    """Return the immutable historical WTA v1 specification."""
    return {
        "action_protocol_version": LEGACY_ACTION_PROTOCOL_VERSION,
        "mapping_version": LEGACY_ACTION_MAPPING_VERSION,
        "policy_action_dim": 3,
        "policy_action_channels": ["throttle_latent", "steer", "brake_latent"],
        "environment_action_dim": ENV_ACTION_DIM,
        "environment_action_channels": list(ENV_ACTION_CHANNELS),
        "longitudinal_deadzone": None,
    }


def validate_action_spec(
    payload: Mapping[str, object],
    *,
    legacy: bool = False,
    context: str = "action_spec",
) -> None:
    expected = legacy_action_protocol_spec() if legacy else action_protocol_spec()
    if not isinstance(payload, Mapping) or dict(payload) != expected:
        raise ActionProtocolError(
            f"{context} differs from "
            f"{'legacy WTA v1' if legacy else 'signed-longitudinal v2'}"
        )


def _validated_action_array(
    action: object,
    *,
    expected_dim: int,
    context: str,
) -> np.ndarray:
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ActionProtocolError(f"{context} must be numeric") from error
    if array.ndim < 1 or array.shape[-1] != expected_dim:
        raise ActionProtocolError(
            f"{context} must have final dimension {expected_dim}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ActionProtocolError(f"{context} contains NaN/Inf")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if (
        minimum < -1.0 - ACTION_BOUND_TOLERANCE
        or maximum > 1.0 + ACTION_BOUND_TOLERANCE
    ):
        raise ActionProtocolError(
            f"{context} is outside [-1,1]: min={minimum}, max={maximum}"
        )
    # Tanh is already bounded.  The tolerance only absorbs floating-point
    # roundoff at the endpoints and must not become a silent generic clip.
    return np.clip(array, -1.0, 1.0)


def signed_longitudinal_command(
    signed_longitudinal: object,
    *,
    deadzone: float = LONGITUDINAL_DEADZONE,
) -> np.ndarray:
    """Apply the odd, continuous dead-zone rescaling ``g_d(l)``."""
    if not np.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
        raise ActionProtocolError("longitudinal deadzone must be in [0,1)")
    values = np.asarray(signed_longitudinal, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ActionProtocolError("signed_longitudinal contains NaN/Inf")
    magnitude = np.maximum(np.abs(values) - deadzone, 0.0) / (1.0 - deadzone)
    return np.sign(values) * magnitude


def action_tanh_to_env(action_tanh: object) -> np.ndarray:
    """Map current policy action ``[...,2]`` to CARLA action ``[...,3]``.

    Positive signed longitudinal values produce throttle, negative values
    produce brake, and the frozen central dead zone produces coast.  The
    output is continuous at both dead-zone boundaries and throttle/brake are
    mutually exclusive by construction.
    """
    array = _validated_action_array(
        action_tanh,
        expected_dim=POLICY_ACTION_DIM,
        context="signed-longitudinal policy action",
    )
    longitudinal = signed_longitudinal_command(array[..., 0])
    throttle = np.maximum(longitudinal, 0.0)
    brake = np.maximum(-longitudinal, 0.0)
    result = np.stack((throttle, array[..., 1], brake), axis=-1)
    return result.astype(np.float32, copy=False)


def legacy_wta_action_tanh_to_env(action_tanh: object) -> np.ndarray:
    """Read-only historical WTA v1 adapter; never use for new training."""
    array = _validated_action_array(
        action_tanh,
        expected_dim=3,
        context="legacy WTA policy action",
    )
    throttle_raw = (array[..., 0] + 1.0) * 0.5
    brake_raw = (array[..., 2] + 1.0) * 0.5
    throttle_wins = throttle_raw >= brake_raw
    throttle = np.where(throttle_wins, throttle_raw, 0.0)
    brake = np.where(throttle_wins, 0.0, brake_raw)
    result = np.stack((throttle, array[..., 1], brake), axis=-1)
    return result.astype(np.float32, copy=False)


def policy_action_diagnostics(action_tanh: object) -> Dict[str, float]:
    """Classify one current two-dimensional policy action for audit logs."""
    array = _validated_action_array(
        action_tanh,
        expected_dim=POLICY_ACTION_DIM,
        context="policy action diagnostics",
    )
    if array.shape != (POLICY_ACTION_DIM,):
        raise ActionProtocolError(
            "policy action diagnostics requires one shape-(2,) action"
        )
    longitudinal = float(array[0])
    return {
        "signed_longitudinal_deadzone": float(
            abs(longitudinal) <= LONGITUDINAL_DEADZONE
        ),
        "signed_longitudinal_saturation": float(
            abs(longitudinal) >= ACTION_SATURATION_THRESHOLD
        ),
    }
