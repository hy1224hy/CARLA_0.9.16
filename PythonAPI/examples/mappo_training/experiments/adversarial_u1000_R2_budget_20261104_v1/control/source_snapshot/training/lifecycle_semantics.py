"""Shared terminated/truncated protocol checks for current training entrypoints."""

from typing import Optional, Tuple

import numpy as np


def validate_lifecycle_flags(
    done: bool,
    terminated: bool,
    truncated: bool,
    *,
    context: str = "transition",
) -> Tuple[bool, bool, bool]:
    """Validate and normalize one agent's lifecycle flags.

    ``done`` is the rollout boundary, while ``terminated`` and ``truncated``
    are mutually exclusive reasons for that boundary.
    """
    done = bool(done)
    terminated = bool(terminated)
    truncated = bool(truncated)

    if terminated and truncated:
        raise ValueError(f"{context}: terminated and truncated cannot both be true")
    if done != (terminated or truncated):
        raise ValueError(
            f"{context}: done={done} must equal "
            f"terminated|truncated={terminated or truncated}"
        )
    return done, terminated, truncated


def bootstrap_state_for_boundary(
    done: bool,
    terminated: bool,
    truncated: bool,
    next_global_state: np.ndarray,
    *,
    context: str = "episode boundary",
) -> Optional[np.ndarray]:
    """Return a copied bootstrap state only for a valid truncation boundary."""
    done, terminated, truncated = validate_lifecycle_flags(
        done, terminated, truncated, context=context
    )
    if not done:
        raise ValueError(f"{context}: bootstrap selection requires an episode boundary")
    if terminated:
        return None
    return np.asarray(next_global_state, dtype=np.float32).copy()
