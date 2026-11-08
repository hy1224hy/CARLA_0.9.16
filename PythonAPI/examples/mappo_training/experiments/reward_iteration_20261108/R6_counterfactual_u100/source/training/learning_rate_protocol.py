"""Versioned actor/critic learning-rate schedules for R7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict


LEARNING_RATE_PROTOCOL_VERSION = "separate_actor_critic_lr_r7_v1"


@dataclass(frozen=True)
class R7LearningRateConfig:
    actor_initial_lr: float = 3e-4
    actor_final_lr: float = 1.5e-5
    actor_decay_updates: int = 600
    critic_initial_lr: float = 3e-4
    critic_hold_updates: int = 300
    critic_final_lr: float = 1e-4
    critic_decay_updates: int = 600


DEFAULT_R7_LR_CONFIG = R7LearningRateConfig()


def validate_r7_lr_config(config: R7LearningRateConfig) -> None:
    if not isinstance(config, R7LearningRateConfig):
        raise ValueError("config must be R7LearningRateConfig")
    for name in (
        "actor_initial_lr",
        "actor_final_lr",
        "critic_initial_lr",
        "critic_final_lr",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.actor_final_lr > config.actor_initial_lr:
        raise ValueError("actor final lr cannot exceed initial lr")
    if config.critic_final_lr > config.critic_initial_lr:
        raise ValueError("critic final lr cannot exceed initial lr")
    if not 0 <= config.critic_hold_updates < config.critic_decay_updates:
        raise ValueError("critic hold must end before critic decay horizon")
    if config.actor_decay_updates <= 1 or config.critic_decay_updates <= 1:
        raise ValueError("lr decay horizons must exceed one update")


validate_r7_lr_config(DEFAULT_R7_LR_CONFIG)


def _linear(initial: float, final: float, position: int, horizon: int) -> float:
    fraction = min(1.0, max(0.0, (int(position) - 1) / max(horizon - 1, 1)))
    return initial + fraction * (final - initial)


def r7_learning_rates(
    update: int, config: R7LearningRateConfig = DEFAULT_R7_LR_CONFIG
) -> Dict[str, float]:
    validate_r7_lr_config(config)
    if isinstance(update, bool) or int(update) <= 0:
        raise ValueError("update must be positive")
    actor_lr = _linear(
        config.actor_initial_lr,
        config.actor_final_lr,
        int(update),
        config.actor_decay_updates,
    )
    if int(update) <= config.critic_hold_updates:
        critic_lr = config.critic_initial_lr
    else:
        critic_lr = _linear(
            config.critic_initial_lr,
            config.critic_final_lr,
            int(update) - config.critic_hold_updates,
            config.critic_decay_updates - config.critic_hold_updates,
        )
    return {"actor_lr": actor_lr, "critic_lr": critic_lr}


def learning_rate_protocol_spec(
    config: R7LearningRateConfig = DEFAULT_R7_LR_CONFIG,
) -> Dict[str, object]:
    validate_r7_lr_config(config)
    return {
        "learning_rate_protocol_version": LEARNING_RATE_PROTOCOL_VERSION,
        "parameters": asdict(config),
        "actor_schedule": "linear_initial_to_final",
        "critic_schedule": "hold_then_linear_initial_to_final",
        "rationale": "avoid_the_r6_u160_critic_floor_before_value_fit_is_established",
    }
