"""Versioned value-target normalization for the MAPPO centralized critic.

The implementation follows the running-moment ValueNorm path used by the
official MAPPO implementation.  The critic predicts normalized values, while
rollout storage, GAE, diagnostics, and public metrics remain in reward units.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


VALUE_NORMALIZATION_PROTOCOL_VERSION = "mappo_valuenorm_v1"
VALUE_LOSS_PROTOCOL_VERSION = "normalized_huber_delta_10_v1"


def value_normalization_spec(
    *,
    beta: float = 0.99999,
    epsilon: float = 1e-5,
    variance_floor: float = 1e-2,
    huber_delta: float = 10.0,
) -> Dict[str, object]:
    return {
        "protocol_version": VALUE_NORMALIZATION_PROTOCOL_VERSION,
        "value_loss_protocol_version": VALUE_LOSS_PROTOCOL_VERSION,
        "enabled": True,
        "target_scope": "pooled_active_agent_return_targets",
        "network_output": "normalized_value",
        "rollout_and_gae_value": "denormalized_reward_units",
        "statistics_update": "once_per_complete_ppo_update_before_minibatches",
        "beta": float(beta),
        "epsilon": float(epsilon),
        "variance_floor": float(variance_floor),
        "loss": "huber",
        "huber_delta": float(huber_delta),
    }


class ValueNormalizer(nn.Module):
    """Bias-corrected exponential running moments for scalar return targets."""

    def __init__(
        self,
        *,
        beta: float = 0.99999,
        epsilon: float = 1e-5,
        variance_floor: float = 1e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("ValueNorm beta must be in [0, 1)")
        if float(epsilon) <= 0.0:
            raise ValueError("ValueNorm epsilon must be positive")
        if float(variance_floor) <= 0.0:
            raise ValueError("ValueNorm variance_floor must be positive")
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.variance_floor = float(variance_floor)
        self.register_buffer("running_mean", torch.zeros((), device=device))
        self.register_buffer("running_mean_sq", torch.zeros((), device=device))
        self.register_buffer("debiasing_term", torch.zeros((), device=device))
        self.register_buffer(
            "update_count", torch.zeros((), dtype=torch.long, device=device)
        )

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        if targets.numel() == 0:
            raise ValueError("ValueNorm cannot update from an empty target batch")
        detached = targets.detach().to(
            device=self.running_mean.device, dtype=self.running_mean.dtype
        )
        if not torch.isfinite(detached).all():
            raise FloatingPointError("ValueNorm targets contain NaN/Inf")
        batch_mean = detached.mean()
        batch_mean_sq = detached.square().mean()
        weight = 1.0 - self.beta
        self.running_mean.mul_(self.beta).add_(batch_mean, alpha=weight)
        self.running_mean_sq.mul_(self.beta).add_(batch_mean_sq, alpha=weight)
        self.debiasing_term.mul_(self.beta).add_(weight)
        self.update_count.add_(1)

    def mean_variance(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = torch.clamp(self.debiasing_term, min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        variance = torch.clamp(
            mean_sq - mean.square(), min=self.variance_floor
        )
        return mean, variance

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, variance = self.mean_variance()
        result = (values - mean) / torch.sqrt(variance)
        if not torch.isfinite(result).all():
            raise FloatingPointError("ValueNorm normalized values contain NaN/Inf")
        return result

    def validate_state(self) -> None:
        for name in ("running_mean", "running_mean_sq", "debiasing_term"):
            value = getattr(self, name)
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"ValueNorm {name} contains NaN/Inf")
        if int(self.update_count.item()) < 0:
            raise ValueError("ValueNorm update_count cannot be negative")
        debias = float(self.debiasing_term.item())
        if not 0.0 <= debias <= 1.0 + self.epsilon:
            raise ValueError("ValueNorm debiasing_term is outside [0,1]")
        if int(self.update_count.item()) > 0 and debias <= 0.0:
            raise ValueError("updated ValueNorm must have positive debiasing_term")
        mean, variance = self.mean_variance()
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise FloatingPointError("ValueNorm decoded moments contain NaN/Inf")

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        mean, variance = self.mean_variance()
        result = normalized_values * torch.sqrt(variance) + mean
        if not torch.isfinite(result).all():
            raise FloatingPointError("ValueNorm denormalized values contain NaN/Inf")
        return result

    def spec(self) -> Dict[str, object]:
        return value_normalization_spec(
            beta=self.beta,
            epsilon=self.epsilon,
            variance_floor=self.variance_floor,
        )

    def statistics(self) -> Dict[str, float]:
        self.validate_state()
        mean, variance = self.mean_variance()
        return {
            "mean": float(mean.detach().cpu().item()),
            "std": float(torch.sqrt(variance).detach().cpu().item()),
            "variance": float(variance.detach().cpu().item()),
            "debiasing_term": float(self.debiasing_term.detach().cpu().item()),
            "update_count": float(self.update_count.detach().cpu().item()),
        }


def normalized_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    delta: float = 10.0,
) -> torch.Tensor:
    """Return mean Huber loss with explicit shape/finite validation."""
    if predictions.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError(
            "value loss shape mismatch: "
            f"predictions={tuple(predictions.shape)}, targets={tuple(targets.shape)}"
        )
    if float(delta) <= 0.0:
        raise ValueError("Huber delta must be positive")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("Huber predictions/targets contain NaN/Inf")
    error = targets - predictions
    absolute_error = error.abs()
    quadratic = torch.minimum(
        absolute_error, torch.as_tensor(delta, device=error.device, dtype=error.dtype)
    )
    linear = absolute_error - quadratic
    loss = (0.5 * quadratic.square() + float(delta) * linear).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Huber loss is NaN/Inf")
    return loss
