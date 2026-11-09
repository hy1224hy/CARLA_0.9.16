"""Controllable policy distillation with fixed sample and teacher semantics.

This is the project's continuous-policy adaptation, not an exact Mut-DQN or
DPD reproduction. Distribution KL is measured before the policy tanh; the
subsequent physical-control deadzone is not a bijection and is not this KL.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

DIRECTIONS = ("student_to_teacher", "teacher_to_student", "symmetric")
GATE_VARIANTS = ("evidence", "random", "strength")


def validate_distillation_options(options):
    options = dict(options or {})
    unknown = set(options) - {"direction", "reduction", "gate_variant"}
    if unknown:
        raise ValueError(f"unknown distillation controls: {sorted(unknown)}")
    options.setdefault("direction", "student_to_teacher")
    options.setdefault("reduction", "all_valid")
    options.setdefault("gate_variant", "evidence")
    if options["direction"] not in DIRECTIONS:
        raise ValueError("unknown KL direction")
    if options["reduction"] != "all_valid":
        raise ValueError("all_valid is the versioned denominator; zero-gate rows cannot be discarded")
    if options["gate_variant"] not in GATE_VARIANTS:
        raise ValueError("unknown gate variant")
    return options


def student_teacher_kl(student, teacher, obs, gate=None, *, direction="student_to_teacher", reduction="all_valid"):
    validate_distillation_options({"direction": direction, "reduction": reduction})
    mu, log_std = student(obs)
    with torch.no_grad():
        peer_mu, peer_log_std = teacher(obs)
    own, peer = Normal(mu, log_std.exp()), Normal(peer_mu, peer_log_std.exp())
    forward = kl_divergence(own, peer).sum(-1)
    reverse = kl_divergence(peer, own).sum(-1)
    per_sample = forward if direction == "student_to_teacher" else reverse if direction == "teacher_to_student" else (forward + reverse) * 0.5
    if not per_sample.numel() or not torch.isfinite(per_sample).all():
        raise ValueError("KL requires nonempty finite policy distributions")
    weights = torch.ones_like(per_sample) if gate is None else torch.as_tensor(gate, dtype=per_sample.dtype, device=per_sample.device)
    if weights.shape != per_sample.shape or not torch.isfinite(weights).all() or not ((weights >= 0) & (weights <= 1)).all():
        raise ValueError("distillation needs one weight in [0,1] per valid row")
    return (per_sample * weights).mean()


def build_distillation_gates(base_gates, variant="evidence", rng=None):
    """Diagnostic controls matched to this update's own evidence coverage.

    random permutes exactly the same number of enabled valid rows. strength
    weights every valid row by the evidence fraction. Neither guarantees equal
    *realized KL* across independently trained runs. A matched external replay
    schedule is needed for stronger cross-run mechanism attribution.
    """
    weights = np.asarray(base_gates, dtype=np.float32)
    if weights.ndim != 1 or not np.isfinite(weights).all() or not np.all((weights == 0) | (weights == 1)):
        raise ValueError("evidence gates must be a finite 1D binary array")
    if variant not in GATE_VARIANTS:
        raise ValueError("unknown gate variant")
    if variant == "evidence" or not len(weights):
        return weights.copy()
    if variant == "random":
        generator = rng if rng is not None else np.random
        return weights[generator.permutation(len(weights))].copy()
    return np.full_like(weights, float(weights.mean()))
