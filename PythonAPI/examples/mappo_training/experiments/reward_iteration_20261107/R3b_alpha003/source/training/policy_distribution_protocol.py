"""Frozen tanh-squashed Gaussian policy-distribution contract for R6."""

from __future__ import annotations

from typing import Dict


POLICY_DISTRIBUTION_PROTOCOL_VERSION = "tanh_squashed_gaussian_entropy_v2_r6"
R7_POLICY_DISTRIBUTION_PROTOCOL_VERSION = "tanh_squashed_gaussian_entropy_v3_r7"


def policy_distribution_spec() -> Dict[str, object]:
    return {
        "policy_distribution_protocol_version": (
            POLICY_DISTRIBUTION_PROTOCOL_VERSION
        ),
        "base_distribution": "diagonal_normal",
        "transform": "elementwise_tanh",
        "log_probability": "normal_log_prob_minus_exact_tanh_log_jacobian",
        "entropy_objective": "fresh_current_policy_monte_carlo_post_tanh",
        "pre_tanh_entropy_role": "diagnostic_only",
        "rollout_sample_entropy_reuse": False,
        "log_std_bounds": {"minimum": -2.0, "maximum": 0.0},
        "initial_log_std": -0.5,
        "entropy_coefficient_schedule": {
            "kind": "linear",
            "start": 0.01,
            "final": 0.001,
            "final_update": 160,
        },
        "deterministic_action_saturation_threshold": 0.99,
    }


def r7_policy_distribution_spec() -> Dict[str, object]:
    spec = policy_distribution_spec()
    spec["policy_distribution_protocol_version"] = (
        R7_POLICY_DISTRIBUTION_PROTOCOL_VERSION
    )
    spec["entropy_coefficient_schedule"] = {
        "kind": "linear",
        "start": 0.01,
        "final": 0.001,
        "final_update": 600,
    }
    return spec
