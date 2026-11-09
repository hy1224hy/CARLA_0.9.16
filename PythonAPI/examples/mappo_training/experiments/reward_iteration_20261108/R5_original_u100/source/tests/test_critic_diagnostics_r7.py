import unittest

import numpy as np

from training.critic_diagnostics import (
    build_critic_probe_batch,
    critic_fit_statistics,
    grouped_critic_fit_statistics,
    validate_critic_probe_batch,
)
from training.learning_rate_protocol import (
    DEFAULT_R7_LR_CONFIG,
    r7_learning_rates,
)


class CriticDiagnosticsR7Tests(unittest.TestCase):
    def test_exact_fit_has_unit_explained_variance(self):
        stats = critic_fit_statistics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        self.assertAlmostEqual(stats["explained_variance"], 1.0)
        self.assertEqual(stats["absolute_error_mean"], 0.0)

    def test_constant_prediction_exposes_std_collapse(self):
        stats = critic_fit_statistics([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])
        self.assertEqual(stats["prediction_std"], 0.0)
        self.assertEqual(stats["prediction_target_std_ratio"], 0.0)

    def test_group_statistics_do_not_hide_agent_failure(self):
        grouped = grouped_critic_fit_statistics(
            [0.0, 1.0, 0.0, 10.0],
            [0.0, 1.0, 5.0, 5.0],
            ["agent_0", "agent_0", "agent_1", "agent_1"],
        )
        self.assertAlmostEqual(grouped["agent_0"]["explained_variance"], 1.0)
        self.assertLess(grouped["agent_1"]["explained_variance"], 1.0)

    def test_probe_is_bounded_deterministic_and_validated(self):
        states = np.arange(40, dtype=np.float32).reshape(10, 4)
        first = build_critic_probe_batch(
            states=states,
            agent_ids=[0, 1] * 5,
            targets=np.arange(10),
            maximum_rows=4,
        )
        second = build_critic_probe_batch(
            states=states,
            agent_ids=[0, 1] * 5,
            targets=np.arange(10),
            maximum_rows=4,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["row_count"], 4)
        self.assertEqual(validate_critic_probe_batch(first, expected_state_dim=4), first)

    def test_critic_lr_is_held_then_decays_to_non_degenerate_floor(self):
        config = DEFAULT_R7_LR_CONFIG
        self.assertEqual(r7_learning_rates(1)["critic_lr"], config.critic_initial_lr)
        self.assertEqual(
            r7_learning_rates(config.critic_hold_updates)["critic_lr"],
            config.critic_initial_lr,
        )
        self.assertAlmostEqual(
            r7_learning_rates(config.critic_decay_updates)["critic_lr"],
            config.critic_final_lr,
        )
        self.assertGreater(config.critic_final_lr, config.actor_final_lr)


if __name__ == "__main__":
    unittest.main()
