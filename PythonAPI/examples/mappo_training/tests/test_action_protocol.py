import unittest

import numpy as np
import torch

from models.actor_critic import Actor
from training.action_protocol import (
    ACTION_MAPPING_VERSION,
    ACTION_PROTOCOL_VERSION,
    LONGITUDINAL_DEADZONE,
    ActionProtocolError,
    action_protocol_spec,
    action_tanh_to_env,
    legacy_wta_action_tanh_to_env,
    policy_action_diagnostics,
)


class SignedLongitudinalActionProtocolTests(unittest.TestCase):
    def test_frozen_spec_declares_two_policy_and_three_environment_channels(self):
        spec = action_protocol_spec()
        self.assertEqual(spec["action_protocol_version"], ACTION_PROTOCOL_VERSION)
        self.assertEqual(spec["mapping_version"], ACTION_MAPPING_VERSION)
        self.assertEqual(spec["policy_action_channels"], ["signed_longitudinal", "steer"])
        self.assertEqual(spec["environment_action_channels"], ["throttle", "steer", "brake"])
        self.assertEqual(spec["longitudinal_deadzone"], LONGITUDINAL_DEADZONE)

    def test_mapping_is_odd_continuous_bounded_and_batch_safe(self):
        policy = np.asarray(
            [
                [-1.0, -1.0],
                [-0.525, -0.5],
                [-0.05, 0.0],
                [0.0, 0.25],
                [0.05, 0.0],
                [0.525, 0.5],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
        actual = action_tanh_to_env(policy)
        expected = np.asarray(
            [
                [0.0, -1.0, 1.0],
                [0.0, -0.5, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertTrue(np.all((actual[:, 0] == 0.0) | (actual[:, 2] == 0.0)))

        epsilon = 1e-7
        below = action_tanh_to_env([LONGITUDINAL_DEADZONE - epsilon, 0.0])
        above = action_tanh_to_env([LONGITUDINAL_DEADZONE + epsilon, 0.0])
        self.assertLess(float(np.linalg.norm(above - below)), 2.0 * epsilon)

    def test_invalid_shape_bounds_and_nonfinite_values_fail_fast(self):
        invalid = ([0.0, 0.0, 0.0], [1.01, 0.0], [np.nan, 0.0])
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ActionProtocolError):
                action_tanh_to_env(value)

    def test_policy_diagnostics_separate_deadzone_and_saturation(self):
        self.assertEqual(
            policy_action_diagnostics([0.0, 0.2]),
            {
                "signed_longitudinal_deadzone": 1.0,
                "signed_longitudinal_saturation": 0.0,
            },
        )
        self.assertEqual(
            policy_action_diagnostics([-1.0, 0.0])["signed_longitudinal_saturation"],
            1.0,
        )

    def test_legacy_adapter_is_explicit_and_not_accepted_by_current_adapter(self):
        legacy = np.asarray([0.0, 0.25, 0.0], dtype=np.float32)
        np.testing.assert_allclose(
            legacy_wta_action_tanh_to_env(legacy),
            [0.5, 0.25, 0.0],
            atol=1e-7,
        )
        with self.assertRaises(ActionProtocolError):
            action_tanh_to_env(legacy)


class TwoDimensionalPpoDensityTests(unittest.TestCase):
    def test_pre_tanh_log_prob_round_trip_uses_exact_two_dimensional_density(self):
        torch.manual_seed(20260810)
        actor = Actor(obs_dim=19, act_dim=2, hidden=16, init_log_std=0.0)
        observations = torch.randn(8, 19)
        action_tanh, old_log_prob, action_z = actor.sample_action_z(observations)
        new_log_prob, entropy = actor.evaluate_actions_z(observations, action_z)

        self.assertEqual(tuple(action_tanh.shape), (8, 2))
        self.assertEqual(tuple(action_z.shape), (8, 2))
        self.assertEqual(tuple(old_log_prob.shape), (8,))
        self.assertEqual(tuple(entropy.shape), (8,))
        torch.testing.assert_close(new_log_prob, old_log_prob, rtol=0.0, atol=0.0)
        self.assertTrue(torch.equal(torch.exp(new_log_prob - old_log_prob), torch.ones(8)))
        self.assertEqual(action_tanh_to_env(action_tanh.detach().numpy()).shape, (8, 3))


if __name__ == "__main__":
    unittest.main()
