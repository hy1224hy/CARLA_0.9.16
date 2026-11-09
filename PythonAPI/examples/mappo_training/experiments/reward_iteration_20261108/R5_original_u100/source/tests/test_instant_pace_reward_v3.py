"""Necessary v3 reward checks; no simulator connection or integration run."""

from dataclasses import replace
import math
import unittest

from environment import instant_pace_reward as v2
from environment import instant_pace_reward_v3 as v3


class InstantPaceV3Tests(unittest.TestCase):
    def test_stronger_linear_quadratic_cost_and_v2_remains_unchanged(self):
        original_spec = v2.instant_pace_spec()
        for speed_kmh in (20.0, 30.0, 50.0):
            speed = speed_kmh / 3.6
            old_cost, old_risk = v2.continuous_overspeed_penalty(speed, timestep=0.1)
            cost, risk = v3.continuous_overspeed_penalty(speed, timestep=0.1)
            excess = (speed_kmh - 15.0) / 15.0
            self.assertAlmostEqual(old_cost, -0.02 * excess ** 2)
            self.assertAlmostEqual(cost, -0.1 * (0.5 * excess + 0.5 * excess ** 2))
            self.assertLess(cost, old_cost)
            self.assertEqual(risk, old_risk)
        self.assertAlmostEqual(v3.continuous_overspeed_penalty(30 / 3.6, timestep=0.1)[0], -0.1)
        self.assertAlmostEqual(v3.continuous_overspeed_penalty(50 / 3.6, timestep=0.1)[0], -0.3888888888888889)
        v3.instant_pace_spec()
        self.assertEqual(v2.instant_pace_spec(), original_spec)

    def test_upper_boundary_right_slope_and_monotonicity(self):
        upper = v3.CONFIG.overspeed_start_mps
        for speed in (0.0, upper - 1e-6, upper):
            self.assertEqual(v3.continuous_overspeed_penalty(speed, timestep=0.1), (0.0, 0.0))
        epsilon = 1e-5
        right_cost = v3.continuous_overspeed_penalty(upper + epsilon, timestep=0.1)[0]
        expected_slope = -0.1 * 0.5 / v3.CONFIG.overspeed_scale_mps
        self.assertAlmostEqual(right_cost / epsilon, expected_slope, delta=1e-7)
        costs = [v3.continuous_overspeed_penalty(speed / 3.6, timestep=0.1)
                 for speed in (20.0, 30.0, 40.0, 50.0)]
        self.assertTrue(all(a[0] > b[0] for a, b in zip(costs, costs[1:])))
        self.assertTrue(all(a[1] < b[1] for a, b in zip(costs, costs[1:])))
        self.assertGreater(costs[-1][1], 1.0)

    def test_dt_consistency_and_invalid_inputs(self):
        first = v3.continuous_overspeed_penalty(10.0, timestep=0.1)
        second = v3.continuous_overspeed_penalty(10.0, timestep=0.05)
        self.assertAlmostEqual(10 * first[0], 20 * second[0])
        self.assertEqual(first[1], second[1])
        for bad in (math.nan, math.inf, -math.inf, True, None, "10", -1.0, 1e308):
            with self.subTest(speed=bad), self.assertRaises(ValueError):
                v3.continuous_overspeed_penalty(bad, timestep=0.1)
        for bad in (0.0, -0.1, math.nan, math.inf, True, None, "0.1"):
            with self.subTest(timestep=bad), self.assertRaises(ValueError):
                v3.continuous_overspeed_penalty(10.0, timestep=bad)
        with self.assertRaises(ValueError):
            v3.continuous_overspeed_penalty(10.0, timestep=0.1,
                config=replace(v3.CONFIG, overspeed_scale_mps=0.0))

    def test_pace_parameters_and_all_exemptions_match_v2(self):
        defaults = {"distance_to_goal_m": 20.0, "elapsed_seconds": 3.0, "timestep": 0.1}
        overrides = ({}, {"eligible": False}, {"front_blocker": True},
                     {"complex_interaction": True}, {"terminal_event": True},
                     {"elapsed_seconds": 2.0}, {"distance_to_goal_m": 5.0})
        for speed in (-1.0, 0.0, 1.25, 2.5, 5.0):
            for change in overrides:
                with self.subTest(speed=speed, change=change):
                    options = {**defaults, **change}
                    self.assertEqual(v3.instant_pace_deficit(speed, **options),
                                     v2.instant_pace_deficit(speed, **options))
        self.assertEqual(v3.instant_pace_spec()["pace"], v2.instant_pace_spec()["pace"])

    def test_spec_discloses_candidate_formula_and_only_v2_speed_change(self):
        spec = v3.instant_pace_spec()
        self.assertEqual(spec["profile"], "instant_pace_v3")
        self.assertEqual(spec["difference_from_v2"]["changed_components"], ["overspeed"])
        self.assertTrue(spec["difference_from_v2"]["pace_math_and_exemptions_unchanged"])
        self.assertIn("linear_cost_per_second", spec["overspeed"]["formula"])
        self.assertIn("quadratic_cost_per_second", spec["overspeed"]["formula"])
        self.assertEqual(spec["parameters"]["overspeed_linear_cost_per_second"], 0.5)
        self.assertEqual(spec["parameters"]["overspeed_quadratic_cost_per_second"], 0.5)
        self.assertNotIn("overspeed_cost_per_second", spec["parameters"])
        self.assertFalse(spec["policy_invariance_guarantee"])


if __name__ == "__main__":
    unittest.main()
