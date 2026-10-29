"""Necessary pure v5 reference-lane checks; no CARLA client is created."""

import math
import unittest

from environment import instant_pace_reward_v4 as v4
from environment import instant_pace_reward_v5 as v5


class InstantPaceV5Tests(unittest.TestCase):
    def test_deadzone_and_immediate_outside_feedback(self):
        for distance in (0.0, 0.1, 0.35):
            self.assertEqual(v5.continuous_lane_deviation_penalty(distance, timestep=0.1),
                             (0.0, 0.0, False))
        cost, risk, exemption = v5.continuous_lane_deviation_penalty(0.359, timestep=0.1)
        self.assertAlmostEqual(risk, 0.01 + 0.01 ** 2)
        self.assertAlmostEqual(cost, -0.01 * risk)
        self.assertFalse(exemption)

    def test_large_lane_bias_cost_grows_without_saturation(self):
        values = [v5.continuous_lane_deviation_penalty(distance, timestep=0.1)
                  for distance in (0.5, 1.25, 3.5, 10.0)]
        self.assertTrue(all(a[0] > b[0] for a, b in zip(values, values[1:])))
        self.assertTrue(all(a[1] < b[1] for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[1][1], 2.0)
        self.assertAlmostEqual(values[1][0], -0.02)
        self.assertGreater(values[-1][1], 1.0)
        self.assertLess(values[-1][0], -1.0)

    def test_legal_exemption_and_junction_ambiguity_are_preserved(self):
        legal = v5.continuous_lane_deviation_penalty(3.5, timestep=0.1, exempt=True)
        self.assertEqual(legal[0], 0.0)
        self.assertGreater(legal[1], 1.0)
        self.assertTrue(legal[2])
        for road, opposing in ((True, False), (False, True), (True, True)):
            result = v5.continuous_lane_deviation_penalty(
                3.5, timestep=0.1, exempt=True, road_violation=road,
                opposing_lane=opposing, in_junction=True,
            )
            self.assertEqual(result, legal)
        no_original_exemption = v5.continuous_lane_deviation_penalty(
            3.5, timestep=0.1, exempt=False, in_junction=True,
        )
        self.assertFalse(no_original_exemption[2])
        self.assertLess(no_original_exemption[0], 0.0)

    def test_confirmed_nonjunction_illegality_cancels_only_exemption(self):
        unexempted = v5.continuous_lane_deviation_penalty(3.5, timestep=0.1)
        for road, opposing in ((True, False), (False, True), (True, True)):
            result = v5.continuous_lane_deviation_penalty(
                3.5, timestep=0.1, exempt=True,
                road_violation=road, opposing_lane=opposing,
            )
            self.assertEqual(result, unexempted)
            self.assertFalse(result[2])

    def test_timestep_and_invalid_values(self):
        first = v5.continuous_lane_deviation_penalty(2.0, timestep=0.1)
        second = v5.continuous_lane_deviation_penalty(2.0, timestep=0.05)
        self.assertAlmostEqual(10 * first[0], 20 * second[0])
        self.assertEqual(first[1:], second[1:])
        for bad in (math.nan, math.inf, -math.inf, True, None, "1.0", -1.0, 1e308):
            with self.subTest(distance=bad), self.assertRaises(ValueError):
                v5.continuous_lane_deviation_penalty(bad, timestep=0.1)
        for bad in (0.0, -0.1, math.nan, math.inf, True, None, "0.1"):
            with self.subTest(dt=bad), self.assertRaises(ValueError):
                v5.continuous_lane_deviation_penalty(2.0, timestep=bad)
        for flag in ("exempt", "road_violation", "opposing_lane", "in_junction"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                v5.continuous_lane_deviation_penalty(2.0, timestep=0.1, **{flag: 1})

    def test_retains_v4_math_and_spec_discloses_only_dense_lane_change(self):
        original = v4.instant_pace_spec()
        for distance in (3.0, 20.0):
            for speed in (0.0, 0.5, 2.5):
                options = {"distance_to_goal_m": distance,
                           "elapsed_seconds": 3.0, "timestep": 0.1}
                self.assertEqual(v5.instant_pace_deficit(speed, **options),
                                 v4.instant_pace_deficit(speed, **options))
        self.assertEqual(v5.continuous_overspeed_penalty(10.0, timestep=0.1),
                         v4.continuous_overspeed_penalty(10.0, timestep=0.1))
        spec = v5.instant_pace_spec()
        self.assertEqual(spec["profile"], "instant_pace_v5")
        self.assertEqual(spec["difference_from_v4"]["changed_components"], ["lane_deviation"])
        self.assertEqual(spec["pace"], original["pace"])
        self.assertEqual(spec["overspeed"], original["overspeed"])
        self.assertFalse(spec["lane_deviation"]["termination_change"])
        self.assertFalse(spec["lane_deviation"]["new_terminal_penalty"])
        self.assertIn("lane_deviation", spec["replaced_components"])
        self.assertFalse(spec["policy_invariance_guarantee"])
        self.assertEqual(v4.instant_pace_spec(), original)


if __name__ == "__main__":
    unittest.main()
