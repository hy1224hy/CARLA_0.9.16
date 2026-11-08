"""Pure v4 goal-approach checks; never connects to CARLA."""

from dataclasses import replace
import math
import unittest

from environment import instant_pace_reward_v3 as v3
from environment import instant_pace_reward_v4 as v4


class InstantPaceV4Tests(unittest.TestCase):
    def decision(self, speed, **overrides):
        options = {"distance_to_goal_m": 4.0, "elapsed_seconds": 3.0,
                   "timestep": 0.1}
        options.update(overrides)
        return v4.instant_pace_deficit(speed, **options)

    def test_unfinished_near_goal_stop_costs_and_slow_approach_is_neutral(self):
        for distance in (0.0, 4.0, 5.0):
            stopped = self.decision(0.0, distance_to_goal_m=distance)
            self.assertAlmostEqual(stopped.reward, -0.05)
            self.assertEqual(stopped.raw_deficit, 0.5)
            self.assertEqual(stopped.risk, 1.0)
            self.assertTrue(stopped.eligible)
            self.assertFalse(stopped.near_goal_exempt)
        halfway = self.decision(0.25)
        self.assertAlmostEqual(halfway.reward, -0.025)
        self.assertEqual(halfway.raw_deficit, 0.25)
        for speed in (0.5, 1.0, 3.0):
            self.assertEqual(self.decision(speed).reward, 0.0)
        self.assertEqual(self.decision(-1.0).reward, self.decision(0.0).reward)

    def test_startup_traffic_complex_and_success_terminal_remain_exempt(self):
        for overrides in ({"eligible": False}, {"front_blocker": True},
                          {"complex_interaction": True}, {"terminal_event": True},
                          {"elapsed_seconds": 2.0}):
            with self.subTest(overrides=overrides):
                result = self.decision(0.0, **overrides)
                self.assertEqual(result.reward, 0.0)
                self.assertFalse(result.eligible)
                self.assertFalse(result.near_goal_exempt)
                self.assertEqual(result.raw_deficit, 0.5)
        self.assertTrue(self.decision(0.0, terminal_event=True).terminal_exempt)
        self.assertTrue(self.decision(0.0, elapsed_seconds=2.0).grace_exempt)
        self.assertTrue(self.decision(0.0, elapsed_seconds=2.0001).eligible)

    def test_far_math_and_v3_overspeed_are_unchanged(self):
        original_spec = v3.instant_pace_spec()
        for speed in (-1.0, 0.0, 1.25, 2.5, 5.0):
            options = {"distance_to_goal_m": 5.0001, "elapsed_seconds": 3.0,
                       "timestep": 0.1}
            self.assertEqual(v4.instant_pace_deficit(speed, **options),
                             v3.instant_pace_deficit(speed, **options))
        for speed in (0.0, 15.0, 30.0, 50.0):
            self.assertEqual(v4.continuous_overspeed_penalty(speed / 3.6, timestep=0.1),
                             v3.continuous_overspeed_penalty(speed / 3.6, timestep=0.1))
        v4.instant_pace_spec()
        self.assertEqual(v3.instant_pace_spec(), original_spec)

    def test_dt_and_invalid_values(self):
        self.assertAlmostEqual(10 * self.decision(0.25, timestep=0.1).reward,
                               20 * self.decision(0.25, timestep=0.05).reward)
        for bad in (math.nan, math.inf, -math.inf, True, None, "0.5"):
            with self.subTest(speed=bad), self.assertRaises(ValueError):
                self.decision(bad)
            for name in ("distance_to_goal_m", "elapsed_seconds", "timestep"):
                with self.subTest(name=name, value=bad), self.assertRaises(ValueError):
                    self.decision(0.0, **{name: bad})
        for dt in (0.0, -0.1):
            with self.assertRaises(ValueError):
                self.decision(0.0, timestep=dt)
        with self.assertRaises(ValueError):
            self.decision(0.0, distance_to_goal_m=-1.0)
        with self.assertRaises(ValueError):
            self.decision(0.0, elapsed_seconds=-1.0)
        with self.assertRaises(ValueError):
            self.decision(0.0, front_blocker=1)
        invalid = replace(v4.CONFIG, reference_speed_mps=0.4)
        with self.assertRaises(ValueError):
            self.decision(0.0, config=invalid)
        with self.assertRaises(ValueError):
            v4.instant_pace_spec(invalid)

    def test_spec_discloses_reference_switch_and_retained_speed_math(self):
        spec = v4.instant_pace_spec()
        self.assertEqual(spec["profile"], "instant_pace_v4")
        self.assertEqual(spec["parameters"]["approach_reference_speed_mps"], 0.5)
        self.assertFalse(spec["pace"]["near_goal_exemption"])
        self.assertNotIn("near_goal", spec["pace"]["exemptions"])
        self.assertEqual(spec["pace"]["raw_deficit_units"], "m/s")
        self.assertEqual(spec["overspeed"], v3.instant_pace_spec()["overspeed"])
        self.assertTrue(spec["difference_from_v3"]["far_goal_pace_math_unchanged"])
        self.assertTrue(spec["difference_from_v3"]["overspeed_math_unchanged"])
        self.assertFalse(spec["policy_invariance_guarantee"])


if __name__ == "__main__":
    unittest.main()
