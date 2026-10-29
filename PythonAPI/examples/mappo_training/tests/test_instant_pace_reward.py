"""Offline candidate-reward checks; no CARLA client is created."""

from dataclasses import replace
import math
import unittest

from environment.base_reward_v6 import PaceDeficitDecision
from environment.instant_pace_reward import (
    CONFIG, PROFILE, continuous_overspeed_penalty, instant_pace_deficit,
    instant_pace_spec, validate_instant_pace_config,
)


class InstantPaceTests(unittest.TestCase):
    def decision(self, speed, **kwargs):
        options = {"distance_to_goal_m": 20.0, "elapsed_seconds": 3.0,
                   "timestep": 0.1}
        options.update(kwargs)
        return instant_pace_deficit(speed, **options)

    def test_current_speed_recovery_clears_cost_without_history(self):
        stopped = self.decision(0.0)
        self.assertIsInstance(stopped, PaceDeficitDecision)
        self.assertAlmostEqual(stopped.reward, -0.05)
        self.assertEqual(stopped.raw_deficit, 2.5)
        self.assertEqual(stopped.risk, 1.0)
        slow = self.decision(1.25)
        self.assertAlmostEqual(slow.reward, -0.025)
        for speed in (2.5, 4.0, 10.0):
            recovered = self.decision(speed, elapsed_seconds=100.0)
            self.assertEqual(recovered.reward, 0.0)
            self.assertEqual(recovered.raw_deficit, 0.0)
        self.assertEqual(self.decision(-3.0).reward, stopped.reward)

    def test_exemptions_and_inclusive_boundaries_preserve_raw_evidence(self):
        for overrides in (
            {"eligible": False}, {"front_blocker": True},
            {"complex_interaction": True}, {"terminal_event": True},
            {"elapsed_seconds": CONFIG.grace_seconds},
            {"distance_to_goal_m": CONFIG.near_goal_distance_m},
        ):
            with self.subTest(overrides=overrides):
                result = self.decision(0.0, **overrides)
                self.assertEqual(result.reward, 0.0)
                self.assertFalse(result.eligible)
                self.assertEqual(result.raw_deficit, 2.5)
                self.assertEqual(result.risk, 1.0)
        self.assertTrue(self.decision(0.0, elapsed_seconds=2.0001,
                                     distance_to_goal_m=5.0001).eligible)
        self.assertTrue(self.decision(0.0, terminal_event=True).terminal_exempt)
        self.assertTrue(self.decision(0.0, elapsed_seconds=2.0).grace_exempt)
        self.assertTrue(self.decision(0.0, distance_to_goal_m=5.0).near_goal_exempt)

    def test_overspeed_is_neutral_below_upper_and_continues_past_thirty(self):
        for speed in (0.0, 2.5, CONFIG.overspeed_start_mps):
            self.assertEqual(continuous_overspeed_penalty(speed, timestep=0.1),
                             (0.0, 0.0))
        speeds = (20 / 3.6, 30 / 3.6, 40 / 3.6, 50 / 3.6)
        values = [continuous_overspeed_penalty(speed, timestep=0.1)
                  for speed in speeds]
        self.assertTrue(all(a[0] > b[0] for a, b in zip(values, values[1:])))
        self.assertTrue(all(a[1] < b[1] for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[1][0], -0.02)
        self.assertAlmostEqual(values[1][1], 1.0)
        self.assertGreater(values[-1][1], 1.0)

    def test_time_integration_is_consistent_for_both_costs(self):
        self.assertAlmostEqual(
            10 * self.decision(1.0, timestep=0.1).reward,
            20 * self.decision(1.0, timestep=0.05).reward,
        )
        first = continuous_overspeed_penalty(10.0, timestep=0.1)[0]
        second = continuous_overspeed_penalty(10.0, timestep=0.05)[0]
        self.assertAlmostEqual(10 * first, 20 * second)

    def test_bad_numbers_flags_and_parameters_are_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, True, None, "2.0"):
            with self.subTest(speed=bad), self.assertRaises(ValueError):
                self.decision(bad)
            with self.subTest(overspeed=bad), self.assertRaises(ValueError):
                continuous_overspeed_penalty(bad, timestep=0.1)
            for key in ("distance_to_goal_m", "elapsed_seconds", "timestep"):
                with self.subTest(field=key, value=bad), self.assertRaises(ValueError):
                    self.decision(1.0, **{key: bad})
        for dt in (0.0, -0.1):
            with self.assertRaises(ValueError):
                self.decision(1.0, timestep=dt)
            with self.assertRaises(ValueError):
                continuous_overspeed_penalty(1.0, timestep=dt)
        for key in ("distance_to_goal_m", "elapsed_seconds"):
            with self.assertRaises(ValueError):
                self.decision(1.0, **{key: -1.0})
        for key in ("eligible", "front_blocker", "complex_interaction", "terminal_event"):
            with self.assertRaises(ValueError):
                self.decision(1.0, **{key: 1})
        with self.assertRaises(ValueError):
            continuous_overspeed_penalty(-1.0, timestep=0.1)
        with self.assertRaises(ValueError):
            continuous_overspeed_penalty(1e308, timestep=0.1)
        for name in (
            "reference_speed_mps", "cost_per_second", "overspeed_start_mps",
            "overspeed_scale_mps", "overspeed_cost_per_second",
        ):
            for value in (0.0, -1.0, math.nan, math.inf, True):
                with self.subTest(parameter=name, value=value), self.assertRaises(ValueError):
                    validate_instant_pace_config(replace(CONFIG, **{name: value}))
        for name in ("grace_seconds", "near_goal_distance_m"):
            with self.assertRaises(ValueError):
                validate_instant_pace_config(replace(CONFIG, **{name: -1.0}))
        with self.assertRaises(ValueError):
            validate_instant_pace_config(replace(CONFIG, reference_speed_mps=5.0))
        with self.assertRaises(ValueError):
            validate_instant_pace_config({})

    def test_spec_discloses_both_replacements_and_new_units(self):
        spec = instant_pace_spec()
        self.assertEqual(spec["profile"], PROFILE)
        self.assertEqual(set(spec["replaced_components"]),
                         {"free_flow_pace_deficit", "overspeed"})
        self.assertEqual(spec["pace"]["raw_deficit_units"], "m/s")
        self.assertFalse(spec["policy_invariance_guarantee"])
        self.assertFalse(spec["observation_action_change"])


if __name__ == "__main__":
    unittest.main()
