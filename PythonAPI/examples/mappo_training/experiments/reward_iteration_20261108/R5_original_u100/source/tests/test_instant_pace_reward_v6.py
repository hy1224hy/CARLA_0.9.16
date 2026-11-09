"""Reward-contract checks; no simulator connection or rollout."""
import importlib
import math
import unittest

from environment import instant_pace_reward_v5 as v5


class InstantPaceV6Tests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("environment.instant_pace_reward_v6")
        self.assertIsNotNone(spec, "V6 reward profile is not implemented")
        self.v6 = importlib.import_module("environment.instant_pace_reward_v6")

    def test_speed_preference_and_high_speed_shoulder(self):
        costs = []
        for kmh, expected in ((0, 0), (15, 0), (30, -.2),
                              (45, -.8), (60, -2.0)):
            cost, risk = self.v6.continuous_overspeed_penalty(kmh / 3.6, timestep=.1)
            self.assertAlmostEqual(cost, expected)
            self.assertAlmostEqual(risk, max(0, (kmh-15)/15)**2)
            costs.append(cost)
        self.assertTrue(all(a > b for a, b in zip(costs[1:], costs[2:])))
        self.assertAlmostEqual(self.v6.continuous_overspeed_penalty(45/3.6, timestep=.05)[0], -.4)

    def test_small_lane_errors_cost_without_weakening_existing_large_error_cost(self):
        self.assertEqual(self.v6.continuous_lane_deviation_penalty(0, timestep=.1), (0, 0, False))
        for distance in (.1, .3, .35, .5, 1.25, 3.5):
            old, risk, _ = v5.continuous_lane_deviation_penalty(distance, timestep=.1)
            cost, new_risk, _ = self.v6.continuous_lane_deviation_penalty(distance, timestep=.1)
            self.assertAlmostEqual(cost, old - .01*(distance/.9)**2)
            self.assertAlmostEqual(new_risk, risk + (distance/.9)**2)
            self.assertLess(cost, old)

    def test_lane_grace_and_illegality_keep_existing_contract(self):
        for violation, opposing, junction, exempted in (
            (False, False, False, True), (True, False, False, False),
            (False, True, False, False), (True, True, True, True),
        ):
            cost, risk, exempt = self.v6.continuous_lane_deviation_penalty(
                1.25, timestep=.1, exempt=True, road_violation=violation,
                opposing_lane=opposing, in_junction=junction)
            self.assertEqual(exempt, exempted)
            self.assertGreater(risk, 0)
            self.assertEqual(cost == 0, exempted)

    def test_steer_changes_do_not_saturate_and_first_control_is_free(self):
        self.assertEqual(self.v6.steer_rate_penalty(.8, None, timestep=.1), (0, 0))
        for delta, expected in ((0, 0), (.2, -.001), (.5, -.00625), (1, -.025)):
            cost, risk = self.v6.steer_rate_penalty(delta, 0, timestep=.1)
            self.assertAlmostEqual(cost, expected)
            self.assertAlmostEqual(risk, (delta/.2)**2)
        self.assertAlmostEqual(self.v6.steer_rate_penalty(.1, 0, timestep=.05)[0]*20,
                               self.v6.steer_rate_penalty(.2, 0, timestep=.1)[0]*10)

    def test_invalid_or_overflow_inputs_fail_closed(self):
        for invalid in (math.nan, math.inf, -math.inf, True, None, "1", -1, 1e308):
            for function in (self.v6.continuous_lane_deviation_penalty,
                             self.v6.continuous_overspeed_penalty):
                with self.subTest(function=function.__name__, value=invalid), self.assertRaises(ValueError):
                    function(invalid, timestep=.1)
        for invalid in (math.nan, math.inf, True, "0.1", 1.1, -1.1):
            with self.subTest(steer=invalid), self.assertRaises(ValueError):
                self.v6.steer_rate_penalty(invalid, 0, timestep=.1)
        for invalid in (0, -.1, math.nan, True, "0.1"):
            with self.subTest(dt=invalid), self.assertRaises(ValueError):
                self.v6.steer_rate_penalty(.1, None, timestep=invalid)

    def test_pace_safety_and_old_spec_stay_frozen(self):
        old_spec = v5.instant_pace_spec()
        for distance in (3, 20):
            for speed in (0, .5, 2.5):
                opts = dict(distance_to_goal_m=distance, elapsed_seconds=3, timestep=.1)
                self.assertEqual(self.v6.instant_pace_deficit(speed, **opts),
                                 v5.instant_pace_deficit(speed, **opts))
        spec = self.v6.instant_pace_spec()
        self.assertEqual(spec['profile'], 'instant_pace_v6')
        self.assertEqual(spec['difference_from_v5']['changed_components'],
                         ['overspeed', 'lane_deviation', 'steer_rate'])
        self.assertFalse(spec['observation_action_change'])
        self.assertFalse(spec['policy_invariance_guarantee'])
        self.assertEqual(spec['positive_dense_components'], ['progress'])
        self.assertEqual(spec['pace'], old_spec['pace'])
        self.assertEqual(v5.instant_pace_spec(), old_spec)


if __name__ == '__main__':
    unittest.main()
