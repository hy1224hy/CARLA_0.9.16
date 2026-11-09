import unittest

from environment.base_reward_v5 import (
    DEFAULT_REWARD_V5_CONFIG,
    REWARD_COMPONENTS,
    lane_change_event_penalty,
    lane_deviation_penalty,
    mission_timeout_reward,
    reward_v5_spec,
    road_rule_terminal_reward,
)


class RewardV5Tests(unittest.TestCase):
    def test_only_progress_is_positive_dense_reward(self):
        spec = reward_v5_spec()
        self.assertEqual(spec["positive_dense_components"], ["progress"])
        self.assertIn("road_rule", REWARD_COMPONENTS)
        self.assertEqual(spec["legal_lane_change_reward"], "no_bonus_no_event_penalty")

    def test_legal_change_has_no_bonus_or_event_cost(self):
        self.assertEqual(
            lane_change_event_penalty(sensor_event=True, legal_lane_change=True),
            0.0,
        )
        penalty, risk = lane_deviation_penalty(2.0, exempt=True)
        self.assertEqual((penalty, risk), (0.0, 0.0))

    def test_rejected_crossing_is_penalized_even_if_marking_was_broken(self):
        self.assertEqual(
            lane_change_event_penalty(sensor_event=True, legal_lane_change=False),
            DEFAULT_REWARD_V5_CONFIG.illegal_lane_change_penalty,
        )

    def test_road_rule_terminal_dominates_route_potential(self):
        penalty = road_rule_terminal_reward(road_violation=True)
        self.assertLess(penalty, -DEFAULT_REWARD_V5_CONFIG.route_potential_scale)

    def test_terminal_priority_suppresses_road_rule_and_timeout(self):
        self.assertEqual(
            road_rule_terminal_reward(
                road_violation=True, collision=True, offroad=False
            ),
            0.0,
        )
        self.assertEqual(
            mission_timeout_reward(
                1.0, mission_timeout=True, road_violation=True
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
