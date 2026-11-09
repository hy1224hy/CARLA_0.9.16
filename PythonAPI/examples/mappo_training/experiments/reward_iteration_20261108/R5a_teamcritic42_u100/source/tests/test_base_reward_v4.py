import unittest

from environment.base_reward_v4 import (
    DEFAULT_REWARD_V4_CONFIG,
    approach_gated_lidar_penalty,
    control_rate_penalties,
    discounted_return,
    lane_deviation_penalty,
    mission_timeout_reward,
    projected_ttc_seconds,
    route_potential_reward,
    terminal_rewards,
)


GAMMA = 0.999
HORIZON = 500


def _trajectory_return(completions, *, terminal_reward):
    rewards = []
    for previous, current in zip(completions[:-1], completions[1:]):
        rewards.append(
            route_potential_reward(previous, current, gamma=GAMMA)
        )
    rewards[-1] += terminal_reward
    return discounted_return(rewards, gamma=GAMMA)


def _trajectory_return_at(completions, *, terminal_reward, gamma):
    rewards = [
        route_potential_reward(previous, current, gamma=gamma)
        for previous, current in zip(completions[:-1], completions[1:])
    ]
    rewards[-1] += terminal_reward
    return discounted_return(rewards, gamma=gamma)


class RewardV4BudgetTests(unittest.TestCase):
    def test_seven_scripted_trajectory_ordering(self):
        stopped = [0.0] * (HORIZON + 1)
        partial = [0.8 * step / HORIZON for step in range(HORIZON + 1)]
        success = [1.0 * step / HORIZON for step in range(HORIZON + 1)]
        cyclic = [0.0]
        for step in range(1, HORIZON + 1):
            cyclic.append(0.4 * (step / 250.0) if step <= 250 else 0.4 * ((500 - step) / 250.0))
        early_linger = [min(0.8, 0.8 * step / 100.0) for step in range(HORIZON + 1)]

        stopped_return = _trajectory_return(
            stopped,
            terminal_reward=mission_timeout_reward(1.0, mission_timeout=True),
        )
        partial_return = _trajectory_return(
            partial,
            terminal_reward=mission_timeout_reward(0.2, mission_timeout=True),
        )
        success_return = _trajectory_return(
            success,
            terminal_reward=terminal_rewards(success=True, collision=False)[1],
        )
        collision_return = _trajectory_return(
            partial,
            terminal_reward=terminal_rewards(success=False, collision=True)[0],
        )
        offroad_return = _trajectory_return(
            partial,
            terminal_reward=DEFAULT_REWARD_V4_CONFIG.offroad_penalty,
        )
        cyclic_return = _trajectory_return(
            cyclic,
            terminal_reward=mission_timeout_reward(1.0, mission_timeout=True),
        )
        linger_return = _trajectory_return(
            early_linger,
            terminal_reward=mission_timeout_reward(0.2, mission_timeout=True),
        )

        self.assertGreater(success_return, partial_return)
        self.assertGreater(partial_return, stopped_return)
        self.assertLess(collision_return, stopped_return)
        self.assertLess(offroad_return, stopped_return)
        self.assertAlmostEqual(cyclic_return, stopped_return, places=10)
        # Potential shaping depends on the final potential, not when it was
        # collected, so reaching q=.8 early and lingering cannot farm return.
        self.assertAlmostEqual(linger_return, partial_return, places=10)

    def test_timeout_is_unfinished_terminal_failure_and_is_exclusive(self):
        self.assertEqual(mission_timeout_reward(0.0, mission_timeout=False), 0.0)
        self.assertEqual(
            mission_timeout_reward(1.0, mission_timeout=True), -20.0
        )
        self.assertEqual(
            mission_timeout_reward(0.2, mission_timeout=True), -12.0
        )
        self.assertEqual(
            mission_timeout_reward(
                1.0, mission_timeout=True, collision=True
            ),
            0.0,
        )

    def test_lane_center_is_never_positive_and_legal_merge_can_be_exempt(self):
        centered, centered_risk = lane_deviation_penalty(0.0)
        edge, edge_risk = lane_deviation_penalty(1.5)
        exempted, exempted_risk = lane_deviation_penalty(1.5, exempt=True)
        self.assertEqual((centered, centered_risk), (0.0, 0.0))
        self.assertLess(edge, 0.0)
        self.assertEqual(edge_risk, 1.0)
        self.assertEqual((exempted, exempted_risk), (0.0, 0.0))

    def test_safety_ordering_is_stable_across_gamma_and_horizon_sensitivity(self):
        for gamma in (0.997, 0.999, 1.0):
            for horizon in (250, 500):
                stopped = [0.0] * (horizon + 1)
                partial = [0.8 * step / horizon for step in range(horizon + 1)]
                stopped_return = _trajectory_return_at(
                    stopped,
                    terminal_reward=mission_timeout_reward(1.0, mission_timeout=True),
                    gamma=gamma,
                )
                partial_return = _trajectory_return_at(
                    partial,
                    terminal_reward=mission_timeout_reward(0.2, mission_timeout=True),
                    gamma=gamma,
                )
                collision_return = _trajectory_return_at(
                    partial,
                    terminal_reward=terminal_rewards(
                        success=False, collision=True
                    )[0],
                    gamma=gamma,
                )
                offroad_return = _trajectory_return_at(
                    partial,
                    terminal_reward=DEFAULT_REWARD_V4_CONFIG.offroad_penalty,
                    gamma=gamma,
                )
                self.assertGreater(partial_return, stopped_return)
                self.assertLess(collision_return, stopped_return)
                self.assertLess(offroad_return, stopped_return)

    def test_control_rate_cost_prefers_smooth_commands_to_oscillation(self):
        smooth_commands = [0.5] * 100
        oscillating_commands = [0.5 if step % 2 == 0 else -0.5 for step in range(100)]

        def total_cost(commands):
            previous = 0.0
            result = 0.0
            for current in commands:
                longitudinal, steer, _, _ = control_rate_penalties(
                    current_longitudinal=current,
                    previous_longitudinal=previous,
                    current_steer=current,
                    previous_steer=previous,
                    timestep_s=0.1,
                )
                result += longitudinal + steer
                previous = current
            return result

        self.assertGreater(total_cost(smooth_commands), total_cost(oscillating_commands))


class RewardV4SafetyMathTests(unittest.TestCase):
    def test_static_side_obstacle_does_not_create_lidar_cost(self):
        current = [1.0, 0.2, 1.0]
        penalty, risk, _, _, closing_speed, gate = approach_gated_lidar_penalty(
            current,
            current,
            timestep_s=0.1,
            lidar_range_m=4.0,
        )
        self.assertEqual((penalty, risk, closing_speed, gate), (0.0, 0.0, 0.0, 0.0))

    def test_approaching_obstacle_activates_bounded_lidar_cost(self):
        penalty, risk, current_minimum, previous_minimum, closing_speed, gate = (
            approach_gated_lidar_penalty(
                [1.0, 0.2, 1.0],
                [1.0, 0.3, 1.0],
                timestep_s=0.1,
                lidar_range_m=4.0,
            )
        )
        self.assertLess(penalty, 0.0)
        self.assertGreater(risk, 0.0)
        self.assertAlmostEqual(current_minimum, 0.2)
        self.assertAlmostEqual(previous_minimum, 0.3)
        self.assertAlmostEqual(closing_speed, 4.0)
        self.assertEqual(gate, 1.0)

    def test_projected_ttc_uses_relative_velocity_not_speed_difference(self):
        approaching = projected_ttc_seconds(
            (3.0, 4.0, 0.0),
            (3.0, 4.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        separating = projected_ttc_seconds(
            (3.0, 4.0, 0.0),
            (0.0, 0.0, 0.0),
            (3.0, 4.0, 0.0),
        )
        self.assertAlmostEqual(approaching, 1.0)
        self.assertIsNone(separating)


if __name__ == "__main__":
    unittest.main()
