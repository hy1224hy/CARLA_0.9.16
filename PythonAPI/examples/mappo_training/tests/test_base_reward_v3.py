import itertools
import math
import unittest
from dataclasses import replace

from environment.base_reward_v3 import (
    DEFAULT_REWARD_V3_CONFIG,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
    REWARD_PROTOCOL_VERSION,
    RewardV3Error,
    control_rate_penalties,
    discounted_return,
    discounted_step_budget,
    is_legal_lane_change_marking,
    lane_deviation_penalty,
    lane_invasion_penalty,
    lateral_distance_to_lane_center,
    lidar_risk_penalty,
    overspeed_penalty,
    offroad_terminal_reward,
    potential_discounted_return,
    reward_v3_spec,
    route_completion,
    route_potential_reward,
    terminal_rewards,
    ttc_risk_penalty,
    validate_reward_v3_config,
)


GAMMA = 0.997


def _trajectory_return(
    completions,
    longitudinal,
    steer,
    *,
    terminal=None,
    config=DEFAULT_REWARD_V3_CONFIG,
):
    validate_reward_v3_config(config)
    if not (
        len(completions) == len(longitudinal) == len(steer)
        and len(completions) >= 2
    ):
        raise ValueError("trajectory arrays must be aligned state sequences")
    rewards = []
    last_transition = len(completions) - 2
    for index in range(len(completions) - 1):
        progress = route_potential_reward(
            completions[index],
            completions[index + 1],
            gamma=GAMMA,
            config=config,
        )
        long_penalty, steer_penalty, _, _ = control_rate_penalties(
            current_longitudinal=longitudinal[index + 1],
            previous_longitudinal=longitudinal[index],
            current_steer=steer[index + 1],
            previous_steer=steer[index],
            timestep_s=0.1,
            config=config,
        )
        collision = terminal == "collision" and index == last_transition
        success = terminal == "success" and index == last_transition
        collision_reward, success_reward = terminal_rewards(
            success=success,
            collision=collision,
            config=config,
        )
        rewards.append(
            math.fsum(
                (
                    progress,
                    long_penalty,
                    steer_penalty,
                    collision_reward,
                    success_reward,
                )
            )
        )
    return discounted_return(rewards, gamma=GAMMA)


def _linear_completion(final, transitions=100):
    return [final * index / transitions for index in range(transitions + 1)]


class RewardV3MathematicsTests(unittest.TestCase):
    def test_offroad_terminal_penalty_is_one_shot_and_collision_has_priority(self):
        self.assertEqual(
            offroad_terminal_reward(offroad=True, collision=False), -20.0
        )
        self.assertEqual(
            offroad_terminal_reward(offroad=True, collision=True), 0.0
        )
        self.assertEqual(
            offroad_terminal_reward(offroad=False, collision=False), 0.0
        )

    def test_lane_center_distance_is_per_lane_and_rotation_invariant(self):
        # 两条平行车道各自居中时都必须得到 0，而不能拿道路整体中线比较。
        self.assertEqual(
            lateral_distance_to_lane_center(
                vehicle_x=-195.09,
                vehicle_y=-64.58,
                lane_center_x=-195.09,
                lane_center_y=-64.58,
                lane_right_x=-1.0,
                lane_right_y=0.0,
            ),
            0.0,
        )
        self.assertEqual(
            lateral_distance_to_lane_center(
                vehicle_x=-191.59,
                vehicle_y=-64.58,
                lane_center_x=-191.59,
                lane_center_y=-64.58,
                lane_right_x=-1.0,
                lane_right_y=0.0,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            lateral_distance_to_lane_center(
                vehicle_x=2.0,
                vehicle_y=3.0,
                lane_center_x=1.0,
                lane_center_y=1.0,
                lane_right_x=0.0,
                lane_right_y=2.0,
            ),
            2.0,
        )

    def test_town05_legal_lane_change_requires_only_broken_markings(self):
        self.assertTrue(is_legal_lane_change_marking(["Broken"]))
        self.assertTrue(is_legal_lane_change_marking(["Broken", "Broken"]))
        self.assertFalse(is_legal_lane_change_marking([]))
        self.assertFalse(is_legal_lane_change_marking(["Solid"]))
        self.assertFalse(is_legal_lane_change_marking(["Broken", "Solid"]))
        self.assertFalse(is_legal_lane_change_marking(["NONE"]))

    def test_machine_spec_freezes_components_diagnostics_and_budget(self):
        spec = reward_v3_spec()
        self.assertEqual(spec["reward_protocol_version"], REWARD_PROTOCOL_VERSION)
        self.assertEqual(spec["components"], list(REWARD_COMPONENTS))
        self.assertEqual(spec["diagnostics"], list(REWARD_DIAGNOSTICS))
        self.assertEqual(spec["parameters"]["route_potential_scale"], 10.0)
        self.assertEqual(spec["parameters"]["success_bonus"], 10.0)
        self.assertEqual(spec["parameters"]["collision_penalty"], -20.0)
        self.assertIn(
            "forward_reverse_cycle_cannot_farm_progress",
            spec["anti_hacking_invariants"],
        )

    def test_route_completion_is_bounded_and_rejects_invalid_distance(self):
        self.assertEqual(route_completion(100.0, 100.0), 0.0)
        self.assertEqual(route_completion(100.0, 40.0), 0.6)
        self.assertEqual(route_completion(100.0, 0.0), 1.0)
        self.assertEqual(route_completion(100.0, 120.0), 0.0)
        with self.assertRaises(RewardV3Error):
            route_completion(0.0, 0.0)
        with self.assertRaises(RewardV3Error):
            route_completion(100.0, float("nan"))

    def test_discounted_progress_telescopes_and_cycle_cannot_farm(self):
        completions = [0.0, 0.35, 0.7, 0.2, 0.0]
        rewards = [
            route_potential_reward(a, b, gamma=GAMMA)
            for a, b in zip(completions, completions[1:])
        ]
        actual = discounted_return(rewards, gamma=GAMMA)
        expected = potential_discounted_return(
            completions[0],
            completions[-1],
            steps=len(rewards),
            gamma=GAMMA,
        )
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(actual, 0.0, places=12)

    def test_lingering_near_goal_is_worse_than_finishing_immediately(self):
        direct = _linear_completion(1.0, transitions=50)
        direct_return = _trajectory_return(
            direct,
            [0.5] * len(direct),
            [0.0] * len(direct),
            terminal="success",
        )
        delayed = _linear_completion(0.99, transitions=50) + [0.99] * 49 + [1.0]
        delayed_return = _trajectory_return(
            delayed,
            [0.5] * len(delayed),
            [0.0] * len(delayed),
            terminal="success",
        )
        self.assertGreater(direct_return, delayed_return)

    def test_throttle_and_brake_changes_have_symmetric_rate_cost(self):
        throttle = control_rate_penalties(
            current_longitudinal=0.5,
            previous_longitudinal=0.0,
            current_steer=0.0,
            previous_steer=0.0,
            timestep_s=0.1,
        )
        brake = control_rate_penalties(
            current_longitudinal=-0.5,
            previous_longitudinal=0.0,
            current_steer=0.0,
            previous_steer=0.0,
            timestep_s=0.1,
        )
        self.assertEqual(throttle, brake)
        self.assertLess(throttle[0], 0.0)

    def test_dense_costs_are_bounded_and_speed_has_no_positive_farm(self):
        config = DEFAULT_REWARD_V3_CONFIG
        for speed in (0.0, 1.0, config.overspeed_start_mps):
            penalty, risk = overspeed_penalty(speed)
            self.assertEqual((penalty, risk), (0.0, 0.0))
        penalty, risk = overspeed_penalty(100.0)
        self.assertEqual(penalty, -config.overspeed_weight)
        self.assertEqual(risk, 1.0)

        long_penalty, steer_penalty, long_cost, steer_cost = control_rate_penalties(
            current_longitudinal=1.0,
            previous_longitudinal=-1.0,
            current_steer=1.0,
            previous_steer=-1.0,
            timestep_s=0.1,
        )
        self.assertEqual(long_penalty, -config.longitudinal_rate_weight)
        self.assertEqual(steer_penalty, -config.steer_rate_weight)
        self.assertEqual((long_cost, steer_cost), (1.0, 1.0))

        lane_penalty, lane_risk = lane_deviation_penalty(100.0)
        self.assertEqual(lane_penalty, -config.lane_deviation_weight)
        self.assertEqual(lane_risk, 1.0)

        ttc_penalty, ttc_risk = ttc_risk_penalty(0.0)
        self.assertEqual(ttc_penalty, -config.ttc_risk_weight)
        self.assertEqual(ttc_risk, 1.0)

    def test_ttc_cost_is_continuous_at_registered_threshold(self):
        threshold = DEFAULT_REWARD_V3_CONFIG.ttc_threshold_s
        at_threshold = ttc_risk_penalty(threshold)
        just_below = ttc_risk_penalty(threshold - 1e-8)
        self.assertEqual(at_threshold, (0.0, 0.0))
        self.assertLess(abs(just_below[0]), 1e-8)
        self.assertLess(just_below[1], 1e-6)

    def test_lane_event_budget_uses_severity_not_sensor_list_length(self):
        self.assertEqual(lane_invasion_penalty(["Broken"]), -0.5)
        self.assertEqual(lane_invasion_penalty(["Solid", "Broken"]), -2.0)
        self.assertEqual(
            lane_invasion_penalty(["Curb"] * 100 + ["Solid"]),
            -4.0,
        )

    def test_invalid_budget_hierarchy_is_rejected(self):
        with self.assertRaisesRegex(RewardV3Error, "collision penalty"):
            validate_reward_v3_config(
                replace(DEFAULT_REWARD_V3_CONFIG, collision_penalty=-5.0)
            )
        with self.assertRaisesRegex(RewardV3Error, "lane penalties"):
            validate_reward_v3_config(
                replace(DEFAULT_REWARD_V3_CONFIG, curb_penalty=-0.1)
            )
        with self.assertRaisesRegex(RewardV3Error, "offroad penalty"):
            validate_reward_v3_config(
                replace(DEFAULT_REWARD_V3_CONFIG, offroad_penalty=-5.0)
            )
        with self.assertRaisesRegex(RewardV3Error, "collision_penalty"):
            validate_reward_v3_config(
                replace(
                    DEFAULT_REWARD_V3_CONFIG,
                    collision_penalty=float("nan"),
                )
            )
        with self.assertRaisesRegex(RewardV3Error, "curb_penalty"):
            validate_reward_v3_config(
                replace(
                    DEFAULT_REWARD_V3_CONFIG,
                    curb_penalty=float("-inf"),
                )
            )
        with self.assertRaisesRegex(RewardV3Error, "current command"):
            control_rate_penalties(
                current_longitudinal=1.01,
                previous_longitudinal=0.0,
                current_steer=0.0,
                previous_steer=0.0,
                timestep_s=0.1,
            )

    def test_lidar_budget_is_invariant_to_ray_count(self):
        short = lidar_risk_penalty([0.25, 1.0])
        duplicated = lidar_risk_penalty([0.25] + [1.0] * 100)
        self.assertEqual(short, duplicated)
        with self.assertRaises(RewardV3Error):
            lidar_risk_penalty([])


class RewardV3TrajectoryBudgetTests(unittest.TestCase):
    def _five_returns(self, config=DEFAULT_REWARD_V3_CONFIG):
        stopped_q = [0.0] * 101
        smooth_q = _linear_completion(0.8)
        success_q = _linear_completion(1.0)
        smooth_long = [0.5] * 101
        zero_steer = [0.0] * 101
        oscillatory_long = [0.5 if index % 2 == 0 else -0.5 for index in range(101)]
        oscillatory_steer = [0.3 if index % 2 == 0 else -0.3 for index in range(101)]
        return {
            "stopped": _trajectory_return(
                stopped_q,
                [0.0] * 101,
                zero_steer,
                config=config,
            ),
            "smooth": _trajectory_return(
                smooth_q,
                smooth_long,
                zero_steer,
                config=config,
            ),
            "oscillatory": _trajectory_return(
                smooth_q,
                oscillatory_long,
                oscillatory_steer,
                config=config,
            ),
            "collision": _trajectory_return(
                smooth_q,
                smooth_long,
                zero_steer,
                terminal="collision",
                config=config,
            ),
            "success": _trajectory_return(
                success_q,
                smooth_long,
                zero_steer,
                terminal="success",
                config=config,
            ),
        }

    def test_five_scripted_trajectory_ordering(self):
        returns = self._five_returns()
        self.assertGreater(returns["success"], returns["smooth"])
        self.assertGreater(returns["smooth"], returns["stopped"])
        self.assertGreater(returns["stopped"], returns["collision"])
        self.assertGreater(returns["smooth"], returns["oscillatory"])

    def test_late_collision_is_worse_than_safe_stopping_at_any_progress(self):
        for completion in (0.0, 0.25, 0.5, 0.8, 1.0):
            q = _linear_completion(completion)
            collision_return = _trajectory_return(
                q,
                [0.5] * len(q),
                [0.0] * len(q),
                terminal="collision",
            )
            with self.subTest(completion=completion):
                self.assertLess(collision_return, 0.0)

    def test_each_dense_cost_stays_below_reasonable_route_progress_budget(self):
        config = DEFAULT_REWARD_V3_CONFIG
        reasonable_completion_steps = 250
        progress_budget = potential_discounted_return(
            0.0,
            1.0,
            steps=reasonable_completion_steps,
            gamma=GAMMA,
            config=config,
        )
        individual_weights = {
            "overspeed": config.overspeed_weight,
            "longitudinal_rate": config.longitudinal_rate_weight,
            "steer_rate": config.steer_rate_weight,
            "lidar": config.lidar_risk_weight,
            "ttc": config.ttc_risk_weight,
            "lane_deviation": config.lane_deviation_weight,
        }
        for name, weight in individual_weights.items():
            cost_budget = discounted_step_budget(
                weight,
                steps=reasonable_completion_steps,
                gamma=GAMMA,
            )
            with self.subTest(component=name):
                self.assertLess(cost_budget, progress_budget)

    def test_ordering_survives_registered_plus_minus_25_percent_sensitivity(self):
        base = DEFAULT_REWARD_V3_CONFIG
        scales = (0.75, 1.25)
        for route, success, collision, smoothness in itertools.product(scales, repeat=4):
            config = replace(
                base,
                route_potential_scale=base.route_potential_scale * route,
                success_bonus=base.success_bonus * success,
                collision_penalty=base.collision_penalty * collision,
                longitudinal_rate_weight=(
                    base.longitudinal_rate_weight * smoothness
                ),
                steer_rate_weight=base.steer_rate_weight * smoothness,
            )
            returns = self._five_returns(config)
            with self.subTest(
                route=route,
                success=success,
                collision=collision,
                smoothness=smoothness,
            ):
                self.assertGreater(returns["success"], returns["smooth"])
                self.assertGreater(returns["smooth"], returns["stopped"])
                self.assertGreater(returns["stopped"], returns["collision"])
                self.assertGreater(returns["smooth"], returns["oscillatory"])


if __name__ == "__main__":
    unittest.main()
