"""Behavioral regressions for the reward-driven PPO repair."""
import unittest
import numpy as np
from highwaytest.straight.config import StraightConfig
from highwaytest.straight.environment import StraightEnv
from highwaytest.straight.rewards import RewardState


class RewardRepairTests(unittest.TestCase):
    def test_approach_is_monotone_through_both_lane_directions(self):
        for level in (2, 3):
            env = StraightEnv(StraightConfig(level=level))
            env.reset(seed=0, options={'family': 'adjacent'})
            for target_y, positions in ((0., [4., 3., 2., 1., 0.]),
                                        (4., [0., 1., 2., 3., 4.])):
                env.target.position[:] = [100., target_y]
                env.target.speed = 18.
                values = []
                for y in positions:
                    env.attacker.position[:] = [119.4, y]
                    env.attacker.heading = env.target.heading = 0.
                    values.append(env._phi())
                self.assertTrue(all(b > a for a, b in zip(values, values[1:])), values)

    def test_default_crossing_does_not_pay_without_a_completed_event(self):
        state = RewardState(StraightConfig(level=2))
        for _ in range(10):
            state.observe(dict(attacker_lane_changed=True, target_lane_stable=True), 1/15)
        self.assertEqual(state.components['event'], 0.)

    def test_additional_late_cut_does_not_erase_an_already_held_cut(self):
        state = RewardState(StraightConfig(level=2, duration=4.))
        def cut():
            for i in range(6):
                state.observe(dict(attacker_lane_changed=i == 0,
                    target_lane_stable=True, aligned=True, front=True, thw=.7), 1/15)
            state.observe(dict(reference_speed=18., target_speed=10.), 1/15)
        cut()
        while state.time < 3.1:
            state.observe({}, 1/15)
        cut()
        while not state.done:
            state.observe({}, 1/15)
        self.assertTrue(state.success)
        self.assertEqual(state.cut_count, 2)

    def test_dense_approach_exists_before_any_cut(self):
        cfg = StraightConfig(level=2, geometry_weight=0.)
        env = StraightEnv(cfg)
        env.reset(seed=2, options={'family': 'adjacent'})
        env.attacker.position[:] = [119.4, 4.]
        env.target.position[:] = [100., 0.]
        env.attacker.speed = env.target.speed = 18.
        _, _, _, _, info = env.step([0., 0.])
        self.assertGreater(info['reward_components']['approach'], 0.)
        self.assertFalse(info['cut_in'])


if __name__ == '__main__':
    unittest.main()
