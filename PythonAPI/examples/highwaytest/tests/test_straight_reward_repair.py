"""Behavioral regressions for the reward-driven PPO repair."""
import unittest
import numpy as np
import torch
from highwaytest.straight.config import StraightConfig
from highwaytest.straight.environment import StraightEnv
from highwaytest.straight.rewards import RewardState


class RewardRepairTests(unittest.TestCase):
    def test_tracking_cost_has_no_positive_waiting_wage_and_balances_motion(self):
        from highwaytest.straight.rewards import tracking_score
        self.assertEqual(tracking_score(0.,.7),0.)
        self.assertLess(tracking_score(2.,.7),0.)
        lateral_gain=tracking_score(1.5,.7)-tracking_score(2.,.7)
        speed_noise=abs(tracking_score(.5,6.))
        self.assertGreater(lateral_gain,10*speed_noise)

    def test_cost_mode_penalizes_incomplete_horizon_without_changing_success(self):
        cfg=StraightConfig(level=3,duration=.2,guidance_mode='tracking_cost',guidance_weight=6.)
        state=RewardState(cfg)
        for _ in range(3):result=state.observe({},1/15)
        self.assertFalse(result['success'])
        self.assertEqual(result['components']['failure'],-2.)

    def test_preparation_milestone_cannot_be_farmed_by_repeated_phase_entries(self):
        state=RewardState(StraightConfig(level=3,guidance_mode='tracking_cost'))
        for _ in range(12):state.observe(dict(preparation_complete=True),1/15)
        self.assertEqual(state.components['milestone'],.5)
        self.assertEqual(state.components['event'],0.)

    def test_normal_actor_expansion_preserves_all_existing_outputs(self):
        from highwaytest.straight.model import ActorCritic
        from highwaytest.straight.repair_study import expand_normal_observation
        torch.manual_seed(42)
        model=ActorCritic(obs_dim=24)
        state=dict(config=StraightConfig(level=0).to_dict(),model=model.state_dict(),
                   actor=model.actor.state_dict())
        expanded=expand_normal_observation(state,26)
        new=ActorCritic(obs_dim=26)
        new.load_state_dict(expanded['model'])
        obs=torch.randn(16,24)
        with torch.no_grad():
            expected=model.actor(obs)
            actual=new.actor(torch.cat([obs,torch.randn(16,2)],dim=-1))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

    def test_task_guidance_distinguishes_depart_cut_and_contact(self):
        env = StraightEnv(StraightConfig(level=2, obs_dim=26))
        obs, _ = env.reset(seed=2, options={'family': 'follow'})
        self.assertEqual(obs.shape, (26,))
        self.assertFalse(env._l2_ready)
        self.assertEqual(env._guidance()['goal_y'], 4.)
        env.attacker.position[1] = 4.
        env._update_task_phase()
        self.assertTrue(env._l2_ready)
        self.assertEqual(env._guidance()['goal_y'], 0.)
        env = StraightEnv(StraightConfig(level=3, obs_dim=26))
        env.reset(seed=2, options={'family': 'follow'})
        self.assertEqual(env._guidance()['goal_y'], 4.)
        env.attacker.position[0] = env.target.position[0] + 1.
        env._update_task_phase()
        self.assertEqual(env._guidance()['goal_y'], 0.)

    def test_lateral_motion_toward_target_earns_local_feedback(self):
        env = StraightEnv(StraightConfig(level=2, obs_dim=26))
        env.reset(seed=2, options={'family': 'adjacent'})
        env.attacker.position[:] = [119.4, 4.]
        env.attacker.heading = 0.
        stationary = env._guidance()['lateral_quality']
        env.attacker.heading = -.05
        moving = env._guidance()['lateral_quality']
        self.assertGreater(moving, stationary)

    def test_l1_loss_in_other_lane_cannot_collect_impact_reward(self):
        state = RewardState(StraightConfig(level=1, approach_weight=0.))
        result = state.observe(dict(ell=.2, front=False), 1/15)
        self.assertEqual(result['components']['dense'], 0.)

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
