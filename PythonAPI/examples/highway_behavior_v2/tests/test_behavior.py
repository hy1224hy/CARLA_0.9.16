import unittest
import numpy as np
from highway_behavior_v2.config import HybridConfig
from highway_behavior_v2.environment import HybridEnv
from highway_behavior_v2.straight.rewards import RewardState


def frame(**changes):
    row=dict(front=False,ell=0.,acceleration=0.,target_speed=15.,reference_speed=18.,
             attacker_speed=16.,control_cost=0.,thw=100.,gap=-10.,aligned=False,
             attacker_lane_changed=False,target_lane_stable=True,lateral=4.,
             lane_alignment=0.,collision=False,invalid=False,baseline_valid=True,
             spawn_overlap=False,speed_ratio=1.,normal_progress=0.)
    row.update(changes)
    return row


class RewardBehaviorTests(unittest.TestCase):
    def test_behind_vehicle_never_receives_positive_impact_reward(self):
        for level in (1,2):
            state=RewardState(HybridConfig(level=level))
            result=state.observe(frame(ell=.25,attacker_lane_changed=True),.2)
            self.assertLessEqual(result['components']['dense'],0.)
            self.assertLessEqual(result['components']['event'],0.)

    def test_l1_requires_stable_front_before_effect_payment(self):
        state=RewardState(HybridConfig(level=1))
        result=state.observe(frame(front=True,ell=.2,aligned=True,gap=25.,lateral=0.),.2)
        self.assertEqual(result['components']['dense'],0.)
        for _ in range(8):
            result=state.observe(frame(front=True,ell=.2,aligned=True,gap=25.,lateral=0.),.2)
        self.assertGreater(result['components']['dense'],0.)

    def test_l2_accepts_causal_cut_outside_old_danger_window(self):
        state=RewardState(HybridConfig(level=2))
        state.observe(frame(front=True,thw=2.,gap=30.,attacker_lane_changed=True,lateral=1.9),.2)
        for _ in range(3):
            state.observe(frame(front=True,thw=2.,gap=28.,target_speed=13.8,acceleration=-2.,lateral=0.,aligned=True),.2)
        self.assertEqual(state.cut_count,1)
        self.assertEqual(state.dangerous_cut_count,0)

    def test_existing_braking_is_not_new_cut_response(self):
        state=RewardState(HybridConfig(level=2))
        state.observe(frame(front=True,thw=.8,gap=12.,attacker_lane_changed=True,
                            target_speed=12.,acceleration=-4.,ell=.4,lateral=1.9),.2)
        for _ in range(8):
            state.observe(frame(front=True,thw=.8,gap=12.,target_speed=12.,acceleration=-4.,
                                ell=.4,lateral=0.,aligned=True),.2)
        self.assertEqual(state.cut_count,0)

    def test_remain_in_same_lane_cannot_farm_multiple_cuts(self):
        state=RewardState(HybridConfig(level=2))
        state.observe(frame(front=True,thw=.8,gap=12.,attacker_lane_changed=True,lateral=1.9),.2)
        for _ in range(15):
            state.observe(frame(front=True,thw=.8,gap=10.,target_speed=13.5,
                                acceleration=-2.,lateral=0.,aligned=True),.2)
        self.assertEqual(state.cut_count,1)

    def test_new_evasive_lane_change_counts_without_reference_lane_change(self):
        state=RewardState(HybridConfig(level=2))
        state.observe(frame(front=True,thw=2.,gap=30.,attacker_lane_changed=True,lateral=1.9),.2)
        state.observe(frame(front=False,target_lane_stable=False,reference_lane_stable=True,lateral=2.5),.2)
        self.assertEqual(state.cut_count,1)

    def test_reference_also_changing_lane_is_not_new_evasion(self):
        state=RewardState(HybridConfig(level=2))
        state.observe(frame(front=True,thw=2.,gap=30.,attacker_lane_changed=True,lateral=1.9),.2)
        state.observe(frame(front=False,target_lane_stable=False,reference_lane_stable=False,lateral=2.5),.2)
        self.assertEqual(state.cut_count,0)

    def test_repeat_cut_requires_stable_exit_and_fresh_response(self):
        state=RewardState(HybridConfig(level=2))
        for _ in range(2):
            for _ in range(4):state.observe(frame(lateral=4.,target_speed=15.),.2)
            state.observe(frame(front=True,thw=1.,gap=15.,attacker_lane_changed=True,lateral=1.9),.2)
            for _ in range(3):
                state.observe(frame(front=True,thw=1.,gap=12.,target_speed=13.5,
                                    acceleration=-2.,lateral=0.,aligned=True),.2)
        self.assertEqual(state.cut_count,2)
        self.assertEqual(state.dangerous_cut_count,2)


class LaneBudgetTests(unittest.TestCase):
    def test_budget_and_cooldown_are_observed_and_masked(self):
        env=HybridEnv(HybridConfig(level=1,l1_max_lane_changes=2))
        try:
            obs,_=env.reset(seed=2,options={'family':'follow'})
            self.assertEqual(obs['state'].shape,(env.config.state_dim,))
            self.assertGreater(env.config.state_dim,28)
            env._lane_changes_started=2
            np.testing.assert_array_equal(env.lane_mask(),[True,False,False])
            env._lane_changes_started=0;env._cooldown_until=env.tracker.time+2
            np.testing.assert_array_equal(env.lane_mask(),[True,False,False])
            self.assertTrue(env.target.enable_lane_change)
        finally:env.close()

    def test_cooldown_starts_after_completed_maneuver(self):
        env=HybridEnv(HybridConfig(level=1))
        try:
            env.reset(seed=0,options={'family':'follow'})
            env.step(dict(acceleration=np.array([0.]),lane=2))
            for _ in range(35):
                _,_,done,_,info=env.step(dict(acceleration=np.array([0.]),lane=0))
                if done:break
                if not info['lane_busy']:
                    self.assertGreater(env._cooldown_until,env.tracker.time)
                    np.testing.assert_array_equal(env.lane_mask(),[True,False,False])
                    return
            self.fail('Maneuver did not settle')
        finally:env.close()


if __name__=='__main__':unittest.main()
