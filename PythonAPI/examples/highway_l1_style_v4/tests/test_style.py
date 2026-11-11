import unittest
from highway_l1_style_v4.config import HybridConfig
from highway_l1_style_v4.environment import HybridEnv
from highway_l1_style_v4.straight.rewards import RewardState
from highway_l1_style_v4.runner import rank


def frame(**changes):
    row=dict(front=True,aligned=True,gap=35.,thw=2.,ell=0.,acceleration=0.,target_speed=18.,reference_speed=18.,
        attacker_speed=15.3,control_cost=0.,attacker_lane=0,attacker_y=0.,attacker_heading=0.,
        attacker_lane_changed=False,target_lane_stable=True,lateral=0.,lane_alignment=1.,
        collision=False,invalid=False,baseline_valid=True,spawn_overlap=False,speed_ratio=.85,red_delay_m=0.)
    row.update(changes);return row


class StyleTests(unittest.TestCase):
    def test_selection_prefers_style_on_joint_success_tie(self):
        report=dict(baseline_valid_rate=1.,invalid_rate=0.,collision_rate=0.,grade_violation_rate=0.,
            attack_success_rate=0.,style_success_rate=.8,utility=0.)
        other=dict(report,style_success_rate=0.,utility=.5)
        self.assertGreater(rank(report,1),rank(other,1))
    def test_preparation_must_precede_slow_style_payment(self):
        state=RewardState(HybridConfig())
        for _ in range(8):result=state.observe(frame(front=False,aligned=False,gap=-15.,lateral=4.),.2)
        self.assertFalse(state.style_active)
        self.assertEqual(result['components']['style'],0.)

    def test_target_evades_but_style_lane_remains_latched(self):
        state=RewardState(HybridConfig())
        for _ in range(3):state.observe(frame(),.2)
        self.assertTrue(state.style_active)
        result=state.observe(frame(front=False,aligned=False,lateral=4.,target_lane_stable=False),.2)
        self.assertEqual(state.style_lane,0)
        self.assertGreater(result['components']['style'],0.)
        self.assertEqual(result['components']['impact'],0.)

    def test_style_is_distinct_from_traffic_impact_success(self):
        state=RewardState(HybridConfig())
        for i in range(100):
            state.observe(frame(front=i<3,aligned=i<3,lateral=0. if i<3 else 4.),.2)
        self.assertTrue(state.style_success)
        self.assertFalse(state.impact_success)
        self.assertFalse(state.success)

    def test_style_and_measurable_impact_jointly_succeed(self):
        state=RewardState(HybridConfig())
        for _ in range(100):state.observe(frame(ell=.1,target_speed=16.2,red_delay_m=10.),.2)
        self.assertTrue(state.style_success)
        self.assertTrue(state.impact_success)
        self.assertTrue(state.success)

    def test_parking_does_not_receive_style_reward(self):
        state=RewardState(HybridConfig())
        for _ in range(3):state.observe(frame(),.2)
        result=state.observe(frame(attacker_speed=0.),.2)
        self.assertEqual(result['components']['style'],0.)
        self.assertLess(result['reward'],0.)

    def test_collision_cannot_be_style_or_attack_success(self):
        state=RewardState(HybridConfig())
        for _ in range(99):state.observe(frame(ell=.1),.2)
        result=state.observe(frame(ell=.1,collision=True),.2)
        self.assertFalse(state.style_success)
        self.assertFalse(state.success)
        self.assertEqual(result['components']['failure'],-20.)

    def test_severe_target_loss_still_cannot_earn_more_impact_reward(self):
        values=[]
        for loss in [.1,.5]:
            state=RewardState(HybridConfig())
            for _ in range(3):result=state.observe(frame(ell=loss),.2)
            values.append(result['components']['impact'])
        self.assertAlmostEqual(*values)

    def test_phase_and_accumulated_style_state_are_observed(self):
        env=HybridEnv(HybridConfig())
        try:
            obs,_=env.reset(seed=0,options={'family':'follow'})
            self.assertEqual(obs['state'].shape,(env.config.state_dim,))
            self.assertGreater(env.config.state_dim,45)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(env.target.enable_lane_change)
            env.tracker.style_active=True;env.tracker.style_lane=0
            env.target.position[1]=4.
            self.assertEqual(env._task_targets()['goal_y'],0.)
        finally:env.close()


if __name__=='__main__':unittest.main()
