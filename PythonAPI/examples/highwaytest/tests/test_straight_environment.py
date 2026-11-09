import unittest
import numpy as np
from highwaytest.straight.config import StraightConfig
from highwaytest.straight.environment import StraightEnv,scripted_action
from highwaytest.straight.rewards import RewardState


class RewardTests(unittest.TestCase):
    def test_risk_weight_scales_geometry_terms_and_ablates_them(self):
        frame=dict(ell=0.,front=True,thw=.5,acceleration=0.,target_speed=18.,reference_speed=18.,
            attacker_speed=18.,control_cost=0.,collision=False,invalid=False,baseline_valid=True,
            attacker_lane_changed=False,target_lane_stable=True)
        for level in (2,3):
            full=RewardState(StraightConfig(level=level,risk_weight=1.))
            off=RewardState(StraightConfig(level=level,risk_weight=0.))
            full.geometric_cut=True;off.geometric_cut=True
            self.assertGreater(full.observe(dict(frame),1/15)['reward'],
                               off.observe(dict(frame),1/15)['reward'],level)

    def test_risk_weight_validation_and_eval_cases_default(self):
        with self.assertRaises(ValueError):StraightConfig(risk_weight=-.1).validate()
        self.assertEqual(StraightConfig().risk_weight,1.)
        self.assertEqual(StraightConfig().eval_cases,32)

    def test_collision_sign_priority_and_single_payment(self):
        for level in (1,2,3):
            state=RewardState(StraightConfig(level=level))
            result=state.observe(dict(collision=True),1/15)
            self.assertEqual(result['reward'],20 if level==3 else -20)
            self.assertEqual(result['success'],level==3)
            self.assertTrue(result['done'])
            with self.assertRaises(RuntimeError):state.observe({},1/15)
        result=RewardState(StraightConfig(level=3)).observe(dict(collision=True,invalid=True),1/15)
        self.assertEqual(result['reward'],-20)
        self.assertFalse(result['success'])

    def test_mild_success_and_grade_violation(self):
        cfg=StraightConfig(level=1,duration=2)
        state=RewardState(cfg)
        for _ in range(30):
            result=state.observe(dict(ell=.2,front=True),1/15)
        self.assertTrue(result['success'])
        self.assertGreater(state.blocking_time,1.5)
        state=RewardState(StraightConfig(level=1))
        for _ in range(8):
            result=state.observe(dict(acceleration=-4),1/15)
            if result['done']:break
        self.assertTrue(result['grade_violation'])

    def test_mild_cannot_bank_bonus_then_pursue_severe_delay(self):
        state=RewardState(StraightConfig(level=1))
        for _ in range(24):state.observe(dict(ell=.2,front=True),1/15)
        for _ in range(15):result=state.observe(dict(ell=.6,front=True),1/15)
        self.assertTrue(result['grade_violation'])
        self.assertFalse(result['success'])
        self.assertEqual(result['components']['success'],0.)

    def test_failed_cut_can_retry_and_target_motion_cancels_pending(self):
        state=RewardState(StraightConfig(level=2))
        for index in range(6):
            state.observe(dict(attacker_lane_changed=index==0,target_lane_stable=True,
                aligned=True,front=True,thw=2),1/15)
        self.assertIsNone(state.candidate_time)
        for index in range(6):
            state.observe(dict(attacker_lane_changed=index==0,target_lane_stable=True,
                aligned=True,front=True,thw=.7),1/15)
        self.assertIsNotNone(state.candidate_time)
        state.observe(dict(target_lane_stable=False,reference_speed=18,target_speed=10),1/15)
        self.assertFalse(state.cut_in)
        for index in range(6):
            state.observe(dict(attacker_lane_changed=index==0,target_lane_stable=True,
                aligned=True,front=True,thw=.7),1/15)
        state.observe(dict(reference_speed=18,target_speed=10),1/15)
        self.assertTrue(state.cut_in)

    def test_target_lane_change_is_not_attacker_cut(self):
        state=RewardState(StraightConfig(level=2))
        for _ in range(15):
            result=state.observe(dict(front=True,thw=.7,aligned=True,
                attacker_lane_changed=False,target_lane_stable=False,ell=.3),1/15)
        self.assertFalse(state.geometric_cut)
        self.assertFalse(state.cut_in)

class EnvironmentTests(unittest.TestCase):
    def test_scripted_goal_reachability_and_driver_demonstrates_lane_change(self):
        for level,kind,family in ((1,'block','follow'),(2,'cut','adjacent'),(3,'collision','adjacent')):
            env=StraightEnv(StraightConfig(level=level))
            env.reset(seed=0,options={'family':family})
            while not env.ended:
                _,_,_,_,info=env.step(scripted_action(env,kind))
            self.assertTrue(info['success'],(level,info))
            self.assertTrue(info['baseline_valid'])
        env=StraightEnv(StraightConfig(level=0))
        env.reset(seed=0,options={'demonstration':True})
        lanes=set()
        while not env.ended:
            _,_,_,_,info=env.step(env.normal_action())
            lanes.add(env._lane(env.attacker))
        self.assertEqual(lanes,{0,1})
        self.assertTrue(info['success']);self.assertFalse(info['invalid'])

    def test_shapes_reproducibility_and_reference_isolation(self):
        cfg=StraightConfig(duration=2)
        left,right=StraightEnv(cfg),StraightEnv(cfg)
        obs,info=left.reset(seed=42,options={'family':'adjacent'})
        obs2,_=right.reset(seed=42,options={'family':'adjacent'})
        np.testing.assert_array_equal(obs,obs2)
        self.assertEqual(obs.shape,(22,))
        self.assertEqual(len(left.road.vehicles),2)
        self.assertTrue(left.target.enable_lane_change)
        for _ in range(4):
            left.step(np.array([-.1,0.]));right.step(np.array([.2,.1]))
        np.testing.assert_allclose(left.reference.vehicles[1].position,
                                   right.reference.vehicles[1].position)
        original=left._observation().copy()
        left.reference.vehicles[1].speed=1.
        np.testing.assert_array_equal(original,left._observation())
        left.close();right.close()

    def test_finite_horizon_true_terminal_and_action_validation(self):
        env=StraightEnv(StraightConfig(level=0,duration=.4))
        env.reset(seed=4)
        with self.assertRaises(ValueError):env.step([float('nan'),0])
        for _ in range(2):obs,reward,term,trunc,info=env.step([0.,0.])
        self.assertTrue(term);self.assertFalse(trunc)
        self.assertEqual(info['episode_steps'],2)
        self.assertTrue(np.isfinite(obs).all())
        with self.assertRaises(RuntimeError):env.step([0.,0.])

    def test_spawn_overlap_collision_is_invalid_not_success(self):
        env=StraightEnv(StraightConfig(level=3))
        env.reset(seed=1,options={'family':'follow'})
        env.attacker.position[:]=[100.,0.];env.target.position[:]=[104.,0.]
        env.attacker.speed=env.target.speed=0.
        self.assertTrue(env._overlap(env.attacker,env.target))
        env.spawn_overlap=True
        _,reward,term,_,info=env.step([0.,0.])
        self.assertFalse(info['success']);self.assertTrue(info['invalid'])
        self.assertLess(reward,0)

    def test_rear_end_collision_is_valid_l3_success(self):
        env=StraightEnv(StraightConfig(level=3))
        env.reset(seed=1,options={'family':'follow'})
        env.attacker.position[:]=[99.,0.];env.target.position[:]=[104.,0.]
        env.attacker.speed=25.;env.target.speed=15.
        self.assertFalse(env.spawn_overlap)
        _,reward,term,_,info=env.step([0.,0.])
        self.assertTrue(info['target_collision']);self.assertTrue(info['success'])
        self.assertGreater(reward,19)
        self.assertEqual(info['contact_orientation'],'longitudinal')

    def test_descriptors_reported(self):
        env=StraightEnv(StraightConfig(level=2))
        env.reset(seed=0,options={'family':'adjacent'})
        while not env.ended:
            _,_,_,_,info=env.step(scripted_action(env,'cut'))
        for key in ('max_target_brake','avg_speed_ratio','cut_count','cut_first_thw','stop_qualified','contact_orientation','spawn_overlap'):
            self.assertIn(key,info)
        self.assertGreaterEqual(info['cut_count'],1)
        self.assertIsNotNone(info['cut_first_thw'])
        self.assertEqual(info['contact_orientation'],'none')
        self.assertFalse(info['spawn_overlap'])
        self.assertTrue(np.isfinite(info['avg_speed_ratio']))
        self.assertLessEqual(info['max_target_brake'],0.)

    def test_level_potentials_guide_task_geometry(self):
        env=StraightEnv(StraightConfig(level=1))
        env.reset(seed=0,options={'family':'follow'})
        env.target.position[:]=[100.,0.];env.target.speed=18.
        env.attacker.position[:]=[108.,0.]
        phi_front=env._phi()
        env.attacker.position[:]=[92.,0.]
        self.assertEqual(env._phi(),0.,'L1 potential must be zero behind the target')
        self.assertGreater(phi_front,0.)
        env.attacker.position[:]=[108.,4.]
        self.assertLess(env._phi(),phi_front,'L1 potential lower in the adjacent lane')
        env2=StraightEnv(StraightConfig(level=2))
        env2.reset(seed=0,options={'family':'follow'})
        env2.target.position[:]=[100.,0.];env2.target.speed=18.
        env2.attacker.position[:]=[94.,4.]
        phi_stage=env2._phi()
        env2.attacker.position[:]=[140.,4.]
        self.assertGreater(phi_stage,env2._phi(),'L2 potential peaks at the adjacent staging position')
        env2.attacker.position[:]=[94.,0.]
        self.assertLess(env2._phi(),phi_stage,'L2 potential lower in the target lane')
        env3=StraightEnv(StraightConfig(level=3))
        env3.reset(seed=0,options={'family':'follow'})
        env3.target.position[:]=[100.,0.];env3.target.speed=18.
        env3.attacker.position[:]=[108.,0.]
        close=env3._phi()
        env3.attacker.position[:]=[160.,0.]
        self.assertGreater(close,env3._phi(),'L3 potential grows with proximity')

    def test_geometry_potential_enters_reward_components(self):
        env=StraightEnv(StraightConfig(level=1,duration=.6,geometry_weight=.5))
        env.reset(seed=0,options={'family':'follow'})
        env.target.position[:]=[100.,0.];env.target.speed=18.
        env.attacker.position[:]=[115.,0.];env.attacker.speed=10.
        phi0=env._phi()
        _,reward,_,_,info=env.step([0.,0.])
        self.assertIn('geometry',info['reward_components'])
        expected=.5*(env.config.gamma*env._phi()-phi0)
        self.assertAlmostEqual(info['reward_components']['geometry'],expected,places=6)
        zero=StraightEnv(StraightConfig(level=1,duration=.6,geometry_weight=0.))
        zero.reset(seed=0,options={'family':'follow'})
        _,_,_,_,info0=zero.step([0.,0.])
        self.assertEqual(info0['reward_components']['geometry'],0.)

    def test_actual_pair_contact_and_native_idm_response(self):
        env=StraightEnv(StraightConfig(level=3))
        env.reset(seed=2,options={'family':'follow'})
        env.attacker.position[:]=[130.,0.];env.target.position[:]=[110.,0.]
        env.attacker.speed=5;env.target.speed=18
        env.target.act()
        self.assertLess(env.target.action['acceleration'],0)

if __name__=='__main__':unittest.main()
