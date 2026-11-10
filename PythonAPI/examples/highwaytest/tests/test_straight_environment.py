import unittest
import numpy as np
from highwaytest.straight.config import StraightConfig
from highwaytest.straight.environment import StraightEnv,scripted_action
from highwaytest.straight.rewards import RewardState
from highwaytest.straight.environment import GRID


class RewardTests(unittest.TestCase):
    def test_risk_weight_scales_geometry_terms_and_ablates_them(self):
        frame=dict(ell=0.,front=True,thw=.5,lateral=.5,acceleration=0.,target_speed=18.,reference_speed=18.,
            attacker_speed=18.,control_cost=0.,collision=False,invalid=False,baseline_valid=True,
            attacker_lane_changed=False,target_lane_stable=True)
        full=RewardState(StraightConfig(level=2,risk_weight=1.))
        off=RewardState(StraightConfig(level=2,risk_weight=0.))
        self.assertGreater(full.observe(dict(frame),1/15)['reward'],
                           off.observe(dict(frame),1/15)['reward'],2)

    def test_risk_weight_validation_and_eval_cases_default(self):
        with self.assertRaises(ValueError):StraightConfig(risk_weight=-.1).validate()
        self.assertEqual(StraightConfig().risk_weight,1.)
        self.assertEqual(StraightConfig().eval_cases,32)

    def test_l1_reward_peaks_in_mild_band(self):
        def dense(ell):
            return RewardState(StraightConfig(level=1)).observe(dict(ell=ell,front=True),1/15)['reward']
        self.assertGreater(dense(.25),dense(.5),'mild band must beat stronger slowdown')
        self.assertAlmostEqual(dense(.6),dense(0.),places=6,msg='zero outside the band')
        self.assertGreater(dense(.02),dense(0.),'discovery slope must start at ell=0')
        self.assertGreater(dense(.25),dense(.02),'band guides from below')

    def test_h_band_matches_success_window(self):
        def h(thw):
            state=RewardState(StraightConfig(level=2));state.geometric_cut=True
            return state.observe(dict(thw=thw,front=True,ell=0.),1/15)['reward']
        self.assertGreater(h(.7),h(.1),'in-window beats below-window')
        self.assertGreater(h(.7),h(1.4),'in-window beats above-window')
        self.assertAlmostEqual(h(.1),0.,places=6)
        self.assertAlmostEqual(h(1.4),0.,places=6)

    def test_l2_cut_success_after_one_qualified_cut_and_repeats_earn_more(self):
        cfg=StraightConfig(level=2,duration=4.)
        def do_cut(state):
            for i in range(6):
                state.observe(dict(attacker_lane_changed=i==0,target_lane_stable=True,aligned=True,front=True,thw=.7),1/15)
            state.observe(dict(reference_speed=18.,target_speed=10.),1/15)
        state=RewardState(cfg)
        do_cut(state)
        self.assertEqual(state.cut_count,1)
        self.assertAlmostEqual(state.components['event'],2.5,'cut event + one lane-cross bonus')
        while not state.done:
            state.observe(dict(),1/15)
        self.assertTrue(state.success,'E2 needs ONE qualified cut plus 1s clean, not two')
        self.assertAlmostEqual(state.components['success'],5.)
        # Repetition is encouraged via per-cut event rewards, not the success gate.
        state2=RewardState(cfg)
        do_cut(state2);do_cut(state2)
        self.assertEqual(state2.cut_count,2)
        self.assertAlmostEqual(state2.components['event'],5.,'two cuts plus two capped crossing bonuses')

    def test_cut_chain_diagnostics_record_gate_failures(self):
        state=RewardState(StraightConfig(level=2))
        # Attempt 1: geometric cut outside the THW window -> no reaction check at all.
        for i in range(6):
            state.observe(dict(attacker_lane_changed=i==0,target_lane_stable=True,aligned=True,front=True,thw=2.),1/15)
        for _ in range(16):
            state.observe(dict(),1/15)
        self.assertEqual(state.diag['geometric'],1)
        self.assertEqual(state.diag['thw_miss'],1)
        # Attempt 2: THW-qualified cut but the target never reacts -> reaction miss.
        for i in range(6):
            state.observe(dict(attacker_lane_changed=i==0,target_lane_stable=True,aligned=True,front=True,thw=.7),1/15)
        for _ in range(16):
            state.observe(dict(),1/15)
        self.assertEqual(state.diag['thw_ok'],1)
        self.assertEqual(state.diag['react_miss'],1)
        self.assertEqual(state.diag['react_hit'],0)
        self.assertEqual(state.cut_count,0,'no reaction means no qualified cut')



    def test_lane_cross_bonus_capped_and_directional(self):
        cfg=StraightConfig(level=2)
        state=RewardState(cfg)
        def cross():
            return state.observe(dict(attacker_lane_changed=True,target_lane_stable=True,
                                      aligned=True,front=True,thw=.7),1/15)['reward']
        r1=cross();r2=cross();r3=cross()
        self.assertAlmostEqual(r1-r3,.5*cfg.event_weight,places=4,msg='first crossing into the target lane earns the bonus')
        self.assertAlmostEqual(r2-r3,.5*cfg.event_weight,places=4,msg='second crossing still earns it')
        self.assertLess(r3,.01,'third crossing earns no bonus (dense residue only)')
        # Crossing back out is not an attacker_lane_changed-into-target-lane event: no bonus path exists.
        self.assertEqual(state.lane_cross_count,3)

    def test_l2_lateral_terms_valley_at_the_lane_line(self):
        # The lane boundary must not harvest lateral rewards (line-hugging exploit).
        env=StraightEnv(StraightConfig(level=2))
        env.reset(seed=0,options={'family':'follow'})
        env.target.position[:]=[100.,0.];env.target.speed=18.
        env.attacker.speed=18.
        env.attacker.position[:]=[114.,2.]  # exactly on the line
        line=env._phi()
        env.attacker.position[:]=[114.,1.]
        near=env._phi()
        env.attacker.position[:]=[114.,3.]
        far_side=env._phi()
        env.attacker.position[:]=[114.,0.]
        in_lane=env._phi()
        self.assertLess(line,near,'the line must be a reward valley')
        self.assertLess(line,far_side)
        self.assertGreater(in_lane,near,'the target lane must beat the approach')
        self.assertGreater(far_side,0.,'gradient from the adjacent side must survive')

    def test_l3_crossing_bonus_and_potential_valley(self):
        cfg=StraightConfig(level=3)
        state=RewardState(cfg)
        frame=dict(attacker_lane_changed=True,target_lane_stable=True,aligned=True,front=True,thw=.7)
        r1=state.observe(dict(frame),1/15)['reward']
        r2=state.observe(dict(frame),1/15)['reward']
        r3=state.observe(dict(frame),1/15)['reward']
        self.assertGreater(r1-r3,.4,'L3 crossing bonus pays')
        self.assertGreater(r2-r3,.4)
        self.assertLess(r3,.01,'capped after two crossings')
        env=StraightEnv(StraightConfig(level=3))
        env.reset(seed=0,options={'family':'follow'})
        env.target.position[:]=[100.,0.];env.target.speed=18.
        env.attacker.speed=18.
        env.attacker.position[:]=[103.,2.]
        line=env._phi()
        env.attacker.position[:]=[103.,0.]
        in_lane=env._phi()
        env.attacker.position[:]=[103.,4.]
        adjacent=env._phi()
        self.assertLess(line,in_lane,'the line must stay a valley for L3 too')
        self.assertLess(line,adjacent)
        self.assertGreater(in_lane,adjacent,'the target lane must dominate')

    def test_align_thresholds_are_config_knobs(self):
        relaxed=RewardState(StraightConfig(align_duration=.3,align_lateral=.75,align_heading=15.))
        for i in range(5):
            relaxed.observe(dict(attacker_lane_changed=i==0,target_lane_stable=True,aligned=True,front=True,thw=2),1/15)
        self.assertTrue(relaxed.geometric_cut,'5 frames (0.33s) must confirm with align_duration=0.3')
        strict=RewardState(StraightConfig())
        for i in range(5):
            strict.observe(dict(attacker_lane_changed=i==0,target_lane_stable=True,aligned=True,front=True,thw=2),1/15)
        self.assertFalse(strict.geometric_cut,'default 0.4s must not confirm in 5 frames')
        with self.assertRaises(ValueError):StraightConfig(align_duration=0.).validate()

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
            result=state.observe(dict(acceleration=-4,ell=.5),1/15)
            if result['done']:break
        self.assertTrue(result['grade_violation'])

    def test_hard_brake_alone_is_not_a_grade_violation(self):
        # IDM jerk can brake below -3 briefly without any real speed loss;
        # the violation must co-occur with an actual strong loss (ell above band).
        state=RewardState(StraightConfig(level=1))
        for _ in range(10):
            result=state.observe(dict(acceleration=-4,ell=.05),1/15)
        self.assertFalse(result['grade_violation'])
        self.assertFalse(result['done'])

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
        self.assertEqual(obs.shape,(24,))
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


    def test_potentials_positive_only_in_attack_regions(self):
        env = StraightEnv(StraightConfig(level=1))
        env.reset(seed=0, options={'family': 'follow'})
        env.target.position[:] = [100., 0.]
        env.target.speed = 18.
        env.attacker.position[:] = [110., 0.]  # in front, gap 5, same lane
        env.attacker.speed = 14.
        self.assertGreater(env._phi(), 0., 'L1 potential must guide the front blocking band')
        env2 = StraightEnv(StraightConfig(level=2))
        env2.reset(seed=0, options={'family': 'follow'})
        env2.target.position[:] = [100., 0.]
        env2.target.speed = 18.
        env2.attacker.position[:] = [112., 0.]  # in lane, front, THW ~0.7
        env2.attacker.speed = 18.
        self.assertGreater(env2._phi(), 0., 'L2 potential must guide the post-cut band')
        env3 = StraightEnv(StraightConfig(level=3))
        env3.reset(seed=0, options={'family': 'follow'})
        env3.target.position[:] = [100., 0.]
        env3.target.speed = 18.
        env3.attacker.position[:] = [103., 0.]
        env3.attacker.speed = 18.
        self.assertGreater(env3._phi(), 0., 'L3 potential must guide in-lane proximity')



    def test_l1_triangular_preference(self):
        def dense(ell):
            return RewardState(StraightConfig(level=1)).observe(dict(ell=ell,front=True),1/15)['reward']
        self.assertGreater(dense(.2),dense(.1),'preference rises toward the peak')
        self.assertGreater(dense(.2),dense(.3),'preference falls past the peak')
        self.assertAlmostEqual(dense(.35),dense(.5),places=6,msg='zero at and above the mild boundary')
        self.assertGreater(dense(.05),dense(0.),'slope starts at zero loss')

    def test_l2_smooth_risk_activates_without_geometric_cut(self):
        def r(thw):
            state=RewardState(StraightConfig(level=2))
            return state.observe(dict(thw=thw,front=True,lateral=.5,ell=0.),1/15)['reward']
        self.assertGreater(r(.7),0.,'in-window front proximity gives feedback without any cut history')
        self.assertAlmostEqual(r(1.4),0.,places=6)
        self.assertGreater(r(.7),r(1.4))

    def test_l3_dense_is_control_cost_only(self):
        state=RewardState(StraightConfig(level=3))
        state.geometric_cut=True
        r=state.observe(dict(ell=.5,front=True,thw=.5),1/15)['reward']
        self.assertEqual(r,0.,'no d/h leakage into L3 dense')

    def test_continuous_potentials_guide_levels(self):
        env=StraightEnv(StraightConfig(level=1))
        env.reset(seed=0,options={'family':'follow'})
        env.target.position[:]=[100.,0.];env.target.speed=18.
        env.reference.vehicles[1].speed=18.
        env.attacker.speed=18.
        env.attacker.position[:]=[130.,0.]
        far=env._phi()
        env.attacker.position[:]=[113.,0.]  # gap 8 = preferred front position
        self.assertGreater(env._phi(),far,'L1 potential must prefer the front position')
        env.attacker.position[:]=[95.,0.]
        self.assertEqual(env._phi(),0.,'behind stays zero')
        env2=StraightEnv(StraightConfig(level=2))
        env2.reset(seed=0,options={'family':'follow'})
        env2.target.position[:]=[100.,0.];env2.target.speed=18.
        env2.attacker.speed=18.
        env2.attacker.position[:]=[114.,0.]  # gap 9, close to g*=14.4
        near=env2._phi()
        env2.attacker.position[:]=[140.,0.]  # gap 35, far from g*
        self.assertGreater(near,env2._phi(),'L2 potential peaks near the target spacing')
        env2.attacker.position[:]=[114.,4.]
        self.assertLess(env2._phi(),near,'lateral offset lowers the L2 potential')
        env3=StraightEnv(StraightConfig(level=3))
        env3.reset(seed=0,options={'family':'follow'})
        env3.target.position[:]=[100.,0.];env3.target.speed=18.
        env3.attacker.speed=18.
        env3.attacker.position[:]=[103.,0.]
        close=env3._phi()
        env3.attacker.position[:]=[130.,0.]
        self.assertGreater(close,env3._phi(),'L3 potential grows with proximity')
        env3.attacker.position[:]=[103.,4.]
        self.assertLess(env3._phi(),close,'lateral offset lowers the L3 potential')

    def test_scenario_grid_sampling_is_rng_driven_and_deterministic(self):
        env=StraightEnv(StraightConfig(level=1))
        env.reset(seed=900001)
        cell=(env.family,round(env.attacker.position[0],1),round(env.target.speed,1))
        env.reset(seed=900001)
        self.assertEqual(cell,(env.family,round(env.attacker.position[0],1),round(env.target.speed,1)),
                     'same seed must draw the same grid cell (bit-exact resume)')
        cells=set()
        for i in range(60):
            env.reset(seed=900010+i)
            cells.add((env.family,round(env.attacker.position[0],1),round(env.target.speed,1)))
        self.assertGreater(len(cells),3,'variety across seeds')
        for family,x,v in cells:
            self.assertIn((family,round(x-100.,1),round(v,1)),GRID,'cells must come from the frozen grid')
        env.reset(seed=900001,options={'family':'follow'})
        self.assertEqual(env.family,'follow','explicit family option bypasses the grid')

    def test_l2_success_latches_after_hold_and_extra_cuts_do_not_revoke(self):
        cfg=StraightConfig(level=2,duration=6.)
        state=RewardState(cfg)
        def do_cut():
            for i in range(6):
                state.observe(dict(attacker_lane_changed=i==0,target_lane_stable=True,aligned=True,front=True,thw=.7),1/15)
            state.observe(dict(reference_speed=18.,target_speed=10.),1/15)
        do_cut();do_cut()
        while not state.done:
            state.observe(dict(),1/15)
        self.assertTrue(state.success)
        self.assertEqual(state.cut_count,2,'extra cuts after the hold must not revoke success')

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
