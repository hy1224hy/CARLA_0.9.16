import copy
import unittest
import numpy as np
from evaluation.r6_counterfactual import paired_credit, settle_credit, RedOutcomeIntegral, DEFAULTS
from environment.mut_mappo_outcome_reward_v6 import CounterfactualReward
from training.train_no_rllib_mappo_robust_v2 import RolloutBuffer,Transition
from tests.test_mut_mappo_critical_reward_v3 import R3Tests


def record(debt=4.,burden=2.):
    states={str(i):dict(location=dict(x=float(i*7),y=0.,z=.5),speed=3.,rotation=dict(yaw=0.)) for i in range(4)}
    return dict(valid=True,any_contact=False,case_key='case',seed=42,team='A',
        red_progress_debt_s=debt,red_burden_integral_s=burden,condition='C2',outcome_horizon_s=50.,
        case_geometry=dict(realized_initial_states=states),
        safety=dict(per_agent={'2':{},'3':{}}))


class R6Tests(unittest.TestCase):
    def test_nonnegative_dimensions_do_not_cancel_and_credit_is_individual(self):
        actual=record();cf2=record(3.,3.);cf3=record(5.,1.)
        x=paired_credit(actual,cf2,2);y=paired_credit(actual,cf3,3)
        self.assertAlmostEqual(x['credit'],8*.7)
        self.assertAlmostEqual(y['credit'],8*.3)
        self.assertLess(x['burden_delta_s'],0.)
        self.assertGreater(x['delay_delta_s'],0.)
        self.assertEqual(paired_credit(actual,record(),2)['credit'],0.)

    def test_contact_failure_geometry_and_identity_fail_closed(self):
        a=record();b=record(3.,1.)
        for field,value in (('any_contact',True),('valid',False),('safety',{'collision':True})):
            bad=copy.deepcopy(b);bad[field]=value
            self.assertEqual(paired_credit(a,bad,2)['credit'],0.)
        bad=copy.deepcopy(b);bad['case_geometry']['realized_initial_states']['0']['location']['x']+=1.
        self.assertFalse(paired_credit(a,bad,2)['eligible'])
        bad=copy.deepcopy(a);bad['safety']['per_agent']['3']['offroad']=True
        self.assertEqual(paired_credit(bad,b,2)['credit'],0.)
        bad=copy.deepcopy(b);bad['seed']=43
        with self.assertRaises(ValueError):paired_credit(a,bad,2)

    def test_noise_deadband_and_no_distance_bonus(self):
        self.assertEqual(paired_credit(record(),record(3.99,1.995),2)['credit'],0.)
        # Role position alone (including being behind the red cars) earns nothing;
        # a measured policy replacement effect earns credit without a front-only gate.
        a=record();b=record(3.,2.)
        for r in (a,b):r['case_geometry']['realized_initial_states']['2']['location']['x']=-20.
        self.assertGreater(paired_credit(a,b,2)['credit'],0.)

    def test_online_reward_has_no_mission_pace_or_proximity_incentive(self):
        r=CounterfactualReward()
        args=R3Tests().args(1)
        for aid in (2,3):
            for k in args['base_details'][aid]:args['base_details'][aid][k]=0.
            args['base_details'][aid].update(progress=100.,success=100.,free_flow_pace_deficit=-100.,ttc=-100.,safety_lidar=-100.,lane_deviation=-100.)
        values,parts,_=r.step(**args)
        self.assertEqual(values[2],0.)
        args=R3Tests().args(2);args['base_details'][2]['collision']=-50.;args['safety']={2:{'collision':True}}
        _,parts,_=r.step(**args)
        self.assertEqual(parts[2]['collision'],-20.)
        self.assertEqual(parts[2]['outcome_reward'],0.)

    def test_settlement_only_changes_actual_last_transition_and_reconciles(self):
        b=RolloutBuffer([2,3]);r=record()
        r.update(returns={'2':-.5,'3':-.5},individual_reward_components={str(i):dict(control_cost=-.5,outcome_reward=0.) for i in (2,3)},agent_terminated={},agent_truncated={})
        for i in (2,3):
            b.add(i,Transition(np.zeros(20),np.zeros(80),np.zeros(2),0.,-.5,1.,0.,0.,'case'))
            b.mark_episode_end(i);b.add_episode_bootstrap(i,np.ones(80))
        credits={i:paired_credit(record(),record(3.,2.) if i==2 else record(),i) for i in (2,3)}
        settle_credit(r,b,{2:0,3:0},credits)
        self.assertAlmostEqual(b.data[2][-1].reward,5.1)
        self.assertEqual(b.data[3][-1].reward,-.5)
        self.assertEqual(b.data[2][-1].terminated,1.)
        self.assertIsNone(b.episode_bootstrap_states[2][-1])
        self.assertAlmostEqual(sum(r['individual_reward_components']['2'].values()),r['returns']['2'])

    def test_common_random_numbers_are_role_keyed_not_order_dependent(self):
        import torch
        from training.r6_collection import role_noise_seed
        def noise(i):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(role_noise_seed(42,8,i))
                return torch.randn(2)
        before=torch.get_rng_state().clone();a=noise(3);noise(2);b=noise(3)
        self.assertTrue(torch.equal(a,b));self.assertFalse(torch.equal(a,noise(2)))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_outcome_endpoint_convention_is_explicit(self):
        t=RedOutcomeIntegral(.1,2)
        args=R3Tests().args(1)
        info=dict(mut_metrics=dict(route_progress={'0':.5,'1':1.},contact_events=[]),
            active_mask={0:True,1:False},post_tick_kinematics=args['snapshots'])
        t.step(info);x=t.finish()
        self.assertAlmostEqual(x['red_progress_debt_s'],.05)
        self.assertIn('endpoint_extension_convention',x)


if __name__=='__main__':unittest.main()
