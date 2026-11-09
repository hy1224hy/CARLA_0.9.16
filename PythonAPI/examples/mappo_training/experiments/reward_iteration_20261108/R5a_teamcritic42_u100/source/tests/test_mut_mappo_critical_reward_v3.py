import unittest
from tests import test_mut_mappo_critical_reward_v2 as v2
from environment.base_reward_v6 import REWARD_COMPONENTS
from environment.mut_mappo_critical_reward_v3 import IndividualDrivingCriticalReward as Reward

class R3Tests(unittest.TestCase):
    def args(self, step=1):
        a=v2.NormalCriticalRewardTests().args(step)
        a['base_details']={2:dict.fromkeys(REWARD_COMPONENTS,0.),3:dict.fromkeys(REWARD_COMPONENTS,0.)}
        # Role 3 is active but has a safe, distant geometry.
        from tests.test_mut_mappo_critical_reward import snapshot
        a['snapshots'][3]=snapshot(50., y=5.)
        a['driving_context'][3]=dict(a['driving_context'][2])
        a['active'][3]=True
        return a

    def test_individual_base_and_common_attack(self):
        a=self.args();a['base_details'][2]['progress']=2.;a['base_details'][3]['progress']=1.
        r=Reward();rewards,parts,shared=r.step(**a)
        self.assertAlmostEqual(parts[2]['base_soft']-parts[3]['base_soft'],.3)
        self.assertEqual(parts[3]['individual_adversarial'],0.)
        self.assertAlmostEqual(r.metrics(True)['team_utility_total'],sum(rewards.values())/2)
        self.assertAlmostEqual(sum(r.metrics(True)['team_reward_components'].values()),sum(rewards.values())/2)

    def test_hard_cost_not_discounted_or_shared(self):
        a=self.args();a['base_details'][2]['collision']=-50.;a['safety']={2:{'collision':True}}
        rewards,parts,_=Reward().step(**a)
        self.assertEqual(parts[2]['base_hard'],-50.)
        self.assertEqual(parts[3]['base_hard'],0.)
        self.assertAlmostEqual(rewards[2]-rewards[3],-50.)

    def test_geometry_budget_and_no_efficiency_objective(self):
        r=Reward();total=0.
        for step in range(1,51):
            a=self.args(step);_,_,shared=r.step(**a);total+=shared[2]
        self.assertGreater(total,5.)
        self.assertLessEqual(total,8.+1e-9)
        self.assertNotIn('attack',r.metrics(True)['team_reward_components'])

    def test_other_blue_does_not_receive_event_credit(self):
        r=Reward()
        for step in range(1,51):
            _,parts,adv=r.step(**self.args(step))
            self.assertEqual(adv[3],0.)
        self.assertTrue(r.role_geometry[2].primary_event_paid)
        self.assertFalse(r.role_geometry[3].primary_event_paid)

    def test_invalid_weights_and_missing_components(self):
        with self.assertRaises(ValueError):Reward({'base_soft_weight':-1})
        with self.assertRaises(ValueError):Reward({'adversarial_weight':100})
        a=self.args();del a['base_details'][2]['progress']
        with self.assertRaises(ValueError):Reward().step(**a)
