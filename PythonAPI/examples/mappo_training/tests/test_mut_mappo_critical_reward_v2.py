"""Second candidate incentives; no simulator or claims of learned driving."""
import unittest
from tests import test_mut_mappo_critical_reward as v1tests

from environment.mut_mappo_critical_reward_v2 import NormalDrivingCriticalReward as Reward


class NormalCriticalRewardTests(unittest.TestCase):
    def args(self,step=1):
        args=v1tests.CriticalRewardTests().args()
        args.update(step=step,base_details={2:{'lane_deviation':0.}},driving_context={2:dict(
            legal_forward=True,route_speed_mps=1.,front_blocker=False,near_goal=False,terminal=False)})
        return args

    def test_primary_bonus_requires_credited_persistence_after_grace_and_is_once(self):
        r=Reward({'pace_weight':0.})
        critical=event=0.
        for step in range(1,51):
            _,p=r.step(**self.args(step));critical+=p['critical'];event+=p['primary_event_bonus']
            if step<=12:self.assertEqual(event,0.)
        self.assertAlmostEqual(critical,3.)
        self.assertEqual(event,5.)

    def test_near_threshold_shaping_cannot_earn_primary_bonus(self):
        r=Reward({'pace_weight':0.})
        for step in range(1,101):
            a=self.args(step);a['snapshots'][2]['location']['x']=9.
            r.step(**a)
        self.assertEqual(r.components_total['primary_event_bonus'],0.)
        self.assertLessEqual(r.components_total['critical'],3.+1e-9)

    def test_new_normal_costs_and_legal_lane_exemption(self):
        a=self.args();a['costs']={2:{'overspeed':.1}}
        a['base_details']={2:{'lane_deviation':-.02,'steer_rate':-.01,'longitudinal_rate':-.01}}
        _,p=Reward().step(**a)
        self.assertAlmostEqual(p['normal_overspeed'],-.45)
        self.assertAlmostEqual(p['normal_comfort'],-.04)
        self.assertAlmostEqual(p['normal_lane'],-.1)
        a['base_details'][2]['lane_deviation']=0.
        self.assertEqual(Reward().step(**a)[1]['normal_lane'],0.)

    def test_late_collision_suppresses_credit_and_dominates_positive_budget(self):
        r=Reward({'pace_weight':0.})
        for step in range(1,35):r.step(**self.args(step))
        a=self.args(35);a['safety']={2:{'collision':True}}
        _,p=r.step(**a)
        self.assertEqual(p['primary_event_bonus'],0.)
        self.assertEqual(p['critical'],0.)
        self.assertLess(r.utility_total,0.)
        self.assertEqual(r.metrics(False)['qualified_primary_event_credit'],0.)
        with self.assertRaises(ValueError):Reward({'primary_event_reward':20.})


if __name__=='__main__':unittest.main()
