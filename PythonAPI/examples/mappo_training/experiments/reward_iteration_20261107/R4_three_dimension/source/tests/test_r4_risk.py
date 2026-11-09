import unittest
from tests.test_mut_mappo_critical_reward import snapshot
from tests import test_mut_mappo_critical_reward_v3 as v3
from evaluation.r4_risk_metrics import RiskTracker, pair_risk
from environment.mut_mappo_critical_reward_v4 import RiskReward


class RiskTests(unittest.TestCase):
    def test_same_speed_close_front_has_spacing_without_ttc(self):
        p = pair_risk(snapshot(0, vx=3), snapshot(5, vx=3))
        self.assertIsNone(p['ttc'])
        self.assertGreater(p['spacing_score'], 0)
        self.assertAlmostEqual(p['longitudinal_gap_m'], 1.)

    def test_far_or_rear_chase_is_not_spacing_reward(self):
        self.assertEqual(pair_risk(snapshot(0), snapshot(50))['spacing_score'], 0)
        self.assertEqual(pair_risk(snapshot(0), snapshot(-5))['spacing_score'], 0)

    def test_exposure_not_capped_and_contact_stops_positive_credit(self):
        t = RiskTracker()
        for step in range(1, 41):
            t.step(step, {0:snapshot(0),2:snapshot(5)}, {0:True,2:True}, {}, None)
        self.assertAlmostEqual(t.metrics()['2']['spacing_duration_s'], 4.)
        self.assertAlmostEqual(t.metrics()['2']['spacing_longest_s'], 4.)
        x = t.step(41, {0:snapshot(0),2:snapshot(0)}, {0:True,2:True}, {}, None)
        self.assertEqual(x[2]['risk'], 0)
        x = t.step(42, {0:snapshot(0),2:snapshot(5)}, {0:True,2:True}, {}, None)
        self.assertEqual(x[2]['risk'], 0)

    def test_missing_geometry_fails_instead_of_zero_risk(self):
        with self.assertRaises(ValueError):
            RiskTracker().step(1, {0:snapshot(0)}, {0:True,2:True}, {}, None)

    def test_reward_keeps_individual_cost_and_bounded_risk(self):
        a = v3.R3Tests().args()
        r = RiskReward()
        values, parts, _ = r.step(**a)
        self.assertEqual(parts[3]['risk_reward'], 0)
        a = v3.R3Tests().args(2)
        a['base_details'][2]['collision'] = -50
        a['safety'] = {2:{'collision':True}}
        _, parts, _ = r.step(**a)
        self.assertEqual(parts[2]['collision'], -50)
        self.assertEqual(parts[2]['risk_reward'], 0)

    def test_weak_collision_penalty_is_explicit_and_validated(self):
        a=v3.R3Tests().args();a['base_details'][2]['collision']=-50.;a['safety']={2:{'collision':True}}
        _,parts,_=RiskReward({'collision_scale':.5}).step(**a)
        self.assertEqual(parts[2]['collision'],-25.)
        with self.assertRaises(ValueError):RiskReward({'collision_scale':.01})

    def test_reference_identical_response_has_no_excess_reward(self):
        t=RiskTracker()
        reference={'1':{'0':{'response_score':1.}}, '2':{'0':{'response_score':1.}}}
        t.step(1,{0:snapshot(0,vx=3),2:snapshot(5,vx=3)},{0:True,2:True},{},reference)
        x=t.step(2,{0:snapshot(0,vx=1),2:snapshot(5,vx=1)},{0:True,2:True},{},reference)
        self.assertEqual(x[2]['response_excess'],0)


if __name__=='__main__': unittest.main()
