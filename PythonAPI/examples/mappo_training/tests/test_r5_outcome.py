import unittest
from evaluation.r5_outcome_metrics import OutcomeTracker, PROTOCOL
from environment.mut_mappo_outcome_reward_v5 import OutcomeReward,validate_v5_settings
from tests.test_mut_mappo_critical_reward import snapshot
from tests import test_mut_mappo_critical_reward_v3 as v3

class R5Tests(unittest.TestCase):
    def args(self,step=1,deficit=.1,second=False):
        return dict(step=step,snapshots={0:snapshot(0,vx=3),2:snapshot(7,vx=3),3:snapshot(9 if second else 50,y=0 if second else 10,vx=3)},
            active={0:True,2:True,3:True},reached={},progress={0:.2,1:.2,2:.2,3:.2},safety={},
            reference={'0':[.2]*100,'1':[.2]*100,'outcome_reference':{'protocol':PROTOCOL,'filter_tau_s':.3,'steps':{str(step):{'0':{
                'progress_mean':.2+deficit,'progress_std':0.,'burden_mean':0.,'burden_std':0.,'valid_samples':3}}}}})

    def run_two(self,tracker=None,**kw):
        t=tracker or OutcomeTracker()
        t.step(**self.args(1,**kw)); x=t.step(**self.args(2,**kw))
        return t,x

    def test_closeness_without_outcome_has_zero_positive_credit(self):
        _,x=self.run_two(deficit=0.)
        self.assertEqual(x[2]['outcome'],0.)

    def test_progress_loss_scores_only_interacting_blue(self):
        _,x=self.run_two()
        self.assertGreater(x[2]['outcome'],0.)
        self.assertEqual(x[3]['outcome'],0.)

    def test_helping_red_is_signed_negative(self):
        _,x=self.run_two(deficit=-.1)
        self.assertLess(x[2]['outcome'],0.)

    def test_two_blues_cannot_each_claim_full_red_loss(self):
        _,one=self.run_two();_,two=self.run_two(second=True)
        self.assertAlmostEqual(one[2]['outcome'],two[2]['outcome']+two[3]['outcome'])

    def test_missing_reference_fails_closed_and_counts_coverage(self):
        t=OutcomeTracker()
        for n in (1,2):
            a=self.args(n);a['reference']=None;x=t.step(**a)
        self.assertEqual(x[2]['outcome'],0.)
        self.assertEqual(t.metrics()['2']['reference_coverage'],0.)

    def test_reference_spread_creates_deadband(self):
        t=OutcomeTracker()
        for n in (1,2):
            a=self.args(n);a['reference']['outcome_reference']['steps'][str(n)]['0']['progress_std']=.1;x=t.step(**a)
        self.assertEqual(x[2]['outcome'],0.)

    def test_contact_blocks_credit_permanently(self):
        t,x=self.run_two();a=self.args(3);a['contact_ids']=[0];x=t.step(**a)
        self.assertEqual(x[2]['outcome'],0.)
        self.assertEqual(t.step(**self.args(4))[2]['outcome'],0.)

    def test_finished_red_does_not_create_slowdown_credit(self):
        t=OutcomeTracker()
        for n in (1,2):
            a=self.args(n);a['reached']={0:True};x=t.step(**a)
        self.assertEqual(x[2]['outcome'],0.)

    def test_invalid_reference_and_settings(self):
        with self.assertRaises(ValueError):validate_v5_settings({'min_reference_samples':1})
        with self.assertRaises(ValueError):validate_v5_settings({'delay_fraction':2.})
        with self.assertRaises(ValueError):validate_v5_settings({'outcome_budget':100.})

    def test_reward_geometry_has_no_positive_bonus_and_accounts_reconcile(self):
        r=OutcomeReward();total=0.
        for n in range(1,30):
            a=v3.R3Tests().args(n)
            a['reference']=None
            values,parts,_=r.step(**a)
            self.assertEqual(parts[2]['risk_reward'],0.)
            self.assertEqual(parts[2]['response_reward'],0.)
            self.assertEqual(parts[2]['outcome_reward'],0.)
            total+=sum(values.values())/2.
        self.assertAlmostEqual(r.metrics(True)['team_utility_total'],total)
        self.assertAlmostEqual(sum(r.metrics(True)['team_reward_components'].values()),total)

    def test_budget_and_own_collision_cost(self):
        r=OutcomeReward({'outcome_budget':.02})
        for n in range(1,40):
            a=v3.R3Tests().args(n);b=self.args(n)
            for key in ('snapshots','active','reached','progress','safety','reference'):a[key]=b[key]
            values,parts,_=r.step(**a)
        self.assertAlmostEqual(r.outcome_paid[2],.02)
        a=v3.R3Tests().args(40);a['base_details'][2]['collision']=-50.;a['safety']={2:{'collision':True}}
        _,parts,_=r.step(**a)
        self.assertEqual(parts[2]['collision'],-50.)
        self.assertEqual(parts[3]['collision'],0.)
        self.assertEqual(parts[2]['outcome_reward'],0.)

    def test_repeated_reference_aggregation_does_not_fill_missing(self):
        from utils.prepare_r5_reference import aggregate
        runs=[{'2':{'0':{'progress':v,'burden':0.}}} for v in (.2,.3,.4)]
        ref=aggregate(runs,3)
        self.assertNotIn('1',ref['steps'])
        self.assertAlmostEqual(ref['steps']['2']['0']['progress_mean'],.3)
        self.assertGreater(ref['steps']['2']['0']['progress_std'],0.)
        self.assertNotIn('3',ref['steps'])

    def test_policy_replacement_contrast_and_contact_rejection(self):
        from evaluation.evaluate_r5_counterfactual import contrast
        rs=[dict(replacement=m,case_key='case',seed=42,team='A',valid=True,any_contact=False,
                 red_progress_debt_s=v,red_burden_integral_s=v/2) for m,v in [('none',4),('blue2',2),('blue3',3),('both',1)]]
        out=contrast(rs)
        self.assertEqual(out['effects']['red_progress_debt_s']['blue2_marginal'],2)
        self.assertEqual(out['effects']['red_progress_debt_s']['interaction'],0)
        rs[0]['any_contact']=True
        self.assertIsNone(contrast(rs)['effects'])

if __name__=='__main__':unittest.main()
