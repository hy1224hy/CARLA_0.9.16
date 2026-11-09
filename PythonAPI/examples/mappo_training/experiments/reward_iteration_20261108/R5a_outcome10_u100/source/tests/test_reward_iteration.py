import unittest
from utils.reward_iteration import summarize, improved

def row(team='A',exposure=.4,event=1.,bad=False,valid=True):
    return dict(team=team,valid=valid,blue_eligible=not bad,safety=dict(collision=bad,offroad=False,road_rule=False),individual_adversarial_metrics={str(i):dict(critical_exposure_s=exposure,qualified_primary_event_credit=event) for i in (2,3)})

class IterationTests(unittest.TestCase):
    def test_invalid_evidence_is_not_success(self):
        base=summarize([row() for _ in range(24)])
        candidate=summarize([row(exposure=1.,valid=False) for _ in range(24)])
        self.assertFalse(improved(base,candidate)['passed'])

    def test_more_collisions_without_exposure_gain_is_not_success(self):
        base=summarize([row() for _ in range(24)])
        candidate=summarize([row(bad=True) for _ in range(24)])
        self.assertFalse(improved(base,candidate)['passed'])

    def test_two_roles_must_improve_on_identical_cohort(self):
        base=summarize([dict(row(exposure=.1),case_key=str(i)) for i in range(24)])
        candidate=summarize([dict(row(exposure=.5),case_key=str(i)) for i in range(24)])
        self.assertTrue(improved(base,candidate)['passed'])
        candidate['A']['roles']['2']['exposure']=.1
        self.assertFalse(improved(base,candidate)['passed'])
