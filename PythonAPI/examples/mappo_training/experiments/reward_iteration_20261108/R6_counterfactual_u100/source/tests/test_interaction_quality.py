"""Check observational conflict endpoints without CARLA."""
import copy
import unittest

from evaluation.critical_interaction_metrics import PROTOCOL_DEFAULT, analyze_trace
from evaluation.interaction_quality import analyze_outcomes, box_gap, contract, driving_quality
from tests.test_critical_interaction_metrics import snap, trace


class InteractionQualityTests(unittest.TestCase):
    @staticmethod
    def episode(**extra):
        return dict(contact_events=[], safety=dict(collision=False), blue_eligible=True, **extra)

    def analyze(self, rows, episode=None):
        return analyze_outcomes(rows, episode or self.episode(), analyze_trace(rows, PROTOCOL_DEFAULT))

    def test_contact_alone_does_not_become_predicted_near_miss(self):
        rows = trace(1)
        rows[1]['kinematics']['2']['location']['x'] = 3
        legacy = analyze_trace(rows, PROTOCOL_DEFAULT)
        self.assertTrue(legacy['critical_event'])
        result = analyze_outcomes(rows, self.episode(), legacy)
        self.assertFalse(result['predicted_conflict'])
        self.assertTrue(result['geometry_contact_unconfirmed'])
        self.assertFalse(result['observed_contact_free_resolved'])

    def test_low_ttc_then_observed_clearance_resolves(self):
        rows = trace(6)
        for row in rows[4:]:
            row['kinematics']['2']['location']['y'] = 10
        result = self.analyze(rows)
        self.assertTrue(result['predicted_conflict'])
        self.assertTrue(result['observed_contact_free_resolved'])
        self.assertFalse(result['contact_free_censored'])
        self.assertAlmostEqual(result['precontact_tet_s'], .3)
        self.assertGreater(result['minimum_box_clearance_m'], 0)

    def test_end_of_episode_is_censored_not_resolved(self):
        result = self.analyze(trace(3))
        self.assertTrue(result['predicted_conflict'])
        self.assertTrue(result['contact_free_censored'])
        self.assertFalse(result['observed_contact_free_resolved'])

    def test_real_contact_and_timing_preserve_precontact_event(self):
        rows = trace(8)
        for row in rows[5:]:
            row['kinematics']['2']['location']['x'] = row['step'] + 2
        episode = self.episode(agent_results={'0': dict(evidence_frame=108, evidence_step=8)})
        episode['contact_events'] = [dict(agent_id=0, other_agent_id=2, frame=105, pair_class='red_blue')]
        episode['blue_eligible'] = False
        episode['safety']['collision'] = True
        result = self.analyze(rows, episode)
        self.assertTrue(result['predicted_conflict'])
        self.assertEqual(result['events'][0]['outcome'], 'physical_contact')
        self.assertFalse(result['joint_normal_resolved'])
        self.assertTrue(result['physical_red_blue_contact'])

    def test_contact_with_unknown_timing_is_unmeasured(self):
        episode = self.episode()
        episode['contact_events'] = [dict(agent_id=0, other_agent_id=2, frame=105)]
        result = self.analyze(trace(3), episode)
        self.assertEqual(result['status'], 'CONTACT_TIMING_UNMEASURED')
        self.assertIsNone(result['predicted_conflict'])

    def test_multiple_pairs_do_not_multiply_exposure(self):
        rows = trace(3)
        for row in rows:
            row['active_mask']['3'] = True
            row.setdefault('next_active_mask', {})['3'] = row['step'] < 3
            row['kinematics']['3'] = copy.deepcopy(row['kinematics']['2'])
        # The legacy initial row must retain exactly the same lifecycle schema.
        rows[0].pop('next_active_mask')
        result = self.analyze(rows)
        self.assertAlmostEqual(result['precontact_tet_s'], .3)
        self.assertEqual(len(result['events']), 2)

    def test_clearance_uses_real_vehicle_extents(self):
        self.assertAlmostEqual(box_gap(snap(0, 0), snap(10, 0)), 6)
        self.assertEqual(box_gap(snap(0, 0), snap(3, 0)), 0)


class DrivingQualityTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for step in range(1, 51):
            rows.append(dict(step=step, active_mask={'2': True, '3': True},
                driving_diagnostics={aid: dict(lateral_distance_m=.1, lane_deviation_exempted=0.,
                    pace_front_blocker_exempt=0., distance_to_goal_m=20.) for aid in ('2', '3')}))
        metrics = dict(measured_state_fraction=1., actual_control_fraction=1., peak_speed_mps=3.,
            above_preferred_speed_fraction=0., preferred_speed_fraction=1., unblocked_low_speed_fraction=0.,
            unblocked_low_speed_longest_seconds=0., actual_steer_rate_p95_per_second=.2)
        episode = dict(blue_eligible=True, driving_metrics={aid: copy.deepcopy(metrics) for aid in ('2', '3')})
        return rows, episode

    def test_only_recorded_legal_transition_is_exempt(self):
        rows, episode = self.fixture()
        for row in rows[10:20]:
            row['driving_diagnostics']['2'].update(lateral_distance_m=2., lane_deviation_exempted=1.)
        result = driving_quality(rows, episode, contract())
        self.assertTrue(result['eligible'])
        self.assertEqual(result['per_blue']['2']['legal_transition_exempt_steps'], 10)
        rows[10]['driving_diagnostics']['2']['lane_deviation_exempted'] = 0.
        self.assertFalse(driving_quality(rows, episode, contract())['eligible'])

    def test_missing_measurement_is_unknown_not_safe(self):
        rows, episode = self.fixture()
        del episode['driving_metrics']['2']['actual_control_fraction']
        self.assertIsNone(driving_quality(rows, episode, contract())['eligible'])
        episode['blue_eligible'] = False
        self.assertFalse(driving_quality(rows, episode, contract())['eligible'])


if __name__ == '__main__':
    unittest.main()
