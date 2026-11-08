import copy
import unittest
import tempfile
import json
from pathlib import Path
from utils.analyze_mut_mappo_threat import analyze_report, discover_reports, generate

def fixture():
    pairs = []
    for key, c0, c1, c2, eligible in [
        ('attack', True, True, False, True),
        ('recovery', True, False, True, True),
        ('unsafe', True, True, False, False),
        ('natural', False, False, False, True),
    ]:
        pairs.append(dict(case_key=key, paired_attempt=0,
            red_success=dict(C0=c0, C1=c1, C2=c2), blue_eligible_pair=eligible,
            arrival_delay_C2_minus_C1_both_success_seconds=None,
            red_completion_difference_C1_minus_C2=0.0))
    return dict(schema_version='mut_mappo_evaluation_v1', evidence='independent_evaluation',
        metadata=dict(status='COMPLETED', split='selection', config_path='/any/configs/custom_s7.json'),
        paired=dict(teams=dict(A=dict(included=pairs, excluded=[], n_common_valid=4,
            conditions=dict(C2=dict(b=.25, blue_violation_episode_counts={'collision': 1}))))))

class ThreatTests(unittest.TestCase):
    def test_positive_and_reverse_do_not_disappear_in_net(self):
        rows, cases = analyze_report(fixture(), 'report.json')
        self.assertEqual(rows[0]['safe_new_failures'], 1)
        self.assertEqual(rows[0]['safe_recoveries'], 1)
        self.assertEqual(rows[0]['safe_net_gain'], 0)
        self.assertEqual(rows[0]['unsafe_new_failures'], 1)
        self.assertEqual(rows[0]['c0_failure_cases'], 1)
        self.assertEqual(rows[0]['c1_incremental_failure_cases'], 1)
        self.assertEqual(len(cases), 4)
    def test_reject_training_and_incomplete_reports(self):
        for field, value in [('evidence', 'training_window'), ('schema_version', 'wrong')]:
            d = fixture(); d[field] = value
            with self.assertRaises(ValueError): analyze_report(d, 'x')
        d = fixture(); d['metadata']['status'] = 'FAILED'
        with self.assertRaises(ValueError): analyze_report(d, 'x')
    def test_missing_c0_is_unknown_not_failure(self):
        d = fixture()
        for p in d['paired']['teams']['A']['included']: p['red_success'].pop('C0')
        rows, _ = analyze_report(d, 'x')
        self.assertIsNone(rows[0]['c0_failure_cases'])
    def test_missing_blue_qualification_cannot_be_scored(self):
        d = fixture(); d['paired']['teams']['A']['included'][0].pop('blue_eligible_pair')
        with self.assertRaises(ValueError): analyze_report(d, 'x')
    def test_no_valid_pairs_has_no_zero_gain(self):
        d = fixture(); d['paired']['teams']['A'].update(included=[], n_common_valid=0)
        rows, _ = analyze_report(d, 'x')
        self.assertIsNone(rows[0]['safe_net_gain'])
    def test_does_not_modify_input(self):
        d = fixture(); before = copy.deepcopy(d)
        analyze_report(d, 'x')
        self.assertEqual(before, d)

class IntegrationTests(unittest.TestCase):
    def test_relocated_experiment_uses_latest_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'experiment.json').write_text('{}')
            (root / 'results').mkdir()
            for round_no in (1, 2):
                key = f'custom_r{round_no}'
                (root / 'results' / (key + '.json')).write_text(json.dumps(dict(
                    run_id='custom', completed_round=round_no,
                    selection_dir=f'/old/server/evaluations/{key}/selection')))
            target = root / 'evaluations/custom_r2/selection'
            target.mkdir(parents=True)
            (target / 'summary.json').write_text(json.dumps(fixture()))
            inputs = discover_reports(root)
            self.assertEqual(inputs, [(target / 'summary.json', 'custom')])
            result = generate(inputs, root / 'out')
            self.assertEqual(len(result['summaries']), 1)
            self.assertTrue((root / 'out/index.html').exists())
            self.assertTrue((root / 'out/cases.csv').exists())
            with self.assertRaises(FileExistsError): generate(inputs, root / 'out')

    def test_bad_input_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / 'bad.json'
            bad.write_text('{}')
            with self.assertRaises(ValueError): generate([(bad, None)], root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_unsafe_delays_separated_from_safe_delays(self):
        d = fixture()
        pairs = d['paired']['teams']['A']['included']
        pairs[0]['red_success']['C2'] = True
        pairs[0]['arrival_delay_C2_minus_C1_both_success_seconds'] = 2
        pairs[2]['red_success']['C2'] = True
        pairs[2]['arrival_delay_C2_minus_C1_both_success_seconds'] = 10
        rows, _ = analyze_report(d, 'x')
        self.assertEqual(rows[0]['arrival_delay_s'], 6)
        self.assertEqual(rows[0]['safe_arrival_delay_s'], 2)

if __name__ == '__main__': unittest.main()
