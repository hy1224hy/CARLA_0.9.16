import json
import tempfile
import unittest
from pathlib import Path
from highwaytest.straight.config import StraightConfig


class ReportingTests(unittest.TestCase):
    def _report(self,root,out,source='unit-test-fixture'):
        from unittest.mock import patch
        from highwaytest.straight.reporting import report
        with patch('highwaytest.training.checkpoint.source_digest',return_value=source):
            return report(root,out)

    def test_beta_panel_categories_per_beta(self):
        from highwaytest.straight.figures import _panel_categories,_is_beta_row,_beta_name
        def row(beta,method='mutual'):
            return dict(level=2,method=method,beta=beta,ablation='full',attack_success_rate=.5,utility=.1,drawdown_max_to_min=0.)
        rows=[row(0.,'independent'),row(.005),row(.02),row(.08),row(.02,method='independent')]
        self.assertEqual(_panel_categories([r for r in rows if _is_beta_row(r)],_beta_name),['0','0.005','0.02','0.08'])

    def test_main_row_predicate(self):
        from highwaytest.straight.reporting import main_row
        base=dict(level=2,method='mutual',beta=.02,risk_weight=1.,event_weight=1.,effect_weight=1.,geometry_weight=0.)
        self.assertTrue(main_row(base))
        self.assertFalse(main_row(dict(base,beta=.08)))
        self.assertFalse(main_row(dict(base,risk_weight=0.)))
        self.assertFalse(main_row(dict(base,event_weight=0.)))
        self.assertTrue(main_row(dict(base,method='independent',beta=0.)))
        self.assertFalse(main_row(dict(base,level=0)))

    def test_report_rejects_evaluation_predating_current_source(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);self.write(root,'a')
            with self.assertRaises(ValueError):self._report(root,root/'out',source='newer-source')

    def test_report_series_exclude_study_treatments(self):
        import json as _json
        def add_validation(run,beta,method='mutual'):
            overall=dict(attack_success_rate=.5,invalid_rate=0.,collision_rate=0.,grade_violation_rate=0.,utility=.1,baseline_valid_rate=1.)
            rows=[dict(train_steps=10,reports=dict(A=dict(overall=overall),B=dict(overall=overall)))]
            (run/'validation.jsonl').write_text(''.join(_json.dumps(r)+'\n' for r in rows))
            (run/'summary.json').write_text(_json.dumps(dict(selection=dict(team='A'))))
        with tempfile.TemporaryDirectory() as f:
            root=Path(f)
            self.write(root,'main_m',seed=42);add_validation(root/'main_m',.02)
            self.write(root,'beta008',seed=43);add_validation(root/'beta008',.08)
            data=_json.loads((root/'beta008'/'evaluation.json').read_text());data['config']['beta']=.08
            (root/'beta008'/'evaluation.json').write_text(_json.dumps(data))
            self._report(root,root/'out')
            import csv
            with open(root/'out/results_series.csv',encoding='utf-8-sig') as stream:
                series=list(csv.DictReader(stream))
            self.assertEqual({s['seed'] for s in series},{'42'})

    def test_winner_series_and_selected_lineage(self):
        from highwaytest.straight.reporting import winner_series,selected_lineage
        def overall(rate):
            return dict(attack_success_rate=rate,invalid_rate=0.,collision_rate=0.,
                        grade_violation_rate=0.,utility=0.,baseline_valid_rate=1.)
        rows=[dict(train_steps=10,reports=dict(A=dict(overall=overall(.6)),B=dict(overall=overall(.4)))),
              dict(train_steps=20,reports=dict(A=dict(overall=overall(.3)),B=dict(overall=overall(.5))))]
        self.assertEqual(winner_series(rows,2),[(10,.6),(20,.5)])
        self.assertEqual(selected_lineage(rows,2,'B'),[(10,.4),(20,.5)])

    def test_drawdown_second_half(self):
        from highwaytest.straight.reporting import drawdown
        d=drawdown([.6,.4,.8,.2,.9],half_window=3)
        self.assertAlmostEqual(d['max_to_min'],.7)
        self.assertAlmostEqual(d['mean_adjacent_drop'],.3)
        d=drawdown([.5,.1,.3,.2])
        self.assertAlmostEqual(d['max_to_min'],.1)
        self.assertAlmostEqual(d['mean_adjacent_drop'],.1)
        self.assertEqual(drawdown([]),{'max_to_min':0.,'mean_adjacent_drop':0.})

    def write(self,root,name,seed=42,target=18):
        path=root/name/'evaluation.json';path.parent.mkdir()
        cfg=StraightConfig(seed=seed,target_speed=target)
        path.write_text(json.dumps(dict(config=cfg.to_dict(),split='test',source_changed=False,
            source_sha256='unit-test-fixture',evaluation_source_sha256='unit-test-fixture',
            episodes=[dict(case_seed=3000000,family='follow')],
            overall={'attack_success_rate':.5,'collision_rate':0.,'utility':-.1})))

    def test_missing_descriptor_fields_are_skipped_not_fabricated(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);self.write(root,'a');self.write(root,'b',seed=43)
            result=self._report(root,root/'out')
            note=json.loads((root/'out/figures_note.json').read_text())
            self.assertIn('attack_weight_study',note)
            self.assertTrue((root/'out/results.csv').is_file())

    def test_invalid_baseline_reference_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);self.write(root,'a',seed=42)
            path=root/'a'/'evaluation.json'
            data=json.loads(path.read_text());data['overall']['baseline_valid_rate']=.9
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):self._report(root,root/'out')

    def test_pool_gates_by_level_descriptors_and_skips_study_treatments(self):
        import torch
        from unittest.mock import patch
        from highwaytest.straight.reporting import build_pool
        from highwaytest.straight.runner import PROTOCOL
        def make(root,name,level,overall,method='mutual',beta=.02):
            path=root/name/'selected.pt';path.parent.mkdir(parents=True)
            cfg=StraightConfig(level=level,seed=42,method=method,beta=beta).to_dict()
            torch.save(dict(protocol=PROTOCOL,config=cfg,
                source_sha256='fixture',model={},selection=dict(team='A',train_steps=8,eligible=True,overall=overall)),path)
        with tempfile.TemporaryDirectory() as f,patch('highwaytest.training.checkpoint.source_digest',return_value='fixture'):
            root=Path(f)
            base=dict(attack_success_rate=.5,invalid_rate=0.,baseline_valid_rate=1.,utility=.1)
            make(root,'l2_nocut',2,dict(base,cut_in_rate=0.,collision_rate=0.))
            make(root,'l2_good',2,dict(base,cut_in_rate=.1,collision_rate=0.))
            make(root,'l2_beta008',2,dict(base,cut_in_rate=.5,collision_rate=0.),beta=.08)
            make(root,'l1_good',1,dict(base,blocking_time=2.,collision_rate=0.,grade_violation_rate=0.))
            make(root,'l3_good',3,dict(base,collision_rate=.6))
            result=build_pool(root,root/'pool.json')
            self.assertEqual(sorted(m['level'] for m in result['members']),[1,2,3])
            self.assertIn('l2_beta008',result['skipped_non_main'])
            self.assertEqual(result['members'][1]['checkpoint'].split('\\')[-2],'l2_good')

    def test_duplicate_seed_is_not_an_independent_repeat(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);self.write(root,'a');self.write(root,'b')
            with self.assertRaises(ValueError):self._report(root,root/'out')

    def test_different_target_contract_rejected(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);self.write(root,'a');self.write(root,'b',seed=43,target=22)
            with self.assertRaises(ValueError):self._report(root,root/'out')

    def test_exports_negative_effect_and_independent_seed_spread(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f);self.write(root,'a');self.write(root,'b',seed=43)
            result=self._report(root,root/'out')
            self.assertEqual(result['runs'],2)
            row=json.loads((root/'out/summary.json').read_text())[0]
            self.assertEqual(row['n'],2)
            self.assertAlmostEqual(row['utility']['mean'],-.1)
            self.assertTrue((root/'out/attack_success_rate.pdf').is_file())
