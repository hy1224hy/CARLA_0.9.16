import unittest
import tempfile
import sys
import io
import contextlib
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
from evaluation.critical_interaction_report import paired_diagnostics, generate_report
from utils.evaluate_mut_mappo_critical import enrich_snapshots, geometry_capture


def row(c, critical, qualified=True, attempt=0, valid=True, seed=9):
    return dict(case_key='case', team='A', seed=seed, paired_attempt=attempt, condition=c,
        valid=valid, blue_eligible=qualified, red_success=True,
        critical_interaction=dict(status='COMPLETE', critical_event=critical,
            tet_s={'1.5': float(critical)}, tit_s={'1.5': float(critical)/2,}, duration_s=5))


class PairingTests(unittest.TestCase):
    def test_qualification_improvement_is_not_risk_improvement(self):
        result = paired_diagnostics([row('C1', True, False), row('C2', True)], ('C1','C2'))
        r = result['teams']['A']
        self.assertEqual(r['delta_raw_critical_rate'], 0)
        self.assertEqual(r['delta_qualified_critical_rate'], 1)
        self.assertEqual(r['qualification_only_improvements'], 1)
    def test_do_not_pair_different_attempts_or_seeds(self):
        for records in ([row('C1', True, attempt=0),row('C2', True, attempt=1)],
                        [row('C1', True, seed=1),row('C2', True, seed=2)]):
            self.assertEqual(paired_diagnostics(records, ('C1','C2'))['teams']['A']['n_measured_pairs'], 0)
    def test_use_latest_whole_valid_retry(self):
        records = [row('C1',False), row('C2',True,valid=False),
                   row('C1',True,attempt=1),row('C2',False,attempt=1)]
        r=paired_diagnostics(records,('C1','C2'))['teams']['A']
        self.assertEqual(r['n_measured_pairs'],1)
        self.assertEqual(r['delta_raw_critical_rate'],-1)
    def test_missing_geometry_not_scored_as_safe(self):
        r=row('C2',False)
        r['critical_interaction']['status']='MISSING_GEOMETRY'
        result=paired_diagnostics([row('C1',True),r],('C1','C2'))['teams']['A']
        self.assertEqual(result['n_measured_pairs'],0)
        self.assertIsNone(result['delta_raw_critical_rate'])
    def test_duplicate_condition_rejected(self):
        with self.assertRaises(ValueError): paired_diagnostics([row('C1',True),row('C1',True),row('C2',True)],('C1','C2'))

class CaptureTests(unittest.TestCase):
    def test_real_bbox_copied_without_mutating_original_snapshots(self):
        box=NS(extent=NS(x=2.2,y=.9,z=.7),location=NS(x=.3,y=.1,z=.2),rotation=NS(yaw=10,pitch=0,roll=0))
        env=NS(vehicles=[NS(is_alive=True,bounding_box=box)])
        original={0:{'location':{'x':1,'y':2}}}
        enriched=enrich_snapshots(env, original)
        self.assertEqual(enriched[0]['bounding_box']['extent']['x'],2.2)
        self.assertNotIn('bounding_box',original[0])
        self.assertEqual(box.location.x,.3)
    def test_missing_live_actor_does_not_invent_dimensions(self):
        result=enrich_snapshots(NS(vehicles=[]),{0:{}})
        self.assertNotIn('bounding_box',result[0])


class OfflineIntegrationTests(unittest.TestCase):
    @staticmethod
    def snapshot(x,v):
        return dict(location=dict(x=x,y=0,z=0),rotation=dict(yaw=0),velocity=dict(x=v,y=0),
            bounding_box=dict(extent=dict(x=2,y=1),location=dict(x=0,y=0),rotation=dict(yaw=0)))

    def test_end_to_end_real_geometry_and_missing_old_trace(self):
        from evaluation.critical_interaction_metrics import PROTOCOL_DEFAULT
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'evaluation'; source.mkdir()
            (source/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',conditions=['C1','C2'],
                split='gate',config_path='/server/configs/arbitrary_s9.json',checkpoint_completed_round=7)))
            (source/'resolved_config.json').write_text(json.dumps(dict(task=dict(timestep=.1))))
            records=[]
            for condition in ('C1','C2'):
                r=row(condition,True)
                r.update(evidence='independent_evaluation',trace_path=condition+'.jsonl',steps=3)
                records.append(r)
                samples=[dict(step=0,active_mask={'0':True,'2':True},case_geometry=dict(realized_initial_states={
                    '0':self.snapshot(0,10),'2':self.snapshot(8,5)}))]
                for step in range(1,4):
                    samples.append(dict(step=step,active_mask={'0':True,'2':True},
                        next_active_mask={'0':step<3,'2':step<3},kinematics={
                            '0':self.snapshot(step,10),'2':self.snapshot(8+.5*step,5)}))
                (source/r['trace_path']).write_text(''.join(json.dumps(x)+'\n' for x in samples))
            (source/'cases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
            protocol=dict(PROTOCOL_DEFAULT,red_ids=[0],blue_ids=[2])
            report=generate_report([source],root/'output',protocol)
            self.assertEqual(report['team_rows'][0]['n_measured_pairs'],1)
            self.assertEqual(report['team_rows'][0]['delta_raw_critical_rate'],0)
            self.assertTrue((root/'output/index.html').exists())
            self.assertTrue((root/'output/pairs.csv').exists())
            self.assertTrue((root/'output/events.csv').exists())
            manifest=json.loads((source/'manifest.json').read_text())
            manifest.update(case_keys=['case','missing_case'],teams=['A'])
            (source/'manifest.json').write_text(json.dumps(manifest))
            coverage=generate_report([source],root/'coverage',protocol)['team_rows'][0]
            self.assertEqual(coverage['n_missing_planned_cases'],1)
            self.assertEqual(coverage['measurement_coverage'],.5)
            with self.assertRaises(FileExistsError): generate_report([source],root/'output',protocol)
            # Old trace without geometry must never be measured as harmless.
            path=source/'C2.jsonl'
            text=path.read_text()
            samples=[json.loads(line) for line in text.splitlines()]
            for sample in samples:
                for snap in sample.get('kinematics',{}).values(): snap.pop('bounding_box')
            path.write_text(''.join(json.dumps(x)+'\n' for x in samples))
            old=generate_report([source],root/'missing',protocol)
            self.assertEqual(old['team_rows'][0]['n_measured_pairs'],0)
            self.assertIsNone(old['team_rows'][0]['delta_raw_critical_rate'])
            # Two seeds for one observed case must not fill the slot of a missing planned case.
            expanded=records+[dict(r,seed=10) for r in records]
            (source/'cases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in expanded))
            with self.assertRaisesRegex(ValueError,'multiple evaluation seeds'):
                generate_report([source],root/'mixed_seed',protocol)


    def test_scoped_runtime_factory_restored_on_error(self):
        from training import train_mut_mappo as runner
        env=NS(vehicles=[],_snapshot_kinematics=lambda:{0:{}})
        with patch.object(runner,'create_runtime',return_value=NS(env=env)) as factory:
            with self.assertRaisesRegex(RuntimeError,'probe'):
                with geometry_capture():
                    runtime=runner.create_runtime()
                    self.assertEqual(runtime.env._snapshot_kinematics(),{0:{}})
                    raise RuntimeError('probe')
            self.assertIs(runner.create_runtime,factory)


class CliTests(unittest.TestCase):
    def test_validate_only_never_creates_output_or_connects(self):
        from utils.evaluate_mut_mappo_critical import main
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new_output'
            args=['critical','live','--config','any.json','--checkpoint','any.pt',
                  '--output-dir',str(out),'--validate-only']
            resolved=dict(config=dict(task=dict(timestep=.1)),metadata=dict(split='gate'))
            with patch.object(sys,'argv',args), patch('evaluation.evaluate_mut_mappo.preflight',return_value=resolved), \
                 patch('evaluation.evaluate_mut_mappo.evaluate') as evaluate, \
                 patch('utils.evaluate_mut_mappo_critical.geometry_capture') as capture, \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            evaluate.assert_not_called();capture.assert_not_called()
            self.assertFalse(out.exists())

if __name__ == '__main__': unittest.main()
