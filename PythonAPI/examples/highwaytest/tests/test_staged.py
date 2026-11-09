"""Curriculum and initialization contracts, not algorithm superiority tests."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch
from ..config import Config, Task, ROOT
from ..models.team import Team
from ..training.curriculum import Curriculum
from ..training.runner import run
from ..training.checkpoint import read


EXPLORATION = {'initial': [0.3, 0.08], 'minimum': [0.1, 0.03], 'maximum': [0.6, 0.2]}
COURSE = {'weights': [0.1, 0.3, 1.0], 'min_steps': 10000, 'qualified': 0.875,
          'max_bad': 0.0625, 'patience': 2}

def evidence(success=1., bad=0.):
    return {t: {'overall': {'success_rate': success, 'bad_rate': bad, 'baseline_invalid_count': 0}}
            for t in ('A','B')}

class StagedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_config_retains_legacy_serialization(self):
        raw = Config().to_dict()
        self.assertNotIn('exploration', raw)
        self.assertNotIn('curriculum', raw)

    def test_axis_exploration_and_bounds(self):
        cfg = Config(exploration=EXPLORATION).validate()
        model = Team(cfg)
        torch.testing.assert_close(model.action_std(), torch.tensor([[.3,.08],[.3,.08]]))
        with torch.no_grad():
            model.log_std.fill_(-100)
        torch.testing.assert_close(model.action_std(), torch.tensor([[.1,.03],[.1,.03]]))
        with self.assertRaises(ValueError):
            replace(cfg, exploration=EXPLORATION | {'minimum':[.4,.03]}).validate()

    def test_stage_requires_both_teams_and_consecutive_checks(self):
        c = Curriculum(COURSE)
        self.assertEqual(c.weight, .1)
        c.observe(evidence(), 10000)
        one_bad = evidence(); one_bad['B']['overall']['success_rate'] = .5
        c.observe(one_bad, 11000)
        self.assertEqual(c.weight, .1)
        c.observe(evidence(), 12000); c.observe(evidence(), 13000)
        self.assertEqual(c.weight, .3)
        self.assertFalse(c.full_strength)
        c.observe(evidence(), 14000); c.observe(evidence(), 15000)
        self.assertEqual(c.weight, .3)  # minimum residence in each stage

    def test_curriculum_restore_and_regression(self):
        c = Curriculum(COURSE)
        c.observe(evidence(), 10000)
        restored = Curriculum(COURSE, c.state_dict())
        restored.observe(evidence(), 11000)
        self.assertEqual(restored.weight, .3)
        restored.observe(evidence(success=.5), 12000)
        restored.observe(evidence(success=.5), 13000)
        self.assertEqual(restored.weight, .1)
        self.assertEqual(restored.state_dict()['transitions'][-1]['reason'], 'qualification_regression')

    def test_curriculum_does_not_accept_invalid_reference(self):
        c=Curriculum(COURSE)
        ev=evidence()
        ev['A']['overall']['baseline_invalid_count']=1
        c.observe(ev,10000); c.observe(ev,20000)
        self.assertEqual(c.weight,.1)

    def test_driver_export_rejects_unqualified_or_attack_candidate(self):
        from ..training.driving import validate_driver
        cfg=Config(task=Task(research_protocol='route_relative_paired_attack_v3'))
        reports={split:{'families':{f:{'success_rate':1.,'bad_rate':0.,'baseline_invalid_count':0}
                    for f in ('parallel','closing')}} for split in ('gate','selection')}
        validate_driver(cfg,reports)
        reports['gate']['families']['closing']['success_rate']=.5
        with self.assertRaises(ValueError):
            validate_driver(cfg,reports)
        with self.assertRaises(ValueError):
            validate_driver(replace(cfg,task=replace(cfg.task,attack_weight=1)),reports)

    def test_demonstrations_use_local_features_and_training_seeds(self):
        from ..training.driving import demonstrations
        cfg=Config(task=Task(research_protocol='route_relative_paired_attack_v3',duration=.4,goal_distance=.1))
        data=demonstrations(cfg,2)
        self.assertEqual(data['obs'].shape,(4,2,25))
        self.assertEqual(data['actions'].shape,(4,2,2))
        self.assertTrue((abs(data['actions'])<=1).all())
        self.assertTrue(all(0<=s<900000 for s in data['seeds']))

    def test_study_generates_seed42_matched_attack_matrix(self):
        import json
        from ..staged import prepare
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp)/'study'
            prepare(root,42,1000,2000)
            configs=[json.loads(p.read_text()) for p in (root/'configs/attack').glob('*.json')]
            self.assertEqual(len(configs),8)
            self.assertEqual({c['seed'] for c in configs},{42})
            self.assertEqual({c['total_train_steps'] for c in configs},{2000})
            self.assertTrue(all(c['curriculum']==COURSE for c in configs))
            with self.assertRaises(FileExistsError):
                prepare(root,42,1000,2000)

    def test_diagnostics_expose_stalled_role_instead_of_only_mean(self):
        from ..common.contract import SceneFrame, VehicleFrame
        from ..common.outcomes import OutcomeTracker
        vehicles={i:VehicleFrame(i,(i*10.,0.),(0. if i==2 else 12.,0.),0.,0.,0.,
                    20. if i==2 else 240.) for i in range(4)}
        tracker=OutcomeTracker(Task())
        tracker.update(SceneFrame(vehicles,0.,(0.,0.)))
        tracker.update(SceneFrame(vehicles,2.,(0.,0.)))
        summary=tracker.summary()
        self.assertEqual(summary['blue_min_progress_m'],20.)
        self.assertEqual(summary['blue_0_stopped_s'],2.)
        self.assertEqual(summary['blue_1_stopped_s'],0.)

    def test_full_strength_budget_handles_integer_and_float_weights(self):
        from ..research import attack_exposure
        histogram,full=attack_exposure([{'attack_weight':.3,'sample_steps':3},
            {'attack_weight':1.,'sample_steps':7}],1)
        self.assertEqual(full,7)
        self.assertEqual(sum(histogram.values()),10)

    def test_source_identity_excludes_distribution_copies(self):
        from ..training.checkpoint import source_digest,provenance
        before=source_digest()
        (ROOT/'dist').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'dist') as tmp:
            (Path(tmp)/'old_distribution.py').write_text('print("not active source")\n')
            self.assertEqual(source_digest(),before)
            with tempfile.TemporaryDirectory(dir=ROOT/'runs') as run_tmp:
                out=Path(run_tmp)/'run'
                provenance(out,Config())
                self.assertFalse((out/'source/dist').exists())

    def test_batch_passes_same_initialization_to_every_fresh_group(self):
        import json,hashlib
        from ..experiments import batch
        from ..training.checkpoint import save,source_digest,PROTOCOL
        cfg=Config(total_train_steps=16,rollout_steps=8,epochs=1,minibatch=8,gate_cases_per_family=1,
            task=Task(research_protocol='route_relative_paired_attack_v3',duration=.4,goal_distance=.1,attack_weight=1))
        prior=replace(cfg,task=replace(cfg.task,attack_weight=0))
        model=Team(prior)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp); configs=root/'configs'; configs.mkdir()
            for group in ('G0','G3'):
                (configs/f'{group}.json').write_text(json.dumps(replace(cfg,group=group).to_dict()))
            initial=root/'driver.pt'
            save(initial,{'protocol':PROTOCOL,'source_sha256':source_digest(),'config':prior.to_dict(),
                'models':{t:model.state_dict() for t in ('A','B')},'train_steps':32})
            batch(configs,root/'attack',2,initialize_from=initial)
            digest=hashlib.sha256(initial.read_bytes()).hexdigest()
            for group in ('G0','G3'):
                manifest=json.loads((root/'attack'/f'{group}_s42/manifest.json').read_text())
                self.assertEqual(manifest['initialization']['sha256'],digest)
                self.assertEqual(manifest['initialization']['prior_train_steps'],32)

    def test_staged_resume_reproduces_models_and_curriculum(self):
        task=Task(research_protocol='route_relative_paired_attack_v3',duration=.4,
                  goal_distance=.1,attack_weight=1)
        cfg=Config(total_train_steps=32,rollout_steps=4,epochs=1,minibatch=4,
            gate_cases_per_family=1,refresh_rounds=1,task=task,
            exploration=EXPLORATION,curriculum=COURSE | {'min_steps':8,'patience':1})
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            run(cfg,a)
            run(cfg,b,max_rounds=2)
            run(cfg,b,resume=b/'latest.pt')
            x,y=read(a/'latest.pt'),read(b/'latest.pt')
            self.assertEqual(x['curriculum'],y['curriculum'])
            for t in ('A','B'):
                for k in x['models'][t]:
                    torch.testing.assert_close(x['models'][t][k],y['models'][t][k],rtol=0,atol=0)

if __name__=='__main__':
    unittest.main()
