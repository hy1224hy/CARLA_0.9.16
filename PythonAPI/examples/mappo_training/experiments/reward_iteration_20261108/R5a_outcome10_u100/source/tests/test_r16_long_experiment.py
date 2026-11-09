import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class R16LongTests(unittest.TestCase):
    def test_tick_segments_preserve_identity_and_optimizer_progress(self):
        import carla
        from tests.test_first_point_experiment import FakeTeamEnv
        from training.mut_mappo_config import load_config,portable_config,validate_config,make_actors
        from training.train_mut_mappo import train,Runtime
        cfg=portable_config(load_config('experiments/adversarial_u1000_learning_diagnosis_20261104_v3/configs/R16.json'))
        cfg['run']['train_tick_budget']=1000
        cfg['learning'].update(ppo_epochs=1,minibatch_size=128)
        cfg=validate_config(cfg)
        class Env(FakeTeamEnv):
            def step(self,controls):
                obs,rewards,dones,info=super().step(controls)
                info['mut_metrics'].update(safe_critical_event=0.,critical_event=False,
                    critical_exposure_s=0.,critical_tet_s=0.,critical_tit_s=0.,team_reward_components={})
                return obs,rewards,dones,info
        def factory(config,checkpoint=None,condition='C2',team='A',audit_callback=None):
            source=config['_red_payload']
            red=make_actors(source['actors'],source['training_config'],(0,1))
            blue=make_actors({2:source['actors'][0],3:source['actors'][1]},source['training_config'],(2,3))
            return Runtime(config,Env(audit_callback,team,condition),red,blue,condition,team)
        with tempfile.TemporaryDirectory() as tmp,patch.object(carla,'Client',side_effect=AssertionError('offline test called CARLA')),contextlib.redirect_stdout(io.StringIO()):
            run=Path(tmp)/'run'
            first=train(cfg,run,stop_train_ticks=2,env_factory=factory)
            identity=json.loads((run/'checkpoint_manifest.json').read_text())['identity']
            self.assertEqual(first['completed_round'],1)
            self.assertLess(first['counters']['train'],1000)
            second=train(cfg,run,resume=True,stop_train_ticks=200,env_factory=factory)
            self.assertEqual(second['completed_round'],2)
            self.assertEqual(identity,json.loads((run/'checkpoint_manifest.json').read_text())['identity'])
            for team in ('A','B'):
                self.assertEqual(second['teams'][team]['update_count'],2)
                self.assertGreater(second['teams'][team]['schedule_state']['actor_lr'],.99*cfg['learning']['lr_actor'])
                for aid in (2,3):
                    first_steps=[v['step'].item() for v in first['teams'][team]['actor_optims'][aid]['state'].values()]
                    later_steps=[v['step'].item() for v in second['teams'][team]['actor_optims'][aid]['state'].values()]
                    self.assertGreater(min(later_steps),max(first_steps))
            status=json.loads((run/'session/status.json').read_text())
            self.assertTrue(status['segment_tick_cap_reached'])
            self.assertFalse(status['training_budget_reached'])

    def test_shutdown_refuses_incomplete_pipeline(self):
        from utils.r16_long_experiment import close_server
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'control').mkdir()
            (root/'control/pipeline_status.json').write_text('{"status":"RUNNING"}')
            with patch('utils.r16_long_experiment.powershell',side_effect=AssertionError('should not stop process')):
                with self.assertRaises(ValueError):close_server(root)


if __name__=='__main__':unittest.main()
