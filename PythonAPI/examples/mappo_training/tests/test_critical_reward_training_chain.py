"""Real PPO plumbing on synthetic dynamics, explicitly not driving evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CriticalTrainingChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_mut_mappo_runner import RunnerIntegrationTests
        RunnerIntegrationTests.setUpClass()
        cls.addClassCleanup(RunnerIntegrationTests.doClassCleanups)
        cls.synthetic_red=RunnerIntegrationTests.config['red_checkpoint']

    def test_new_profile_complete_ab_update_and_source_identity_without_carla(self):
        self.run_profile('critical_interaction_team_v1')

    def test_v2_complete_ab_update_and_source_identity_without_carla(self):
        self.run_profile('normal_driving_critical_team_v2')

    def run_profile(self, profile):
        import carla
        from tests.test_first_point_experiment import FakeTeamEnv
        from training.mut_mappo_config import load_config,portable_config,validate_config,make_actors,cases_for
        from training.train_mut_mappo import Runtime,train
        from environment.mut_mappo_efficiency_reward import reference_identity,ReferenceStore
        class Env(FakeTeamEnv):
            def step(self,controls):
                obs,rewards,dones,info=super().step(controls)
                info['mut_metrics'].update(safe_critical_event=1.,critical_event=True,
                    critical_exposure_s=.2,critical_tet_s=.3,critical_tit_s=.1,
                    team_reward_components={'critical':.5})
                return obs,rewards,dones,info
        with tempfile.TemporaryDirectory() as folder, patch.object(carla,'Client',side_effect=AssertionError('offline test opened Client')):
            root=Path(folder)
            cfg=portable_config(load_config('configs/mut_mappo/local_candidate_v1.json'))
            cfg['red_checkpoint']=dict(self.synthetic_red)
            cfg['learning']['gamma']=.999
            cfg['task'].update(reward_profile=profile,c1_controller='lane_pid_v1',
                team_reward={'reference_dir':str(root/'references'),'attack_weight':2.,'gamma':cfg['learning']['gamma']})
            cfg['method']='off'; cfg['mutual']['quality_metric']='critical'
            cfg['mutual'].update(schedule_clock='rounds',warmup_rounds=1,refresh_rounds=10)
            cfg['run'].update(rounds=1,episodes_per_team_update=4,stop_rule='rounds',schedule_clock='rounds')
            cfg['learning'].update(ppo_epochs=1,minibatch_size=16)
            cfg=validate_config(cfg)
            refs=root/'references';refs.mkdir()
            header={'identity':reference_identity(cfg),'cases':{}}
            for split in ('train','gate'):
                for case in cases_for(cfg,split):
                    key=case['case_key'];path=refs/(hashlib.sha256(key.encode()).hexdigest()+'.json')
                    path.write_text(json.dumps({'case_key':key,'progress':{'0':[0.,1.],'1':[0.,1.]},
                        'scope':'SYNTHETIC_UNIT_TEST_ONLY'}),encoding='utf-8')
                    header['cases'][key]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            (refs/'manifest.json').write_text(json.dumps(header),encoding='utf-8')
            ReferenceStore(refs,reference_identity(cfg)).curve(cases_for(cfg,'train')[0]['case_key'])
            def factory(config,checkpoint=None,condition='C2',team='A',audit_callback=None):
                source=config['_red_payload']
                red=make_actors(source['actors'],source['training_config'],(0,1))
                blue=make_actors({2:source['actors'][0],3:source['actors'][1]},source['training_config'],(2,3))
                self.assertTrue(all(not p.requires_grad for a in red.values() for p in a.parameters()))
                return Runtime(config,Env(audit_callback,team,condition),red,blue,condition,team)
            import contextlib,io
            with contextlib.redirect_stdout(io.StringIO()):
                result=train(cfg,root/'run',env_factory=factory)
            self.assertEqual(result['completed_round'],1)
            manifest=json.loads((root/'run/checkpoint_manifest.json').read_text(encoding='utf-8'))
            self.assertIn('@carla/agents/navigation/controller.py',manifest['identity']['source_hashes'])
            if profile == 'normal_driving_critical_team_v2':
                self.assertIn('environment/mut_mappo_critical_reward_v2.py',manifest['identity']['source_hashes'])
            for team in ('A','B'):
                self.assertEqual(result['teams'][team]['update_count'],1)
                self.assertEqual(result['last_metrics']['training_window'][team]['safe_critical_event'],1.)
                self.assertEqual(result['last_metrics']['training_window'][team]['reward_components_mean']['critical'],.5)


if __name__=='__main__':unittest.main()
