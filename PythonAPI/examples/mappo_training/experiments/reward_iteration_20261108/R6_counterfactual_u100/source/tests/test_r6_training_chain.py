"""Actual R6 rollout/PPO/checkpoint plumbing on synthetic dynamics, never driving evidence."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tests.test_mut_mappo_critical_reward import snapshot


class SyntheticEnv:
    def __init__(self,callback):
        self.callback=callback;self.num_agents=4;self._world_tick_counts=dict(total=0,setup=0,decision=0)
    def tick(self,kind):
        self._world_tick_counts[kind]+=1;self._world_tick_counts['total']+=1
        self.callback(dict(kind=kind,ticks=1))
    def reset_case(self,case):
        self.n=0;self.tick('setup');self.progress=0.
        self.obs={i:np.zeros(20,np.float32) for i in range(4)}
        states={str(i):dict(location=dict(x=float(i*7),y=0.,z=.5),speed=3.,rotation=dict(yaw=0.)) for i in range(4)}
        return self.obs,dict(active_mask=dict.fromkeys(range(4),True),case_geometry=dict(realized_initial_states=states))
    def get_global_state(self):return np.concatenate(list(self.obs.values()))
    def set_cost_coefficient(self,value):pass
    def close(self):pass
    def step(self,controls):
        self.n+=1;self.tick('decision');done=self.n==4
        self.progress=min(1.,self.progress+.15+.03*sum(c[0] for i,c in controls.items() if i in (2,3)))
        self.obs={i:np.full(20,.01*self.n,np.float32) for i in range(4)}
        snaps={i:snapshot(i*7+.3*self.n,vx=3.) for i in range(4)}
        parts={str(i):dict(control_cost=-.01*self.n,outcome_reward=0.) for i in (2,3)}
        metrics=dict(valid=True,red_success=False,blue_eligible=True,response=False,
            route_progress={str(i):self.progress for i in range(4)},contact_events=[],
            safety=dict(per_agent={'2':{},'3':{}}),safe_efficiency_utility=0.,
            individual_reward_components=parts,team_reward_components=dict(control_cost=-.01*self.n,outcome_reward=0.),
            team_utility_total=-.01*self.n,safe_critical_event=0.,critical_exposure_s=0.,critical_tet_s=0.,critical_tit_s=0.)
        info=dict(active_mask=dict.fromkeys(range(4),not done),agent_terminated=dict.fromkeys(range(4),done),
            agent_truncated=dict.fromkeys(range(4),False),episode_end_reason='all_agents_reached' if done else 'running',
            post_tick_kinematics=snaps,mut_metrics=metrics,world_tick_counts=dict(self._world_tick_counts))
        return self.obs,dict.fromkeys(range(4),-.01),dict.fromkeys(range(4),done),info


class R6ChainTests(unittest.TestCase):
    def test_complete_ab_update_has_only_actual_ppo_data_and_real_checkpoint(self):
        import carla
        from training.mut_mappo_config import load_config,validate_config,make_actors
        from training.train_mut_mappo import Runtime,train
        cfg=load_config(Path(__file__).resolve().parents[2]/'config.json')
        cfg['run'].update(rounds=1,episodes_per_team_update=4)
        cfg['learning'].update(ppo_epochs=1,minibatch_size=16)
        cfg=validate_config(cfg)
        def factory(config,checkpoint=None,condition='C2',team='A',audit_callback=None):
            p=config['_red_payload'];red=make_actors(p['actors'],p['training_config'],(0,1))
            blue=make_actors({2:p['actors'][0],3:p['actors'][1]},p['training_config'],(2,3))
            return Runtime(config,SyntheticEnv(audit_callback),red,blue,condition,team)
        with tempfile.TemporaryDirectory() as tmp,patch.object(carla,'Client',side_effect=AssertionError('offline test called CARLA')),contextlib.redirect_stdout(io.StringIO()):
            run=Path(tmp)/'run';result=train(cfg,run,env_factory=factory)
            actual=[json.loads(x) for x in (run/'episodes.jsonl').read_text().splitlines()]
            auxiliary=[json.loads(x) for x in (run/'counterfactual_episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(actual),8);self.assertEqual(len(auxiliary),16)
            self.assertTrue(all(x['r6_settled'] for x in actual))
            self.assertTrue(all(x['ppo_samples'] is False for x in auxiliary))
            self.assertEqual(result['completed_round'],1)
            self.assertEqual(result['counters']['train_decision'],96)
            self.assertTrue(all(r['individual_counterfactual_metrics'][str(i)]['eligible'] for r in actual for i in (2,3)))
            self.assertTrue(any(r['individual_counterfactual_metrics'][str(i)]['credit']>0 for r in actual for i in (2,3)))
            for team in ('A','B'):
                t=result['teams'][team]
                self.assertEqual(t['update_count'],1)
                self.assertEqual(tuple(t['critic']['net.0.weight'].shape),(128,42))
                self.assertTrue(any(not torch.equal(v,cfg['_red_payload']['critic'][k]) for k,v in t['critic'].items()))
            m=json.loads((run/'checkpoint_manifest.json').read_text())
            self.assertIn('training/r6_collection.py',m['identity']['source_hashes'])
            self.assertNotIn('training_reference_hashes',m['identity'])


    def test_four_mode_evaluation_reuses_two_runtimes_and_keeps_all_cases(self):
        import carla
        import training.train_mut_mappo as training
        from training.mut_mappo_config import load_config,validate_config,make_actors,cases_for
        import evaluation.evaluate_r6_counterfactual as evaluation
        cfg=validate_config(load_config(Path(__file__).resolve().parents[2]/'config.json'))
        case=cases_for(cfg,'selection')[0];created=[]
        def factory(config,checkpoint=None,condition='C2',team='A',audit_callback=None):
            p=config['_red_payload'];red=make_actors(p['actors'],p['training_config'],(0,1))
            blue=make_actors({2:p['actors'][0],3:p['actors'][1]},p['training_config'],(2,3))
            created.append(team)
            return training.Runtime(config,SyntheticEnv(audit_callback or (lambda e:None)),red,blue,condition,team)
        with tempfile.TemporaryDirectory() as tmp,patch.object(carla,'Client',side_effect=AssertionError('offline test called CARLA')),patch.object(training,'create_runtime',factory),patch.object(evaluation,'cases_for',return_value=[case]),patch('utils.mut_mappo_session.CarlaOwner',side_effect=lambda *a:contextlib.nullcontext()):
            folder=Path(tmp)/'evaluation'
            evaluation.evaluate(Path(__file__).resolve().parents[2]/'config.json','offline_factory_checkpoint',folder)
            analysis=json.loads((folder/'analysis.json').read_text());records=[json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()]
            self.assertEqual(created,['A','B']);self.assertEqual(analysis['records'],8)
            self.assertEqual(analysis['matched_groups'],2)
            self.assertTrue(all(x['four_mode_joint']['eligible'] for x in analysis['effects']))
            self.assertTrue(all((folder/r['red_outcome_timeline']).exists() for r in records))
            self.assertEqual(sum(bool(r.get('r6_settled')) for r in records),2)


if __name__=='__main__':unittest.main()
