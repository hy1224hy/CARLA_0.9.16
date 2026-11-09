import unittest
from dataclasses import replace
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch
from highwaytest.straight.model import ActorCritic
from highwaytest.straight.learning import peer_loss, beta_at
from highwaytest.straight.runner import rank_report


class TrainingTests(unittest.TestCase):
    def test_round_boundary_resume_matches_uninterrupted_weights(self):
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import train,load_state
        cfg=StraightConfig(total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,
            warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'):
            full=Path(folder)/'full';partial=Path(folder)/'partial'
            train(cfg,full)
            train(cfg,partial,max_rounds=1)
            train(cfg,partial,resume=partial/'latest.pt')
            a,b=load_state(full/'latest.pt'),load_state(partial/'latest.pt')
            self.assertEqual(a['train_steps'],b['train_steps'])
            for replica in ('A','B'):
                for key in a['models'][replica]:torch.testing.assert_close(a['models'][replica][key],b['models'][replica][key],rtol=0,atol=0)

    def test_same_input_peer_is_frozen(self):
        a,b=ActorCritic(),ActorCritic()
        obs=torch.randn(9,22)
        loss=peer_loss(a,b,obs)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in a.parameters()))
        self.assertTrue(all(p.grad is None for p in b.parameters()))

    def test_action_probability(self):
        model=ActorCritic()
        obs=torch.randn(7,22)
        action,z,lp=model.sample(obs)
        self.assertTrue((action.abs()<=1).all())
        torch.testing.assert_close(lp,model.log_prob(model.distribution(obs),z))

    def test_beta_total_clock(self):
        self.assertEqual(beta_at(10,.02,10,10),0)
        self.assertAlmostEqual(beta_at(15,.02,10,10),.01)
        self.assertAlmostEqual(beta_at(20,.02,10,10),.02)

    def test_collision_success_not_rejected_for_l3(self):
        good={'attack_success_rate':1.,'invalid_rate':0.,'collision_rate':1.,'utility':0.}
        empty={'attack_success_rate':0.,'invalid_rate':0.,'collision_rate':0.,'utility':0.}
        self.assertGreater(rank_report(good,3),rank_report(empty,3))
        self.assertLess(rank_report(good,1),rank_report(empty,1))

    def test_normal_selection_uses_normal_success(self):
        good={'success_rate':1.,'attack_success_rate':0.,'invalid_rate':0.,'collision_rate':0.}
        poor=dict(good,success_rate=0.)
        self.assertGreater(rank_report(good,0),rank_report(poor,0))

    def test_summary_records_cumulative_wall_seconds(self):
        import json
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import train
        cfg=StraightConfig(total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'):
            out=Path(folder)/'run'
            train(cfg,out,max_rounds=1)
            first=json.loads((out/'summary.json').read_text())
            self.assertGreater(first['wall_seconds'],0)
            train(cfg,out,resume=out/'latest.pt')
            second=json.loads((out/'summary.json').read_text())
            self.assertGreaterEqual(second['wall_seconds'],first['wall_seconds'])

    def test_resume_rejects_source_change(self):
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import train
        cfg=StraightConfig(total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'run'
            with patch('highwaytest.straight.runner.source_digest',return_value='src-A'):
                train(cfg,out)
            with patch('highwaytest.straight.runner.source_digest',return_value='src-B'):
                with self.assertRaises(ValueError):train(cfg,out,resume=out/'latest.pt')

    def test_evaluate_cases_default_per_split(self):
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import train,evaluate
        cfg=StraightConfig(total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'):
            out=Path(folder)/'run';train(cfg,out)
            self.assertEqual(evaluate(out/'selected.pt',split='selection')['cases'],32)
            self.assertEqual(evaluate(out/'selected.pt',split='test')['cases'],100)

    def test_capture_exports_three_representative_trajectories(self):
        import json
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import train,evaluate
        cfg=StraightConfig(total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'):
            out=Path(folder)/'run';train(cfg,out)
            evaluate(out/'selected.pt',split='test',cases=4,output=Path(folder)/'test',gif=True)
            traj=json.loads((Path(folder)/'test'/'trajectory.json').read_text())
            self.assertLessEqual(len(traj['cases']),3)
            self.assertEqual(len({c['case_seed'] for c in traj['cases']}),len(traj['cases']))
            self.assertEqual(traj['cases'][0]['case_seed'],3000000)
            step=traj['cases'][0]['steps'][0]
            for key in ('time','attacker_position','target_position','attacker_speed','target_speed'):
                self.assertIn(key,step)
            ev=json.loads((Path(folder)/'test'/'evaluation.json').read_text())
            self.assertEqual(ev['train_steps_total'],16)

    def test_attack_bootstrap_produces_level_initializers(self):
        import json as _json
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import bootstrap
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)/'base'
            bootstrap(StraightConfig(level=0,seed=42),base,episodes=32,epochs=10,dagger_rounds=0)
            for level,script in ((1,'block'),(2,'cut'),(3,'collision')):
                result=bootstrap(StraightConfig(level=0,seed=42),base,attack_level=level)
                path=base/f'attack_L{level}.pt'
                self.assertTrue(path.is_file(),(level,result))
                state=torch.load(path,map_location='cpu',weights_only=False)
                self.assertEqual(state['attack_level'],level)
                self.assertIn('actor',state)
                qual=state['qualification']['overall']
                if level==1:self.assertGreater(qual['blocking_time'],0.,state['qualification'])
                elif level==2:self.assertGreater(qual['cut_in_rate'],0.,state['qualification'])
                else:self.assertGreater(qual['collision_rate'],.5,state['qualification'])

    def test_attack_initializer_accepted_by_matching_level_train(self):
        import json as _json
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import bootstrap,train,load_state
        cfg=StraightConfig(total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=20.,eval_cases=8,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'):
            base=Path(folder)/'base'
            bootstrap(cfg,base,episodes=32,epochs=10,dagger_rounds=0)
            bootstrap(cfg,base,attack_level=1)
            train(replace(cfg,level=1),Path(folder)/'run',initialize_from=base/'attack_L1.pt')
            self.assertTrue((Path(folder)/'run'/'summary.json').is_file())

    def test_bootstrap_produces_qualified_driver(self):
        import numpy as np
        from highwaytest.straight.config import StraightConfig
        from highwaytest.straight.runner import bootstrap
        with tempfile.TemporaryDirectory() as folder:
            result=bootstrap(StraightConfig(level=0,seed=42),Path(folder)/'base',episodes=128,epochs=40)
            self.assertTrue(result['qualified'],result)
            self.assertTrue((Path(folder)/'base'/'driver.pt').is_file())
            with np.load(Path(folder)/'base'/'demonstrations.npz') as data:
                dx=data['obs'][:,11]*60
                self.assertLess(dx.min(),-30,'demonstrations must cover normal-speed follow with attacker ahead')
                ea=data['obs'][:,4]*2
                self.assertGreater(float(np.abs(ea).max()),.5,'demonstrations must cover lateral recovery')
