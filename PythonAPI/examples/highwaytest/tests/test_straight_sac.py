import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from highwaytest.straight.config import StraightConfig
from highwaytest.straight.model import ActorCritic
from highwaytest.straight.sac import SACActorCritic,ReplayBuffer,sac_update


class SACTests(unittest.TestCase):
    def test_learner_validation_and_dispatch(self):
        self.assertEqual(StraightConfig(learner='sac').validate().learner,'sac')
        self.assertEqual(StraightConfig(learner='ppo').validate().learner,'ppo')
        with self.assertRaises(ValueError):StraightConfig(learner='x').validate()

    def test_sac_update_reduces_q_loss_and_is_finite(self):
        torch.manual_seed(0)
        cfg=StraightConfig(learner='sac',alpha=.2)
        model=SACActorCritic(hidden=16)
        target=SACActorCritic(hidden=16);target.load_state_dict(model.state_dict());target.eval()
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        buffer=ReplayBuffer(capacity=4000)
        rng=np.random.default_rng(7)
        for i in range(2000):
            buffer.push(rng.standard_normal(22).astype(np.float32),rng.standard_normal(2).astype(np.float32),
                        float(rng.standard_normal()),rng.standard_normal(22).astype(np.float32),bool(i%5==0))
        first=None
        for _ in range(150):
            stats=sac_update(model,target,optimizer,buffer,256,cfg,None,0.,rng)
            self.assertTrue(np.isfinite(list(stats.values())).all(),stats)
            if first is None:first=stats['q_loss']
        self.assertLess(stats['q_loss'],first,'Q loss should decrease over updates')
        self.assertGreater(first,0.)

    def test_sac_mutual_peer_frozen_and_enters_loss(self):
        import copy
        cfg=StraightConfig(learner='sac',beta=.02,alpha=.2)
        a=SACActorCritic(hidden=16);b=SACActorCritic(hidden=16)
        teacher=SACActorCritic(hidden=16);teacher.load_state_dict(b.state_dict())
        teacher.eval().requires_grad_(False)
        target=copy.deepcopy(a).eval().requires_grad_(False)
        opt=torch.optim.Adam(a.parameters(),lr=3e-4)
        buffer=ReplayBuffer(capacity=1000)
        rng=np.random.default_rng(3)
        for i in range(500):
            buffer.push(rng.standard_normal(22).astype(np.float32),rng.standard_normal(2).astype(np.float32),
                        float(rng.standard_normal()),rng.standard_normal(22).astype(np.float32),bool(i%5==0))
        before=[p.detach().clone() for p in teacher.parameters()]
        stats=sac_update(a,target,opt,buffer,128,cfg,teacher,.02,rng)
        self.assertTrue(np.isfinite(stats['mutual_kl']))
        self.assertGreater(stats['mutual_kl'],0.)
        for old,new in zip(before,teacher.parameters()):
            torch.testing.assert_close(new,old)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_sac_twin_train_completes_and_evaluate_loads(self):
        from highwaytest.straight.runner import train,evaluate,load_state
        cfg=StraightConfig(learner='sac',total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,\
             patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'),\
             patch('highwaytest.straight.sac.source_digest',return_value='fixed-test-source'):
            out=Path(folder)/'run'
            train(cfg,out)
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['train_steps'],16)
            state=load_state(out/'latest.pt')
            self.assertEqual(state['learner'],'sac')
            self.assertIn('buffers',state)
            self.assertIn('targets',state)
            result=evaluate(out/'selected.pt',split='selection')
            self.assertEqual(result['cases'],32)

    def test_sac_resume_matches_uninterrupted_weights(self):
        from highwaytest.straight.runner import train,load_state
        cfg=StraightConfig(learner='sac',total_steps=16,rollout_steps=4,epochs=1,minibatch=4,
            hidden=16,duration=1.,eval_cases=2,eval_interval=8,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder,\
             patch('highwaytest.straight.runner.source_digest',return_value='fixed-test-source'),\
             patch('highwaytest.straight.sac.source_digest',return_value='fixed-test-source'):
            full=Path(folder)/'full';partial=Path(folder)/'partial'
            train(cfg,full)
            train(cfg,partial,max_rounds=1)
            train(cfg,partial,resume=partial/'latest.pt')
            a,b=load_state(full/'latest.pt'),load_state(partial/'latest.pt')
            self.assertEqual(a['train_steps'],b['train_steps'])
            for replica in ('A','B'):
                for key in a['models'][replica]:
                    torch.testing.assert_close(a['models'][replica][key],b['models'][replica][key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
