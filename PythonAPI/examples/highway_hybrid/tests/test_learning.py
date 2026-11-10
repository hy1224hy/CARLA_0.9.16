import copy
import unittest
import numpy as np
import torch
from highway_hybrid.config import HybridConfig
from highway_hybrid.environment import HybridEnv
from highway_hybrid.model import HybridActorCritic
from highway_hybrid.learning import collect,update


class HybridLearningTests(unittest.TestCase):
    def test_real_rollout_and_joint_ppo_with_frozen_peer(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        cfg=HybridConfig(level=3,duration=2.,hidden=32,rollout_steps=24,minibatch=12,epochs=2)
        model=HybridActorCritic(32);teacher=copy.deepcopy(model).eval().requires_grad_(False)
        initial=copy.deepcopy(teacher.state_dict())
        env=HybridEnv(cfg)
        batch,episodes=collect(model,env,24,np.random.default_rng(42))
        self.assertEqual(batch['mask'].shape,(24,3))
        self.assertGreater(len(episodes),0)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
        result=update(model,optimizer,batch,cfg,teacher,.02,.03)
        self.assertLess(result['ratio_error'],1e-4)
        self.assertEqual(result['effective_beta_lane'],.03)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        for key,value in initial.items():
            torch.testing.assert_close(value,teacher.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertGreater(result['lane_decision_fraction'],0.)
        env.close()


if __name__=='__main__':unittest.main()
