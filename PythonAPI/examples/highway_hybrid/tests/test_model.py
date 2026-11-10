import unittest
import torch
from highway_hybrid.model import HybridActorCritic, mutual_terms


class HybridModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model=HybridActorCritic(32)
        self.obs=torch.randn(12,28)
        self.mask=torch.tensor([[True,False,True]]*12)

    def test_joint_probability_and_masked_sampling(self):
        acceleration,lane,z,old=self.model.sample(self.obs,self.mask)
        self.assertEqual(acceleration.shape,(12,1))
        self.assertFalse((lane==1).any())
        self.assertTrue((acceleration.abs()<=1).all())
        torch.testing.assert_close(old,self.model.log_prob(self.obs,self.mask,z,lane))
        normal,category=self.model.distributions(self.obs,self.mask)
        expected=self.model.acceleration_log_prob(normal,z)+category.log_prob(lane)
        torch.testing.assert_close(old,expected)

    def test_busy_mask_has_zero_lane_entropy_and_kl(self):
        mask=torch.tensor([[True,False,False]]*12)
        peer=HybridActorCritic(32)
        _,category=self.model.distributions(self.obs,mask)
        self.assertTrue((category.sample()==0).all())
        torch.testing.assert_close(category.entropy(),torch.zeros(12),atol=1e-6,rtol=0)
        _,lane_kl=mutual_terms(self.model,peer,self.obs,mask)
        self.assertAlmostEqual(float(lane_kl.detach()),0.,places=6)

    def test_mutual_peer_is_frozen_and_heads_receive_gradients(self):
        peer=HybridActorCritic(32)
        acceleration,lane=mutual_terms(self.model,peer,self.obs,self.mask)
        (acceleration+lane).backward()
        self.assertIsNotNone(self.model.acceleration_head.weight.grad)
        self.assertIsNotNone(self.model.lane_head.weight.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in peer.parameters()))

    def test_empty_action_mask_rejected(self):
        with self.assertRaises(ValueError):
            self.model.distributions(self.obs,torch.zeros_like(self.mask))


if __name__=='__main__':unittest.main()
