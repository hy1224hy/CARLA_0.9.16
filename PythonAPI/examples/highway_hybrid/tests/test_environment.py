import unittest
import numpy as np
from highway_hybrid.config import HybridConfig
from highway_hybrid.environment import HybridEnv


class HybridEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env=HybridEnv(HybridConfig(level=0))
        self.obs,_=self.env.reset(seed=0,options={'family':'follow'})

    def tearDown(self):self.env.close()

    def test_observation_and_boundary_masks(self):
        self.assertEqual(self.obs['state'].shape,(28,))
        np.testing.assert_array_equal(self.obs['lane_mask'],[True,False,True])
        self.assertTrue(self.env.observation_space.contains(self.obs))
        with self.assertRaises(ValueError):self.env.step(dict(acceleration=[0.],lane=1))

    def test_lane_commitment_completes_then_allows_reverse(self):
        obs,_,done,_,_=self.env.step(dict(acceleration=[0.],lane=2))
        self.assertFalse(done)
        np.testing.assert_array_equal(obs['lane_mask'],[True,False,False])
        with self.assertRaises(ValueError):self.env.step(dict(acceleration=[0.],lane=1))
        for _ in range(35):
            obs,_,done,_,info=self.env.step(dict(acceleration=[0.],lane=0))
            self.assertFalse(done)
            if not info['lane_busy']:break
        self.assertFalse(info['lane_busy'])
        self.assertLess(abs(self.env.attacker.position[1]-4.),.35)
        np.testing.assert_array_equal(obs['lane_mask'],[True,True,False])
        self.env.step(dict(acceleration=[0.],lane=1))
        for _ in range(35):
            obs,_,done,_,info=self.env.step(dict(acceleration=[0.],lane=0))
            if not info['lane_busy']:break
        self.assertFalse(done)
        self.assertLess(abs(self.env.attacker.position[1]),.35)

    def test_longitudinal_action_remains_continuous(self):
        left=HybridEnv(HybridConfig(level=0));right=HybridEnv(HybridConfig(level=0))
        left.reset(seed=2,options={'family':'adjacent'});right.reset(seed=2,options={'family':'adjacent'})
        left.step(dict(acceleration=[-.2],lane=0));right.step(dict(acceleration=[.2],lane=0))
        self.assertLess(left.attacker.speed,right.attacker.speed)
        self.assertTrue(left.target.enable_lane_change)
        np.testing.assert_array_equal(left.reference.vehicles[1].position,right.reference.vehicles[1].position)

    def test_reset_clears_controller_commitment(self):
        self.env.step(dict(acceleration=[0.],lane=2))
        obs,info=self.env.reset(seed=0,options={'family':'adjacent'})
        self.assertFalse(info['lane_busy'])
        np.testing.assert_array_equal(obs['lane_mask'],[True,True,False])

    def test_controller_settle_history_is_visible_in_observation(self):
        self.env.step(dict(acceleration=[0.],lane=2))
        first=self.env._observation()['state'].copy()
        self.env._settled_time=self.env.config.lane_settle_duration/2
        second=self.env._observation()['state']
        self.assertEqual(first[-1],1.)
        self.assertEqual(second[-1],1.5)
        np.testing.assert_array_equal(first[:-1],second[:-1])


if __name__=='__main__':unittest.main()
