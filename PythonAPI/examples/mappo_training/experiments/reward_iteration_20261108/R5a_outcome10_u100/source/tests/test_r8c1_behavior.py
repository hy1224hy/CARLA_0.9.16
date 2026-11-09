import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from evaluation.diagnose_r8c1_behavior import variants, make_actors, ROOT, RUN
from training.checkpoint_protocol import load_checkpoint_file


class BehaviorDiagnosticTests(unittest.TestCase):
    def test_variant_chain_changes_only_planned_fields(self):
        p=ROOT/'data/training_scenarios_town05_r8a_v1/scenario_01_north_short_rolling_a0_left.json'
        s=json.loads(p.read_text()); original=copy.deepcopy(s)
        a,b,c=variants(s).values()
        self.assertEqual(s,original)
        self.assertEqual(a['episode_max_steps'],200)
        self.assertEqual(a['agents'],s['agents'])
        self.assertEqual(a['tags'],s['tags'])
        expected=copy.deepcopy(a); expected['episode_max_steps']=500
        self.assertEqual(b,expected)
        expected=copy.deepcopy(b); expected['agents']['agent1']['initial_speed_mps']=4.0
        self.assertEqual(c,expected)
        self.assertAlmostEqual(a['agent_goal_zones']['agent0']['y']-s['agent_goal_zones']['agent0']['y'],30)

    def test_actual_checkpoint_actor_distribution_matches_training_bounds(self):
        c,_=load_checkpoint_file(RUN/'periodic/ckpt_update_000185.pt',map_location='cpu')
        actors=make_actors(c)
        for aid,a in actors.items():
            self.assertFalse(a.training)
            self.assertFalse(any(p.requires_grad for p in a.parameters()))
            self.assertEqual(a.min_log_std,c['training_config']['min_log_std'])
            self.assertEqual(a.max_log_std,c['training_config']['max_log_std'])
            x=torch.zeros(1,20)
            torch.manual_seed(62001); first=a.sample_action_z(x)[0]
            torch.manual_seed(62001); second=a.sample_action_z(x)[0]
            self.assertTrue(torch.equal(first,second))
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue((first.abs()<=1).all())
            mu,_=a(x)
            self.assertTrue(torch.equal(a.get_deterministic_action(x),mu.tanh()))


if __name__=='__main__': unittest.main()
