import json
from pathlib import Path
import tempfile
import unittest
import torch
from highway_hybrid.config import HybridConfig
from highway_hybrid.runner import train,load_checkpoint,evaluate_suite


class HybridRunnerTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_evaluation_freezes_checkpoint(self):
        cfg=HybridConfig(level=3,total_steps=32,rollout_steps=4,epochs=1,minibatch=4,
                         hidden=16,duration=1.,eval_cases=2,eval_interval=16,warmup_steps=0,ramp_steps=0)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            train(cfg,root/'full')
            train(cfg,root/'partial',max_rounds=2)
            train(cfg,root/'partial',resume=root/'partial'/'latest.pt')
            left=load_checkpoint(root/'full'/'latest.pt')
            right=load_checkpoint(root/'partial'/'latest.pt')
            self.assertEqual(left['train_steps'],32)
            for team in ('A','B'):
                for key in left['models'][team]:
                    torch.testing.assert_close(left['models'][team][key],right['models'][team][key],atol=0,rtol=0)
            result=evaluate_suite(root/'partial',root/'audit','development',2,'latest','B')
            self.assertEqual(result[0]['overall']['cases'],2)
            self.assertTrue((root/'audit'/'partial'/'checkpoint.pt').exists())
            self.assertEqual(result[0]['config']['state_dim'],28)
            from highway_eval.evaluate import evaluate_suite as unified_evaluate
            routed=unified_evaluate(root/'partial',root/'unified','development',2,'latest','A')
            self.assertEqual(routed[0]['protocol'],'highway_hybrid_twin_v1')
            with self.assertRaises(FileExistsError):
                evaluate_suite(root/'partial',root/'audit','development',2,'latest','B')

    def test_old_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'old.pt'
            torch.save(dict(protocol='straight_twin_v1'),path)
            with self.assertRaises(ValueError):load_checkpoint(path)

    def test_config_rejects_unsupported_action_model(self):
        with self.assertRaises(ValueError):HybridConfig(learner='sac').validate()
        with self.assertRaises(ValueError):HybridConfig(obs_dim=24).validate()


if __name__=='__main__':unittest.main()
