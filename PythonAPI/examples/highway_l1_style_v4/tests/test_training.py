import tempfile
import unittest
from pathlib import Path
from highway_l1_style_v4.config import HybridConfig
from highway_l1_style_v4.runner import train,evaluate_suite,load_checkpoint


class TrainingContractTests(unittest.TestCase):
    def test_checkpoint_resume_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cfg=HybridConfig(level=1,total_steps=128,rollout_steps=32,minibatch=32,epochs=1,
                eval_cases=2,eval_interval=128,hidden=16,warmup_steps=0,ramp_steps=0)
            train(cfg,root/'run',max_rounds=1)
            result=train(cfg,root/'run',resume=root/'run'/'latest.pt')
            self.assertEqual(result['train_steps'],128)
            state=load_checkpoint(root/'run'/'latest.pt')
            self.assertEqual(state['protocol'],'highway_l1_style_twin_v4')
            audit=evaluate_suite(root/'run',root/'audit','development',2,'latest','A')
            self.assertEqual(audit[0]['overall']['cases'],2)
            self.assertEqual(audit[0]['overall']['behind_positive_impact'],0.)


if __name__=='__main__':unittest.main()
