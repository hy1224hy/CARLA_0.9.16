import json
from pathlib import Path
import tempfile
import unittest
import torch
from highwaytest.straight.config import StraightConfig
from highwaytest.straight.model import ActorCritic
from highwaytest.training.checkpoint import source_digest
from highway_eval.evaluate import case_contract, summarize, evaluate_suite


class AuditTests(unittest.TestCase):
    def test_case_sets_are_balanced_reproducible_and_disjoint(self):
        dev=case_contract('development',4)
        test=case_contract('test',4)
        self.assertEqual(dev,case_contract('development',4))
        self.assertEqual([c['family'] for c in dev],['follow','adjacent']*2)
        self.assertTrue({c['seed'] for c in dev}.isdisjoint(c['seed'] for c in test))
        with self.assertRaises(ValueError):case_contract('development',3)

    def test_failures_remain_in_success_denominator(self):
        rows=[dict(attack_success=True,invalid=False,cut_count=2,cut_diag={}),
              dict(attack_success=False,invalid=True,cut_count=0,cut_diag={})]
        result=summarize(rows)
        self.assertEqual(result['attack_success_rate'],.5)
        self.assertEqual(result['invalid_rate'],.5)
        self.assertEqual(result['repeated_cut_rate'],.5)

    def test_real_rollout_snapshot_and_source_rejection(self):
        torch.set_num_threads(1)
        cfg=StraightConfig(level=3,duration=.4,hidden=16)
        model=ActorCritic(cfg.hidden,cfg.obs_dim)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'runs'/'L3';run.mkdir(parents=True)
            state=dict(protocol='straight_twin_v1',config=cfg.to_dict(),
                       source_sha256=source_digest(),model=model.state_dict(),
                       selection={'team':'A','train_steps':16})
            checkpoint=run/'selected.pt';torch.save(state,checkpoint)
            result=evaluate_suite(root/'runs',root/'audit','development',2,'selected','A')
            self.assertEqual(result[0]['overall']['cases'],2)
            self.assertTrue((root/'audit'/'L3'/'checkpoint.pt').exists())
            trace=json.loads((root/'audit'/'L3'/'trajectories.jsonl').read_text().splitlines()[0])
            self.assertEqual(len(trace['steps']),2)
            state['source_sha256']='different';torch.save(state,checkpoint)
            with self.assertRaises(ValueError):
                evaluate_suite(root/'runs',root/'rejected','development',2,'selected','A')


if __name__=='__main__':unittest.main()
