import tempfile
import unittest
from pathlib import Path


class StraightCLITests(unittest.TestCase):
    def test_plan_is_balanced_and_does_not_train(self):
        from highwaytest.straight.__main__ import write_plan
        import json
        with tempfile.TemporaryDirectory() as folder:
            paths=write_plan(Path(folder)/'configs',steps=1024,seeds=[42,43,44])
            self.assertEqual(len(paths),18)
            configs=[json.loads(p.read_text()) for p in paths]
            self.assertEqual({c['level'] for c in configs},{1,2,3})
            self.assertEqual({c['method'] for c in configs},{'independent','mutual'})
            self.assertEqual({c['total_steps'] for c in configs},{1024})
            self.assertFalse(list(Path(folder).rglob('*.pt')))
            with self.assertRaises(FileExistsError):
                write_plan(Path(folder)/'configs',steps=1024,seeds=[42])

    def test_parser_requires_subcommand(self):
        from highwaytest.straight.__main__ import parser
        with self.assertRaises(SystemExit):
            parser().parse_args([])

    def test_existing_matrix_cannot_silently_ignore_new_budget(self):
        from highwaytest.straight.__main__ import write_plan,check_matrix
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'configs';write_plan(path,steps=1024,seeds=[42])
            with self.assertRaises(ValueError):check_matrix(path,steps=2048,seeds=[42])
            with self.assertRaises(ValueError):check_matrix(path,steps=1024,seeds=[43])

    def test_beta_and_reward_studies_change_real_objectives(self):
        from highwaytest.straight.__main__ import write_study_plan,load_config
        with tempfile.TemporaryDirectory() as folder:
            beta=write_study_plan(Path(folder)/'beta','beta',1024,[42])
            self.assertEqual({load_config(p).beta for p in beta},{0.005,.08})
            self.assertTrue(all(load_config(p).method=='mutual' for p in beta))
            reward=write_study_plan(Path(folder)/'reward','reward',1024,[42])
            self.assertEqual({(load_config(p).risk_weight,load_config(p).event_weight) for p in reward},{(0.,1.),(1.,0.)})
            self.assertTrue(all(load_config(p).method=='mutual' for p in reward))

    def test_check_rejects_stale_eval_cases(self):
        import json
        from highwaytest.straight.__main__ import write_plan,check_matrix
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'configs';write_plan(path,steps=1024,seeds=[42])
            stale=json.loads((path/'L1_independent_s42.json').read_text())
            stale['eval_cases']=16
            (path/'L1_independent_s42.json').write_text(json.dumps(stale))
            with self.assertRaises(ValueError):check_matrix(path,steps=1024,seeds=[42])
