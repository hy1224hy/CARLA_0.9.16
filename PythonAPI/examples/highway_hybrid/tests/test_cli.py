import tempfile
from pathlib import Path
import json
import unittest
from highway_hybrid.__main__ import parser,write_plan


class HybridCLITests(unittest.TestCase):
    def test_balanced_plan_and_distinct_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'configs'
            paths=write_plan(output,200000,[42,43,44],[1,2,3],['independent','mutual'],.02)
            self.assertEqual(len(paths),18)
            for path in paths:
                cfg=json.loads(Path(path).read_text())
                self.assertEqual(cfg['obs_dim'],26)
                self.assertEqual(cfg['guidance_weight'],0.)
                self.assertEqual(cfg['approach_weight'],0.)
            with self.assertRaises(FileExistsError):write_plan(output,200000,[42],[1],['mutual'],.02)

    def test_cli_exposes_bounded_training_and_independent_evaluation(self):
        args=parser().parse_args(['train','--config','a.json','--output','run','--max-rounds','2'])
        self.assertEqual(args.max_rounds,2)
        args=parser().parse_args(['evaluate','--runs','run','--output','audit','--checkpoint','latest','--team','B'])
        self.assertEqual(args.split,'development')
        self.assertEqual(args.team,'B')


if __name__=='__main__':unittest.main()
