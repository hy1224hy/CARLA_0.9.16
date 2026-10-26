import unittest
from unittest.mock import patch
from training.mut_mappo_config import load_config
from training.mut_mappo_experiments import experiment_config
from utils.plan_mut_mappo_experiments import plan


class ExperimentRegistryTests(unittest.TestCase):
    def test_groups_change_algorithm_topology_without_changing_common_task(self):
        base = load_config("configs/mut_mappo/local_candidate_v1.json")
        for name, algorithm, count, method in (("MAPPO", "mappo", 1, "off"), ("IPPO", "ippo", 1, "off"),
                ("B0", "mappo", 2, "off"), ("B1", "mappo", 2, "plain"), ("B2", "mappo", 2, "selective")):
            cfg = experiment_config(base, name)
            self.assertEqual((cfg["algorithm"], cfg["team_count"], cfg["method"]), (algorithm, count, method))
            self.assertEqual(cfg["task"], base["task"])
            self.assertEqual(cfg["red_checkpoint"], base["red_checkpoint"])
            self.assertEqual(cfg["case_manifest"], base["case_manifest"])
            self.assertEqual(cfg["run"]["schedule_clock"], "train_ticks")

    def test_suite_budget_scope_and_planned_status_are_explicit(self):
        base = load_config("configs/mut_mappo/local_candidate_v1.json")
        with self.assertRaisesRegex(ValueError, "frozen"):
            plan(base, phase="formal", suite="comparisons", formal_budget_ticks=60000)
        frozen = dict(base, research_status="frozen")
        with self.assertRaisesRegex(ValueError, "explicit positive"):
            plan(frozen, phase="formal", suite="comparisons")
        with patch("utils.plan_mut_mappo_experiments.validate_config", return_value=frozen):
            primary = plan(frozen, phase="formal", suite="comparisons", formal_budget_ticks=60000)
        self.assertEqual(len(primary["entries"]), 15)
        self.assertTrue(all(r["config"]["run"]["train_tick_budget"] == 60000 for r in primary["entries"]))
        pilot = plan(base, phase="pilot", suite="all")
        self.assertEqual(len(pilot["entries"]), 7)
        self.assertTrue(all(r["config"]["run"]["train_tick_budget"] == 120000 for r in pilot["entries"]))
        self.assertTrue(all(r["status"] == "PLANNED_NOT_RUN" for r in pilot["entries"]))
        random = experiment_config(base, "G_RANDOM")
        self.assertEqual(random["learning"]["distillation"]["gate_variant"], "random")
        self.assertEqual(experiment_config(base, "G_STRENGTH")["learning"]["distillation"]["gate_variant"], "strength")


if __name__ == "__main__":
    unittest.main()
