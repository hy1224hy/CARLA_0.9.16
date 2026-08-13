import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from training.train_no_rllib_mappo_robust_v2 import (
    CurriculumLearning,
    compute_scenario_sampling_priorities,
    sample_scenarios_stratified,
)


class AdaptiveScenarioSamplingTests(unittest.TestCase):
    def test_unobserved_scenario_uses_neutral_prior(self):
        priorities = compute_scenario_sampling_priorities(
            ["unknown.json"],
            success_rates={"unknown.json": 0.0},
            sample_counts={"unknown.json": 0},
        )
        self.assertAlmostEqual(priorities["unknown.json"], 0.75)

    def test_failure_history_increases_priority_after_smoothing(self):
        priorities = compute_scenario_sampling_priorities(
            ["hard.json", "easy.json"],
            success_rates={"hard.json": 0.0, "easy.json": 1.0},
            sample_counts={"hard.json": 20, "easy.json": 20},
        )
        self.assertGreater(priorities["hard.json"], priorities["easy.json"])

    def test_weighted_batch_keeps_coverage_and_caps_concentration(self):
        scenarios = [f"scenario_{index}.json" for index in range(4)]
        np.random.seed(42)
        sampled = sample_scenarios_stratified(
            scenarios,
            n_episodes=16,
            success_rates={
                scenarios[0]: 0.0,
                scenarios[1]: 1.0,
                scenarios[2]: 1.0,
                scenarios[3]: 1.0,
            },
            sample_counts={scenario: 20 for scenario in scenarios},
            mode="weighted",
        )
        counts = Counter(sampled)
        self.assertEqual(set(counts), set(scenarios))
        self.assertEqual(len(sampled), 16)
        self.assertLessEqual(max(counts.values()), 6)
        self.assertGreater(counts[scenarios[0]], min(counts.values()))

    def test_family_coverage_keeps_adaptive_budget_for_sixteen_scenarios(self):
        scenarios = [f"scenario_{index:02d}.json" for index in range(16)]
        groups = {
            scenario: f"family_{index // 4}"
            for index, scenario in enumerate(scenarios)
        }
        np.random.seed(7)
        sampled = sample_scenarios_stratified(
            scenarios,
            n_episodes=16,
            success_rates={
                scenario: (0.0 if index == 0 else 1.0)
                for index, scenario in enumerate(scenarios)
            },
            sample_counts={scenario: 20 for scenario in scenarios},
            mode="weighted",
            coverage_groups=groups,
        )
        counts = Counter(sampled)
        sampled_groups = {groups[scenario] for scenario in sampled}
        self.assertEqual(len(sampled), 16)
        self.assertEqual(sampled_groups, set(groups.values()))
        self.assertGreater(counts[scenarios[0]], 1)

    def test_family_coverage_rotates_to_least_sampled_member(self):
        scenarios = ["a.json", "b.json", "c.json", "d.json"]
        sampled = sample_scenarios_stratified(
            scenarios,
            n_episodes=2,
            success_rates={scenario: 0.5 for scenario in scenarios},
            sample_counts={
                "a.json": 10,
                "b.json": 0,
                "c.json": 8,
                "d.json": 1,
            },
            coverage_counts={
                "a.json": 100,
                "b.json": 20,
                "c.json": 80,
                "d.json": 10,
            },
            mode="weighted",
            coverage_groups={
                "a.json": "family_1",
                "b.json": "family_1",
                "c.json": "family_2",
                "d.json": "family_2",
            },
        )
        self.assertEqual(set(sampled), {"b.json", "d.json"})

    def test_empty_pool_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            sample_scenarios_stratified([], 16, mode="weighted")


class MetadataCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.scenario_dir = Path(self.temp_dir.name)
        for difficulty in range(1, 4):
            for suffix in ("a", "b"):
                path = self.scenario_dir / f"d{difficulty}_{suffix}.json"
                path.write_text(
                    json.dumps({
                        "difficulty": difficulty,
                        "family": f"family_{difficulty}",
                    }),
                    encoding="utf-8",
                )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_stages_are_balanced_and_cumulative(self):
        curriculum = CurriculumLearning(
            str(self.scenario_dir),
            mode="metadata",
            warmup_updates=10,
            full_scenario_updates=40,
            transition_window=10,
        )
        self.assertEqual([len(stage) for stage in curriculum.curriculum_stages], [2, 4, 6])
        self.assertTrue(
            set(curriculum.curriculum_stages[0])
            < set(curriculum.curriculum_stages[1])
            < set(curriculum.curriculum_stages[2])
        )

    def test_warmup_boundary_and_soft_weight_apply_to_new_scenarios(self):
        curriculum = CurriculumLearning(
            str(self.scenario_dir),
            mode="metadata",
            warmup_updates=10,
            full_scenario_updates=40,
            transition_window=10,
        )
        self.assertEqual(len(curriculum.get_available_scenarios(9)), 2)
        self.assertEqual(len(curriculum.get_available_scenarios(10)), 4)
        weights = curriculum.get_scenario_sampling_weights(10)
        old_scenarios = set(curriculum.curriculum_stages[0])
        new_scenarios = set(curriculum.curriculum_stages[1]) - old_scenarios
        self.assertEqual({weights[name] for name in old_scenarios}, {1.0})
        self.assertEqual({weights[name] for name in new_scenarios}, {0.1})
        self.assertEqual(
            set(curriculum.get_scenario_sampling_weights(19).values()),
            {1.0},
        )

    def test_metadata_exposes_family_coverage_groups(self):
        curriculum = CurriculumLearning(
            str(self.scenario_dir),
            mode="metadata",
        )
        scenarios = curriculum.curriculum_stages[-1]
        groups = curriculum.get_scenario_coverage_groups(scenarios)
        self.assertEqual(set(groups), set(scenarios))
        self.assertEqual(set(groups.values()), {
            "family_1", "family_2", "family_3"
        })

    def test_total_sample_count_survives_sliding_success_window(self):
        curriculum = CurriculumLearning(
            str(self.scenario_dir),
            mode="metadata",
        )
        scenario = curriculum.curriculum_stages[0][0]
        for _ in range(25):
            curriculum.record_scenario_result(scenario, True)
        self.assertEqual(curriculum.get_scenario_sample_count(scenario), 20)
        self.assertEqual(curriculum.get_scenario_total_sample_count(scenario), 25)

    def test_town05_v2_curriculum_has_four_cumulative_balanced_stages(self):
        project_dir = Path(__file__).resolve().parents[1]
        curriculum = CurriculumLearning(
            str(project_dir / "data" / "training_scenarios_town05_v2"),
            mode="metadata",
            warmup_updates=20,
            full_scenario_updates=80,
            transition_window=10,
        )
        self.assertEqual(
            [len(stage) for stage in curriculum.curriculum_stages],
            [8, 16, 24, 32],
        )
        self.assertEqual(len(curriculum.get_available_scenarios(19)), 8)
        self.assertEqual(len(curriculum.get_available_scenarios(20)), 16)
        self.assertEqual(len(curriculum.get_available_scenarios(50)), 24)
        self.assertEqual(len(curriculum.get_available_scenarios(80)), 32)
        groups = curriculum.get_scenario_coverage_groups(
            curriculum.curriculum_stages[-1]
        )
        self.assertEqual(len(set(groups.values())), 8)


if __name__ == "__main__":
    unittest.main()
