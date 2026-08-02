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
                    json.dumps({"difficulty": difficulty}), encoding="utf-8"
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


if __name__ == "__main__":
    unittest.main()
