import copy
import json
from pathlib import Path
import unittest

from training.base_driving_curriculum import BaseDrivingCurriculumConfig
from training.base_driving_runtime import BaseDrivingCurriculumAdapter, load_adapter


PROJECT = Path(__file__).resolve().parents[1]
POOL = PROJECT / "data/training_scenarios_town05_base_vnext_v1"


def _episode(name):
    return {
        "scenario_name": name, "success": True, "timeout": False,
        "collision": False, "offroad": False, "road_violation": False,
        "agent_metrics": {
            aid: {
                "mean_speed_mps": 2.5, "completion": 1.0,
                "brake_mode_rate": 0.0, "active_decision_steps": 100,
            } for aid in (0, 1)
        },
    }


def _update(adapter, update):
    names = adapter.sample_scenarios(16, phase=update - 1)
    for name in names:
        adapter.record_scenario_result(name, True)
    return adapter.observe_update(update, [_episode(name) for name in names])


class BaseDrivingRuntimeTests(unittest.TestCase):
    def test_legacy_facade_retains_counts_without_enabling_legacy_mastery(self):
        adapter = BaseDrivingCurriculumAdapter(POOL)
        self.assertIsNone(adapter.mastery_gate)
        self.assertEqual([len(stage) for stage in adapter.curriculum_stages], [16, 48])
        self.assertEqual(adapter._get_stage_idx(600), 0)
        names = adapter.get_available_scenarios(1)
        name = names[0]
        self.assertEqual(adapter.get_scenario_success_rate(name), 0.5)
        for index in range(25):
            adapter.record_scenario_result(name, index >= 5)
        self.assertEqual(adapter.get_scenario_sample_count(name), 20)
        self.assertEqual(adapter.get_scenario_total_sample_count(name), 25)
        self.assertEqual(adapter.total_count, 25)
        self.assertEqual(adapter.get_scenario_success_rate(name), 1.0)
        groups = adapter.get_scenario_coverage_groups(names)
        self.assertEqual(len(set(groups.values())), 8)
        locked = next(name for name in adapter.scenario_metadata if name not in names)
        with self.assertRaisesRegex(ValueError, "curriculum-locked"):
            adapter.record_scenario_result(locked, True)

    def test_checkpoint_restores_pending_window_exact_sampler_and_diagnostic_counts(self):
        adapter = BaseDrivingCurriculumAdapter(POOL)
        for update in range(1, 4):
            _update(adapter, update)
        state = json.loads(json.dumps(adapter.state_dict(completed_update=3)))
        restored = BaseDrivingCurriculumAdapter(POOL)
        restored.load_state_dict(state, expected_completed_update=3)
        self.assertEqual(restored.state_dict(), adapter.state_dict())
        self.assertEqual(adapter.sample_scenarios(16), restored.sample_scenarios(16))
        _update(adapter, 4)
        _update(restored, 4)
        self.assertEqual(adapter.state_dict(), restored.state_dict())
        self.assertEqual(restored.total_count, 64)
        restored.validate_checkpoint_histories({
            "success_history": restored.scenario_success_history,
            "total_sample_counts": restored.scenario_total_sample_counts,
        })
        with self.assertRaisesRegex(ValueError, "sample counts disagree"):
            restored.validate_checkpoint_histories({
                "success_history": restored.scenario_success_history,
                "total_sample_counts": {},
            })

    def test_sampler_boundary_is_next_support_after_promotion(self):
        small_test_config = BaseDrivingCurriculumConfig(
            minimum_window_updates=8, maximum_window_updates=48,
            minimum_window_episodes=128, minimum_stratum_episodes=8,
        )
        adapter = BaseDrivingCurriculumAdapter(POOL, config=small_test_config)
        for update in range(1, 17):
            evidence = _update(adapter, update)
        self.assertTrue(evidence["promoted"])
        self.assertEqual(len(adapter.get_available_scenarios(17)), 48)
        state = adapter.state_dict(completed_update=16)
        self.assertEqual(state["sampler_state"]["completed_update"], 16)
        restored = BaseDrivingCurriculumAdapter(POOL, config=small_test_config)
        restored.load_state_dict(state, expected_completed_update=16)
        self.assertEqual(restored.sample_scenarios(16), adapter.sample_scenarios(16))
        self.assertEqual(len(set(adapter.sample_scenarios(16))), 16)

    def test_mismatched_boundary_sampler_and_history_rejected_without_partial_restore(self):
        adapter = BaseDrivingCurriculumAdapter(POOL)
        _update(adapter, 1)
        state = adapter.state_dict(completed_update=1)
        with self.assertRaisesRegex(ValueError, "checkpoint update"):
            adapter.state_dict(completed_update=2)
        with self.assertRaisesRegex(ValueError, "sampler phase"):
            adapter.sample_scenarios(16, phase=0)
        for corrupt in ("sampler", "history", "pool"):
            candidate = copy.deepcopy(state)
            if corrupt == "sampler":
                candidate["sampler_state"]["completed_update"] = 0
            elif corrupt == "history":
                observed = next(name for name, count in candidate["scenario_total_sample_counts"].items() if count)
                candidate["scenario_success_history"][observed] = []
            else:
                candidate["pool_manifest_sha256"] = "changed"
            restored = BaseDrivingCurriculumAdapter(POOL)
            before = restored.state_dict()
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                restored.load_state_dict(candidate, expected_completed_update=1)
            self.assertEqual(restored.state_dict(), before)
        restored = BaseDrivingCurriculumAdapter(POOL)
        with self.assertRaisesRegex(ValueError, "checkpoint update"):
            restored.load_state_dict(state, expected_completed_update=2)

    def test_loader_resolves_project_path_and_rejects_validation_split(self):
        adapter = load_adapter(
            "data/training_scenarios_town05_base_vnext_v1", project_dir=PROJECT,
            config={"minimum_agent_mean_speed_mps": 2.5}, completed_update=327,
        )
        self.assertEqual(adapter.base_gate.last_update, 327)
        self.assertEqual(adapter.base_gate.config.minimum_agent_mean_speed_mps, 2.5)
        with self.assertRaisesRegex(ValueError, "training split"):
            load_adapter(PROJECT / "data/training_scenarios_town05_base_vnext_v1_validation")
        with self.assertRaisesRegex(ValueError, "configuration"):
            load_adapter(POOL, config={"unknown_option": 1})


if __name__ == "__main__":
    unittest.main()
