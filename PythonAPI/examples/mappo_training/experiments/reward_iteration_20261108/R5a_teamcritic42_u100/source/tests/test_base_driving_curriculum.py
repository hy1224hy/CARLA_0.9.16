import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from training.base_driving_curriculum import (
    BaseDrivingCurriculum,
    BaseDrivingCurriculumConfig,
    validate_base_driving_curriculum_config,
)


def _metadata():
    result = {}
    for stage in (0, 1):
        for direction in ("north", "south"):
            for swapped in (False, True):
                roles = ["left_parallel", "right_parallel"] if stage == 0 else ["left_ahead", "right_behind"]
                if swapped:
                    roles.reverse()
                result[f"s{stage}_{direction}_{int(swapped)}.json"] = {
                    "curriculum_stage": stage,
                    "task_family": "adjacent_free_flow",
                    "direction": direction,
                    "startup_mode": "stationary",
                    "agent_roles": {"agent0": roles[0], "agent1": roles[1]},
                }
    return result


def _episode(name, *, weak_agent=None, collision=False):
    return {
        "scenario_name": name,
        "success": not collision,
        "timeout": False,
        "collision": collision,
        "offroad": False,
        "road_violation": False,
        "agent_metrics": {
            aid: {
                "mean_speed_mps": 0.1 if aid == weak_agent else 2.5,
                "completion": 0.9,
                "brake_mode_rate": 0.1,
                "active_decision_steps": 100,
            } for aid in (0, 1)
        },
    }


def _config(**overrides):
    return replace(BaseDrivingCurriculumConfig(), **{
        "minimum_window_updates": 2,
        "maximum_window_updates": 6,
        "minimum_window_episodes": 8,
        "minimum_stratum_episodes": 2,
        **overrides,
    })


class BaseDrivingCurriculumTests(unittest.TestCase):
    def test_default_twenty_episode_stratum_accepts_ten_percent_but_rejects_fifteen_percent(self):
        root = Path(__file__).resolve().parents[1] / "data/training_scenarios_town05_base_vnext_v1"
        metadata = {path.name: json.loads(path.read_text(encoding="utf-8"))
                    for path in root.glob("*.json")}
        default = BaseDrivingCurriculumConfig()
        self.assertEqual((default.minimum_window_updates, default.maximum_window_updates,
                          default.minimum_window_episodes, default.minimum_stratum_episodes),
                         (20, 80, 320, 20))
        for collision_episodes in (2, 3):
            gate = BaseDrivingCurriculum(metadata)
            target = gate.available_scenarios()[0]
            for update in range(1, 21):
                batch = [_episode(name, collision=(name == target and update <= collision_episodes))
                         for name in gate.available_scenarios()]
                evidence = gate.observe_update(update, batch)
                self.assertEqual(evidence["window_closed"], update == 20)
            self.assertTrue(evidence["coverage_sufficient"])
            self.assertEqual(evidence["window_episode_count"], 320)
            self.assertEqual(set(evidence["stratum_episode_counts"].values()), {20})
            affected = [summary for summary in evidence["strata"].values() if summary["collision_rate"]]
            self.assertEqual(len(affected), 2)
            self.assertEqual({row["collision_rate"] for row in affected}, {collision_episodes / 20})
            self.assertEqual(evidence["passed"], collision_episodes == 2)
            if collision_episodes == 3:
                self.assertEqual(len(evidence["failures"]), 2)
                self.assertTrue(all(value.endswith(".collision_rate") for value in evidence["failures"]))

    def test_brake_rate_weights_actual_readbacks_not_missing_control_steps(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        for update in (1, 2):
            batch = [_episode(name) for name in gate.available_scenarios()]
            target = next(row for row in batch if row["scenario_name"] == "s0_north_0.json")
            target["agent_metrics"][1].update({
                "active_decision_steps": 100 if update == 1 else 1,
                "brake_observation_steps": 1,
                "brake_mode_rate": 1.0 if update == 1 else 0.0,
            })
            evidence = gate.observe_update(update, batch)
        summary = next(value for key, value in evidence["strata"].items()
                       if '"north","stationary",1,"right_parallel"' in key)
        self.assertEqual(summary["brake_mode_rate"], 0.5)
        self.assertEqual(summary["brake_observation_steps"], 2)
        self.assertEqual(summary["missing_control_steps"], 99)
        self.assertTrue(evidence["passed"])

    def test_no_actual_control_samples_fail_gate_instead_of_looking_like_no_braking(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        for update in (1, 2):
            batch = [_episode(name) for name in gate.available_scenarios()]
            target = next(row for row in batch if row["scenario_name"] == "s0_north_0.json")
            target["agent_metrics"][1].update({
                "brake_observation_steps": 0, "brake_mode_rate": 0.0,
            })
            evidence = gate.observe_update(update, batch)
        self.assertFalse(evidence["passed"])
        self.assertEqual(len(evidence["failures"]), 1)
        self.assertTrue(evidence["failures"][0].endswith(".missing_control"))
        summary = next(value for key, value in evidence["strata"].items()
                       if '"north","stationary",1,"right_parallel"' in key)
        self.assertIsNone(summary["brake_mode_rate"])
        self.assertEqual(summary["missing_control_steps"], 200)

    def test_invalid_readback_counts_rejected_and_partial_readbacks_resume_exactly(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        names = gate.available_scenarios()
        before = gate.state_dict()
        for count in (-1, 101, 0.5, True):
            batch = [_episode(name) for name in names]
            batch[0]["agent_metrics"][0]["brake_observation_steps"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                gate.observe_update(1, batch)
            self.assertEqual(before, gate.state_dict())
        batch = [_episode(name) for name in names]
        batch[0]["agent_metrics"][0]["brake_observation_steps"] = 2
        gate.observe_update(1, batch)
        state = json.loads(json.dumps(gate.state_dict()))
        restored = BaseDrivingCurriculum(_metadata(), config=_config())
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertEqual(restored.observe_update(2, batch), gate.observe_update(2, batch))
        for change in ("old_version", "bad_steps", "bad_sum"):
            changed = copy.deepcopy(state)
            bucket = next(value for value in changed["pending_strata"].values()
                          if value["brake_observation_steps"] == 2)
            if change == "old_version":
                changed["curriculum_protocol_version"] = "town05_base_driving_cross_update_v1"
            elif change == "bad_steps":
                bucket["brake_observation_steps"] = bucket["active_steps"] + 1
            else:
                bucket["brake_sum"] = bucket["brake_observation_steps"] + 0.5
            with self.subTest(change=change), self.assertRaises(ValueError):
                BaseDrivingCurriculum(_metadata(), config=_config()).load_state_dict(changed)

    def test_real_pool_sampler_exposes_all_roles_before_unlock_and_preserves_resume(self):
        from training.base_driving_scenarios_v1 import sample_base_driving_balanced

        root = Path(__file__).resolve().parents[1] / "data/training_scenarios_town05_base_vnext_v1"
        metadata = {path.name: json.loads(path.read_text(encoding="utf-8"))
                    for path in root.glob("*.json")}
        self.assertEqual(len(metadata), 48)
        small_test_config = BaseDrivingCurriculumConfig(
            minimum_window_updates=8, maximum_window_updates=48,
            minimum_window_episodes=128, minimum_stratum_episodes=8,
        )
        gate = BaseDrivingCurriculum(metadata, config=small_test_config)
        self.assertEqual(len(gate.available_scenarios()), 16)
        first_stage = set(gate.available_scenarios())
        closed = []
        for update in range(1, 65):
            names = sample_base_driving_balanced(gate.available_scenarios(), 16, phase=update - 1)
            evidence = gate.observe_update(update, [_episode(name) for name in names])
            if evidence["window_closed"]:
                closed.append(update)
                self.assertTrue(evidence["passed"])
            if update == 23:
                resumed = BaseDrivingCurriculum(metadata, config=small_test_config)
                resumed.load_state_dict(json.loads(json.dumps(gate.state_dict())))
                self.assertEqual(resumed.state_dict(), gate.state_dict())
                gate = resumed
        self.assertEqual(closed, [8, 16, 40, 64])
        self.assertEqual(gate.stage_index, 1)
        self.assertEqual(len(gate.available_scenarios()), 48)
        self.assertTrue(first_stage.issubset(gate.available_scenarios()))

    def test_units_are_separate_and_invalid_proportions_are_rejected(self):
        validate_base_driving_curriculum_config(_config(minimum_agent_mean_speed_mps=2.5))
        for changes in (
            {"minimum_agent_mean_completion": 1.5},
            {"minimum_agent_mean_speed_mps": float("nan")},
            {"minimum_window_updates": 1.5},
            {"consecutive_passing_windows": True},
            {"maximum_window_updates": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_base_driving_curriculum_config(_config(**changes))

    def test_rotate_partial_batches_and_require_disjoint_passing_windows(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        initial = gate.available_scenarios()
        self.assertEqual(len(initial), 4)
        # Each update has only one direction and two scenarios. Full coverage
        # accumulates across updates; an overlapping recheck is not a new pass.
        for update in range(1, 9):
            direction = "north" if update % 2 else "south"
            batch = [_episode(name) for name in initial if direction in name]
            evidence = gate.observe_update(update, batch)
            self.assertEqual(evidence["window_closed"], update in (4, 8))
            self.assertEqual(gate.stage_index, int(update == 8))
        self.assertTrue(evidence["promoted"])
        self.assertEqual(evidence["window_start_update"], 5)
        self.assertEqual(evidence["window_episode_count"], 8)
        self.assertTrue(set(initial).issubset(gate.available_scenarios()))
        self.assertEqual(len(gate.available_scenarios()), 8)
        self.assertIn("not_frozen", evidence["evidence_source"])

    def test_strata_do_not_hide_one_weak_actor_with_pooled_results(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        for update in (1, 2):
            batch = [
                _episode(name, weak_agent=(1 if "south_1" in name else None))
                for name in gate.available_scenarios()
            ]
            evidence = gate.observe_update(update, batch)
        self.assertTrue(evidence["coverage_sufficient"])
        self.assertFalse(evidence["passed"])
        self.assertEqual(gate.stage_index, 0)
        failures = evidence["failures"]
        self.assertEqual(len(failures), 1)
        self.assertIn('"south","stationary",1,', failures[0])
        self.assertTrue(failures[0].endswith(".mean_speed_mps"))

    def test_restore_pending_window_and_streak_matches_uninterrupted_run(self):
        source = BaseDrivingCurriculum(_metadata(), config=_config())
        names = source.available_scenarios()
        for update in (1, 2, 3):
            source.observe_update(update, [_episode(name) for name in names])
        state = json.loads(json.dumps(source.state_dict()))
        restored = BaseDrivingCurriculum(_metadata(), config=_config())
        restored.load_state_dict(state)
        self.assertEqual(source.state_dict(), restored.state_dict())
        source_evidence = source.observe_update(4, [_episode(name) for name in names])
        restored_evidence = restored.observe_update(4, [_episode(name) for name in names])
        self.assertEqual(source_evidence, restored_evidence)
        self.assertEqual(source.state_dict(), restored.state_dict())
        self.assertEqual(restored.stage_index, 1)

    def test_review_hysteresis_retains_learned_support_and_never_stops(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        original = gate.available_scenarios()
        for update in range(1, 5):
            gate.observe_update(update, [_episode(name) for name in original])
        self.assertEqual(gate.stage_index, 1)
        unlocked = gate.available_scenarios()
        # Train only new tasks: the old-direction/role evidence is now missing.
        new_only = [name for name in unlocked if name not in original]
        for update in range(5, 17):
            evidence = gate.observe_update(update, [_episode(name) for name in new_only])
        self.assertTrue(evidence["window_closed"])
        self.assertFalse(evidence["coverage_sufficient"])
        self.assertTrue(gate.review_required)
        self.assertEqual(gate.stage_index, 1)
        self.assertEqual(gate.available_scenarios(), unlocked)
        self.assertEqual(gate.consecutive_failures, 2)
        # One good window is insufficient to clear review; two clear it.
        for update in (17, 18):
            gate.observe_update(update, [_episode(name) for name in unlocked])
        self.assertTrue(gate.review_required)
        for update in (19, 20):
            gate.observe_update(update, [_episode(name) for name in unlocked])
        self.assertFalse(gate.review_required)
        self.assertEqual(gate.stage_index, 1)

    def test_bad_batch_is_transactional_and_no_duplicate_or_skipped_updates(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        names = gate.available_scenarios()
        bad = [_episode(names[0]), _episode("s1_north_0.json")]
        before = gate.state_dict()
        with self.assertRaisesRegex(ValueError, "still locked"):
            gate.observe_update(1, bad)
        self.assertEqual(before, gate.state_dict())
        gate.observe_update(1, [_episode(names[0])])
        for update in (1, 3):
            with self.assertRaisesRegex(ValueError, "contiguous"):
                gate.observe_update(update, [_episode(names[0])])

    def test_restore_rejects_protocol_metadata_and_pending_aggregate_drift(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config())
        gate.observe_update(1, [_episode(name) for name in gate.available_scenarios()])
        state = gate.state_dict()
        variants = []
        for key, value in (
            ("source_protocol_version", "an_old_curriculum"),
            ("curriculum_protocol_version", "old_r8a_protocol"),
            ("pending_window_episode_count", 99),
            ("pending_window_updates", [1, 3]),
        ):
            changed = copy.deepcopy(state)
            changed[key] = value
            variants.append(changed)
        changed = copy.deepcopy(state)
        first_stratum = next(iter(changed["pending_strata"].values()))
        first_stratum["speed_sum_mps"] = float("inf")
        variants.append(changed)
        for changed in variants:
            restored = BaseDrivingCurriculum(_metadata(), config=_config())
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                restored.load_state_dict(changed)
        metadata = _metadata()
        metadata["s0_north_0.json"]["direction"] = "west"
        with self.assertRaisesRegex(ValueError, "scenario_metadata"):
            BaseDrivingCurriculum(metadata, config=_config()).load_state_dict(state)

    def test_warm_start_origin_is_explicit_and_cannot_load_another_origin(self):
        gate = BaseDrivingCurriculum(_metadata(), config=_config(), completed_update=327)
        evidence = gate.observe_update(328, [_episode(name) for name in gate.available_scenarios()])
        self.assertEqual(evidence["window_start_update"], 328)
        restored = BaseDrivingCurriculum(_metadata(), config=_config(), completed_update=327)
        restored.load_state_dict(gate.state_dict())
        with self.assertRaisesRegex(ValueError, "origin_completed_update"):
            BaseDrivingCurriculum(_metadata(), config=_config()).load_state_dict(gate.state_dict())


if __name__ == "__main__":
    unittest.main()
