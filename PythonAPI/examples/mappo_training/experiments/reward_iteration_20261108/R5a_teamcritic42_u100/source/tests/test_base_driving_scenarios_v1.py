import copy
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.base_driving_scenarios_v1 import (
    BASE_DRIVING_DIRECTIONS,
    base_driving_sampler_state,
    load_base_driving_manifest,
    sample_base_driving_balanced,
    scenario_axes,
    validate_base_driving_sampler_state,
)
from utils.build_base_driving_scenarios_v1 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VALIDATION_DIR,
    build_payloads,
    validate_payloads,
    write_payloads,
)


class BaseDrivingScenarioTests(unittest.TestCase):
    def test_actor_swap_includes_goals_and_keeps_equal_individual_routes(self):
        train = build_payloads()
        self.assertEqual(len(train), 48)
        self.assertEqual(Counter(p["curriculum_stage"] for p in train.values()), {0: 16, 1: 32})
        self.assertEqual({p["goal_mode"] for p in train.values()}, {"per_agent_explicit"})
        validate_payloads(train)
        for name, payload in train.items():
            first, second = payload["agents"]["agent0"], payload["agents"]["agent1"]
            yaw = math.radians(first["yaw"])
            lateral_along_carla_right = -(first["x"] - second["x"]) * math.sin(yaw) + (first["y"] - second["y"]) * math.cos(yaw)
            expected = -3.5 if scenario_axes(name)["agent0_lane"] == "left" else 3.5
            self.assertAlmostEqual(lateral_along_carla_right, expected, places=7)
        broken = copy.deepcopy(train)
        broken[next(iter(broken))]["agent_goal_zones"]["agent0"]["x"] += 1
        with self.assertRaisesRegex(ValueError, "actor swap|route length"):
            validate_payloads(broken)

    def test_split_keeps_whole_physical_templates_out_of_training(self):
        train, validation = build_payloads(), build_payloads(split="validation")
        self.assertEqual(len(validation), 32)
        self.assertFalse({p["physical_template_id"] for p in train.values()} & {p["physical_template_id"] for p in validation.values()})
        for pool in (train, validation):
            self.assertEqual(set(Counter(p["physical_template_id"] for p in pool.values()).values()), {2})
            self.assertTrue(all(p["symmetry_partner"] in pool for p in pool.values()))

    def test_full_and_parallel_support_have_pair_direction_and_startup_balance(self):
        payloads = build_payloads()
        for names in (list(payloads), [name for name, p in payloads.items() if p["curriculum_stage"] == 0]):
            for phase in range(6):
                batch = sample_base_driving_balanced(names, 16, phase=phase)
                self.assertEqual(len(batch), 16)
                self.assertEqual(Counter(scenario_axes(name)["direction"] for name in batch), dict.fromkeys(BASE_DRIVING_DIRECTIONS, 4))
                self.assertEqual(set(Counter(scenario_axes(name)["pair_id"] for name in batch).values()), {2})
                self.assertEqual(Counter(scenario_axes(name)["startup_mode"] for name in batch), {"stationary": 8, "rolling": 8})
        exposure = Counter(name for phase in range(3) for name in sample_base_driving_balanced(list(payloads), 16, phase=phase))
        self.assertEqual(exposure, Counter(dict.fromkeys(payloads, 1)))

    def test_restore_is_deterministic_and_rejects_changed_phase_or_support(self):
        names = list(build_payloads())
        state = json.loads(json.dumps(base_driving_sampler_state(17, names)))
        validate_base_driving_sampler_state(state, expected_completed_update=17, available_scenarios=names)
        self.assertEqual(sample_base_driving_balanced(names, 16, phase=17), sample_base_driving_balanced(list(reversed(names)), 16, phase=17))
        with self.assertRaisesRegex(ValueError, "sampler state mismatch"):
            validate_base_driving_sampler_state(state, expected_completed_update=18, available_scenarios=names)
        parallel = [name for name in names if scenario_axes(name)["curriculum_stage"] == 0]
        with self.assertRaisesRegex(ValueError, "sampler state mismatch"):
            validate_base_driving_sampler_state(state, expected_completed_update=17, available_scenarios=parallel)
        with self.assertRaisesRegex(ValueError, "complete two-member"):
            sample_base_driving_balanced(names[1:], 16, phase=0)
        with self.assertRaisesRegex(ValueError, "multiple of 8"):
            sample_base_driving_balanced(names, 15, phase=0)

    def test_manifest_detects_changed_payload_and_generated_artifacts_are_current(self):
        write_payloads(check=True)
        self.assertEqual(load_base_driving_manifest(DEFAULT_OUTPUT_DIR)["scenario_count"], 48)
        self.assertEqual(load_base_driving_manifest(DEFAULT_VALIDATION_DIR)["scenario_count"], 32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_payloads(output_dir=root / "train", validation_dir=root / "validation")
            name = next(iter(build_payloads()))
            path = root / "train" / name
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_base_driving_manifest(root / "train")


if __name__ == "__main__":
    unittest.main()
