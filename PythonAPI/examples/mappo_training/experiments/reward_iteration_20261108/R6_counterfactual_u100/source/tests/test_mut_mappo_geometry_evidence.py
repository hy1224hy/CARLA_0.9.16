"""Reset-evidence persistence checks with a synthetic environment, no CARLA client."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from training.train_mut_mappo import Runtime, run_case


class ResetEvidenceEnv:
    num_agents = 4

    def __init__(self):
        self._world_tick_counts = {"total": 0, "setup": 0, "decision": 0}
        self.obs = {aid: np.zeros(20, dtype=np.float32) for aid in range(4)}
        self.geometry = {
            "case_key": "paired_case", "status": "runtime_map_and_spawn_checked",
            "waypoints": {"agent0": {"road_id": 44, "lane_id": -1}},
            "realized_initial_states": {
                0: {"location": {"x": 1.25, "y": 2.5, "z": 0.5}, "speed": 3.0}
            },
        }
        self.contacts = {0: [{"frame": 100, "classification": "physical_contact"}]}

    def reset_case(self, case):
        self.actual_seed = case["seed"]
        self._world_tick_counts.update(total=1, setup=1, decision=0)
        return self.obs, {
            "active_mask": dict.fromkeys(range(4), True),
            "case_geometry": self.geometry, "setup_contacts": self.contacts,
        }

    def step(self, controls):
        # Simulate a later mutation/cleanup of the reset info and no repetition
        # of geometry by the final step. The result must retain the first state.
        self.geometry["realized_initial_states"][0]["location"]["x"] = 999.0
        self.contacts[0].clear()
        self._world_tick_counts.update(total=2, setup=1, decision=1)
        done = dict.fromkeys(range(4), True)
        return self.obs, dict.fromkeys(range(4), 0.0), done, {
            "active_mask": dict.fromkeys(range(4), False),
            "terminated": done, "truncated": dict.fromkeys(range(4), False),
            "episode_end_reason": "all_agents_reached",
            "mut_metrics": {"valid": True, "red_success": True},
            "world_tick_counts": dict(self._world_tick_counts),
        }


class GeometryEvidenceTests(unittest.TestCase):
    def test_result_and_first_trace_preserve_reset_evidence_with_pair_identity(self):
        config = {
            "device": "cpu", "seed": 42, "task": {"max_steps": 5},
            "_root_validation": {
                "case_manifest_sha256": "manifest_digest", "red_sha256": "red_digest",
                "scenario_hashes": {"data/base.json": "scenario_digest"},
            },
        }
        case = {"case_key": "paired_case", "family": "blocking", "seed": 42,
                "base_scenario": "data/base.json"}
        env = ResetEvidenceEnv()
        expected_geometry, expected_contacts = copy.deepcopy(env.geometry), copy.deepcopy(env.contacts)
        runtime = Runtime(config, env, {}, {}, condition="C1", team="A")
        with tempfile.TemporaryDirectory() as directory, patch(
                "training.train_mut_mappo.policy_controls", return_value={}):
            trace = Path(directory) / "trace.jsonl"
            result = run_case(runtime, case, seed=62001, trace_path=trace)
            records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        # Result serialization normalizes physical-ID dict keys to strings.
        expected_geometry = json.loads(json.dumps(expected_geometry))
        expected_contacts = json.loads(json.dumps(expected_contacts))
        self.assertEqual(result["case_geometry"], expected_geometry)
        self.assertEqual(result["setup_contacts"], expected_contacts)
        self.assertEqual(result["seed"], 62001)
        self.assertEqual(env.actual_seed, 62001)
        self.assertEqual(case["seed"], 42)
        identity = result["initial_evidence_identity"]
        self.assertEqual(identity, {
            "case_key": "paired_case", "family": "blocking", "seed": 62001,
            "condition": "C1", "team": "A", "case_manifest_sha256": "manifest_digest",
            "base_scenario_sha256": "scenario_digest", "red_checkpoint_sha256": "red_digest",
        })
        self.assertEqual(records[0]["record_kind"], "case_initial_geometry")
        self.assertEqual(records[0]["step"], 0)
        self.assertEqual(records[0]["case_geometry"], result["case_geometry"])
        self.assertEqual(records[0]["setup_contacts"], result["setup_contacts"])
        self.assertTrue(all(records[0][key] == value for key, value in identity.items()))
        self.assertEqual(records[1]["step"], 1)
        self.assertEqual(result["sensor_timing"]["collision_callback_frame_barrier"], "not_enforced")


if __name__ == "__main__":
    unittest.main()
