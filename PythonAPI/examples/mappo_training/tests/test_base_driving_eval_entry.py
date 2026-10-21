"""Offline compatibility gates for the new evaluator; no CARLA imports."""
import copy
import unittest
from unittest.mock import patch

from evaluation.base_driving_qualification import PROTOCOL, ROOT, load_contract
from evaluation.evaluate_base_driving_vnext import checkpoint_environment, load_scenarios, select_evaluation_scenarios
from evaluation import evaluate_base_driving_vnext as evaluator


def payload():
    return {
        "training_config": {"reward_version": "v6", "pace_profile": "instant_pace_v6",
                            "base_driving_protocol": PROTOCOL, "base_driving_reward_profile": "startup_v1",
                            "lidar_range_m": 20.0, "town": "Town05", "straight_road_length": 100.0,
                            "timestep": .1, "max_steps": 400, "ttc_max_range_m": 20.0, "gamma": .999},
        "model_spec": {"obs_dim": 20, "action_dim": 2, "num_agents": 2},
        "protocol_versions": {"observation": "carla_obs_20_lidar20m_time_v3",
                              "action": "tanh_gaussian_signed_longitudinal_2d_pre_tanh_z_v2",
                              "reward": "v6", "pace_reward": "instant_pace_v6", "base_driving": PROTOCOL},
        "base_driving_contract": {"protocol": PROTOCOL, "qualification_contract_sha256": "contract",
                                  "scenario_pool_sha256": "a" * 64, "reward_profile": "startup_v1"},
    }


class EvalEntryTests(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()
        self.header = patch("training.checkpoint_protocol.validate_checkpoint_header")
        self.header.start()
        self.addCleanup(self.header.stop)

    def test_profile_and_binding_are_explicit(self):
        environment, eligible = checkpoint_environment(payload(), self.contract, "contract")
        self.assertTrue(eligible)
        self.assertEqual(environment["base_driving_reward_profile"], "startup_v1")
        self.assertEqual(environment["pace_profile"], "instant_pace_v6")
        changed = copy.deepcopy(payload())
        changed["base_driving_contract"]["qualification_contract_sha256"] = "old"
        with self.assertRaises(ValueError):
            checkpoint_environment(changed, self.contract, "contract", allow_legacy_reference=True)

    def test_old_reference_requires_opt_in_and_cannot_qualify(self):
        old = payload()
        del old["base_driving_contract"]
        with self.assertRaises(ValueError):
            checkpoint_environment(old, self.contract, "contract")
        environment, eligible = checkpoint_environment(old, self.contract, "contract", allow_legacy_reference=True)
        self.assertFalse(eligible)
        self.assertEqual(environment["base_driving_reward_profile"], "v6")

    def test_actor_shape_and_observation_semantics_cannot_drift(self):
        for field in ("observation", "action", "pace_reward"):
            changed = copy.deepcopy(payload())
            changed["protocol_versions"][field] = "different"
            with self.assertRaises(ValueError):
                checkpoint_environment(changed, self.contract, "contract")
        changed = payload()
        changed["model_spec"]["obs_dim"] = 22
        with self.assertRaises(ValueError):
            checkpoint_environment(changed, self.contract, "contract")

    def test_parallel_regression_is_an_explicit_complete_paired_subset(self):
        pool, scenarios, digest = load_scenarios(ROOT / "data/training_scenarios_town05_base_vnext_v1/protocol/manifest.json")
        selected, subset_sha, definition = select_evaluation_scenarios(pool, scenarios, digest, parallel_regression=True)
        self.assertEqual(len(scenarios), 48)
        self.assertEqual(len(selected), 16)
        self.assertNotEqual(subset_sha, digest)
        self.assertEqual(definition["pool_manifest_sha256"], digest)
        self.assertEqual(definition["scenario_names"], sorted(selected))
        self.assertEqual(len({row["config"]["pair_id"] for row in selected.values()}), 8)
        self.assertTrue(all(row["config"]["delta_s_m"] == 0 for row in selected.values()))
        _, altered_sha, _ = select_evaluation_scenarios(pool, scenarios, "b" * 64, parallel_regression=True)
        self.assertNotEqual(subset_sha, altered_sha)
        validation = load_scenarios(ROOT / "data/training_scenarios_town05_base_vnext_v1_validation/protocol/manifest.json")
        with self.assertRaisesRegex(ValueError, "48-scene training pool"):
            select_evaluation_scenarios(*validation, parallel_regression=True)
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            select_evaluation_scenarios(pool, scenarios, digest, parallel_regression=None)

    def test_live_parallel_subset_cannot_bypass_full_pool_geometry_guard(self):
        manifest = ROOT / "data/training_scenarios_town05_base_vnext_v1/protocol/manifest.json"
        with patch("utils.preflight_base_driving_scenarios_v1.require_geometry_preflight",
                   side_effect=ValueError("TEST_ONLY partial geometry refused")) as geometry:
            with patch.object(evaluator, "select_evaluation_scenarios") as selector:
                with self.assertRaisesRegex(ValueError, "partial geometry refused"):
                    evaluator.evaluate([], None, manifest, geometry_report="TEST_ONLY_NOT_A_REAL_REPORT",
                                       parallel_regression=True)
                geometry.assert_called_once_with(manifest.parent.parent, "TEST_ONLY_NOT_A_REAL_REPORT")
                selector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
