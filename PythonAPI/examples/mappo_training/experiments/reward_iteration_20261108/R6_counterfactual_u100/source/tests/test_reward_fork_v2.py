"""Small offline checks for u456 origin and complete-state reward transfers."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from training.checkpoint_protocol import CheckpointProtocolError, load_checkpoint_file
from training.reward_fork_v2 import (
    FORK_VERSION, ROOT, SOURCE, SOURCE_SHA, SOURCE_UPDATE,
    _runtime_fingerprint, fork_payload, validate_fork_identity,
    validate_inherited_contract, validate_registered_source,
)


def assert_equal_state(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_equal_state(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_equal_state(test, a, b)
    elif hasattr(left, "shape"):
        test.assertTrue((left == right).all())
    else:
        test.assertEqual(left, right)


@unittest.skipUnless(SOURCE.is_file(), "registered u456 artifact unavailable")
class U456RewardForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.digest = load_checkpoint_file(SOURCE, map_location="cpu")

    def derive(self, profile="instant_pace_v2"):
        return fork_payload(
            self.source, source_path=SOURCE, digest=self.digest,
            run_id="offline_u456_fork", profile=profile,
            fingerprint="test_fingerprint", git_sha="test_git",
        )

    @staticmethod
    def manifest(checkpoint):
        return {"run_id": checkpoint["run_metadata"]["run_id"],
                "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
                "training_protocol_version": checkpoint["training_protocol_version"],
                "latest": {"path": "latest.pt", "update": checkpoint["update"],
                           "sha256": "prepared_digest"}}

    def test_complete_learning_and_sampling_state_preserved_without_old_metrics(self):
        result = self.derive()
        for key in ("actors", "actor_optims", "critic", "critic_optim",
                    "value_normalizer", "rng_state", "lr_schedule",
                    "curriculum_state", "critic_probe_batch", "critic_spec",
                    "action_spec", "model_spec", "update", "total_episodes",
                    "total_successes", "total_collisions", "total_timeouts",
                    "total_offroads", "total_road_violations"):
            with self.subTest(key=key):
                assert_equal_state(self, self.source[key], result[key])
        self.assertIsNone(result["last_metrics"])
        self.assertEqual(result["checkpoint_kind"], "reward_fork_seed")
        self.assertNotEqual(result["run_metadata"]["run_id"], self.source["run_metadata"]["run_id"])
        self.assertNotIn("continuation", result["run_metadata"])
        self.assertEqual(result["training_config"]["pace_profile"], "instant_pace_v2")
        self.assertEqual(result["protocol_versions"]["pace_reward"], "instant_pace_v2")
        for key, value in self.source["training_config"].items():
            if key != "pace_profile":
                self.assertEqual(value, result["training_config"][key])
        origin = result["run_metadata"]["reward_fork"]
        self.assertEqual(origin["version"], FORK_VERSION)
        self.assertEqual(origin["source_critic_and_probe_sha256"], SOURCE_SHA)
        self.assertEqual(origin["source_critic_and_probe_update"], SOURCE_UPDATE)
        self.assertIn("source-reward", origin["critic_probe_scope"])
        self.assertEqual(self.source["training_config"]["pace_profile"], "route_pace_v1")

    def test_existing_reward_control_uses_same_source_state(self):
        result = self.derive("route_pace_v1")
        self.assertEqual(result["training_config"], self.source["training_config"])
        assert_equal_state(self, result["actors"], self.source["actors"])
        validate_fork_identity(result, self.manifest(result), "prepared_digest", "test_fingerprint")

    def test_v3_uses_registered_u456_with_full_inherited_contract(self):
        result = self.derive("instant_pace_v3")
        self.assertEqual(result["training_config"]["pace_profile"], "instant_pace_v3")
        self.assertEqual(result["protocol_versions"]["pace_reward"], "instant_pace_v3")
        validate_inherited_contract(result, self.source)
        validate_fork_identity(result, self.manifest(result), "prepared_digest", "test_fingerprint")
        for key in ("actors", "critic", "actor_optims", "critic_optim",
                    "value_normalizer", "rng_state", "lr_schedule", "curriculum_state"):
            assert_equal_state(self, result[key], self.source[key])

    def test_v4_identity_retains_registered_u456_and_source_value_state(self):
        result = self.derive("instant_pace_v4")
        self.assertEqual(result["training_config"]["pace_profile"], "instant_pace_v4")
        self.assertEqual(result["protocol_versions"]["pace_reward"], "instant_pace_v4")
        self.assertEqual(result["run_metadata"]["reward_fork"]["source_sha256"], SOURCE_SHA)
        self.assertEqual(result["run_metadata"]["reward_fork"]["source_update"], SOURCE_UPDATE)
        validate_inherited_contract(result, self.source)
        validate_fork_identity(result, self.manifest(result), "prepared_digest", "test_fingerprint")
        for key in ("actors", "critic", "actor_optims", "critic_optim",
                    "value_normalizer", "rng_state", "lr_schedule", "curriculum_state"):
            assert_equal_state(self, result[key], self.source[key])

    def test_v5_identity_retains_registered_u456_and_source_value_state(self):
        result = self.derive("instant_pace_v5")
        self.assertEqual(result["training_config"]["pace_profile"], "instant_pace_v5")
        self.assertEqual(result["protocol_versions"]["pace_reward"], "instant_pace_v5")
        self.assertEqual(result["run_metadata"]["reward_fork"]["source_sha256"], SOURCE_SHA)
        self.assertEqual(result["run_metadata"]["reward_fork"]["source_update"], SOURCE_UPDATE)
        validate_inherited_contract(result, self.source)
        validate_fork_identity(result, self.manifest(result), "prepared_digest", "test_fingerprint")
        for key in ("actors", "critic", "actor_optims", "critic_optim",
                    "value_normalizer", "rng_state", "lr_schedule", "curriculum_state"):
            assert_equal_state(self, result[key], self.source[key])

    def test_v6_identity_retains_complete_state_and_evaluator_contract(self):
        from evaluation.evaluate_reward_fork import checkpoint_environment
        result = self.derive("instant_pace_v6")
        self.assertEqual(result["training_config"]["pace_profile"], "instant_pace_v6")
        validate_inherited_contract(result, self.source)
        validate_fork_identity(result, self.manifest(result), "prepared_digest", "test_fingerprint")
        self.assertEqual(checkpoint_environment(result)["pace_profile"], "instant_pace_v6")
        for key in ("actors", "critic", "actor_optims", "critic_optim", "value_normalizer",
                    "rng_state", "lr_schedule", "curriculum_state"):
            assert_equal_state(self, result[key], self.source[key])

    def test_rejects_wrong_source_sha_and_reward_registration(self):
        with self.assertRaises(CheckpointProtocolError):
            validate_registered_source(self.source, "0" * 64)
        for key, value in (("pace_profile", "frozen_v6"), ("reward_version", "v5")):
            source = copy.deepcopy(self.source)
            source["training_config"][key] = value
            with self.subTest(key=key), self.assertRaises(CheckpointProtocolError):
                validate_registered_source(source, self.digest)
        with self.assertRaises(CheckpointProtocolError):
            self.derive("unregistered_profile")

    def test_rejects_changed_model_curriculum_and_lr_phase(self):
        variants = (
            ("model_spec", "obs_dim", 19),
            ("model_spec", "action_dim", 3),
            ("lr_schedule", "decay_horizon", 500),
        )
        for section, key, value in variants:
            source = copy.deepcopy(self.source)
            source[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(CheckpointProtocolError):
                validate_registered_source(source, self.digest)
        source = copy.deepcopy(self.source)
        source["curriculum_state"]["mastery_state"]["stage_index"] = 2
        with self.assertRaises(CheckpointProtocolError):
            validate_registered_source(source, self.digest)

    def test_rejects_resume_identity_drift_and_manifest_mismatch(self):
        result = self.derive()
        manifest = self.manifest(result)
        validate_fork_identity(result, manifest, "prepared_digest", "test_fingerprint")
        with self.assertRaises(CheckpointProtocolError):
            validate_fork_identity(result, manifest, "prepared_digest", "changed_source")
        for key, value in (("sha256", "changed_digest"), ("update", SOURCE_UPDATE + 1),
                           ("path", "seed.pt")):
            changed = copy.deepcopy(manifest)
            changed["latest"][key] = value
            with self.subTest(key=key), self.assertRaises(CheckpointProtocolError):
                validate_fork_identity(result, changed, "prepared_digest", "test_fingerprint")
        changed = copy.deepcopy(result)
        changed["run_metadata"]["reward_fork"]["source_update"] = 356
        with self.assertRaises(CheckpointProtocolError):
            validate_fork_identity(changed, manifest, "prepared_digest", "test_fingerprint")
        changed = copy.deepcopy(result)
        changed["protocol_versions"]["pace_reward"] = "route_pace_v1"
        with self.assertRaises(CheckpointProtocolError):
            validate_fork_identity(changed, manifest, "prepared_digest", "test_fingerprint")

    def test_reward_only_contract_rejects_optimizer_and_observation_protocol_changes(self):
        result = self.derive()
        validate_inherited_contract(result, self.source)
        changed = copy.deepcopy(result)
        changed["training_config"]["gamma"] = 0.99
        with self.assertRaises(CheckpointProtocolError):
            validate_inherited_contract(changed, self.source)
        changed = copy.deepcopy(result)
        changed["protocol_versions"]["observation"] = "changed_observation"
        with self.assertRaises(CheckpointProtocolError):
            validate_inherited_contract(changed, self.source)
        changed = copy.deepcopy(result)
        changed["critic_spec"]["beta"] = 0.9
        with self.assertRaises(CheckpointProtocolError):
            validate_inherited_contract(changed, self.source)


class RewardProfileFingerprintTests(unittest.TestCase):
    def test_v3_implementation_must_be_included_in_fingerprint(self):
        from training import reward_fork_v2
        for version in (3, 6):
            with self.subTest(version=version):
                profile = f"instant_pace_v{version}"
                filename = f"instant_pace_reward_v{version}.py"
                paths = [Path(reward_fork_v2.__file__).resolve()]
                parser = SimpleNamespace(_actions=[SimpleNamespace(dest="pace_profile", choices=[profile])])
                trainer = SimpleNamespace(build_parser=lambda: parser,
                                          training_source_paths=lambda root: paths)
                with patch("training.reward_fork_v2.files_fingerprint", return_value="checked") as fingerprint:
                    with self.assertRaisesRegex(CheckpointProtocolError, filename):
                        _runtime_fingerprint(trainer, profile)
                    fingerprint.assert_not_called()
                    paths.append(ROOT / "environment" / filename)
                    self.assertEqual(_runtime_fingerprint(trainer, profile), "checked")
                    fingerprint.assert_called_once_with([path.resolve() for path in paths])


if __name__ == "__main__":
    unittest.main()
