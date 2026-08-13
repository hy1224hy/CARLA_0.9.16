import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.actor_critic import Actor, AgentConditionedCritic
from training.checkpoint_protocol import (
    ACTION_PROTOCOL_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    LIFECYCLE_PROTOCOL_VERSION,
    OBSERVATION_PROTOCOL_VERSION,
    TRAINING_PROTOCOL_VERSION,
    CheckpointProtocolError,
    atomic_copy_checkpoint,
    atomic_write_json,
    capture_rng_state,
    checksum_path,
    ensure_fresh_run_directory,
    load_checkpoint_file,
    load_manifest,
    resolve_resume_path,
    restore_rng_state,
    save_checkpoint_file,
    sha256_file,
    validate_checkpoint_payload,
)
from training.train_no_rllib_mappo_robust_v2 import (
    CurriculumLearning,
    RESUME_IMMUTABLE_ARG_NAMES,
    apply_resume_configuration,
    build_parser,
    checkpoint_training_config,
    train,
    validate_p0_smoke_interrupt_configuration,
)
from training.metrics_protocol import METRICS_PROTOCOL_VERSION
from training.value_normalization import (
    VALUE_LOSS_PROTOCOL_VERSION,
    VALUE_NORMALIZATION_PROTOCOL_VERSION,
    ValueNormalizer,
)
from training.action_protocol import action_protocol_spec, legacy_action_protocol_spec


class P0SmokeInterruptConfigurationTests(unittest.TestCase):
    def test_interrupt_hook_requires_resume(self):
        args = build_parser().parse_args(
            ["--p0-smoke-interrupt-after-episodes", "1"]
        )
        with self.assertRaisesRegex(ValueError, "only valid with --resume"):
            validate_p0_smoke_interrupt_configuration(args)

    def test_interrupt_hook_must_fit_frozen_rollout(self):
        args = build_parser().parse_args(
            [
                "--resume",
                "latest",
                "--rollout-episodes",
                "4",
                "--p0-smoke-interrupt-after-episodes",
                "5",
            ]
        )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            validate_p0_smoke_interrupt_configuration(args)

    def test_interrupt_hook_accepts_resume_episode_boundary(self):
        args = build_parser().parse_args(
            [
                "--resume",
                "latest",
                "--rollout-episodes",
                "4",
                "--p0-smoke-interrupt-after-episodes",
                "1",
            ]
        )
        self.assertEqual(validate_p0_smoke_interrupt_configuration(args), 1)


class CurriculumResumeDeterminismTests(unittest.TestCase):
    def test_classic_stage_order_is_stable_and_deduplicated(self):
        scenario_files = [
            "scenario_01_basic.json",
            "scenario_02_left.json",
            "scenario_03_parallel.json",
            "scenario_04_right.json",
            "scenario_05_far.json",
        ]
        stages = CurriculumLearning._build_default_stages(scenario_files)
        self.assertEqual(
            stages[1],
            [
                "scenario_01_basic.json",
                "scenario_02_left.json",
                "scenario_03_parallel.json",
                "scenario_04_right.json",
            ],
        )
        self.assertEqual(len(stages[1]), len(set(stages[1])))


def _protocol_versions():
    return {
        "observation": OBSERVATION_PROTOCOL_VERSION,
        "action": ACTION_PROTOCOL_VERSION,
        "lifecycle": LIFECYCLE_PROTOCOL_VERSION,
        "reward": "v2",
        "metrics": METRICS_PROTOCOL_VERSION,
        "value_normalization": VALUE_NORMALIZATION_PROTOCOL_VERSION,
        "value_loss": VALUE_LOSS_PROTOCOL_VERSION,
    }


def _model_spec():
    return {
        "obs_dim": 19,
        "action_dim": 2,
        "state_dim": 38,
        "num_agents": 2,
        "agent_ids": [0, 1],
        "hidden_dim": 32,
        "critic_type": "AgentConditionedCritic",
        "critic_output_semantics": "normalized_value",
    }


def _training_config():
    return {
        "gamma": 0.997,
        "gae_lambda": 0.95,
        "clip_eps": 0.15,
        "reward_version": "v2",
    }


def _checkpoint_payload():
    value_normalizer = ValueNormalizer()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "protocol_versions": _protocol_versions(),
        "checkpoint_kind": "latest",
        "update": 7,
        "actors": {0: {"weight": torch.tensor([1.0])}, 1: {"weight": torch.tensor([2.0])}},
        "critic": {"weight": torch.tensor([3.0])},
        "value_normalizer": value_normalizer.state_dict(),
        "critic_spec": value_normalizer.spec(),
        "actor_optims": {0: {"state": {}}, 1: {"state": {}}},
        "critic_optim": {"state": {}},
        "args": {"gamma": 0.997},
        "action_spec": action_protocol_spec(),
        "model_spec": _model_spec(),
        "training_config": _training_config(),
        "lr_schedule": {"decay_horizon": 100},
        "run_metadata": {
            "run_id": "P0_test_s42",
            "seed": 42,
            "origin_git_sha": "abc123",
            "last_git_sha": "abc123",
            "code_fingerprint": "c" * 64,
            "config_path": "",
        },
        "curriculum_state": {
            "success_history": {},
            "stages": [["scenario.json"]],
            "scenario_pool_fingerprint": "f" * 64,
        },
        "rng_state": capture_rng_state(),
        "last_metrics": {
            "metrics_protocol_version": METRICS_PROTOCOL_VERSION,
            "run_id": "P0_test_s42",
            "update": 7,
            "scalars": {"reward_audit/max_abs_step_residual": 0.0},
        },
        "total_episodes": 10,
        "total_successes": 4,
        "total_collisions": 3,
        "total_timeouts": 3,
        "total_offroads": 0,
    }


def _assert_nested_equal(test_case, left, right):
    test_case.assertEqual(type(left), type(right))
    if isinstance(left, torch.Tensor):
        test_case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test_case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test_case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test_case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(test_case, left_item, right_item)
    else:
        test_case.assertEqual(left, right)


class AtomicCheckpointFileTests(unittest.TestCase):
    def test_atomic_save_load_checksum_and_no_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            payload = _checkpoint_payload()
            digest = save_checkpoint_file(payload, path)
            loaded, loaded_digest = load_checkpoint_file(path)

            self.assertEqual(digest, loaded_digest)
            self.assertEqual(digest, sha256_file(path))
            self.assertTrue(checksum_path(path).is_file())
            self.assertEqual(loaded["update"], 7)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))
            self.assertFalse(list(Path(tmp).glob(".*.tmp")))

    def test_immutable_periodic_checkpoint_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt_update_000007.pt"
            save_checkpoint_file(_checkpoint_payload(), path, immutable=True)
            original_digest = sha256_file(path)
            with self.assertRaises(FileExistsError):
                save_checkpoint_file(_checkpoint_payload(), path, immutable=True)
            self.assertEqual(sha256_file(path), original_digest)

    def test_corruption_is_detected_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            save_checkpoint_file(_checkpoint_payload(), path)
            with open(path, "ab") as file_obj:
                file_obj.write(b"corrupt")
            with self.assertRaisesRegex(CheckpointProtocolError, "SHA256 mismatch"):
                load_checkpoint_file(path)

    def test_missing_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save(_checkpoint_payload(), path)
            with self.assertRaisesRegex(CheckpointProtocolError, "sidecar"):
                load_checkpoint_file(path)

    def test_interrupted_copy_matches_last_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest.pt"
            interrupted = Path(tmp) / "interrupted.pt"
            digest = save_checkpoint_file(_checkpoint_payload(), latest)
            copied_digest = atomic_copy_checkpoint(latest, interrupted)

            self.assertEqual(copied_digest, digest)
            self.assertEqual(sha256_file(interrupted), sha256_file(latest))
            loaded, _ = load_checkpoint_file(interrupted)
            self.assertEqual(loaded["update"], 7)


class RngAndOptimizerContinuityTests(unittest.TestCase):
    def test_python_numpy_and_torch_rng_continue_exactly(self):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        state = capture_rng_state()
        expected = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        random.random()
        np.random.random(10)
        torch.rand(10)
        warnings = restore_rng_state(state)
        actual = (
            random.random(),
            np.random.random(4),
            torch.rand(4),
        )

        self.assertEqual(warnings, ())
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_optimizer_next_step_matches_after_round_trip(self):
        torch.manual_seed(77)
        model = nn.Linear(3, 1)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.tensor([[0.2, -0.4, 0.8], [1.0, 0.5, -0.3]])
        targets = torch.tensor([[0.7], [-0.2]])

        def take_step(current_model, current_optimizer):
            loss = (current_model(inputs) - targets).pow(2).mean()
            current_optimizer.zero_grad()
            loss.backward()
            current_optimizer.step()

        take_step(model, optimizer)
        saved_model = copy.deepcopy(model.state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        take_step(model, optimizer)
        expected = copy.deepcopy(model.state_dict())

        resumed_model = nn.Linear(3, 1)
        resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=1e-3)
        resumed_model.load_state_dict(saved_model)
        resumed_optimizer.load_state_dict(saved_optimizer)
        take_step(resumed_model, resumed_optimizer)

        for name, tensor in expected.items():
            self.assertTrue(torch.equal(tensor, resumed_model.state_dict()[name]))

    def test_all_mappo_models_and_optimizers_round_trip(self):
        torch.manual_seed(20260727)
        actors = {aid: Actor(19, 2, hidden=16) for aid in (0, 1)}
        critic = AgentConditionedCritic(38, 2, hidden=16)
        actor_optims = {
            aid: optim.Adam(actor.parameters(), lr=3e-4)
            for aid, actor in actors.items()
        }
        critic_optim = optim.Adam(critic.parameters(), lr=3e-4)

        for aid, actor in actors.items():
            mu, log_std = actor(torch.randn(4, 19))
            loss = mu.pow(2).mean() + 0.01 * log_std.pow(2).mean()
            actor_optims[aid].zero_grad()
            loss.backward()
            actor_optims[aid].step()
        values = critic(torch.randn(6, 38), torch.tensor([0, 1, 0, 1, 0, 1]))
        critic_loss = values.pow(2).mean()
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        payload = _checkpoint_payload()
        payload["actors"] = {
            aid: copy.deepcopy(actor.state_dict()) for aid, actor in actors.items()
        }
        payload["critic"] = copy.deepcopy(critic.state_dict())
        payload["actor_optims"] = {
            aid: copy.deepcopy(optimizer.state_dict())
            for aid, optimizer in actor_optims.items()
        }
        payload["critic_optim"] = copy.deepcopy(critic_optim.state_dict())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.pt"
            save_checkpoint_file(payload, path)
            loaded, _ = load_checkpoint_file(path)

        resumed_actors = {aid: Actor(19, 2, hidden=16) for aid in (0, 1)}
        resumed_critic = AgentConditionedCritic(38, 2, hidden=16)
        resumed_actor_optims = {
            aid: optim.Adam(actor.parameters(), lr=3e-4)
            for aid, actor in resumed_actors.items()
        }
        resumed_critic_optim = optim.Adam(
            resumed_critic.parameters(), lr=3e-4
        )
        for aid in (0, 1):
            resumed_actors[aid].load_state_dict(loaded["actors"][aid])
            resumed_actor_optims[aid].load_state_dict(
                loaded["actor_optims"][aid]
            )
            _assert_nested_equal(
                self, actors[aid].state_dict(), resumed_actors[aid].state_dict()
            )
            _assert_nested_equal(
                self,
                actor_optims[aid].state_dict(),
                resumed_actor_optims[aid].state_dict(),
            )
        resumed_critic.load_state_dict(loaded["critic"])
        resumed_critic_optim.load_state_dict(loaded["critic_optim"])
        _assert_nested_equal(
            self, critic.state_dict(), resumed_critic.state_dict()
        )
        _assert_nested_equal(
            self, critic_optim.state_dict(), resumed_critic_optim.state_dict()
        )


class ResumeProtocolValidationTests(unittest.TestCase):
    def test_complete_payload_passes_and_incompatible_config_fails(self):
        payload = _checkpoint_payload()
        validate_checkpoint_payload(
            payload,
            expected_protocol_versions=_protocol_versions(),
            expected_action_spec=action_protocol_spec(),
            expected_model_spec=_model_spec(),
            expected_training_config=_training_config(),
        )

        legacy_action = copy.deepcopy(payload)
        legacy_action["action_spec"] = legacy_action_protocol_spec()
        with self.assertRaisesRegex(CheckpointProtocolError, "action_spec"):
            validate_checkpoint_payload(
                legacy_action,
                expected_protocol_versions=_protocol_versions(),
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=_model_spec(),
                expected_training_config=_training_config(),
            )

        incompatible = dict(_training_config())
        incompatible["gamma"] = 0.9
        with self.assertRaisesRegex(CheckpointProtocolError, "training_config.gamma"):
            validate_checkpoint_payload(
                payload,
                expected_protocol_versions=_protocol_versions(),
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=_model_spec(),
                expected_training_config=incompatible,
            )

        incompatible_spec = dict(_model_spec())
        incompatible_spec["state_dim"] = 40
        with self.assertRaisesRegex(CheckpointProtocolError, "model_spec.state_dim"):
            validate_checkpoint_payload(
                payload,
                expected_protocol_versions=_protocol_versions(),
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=incompatible_spec,
                expected_training_config=_training_config(),
            )

        incompatible_protocol = dict(_protocol_versions())
        incompatible_protocol["lifecycle"] = "legacy_done_v0"
        with self.assertRaisesRegex(
            CheckpointProtocolError, "protocol_versions.lifecycle"
        ):
            validate_checkpoint_payload(
                payload,
                expected_protocol_versions=incompatible_protocol,
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=_model_spec(),
                expected_training_config=_training_config(),
            )

        invalid_metrics = copy.deepcopy(payload)
        invalid_metrics["last_metrics"]["scalars"]["invalid"] = float("nan")
        with self.assertRaisesRegex(CheckpointProtocolError, "NaN/Inf"):
            validate_checkpoint_payload(
                invalid_metrics,
                expected_protocol_versions=_protocol_versions(),
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=_model_spec(),
                expected_training_config=_training_config(),
            )

    def test_missing_optimizer_and_agent_id_mismatch_fail_fast(self):
        payload = _checkpoint_payload()
        del payload["critic_optim"]
        with self.assertRaisesRegex(CheckpointProtocolError, "critic_optim"):
            validate_checkpoint_payload(
                payload,
                expected_protocol_versions=_protocol_versions(),
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=_model_spec(),
                expected_training_config=_training_config(),
            )

        payload = _checkpoint_payload()
        del payload["actors"][1]
        with self.assertRaisesRegex(CheckpointProtocolError, "actors agent ids"):
            validate_checkpoint_payload(
                payload,
                expected_protocol_versions=_protocol_versions(),
                expected_action_spec=action_protocol_spec(),
                expected_model_spec=_model_spec(),
                expected_training_config=_training_config(),
            )

    def test_latest_selector_never_silently_starts_from_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "refusing to start from zero"):
                resolve_resume_path("latest", tmp)

            path = Path(tmp) / "latest.pt"
            save_checkpoint_file(_checkpoint_payload(), path)
            self.assertEqual(resolve_resume_path("latest", tmp), path.resolve())

    def test_fresh_run_rejects_existing_checkpoint_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = ensure_fresh_run_directory(Path(tmp) / "new_run")
            self.assertTrue(run_dir.is_dir())
            save_checkpoint_file(_checkpoint_payload(), run_dir / "latest.pt")
            with self.assertRaisesRegex(FileExistsError, "run artifacts"):
                ensure_fresh_run_directory(run_dir)

    def test_fresh_run_rejects_existing_metric_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "new_run"
            run_dir.mkdir()
            (run_dir / "episodes.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "run artifacts"):
                ensure_fresh_run_directory(run_dir)

    def test_manifest_round_trip_is_atomic_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_manifest.json"
            manifest = {
                "run_id": "P0_test_s42",
                "latest": {"update": 7, "sha256": "f" * 64},
                "periodic": [],
            }
            atomic_write_json(manifest, path)
            self.assertEqual(load_manifest(path), manifest)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)


class RobustV2CheckpointIntegrationTests(unittest.TestCase):
    def _resume_header(self, saved_args):
        return {
            "training_config": checkpoint_training_config(saved_args),
            "run_metadata": {
                "seed": 7,
                "run_id": "P0_resume_s7",
            },
            "lr_schedule": {"decay_horizon": 200},
        }

    def test_resume_restores_unspecified_config_and_rejects_explicit_drift(self):
        saved_args = build_parser().parse_args([])
        saved_args.gamma = 0.91
        saved_args.seed = 7
        saved_args.run_id = "P0_resume_s7"
        saved_args.lr_decay_horizon = 200
        saved_args.save_interval = 1
        saved_args.log_interval = 2
        saved_args.tb_histogram_interval = 3
        checkpoint = self._resume_header(saved_args)

        resumed_args = build_parser().parse_args([])
        resumed_args._explicit_cli_options = set()
        apply_resume_configuration(resumed_args, checkpoint)
        self.assertEqual(resumed_args.gamma, 0.91)
        self.assertEqual(resumed_args.seed, 7)
        self.assertEqual(resumed_args.run_id, "P0_resume_s7")
        self.assertEqual(resumed_args.lr_decay_horizon, 200)
        for name in RESUME_IMMUTABLE_ARG_NAMES:
            self.assertEqual(
                getattr(resumed_args, name), checkpoint["training_config"][name]
            )

        drifted_args = build_parser().parse_args(["--gamma", "0.8"])
        drifted_args._explicit_cli_options = {"--gamma"}
        with self.assertRaisesRegex(CheckpointProtocolError, "cannot change --gamma"):
            apply_resume_configuration(drifted_args, checkpoint)

        save_drift = build_parser().parse_args(["--save-interval", "5"])
        save_drift._explicit_cli_options = {"--save-interval"}
        with self.assertRaisesRegex(
            CheckpointProtocolError, "cannot change --save-interval"
        ):
            apply_resume_configuration(save_drift, checkpoint)

    def test_resume_and_pretrained_are_rejected_before_carla_initialization(self):
        args = build_parser().parse_args(
            [
                "--resume",
                "latest",
                "--pretrained-checkpoint",
                "weights.pt",
            ]
        )
        args._explicit_cli_options = {
            "--resume",
            "--pretrained-checkpoint",
        }
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            train(args)

    def test_training_source_contains_no_rollout_selected_best_checkpoint(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "training"
            / "train_no_rllib_mappo_robust_v2.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn('"best.pt"', source)
        self.assertNotIn("best_success_rate", source)


if __name__ == "__main__":
    unittest.main()
