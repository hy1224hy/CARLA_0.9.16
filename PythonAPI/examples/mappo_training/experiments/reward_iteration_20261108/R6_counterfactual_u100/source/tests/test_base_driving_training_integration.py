"""Synthetic trainer integration: real CPU PPO/checkpoints, no simulator client.

All rollout values below are labelled TEST_ONLY fixtures, not CARLA driving
evidence. Temporary artifacts are removed. The geometry guard is mocked only
inside this test and never writes a report claiming live geometry passed.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from environment.base_reward_v6 import REWARD_COMPONENTS, REWARD_DIAGNOSTICS
from training import train_no_rllib_mappo_robust_v2 as trainer
from training.checkpoint_protocol import CheckpointProtocolError, load_checkpoint_file


PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT / "configs/base_driving/base_driving_startup_v1_v1.json"


class SyntheticBaseDrivingEnv:
    """Two-transition fixture implementing only the trainer's public env API."""
    instances = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.num_agents = 2
        self.step_number = 0
        self.closed = False
        self.vehicles = [SimpleNamespace(is_alive=True, get_control=lambda: SimpleNamespace(throttle=.2, steer=0., brake=0.))
                         for _ in (0, 1)]
        self.scenario = None
        self.__class__.instances.append(self)

    def load_scenario_by_name(self, name, directory):
        return json.loads((Path(directory) / name).read_text(encoding="utf-8"))

    def set_scenario(self, scene):
        self.scenario = scene

    def _obs(self):
        return {aid: np.full(20, .05 * (aid + 1) + .01 * self.step_number, dtype=np.float32)
                for aid in (0, 1)}

    def get_global_state(self):
        return np.concatenate([self._obs()[aid] for aid in (0, 1)])

    def _info(self, terminal=False):
        details, diagnostics = {}, {}
        for aid in (0, 1):
            details[aid] = {name: 0. for name in REWARD_COMPONENTS}
            details[aid]["progress"] = .4 + .1 * aid + .05 * self.step_number
            diagnostics[aid] = {name: 0. for name in REWARD_DIAGNOSTICS}
            diagnostics[aid].update(speed_mps=2.5, route_completion=.5 * self.step_number,
                                    distance_to_goal_m=10. - 5. * self.step_number,
                                    remaining_route_fraction=1. - .5 * self.step_number,
                                    success_event=float(terminal))
        return {"active_mask": {aid: not terminal for aid in (0, 1)},
                "terminated": {aid: terminal for aid in (0, 1)},
                "truncated": {0: False, 1: False},
                "episode_end_reason": "all_agents_reached" if terminal else "running",
                "offroad_terminated": {0: False, 1: False},
                "first_offroad_evidence": {0: None, 1: None},
                "max_offroad_consecutive_ticks": {0: 0, 1: 0},
                "max_footprint_outside_fraction": {0: 0., 1: 0.},
                "road_rule_terminated": {0: False, 1: False},
                "reward_details": details, "reward_diagnostics": diagnostics}

    def reset(self):
        self.step_number = 0
        return self._obs(), self._info()

    def step(self, actions):
        if set(actions) != {0, 1}:
            raise AssertionError("synthetic fixture expects both active agents")
        self.step_number += 1
        terminal = self.step_number == 2
        info = self._info(terminal)
        rewards = {aid: sum(info["reward_details"][aid].values()) for aid in (0, 1)}
        return self._obs(), rewards, {aid: terminal for aid in (0, 1)}, info

    def close(self):
        self.closed = True


def assert_state_equal(test, left, right):
    if torch.is_tensor(left):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for name in left:
            assert_state_equal(test, left[name], right[name])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for before, after in zip(left, right):
            assert_state_equal(test, before, after)
    else:
        test.assertEqual(left, right)


class BaseDrivingTrainingIntegrationTests(unittest.TestCase):
    def test_real_trainer_schema8_strict_resume_and_fail_closed_protocols(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        SyntheticBaseDrivingEnv.instances = []
        with tempfile.TemporaryDirectory(prefix="TEST_ONLY_base_driving_integration_", dir=PROJECT / "diagnostics") as temp:
            root = Path(temp)
            contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
            contract["status"] = "TEST_ONLY_SYNTHETIC_NOT_CARLA_EVIDENCE"
            contract["training_overrides"].update(hidden_dim=16, ppo_epochs=1, minibatch_size=64,
                min_transitions_per_update=1, save_interval=1, log_interval=100,
                tb_histogram_interval=0, critic_probe_batch_size=8)
            contract_path = root / "TEST_ONLY_contract.json"
            contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")

            def args(run, stop, resume=False, path=contract_path):
                argv = ["--base-driving-contract", str(path), "--save-dir", str(run),
                        "--run-id", "TEST_ONLY_synthetic_base_driving", "--updates", str(stop), "--cpu"]
                if resume:
                    argv += ["--resume", "latest"]
                parsed = trainer.build_parser().parse_args(argv)
                parsed._explicit_cli_options = {v for v in argv if v.startswith("--")}
                parsed._base_driving_clock_owned = True  # TEST ONLY; no real clock exists.
                parsed.base_driving_geometry_report = "TEST_ONLY_NO_LIVE_REPORT"
                return parsed

            def train(run, stop, resume=False, path=contract_path):
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.train(args(run, stop, resume, path))

            with contextlib.ExitStack() as stack:
                no_client = stack.enter_context(patch("carla.Client", side_effect=AssertionError("TEST MUST NOT CONNECT TO CARLA")))
                env_mock = stack.enter_context(patch("environment.base_driving_env.BaseDrivingEnv", side_effect=SyntheticBaseDrivingEnv))
                geometry_mock = stack.enter_context(patch("utils.preflight_base_driving_scenarios_v1.require_geometry_preflight",
                    return_value={"TEST_ONLY_FAKE_GEOMETRY_GUARD": True}))
                stack.enter_context(patch("utils.training_metrics_recorder.SummaryWriter"))
                stack.enter_context(patch.object(trainer, "get_git_metadata",
                    return_value={"commit": "TEST_ONLY_SYNTHETIC_SOURCE", "dirty": False, "dirty_paths": []}))
                stack.enter_context(patch.object(trainer, "runtime_metadata_with_carla",
                    return_value={"TEST_ONLY_SYNTHETIC_ENVIRONMENT": True}))
                # A stable synthetic source identity isolates this test from
                # concurrent edits elsewhere in the shared checkout. A separate
                # assertion below changes it and requires an early refusal.
                stack.enter_context(patch.object(trainer, "files_fingerprint", return_value="a" * 64))

                continuous, resumed = root / "continuous", root / "strict_resume"
                train(continuous, 2)
                uninterrupted, _ = load_checkpoint_file(continuous / "latest.pt")
                train(resumed, 1)
                first, _ = load_checkpoint_file(resumed / "latest.pt")
                self.assertEqual(first["checkpoint_schema_version"], 8)
                self.assertEqual(first["update"], 1)
                self.assertEqual(first["total_episodes"], 16)
                self.assertEqual(first["protocol_versions"]["base_driving"], "base_driving_vnext_v1")
                self.assertEqual(first["base_driving_contract"]["reward_profile"], "startup_v1")
                self.assertEqual(first["base_driving_state"]["sampler_state"]["completed_update"], 1)
                self.assertEqual(first["curriculum_state"]["sampler_state"]["completed_update"], 1)
                self.assertEqual(first["lr_schedule"]["decay_horizon"], 600)
                self.assertIsNotNone(first["critic_probe_batch"])
                for aid in (0, 1):
                    self.assertTrue(first["actor_optims"][aid]["state"])
                    self.assertTrue(all(int(row["step"]) == 1 for row in first["actor_optims"][aid]["state"].values()))
                self.assertTrue(first["critic_optim"]["state"])
                self.assertGreater(float(first["value_normalizer"]["debiasing_term"]), 0.)

                train(resumed, 2, True)
                restored, _ = load_checkpoint_file(resumed / "latest.pt")
                self.assertEqual(restored["update"], 2)
                self.assertEqual(restored["total_episodes"], 32)
                self.assertEqual(restored["run_metadata"]["resume_count"], 1)
                for field in ("actors", "critic", "actor_optims", "critic_optim", "value_normalizer",
                              "rng_state", "protocol_versions", "base_driving_contract", "base_driving_state",
                              "curriculum_state", "lr_schedule", "critic_probe_batch"):
                    with self.subTest(strict_roundtrip=field):
                        assert_state_equal(self, uninterrupted[field], restored[field])
                self.assertEqual([json.loads(line)["update"] for line in (resumed / "metrics.jsonl").read_text(encoding="utf-8").splitlines()], [1, 2])

                before_env_count = env_mock.call_count
                with patch.object(trainer, "files_fingerprint", return_value="b" * 64):
                    with self.assertRaisesRegex(CheckpointProtocolError, "code identity mismatch"):
                        train(resumed, 3, True)
                self.assertEqual(env_mock.call_count, before_env_count)

                changed_path = root / "TEST_ONLY_changed_profile.json"
                changed = copy.deepcopy(contract)
                changed["reward_profile"] = "v6"
                changed_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed on resume"):
                    train(resumed, 3, True, changed_path)
                self.assertEqual(env_mock.call_count, before_env_count)

                old_v6 = copy.deepcopy(restored)
                del old_v6["base_driving_contract"]
                del old_v6["base_driving_state"]
                with patch.object(trainer, "load_checkpoint_file", return_value=(old_v6, "synthetic_legacy_reference")):
                    with self.assertRaisesRegex(ValueError, "cannot be silently resumed|missing its explicit contract binding"):
                        train(resumed, 3, True)
                self.assertEqual(env_mock.call_count, before_env_count)
                no_client.assert_not_called()
                self.assertEqual(geometry_mock.call_count, 3)
                self.assertTrue(all(env.closed for env in SyntheticBaseDrivingEnv.instances))


if __name__ == "__main__":
    unittest.main()
