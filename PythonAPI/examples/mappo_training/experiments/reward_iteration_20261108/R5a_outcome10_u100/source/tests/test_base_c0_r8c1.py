"""R8-C1 diagnostic evaluation regressions; no CARLA server or logs required."""

import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from evaluation import base_c0_protocol as c0
from evaluation import evaluate_base_policy_c0 as runner
from evaluation import r8c1_diagnostic_protocol as r8
from evaluation import validate_base_c0_run as validator
from evaluation.evaluate_base_policy_c0 import _run_case
from models.actor_critic import Actor
from tests.test_checkpoint_resume import _checkpoint_payload
from training.checkpoint_protocol import (
    CheckpointProtocolError, atomic_write_json, files_fingerprint, save_checkpoint_file, sha256_file,
)
from training.critic_diagnostics import build_critic_probe_batch
from training.metrics_protocol import REWARD_COMPONENTS, REWARD_DIAGNOSTICS


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_DIR / "configs/evaluation/p0_base_c0_town05_r8c1_probe_v1.json"
PROTOCOL = "p0_base_c0_diag_town05_r8c1_v1"
CHECKPOINT_SHA = "c" * 64


class _CheckpointFixture:
    """Self-contained schema-8 files; real run artifacts are never required."""

    def __init__(self, root, contract):
        self.root = root
        self.contract = copy.deepcopy(contract)
        policy = self.contract["diagnostic_policy"]
        policy.update({
            "training_run_id": "P0_fixture_s42", "training_run_path": "logs/P0_fixture_s42",
            "training_contract_path": "configs/training_fixture.json",
            "checkpoint_manifest_path": "logs/P0_fixture_s42/checkpoint_manifest.json",
            "training_git_sha": "1" * 40,
            "training_fingerprint_paths": ["shared/actor.py", "shared/environment.py"],
            "evaluator_source_paths": ["evaluation/runner.py"],
        })
        for relative, contents in {
            "shared/actor.py": "actor behavior\n",
            "shared/environment.py": "environment behavior\n",
            "evaluation/runner.py": "evaluator behavior\n",
            "configs/training_fixture.json": '{"fixture":true}\n',
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        policy["training_contract_sha256"] = sha256_file(root / policy["training_contract_path"])
        policy["training_code_fingerprint"] = files_fingerprint(
            root / relative for relative in policy["training_fingerprint_paths"]
        )
        policy["training_source_sha256"] = {
            relative: sha256_file(root / relative) for relative in policy["training_fingerprint_paths"]
        }
        self.contract["checkpoint"]["required_training_contract_sha256"] = policy["training_contract_sha256"]
        policy["checkpoints"] = {
            str(update): {"path": f"{policy['training_run_path']}/periodic/ckpt_update_{update:06d}.pt",
                          "sha256": "0" * 64} for update in (20, 60, 100)
        }
        self.path = root / policy["checkpoints"]["100"]["path"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / policy["checkpoint_manifest_path"]
        env = self.contract["environment"]
        payload = _checkpoint_payload()
        payload.update({
            "checkpoint_schema_version": 8,
            "training_protocol_version": r8.R8C1_TRAINING_PROTOCOL_VERSION,
            "protocol_versions": copy.deepcopy(r8.R8C1_EXPECTED_PROTOCOL_VERSIONS),
            "checkpoint_kind": "periodic", "update": 100,
            "actors": {i: Actor(obs_dim=20, act_dim=2, hidden=128).state_dict() for i in (0, 1)},
            "model_spec": copy.deepcopy(r8.R8C1_MODEL_SPEC),
            "training_config": {
                "reward_version": "v6", "gamma": env["reward_gamma"], "town": env["town"],
                "timestep": env["timestep"], "max_steps": env["max_steps"],
                "lidar_range_m": env["lidar_range_m"], "ttc_max_range_m": env["ttc_max_range_m"],
                "curriculum_protocol": "r8a_mastery_v1", "scenario_sampling_mode": "r8a_balanced",
            },
            "critic_probe_batch": build_critic_probe_batch(
                states=[[float(row + column) for column in range(40)] for row in range(4)],
                agent_ids=[0, 1, 0, 1], targets=[0.0, 1.0, -1.0, 2.0], maximum_rows=4,
            ),
        })
        payload["run_metadata"].update({
            "run_id": policy["training_run_id"], "seed": 42,
            "origin_git_sha": policy["training_git_sha"], "last_git_sha": policy["training_git_sha"],
            "code_fingerprint": policy["training_code_fingerprint"],
            "origin_code_fingerprint": policy["training_code_fingerprint"],
            "experiment_contract_id": r8.R8C1_TRAINING_CONTRACT_ID,
            "experiment_contract_sha256": policy["training_contract_sha256"],
            "config_sha256": policy["training_contract_sha256"],
            "source_type": "fresh", "code_drift_override": False,
            "runtime": {"environment_reward_gamma": env["reward_gamma"],
                        "offroad_protocol": copy.deepcopy(env["offroad_protocol"]),
                        "road_legality_protocol": copy.deepcopy(env["road_legality_protocol"])},
        })
        payload["last_metrics"].update({
            "metrics_protocol_version": r8.R8C1_EXPECTED_PROTOCOL_VERSIONS["metrics"],
            "run_id": policy["training_run_id"], "update": 100,
        })
        self.payload = payload
        self.write(payload)
        # Copy frozen, tracked scenarios only. No logs from the workspace enter
        # this fixture, and case expansion still reads and hashes real files.
        for profile in contract["profiles"].values():
            for name in profile["scenario_names"]:
                relative = Path(profile["scenario_dir"]) / name
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((PROJECT_DIR / relative).read_bytes())

    def write(self, payload):
        self.sha256 = save_checkpoint_file(payload, self.path)
        policy = self.contract["diagnostic_policy"]
        policy["checkpoints"]["100"]["sha256"] = self.sha256
        atomic_write_json({
            "run_id": policy["training_run_id"], "checkpoint_schema_version": 8,
            "training_protocol_version": r8.R8C1_TRAINING_PROTOCOL_VERSION,
            "periodic": [{"update": 100, "sha256": self.sha256,
                          "path": "periodic/ckpt_update_000100.pt"}],
        }, self.manifest_path)
        policy["checkpoint_manifest_sha256"] = sha256_file(self.manifest_path)

    def load(self, *, allow_diagnostic=False):
        return c0.load_and_validate_c0_checkpoint(
            self.path, contract=self.contract, project_dir=self.root,
            allow_diagnostic_checkpoint=allow_diagnostic,
        )


def _road_evidence(violation=False):
    return {
        "classification": "opposing_lane" if violation else "reference_lane",
        "reference_lane": [44, -1],
        "legal_lane_change": False,
        "lane_deviation_exempted": False,
        "violation_reason": "opposing_lane" if violation else "",
        "violation_raw": violation,
        "consecutive_violation_ticks": 5 if violation else 0,
        "violation_confirmed": violation,
        "legal_transition_grace_remaining": 0,
        "success_eligible": not violation,
        "update_reference_lane": not violation,
        "same_reference_lane": not violation,
        "adjacent_to_reference": False,
        "lane_change_permitted": False,
        "current_to_reference_cosine": -1.0 if violation else 1.0,
        "vehicle_to_current_cosine": 1.0,
        "in_junction": False,
    }


class _TrafficManager:
    def set_random_device_seed(self, seed):
        self.seed = seed


class _CountingActor(torch.nn.Module):
    """A real immutable state_dict, with observable inference calls."""

    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor([0.25, 0.0]))
        self.calls = 0
        self.grad_modes = []
        self.eval()

    def get_deterministic_action(self, observations):
        self.calls += 1
        self.grad_modes.append(torch.is_grad_enabled())
        return torch.tanh(self.mean).expand(observations.shape[0], -1)


class _FakeEnv:
    """Models terminal events and inactive samples, not vehicle dynamics.

    All observations for inactive agents are deliberately omitted, and reward
    payloads contain only the agents that actually acted.  This catches an
    evaluator that keeps evaluating an inactive policy or dilutes its totals.
    """

    REASONS = {
        "success": "cooperative_success",
        "collision": "cooperative_collision",
        "offroad": "cooperative_offroad",
        "road_violation": "cooperative_road_violation",
        "timeout": "mission_timeout",
    }

    def __init__(self, outcome="timeout", *, end_after=None, early_goal=False,
                 ignore_horizon=False, never_end=False):
        self.traffic_manager = _TrafficManager()
        self.max_steps = 500
        self.outcome = outcome
        self.end_after = end_after
        self.early_goal = early_goal
        self.ignore_horizon = ignore_horizon
        self.never_end = never_end
        self.horizon_history = []
        self.action_ids = []

    def apply_scenario_config(self, config):
        self.config = copy.deepcopy(config)
        if not self.ignore_horizon:
            self.max_steps = config.get("episode_max_steps", 500)

    def reset(self):
        self.tick = 0
        self.active = {0: True, 1: True}
        self.reached_target = {0: False, 1: False}
        self.horizon_history.append(self.max_steps)
        return self._obs(), {"active_mask": dict(self.active), "episode_end_reason": "running"}

    def _obs(self):
        return {i: np.zeros(20, dtype=np.float32) for i in (0, 1) if self.active[i]}

    def step(self, actions):
        expected = {i for i in (0, 1) if self.active[i]}
        if set(actions) != expected:
            raise AssertionError(f"actions for inactive agent: {set(actions)} != {expected}")
        self.action_ids.append(tuple(sorted(actions)))
        self.tick += 1
        terminal = not self.never_end and self.tick >= (self.end_after or self.max_steps)
        reason = self.REASONS[self.outcome] if terminal else "running"
        if self.early_goal and self.tick == 1:
            self.reached_target[0] = True
            self.active[0] = False
        if terminal:
            if self.outcome == "success":
                self.reached_target = {0: True, 1: True}
            self.active = {0: False, 1: False}

        road = {i: terminal and self.outcome == "road_violation" and i == 0 for i in (0, 1)}
        offroad = {i: terminal and self.outcome == "offroad" and i == 0 for i in (0, 1)}
        truncated = {i: terminal and i == 1 and self.outcome in {"collision", "offroad", "road_violation"}
                     for i in (0, 1)}
        dones = {i: not self.active[i] for i in (0, 1)}
        terminated = {i: dones[i] and not truncated[i] for i in (0, 1)}
        details, diagnostics, rewards = {}, {}, {}
        for i in expected:
            parts = {name: 0.0 for name in REWARD_COMPONENTS["v6"]}
            raw = {name: 0.0 for name in REWARD_DIAGNOSTICS["v6"]}
            raw.update({"road_same_reference_lane": 1.0,
                        "road_success_eligible": 1.0,
                        "road_current_to_reference_cosine": 1.0,
                        "road_vehicle_to_current_cosine": 1.0})
            if not terminal:
                parts["free_flow_pace_deficit"] = -0.002
                raw.update({"pace_free_flow_tag": 1.0, "pace_non_interactive_tag": 1.0,
                            "pace_deficit_eligible": 1.0, "pace_deficit_raw": 0.2,
                            "pace_deficit_risk": 0.2, "pace_deficit_reward": -0.002})
            if self.reached_target[i]:
                parts["success"] = 20.0
                raw["success_event"] = 1.0
            if terminal and i == 0 and self.outcome == "collision":
                parts["collision"] = -50.0
                raw["collision_event"] = 1.0
            if offroad[i]:
                parts["offroad"] = -50.0
                raw.update({"offroad_raw": 1.0, "offroad_terminated": 1.0,
                            "offroad_consecutive_ticks": 3.0, "footprint_outside_fraction": 0.5})
            if road[i]:
                parts["road_rule"] = -50.0
                raw.update({"road_violation_raw": 1.0, "road_violation_terminated": 1.0,
                            "road_violation_consecutive_ticks": 5.0, "road_success_eligible": 0.0,
                            "road_current_to_reference_cosine": -1.0})
            if terminal and self.outcome == "timeout" and not self.reached_target[i]:
                parts["mission_timeout"] = -10.0
                raw["mission_timeout_event"] = 1.0
            details[i], diagnostics[i], rewards[i] = parts, raw, sum(parts.values())

        first_offroad = {i: None for i in (0, 1)}
        if offroad[0]:
            first_offroad[0] = {"episode_step": self.tick, "world_frame": self.tick,
                                "location": {"x": 0.0, "y": 0.0, "z": 0.5}, "yaw": 0.0,
                                "footprint_outside_count": 4, "footprint_outside_fraction": 0.5}
        info = {
            "reward_details": details, "reward_diagnostics": diagnostics,
            "terminated": terminated, "truncated": truncated,
            "active_mask": dict(self.active), "episode_end_reason": reason,
            "offroad_terminated": offroad, "first_offroad_evidence": first_offroad,
            "max_offroad_consecutive_ticks": {i: 3 if offroad[i] else 0 for i in (0, 1)},
            "max_footprint_outside_fraction": {i: 0.5 if offroad[i] else 0.0 for i in (0, 1)},
            "road_violation_confirmed": road, "road_rule_terminated": road,
            "road_legality": {i: _road_evidence(road[i]) for i in (0, 1)},
        }
        return self._obs(), rewards, dones, info


class R8C1EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_record = c0.load_c0_contract(CONTRACT_PATH, project_dir=PROJECT_DIR)
        cls.contract = cls.contract_record["payload"]
        cls.short_cases = c0.expand_profile_cases(cls.contract, "short_probe", project_dir=PROJECT_DIR)
        cls.long_cases = c0.expand_profile_cases(cls.contract, "long_probe", project_dir=PROJECT_DIR)

    def run_case(self, case=None, *, env=None, outcome="timeout", end_after=None, profile="short_probe"):
        case = case or self.short_cases[0]
        if end_after is None and outcome != "timeout":
            end_after = 3
        env = env or _FakeEnv(outcome, end_after=end_after)
        actors = {i: _CountingActor() for i in (0, 1)}
        record = _run_case(env=env, actors=actors, case=case, contract=self.contract,
                           eval_id="fixture", profile=profile, checkpoint_sha256=CHECKPOINT_SHA)
        return record, env, actors

    def validate(self, record, case=None):
        c0.validate_episode_record(record, eval_id="fixture", profile=record["profile"],
                                   checkpoint_sha256=CHECKPOINT_SHA,
                                   expected_case=case or self.short_cases[0])

    def prepare_fixture_run(self, fixture):
        contract_path = fixture.root / "configs/evaluation/r8_fixture.json"
        atomic_write_json(fixture.contract, contract_path)
        contract_record = {
            "payload": fixture.contract, "path": str(contract_path),
            "sha256": sha256_file(contract_path),
            "canonical_sha256": c0.canonical_json_sha256(fixture.contract),
        }
        checkpoint_record = fixture.load()
        cases = c0.expand_profile_cases(fixture.contract, "short_probe", project_dir=fixture.root)
        eval_id = "P0_BASE_C0_short_probe_u000100_20260914_120000"
        output_dir = fixture.root / fixture.contract["outputs"]["evaluation_root"] / eval_id
        args = Namespace(eval_id=eval_id, profile="short_probe", output_dir=str(output_dir),
                         resume=False, host="127.0.0.1", port=2000)
        runner._prepare_output(args=args, contract_record=contract_record,
                               checkpoint_record=checkpoint_record, cases=cases)
        records = [
            _run_case(env=_FakeEnv("success", end_after=3),
                      actors={i: _CountingActor() for i in (0, 1)}, case=case,
                      contract=fixture.contract, eval_id=eval_id, profile=args.profile,
                      checkpoint_sha256=fixture.sha256)
            for case in cases
        ]
        return args, contract_record, checkpoint_record, cases, records

    def test_registered_matrix_is_diagnostic_and_has_42_executions(self):
        self.assertEqual(self.contract["protocol_version"], PROTOCOL)
        self.assertEqual(set(self.contract["profiles"]), {"smoke", "short_probe", "long_probe"})
        smoke = c0.expand_profile_cases(self.contract, "smoke", project_dir=PROJECT_DIR)
        self.assertEqual((len(smoke), len(self.short_cases), len(self.long_cases)), (1, 4, 8))
        self.assertEqual(2 * (len(smoke) + 3 * len(self.short_cases) + len(self.long_cases)), 42)
        self.assertEqual({case["effective_episode_max_steps"] for case in self.short_cases}, {200})
        self.assertEqual({case["effective_episode_max_steps"] for case in self.long_cases}, {500})
        for case in self.short_cases + self.long_cases:
            self.assertEqual(case["effective_episode_max_steps"], case["scenario_config"]["episode_max_steps"])

    def test_frozen_matrix_rejects_seed_selection_repeat_and_scenario_hash_drift(self):
        changes = {
            "smoke_seed": lambda c: c["profiles"]["smoke"].update(episode_seeds=[12345]),
            "probe_seed": lambda c: c["profiles"]["short_probe"].update(episode_seeds=[12345]),
            "selection": lambda c: c["profiles"]["short_probe"].update(formal_use="checkpoint_selection_validation"),
            "repeat_count": lambda c: c["repeatability"].update(required_repeats=3),
            "scenario_order": lambda c: c["profiles"]["short_probe"]["scenario_names"].reverse(),
            "scenario_hash": lambda c: c["diagnostic_policy"]["scenario_sha256"].update({r8.R8C1_SCENARIO_NAMES[0]: "0" * 64}),
            "source_hash": lambda c: c["diagnostic_policy"]["training_source_sha256"].update({f"{r8.R8C1_SCENARIO_DIR}/{r8.R8C1_SCENARIO_NAMES[0]}": "0" * 64}),
        }
        for name, mutate in changes.items():
            contract = copy.deepcopy(self.contract)
            mutate(contract)
            with self.subTest(name=name), self.assertRaises(c0.C0ProtocolError):
                c0.validate_c0_contract(contract, project_dir=PROJECT_DIR)

    def test_new_and_historical_profiles_are_isolated(self):
        with self.assertRaises(c0.C0ProtocolError):
            c0.expand_profile_cases(self.contract, "validation", project_dir=PROJECT_DIR)
        for filename in ("p0_base_c0_town05_r6_v4.json", "p0_base_c0_town05_r7_v5.json"):
            old = c0.load_c0_contract(PROJECT_DIR / "configs/evaluation" / filename,
                                      project_dir=PROJECT_DIR)["payload"]
            self.assertEqual(set(old["profiles"]), {"smoke", "gate", "validation"})
            with self.assertRaises(c0.C0ProtocolError):
                c0.expand_profile_cases(old, "short_probe", project_dir=PROJECT_DIR)

    def test_horizon_resets_200_500_200_and_timeout_occurs_at_limit(self):
        env = _FakeEnv()
        records = []
        for case, profile in ((self.short_cases[0], "short_probe"),
                              (self.long_cases[0], "long_probe"),
                              (self.short_cases[2], "short_probe")):
            record, _, _ = self.run_case(case, env=env, profile=profile)
            records.append(record)
        self.assertEqual(env.horizon_history, [200, 500, 200])
        self.assertEqual([r["episode_length"] for r in records], [200, 500, 200])
        for record in records:
            self.assertEqual(record["outcome"], "timeout")
            self.assertEqual(record["reset_episode_max_steps"], record["effective_episode_max_steps"])
            self.assertEqual(record["agents"]["0"]["active_decision_steps"], record["episode_length"])

    def test_horizon_mismatch_rejected_before_action_and_missing_termination_is_bounded(self):
        wrong = _FakeEnv(ignore_horizon=True)
        with self.assertRaises(c0.C0ProtocolError):
            self.run_case(env=wrong)
        self.assertEqual(wrong.action_ids, [])
        missing_terminal = _FakeEnv(never_end=True)
        with self.assertRaises(c0.C0ProtocolError):
            self.run_case(env=missing_terminal)
        self.assertEqual(missing_terminal.tick, 200)

    def test_inactive_agent_has_no_observation_action_reward_or_denominator_padding(self):
        record, env, actors = self.run_case(env=_FakeEnv("success", end_after=3, early_goal=True))
        self.assertEqual(env.action_ids, [(0, 1), (1,), (1,)])
        self.assertEqual([actors[i].calls for i in (0, 1)], [1, 3])
        self.assertTrue(all(not mode for actor in actors.values() for mode in actor.grad_modes))
        self.assertEqual(record["team"]["total_active_decision_steps"], 4)
        self.assertEqual(record["agents"]["0"]["actions"]["count"], 1)
        self.assertEqual(record["agents"]["0"]["diagnostics"]["pace_deficit_reward"]["count"], 1)
        self.assertAlmostEqual(record["agents"]["0"]["episode_return"], 19.998)

    def test_runner_and_validator_agree_on_all_five_outcomes(self):
        for outcome in _FakeEnv.REASONS:
            with self.subTest(outcome=outcome):
                record, _, _ = self.run_case(outcome=outcome)
                self.assertEqual(record["outcome"], outcome)
                self.validate(record)
                self.assertEqual(record["evaluation_purpose"], "diagnostic_only")
                self.assertIs(record["selection_allowed"], False)
                self.assertEqual(record["performance_decision"], "NOT_EVALUATED")

    def test_record_rejects_horizon_reward_pace_and_road_tampering(self):
        record, _, _ = self.run_case(outcome="road_violation")
        mutations = (
            lambda r: r.update(effective_episode_max_steps=500),
            lambda r: r.update(reset_episode_max_steps=500),
            lambda r: r.update(episode_length=201),
            lambda r: r.update(selection_allowed=True),
            lambda r: r["agents"]["0"]["reward_component_sums"].update(free_flow_pace_deficit=9.0),
            lambda r: r["agents"]["0"].update(road_violation_confirmed=False),
            lambda r: r["agents"]["0"]["road_legality"].update(success_eligible=True),
            lambda r: r.update(outcome="timeout"),
        )
        for mutation in mutations:
            altered = copy.deepcopy(record)
            mutation(altered)
            with self.subTest(mutation=mutation), self.assertRaises(c0.C0ProtocolError):
                self.validate(altered)

    def test_r8_case_rejects_downgraded_record_and_false_terminal_evidence(self):
        record, _, _ = self.run_case()
        downgraded = copy.deepcopy(record)
        downgraded["record_version"] = c0.TOWN05_R7_C0_EPISODE_RECORD_VERSION
        for name in (*r8.diagnostic_usage(), "effective_episode_max_steps", "reset_episode_max_steps", "scenario_group", "direction"):
            downgraded.pop(name)
        with self.assertRaisesRegex(c0.C0ProtocolError, "R8C1.*version"):
            self.validate(downgraded)
        for done, term, trunc in ((False, False, False), (True, False, True)):
            altered = copy.deepcopy(record)
            agent = altered["agents"]["0"]
            agent.update(final_done=done, final_terminated=term, final_truncated=trunc)
            agent["lifecycle_transition"].update(done=done, terminated=term, truncated=trunc)
            with self.subTest(flags=(done, term, trunc)), self.assertRaises(c0.C0ProtocolError):
                self.validate(altered)
        altered = copy.deepcopy(record)
        agent = altered["agents"]["0"]
        agent["reward_component_sums"]["free_flow_pace_deficit"] += 0.5
        agent["episode_return"] += 0.5
        altered["team"]["mean_episode_return"] += 0.25
        with self.assertRaisesRegex(c0.C0ProtocolError, "pace component"):
            self.validate(altered)

    def test_partial_constructor_failure_restores_timing_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _CheckpointFixture(root, self.contract)
            contract_path = root / "configs/evaluation/r8_fixture.json"
            atomic_write_json(fixture.contract, contract_path)
            contract_record = {"payload": fixture.contract, "path": str(contract_path),
                               "sha256": sha256_file(contract_path),
                               "canonical_sha256": c0.canonical_json_sha256(fixture.contract)}
            eval_id = "P0_BASE_C0_smoke_u000100_20260914_121000"
            args = Namespace(eval_contract=contract_path, checkpoint=fixture.path,
                             eval_id=eval_id, profile="smoke", output_dir=root / "evaluations" / eval_id,
                             resume=False, host="127.0.0.1", port=2000, allow_diagnostic_checkpoint=False)
            world = MagicMock()
            world.get_settings.return_value = Namespace(synchronous_mode=False, fixed_delta_seconds=None)
            world.get_actors.return_value.filter.return_value = []
            client = MagicMock()
            client.get_world.return_value = world
            client.get_client_version.return_value = "0.9.16"
            client.get_server_version.return_value = "0.9.16"

            def fail_constructor(**kwargs):
                self.assertIs(kwargs["spawn_vehicles"], False)
                settings = world.get_settings()
                settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.1
                raise RuntimeError("constructor RPC failed after synchronous mode")

            with patch.object(runner, "PROJECT_DIR", root), patch.object(runner, "REPOSITORY_ROOT", root), \
                    patch.object(runner, "load_c0_contract", return_value=contract_record), \
                    patch.object(runner.carla, "Client", return_value=client), \
                    patch.object(runner, "CarlaMultiAgentEnv", side_effect=fail_constructor):
                with self.assertRaisesRegex(RuntimeError, "constructor RPC failed"):
                    runner.run_evaluation(args)
            manifest = c0.load_strict_json(args.output_dir / fixture.contract["outputs"]["manifest"])
            self.assertEqual(manifest["status"], "HOLD_ERROR")
            self.assertEqual(manifest["cleanup_error"], "")
            self.assertTrue(manifest["post_cleanup"]["world_timing_restored"])
            self.assertFalse(world.get_settings().synchronous_mode)
            self.assertIsNone(world.get_settings().fixed_delta_seconds)
            client.get_trafficmanager.return_value.set_synchronous_mode.assert_called_once_with(False)

            def working_constructor(**kwargs):
                env = _FakeEnv("timeout")
                env.client, env.world = client, world
                settings = world.get_settings()
                settings.synchronous_mode, settings.fixed_delta_seconds = True, kwargs["timestep"]

                def close():
                    settings.synchronous_mode = env._original_synchronous_mode
                    settings.fixed_delta_seconds = env._original_fixed_delta_seconds

                env.close = close
                return env

            # Resume the actual production entry after a recoverable constructor
            # failure. Real checkpoint/source/rollout/file validation still runs.
            args.resume = True
            with patch.object(runner, "PROJECT_DIR", root), patch.object(runner, "REPOSITORY_ROOT", root), \
                    patch.object(validator, "PROJECT_DIR", root), patch.object(validator, "REPOSITORY_ROOT", root), \
                    patch.object(runner, "load_c0_contract", return_value=contract_record), \
                    patch.object(validator, "load_c0_contract", return_value=contract_record), \
                    patch.object(runner.carla, "Client", return_value=client), \
                    patch.object(runner, "CarlaMultiAgentEnv", side_effect=working_constructor):
                report = runner.run_evaluation(args)
            self.assertEqual(report["status"], "PASS", report.get("errors"))
            self.assertEqual(report["performance_decision"], "NOT_EVALUATED")
            self.assertFalse(report["formal_eligible"])
            saved = c0.read_jsonl(args.output_dir / fixture.contract["outputs"]["episodes_jsonl"])
            self.assertEqual(saved[0]["episode_length"], 200)
            self.assertEqual(saved[0]["outcome"], "timeout")

    def test_summary_keeps_rolling_and_stationary_separate_and_preserves_v6(self):
        records = [self.run_case(case, outcome="success" if index < 2 else "road_violation")[0]
                   for index, case in enumerate(self.short_cases)]
        summary = c0.build_summary(records, eval_id="fixture", profile="short_probe",
                                   checkpoint_sha256=CHECKPOINT_SHA)
        rolling = summary["by_family"]["r8a_short_free_flow_rolling"]
        stationary = summary["by_family"]["r8a_short_free_flow_stationary"]
        self.assertEqual(rolling["outcome_counts"]["success"], 2)
        self.assertEqual(stationary["outcome_counts"]["road_violation"], 2)
        self.assertEqual(set(summary["overall"]["outcome_counts"]), set(_FakeEnv.REASONS))
        self.assertIn("free_flow_pace_deficit", summary["agents"]["0"]["mean_reward_component_sum"])
        self.assertIs(summary["selection_allowed"], False)
        rows = c0.summary_csv_rows(summary)
        self.assertTrue(all("road_violation_count" in row for row in rows))

    def test_repeatability_catches_changed_road_agent_and_discrete_cause(self):
        record, _, _ = self.run_case(outcome="road_violation")
        mutations = (
            lambda r: r["agents"].update({"0": copy.deepcopy(record["agents"]["1"]),
                                          "1": copy.deepcopy(record["agents"]["0"])}),
            lambda r: r["agents"]["0"]["road_legality"].update(violation_reason="wrong_way"),
            lambda r: r["agents"]["0"]["road_legality"].update(reference_lane=[44, -2]),
            lambda r: r["agents"]["0"]["road_legality"].update(classification="wrong_way"),
        )
        for mutation in mutations:
            changed = copy.deepcopy(record)
            mutation(changed)
            comparison = c0.compare_repeatability_records([record], [changed], contract=self.contract)
            self.assertFalse(comparison["gate_pass"])

    def test_repeatability_does_not_treat_road_cosines_as_bitwise_identity(self):
        record, _, _ = self.run_case(outcome="road_violation")
        changed = copy.deepcopy(record)
        changed["agents"]["0"]["road_legality"]["current_to_reference_cosine"] = -0.99
        comparison = c0.compare_repeatability_records([record], [changed], contract=self.contract)
        self.assertTrue(comparison["gate_pass"])

    def test_timeout_before_effective_horizon_is_rejected(self):
        with self.assertRaises(c0.C0ProtocolError):
            self.run_case(outcome="timeout", end_after=3)

    def test_actor_weight_change_is_rejected(self):
        class MutatingActor(_CountingActor):
            def get_deterministic_action(self, observations):
                with torch.no_grad():
                    self.mean.add_(0.01)
                return super().get_deterministic_action(observations)

        with self.assertRaisesRegex(c0.C0ProtocolError, "Actor state changed"):
            _run_case(env=_FakeEnv("success", end_after=3),
                      actors={0: MutatingActor(), 1: _CountingActor()},
                      case=self.short_cases[0], contract=self.contract,
                      eval_id="fixture", profile="short_probe", checkpoint_sha256=CHECKPOINT_SHA)

    def test_schema8_checkpoint_loads_with_identity_but_never_selection_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CheckpointFixture(Path(directory), self.contract)
            loaded = fixture.load()
            self.assertEqual(loaded["update"], 100)
            self.assertEqual(loaded["sha256"], fixture.sha256)
            self.assertFalse(loaded["formal_eligible"])
            self.assertTrue(any("diagnostic_only" in reason for reason in loaded["formal_ineligibility_reasons"]))
            actors = runner._build_actors(loaded["checkpoint"])
            self.assertEqual(r8.actor_state_hashes(actors),
                             r8.actor_state_hashes(loaded["checkpoint"]["actors"]))
            self.assertTrue(all(not actor.training for actor in actors.values()))

    def test_semantic_checkpoint_tampering_fails_even_with_diagnostic_switch(self):
        mutations = {
            "schema": lambda p: p.update(checkpoint_schema_version=7),
            "model": lambda p: p["model_spec"].update(state_dim=12),
            "action": lambda p: p["action_spec"].update(action_dim=3),
            "reward_protocol": lambda p: p["protocol_versions"].update(reward="v5"),
            "sampling_protocol": lambda p: p["protocol_versions"].update(scenario_sampling="other"),
            "curriculum": lambda p: p["training_config"].update(curriculum_protocol="r7"),
            "sampling": lambda p: p["training_config"].update(scenario_sampling_mode="random"),
            "gamma": lambda p: p["training_config"].update(gamma=0.99),
            "actual_gamma": lambda p: p["run_metadata"]["runtime"].update(environment_reward_gamma=0.99),
            "run": lambda p: p["run_metadata"].update(run_id="unregistered"),
            "contract": lambda p: p["run_metadata"].update(experiment_contract_sha256="0" * 64),
            "source": lambda p: p["run_metadata"].update(code_fingerprint="0" * 64),
            "code_drift": lambda p: p["run_metadata"].update(code_drift_override=True),
            "probe": lambda p: p.update(critic_probe_batch=None),
            "actor_shape": lambda p: p["actors"].update({0: _CountingActor().state_dict()}),
            "latest": lambda p: p.update(checkpoint_kind="latest"),
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CheckpointFixture(Path(directory), self.contract)
            for name, mutation in mutations.items():
                altered = copy.deepcopy(fixture.payload)
                mutation(altered)
                # Re-sign both file and manifest, so failure must detect the
                # semantic violation instead of relying on stale file hashes.
                fixture.write(altered)
                with self.subTest(name=name), self.assertRaises((c0.C0ProtocolError, CheckpointProtocolError)):
                    fixture.load(allow_diagnostic=True)

    def test_checkpoint_sidecar_manifest_and_registered_hash_are_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CheckpointFixture(Path(directory), self.contract)
            sidecar = fixture.path.with_name(fixture.path.name + ".sha256")
            sidecar.write_text("0" * 64, encoding="utf-8")
            with self.assertRaises(CheckpointProtocolError):
                fixture.load()
            fixture.write(fixture.payload)
            manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
            manifest["periodic"][0]["sha256"] = "0" * 64
            atomic_write_json(manifest, fixture.manifest_path)
            fixture.contract["diagnostic_policy"]["checkpoint_manifest_sha256"] = sha256_file(fixture.manifest_path)
            with self.assertRaises(c0.C0ProtocolError):
                fixture.load()
            fixture.write(fixture.payload)
            fixture.contract["diagnostic_policy"]["checkpoints"]["100"]["sha256"] = "0" * 64
            with self.assertRaises(c0.C0ProtocolError):
                fixture.load()

    def test_source_provenance_separates_evaluator_from_shared_training_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CheckpointFixture(Path(directory), self.contract)
            kwargs = {"contract": fixture.contract, "project_dir": fixture.root,
                      "repository_root": fixture.root, "checkpoint_record": fixture.load()}
            with patch.object(r8, "get_git_metadata", return_value={"commit": "2" * 40}):
                saved = r8.build_source_provenance(**kwargs)
            self.assertEqual(saved["training"]["git_sha"], "1" * 40)
            self.assertEqual(saved["evaluation"]["git_sha"], "2" * 40)
            with patch.object(r8, "get_git_metadata", return_value={"commit": "3" * 40}):
                r8.validate_source_provenance(saved, **kwargs)
            evaluator_path = fixture.root / "evaluation/runner.py"
            evaluator_path.write_text("changed evaluator\n", encoding="utf-8")
            with self.assertRaisesRegex(r8.R8C1DiagnosticError, "evaluator"):
                r8.validate_source_provenance(saved, **kwargs)
            evaluator_path.write_text("evaluator behavior\n", encoding="utf-8")
            (fixture.root / "shared/environment.py").write_text("changed behavior\n", encoding="utf-8")
            with self.assertRaisesRegex(r8.R8C1DiagnosticError, "source drift"):
                r8.build_source_provenance(**kwargs)

    def test_resume_requires_a_complete_ordered_episode_prefix(self):
        records = [self.run_case(case, outcome="success")[0] for case in self.short_cases]
        c0.ensure_completed_prefix(records[:2], self.short_cases)
        for wrong in ([records[1]], [records[0], records[0]], records + [records[0]]):
            with self.subTest(case_ids=[row["case_id"] for row in wrong]), self.assertRaises(c0.C0ProtocolError):
                c0.ensure_completed_prefix(wrong, self.short_cases)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            c0.append_jsonl(path, records[0])
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"partial":')
            with self.assertRaises(c0.C0ProtocolError):
                c0.read_jsonl(path)

    def test_prepare_resume_rejects_changed_implementation_cases_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _CheckpointFixture(root, self.contract)
            with patch.object(runner, "PROJECT_DIR", root), patch.object(runner, "REPOSITORY_ROOT", root):
                args, contract_record, checkpoint_record, cases, records = self.prepare_fixture_run(fixture)
                kwargs = {"args": args, "contract_record": contract_record,
                          "checkpoint_record": checkpoint_record, "cases": cases}
                with self.assertRaisesRegex(c0.C0ProtocolError, "already exists"):
                    runner._prepare_output(**kwargs)
                output_dir = Path(args.output_dir)
                outputs = fixture.contract["outputs"]
                c0.append_jsonl(output_dir / outputs["episodes_jsonl"], records[0])
                args.resume = True
                _, recovered = runner._prepare_output(**kwargs)
                self.assertEqual([row["case_id"] for row in recovered], [cases[0]["case_id"]])
                resolved_path = output_dir / outputs["resolved_config"]
                manifest_path = output_dir / outputs["manifest"]
                resolved = c0.load_strict_json(resolved_path)
                manifest = c0.load_strict_json(manifest_path)
                for name, mutate in {
                    "horizon": lambda r: r["cases"][0].update(effective_episode_max_steps=500),
                    "case_order": lambda r: r["cases"].reverse(),
                    "environment": lambda r: r["environment_spec"].update(reward_gamma=0.99),
                    "actor_identity": lambda r: r["actor_state_sha256_before"].update({"0": "0" * 64}),
                }.items():
                    changed = copy.deepcopy(resolved)
                    mutate(changed)
                    atomic_write_json(changed, resolved_path)
                    resigned = {**manifest, "resolved_config_sha256": sha256_file(resolved_path)}
                    atomic_write_json(resigned, manifest_path)
                    with self.subTest(name=name), self.assertRaises(c0.C0ProtocolError):
                        runner._prepare_output(**kwargs)
                atomic_write_json(resolved, resolved_path)
                atomic_write_json(manifest, manifest_path)
                (root / "evaluation/runner.py").write_text("new evaluator\n", encoding="utf-8")
                with self.assertRaisesRegex(r8.R8C1DiagnosticError, "evaluator"):
                    runner._prepare_output(**kwargs)

    def test_independent_validator_rebuilds_files_and_ignores_cached_pass_after_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _CheckpointFixture(root, self.contract)
            with patch.object(runner, "PROJECT_DIR", root), patch.object(runner, "REPOSITORY_ROOT", root):
                args, contract_record, _, cases, records = self.prepare_fixture_run(fixture)
            output_dir, outputs = Path(args.output_dir), fixture.contract["outputs"]
            resolved_path = output_dir / outputs["resolved_config"]
            manifest_path = output_dir / outputs["manifest"]
            episodes_path = output_dir / outputs["episodes_jsonl"]
            summary_path = output_dir / outputs["summary_json"]
            csv_path = output_dir / outputs["summary_csv"]
            resolved = c0.load_strict_json(resolved_path)
            resolved["runtime"].update(carla_client_version="0.9.16-test", carla_server_version="0.9.16-test")
            atomic_write_json(resolved, resolved_path)
            for record in records:
                c0.append_jsonl(episodes_path, record)
            summary = c0.build_summary(records, eval_id=args.eval_id, profile=args.profile,
                                       checkpoint_sha256=fixture.sha256)
            atomic_write_json(summary, summary_path)
            c0.atomic_write_csv(csv_path, c0.summary_csv_rows(summary))
            manifest = c0.load_strict_json(manifest_path)
            manifest.update({
                "status": "DIAGNOSTIC_PASS", "completed_cases": len(cases),
                "resolved_config_sha256": sha256_file(resolved_path),
                "episodes_sha256": sha256_file(episodes_path),
                "summary_sha256": sha256_file(summary_path), "summary_csv_sha256": sha256_file(csv_path),
                "actor_state_sha256_after": manifest["actor_state_sha256_before"],
                "checkpoint_sha256_after": fixture.sha256,
                "preflight": {"carla_client_version": "0.9.16-test", "carla_server_version": "0.9.16-test",
                              "vehicle_actor_count": 0, "sensor_actor_count": 0,
                              "synchronous_mode": False, "fixed_delta_seconds": None},
                "post_cleanup": {"vehicle_actor_count": 0, "sensor_actor_count": 0,
                                 "world_timing_restored": True, "synchronous_mode": False,
                                 "fixed_delta_seconds": None},
            })
            atomic_write_json(manifest, manifest_path)
            # Only frozen contract admission is substituted for this synthetic
            # run. Checkpoint, source, case, JSONL and summary validation remain
            # real and re-read disk independently on every call.
            with patch.object(validator, "PROJECT_DIR", root), \
                    patch.object(validator, "REPOSITORY_ROOT", root), \
                    patch.object(validator, "load_c0_contract", return_value=contract_record):
                report = validator.validate_c0_run(output_dir, expected_profile="short_probe")
                self.assertEqual(report["status"], "PASS", report.get("errors"))
                self.assertEqual(report["performance_decision"], "NOT_EVALUATED")
                self.assertFalse(report["formal_eligible"])
                original_episode_bytes = episodes_path.read_bytes()
                altered = copy.deepcopy(records)
                altered[0]["effective_episode_max_steps"] = 500
                episodes_path.write_text("".join(json.dumps(row) + "\n" for row in altered), encoding="utf-8")
                manifest["episodes_sha256"] = sha256_file(episodes_path)
                atomic_write_json(manifest, manifest_path)
                report = validator.validate_c0_run(output_dir)
                self.assertEqual(report["status"], "HOLD")
                episodes_path.write_bytes(original_episode_bytes)
                manifest["episodes_sha256"] = sha256_file(episodes_path)
                atomic_write_json(manifest, manifest_path)
                report = validator.validate_c0_run(output_dir)
                self.assertEqual(report["status"], "PASS", report.get("errors"))
                changed_summary = copy.deepcopy(summary)
                changed_summary["overall"]["outcome_counts"]["success"] = 0
                atomic_write_json(changed_summary, summary_path)
                manifest["summary_sha256"] = sha256_file(summary_path)
                atomic_write_json(manifest, manifest_path)
                report = validator.validate_c0_run(output_dir)
                self.assertEqual(report["status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
