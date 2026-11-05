"""Small synthetic lifecycle integration; never a CARLA performance test."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.actor_critic import Actor
from training.checkpoint_protocol import save_checkpoint_file

from training.train_mut_mappo import Runtime, train, validate_checkpoint_for_config
from training.mut_mappo_config import load_config, validate_config, run_identity
from training.mut_mappo_learning import TeamLearner
from training.mut_mappo_checkpoint import CheckpointStore
from training.mut_mappo_schedule import learning_rate_multiplier, validate_lr_schedule


class FakeLifecycleEnv:
    num_agents = 4

    def __init__(self, callback):
        self.callback = callback
        self.steps = 0
        self.obs = {aid: np.zeros(20, dtype=np.float32) for aid in range(4)}
        self.callback({"kind": "setup", "ticks": 1})

    def reset_case(self, case):
        self.steps = 0
        self.obs = {aid: np.full(20, 0.02 * (aid+1), dtype=np.float32) for aid in range(4)}
        self.callback({"kind": "setup", "ticks": 1})
        return self.obs, {"active_mask": dict.fromkeys(range(4), True)}

    def step(self, actions):
        self.steps += 1
        self.callback({"kind": "decision", "ticks": 1})
        # blue2 exits early; both red and blue3 continue to a real boundary.
        active = {aid: self.steps < (2 if aid == 2 else 4) for aid in range(4)}
        done = {aid: not flag for aid, flag in active.items()}
        for aid in range(4):
            self.obs[aid] = np.full(20, 0.01 * self.steps if active[aid] else 0, dtype=np.float32)
        info = {"active_mask": active, "terminated": done, "truncated": dict.fromkeys(range(4), False),
                "episode_end_reason": "running" if any(active.values()) else "all_agents_reached",
                "mut_metrics": {"valid": True, "red_success": True, "blue_eligible": True,
                                "blue_safe": True, "response": False, "safety": {}}, "world_tick_counts": {}}
        return self.obs, {aid: (0.2+self.steps*0.1) if aid in actions else 0.0 for aid in range(4)}, done, info

    def get_global_state(self):
        return np.concatenate([self.obs[aid] for aid in range(4)])

    def set_cost_coefficient(self, value):
        self.cost = value

    def close(self):
        self.callback({"kind": "setup", "ticks": 1})


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Integration tests must not require a private trained model in Git.
        directory = tempfile.TemporaryDirectory(prefix="TEST_ONLY_mut_red_")
        cls.addClassCleanup(directory.cleanup)
        path = Path(directory.name) / "synthetic_red.pt"
        raw = load_config("configs/mut_mappo/local_candidate_v1.json")
        with torch.random.fork_rng():
            torch.manual_seed(71)
            states = {aid: Actor(20, 2, hidden=128).state_dict() for aid in (0, 1)}
        digest = save_checkpoint_file({
            "update": 1, "model_spec": {"obs_dim": 20, "action_dim": 2, "hidden_dim": 128},
            "actors": states,
            "training_config": {"hidden_dim": 128, "gamma": .999,
                                "min_log_std": -2., "max_log_std": 0.},
        }, path)
        raw["red_checkpoint"] = {"path": str(path), "sha256": digest,
                                 "capability_status": "TEST_ONLY_SYNTHETIC_NOT_DRIVING_EVIDENCE"}
        cls.config = validate_config(raw)
        cls.config["run"].update(episodes_per_team_update=1, rounds=2)
        cls.config["learning"].update(ppo_epochs=1, minibatch_size=16)

    @staticmethod
    def factory(config, audit_callback):
        # Frozen red actor presence is exercised without running real physics.
        from training.mut_mappo_config import make_actors
        source = config["_red_payload"]
        red = make_actors(source["actors"], source["training_config"], (0, 1))
        return Runtime(config, FakeLifecycleEnv(audit_callback), red, {}, "C2", "A")

    def test_partial_b_round_retains_complete_ab_and_cost_on_resume(self):
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            safe = train(config, run, stop_round=1, env_factory=self.factory)
            self.assertEqual(safe["completed_round"], 1)
            self.assertEqual([safe["teams"][n]["update_count"] for n in ("A", "B")], [1, 1])
            self.assertEqual(validate_checkpoint_for_config(config, run / "latest.pt")["completed_round"], 1)
            changed_task = copy.deepcopy(config)
            changed_task["task"]["max_steps"] += 1
            with self.assertRaisesRegex(ValueError, "task/reward/lifecycle"):
                validate_checkpoint_for_config(changed_task, run / "latest.pt")
            monitor_only = copy.deepcopy(config)
            monitor_only["task"]["follow_spectator"] = False
            self.assertEqual(validate_checkpoint_for_config(monitor_only, run / "latest.pt")["completed_round"], 1)
            actual_before = safe["counters"]["train"]
            original_update = TeamLearner.update

            def interrupt_b(team, buffer, *args, **kwargs):
                if team.name == "B":
                    raise RuntimeError("fixture interruption before B update")
                return original_update(team, buffer, *args, **kwargs)

            with patch.object(TeamLearner, "update", interrupt_b):
                with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
                    train(config, run, resume=True, stop_round=2, env_factory=self.factory)
            store = CheckpointStore(run, run_identity(config))
            self.assertEqual(store.load()["completed_round"], 1)
            resumed = train(config, run, resume=True, stop_round=2, env_factory=self.factory)
            self.assertEqual([resumed["teams"][n]["update_count"] for n in ("A", "B")], [2, 2])
            self.assertGreater(resumed["counters"]["train"], actual_before + 10)
            rows = [json.loads(line) for line in store.metrics_path.read_text().splitlines()]
            self.assertEqual([row["completed_round"] for row in rows], [0, 1, 2])
            self.assertTrue(all("not_independent" in row.get("source", "not_independent") for row in rows))

    def test_single_mappo_and_ippo_are_real_single_team_boundaries(self):
        for algorithm in ("mappo", "ippo"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                config = copy.deepcopy(self.config)
                config.update(algorithm=algorithm, team_count=1)
                run = Path(directory) / "single"
                first = train(config, run, stop_round=1, env_factory=self.factory)
                self.assertEqual(set(first["teams"]), {"A"})
                self.assertEqual(set(first["teachers"]), {"A"})
                self.assertEqual(first["team_count"], 1)
                self.assertEqual(first["next_team"], "A")
                restored = train(config, run, resume=True, stop_round=2, env_factory=self.factory)
                self.assertEqual(restored["teams"]["A"]["update_count"], 2)
                self.assertEqual(validate_checkpoint_for_config(config, run / "latest.pt")["completed_round"], 2)
                if algorithm == "ippo":
                    self.assertEqual(set(restored["teams"]["A"]["value_normalizer"]), {2, 3})
                    self.assertEqual(restored["teams"]["A"]["model_spec"]["critic_input_scope"],
                                     "own_physical_slot_local_observation_only")

    def test_budget_stop_does_not_give_single_team_half_the_dual_budget(self):
        for team_count in (1, 2):
            with self.subTest(team_count=team_count), tempfile.TemporaryDirectory() as directory:
                config = copy.deepcopy(self.config)
                config.update(team_count=team_count)
                config["run"].update(rounds=1, schedule_clock="train_ticks", stop_rule="train_ticks", train_tick_budget=17)
                payload = train(config, Path(directory) / "ticks", env_factory=self.factory)
                self.assertGreaterEqual(payload["counters"]["train"], 17)
                self.assertEqual(payload["last_metrics"]["stop_rule"], "train_ticks")
                self.assertGreaterEqual(payload["last_metrics"]["train_tick_overshoot"], 0)

    def test_lr_schedule_defaults_overrides_and_invalid_options(self):
        tick = {"schedule_clock": "train_ticks", "train_tick_budget": 100}
        rounds = {"lr_horizon_rounds": 100}
        for run, ticks, round_no, expected in (
                (tick, 0, 1, 1.), (tick, 25, 1, .75), (tick, 100, 1, .1),
                (tick, 150, 1, .1), (rounds, 99, 1, 1.), (rounds, 99, 26, .75),
                (rounds, 99, 101, .1), (rounds, 99, 150, .1),
                (dict(tick, lr_decay_train_ticks=200, lr_min_factor=.2), 100, 1, .5),
                (dict(tick, lr_decay_train_ticks=200, lr_min_factor=.2), 300, 1, .2),
                (dict(tick, lr_horizon_train_ticks=200), 100, 1, .5),
                (dict(tick, lr_horizon_train_ticks=200, lr_decay_train_ticks=200), 100, 1, .5),
                ({"schedule_clock": "train_ticks", "train_tick_budget": None,
                  "lr_horizon_train_ticks": 200}, 100, 1, .5),
                (dict(tick, lr_min_factor=0.), 100, 1, 0.),
                (dict(tick, lr_schedule="constant"), 150, 1, 1.),
                ({"lr_schedule": "constant"}, 150, 150, 1.)):
            with self.subTest(run=run, ticks=ticks, round_no=round_no):
                before = copy.deepcopy(run)
                validate_lr_schedule(run)
                self.assertAlmostEqual(learning_rate_multiplier(run, ticks, round_no), expected)
                self.assertEqual(run, before)
        invalid = [{"lr_schedule": value} for value in ("cosine", None, True)]
        invalid += [{"schedule_clock": "wall_time"}, dict(rounds, lr_decay_train_ticks=10)]
        invalid += [dict(rounds, lr_horizon_train_ticks=10),
                    dict(tick, lr_horizon_train_ticks=200, lr_decay_train_ticks=300)]
        invalid += [dict(tick, lr_decay_train_ticks=value) for value in (0, -1, True, 2.5)]
        invalid += [dict(tick, lr_horizon_train_ticks=value) for value in (0, -1, True, 2.5)]
        invalid += [dict(tick, lr_min_factor=value) for value in
                    (-.1, 1.1, True, "0.1", float("nan"), float("inf"))]
        invalid += [dict(tick, lr_schedule="constant", **{key: value}) for key, value in
                    (("lr_decay_train_ticks", 100), ("lr_horizon_train_ticks", 100), ("lr_min_factor", .1))]
        for run in invalid:
            with self.subTest(invalid=run), self.assertRaises(ValueError):
                validate_lr_schedule(run)

    def test_round_target_ignores_ticks_and_resume_preserves_round_lr(self):
        for budget in (None, 1):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                config = copy.deepcopy(self.config)
                config["run"].update(stop_rule="rounds", schedule_clock="rounds", rounds=3,
                                     lr_horizon_rounds=3, train_tick_budget=budget)
                config = validate_config(config)
                if budget is None:
                    invalid = copy.deepcopy(config)
                    invalid["run"].update(stop_rule="train_ticks", schedule_clock="train_ticks")
                    with self.assertRaisesRegex(ValueError, "positive train_tick_budget"):
                        validate_config(invalid)
                identity = run_identity(config)
                run = Path(directory) / "rounds"
                with patch("training.train_mut_mappo.MutMAPPOMonitor") as monitor:
                    first = train(config, run, stop_round=1, env_factory=self.factory)
                    self.assertEqual(first["completed_round"], 1)
                    self.assertFalse(first["last_metrics"]["training_complete"])
                    resumed = train(config, run, resume=True, env_factory=self.factory)
                    self.assertEqual(monitor.call_args.kwargs["planned_rounds"], 3)
                self.assertEqual(resumed["completed_round"], 3)
                self.assertEqual([resumed["teams"][n]["update_count"] for n in ("A", "B")], [3, 3])
                status = json.loads((run / "session/status.json").read_text())
                self.assertEqual((status["completed_updates"], status["target_updates"]), (3, 3))
                self.assertTrue(status["training_complete"])
                self.assertFalse(status["training_budget_reached"])
                self.assertFalse(status["tick_budget_enforced"])
                self.assertIsNone(status["train_tick_overshoot"])
                self.assertGreater(status["actual_counters"]["train"], budget or 0)
                self.assertEqual(run_identity(config), identity)
                store = CheckpointStore(run, identity)
                rows = [json.loads(line) for line in store.metrics_path.read_text().splitlines()][1:]
                self.assertEqual([row["completed_updates"] for row in rows], [1, 2, 3])
                for row in rows:
                    factor = max(.1, 1 - (row["completed_round"] - 1) / 3)
                    for team in row["teams"].values():
                        for rates in team["learning_rates"].values():
                            self.assertAlmostEqual(rates["actor"][0], config["learning"]["lr_actor"] * factor)
                            self.assertAlmostEqual(rates["critic"][0], config["learning"]["lr_critic"] * factor)
                with self.assertRaisesRegex(ValueError, "registered update target"):
                    train(config, run, resume=True, stop_round=4, env_factory=self.factory)
                with self.assertRaisesRegex(ValueError, "tick-budget run"):
                    train(config, run, resume=True, stop_train_ticks=10, env_factory=self.factory)

    def test_absolute_tick_segment_stop_resume_preserves_identity_and_lr_clock(self):
        config = copy.deepcopy(self.config)
        config["run"].update(schedule_clock="train_ticks", stop_rule="train_ticks", train_tick_budget=100)
        identity = run_identity(config)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "segments"
            first = train(config, run, stop_train_ticks=11, env_factory=self.factory)
            self.assertEqual(first["completed_round"], 1)
            self.assertEqual(first["counters"]["train"], 11)
            self.assertEqual([first["teams"][name]["update_count"] for name in ("A", "B")], [1, 1])
            resolved = (run / "resolved_config.json").read_bytes()
            store = CheckpointStore(run, identity)
            self.assertEqual(store.load()["identity"], identity)
            resumed = train(config, run, resume=True, stop_train_ticks=25, env_factory=self.factory)
            # Each team collects one reset plus four decision ticks. Cleanup
            # and recreation add two real training ticks between segments.
            self.assertEqual(resumed["completed_round"], 3)
            self.assertEqual(resumed["counters"]["train"], 33)
            self.assertEqual([resumed["teams"][name]["update_count"] for name in ("A", "B")], [3, 3])
            self.assertEqual(run_identity(config), identity)
            self.assertEqual(store.load()["identity"], identity)
            self.assertEqual((run / "resolved_config.json").read_bytes(), resolved)
            rows = [json.loads(line) for line in store.metrics_path.read_text().splitlines()][1:]
            self.assertEqual([row["actual_counters"]["train"] for row in rows], [11, 23, 33])
            for row in rows:
                order = ("A", "B") if row["completed_round"] % 2 else ("B", "A")
                for index, name in enumerate(order):
                    update_tick = row["actual_counters"]["train"] - (5 if index == 0 else 0)
                    expected_factor = max(.1, 1. - update_tick / 100)
                    for rates in row["teams"][name]["learning_rates"].values():
                        for kind, key in (("actor", "lr_actor"), ("critic", "lr_critic")):
                            for rate in rates[kind]:
                                self.assertAlmostEqual(rate, config["learning"][key] * expected_factor)


if __name__ == "__main__":
    unittest.main()
