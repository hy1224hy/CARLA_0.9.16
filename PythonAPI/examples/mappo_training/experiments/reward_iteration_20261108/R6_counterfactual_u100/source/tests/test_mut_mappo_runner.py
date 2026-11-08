"""Small synthetic lifecycle integration; never a CARLA performance test."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from training.train_mut_mappo import Runtime, train, validate_checkpoint_for_config
from training.mut_mappo_config import load_config, validate_config, run_identity
from training.mut_mappo_learning import TeamLearner
from training.mut_mappo_checkpoint import CheckpointStore


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
        cls.config = validate_config(load_config("configs/mut_mappo/local_candidate_v1.json"))
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


if __name__ == "__main__":
    unittest.main()
