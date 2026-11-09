"""Small consequential checks: reward, geometry, return credit and runnable chain."""
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from environment.mut_mappo_efficiency_reward import TeamEfficiencyReward, settle_team_tail
from evaluation.driving_policy_metrics import DrivingMetrics
from evaluation.evaluate_mut_mappo_credit import credit_summary
from training.first_point_scenarios import ROOT, build_cases, sample_pairs
from training.mut_mappo_learning import GateState


class FakeTeamEnv:
    def __init__(self, callback, team="A", condition="C2"):
        self.callback = callback or (lambda event: None)
        self.team, self.condition = team, condition
        self.num_agents = 2 if condition == "C0" else 4
        self._world_tick_counts = {"total": 0, "setup": 0, "decision": 0}
        self.callback({"kind": "setup", "ticks": 1})

    def reset_case(self, case):
        self.steps = 0
        self.obs = {aid: np.zeros(20, np.float32) for aid in range(self.num_agents)}
        self.callback({"kind": "setup", "ticks": 1})
        return self.obs, {"active_mask": dict.fromkeys(self.obs, True)}

    def step(self, controls):
        self.steps += 1
        self.callback({"kind": "decision", "ticks": 1})
        active = {aid: self.steps < (2 if aid == 2 else 4) for aid in self.obs}
        done = {aid: not value for aid, value in active.items()}
        self.obs = {aid: np.full(20, .01 * self.steps if active[aid] else 0, np.float32) for aid in self.obs}
        utility = .2 if self.team == "A" else .1
        efficiency = utility if self.condition == "C2" else 0.0
        return self.obs, {aid: .1 * self.steps if aid in controls else 0.0 for aid in self.obs}, done, {
            "active_mask": active, "agent_terminated": done, "agent_truncated": dict.fromkeys(self.obs, False),
            "episode_end_reason": "running" if any(active.values()) else "all_agents_reached",
            "team_step_reward": .1 * self.steps,
            "mut_metrics": {"valid": True, "red_success": True, "blue_eligible": True,
                "response": False, "safety": {}, "safe_efficiency_utility": efficiency,
                "attack_integral": efficiency, "team_utility_total": utility,
                "route_progress": {aid: min(1, self.steps / 4) for aid in self.obs}},
            "world_tick_counts": dict(self._world_tick_counts)}

    def get_global_state(self):
        return np.concatenate(list(self.obs.values()))

    def set_cost_coefficient(self, value):
        self.cost = value

    def close(self):
        self.callback({"kind": "setup", "ticks": 1})


class FirstPointTests(unittest.TestCase):
    def test_pool_and_complete_pair_rotation(self):
        manifest = build_cases()
        self.assertEqual(len(build_cases(parallel_only=True)["splits"]["train"]), 8)
        self.assertEqual({k: len(v) for k, v in manifest["splits"].items()},
                         {"train": 24, "gate": 16, "selection": 16, "test": 16})
        cases = [c for c in manifest["cases"] if c["case_key"] in manifest["splits"]["train"]]
        seen = set()
        for update in range(8):
            selected = sample_pairs(cases, update)
            self.assertEqual(len(selected), 4)
            self.assertEqual(selected[0]["pair_id"], selected[1]["pair_id"])
            self.assertEqual(selected[2]["pair_id"], selected[3]["pair_id"])
            self.assertEqual({c["family"] for c in selected}, {"parallel", "staggered"})
            seen.update(c["case_key"] for c in selected)
        self.assertEqual(len(seen), 24)

    def test_actual_map_resolves_small_pool_and_blue_goals(self):
        import carla
        from environment.mut_mappo_env import MutMAPPOEnv
        from utils.build_mut_mappo_symmetric_cases_v2 import DEFAULT_XODR
        env = MutMAPPOEnv.__new__(MutMAPPOEnv)
        env.map = carla.Map("Town05", DEFAULT_XODR.read_text(encoding="utf-8"))
        env.red_ids, env.blue_ids, env.num_agents, env.default_max_steps = (0, 1), (2, 3), 4, 500
        signatures = set()
        for case in build_cases()["cases"]:
            scenario, evidence = env._resolve_case(case)
            signatures.add(tuple(tuple(round(float(a[k]), 5) for k in ("x", "y", "yaw")) for a in scenario["agents"].values()))
            for aid in range(4):
                goal = scenario["agent_goal_zones"][f"agent{aid}"]
                wp = env._get_exact_driving_waypoint(carla.Location(x=goal["x"], y=goal["y"], z=goal["z"]))
                self.assertIsNotNone(wp)
                self.assertEqual(wp.lane_id, evidence["waypoints"][f"agent{aid}"]["lane_id"])
        self.assertEqual(len(signatures), 72)

    def test_sustained_lag_and_first_team_safety_cost(self):
        reward = TeamEfficiencyReward(horizon=10)
        reference = {str(r): [i / 10 for i in range(11)] for r in (0, 1)}
        for step in range(1, 11):
            _, parts = reward.step(step=step, progress={0: step / 20, 1: step / 20, 2: 0, 3: 0},
                reached={}, eligible=True, safety={}, costs={}, base_details={}, reference=reference)
            self.assertGreater(parts["attack"], 0)
        self.assertAlmostEqual(reward.attack_integral, .275)
        for step in (11, 12):
            _, parts = reward.step(step=step, progress={0: .5, 1: .5, 2: 0, 3: 0}, reached={},
                eligible=False, safety={2: {"collision": True}, 3: {"collision": True}}, costs={}, base_details={}, reference=reference)
            self.assertEqual(parts["safety_event"], -50 if step == 11 else 0)
            self.assertEqual(parts["attack"], 0)

    def test_environment_returns_shared_reward_instead_of_legacy_bonuses(self):
        from environment.carla_multi_agent_env import CarlaMultiAgentEnv
        from environment.mut_mappo_events import MutEpisodeEvents
        from tests.test_mut_mappo_events import MutLifecycleTests
        env = MutLifecycleTests.env()
        env._case, env.episode_end_reason, env.follow_spectator = {"case_key": "x"}, "running", False
        env.current_step = 31
        env.team_reward = TeamEfficiencyReward()
        env._team_reference = {"0": [0.5] * 501, "1": [0.5] * 501}
        env.mut_events = MutEpisodeEvents()
        env._last_mut_step_events = {"birth_protected": False}
        env._team_base_details = {}
        env._route_axes = {i: (0, 0, 1, 0, 10) for i in range(4)}
        env._route_progress = dict.fromkeys(range(4), 0.0)
        env._post_tick_kinematics = {i: {"location": {"x": 3, "y": 0}} for i in range(4)}
        env._last_mut_reward_details = {b: {"team_impact": 5, "team_threat": 0, "team_timeout": 0,
            "driving": 0, "lambda": 1, "safety_cost": 0, "costs": {}} for b in (2, 3)}
        env._get_observations = lambda: {i: np.zeros(20, np.float32) for i in range(4)}
        env._mut_info = lambda: {}
        result = (env._get_observations(), {2: 5, 3: 5}, dict.fromkeys(range(4), False), {"reward_details": {}})
        with patch.object(CarlaMultiAgentEnv, "step", return_value=result):
            _, rewards, _, info = env.step({i: np.zeros(3) for i in range(4)})
        self.assertEqual(rewards[2], rewards[3])
        self.assertEqual(rewards[2], info["team_step_reward"])
        self.assertNotIn("team_impact", info["reward_details"][2])

    def test_faster_completion_has_known_negative_absorbing_tail(self):
        reward = TeamEfficiencyReward(horizon=4)
        reference = {"0": [0, .25, .5, .75, 1], "1": [0, .25, .5, .75, 1]}
        _, parts = reward.step(step=2, progress={0: 1, 1: 1, 2: 0, 3: 0}, reached={0: True, 1: True},
            eligible=True, safety={}, costs={}, base_details={}, reference=reference)
        self.assertLess(parts["completed_task_tail"], 0)

    def test_retired_role_keeps_full_discounted_team_return_without_fake_actions(self):
        timeline, gamma = [1, 2, 3, 4, 5], .9
        buffer = SimpleNamespace(data={2: [SimpleNamespace(reward=float(r)) for r in timeline[:2]],
                                       3: [SimpleNamespace(reward=float(r)) for r in timeline]})
        settle_team_tail(buffer, {2: 0, 3: 0}, timeline, gamma)
        expected = sum(gamma ** i * r for i, r in enumerate(timeline))
        self.assertAlmostEqual(sum(gamma ** i * r.reward for i, r in enumerate(buffer.data[2])), expected)
        self.assertEqual(len(buffer.data[2]), 2)

    def test_efficiency_gate_does_not_depend_on_old_response_events(self):
        gate = GateState({"quality_metric": "efficiency", "min_efficiency_gap": .02})
        stats = {team: {"parallel": {"n_all": 8, "q_count": 0, "b_count": 0,
                 "utility_sum": value, "case_keys": list(range(8)), "teacher_hash": team, "version": 1}}
                 for team, value in (("A", 0), ("B", .8))}
        result = gate.refresh({"A": "A", "B": "B"}, stats, version=1, completed_round=1)
        self.assertEqual(result, {"A": {"parallel": 1}, "B": {"parallel": 0}})

    def test_driving_missing_ttc_is_not_zero_and_rolling_is_not_startup(self):
        meter = DrivingMetrics(initial_speeds={0: 2.5})
        for step in range(1, 7):
            meter.update({0: True}, {0: {"speed": 2.5, "rotation": {"yaw": 0},
                "velocity": {"x": 2.5, "y": 0}, "control": {"steer": 0}}},
                {0: {"pace_front_blocker_exempt": False, "distance_to_goal_m": 20}}, {})
        summary = meter.summary()["0"]
        self.assertIsNone(summary["minimum_closing_ttc_s"])
        self.assertIsNone(summary["unblocked_first_motion_seconds"])
        self.assertEqual(summary["abs_acceleration_p95_mps2"], 0)

    def test_negative_shapley_contribution_is_retained(self):
        rows = [{"team": "A", "case_key": "x", "family": "parallel", "valid": True,
                 "blue_eligible": True, "credit_combination": c, "team_utility_total": u}
                for c, u in (("00", 0), ("10", 2), ("01", -3), ("11", 0))]
        case = credit_summary(rows)["cases"][0]
        self.assertAlmostEqual(case["phi_blue2"], 2.5)
        self.assertAlmostEqual(case["phi_blue3"], -2.5)
        self.assertAlmostEqual(case["interaction"], 1)

    def test_real_ppo_resume_selection_export_test_and_credit_chain(self):
        from training.checkpoint_protocol import atomic_write_json, sha256_file
        from training.mut_mappo_config import load_config, validate_config, portable_config, make_actors
        from training.train_mut_mappo import Runtime, train
        from evaluation.evaluate_mut_mappo import evaluate
        from evaluation.evaluate_mut_mappo_credit import evaluate as credit_evaluate
        from environment.mut_mappo_efficiency_reward import reference_identity, ReferenceStore
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / "cases.json"
            atomic_write_json(build_cases(), manifest)
            cfg = portable_config(load_config("configs/mut_mappo/local_candidate_v1.json"))
            cfg.update(case_manifest=str(manifest), method="plain")
            cfg["task"].update(reward_profile="c1_progress_team_v1", team_reward={"reference_dir": str(root / "references")})
            cfg["mutual"].update(schedule_clock="rounds", warmup_rounds=1, refresh_rounds=10, quality_metric="efficiency")
            cfg["run"].update(rounds=2, episodes_per_team_update=4)
            cfg["learning"].update(ppo_epochs=1, minibatch_size=16)
            cfg["evaluation"]["selection_metric"] = "safe_efficiency_utility"
            config_path = root / "config.json"
            atomic_write_json(cfg, config_path)
            cfg = validate_config(cfg)
            reference_dir = root / "references"

            def factory(config, checkpoint=None, condition="C2", team="A", audit_callback=None):
                source = config["_red_payload"]
                red = make_actors(source["actors"], source["training_config"], (0, 1))
                if checkpoint and "teams" in checkpoint:
                    state = checkpoint["teams"][team]
                    blue = make_actors(state["actors"], state["config"], (2, 3))
                elif checkpoint and "actors" in checkpoint:
                    blue = make_actors(checkpoint["actors"], checkpoint["training_config"], (2, 3))
                else:
                    blue = make_actors({2: source["actors"][0], 3: source["actors"][1]}, source["training_config"], (2, 3))
                return Runtime(config, FakeTeamEnv(audit_callback, team, condition), red, blue, condition, team)

            from evaluation.mut_mappo_reference import prepare as prepare_references
            with patch("training.train_mut_mappo.create_runtime", side_effect=factory), \
                 patch("utils.mut_mappo_session.CarlaOwner", side_effect=lambda *a, **k: contextlib.nullcontext()), \
                 contextlib.redirect_stdout(io.StringIO()):
                reference_report = prepare_references(config_path, ("train", "gate", "selection", "test"))
            self.assertEqual(reference_report["cases"], 72)
            self.assertEqual(ReferenceStore(reference_dir, reference_identity(cfg)).curve(cfg["_case_manifest"]["cases"][0]["case_key"])["0"][-1], 1.0)
            with contextlib.redirect_stdout(io.StringIO()):
                first = train(cfg, root / "run", stop_round=1, env_factory=factory)
                second = train(cfg, root / "run", resume=True, stop_round=2, env_factory=factory)
            self.assertEqual(first["completed_round"], 1)
            self.assertEqual(second["completed_round"], 2)
            self.assertGreater(second["last_metrics"]["teams"]["A"]["mutual_kl_2"], 0)
            checkpoint = root / "run/latest.pt"
            with patch("training.train_mut_mappo.create_runtime", side_effect=factory), \
                 patch("utils.mut_mappo_session.CarlaOwner", side_effect=lambda *a, **k: contextlib.nullcontext()), \
                 contextlib.redirect_stdout(io.StringIO()):
                selection = evaluate(config_path, checkpoint, output_dir=root / "selection", select_best=True)
                self.assertEqual(selection["selection"]["status"], "SELECTED")
                frozen = root / "selection/frozen_blue_team.pt"
                test = evaluate(config_path, frozen, split="test", output_dir=root / "test")
                credit = credit_evaluate(config_path, frozen, root / "credit")
            self.assertEqual(test["paired"]["n_common_valid"], 16)
            self.assertEqual(credit["n_complete"], 16)
            from utils.run_first_point_experiments import summarize
            atomic_write_json({"run_id": "B1_s42", "group": "B1", "seed": 42, "completed_round": 2,
                "run_dir": str(root / "run"), "selection_status": "SELECTED", "selected_team": "A",
                "test_dir": str(root / "test"), "credit_dir": str(root / "credit")},
                root / "results/B1_s42_r2.json")
            summary = summarize(root, {"base_config": str(config_path), "entries": [{"run_id": "B1_s42"}], "train_tick_budget": 12000})
            self.assertEqual(summary["groups"]["B1"]["metrics"]["safe_efficiency_utility"]["n"], 1)
            self.assertTrue((root / "comparison.csv").exists())


if __name__ == "__main__":
    unittest.main()
