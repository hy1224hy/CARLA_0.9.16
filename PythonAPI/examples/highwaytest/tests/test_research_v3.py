"""Validity checks, not tests requiring the proposed algorithm to win."""
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from ..config import Config, Task, ROOT
from ..common.contract import PROTOCOL, VehicleFrame, SceneFrame, inputs, rotation
from ..common.objective import PairedObjective
from ..common.outcomes import OutcomeTracker, paired_outcome
from ..environment.paired import PairedAttackEnv
from ..environment.conditions import HighwayConditionBackend
from ..evaluation.paired import evaluate_paired
from ..adapters.environment import PortablePairedEnv
from ..adapters.carla import PolylineRoute, CarlaFrameAdapter, CarlaAccelerationControl
from ..adapters.train_carla import train_carla
from ..models.team import Team
from ..research import plan, report_study
from ..training.runner import run


def task(**kwargs):
    return Task(research_protocol=PROTOCOL, **kwargs)


def scene(time=0, speed=10, progress=0, collision=False):
    return SceneFrame({i: VehicleFrame(i, (i * 15.0, (i % 2) * 4.0), (speed, 0),
            0, 0, 0, progress, collision and i == 2) for i in range(4)}, time, (0, 0))


class ResearchContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        (ROOT / "runs").mkdir(exist_ok=True)

    def test_portable_inputs_rotation_translation_invariance(self):
        base = scene(time=5, progress=10)
        matrix = rotation(-1.1)
        translation = np.array([135, -500])
        transformed = SceneFrame({i: replace(v, position=tuple(matrix @ v.position + translation),
            velocity=tuple(matrix @ v.velocity), heading=v.heading + 1.1,
            route_heading=v.route_heading + 1.1) for i, v in base.vehicles.items()},
            base.time, tuple(translation), 1.1)
        a, b = inputs(base, task()), inputs(transformed, task())
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-6)
        self.assertEqual(a[0].shape, (2, 25))
        self.assertEqual(a[1].shape, (29,))
        self.assertEqual(float(a[0][0, 6]), 0.25)

    def test_qualification_latches_failures_and_overspeed(self):
        tracker = OutcomeTracker(task(duration=1, goal_distance=1, overspeed_grace_s=0.1))
        tracker.update(scene())
        tracker.update(scene(time=0.2, speed=23, progress=2))
        tracker.update(scene(time=1, progress=10))
        self.assertTrue(tracker.summary()["overspeed"])
        self.assertFalse(tracker.summary()["qualified"])
        tracker = OutcomeTracker(task(duration=1, goal_distance=1))
        tracker.update(scene(collision=True))
        tracker.update(scene(time=1, progress=10))
        self.assertTrue(tracker.summary()["collision"])

    def test_censored_or_stalled_attack_never_receives_primary_credit(self):
        t = task(goal_distance=100)
        reference = {"fixed_horizon_complete": True, "bad": False, "qualified": True, "red_progress_m": 200}
        failed = {"fixed_horizon_complete": False, "bad": True, "qualified": False, "red_progress_m": 1}
        value = paired_outcome(reference, reference, failed, t)
        self.assertEqual(value["qualified_attack_utility"], 0)
        self.assertIsNone(value["extra_red_loss_m"])
        stalled = failed | {"fixed_horizon_complete": True, "bad": False}
        self.assertEqual(paired_outcome(reference, reference, stalled, t)["qualified_attack_utility"], 0)
        qualified = stalled | {"qualified": True, "red_progress_m": 150}
        self.assertEqual(paired_outcome(reference, reference, qualified, t)["qualified_attack_utility"], 0.5)

    def test_attack_potential_refunded_on_failure_or_incomplete_goal(self):
        for failure in (True, False):
            t = task(attack_weight=1, duration=0.6, goal_distance=100)
            objective = PairedObjective(t, 0.99)
            discounted_attack = 0
            for index in range(3):
                last = index == 2
                safety = {"bad": last and failure, "fixed_horizon_complete": last,
                          "qualified": False, "collision": last and failure}
                _, parts = objective.step(scene(time=(index + 1) * 0.2, progress=(index + 1) * 5),
                    safety, normal_red_progress=(index + 1) * 10, baseline_valid=True)
                discounted_attack += 0.99 ** index * parts["attack"]
            self.assertAlmostEqual(discounted_attack, 0, places=12)

    def test_goal_does_not_end_before_fixed_horizon(self):
        env = PairedAttackEnv(task(duration=0.4, goal_distance=0.1))
        try:
            env.reset(seed=1)
            _, _, done, trunc, info = env.step(np.zeros((2, 2)))
            self.assertTrue(info["blue_goals_met"])
            self.assertFalse(done)
            _, _, done, trunc, info = env.step(np.zeros((2, 2)))
            self.assertTrue(done)
            self.assertFalse(trunc)  # Finite-horizon task termination, not infinite-horizon timeout.
            self.assertTrue(info["qualified"])
        finally:
            env.close()

    def test_generic_wrapper_matches_highway_objective_and_features(self):
        t = task(duration=0.4, goal_distance=0.1, attack_weight=1)
        native = PairedAttackEnv(t)
        portable = PortablePairedEnv(t, 0.99, lambda c, f, s: HighwayConditionBackend(t, c, f, s))
        try:
            a, ai = native.reset(seed=7, options={"family": "closing"})
            b, bi = portable.reset(seed=7, options={"family": "closing"})
            np.testing.assert_allclose(a, b)
            for _ in range(2):
                ra, rb = native.step(np.zeros((2, 2))), portable.step(np.zeros((2, 2)))
                np.testing.assert_allclose(ra[0], rb[0], atol=1e-6)
                np.testing.assert_allclose(ra[1], rb[1], atol=1e-6)
        finally:
            native.close()
            portable.close()

    def test_normal_condition_negative_control_is_zero(self):
        t = task(duration=1, goal_distance=1, red_lane_change=True, attack_weight=1)
        factory = lambda c, f, s: HighwayConditionBackend(t, "C1" if c == "C2" else c, f, s)
        result, _ = evaluate_paired(None, t, "selection", 2, backend_factory=factory)
        self.assertEqual(result["overall"]["success_rate"], 1)
        self.assertEqual(result["overall"]["baseline_invalid_count"], 0)
        self.assertEqual(result["overall"]["qualified_attack_utility"], 0)
        for row in result["episodes"]:
            self.assertAlmostEqual(row["conditions"]["C0"]["elapsed_s"], 1)
            self.assertAlmostEqual(row["conditions"]["C1"]["elapsed_s"], 1)
            self.assertAlmostEqual(row["conditions"]["C2"]["elapsed_s"], 1)

    def test_unpaired_initial_conditions_are_rejected(self):
        t = task(duration=0.4, goal_distance=0.1)
        def factory(condition, family, seed):
            backend = HighwayConditionBackend(t, "C1" if condition == "C2" else condition, family, seed)
            if condition == "C2":
                cars = dict(backend.frame.vehicles)
                cars[0] = replace(cars[0], position=(cars[0].position[0] + 1, cars[0].position[1]))
                backend.frame = replace(backend.frame, vehicles=cars)
            return backend
        with self.assertRaisesRegex(ValueError, "not paired"):
            evaluate_paired(None, t, "selection", 1, backend_factory=factory)

    def test_fractional_policy_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer number"):
            Config(task=task(duration=20.1)).validate()

    def test_changing_red_policy_does_not_change_normal_blue_controller(self):
        env = PairedAttackEnv(task(red_lane_change=False, red_target_speed=16,
                                  normal_blue_lane_change=True, normal_blue_target_speed=18))
        try:
            env.reset(seed=1)
            self.assertTrue(all(not v.enable_lane_change and v.target_speed == 16 for v in env.road.vehicles[:2]))
            self.assertTrue(all(v.enable_lane_change and v.target_speed == 18 for v in env.normal_reference.vehicles[2:]))
        finally:
            env.close()

    def test_v3_network_and_ppo_minimal_loop(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            cfg = Config(group="G4", hidden=16, total_train_steps=32, rollout_steps=8,
                epochs=1, minibatch=8, gate_cases_per_family=1, refresh_rounds=1,
                task=task(duration=0.4, goal_distance=0.1, attack_weight=1))
            result = run(cfg, Path(temp) / "run")
            self.assertEqual(result["train_steps"], 32)
            self.assertEqual(Team(cfg).encoders[0][0].in_features, 25)

    def test_plan_has_actual_independent_seeds_and_common_attack_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "study"
            plan(target, steps=32)
            files = list((target / "attack").glob("*.json"))
            self.assertEqual(len(files), 24)
            rows = [json.loads(p.read_text()) for p in files]
            self.assertEqual({r["seed"] for r in rows}, {42, 43, 44})
            self.assertEqual({r["task"]["attack_weight"] for r in rows}, {1})

    def test_gate_controls_preserve_this_update_coverage_and_replay(self):
        from ..training.mutual import distillation_weights
        families = ["a"] * 5 + ["b"] * 11
        args = dict(seed=42, round_id=7, team="A")
        evidence = {"a": 1, "b": 0}
        selected = distillation_weights("selective", families, evidence, **args)
        random = distillation_weights("random", families, evidence, **args)
        strength = distillation_weights("strength", families, evidence, **args)
        self.assertEqual(selected.sum(), random.sum())
        self.assertEqual(selected.mean(), strength.mean())
        np.testing.assert_array_equal(random, distillation_weights("random", families, evidence, **args))

    def test_report_uses_training_seeds_and_keeps_negative_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = Config(task=task(attack_weight=1))
            for group, score in (("G0", 0.4), ("G2", 0.2)):
                for seed in (42, 43, 44):
                    record = {"protocol": PROTOCOL, "split": "test", "intervention": "normal",
                        "group": group, "seed": seed, "training_config": replace(cfg, group=group, seed=seed).to_dict(),
                        "source_sha256": "same", "source_changed": False, "selection": {"eligible": True},
                        "training_segments": [{"budget": 100000, "source_sha256": "same"}],
                        "evaluation_contract": {"task": cfg.to_dict()["task"]},
                        "overall": {"baseline_invalid_count": 0, "case_keys": ["paired:1"],
                            "qualified_attack_utility": score, "success_rate": 1, "bad_rate": 0,
                            "qualified_extra_red_loss_m": score * 180}}
                    (root / f"{group}_{seed}.json").write_text(json.dumps(record))
            rows, contrasts = report_study(root, root / "summary")
            self.assertEqual(rows[0]["independent_seeds"], 3)
            contrast = next(c for c in contrasts if c["treatment"] == "attack/G2")
            self.assertAlmostEqual(contrast["mean_difference"], -0.2)
            self.assertLess(contrast["paired_seed_t95_interval"][1], 0)


class FakeActor:
    def __init__(self, identity):
        self.x, self.y, self.speed = (70.0 if identity < 2 else 50.0), (identity % 2) * 4.0, 10.0
        self.bounding_box = NS(extent=NS(x=2.5))

    def get_transform(self):
        return NS(location=NS(x=self.x, y=self.y), rotation=NS(yaw=0))

    def get_velocity(self):
        return NS(x=self.speed, y=0)

    def apply_control(self, command):
        self.command = command


class FakeWorld:
    def __init__(self):
        self.actors = {}

    def get_settings(self):
        return NS(synchronous_mode=True, fixed_delta_seconds=0.1)

    def tick(self):
        for actor in self.actors.values():
            actor.x += actor.speed * 0.1


class FakeMap:
    def get_waypoint(self, location, project_to_road=False):
        return NS(transform=NS(location=NS(x=location.x, y=round(location.y / 4) * 4), rotation=NS(yaw=0)))


class CarlaTransferContracts(unittest.TestCase):
    def test_route_projection_and_physical_control(self):
        route = PolylineRoute([[100, 200], [100, 300]])
        s, lateral, heading = route.project([98, 220])
        self.assertAlmostEqual(s, 20)
        self.assertAlmostEqual(lateral, 2)
        self.assertAlmostEqual(heading, math.pi / 2)
        command = CarlaAccelerationControl(math.pi / 4).command(scene().vehicles[2], [0.4, -0.2], 0.1)
        self.assertGreater(command["throttle"], 0)
        self.assertEqual(command["brake"], 0)
        self.assertLess(command["steer"], 0)

    def test_same_training_runner_runs_against_carla_contract_without_two_world_owners(self):
        world = FakeWorld()

        def loader(condition, family, seed):
            if world.actors:
                raise RuntimeError("previous trial still owns the CARLA world")
            actors = {i: FakeActor(i) for i in (range(2) if condition == "C0" else range(4))}
            world.actors = actors
            routes = {i: PolylineRoute([[0, a.y], [1000, a.y]]) for i, a in actors.items()}
            adapter = CarlaFrameAdapter(actors, routes, origin=(0, 0), route_heading=0,
                collision_flags=lambda: {i: False for i in actors}, road_map=FakeMap())
            return actors, adapter, lambda: setattr(world, "actors", {})

        cfg = Config(group="G4", hidden=16, total_train_steps=16, rollout_steps=8,
            epochs=1, minibatch=8, gate_cases_per_family=1, backend="carla",
            backend_contract={k: "0" * 64 for k in ("scene_manifest_sha256", "red_controller_sha256", "normal_controller_sha256")},
            task=task(duration=0.4, goal_distance=0.1, attack_weight=1, simulation_frequency=10))
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            result = train_carla(cfg, Path(temp) / "mock_carla", world=world, loader=loader,
                traffic_control=lambda identity, frame: {},
                acceleration_controls={i: CarlaAccelerationControl(math.pi / 4) for i in (2, 3)},
                control_factory=lambda **kwargs: kwargs)
            self.assertEqual(result["train_steps"], 16)
            self.assertFalse(world.actors)


if __name__ == "__main__":
    unittest.main()
