"""Scientific contracts that could invalidate an algorithm comparison."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from ..config import Config, Task, ROOT
from ..models.team import Team
from ..environment.highway import HighwayTeamEnv
from ..environment.scenarios import cases
from ..training.ppo import gae
from ..training.ppo import collect, update
from ..training.mutual import snapshots
from ..training.mutual import compute_gates
from ..training.runner import run
from ..training.checkpoint import read


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seeded_environment_and_reference(self):
        a, b = HighwayTeamEnv(Task()), HighwayTeamEnv(Task())
        try:
            oa, ia = a.reset(seed=123)
            ob, ib = b.reset(seed=123)
            np.testing.assert_array_equal(oa, ob)
            rng = np.random.default_rng(77)
            for _ in range(8):
                action = rng.uniform(-0.1, 0.1, (2, 2))
                ra, rb = a.step(action), b.step(action)
                np.testing.assert_array_equal(ra[0], rb[0])
                np.testing.assert_array_equal(ra[1], rb[1])
            self.assertEqual(ia["state"].shape, (28,))
            self.assertTrue(np.isfinite(ra[1]).all())
            self.assertTrue(all(v.target_speed == 18 for v in a.road.vehicles[:2]))
        finally:
            a.close()
            b.close()

    def test_hidden_neighbor_not_in_local_observation(self):
        env = HighwayTeamEnv(Task(sensing_range=1))
        obs, _ = env.reset(seed=3)
        np.testing.assert_array_equal(obs[:, 6:], np.zeros((2, 18)))
        env.close()

    def test_actor_isolation_probability_and_message_gradients(self):
        obs = torch.randn(7, 2, 24)
        links = torch.ones(7, 2, 1)
        for group in ("G0", "G3"):
            model = Team(Config(group=group, hidden=16))
            action, z, old = model.sample(obs, links)
            dist = model.distribution(obs, links)
            torch.testing.assert_close(old, model.log_prob(dist, z))
            self.assertTrue((action.abs() <= 1).all())
            changed = obs.clone()
            changed[:, 1] += 2
            if group == "G0":
                torch.testing.assert_close(dist.loc[:, 0], model.distribution(changed, links).loc[:, 0])
            dist.loc.sum().backward()
            grad = sum(float(p.grad.abs().sum()) for p in model.encoders.parameters() if p.grad is not None)
            self.assertGreater(grad, 0) if group == "G3" else self.assertEqual(grad, 0)
            zero = model.distribution(obs, links, intervention="zero")
            disconnected = model.distribution(obs, links * 0)
            torch.testing.assert_close(zero.loc, disconnected.loc)

    def test_truncation_bootstrap_without_cross_episode_gae(self):
        reward = torch.tensor([[1.], [10.]])
        value = torch.zeros_like(reward)
        next_value = torch.tensor([[2.], [0.]])
        terminal = torch.tensor([[0.], [1.]])
        ended = torch.ones_like(terminal)
        adv, _ = gae(reward, value, next_value, terminal, ended, 0.9, 0.95)
        torch.testing.assert_close(adv, torch.tensor([[2.8], [10.]]))

    def test_split_and_gate_pairing(self):
        splits = [set(cases(s, 8)) for s in ("gate", "selection", "test")]
        self.assertFalse(splits[0] & splits[1] or splits[1] & splits[2])
        def record(q):
            return {"families": {"parallel": {"case_keys": list(range(8)), "episodes": 8,
                                               "quality": q, "bad_rate": 0}}}
        reports = {"A": record(0.1), "B": record(0.4)}
        gates = compute_gates(reports, Config())
        self.assertEqual(gates, {"A": {"parallel": 1.0}, "B": {"parallel": 0.0}})
        reports["B"]["families"]["parallel"]["case_keys"] = list(range(1, 9))
        with self.assertRaises(ValueError):
            compute_gates(reports, Config())

    def test_mutual_update_uses_frozen_teacher(self):
        config = Config(group="G4", hidden=16, epochs=1, minibatch=8, task=Task(duration=2))
        student = Team(config)
        teacher = snapshots({"peer": student})["peer"]
        with torch.no_grad():
            teacher.actors[0][-1].bias.add_(0.4)
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        env = HighwayTeamEnv(config.task)
        batch, _, _, _ = collect(student, env, 8, np.random.default_rng(2), None)
        result = update(student, torch.optim.Adam(student.parameters(), lr=config.lr), batch,
                        config, teacher, torch.ones(8))
        self.assertGreater(result["mutual_kl"], 0)
        self.assertEqual(result["gate_fraction"], 1)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        env.close()

    def test_braking_cannot_reverse_vehicle(self):
        env = HighwayTeamEnv(Task(duration=20))
        env.reset(seed=2, options={"family": "parallel"})
        for _ in range(100):
            _, _, term, trunc, _ = env.step(np.array([[-1, 0], [-1, 0]]))
            self.assertTrue(all(v.speed >= 0 for v in env.road.vehicles))
            if term or trunc:
                break
        env.close()

    def test_exact_budget_and_resume(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        cfg = Config(group="G4", hidden=16, total_train_steps=32, rollout_steps=8,
                     epochs=1, minibatch=8, refresh_rounds=1, warmup_rounds=0,
                     gate_cases_per_family=1, task=Task(duration=0.4))
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            full, resumed = Path(temp) / "full", Path(temp) / "resumed"
            result = run(cfg, full)
            run(cfg, resumed, max_rounds=1)
            with (resumed / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"round":999,"incomplete":')
            recovered = run(cfg, resumed, resume=resumed / "latest.pt")
            self.assertEqual(result["train_steps"], cfg.total_train_steps)
            self.assertEqual(result, recovered)
            x, y = read(full / "latest.pt"), read(resumed / "latest.pt")
            for name in ("A", "B"):
                for key in x["models"][name]:
                    torch.testing.assert_close(x["models"][name][key], y["models"][name][key], rtol=0, atol=0)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(str(resumed / "tensorboard")).Reload()
            self.assertEqual([event.step for event in events.Scalars("train/A/policy_loss")], [16, 32])
            self.assertIn("validation/selection/A/success_rate", events.Tags()["scalars"])

    def test_config_rejects_invalid_server_inputs(self):
        for config in (Config(seed=-1), Config(threads=True), Config(warmup_rounds=0.5),
                       Config(task=Task(policy_frequency=0)), Config(task=Task(duration=0.01)),
                       Config(task=Task(attack_weight=float("nan")))):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            cases("test", 100001)

    def test_resume_rejects_runtime_drift(self):
        from ..training.checkpoint import runtime_versions, validate_runtime
        versions = runtime_versions()
        validate_runtime(versions)
        versions["numpy"] = "changed"
        with self.assertRaisesRegex(ValueError, "versions changed"):
            validate_runtime(versions)

    def test_extend_preserves_full_state_and_parent(self):
        import hashlib
        import json
        from dataclasses import replace
        from ..training.continuation import extend
        cfg = Config(group="G4", hidden=16, total_train_steps=16, rollout_steps=8,
                     epochs=1, minibatch=8, refresh_rounds=1, warmup_rounds=0,
                     gate_cases_per_family=1, task=Task(duration=0.4))
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            parent, child, full = (Path(temp) / name for name in ("parent", "child", "full"))
            run(cfg, parent)
            original = hashlib.sha256((parent / "latest.pt").read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "larger TOTAL"):
                extend(parent / "latest.pt", child, 16)
            self.assertFalse(child.exists())
            result = extend(parent / "latest.pt", child, 32)
            run(replace(cfg, total_train_steps=32), full)
            self.assertEqual(result["train_steps"], 32)
            self.assertEqual(original, hashlib.sha256((parent / "latest.pt").read_bytes()).hexdigest())
            x, y = read(child / "latest.pt"), read(full / "latest.pt")
            for team in ("A", "B"):
                for key in x["models"][team]:
                    torch.testing.assert_close(x["models"][team][key], y["models"][team][key], rtol=0, atol=0)
                torch.testing.assert_close(x["torch_rng"][team], y["torch_rng"][team], rtol=0, atol=0)
                for parameter, state in x["optimizers"][team]["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(value, y["optimizers"][team]["state"][parameter][key], rtol=0, atol=0)
            manifest = json.loads((child / "manifest.json").read_text())
            self.assertFalse(manifest["continuation"]["optimizer_reset"])
            self.assertEqual([s["budget"] for s in manifest["training_segments"]], [16, 32])
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events = EventAccumulator(str(child / "tensorboard")).Reload()
            self.assertEqual([r.step for r in events.Scalars("train/A/policy_loss")], [16, 32])

    def test_run_lock_excludes_second_owner(self):
        from ..training.locking import run_lock
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            output = Path(temp) / "run"
            with run_lock(output):
                with self.assertRaises(RuntimeError):
                    with run_lock(output):
                        pass
            with run_lock(output):
                pass

    def test_critic_state_contains_goal_progress(self):
        env = HighwayTeamEnv(Task())
        env.reset(seed=1)
        before = env._info()["state"].copy()
        env.initial_x[2] -= 15
        after = env._info()["state"]
        self.assertNotEqual(float(before[20]), float(after[20]))
        env.close()

    def test_episode_returns_are_recorded(self):
        config = Config(hidden=16, task=Task(duration=0.2))
        env = HighwayTeamEnv(config.task)
        batch, _, episodes, _ = collect(Team(config), env, 1, np.random.default_rng(2), None)
        self.assertEqual(len(episodes), 1)
        self.assertAlmostEqual(episodes[0]["return"], float(batch["reward"].mean()), places=6)
        env.close()

    def test_live_preview_preserves_training_rollout(self):
        import json
        from ..visualization.live import LivePreview
        config = Config(hidden=16, task=Task(duration=0.4))
        model = Team(config)
        plain_env, preview_env = HighwayTeamEnv(config.task), HighwayTeamEnv(config.task)
        state = torch.get_rng_state()
        plain, _, _, _ = collect(model, plain_env, 8, np.random.default_rng(7), None)
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temp:
            preview = LivePreview(temp)
            preview.context("A", 1, 0)
            torch.set_rng_state(state)
            observed, _, _, _ = collect(model, preview_env, 8, np.random.default_rng(7), None, preview=preview)
            for key in plain:
                torch.testing.assert_close(plain[key], observed[key], rtol=0, atol=0)
            from PIL import Image
            with Image.open(Path(temp) / "preview" / "frame.png") as frame:
                self.assertEqual(frame.size, (900, 220))
            status = json.loads((Path(temp) / "preview" / "status.json").read_text())
            self.assertEqual(status["train_steps"], 8)
            self.assertEqual(status["team"], "A")
        plain_env.close()
        preview_env.close()

    def test_matrix_rejects_unequal_budgets(self):
        import json
        from ..experiments import preflight
        with tempfile.TemporaryDirectory() as temp:
            for group, steps in (("G0", 64), ("G1", 128)):
                (Path(temp) / f"{group}.json").write_text(json.dumps(Config(group=group, total_train_steps=steps).to_dict()))
            with self.assertRaisesRegex(ValueError, "matrix mixes"):
                preflight(temp)

    def test_report_rejects_unmatched_training_contract(self):
        import json
        from ..experiments import comparison_contract
        from ..evaluation.report import report
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for group, steps in (("G0", 64), ("G1", 128)):
                data = {"split": "test", "intervention": "normal", "group": group, "seed": 42,
                        "experiment_contract": comparison_contract(Config(group=group, total_train_steps=steps)),
                        "source_sha256": "same", "overall": {"case_keys": ["parallel:3000000"]}}
                (root / f"{group}.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "different budgets"):
                report(root, root / "summary")

    def test_report_ignores_batch_status_and_previous_summary(self):
        import json
        from ..evaluation.report import report
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "batch_status.json").write_text(json.dumps([{"name": "G0_s42", "status": "complete"}]))
            output = root / "comparison"
            output.mkdir()
            (output / "comparison.json").write_text("[]")
            report(root, output)
            self.assertEqual(json.loads((output / "comparison.json").read_text()), [])


if __name__ == "__main__":
    unittest.main()
