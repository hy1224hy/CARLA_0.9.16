"""Event projection/recovery checks; no simulator or private checkpoints."""
import copy
import builtins
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.mut_mappo_tensorboard import MutMAPPOTensorBoard, committed_records, scalar_projection
from tests import test_mut_mappo_runner as runner_fixtures
from training.train_mut_mappo import train, critical_learning_signal
from training.mut_mappo_checkpoint import CheckpointStore


def fixture(round_no=1, ticks=123):
    return {"completed_round": round_no, "actual_counters": {"train": ticks},
            "teams": {team: {"actor_loss_2": [1., 3.], "mutual_gated_kl_2": 0.,
                               "advantage_stats": {2: {"raw_std": 4.}},
                               "learning_rates": {2: {"actor": [.0003]}}} for team in ("A", "B")},
            "training_window": {team: {"b": .25, "critical_min_ttc_s": None,
                "reward_components_mean": {"critical": .5}} for team in ("A", "B")}}


def events(path):
    return EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()


class TensorBoardProjectionTests(unittest.TestCase):
    def test_event_values_match_committed_json_and_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "checkpoint_manifest.json").write_text(json.dumps({"completed_round": 2}))
            records = [fixture(), fixture(2, 250), fixture(3, 375)]
            (run / "metrics.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records) + '{"partial":', encoding="utf-8")
            for _ in range(2):
                writer = MutMAPPOTensorBoard(run)
                writer.replay()
                writer.close()
            for team in ("A", "B"):
                accumulator = events(run / "tb" / team)
                values = accumulator.Scalars("ppo/actor_loss_2/mean")
                self.assertEqual([(v.step, v.value) for v in values], [(123, 2.), (250, 2.)])
                self.assertEqual(accumulator.Scalars("ppo/advantage_stats/2/raw_std")[-1].value, 4.)
                self.assertNotIn("training_window/critical_min_ttc_s", accumulator.Tags()["scalars"])
            self.assertEqual(len(list(committed_records(run))), 2)

    def test_disabled_writer_has_no_events_and_no_random_state_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            random.seed(10); np.random.seed(10); torch.manual_seed(10)
            before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
            writer = MutMAPPOTensorBoard(directory, enabled=False)
            writer.record(fixture()); writer.close()
            self.assertFalse((Path(directory) / "tb").exists())
            self.assertEqual(random.getstate(), before[0])
            np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
            torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)

    def test_writer_error_does_not_abort_training_or_mask_close(self):
        class Broken:
            def __init__(self, **_kwargs): pass
            def add_scalar(self, *_args): raise OSError("disk full")
            def close(self): raise OSError("close failed")
        with tempfile.TemporaryDirectory() as directory:
            writer = MutMAPPOTensorBoard(directory, writer_factory=Broken)
            with self.assertWarns(RuntimeWarning):
                writer.record(fixture())
            self.assertFalse(writer.enabled)
            self.assertTrue(writer.closed)
            writer.close()

    def test_output_is_bound_to_source_even_when_run_names_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MutMAPPOTensorBoard(root/"one/B0_s42", log_dir=root/"events")
            first.record(fixture()); first.close()
            with self.assertRaisesRegex(ValueError, "another run/identity"):
                MutMAPPOTensorBoard(root/"two/B0_s42", log_dir=root/"events")
            self.assertEqual(len(events(root/"events/A").Scalars("meta/completed_round")), 1)

    def test_nonfinite_values_are_not_zero_imputed(self):
        self.assertEqual(dict(scalar_projection({"null": None, "nan": float("nan"), "zero": 0})), {"zero": 0.})

    def test_reward_normalization_and_conditional_event_failure_have_explicit_denominators(self):
        records = [dict(steps=2, team_reward_components={"critical": 1., "primary_event_bonus": 5.},
                        team_utility_total=6., team_discounted_return=5., blue_eligible=False,
                        critical_min_ttc_s=1., safety={"collision": True}),
                   dict(steps=8, team_reward_components={"critical": 0., "primary_event_bonus": 0.},
                        team_utility_total=0., team_discounted_return=0., blue_eligible=True,
                        critical_min_ttc_s=None, safety={})]
        values = critical_learning_signal(records)
        self.assertEqual(values["reward_components_per_decision_step"]["critical"], .1)
        self.assertEqual(values["primary_bonus_finally_ineligible_rate"], 1.)
        self.assertEqual(values["primary_bonus_positive_episode_rate"], .5)
        self.assertEqual(values["critical_min_ttc_s_mean_observed"], 1.)
        self.assertEqual(values["critical_ttc_observed_episodes"], 1)
        self.assertEqual(values["blue_collision_episode_rate"], .5)


class TensorBoardTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the small synthetic setup, not the test class' test methods.
        runner_fixtures.RunnerIntegrationTests.setUpClass.__func__(cls)

    factory = staticmethod(runner_fixtures.RunnerIntegrationTests.factory)

    def test_missing_dependency_fails_before_run_creation_or_simulator(self):
        original = builtins.__import__
        def missing(name, *args, **kwargs):
            if name == "torch.utils.tensorboard":
                raise ImportError("fixture missing dependency")
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            run, factory = Path(directory)/"run", Mock()
            with patch("builtins.__import__", side_effect=missing):
                with self.assertRaisesRegex(RuntimeError, "no-tensorboard"):
                    train(copy.deepcopy(self.config), run, stop_round=1, env_factory=factory)
            self.assertFalse(run.exists())
            factory.assert_not_called()

    def assert_state_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_state_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for one, two in zip(first, second):
                self.assert_state_equal(one, two)
        else:
            self.assertEqual(first, second)

    def test_on_off_training_is_identical_and_ippo_keeps_role_statistics(self):
        for algorithm in ("mappo", "ippo"):
            config = copy.deepcopy(self.config)
            config.update(algorithm=algorithm, team_count=1)
            with tempfile.TemporaryDirectory() as directory:
                enabled = train(config, Path(directory)/"on", stop_round=1, env_factory=self.factory)
                disabled = train(config, Path(directory)/"off", stop_round=1, env_factory=self.factory,
                                 tensorboard_options={"enabled": False})
                self.assert_state_equal(enabled["teams"]["A"], disabled["teams"]["A"])
                metrics = enabled["last_metrics"]["teams"]["A"]
                self.assertEqual(set(metrics["advantage_stats"]), {2, 3})
                self.assertEqual(set(metrics["learning_rates"]), {2, 3})
                self.assertTrue((Path(directory)/"on/tb/A").exists())
                self.assertFalse((Path(directory)/"off/tb").exists())

    def test_failed_save_publishes_no_uncommitted_boundary_and_closes_writer(self):
        config = copy.deepcopy(self.config)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/"run"
            original = CheckpointStore.save
            def fail_boundary(store, payload, metrics=None):
                if metrics is not None:
                    raise OSError("fixture failed checkpoint save")
                return original(store, payload, metrics)
            with patch.object(CheckpointStore, "save", fail_boundary):
                with self.assertRaisesRegex(OSError, "failed checkpoint"):
                    train(config, run, stop_round=1, env_factory=self.factory)
            self.assertFalse(list(run.rglob("events.out.tfevents.*")))

    def test_cleanup_error_still_closes_tensorboard(self):
        config = copy.deepcopy(self.config)
        closed = []
        original = MutMAPPOTensorBoard.close
        def close(writer):
            closed.append(True)
            return original(writer)
        def factory(cfg, audit_callback):
            runtime = self.factory(cfg, audit_callback)
            runtime.close = lambda: (_ for _ in ()).throw(RuntimeError("fixture cleanup"))
            return runtime
        with tempfile.TemporaryDirectory() as directory, patch.object(MutMAPPOTensorBoard, "close", close):
            with self.assertRaisesRegex(RuntimeError, "fixture cleanup"):
                train(config, Path(directory)/"run", stop_round=1, env_factory=factory)
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
