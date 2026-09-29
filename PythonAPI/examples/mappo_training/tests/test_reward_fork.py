"""Offline fork/state/evaluation checks against the retained u356 artifact."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from training.checkpoint_protocol import load_checkpoint_file, validate_checkpoint_header, CheckpointProtocolError
from training.reward_fork import SOURCE, SOURCE_SHA, prepare, resume_args, fork_payload
from utils.training_metrics_recorder import TrainingMetricsRecorder
from evaluation.evaluate_reward_fork import checkpoint_environment, summarize_trace


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


@unittest.skipUnless(SOURCE.is_file(), "retained u356 artifact unavailable")
class RewardForkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.digest = load_checkpoint_file(SOURCE, map_location="cpu")

    def test_full_state_preserved_and_old_metrics_excluded(self):
        result = fork_payload(self.source, source_path=SOURCE, digest=self.digest,
            run_id="new_reward_test", profile="route_pace_v1", fingerprint="offline", git_sha="offline")
        for key in ("actors", "actor_optims", "critic", "critic_optim", "value_normalizer",
                    "rng_state", "lr_schedule", "curriculum_state", "update", "total_episodes"):
            assert_equal_state(self, self.source[key], result[key])
        self.assertIsNone(result["last_metrics"])
        self.assertNotIn("continuation", result["run_metadata"])
        self.assertEqual(result["curriculum_state"]["mastery_state"]["stage_index"], 3)
        self.assertEqual(checkpoint_environment(result)["pace_profile"], "route_pace_v1")
        self.assertEqual(checkpoint_environment(self.source)["pace_profile"], "frozen_v6")
        invalid = copy.deepcopy(result)
        invalid["update"] += 1
        with self.assertRaises(CheckpointProtocolError):
            validate_checkpoint_header(invalid)

    def test_prepared_fork_passes_actual_trainer_validation_before_carla(self):
        from training.train_no_rllib_mappo_robust_v2 import train
        class ReachedCarlaBoundary(Exception):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "fork"
            prepare(destination)
            args, checkpoint = resume_args(destination)
            self.assertEqual(args.lr_decay_horizon, 600)
            self.assertEqual(args.updates, 406)
            self.assertTrue(args.follow_spectator)
            recorder = TrainingMetricsRecorder(run_dir=destination, run_id="fork",
                agent_ids=[0,1], reward_version="v6", start_update=357, completed_update=356,
                pace_profile="route_pace_v1", origin_update=356)
            try:
                self.assertEqual(recorder._last_committed_update, 356)
                self.assertEqual(recorder._metrics_payloads_by_update, {})
            finally:
                recorder.close()
            with patch("training.train_no_rllib_mappo_robust_v2.CarlaMultiAgentEnv",
                       side_effect=ReachedCarlaBoundary) as boundary:
                with self.assertRaises(ReachedCarlaBoundary):
                    train(args)
                boundary.assert_called_once()
                self.assertEqual(boundary.call_args.kwargs["pace_profile"], "route_pace_v1")


class TraceSummaryTests(unittest.TestCase):
    def test_other_car_completion_at_first_arrival(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            rows = [{"type": "step", "step": step, "info": {"reward_diagnostics": {
                "0": {"success_event": int(step == 10), "route_completion": 1},
                "1": {"success_event": int(step == 30), "route_completion": progress},
            }}} for step,progress in [(10,0.2),(30,1)]]
            trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = summarize_trace(trace, 0.1)
            self.assertEqual(result["arrival_gap_seconds"], 2.0)
            self.assertEqual(result["completion_at_first_arrival"]["1"], 0.2)


if __name__ == "__main__":
    unittest.main()
