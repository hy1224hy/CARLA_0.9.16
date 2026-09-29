import tempfile
import unittest

import torch

from models.actor_critic import Actor
from training.train_no_rllib_mappo_robust_v2 import (
    apply_actor_initialization_mode,
    build_parser,
    checkpoint_training_config,
)
from utils.training_metrics_recorder import TrainingMetricsRecorder


def _actors():
    return {
        agent_id: Actor(
            obs_dim=20,
            act_dim=2,
            hidden=16,
            init_log_std=-0.5,
            min_log_std=-2.0,
            max_log_std=0.0,
        )
        for agent_id in (0, 1)
    }


class ActorInitializationAblationTests(unittest.TestCase):
    def test_identical_mode_copies_weights_without_sharing_parameters(self):
        torch.manual_seed(42)
        actors = _actors()
        report = apply_actor_initialization_mode(actors, "identical")

        self.assertTrue(report["all_exactly_equal"])
        self.assertEqual(report["maximum_abs_parameter_difference"], 0.0)
        self.assertFalse(report["parameters_shared"])
        for first, second in zip(actors[0].parameters(), actors[1].parameters()):
            self.assertTrue(torch.equal(first, second))
            self.assertIsNot(first, second)
            self.assertNotEqual(first.data_ptr(), second.data_ptr())

        optimizers = {
            agent_id: torch.optim.Adam(actor.parameters(), lr=3e-4)
            for agent_id, actor in actors.items()
        }
        self.assertIsNot(optimizers[0], optimizers[1])

    def test_independent_mode_retains_random_initialization(self):
        torch.manual_seed(42)
        actors = _actors()
        report = apply_actor_initialization_mode(actors, "independent")
        self.assertFalse(report["all_exactly_equal"])
        self.assertGreater(report["maximum_abs_parameter_difference"], 0.0)

    def test_actor_init_mode_is_checkpoint_immutable(self):
        args = build_parser().parse_args(["--actor-init-mode", "identical"])
        self.assertEqual(
            checkpoint_training_config(args)["actor_init_mode"], "identical"
        )

    def test_pairwise_metrics_show_divergence_after_separate_update(self):
        torch.manual_seed(42)
        actors = _actors()
        apply_actor_initialization_mode(actors, "identical")
        critic = torch.nn.Linear(4, 1)
        with tempfile.TemporaryDirectory() as run_dir:
            recorder = TrainingMetricsRecorder(
                run_dir=run_dir,
                run_id="identical_init_test",
                agent_ids=[0, 1],
                reward_version="v2",
                start_update=1,
                completed_update=0,
                histogram_interval=0,
            )
            recorder.capture_model_snapshot(actors, critic)
            recorder.record_actor_pairwise_diagnostics(
                actors, phase="pre_update"
            )
            with torch.no_grad():
                next(actors[0].parameters()).add_(0.01)
            recorder.record_model_update(actors, critic)
            recorder.record_actor_pairwise_diagnostics(
                actors, phase="post_update"
            )
            scalars = recorder.external_scalars
            prefix = "model/actor_pair_0_1"
            self.assertEqual(
                scalars[f"{prefix}/pre_update/parameter_distance"], 0.0
            )
            self.assertGreater(
                scalars[f"{prefix}/post_update/parameter_distance"], 0.0
            )
            self.assertLess(
                scalars[
                    f"{prefix}/post_update/parameter_cosine_similarity"
                ],
                1.0,
            )
            recorder.close()


if __name__ == "__main__":
    unittest.main()
