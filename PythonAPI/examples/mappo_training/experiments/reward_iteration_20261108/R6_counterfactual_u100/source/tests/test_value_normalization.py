import unittest

import numpy as np
import torch
import torch.optim as optim

from training.train_no_rllib_mappo_robust_v2 import (
    build_active_step_cache,
    run_mappo_ppo_update,
)
from training.value_normalization import ValueNormalizer, normalized_huber_loss
from tests.test_mappo_algorithm import (
    ACTION_DIM,
    AGENT_IDS,
    OBS_DIM,
    STATE_DIM,
    _make_models,
    _prepare,
)


class _ConstantNormalizedCritic(torch.nn.Module):
    def forward(self, states, agent_ids):
        del agent_ids
        return torch.full((states.shape[0],), 2.0, device=states.device)


class _ZeroActor:
    def sample_action_z(self, observations):
        batch = observations.shape[0]
        zeros = torch.zeros((batch, 2), device=observations.device)
        return zeros, torch.zeros(batch, device=observations.device), zeros


class ValueNormalizerTests(unittest.TestCase):
    def test_first_update_recovers_exact_batch_moments_and_round_trip(self):
        normalizer = ValueNormalizer(beta=0.99999, variance_floor=1e-2)
        targets = torch.tensor([10.0, 14.0])
        normalizer.update(targets)
        mean, variance = normalizer.mean_variance()
        self.assertAlmostEqual(float(mean), 12.0, places=5)
        self.assertAlmostEqual(float(variance), 4.0, places=4)
        normalized = normalizer.normalize(targets)
        self.assertTrue(
            torch.allclose(normalized, torch.tensor([-1.0, 1.0]), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(normalizer.denormalize(normalized), targets, atol=1e-5)
        )

    def test_checkpoint_state_restores_exact_value_decoding(self):
        source = ValueNormalizer()
        source.update(torch.tensor([-20.0, 0.0, 10.0]))
        restored = ValueNormalizer()
        restored.load_state_dict(source.state_dict(), strict=True)
        predictions = torch.tensor([-0.5, 0.0, 1.5])
        self.assertTrue(
            torch.equal(
                source.denormalize(predictions),
                restored.denormalize(predictions),
            )
        )
        self.assertEqual(source.statistics(), restored.statistics())

    def test_huber_caps_outlier_gradient_in_normalized_units(self):
        predictions = torch.tensor([0.0, 0.0], requires_grad=True)
        targets = torch.tensor([1.0, 100.0])
        loss = normalized_huber_loss(predictions, targets, delta=10.0)
        loss.backward()
        self.assertTrue(torch.allclose(predictions.grad, torch.tensor([-0.5, -5.0])))
        self.assertAlmostEqual(float(loss.detach()), 475.25, places=5)

    def test_rollout_cache_stores_denormalized_reward_unit_value(self):
        normalizer = ValueNormalizer()
        normalizer.update(torch.tensor([10.0, 14.0]))
        _, cache = build_active_step_cache(
            obs_dict={0: np.zeros(19, dtype=np.float32)},
            active_mask={0: True},
            global_state_np=np.zeros(19, dtype=np.float32),
            actors={0: _ZeroActor()},
            critic=_ConstantNormalizedCritic(),
            agent_ids=[0],
            device=torch.device("cpu"),
            context="ValueNorm rollout unit test",
            value_normalizer=normalizer,
        )
        self.assertAlmostEqual(cache[0]["value"], 16.0, places=4)

    def test_production_update_normalizes_once_and_reports_both_scales(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [3], 1: [3]})
        normalizer = ValueNormalizer()
        result = run_mappo_ppo_update(
            prepared=prepared,
            actors=actors,
            actor_optims={
                aid: optim.Adam(actors[aid].parameters(), lr=1e-3)
                for aid in AGENT_IDS
            },
            critic=critic,
            critic_optim=optim.Adam(critic.parameters(), lr=1e-3),
            agent_ids=AGENT_IDS,
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            ppo_epochs=2,
            minibatch_size=2,
            clip_eps=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.0,
            value_normalizer=normalizer,
            huber_delta=10.0,
            shuffle=False,
            context="ValueNorm production update test",
        )
        self.assertEqual(int(normalizer.update_count), 1)
        self.assertAlmostEqual(
            result.metrics["normalized_return_target_std"], 1.0, places=5
        )
        self.assertIn("return_target_std", result.metrics)
        self.assertIn("value_prediction_std", result.metrics)
        self.assertIn("value_normalizer_mean", result.metrics)

    def test_rejects_nonfinite_or_empty_updates(self):
        normalizer = ValueNormalizer()
        with self.assertRaisesRegex(ValueError, "empty"):
            normalizer.update(torch.empty(0))
        with self.assertRaisesRegex(FloatingPointError, "NaN/Inf"):
            normalizer.update(torch.tensor([float("nan")]))
        normalizer.running_mean.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "running_mean"):
            normalizer.validate_state()


if __name__ == "__main__":
    unittest.main()
