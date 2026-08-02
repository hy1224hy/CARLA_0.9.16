import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.actor_critic import Actor, AgentConditionedCritic
from training.train_no_rllib_mappo_robust_v2 import (
    compute_gae_per_episode,
    compute_ppo_diagnostics,
    compute_raw_gae_per_episode,
    ensure_finite_module_gradients,
    ensure_finite_module_parameters,
    ensure_finite_tensor,
    normalize_advantages_over_rollout,
    validate_initial_policy_ratio,
)


class GaeAndAdvantageTests(unittest.TestCase):
    def test_raw_gae_matches_hand_calculation_and_does_not_cross_boundaries(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([0.5, 0.25, 1.0], dtype=np.float32)
        dones = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        terminated = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        advantages, returns = compute_raw_gae_per_episode(
            rewards=rewards,
            values=values,
            dones=dones,
            terminateds=terminated,
            boundaries=[(0, 2), (2, 3)],
            gamma=0.9,
            gae_lambda=0.8,
            bootstrap_values=[0.0, 4.0],
            context="hand-calculated trajectory",
        )

        expected_advantages = np.array([1.985, 1.75, 5.6], dtype=np.float32)
        expected_returns = np.array([2.485, 2.0, 6.6], dtype=np.float32)
        np.testing.assert_allclose(
            advantages, expected_advantages, rtol=0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            returns, expected_returns, rtol=0.0, atol=1e-6
        )

        changed_advantages, _ = compute_raw_gae_per_episode(
            rewards=np.array([1.0, 2.0, 300.0], dtype=np.float32),
            values=values,
            dones=dones,
            terminateds=terminated,
            boundaries=[(0, 2), (2, 3)],
            gamma=0.9,
            gae_lambda=0.8,
            bootstrap_values=[0.0, -50.0],
            context="boundary isolation trajectory",
        )
        np.testing.assert_allclose(
            changed_advantages[:2], expected_advantages[:2], rtol=0.0, atol=1e-6
        )

    def test_rollout_normalization_is_once_per_agent_and_returns_stay_raw(self):
        rewards = np.array([20.0, 10.0, 2.0, 1.0], dtype=np.float32)
        values = np.zeros(4, dtype=np.float32)
        dones = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        terminated = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        raw_advantages, raw_returns = compute_raw_gae_per_episode(
            rewards=rewards,
            values=values,
            dones=dones,
            terminateds=terminated,
            boundaries=[(0, 2), (2, 4)],
            gamma=0.0,
            gae_lambda=0.0,
            bootstrap_values=[0.0, 0.0],
            context="normalization scope",
        )
        normalized, stats = normalize_advantages_over_rollout(
            raw_advantages, context="agent 0 rollout"
        )

        self.assertEqual(stats["applied"], 1.0)
        self.assertAlmostEqual(float(normalized.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized.std()), 1.0, places=6)
        np.testing.assert_array_equal(raw_returns, raw_advantages + values)

        per_episode_normalized = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        self.assertFalse(np.allclose(normalized, per_episode_normalized))

        # A second agent is normalized from its own samples; changing it cannot
        # alter the already computed statistics or values of agent 0.
        agent0_snapshot = normalized.copy()
        agent1_normalized, _ = normalize_advantages_over_rollout(
            np.array([-100.0, -50.0, -25.0], dtype=np.float32),
            context="agent 1 rollout",
        )
        np.testing.assert_array_equal(normalized, agent0_snapshot)
        self.assertAlmostEqual(float(agent1_normalized.mean()), 0.0, places=6)

    def test_low_variance_rollout_uses_finite_guard(self):
        raw = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        normalized, stats = normalize_advantages_over_rollout(
            raw, context="constant rollout"
        )
        np.testing.assert_array_equal(normalized, raw)
        self.assertEqual(stats["applied"], 0.0)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_legacy_wrapper_preserves_not_yet_migrated_entrypoint_behavior(self):
        advantages, returns = compute_gae_per_episode(
            rewards=np.array([20.0, 10.0, 2.0, 1.0], dtype=np.float32),
            values=np.zeros(4, dtype=np.float32),
            dones=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            terminateds=np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            boundaries=[(0, 2), (2, 4)],
            gamma=0.0,
            gae_lambda=0.0,
            bootstrap_values=[0.0, 0.0],
        )
        np.testing.assert_allclose(
            advantages,
            np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            returns, np.array([20.0, 10.0, 2.0, 1.0], dtype=np.float32)
        )

    def test_strict_boundaries_and_bootstraps_reject_misalignment(self):
        common = dict(
            rewards=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            values=np.zeros(3, dtype=np.float32),
            dones=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            terminateds=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            gamma=0.9,
            gae_lambda=0.95,
        )
        with self.assertRaisesRegex(ValueError, "contiguous start"):
            compute_raw_gae_per_episode(
                **common,
                boundaries=[(0, 1), (2, 3)],
                bootstrap_values=[0.0, 0.0],
                context="gap",
            )
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            compute_raw_gae_per_episode(
                **common,
                boundaries=[(0, 1), (1, 3)],
                bootstrap_values=[0.0],
                context="bootstrap mismatch",
            )
        with self.assertRaisesRegex(ValueError, "must use zero bootstrap"):
            compute_raw_gae_per_episode(
                rewards=np.array([1.0], dtype=np.float32),
                values=np.array([0.0], dtype=np.float32),
                dones=np.array([1.0], dtype=np.float32),
                terminateds=np.array([1.0], dtype=np.float32),
                boundaries=[(0, 1)],
                gamma=0.9,
                gae_lambda=0.95,
                bootstrap_values=[5.0],
                context="terminal bootstrap",
            )


class PpoUpdateHealthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260723)

    def test_preupdate_ratio_is_one_and_detects_stale_policy(self):
        actor = Actor(obs_dim=19, act_dim=3, hidden=32)
        observations = torch.randn(32, 19)
        with torch.no_grad():
            _actions, old_log_probs, action_z = actor.sample_action_z(observations)

        error = validate_initial_policy_ratio(
            actor,
            observations,
            action_z,
            old_log_probs,
            tolerance=1e-6,
            context="unchanged actor",
        )
        self.assertLess(error, 1e-6)

        with torch.no_grad():
            next(actor.parameters()).add_(0.05)
        with self.assertRaisesRegex(RuntimeError, "pre-update PPO ratio mismatch"):
            validate_initial_policy_ratio(
                actor,
                observations,
                action_z,
                old_log_probs,
                tolerance=1e-6,
                context="stale actor",
            )

    def test_preupdate_check_matches_transitionwise_rollout_geometry(self):
        actor = Actor(obs_dim=19, act_dim=3, hidden=128)
        observations = torch.randn(256, 19)
        old_log_prob_parts = []
        action_z_parts = []
        with torch.no_grad():
            for observation in observations:
                _action, log_prob, action_z = actor.sample_action_z(
                    observation.unsqueeze(0)
                )
                old_log_prob_parts.append(log_prob)
                action_z_parts.append(action_z)

        error = validate_initial_policy_ratio(
            actor,
            observations,
            torch.cat(action_z_parts, dim=0),
            torch.cat(old_log_prob_parts, dim=0),
            tolerance=1e-6,
            context="transitionwise rollout geometry",
        )

        self.assertEqual(error, 0.0)

    def test_one_synthetic_actor_critic_update_stays_finite(self):
        n_samples = 32
        actor = Actor(obs_dim=19, act_dim=3, hidden=32)
        critic = AgentConditionedCritic(state_dim=38, num_agents=2, hidden=32)
        actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
        critic_optim = optim.Adam(critic.parameters(), lr=3e-4)

        observations = torch.randn(n_samples, 19)
        global_states = torch.randn(n_samples, 38)
        agent_ids = torch.arange(n_samples, dtype=torch.long) % 2
        with torch.no_grad():
            _actions, old_log_probs, action_z = actor.sample_action_z(observations)
        normalized_np, _ = normalize_advantages_over_rollout(
            np.linspace(-3.0, 4.0, n_samples, dtype=np.float32),
            context="synthetic PPO advantages",
        )
        advantages = torch.as_tensor(normalized_np)
        critic_returns = torch.linspace(-2.0, 5.0, n_samples)

        validate_initial_policy_ratio(
            actor,
            observations,
            action_z,
            old_log_probs,
            tolerance=1e-6,
            context="synthetic pre-update",
        )

        new_log_probs, entropy = actor.evaluate_actions_z(observations, action_z)
        logratio = new_log_probs - old_log_probs
        ratio = torch.exp(logratio)
        approx_kl, clip_fraction = compute_ppo_diagnostics(
            logratio, ratio, 0.15, context="synthetic actor"
        )
        actor_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.85, 1.15) * advantages,
        ).mean() - 0.03 * entropy.mean()
        ensure_finite_tensor(
            actor_loss, name="actor_loss", context="synthetic actor"
        )
        actor_optim.zero_grad()
        actor_loss.backward()
        ensure_finite_module_gradients(actor, context="synthetic actor")
        actor_grad_norm = nn.utils.clip_grad_norm_(
            actor.parameters(), 0.5, error_if_nonfinite=True
        )
        ensure_finite_tensor(
            actor_grad_norm, name="actor_grad_norm", context="synthetic actor"
        )
        actor_optim.step()
        ensure_finite_module_parameters(actor, context="synthetic actor")

        value_predictions = critic(global_states, agent_ids)
        self.assertEqual(tuple(value_predictions.shape), (n_samples,))
        critic_loss = 0.5 * (critic_returns - value_predictions).pow(2).mean()
        ensure_finite_tensor(
            critic_loss, name="critic_loss", context="synthetic critic"
        )
        critic_optim.zero_grad()
        critic_loss.backward()
        ensure_finite_module_gradients(critic, context="synthetic critic")
        critic_grad_norm = nn.utils.clip_grad_norm_(
            critic.parameters(), 0.5, error_if_nonfinite=True
        )
        ensure_finite_tensor(
            critic_grad_norm, name="critic_grad_norm", context="synthetic critic"
        )
        critic_optim.step()
        ensure_finite_module_parameters(critic, context="synthetic critic")

        self.assertTrue(np.isfinite(float(actor_loss.item())))
        self.assertTrue(np.isfinite(float(critic_loss.item())))
        self.assertTrue(np.isfinite(approx_kl))
        self.assertGreaterEqual(clip_fraction, 0.0)
        self.assertLessEqual(clip_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
