import unittest

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from models.actor_critic import Actor, AgentConditionedCritic
from training.train_no_rllib_mappo_robust_v2 import (
    RolloutBuffer,
    Transition,
    entropy_coefficient_for_update,
    prepare_mappo_update_batch,
    run_mappo_ppo_update,
)


class SquashedGaussianR6Tests(unittest.TestCase):
    def test_stable_change_of_variables_matches_torch_transform(self):
        mu = torch.tensor([[0.2, -0.4], [1.2, -1.5]], dtype=torch.float64)
        std = torch.tensor([[0.7, 1.1], [0.5, 0.9]], dtype=torch.float64)
        z = torch.tensor([[0.3, -0.8], [4.0, -5.0]], dtype=torch.float64)
        dist = Normal(mu, std)
        actual = Actor._squashed_log_prob(dist, z)
        transformed = TransformedDistribution(dist, TanhTransform(cache_size=1))
        expected = transformed.log_prob(torch.tanh(z)).sum(dim=-1)
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)

    def test_post_tanh_entropy_detects_mean_saturation(self):
        torch.manual_seed(17)
        actor = Actor(
            obs_dim=3,
            act_dim=2,
            hidden=8,
            init_log_std=-0.5,
            min_log_std=-2.0,
            max_log_std=0.0,
        )
        with torch.no_grad():
            for parameter in actor.net.parameters():
                parameter.zero_()
            actor.mu_head.weight.zero_()
            actor.mu_head.bias.zero_()
        obs = torch.zeros(4096, 3)
        centered_post, centered_pre = actor.estimate_post_tanh_entropy(obs)
        with torch.no_grad():
            actor.mu_head.bias.fill_(4.0)
        saturated_post, saturated_pre = actor.estimate_post_tanh_entropy(obs)
        self.assertLess(
            saturated_post.mean().item(), centered_post.mean().item() - 5.0
        )
        torch.testing.assert_close(
            saturated_pre.mean(), centered_pre.mean(), rtol=1e-6, atol=1e-6
        )

    def test_entropy_estimator_has_finite_reparameterized_gradients(self):
        torch.manual_seed(3)
        actor = Actor(
            obs_dim=4,
            act_dim=2,
            hidden=8,
            init_log_std=-0.4,
            min_log_std=-2.0,
            max_log_std=0.0,
        )
        entropy, _ = actor.estimate_post_tanh_entropy(torch.randn(32, 4))
        (-entropy.mean()).backward()
        for parameter in actor.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_log_std_projection_enforces_both_bounds(self):
        actor = Actor(
            obs_dim=3,
            act_dim=2,
            init_log_std=-0.5,
            min_log_std=-2.0,
            max_log_std=0.0,
        )
        with torch.no_grad():
            actor.log_std.copy_(torch.tensor([-9.0, 4.0]))
        torch.testing.assert_close(
            actor.effective_log_std(), torch.tensor([-2.0, 0.0])
        )
        actor.project_log_std_()
        torch.testing.assert_close(actor.log_std, torch.tensor([-2.0, 0.0]))

    def test_entropy_coefficient_linear_schedule_is_endpoint_exact(self):
        self.assertAlmostEqual(
            entropy_coefficient_for_update(
                1, start=0.01, final=0.001, anneal_updates=160
            ),
            0.01,
        )
        self.assertAlmostEqual(
            entropy_coefficient_for_update(
                160, start=0.01, final=0.001, anneal_updates=160
            ),
            0.001,
        )
        self.assertAlmostEqual(
            entropy_coefficient_for_update(
                999, start=0.01, final=0.001, anneal_updates=160
            ),
            0.001,
        )

    def test_entropy_sample_count_requires_an_integer(self):
        actor = Actor(obs_dim=3, act_dim=2, min_log_std=-2.0, max_log_std=0.0)
        with self.assertRaises(ValueError):
            actor.estimate_post_tanh_entropy(torch.zeros(2, 3), sample_count=1.5)

    def test_twenty_dimensional_r6_batch_runs_real_ppo_entropy_path(self):
        torch.manual_seed(29)
        np.random.seed(29)
        agent_ids = [0, 1]
        actors = {
            agent_id: Actor(
                obs_dim=20,
                act_dim=2,
                hidden=16,
                init_log_std=-0.5,
                min_log_std=-2.0,
                max_log_std=0.0,
            )
            for agent_id in agent_ids
        }
        critic = AgentConditionedCritic(40, 2, hidden=16)
        buffer = RolloutBuffer(agent_ids)
        for agent_id in agent_ids:
            for step in range(3):
                obs = np.linspace(-0.2, 0.2, 20, dtype=np.float32)
                obs[0] = float(agent_id + step)
                state = np.linspace(-0.4, 0.4, 40, dtype=np.float32)
                state[0] = float(agent_id + step)
                with torch.no_grad():
                    _, log_prob, action_z = actors[agent_id].sample_action_z(
                        torch.as_tensor(obs).unsqueeze(0)
                    )
                    value = critic(torch.as_tensor(state).unsqueeze(0), agent_id)
                terminal = step == 2
                buffer.add(
                    agent_id,
                    Transition(
                        obs=obs,
                        global_state=state,
                        action_z=action_z.squeeze(0).numpy(),
                        log_prob=float(log_prob.item()),
                        reward=1.0,
                        done=float(terminal),
                        terminated=float(terminal),
                        value=float(value.item()),
                        scenario_name="r6_synthetic",
                    ),
                )
            buffer.mark_episode_end(agent_id)
            buffer.add_episode_bootstrap(agent_id, None)
        prepared = prepare_mappo_update_batch(
            buffer=buffer,
            critic=critic,
            agent_ids=agent_ids,
            device=torch.device("cpu"),
            gamma=0.999,
            gae_lambda=0.95,
            obs_dim=20,
            state_dim=40,
            action_dim=2,
            context="R6 20D batch",
        )
        result = run_mappo_ppo_update(
            prepared=prepared,
            actors=actors,
            actor_optims={
                agent_id: optim.Adam(actors[agent_id].parameters(), lr=1e-3)
                for agent_id in agent_ids
            },
            critic=critic,
            critic_optim=optim.Adam(critic.parameters(), lr=1e-3),
            agent_ids=agent_ids,
            obs_dim=20,
            state_dim=40,
            action_dim=2,
            ppo_epochs=1,
            minibatch_size=32,
            clip_eps=0.15,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.0,
            shuffle=False,
            context="R6 20D PPO",
        )
        for agent_id in agent_ids:
            self.assertLess(result.preupdate_ratio_errors[agent_id], 1e-6)
            self.assertTrue(
                np.isfinite(result.metrics[f"post_tanh_entropy_{agent_id}"])
                .all()
            )
            self.assertTrue(
                torch.all(actors[agent_id].log_std >= -2.0)
                and torch.all(actors[agent_id].log_std <= 0.0)
            )


if __name__ == "__main__":
    unittest.main()
