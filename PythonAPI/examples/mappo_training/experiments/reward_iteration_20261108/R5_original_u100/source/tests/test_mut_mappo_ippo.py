"""Information isolation and common-PPO interoperability, without CARLA."""

import copy
import unittest

import numpy as np
import torch

from models.actor_critic import Actor
from models.mut_mappo_critics import LocalObservationCritic
from training.train_no_rllib_mappo_robust_v2 import (
    RolloutBuffer, Transition, prepare_mappo_update_batch, run_mappo_ppo_update,
)
from training.value_normalization import ValueNormalizer


class LocalCriticTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.random.fork_rng()
        self.rng.__enter__()
        torch.manual_seed(18)

    def tearDown(self):
        self.rng.__exit__(None, None, None)

    def test_other_slots_cannot_change_values_or_local_gradients(self):
        critic = LocalObservationCritic(hidden=16)
        for aid in (2, 3):
            start, end = aid * 20, (aid + 1) * 20
            first = torch.randn(5, 80, requires_grad=True)
            second = first.detach().clone()
            second[:, :start] += 100
            second[:, end:] -= 100
            second.requires_grad_(True)
            torch.testing.assert_close(critic(first, aid), critic(second, aid), rtol=0, atol=0)
            left_grad, = torch.autograd.grad(critic(first, aid).sum(), first)
            right_grad, = torch.autograd.grad(critic(second, aid).sum(), second)
            torch.testing.assert_close(left_grad, right_grad, rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(left_grad[:, :start])), 0)
            self.assertEqual(int(torch.count_nonzero(left_grad[:, end:])), 0)
            self.assertGreater(int(torch.count_nonzero(left_grad[:, start:end])), 0)

    def test_parameters_and_optimization_are_role_local(self):
        critic = LocalObservationCritic(hidden=16)
        self.assertFalse({id(p) for p in critic.critics['2'].parameters()}
                         & {id(p) for p in critic.critics['3'].parameters()})
        role3_before = copy.deepcopy(critic.critics['3'].state_dict())
        optim = torch.optim.Adam(critic.parameters(), lr=0.01)
        loss = critic(torch.randn(4, 80), 2).square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in critic.critics['3'].parameters()))
        optim.step()
        for key, before in role3_before.items():
            torch.testing.assert_close(before, critic.critics['3'].state_dict()[key], rtol=0, atol=0)

    def test_mixed_dispatch_and_checkpoint_round_trip(self):
        critic = LocalObservationCritic(hidden=16)
        states, ids = torch.randn(6, 80), torch.tensor([3, 2, 2, 3, 2, 3])
        expected = torch.stack([critic(state, int(aid))[0] for state, aid in zip(states, ids)])
        torch.testing.assert_close(critic(states, ids), expected)
        restored = LocalObservationCritic(hidden=16)
        restored.load_state_dict(critic.state_dict(), strict=True)
        torch.testing.assert_close(restored(states, ids), critic(states, ids), rtol=0, atol=0)
        self.assertEqual(critic.model_spec()['critic_input_scope'],
                         'own_physical_slot_local_observation_only')
        self.assertFalse(critic.model_spec()['critic_parameter_sharing'])

    def test_invalid_layout_and_role_fail_closed(self):
        with self.assertRaises(ValueError):
            LocalObservationCritic(state_dim=79)
        with self.assertRaises(ValueError):
            LocalObservationCritic(agent_ids=(2, 2))
        critic = LocalObservationCritic(hidden=16, agent_ids=(2,))
        for aid in (0, 3, torch.tensor([3]), torch.tensor([2, 2])):
            with self.assertRaises(ValueError):
                critic(torch.zeros(80), aid)
        with self.assertRaises(TypeError):
            critic(torch.zeros(80), torch.tensor([2.0]))
        with self.assertRaises(TypeError):
            critic(torch.zeros(80), True)

    def test_single_role_reuses_production_gae_and_ppo(self):
        """Each role owns its value stats/optimizer, with no second PPO implementation."""
        np_state = np.random.get_state()
        try:
            np.random.seed(21)
            for aid in (2, 3):
                actor = Actor(20, 2, hidden=16)
                critic = LocalObservationCritic(hidden=16, agent_ids=(aid,))
                normalizer = ValueNormalizer()
                actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
                critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)
                buffer = RolloutBuffer([aid])
                for step in range(3):
                    state = np.linspace(0, 1, 80, dtype=np.float32) + step * 0.1
                    local_obs = state[aid * 20:(aid + 1) * 20]
                    with torch.no_grad():
                        obs_t = torch.from_numpy(local_obs).unsqueeze(0)
                        _, logp, z = actor.sample_action_z(obs_t)
                        value = normalizer.denormalize(critic(torch.from_numpy(state), aid))
                    terminal = float(step == 2)
                    buffer.add(aid, Transition(local_obs, state, z[0].numpy(),
                        float(logp.item()), float(aid + step), terminal, terminal,
                        float(value.item()), 'following', 'success'))
                buffer.mark_episode_end(aid)
                buffer.add_episode_bootstrap(aid, None)
                batch = prepare_mappo_update_batch(buffer, critic, [aid], torch.device('cpu'),
                    gamma=0.997, gae_lambda=0.95, obs_dim=20, state_dim=80,
                    action_dim=2, value_normalizer=normalizer, context='IPPO local test')
                result = run_mappo_ppo_update(batch, {aid: actor}, {aid: actor_optim},
                    critic, critic_optim, [aid], obs_dim=20, state_dim=80, action_dim=2,
                    ppo_epochs=1, minibatch_size=8, clip_eps=0.15, ent_coef=0.03,
                    max_grad_norm=0.5, target_kl=0, value_normalizer=normalizer,
                    shuffle=False, context='IPPO local test')
                self.assertLessEqual(result.preupdate_ratio_errors[aid], 1e-6)
                self.assertEqual(normalizer.statistics()['update_count'], 1)
                self.assertEqual(set(critic.critics), {str(aid)})
                self.assertTrue(all(torch.isfinite(p).all() for p in critic.parameters()))
        finally:
            np.random.set_state(np_state)


if __name__ == '__main__':
    unittest.main()
