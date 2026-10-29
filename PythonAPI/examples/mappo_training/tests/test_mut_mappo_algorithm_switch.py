"""Strict IPPO integration and mutation-controlled MAPPO checks; no CARLA."""

import unittest

import numpy as np
import torch

from models.actor_critic import Actor
from training.mut_mappo_learning import TeamLearner, actor_states_hash
from training.train_no_rllib_mappo_robust_v2 import (
    RolloutBuffer, Transition, prepare_mappo_update_batch,
)


def initial_actors():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        return {aid: Actor(20, 2, hidden=16).state_dict() for aid in (0, 1)}


def team(algorithm='ippo', **options):
    config = {'algorithm': algorithm, 'method': 'off', 'hidden': 16,
              'ppo_epochs': 1, 'minibatch_size': 8, 'shuffle': False,
              'target_kl': 0.0}
    config.update(options)
    return TeamLearner('A', 35, initial_actors(), config)


def rollout(learner, *, other_shift=0.0, role3_reward_shift=0.0):
    """Role 2 is fixed; role 3 and red slots can be changed independently."""
    result = RolloutBuffer([2, 3])
    for step in range(4):
        obs = {aid: np.linspace(-0.5, 0.5, 20, dtype=np.float32) + step * 0.1 + aid * 0.02
               for aid in range(4)}
        for aid in (0, 1, 3):
            obs[aid] += other_shift
        state = np.concatenate([obs[aid] for aid in range(4)])
        _, cache = learner.sample(obs, state, {aid: True for aid in range(4)})
        for aid in (2, 3):
            row = cache[aid]
            reward = float(step + aid + (role3_reward_shift if aid == 3 else 0))
            result.add(aid, Transition(row['obs'], row['global_state'], row['action_z'],
                row['logp'], reward, float(step == 3), 0.0, row['value'],
                'following' if step < 2 else 'cut_in', 'timeout'))
    for aid in (2, 3):
        result.mark_episode_end(aid)
        result.add_episode_bootstrap(aid, state + 0.1)
    return result


def role2_prepared(learner, buffer):
    # Production preparation reads only the requested role, even though the
    # public buffer carries both physical blue slots.
    return prepare_mappo_update_batch(buffer, learner.critics[2], [2], learner.device,
        gamma=0.997, gae_lambda=0.95, obs_dim=20, state_dim=80, action_dim=2,
        value_normalizer=learner.value_normalizers[2], context='IPPO role isolation')


class AlgorithmSwitchTests(unittest.TestCase):
    def test_strict_ippo_has_independent_models_optimizer_and_value_norm(self):
        learner = team()
        self.assertIsNone(learner.critic)
        self.assertIsNone(learner.critic_optim)
        self.assertIsNone(learner.value_normalizer)
        self.assertEqual(set(learner.critics), {2, 3})
        self.assertIsNot(learner.value_normalizers[2], learner.value_normalizers[3])
        parameter_sets = []
        for aid in (2, 3):
            self.assertEqual(learner.critics[aid].agent_ids, (aid,))
            self.assertEqual(learner.critics[aid].critics[str(aid)].net[0].in_features, 20)
            owned = {id(p) for group in learner.critic_optims[aid].param_groups for p in group['params']}
            self.assertEqual(owned, {id(p) for p in learner.critics[aid].parameters()})
            parameter_sets.append(owned)
        self.assertFalse(parameter_sets[0] & parameter_sets[1])
        self.assertEqual(learner.model_spec['value_normalizer_scope'], 'independent_per_blue_role')

    def test_other_state_and_return_changes_cannot_change_role2_gae_or_update(self):
        first, changed = team(), team()
        original = rollout(first)
        perturbed = rollout(changed, other_shift=10, role3_reward_shift=100)
        for left, right in zip(original.data[2], perturbed.data[2]):
            self.assertEqual(left.value, right.value)
            np.testing.assert_array_equal(left.action_z, right.action_z)
        left, right = role2_prepared(first, original), role2_prepared(changed, perturbed)
        for field in ('raw_adv', 'adv', 'ret'):
            torch.testing.assert_close(left.per_agent_tensors[2][field],
                                       right.per_agent_tensors[2][field], rtol=0, atol=0)
        first_metrics, changed_metrics = first.update(original), changed.update(perturbed)
        self.assertEqual(first_metrics['algorithm'], 'ippo')
        self.assertEqual(first.update_count, 1)
        self.assertEqual(first_metrics['mutual_beta'], 0)
        self.assertEqual(set(first_metrics['per_role_metrics']), {2, 3})
        for key, value in first.actors[2].state_dict().items():
            torch.testing.assert_close(value, changed.actors[2].state_dict()[key], rtol=0, atol=0)
        for key, value in first.critics[2].state_dict().items():
            torch.testing.assert_close(value, changed.critics[2].state_dict()[key], rtol=0, atol=0)
        self.assertEqual(first.value_normalizers[2].statistics(), changed.value_normalizers[2].statistics())
        self.assertNotEqual(first.value_normalizers[3].statistics()['mean'],
                            changed.value_normalizers[3].statistics()['mean'])
        self.assertEqual(first_metrics['preupdate_ratio_errors'][2],
                         changed_metrics['preupdate_ratio_errors'][2])

    def test_full_independent_resume_matches_sampling_and_next_update(self):
        original = team()
        original.update(rollout(original))
        checkpoint = original.state_dict()
        self.assertEqual(set(checkpoint['critic']), {2, 3})
        self.assertEqual(set(checkpoint['critic_optim']), {2, 3})
        self.assertEqual(set(checkpoint['value_normalizer']), {2, 3})
        restored = team()
        restored.load_state_dict(checkpoint)
        original.update(rollout(original))
        restored.update(rollout(restored))
        self.assertEqual(original.update_count, 2)
        self.assertEqual(actor_states_hash(original.export_actors()), actor_states_hash(restored.export_actors()))
        self.assertEqual(original.next_episode_seed(), restored.next_episode_seed())
        for aid in (2, 3):
            for key, value in original.critics[aid].state_dict().items():
                torch.testing.assert_close(value, restored.critics[aid].state_dict()[key], rtol=0, atol=0)
            self.assertEqual(original.value_normalizers[aid].statistics(),
                             restored.value_normalizers[aid].statistics())

    def test_active_roles_and_algorithm_mismatch_fail_closed(self):
        ippo = team()
        _, cache = ippo.sample({2: np.zeros(20, np.float32)}, np.zeros(80, np.float32),
                               {2: True, 3: False})
        self.assertEqual(set(cache), {2})
        with self.assertRaises(ValueError):
            team(method='plain')
        with self.assertRaises(ValueError):
            team().load_state_dict(team('mappo').state_dict())
        with self.assertRaises(ValueError):
            team('mappo').load_state_dict(team().state_dict())
        corrupted = ippo.state_dict()
        corrupted['value_normalizer'].pop(3)
        with self.assertRaises(ValueError):
            ippo.load_state_dict(corrupted)

    def test_legacy_default_mappo_decoding_requires_the_same_config(self):
        # Source fingerprints and run identity remain outside this narrow
        # decoder; explicit new algorithm configs are never silently migrated.
        config = {'method': 'off', 'hidden': 16}
        original = TeamLearner('A', 35, initial_actors(), config)
        legacy = original.state_dict()
        for field in ('algorithm', 'critic_input_scope', 'value_normalizer_scope'):
            legacy['model_spec'].pop(field)
        restored = TeamLearner('A', 35, initial_actors(), config)
        restored.load_state_dict(legacy)
        explicit = TeamLearner('A', 35, initial_actors(), dict(config, algorithm='mappo'))
        with self.assertRaises(ValueError):
            explicit.load_state_dict(legacy)

    def test_random_and_strength_gate_controls_are_reproducible_on_resume(self):
        teacher = team('mappo')
        with torch.no_grad():
            for actor in teacher.actors.values():
                actor.mu_head.bias.add_(0.4)
        for variant in ('random', 'strength'):
            learner = team('mappo', method='selective',
                distillation={'direction': 'teacher_to_student', 'gate_variant': variant})
            learner.set_peer(teacher.snapshot(1), {'following': 1, 'cut_in': 0})
            metrics = learner.update(rollout(learner))
            self.assertEqual(metrics['distillation']['direction'], 'teacher_to_student')
            self.assertEqual(metrics['distillation']['reduction'], 'all_valid')
            for aid in (2, 3):
                self.assertEqual(metrics[f'evidence_gate_fraction_{aid}'], 0.5)
                self.assertEqual(metrics[f'gate_weight_mean_{aid}'], 0.5)
                self.assertEqual(metrics[f'gate_nonzero_coverage_{aid}'],
                                 0.5 if variant == 'random' else 1.0)
            restored = team('mappo', method='selective',
                distillation={'direction': 'teacher_to_student', 'gate_variant': variant})
            restored.load_state_dict(learner.state_dict())
            left, right = learner.update(rollout(learner)), restored.update(rollout(restored))
            self.assertEqual(actor_states_hash(learner.export_actors()),
                             actor_states_hash(restored.export_actors()))
            self.assertEqual(left['mutual_gated_kl_2'], right['mutual_gated_kl_2'])
        with self.assertRaises(ValueError):
            team('mappo', method='plain', distillation={'gate_variant': 'random'})


if __name__ == '__main__':
    unittest.main()
