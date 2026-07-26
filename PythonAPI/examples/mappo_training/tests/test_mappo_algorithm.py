import copy
import math
import unittest

import numpy as np
import torch
import torch.optim as optim

from models.actor_critic import Actor, AgentConditionedCritic
from training.train_no_rllib_mappo_robust_v2 import (
    RolloutBuffer,
    Transition,
    compute_clipped_actor_loss,
    prepare_mappo_update_batch,
    run_mappo_ppo_update,
    should_stop_for_kl,
    validate_prepared_mappo_batch,
)


OBS_DIM = 19
STATE_DIM = 38
ACTION_DIM = 3
AGENT_IDS = [0, 1]


def _make_models():
    actors = {
        aid: Actor(obs_dim=OBS_DIM, act_dim=ACTION_DIM, hidden=32)
        for aid in AGENT_IDS
    }
    critic = AgentConditionedCritic(
        state_dim=STATE_DIM, num_agents=len(AGENT_IDS), hidden=32
    )
    return actors, critic


def _build_buffer(actors, critic, episode_lengths):
    """Build deterministic ragged trajectories with policy-consistent log-probs."""
    buffer = RolloutBuffer(AGENT_IDS)
    for aid in AGENT_IDS:
        row_index = 0
        for episode_index, episode_length in enumerate(episode_lengths.get(aid, [])):
            for step_index in range(episode_length):
                sentinel = float(aid * 100 + episode_index * 10 + step_index)
                obs = np.linspace(-0.2, 0.2, OBS_DIM, dtype=np.float32)
                obs[0] = sentinel
                state = np.linspace(-0.4, 0.4, STATE_DIM, dtype=np.float32)
                state[0] = sentinel
                obs_tensor = torch.as_tensor(obs).unsqueeze(0)
                state_tensor = torch.as_tensor(state).unsqueeze(0)
                with torch.no_grad():
                    _action, log_prob, action_z = actors[aid].sample_action_z(
                        obs_tensor
                    )
                    value = critic(state_tensor, aid)
                is_last = step_index == episode_length - 1
                buffer.add(
                    aid,
                    Transition(
                        obs=obs.copy(),
                        global_state=state.copy(),
                        action_z=action_z.squeeze(0).cpu().numpy().copy(),
                        log_prob=float(log_prob.item()),
                        reward=float(1.0 + 0.25 * aid + 0.1 * row_index),
                        done=float(is_last),
                        terminated=float(is_last),
                        value=float(value.item()),
                        scenario_name=f"agent{aid}_episode{episode_index}",
                    ),
                )
                row_index += 1
            buffer.mark_episode_end(aid)
            buffer.add_episode_bootstrap(aid, None)
    return buffer


def _prepare(actors, critic, episode_lengths):
    return prepare_mappo_update_batch(
        buffer=_build_buffer(actors, critic, episode_lengths),
        critic=critic,
        agent_ids=AGENT_IDS,
        device=torch.device("cpu"),
        gamma=0.9,
        gae_lambda=0.95,
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        context="algorithm regression fixture",
    )


def _clone_parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_changed(before, module):
    return any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, module.parameters())
    )


def _assert_parameters_equal(test_case, before, module):
    for old, current in zip(before, module.parameters()):
        test_case.assertTrue(torch.equal(old, current.detach()))


class PreparedBatchRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260726)
        np.random.seed(20260726)

    def test_ragged_agent_batches_keep_owner_time_order_and_fixed_dimensions(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [2, 1], 1: [3, 2]})

        self.assertEqual(prepared.transition_counts, {0: 3, 1: 5})
        self.assertEqual(tuple(prepared.critic_states.shape), (8, STATE_DIM))
        self.assertEqual(tuple(prepared.critic_returns.shape), (8,))
        self.assertEqual(tuple(prepared.critic_agent_ids.shape), (8,))
        self.assertEqual(
            prepared.per_agent_tensors[0]["obs"][:, 0].tolist(),
            [0.0, 1.0, 10.0],
        )
        self.assertEqual(
            prepared.per_agent_tensors[1]["obs"][:, 0].tolist(),
            [100.0, 101.0, 102.0, 110.0, 111.0],
        )
        self.assertEqual(
            prepared.critic_agent_ids.tolist(), [0, 0, 0, 1, 1, 1, 1, 1]
        )
        self.assertTrue(
            torch.equal(
                prepared.critic_states,
                torch.cat(
                    [
                        prepared.per_agent_tensors[0]["state"],
                        prepared.per_agent_tensors[1]["state"],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                prepared.critic_returns,
                torch.cat(
                    [
                        prepared.per_agent_tensors[0]["ret"],
                        prepared.per_agent_tensors[1]["ret"],
                    ]
                ),
            )
        )
        for aid in AGENT_IDS:
            tensors = prepared.per_agent_tensors[aid]
            self.assertEqual(tuple(tensors["obs"].shape[1:]), (OBS_DIM,))
            self.assertEqual(tuple(tensors["act"].shape[1:]), (ACTION_DIM,))
            self.assertEqual(tuple(tensors["state"].shape[1:]), (STATE_DIM,))
            self.assertTrue(
                torch.allclose(
                    tensors["ret"], tensors["raw_adv"] + tensors["old_value"]
                )
            )
            self.assertAlmostEqual(float(tensors["adv"].mean()), 0.0, places=5)
            self.assertAlmostEqual(
                float(tensors["adv"].std(unbiased=False)), 1.0, places=5
            )

    def test_prepared_batch_validator_rejects_critic_agent_id_misalignment(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [2], 1: [2]})
        prepared.critic_agent_ids = prepared.critic_agent_ids.roll(1)

        with self.assertRaisesRegex(RuntimeError, "agent ids are misaligned"):
            validate_prepared_mappo_batch(
                prepared,
                AGENT_IDS,
                obs_dim=OBS_DIM,
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                context="misaligned critic ids",
            )


class CriticProtocolRegressionTests(unittest.TestCase):
    def test_agent_conditioned_critic_supports_uniform_and_mixed_id_batches(self):
        torch.manual_seed(17)
        critic = AgentConditionedCritic(STATE_DIM, len(AGENT_IDS), hidden=16)
        states = torch.randn(4, STATE_DIM)

        uniform_int = critic(states, 1)
        uniform_tensor = critic(states, torch.ones(4, dtype=torch.long))
        self.assertEqual(tuple(uniform_int.shape), (4,))
        self.assertTrue(torch.allclose(uniform_int, uniform_tensor))

        mixed_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        mixed = critic(states, mixed_ids)
        rowwise = torch.cat(
            [critic(states[index:index + 1], int(aid)) for index, aid in enumerate(mixed_ids)]
        )
        self.assertTrue(torch.allclose(mixed, rowwise))
        self.assertTrue(torch.isfinite(mixed).all())

    def test_agent_conditioned_critic_rejects_invalid_dimensions_and_ids(self):
        critic = AgentConditionedCritic(STATE_DIM, len(AGENT_IDS), hidden=16)
        states = torch.randn(3, STATE_DIM)
        with self.assertRaisesRegex(ValueError, "outside"):
            critic(states, -1)
        with self.assertRaisesRegex(ValueError, "outside"):
            critic(states, torch.tensor([0, 1, 2]))
        with self.assertRaisesRegex(ValueError, "batch size"):
            critic(states, torch.tensor([0, 1]))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            critic(states, torch.tensor([0.0, 1.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "shape"):
            critic(torch.randn(3, STATE_DIM - 1), 0)


class ProductionPpoUpdateRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260726)
        np.random.seed(20260726)

    def _run_update(
        self, prepared, actors, critic, *, ppo_epochs=2, clip_eps=0.2
    ):
        actor_optims = {
            aid: optim.Adam(actors[aid].parameters(), lr=1e-3)
            for aid in AGENT_IDS
        }
        critic_optim = optim.Adam(critic.parameters(), lr=1e-3)
        return run_mappo_ppo_update(
            prepared=prepared,
            actors=actors,
            actor_optims=actor_optims,
            critic=critic,
            critic_optim=critic_optim,
            agent_ids=AGENT_IDS,
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            ppo_epochs=ppo_epochs,
            minibatch_size=2,
            clip_eps=clip_eps,
            ent_coef=0.01,
            max_grad_norm=0.5,
            target_kl=0.0,
            shuffle=False,
            context="production PPO regression",
        )

    def test_clipped_objective_matches_hand_calculation(self):
        ratios = torch.tensor([1.5, 0.5, 1.5, 0.5])
        old_log_probs = torch.zeros(4)
        new_log_probs = torch.log(ratios)
        advantages = torch.tensor([1.0, 1.0, -1.0, -1.0])
        entropy = torch.zeros(4)

        loss, diagnostics = compute_clipped_actor_loss(
            new_log_probs,
            old_log_probs,
            advantages,
            entropy,
            clip_eps=0.2,
            ent_coef=0.0,
            context="hand-calculated clipping",
        )

        self.assertAlmostEqual(float(loss.item()), 0.15, places=6)
        self.assertAlmostEqual(diagnostics["clip_fraction"], 1.0, places=6)
        expected_kl = float(((ratios - 1.0) - torch.log(ratios)).mean())
        self.assertAlmostEqual(diagnostics["approx_kl"], expected_kl, places=6)

    def test_full_production_update_changes_models_without_mutating_batch(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [2, 1], 1: [3, 2]})
        actor_before = {aid: _clone_parameters(actors[aid]) for aid in AGENT_IDS}
        critic_before = _clone_parameters(critic)
        batch_before = copy.deepcopy(prepared)

        result = self._run_update(prepared, actors, critic, ppo_epochs=2)

        for aid in AGENT_IDS:
            self.assertTrue(_parameters_changed(actor_before[aid], actors[aid]))
            self.assertLess(result.preupdate_ratio_errors[aid], 1e-6)
            self.assertEqual(
                result.metrics[f"actor_samples_seen_{aid}"],
                prepared.transition_counts[aid] * 2,
            )
            for name, tensor in prepared.per_agent_tensors[aid].items():
                self.assertTrue(
                    torch.equal(tensor, batch_before.per_agent_tensors[aid][name])
                )
        self.assertTrue(_parameters_changed(critic_before, critic))
        self.assertEqual(result.metrics["critic_samples_seen"], 16)
        self.assertTrue(
            torch.equal(prepared.critic_states, batch_before.critic_states)
        )
        self.assertTrue(
            torch.equal(prepared.critic_returns, batch_before.critic_returns)
        )
        self.assertTrue(
            torch.equal(prepared.critic_agent_ids, batch_before.critic_agent_ids)
        )
        for metric_name, metric_value in result.metrics.items():
            values = metric_value if isinstance(metric_value, list) else [metric_value]
            for value in values:
                self.assertTrue(
                    math.isfinite(float(value)), msg=f"non-finite {metric_name}"
                )

    def test_invalid_batch_fails_preflight_before_any_parameter_update(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [2], 1: [2]})
        actor_before = {aid: _clone_parameters(actors[aid]) for aid in AGENT_IDS}
        critic_before = _clone_parameters(critic)
        prepared.per_agent_tensors[0]["adv"][0] = float("nan")

        with self.assertRaisesRegex(FloatingPointError, "NaN/Inf"):
            self._run_update(prepared, actors, critic)

        for aid in AGENT_IDS:
            _assert_parameters_equal(self, actor_before[aid], actors[aid])
        _assert_parameters_equal(self, critic_before, critic)

    def test_invalid_clip_setting_fails_before_any_parameter_update(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [2], 1: [2]})
        actor_before = {aid: _clone_parameters(actors[aid]) for aid in AGENT_IDS}
        critic_before = _clone_parameters(critic)

        with self.assertRaisesRegex(ValueError, "clip_eps"):
            self._run_update(prepared, actors, critic, clip_eps=0.0)

        for aid in AGENT_IDS:
            _assert_parameters_equal(self, actor_before[aid], actors[aid])
        _assert_parameters_equal(self, critic_before, critic)

    def test_agent_without_samples_is_not_updated_or_inserted_into_critic_batch(self):
        actors, critic = _make_models()
        prepared = _prepare(actors, critic, {0: [3], 1: []})
        actor_before = {aid: _clone_parameters(actors[aid]) for aid in AGENT_IDS}
        critic_before = _clone_parameters(critic)

        result = self._run_update(prepared, actors, critic, ppo_epochs=2)

        self.assertEqual(prepared.transition_counts, {0: 3, 1: 0})
        self.assertNotIn(1, prepared.per_agent_tensors)
        self.assertEqual(prepared.critic_agent_ids.tolist(), [0, 0, 0])
        self.assertTrue(_parameters_changed(actor_before[0], actors[0]))
        _assert_parameters_equal(self, actor_before[1], actors[1])
        self.assertTrue(_parameters_changed(critic_before, critic))
        self.assertEqual(result.metrics["actor_samples_seen_0"], 6)
        self.assertEqual(result.metrics["actor_samples_seen_1"], 0)
        self.assertEqual(result.metrics["critic_samples_seen"], 6)

    def test_kl_stopping_threshold_is_strict_and_zero_disables_it(self):
        self.assertFalse(should_stop_for_kl(100.0, 0.0))
        self.assertFalse(should_stop_for_kl(0.03, 0.02))
        self.assertTrue(should_stop_for_kl(0.030001, 0.02))


if __name__ == "__main__":
    unittest.main()
