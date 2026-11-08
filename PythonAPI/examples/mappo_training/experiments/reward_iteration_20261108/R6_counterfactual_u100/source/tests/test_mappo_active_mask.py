import unittest

import numpy as np
import torch

from training.train_no_rllib_mappo_robust_v2 import (
    RolloutBuffer,
    build_active_step_cache,
    refresh_observations_after_active_change,
    validate_active_mask,
    validate_episode_transition_counts,
    validate_rollout_step_protocol,
    write_active_transitions,
)


class _SpyActor:
    def __init__(self):
        self.calls = 0
        self.obs_shapes = []

    def sample_action_z(self, obs):
        self.calls += 1
        self.obs_shapes.append(tuple(obs.shape))
        batch = obs.shape[0]
        zeros = torch.zeros((batch, 2), dtype=torch.float32, device=obs.device)
        logp = torch.zeros(batch, dtype=torch.float32, device=obs.device)
        return zeros, logp, zeros.clone()


class _SpyCritic:
    def __init__(self):
        self.calls = []

    def __call__(self, global_state, agent_id):
        self.calls.append((agent_id, tuple(global_state.shape)))
        return torch.full(
            (global_state.shape[0],), float(agent_id),
            dtype=torch.float32, device=global_state.device,
        )


class _EarlyTerminationEnv:
    """Two-agent mock: agent 0 terminates, agent 1 acts three more times."""

    def __init__(self, remaining_steps=3):
        self.agent_ids = [0, 1]
        self.remaining_steps = remaining_steps
        self.step_index = 0
        self.action_key_history = []
        self.obs = {
            0: np.full(19, 1.0, dtype=np.float32),
            1: np.full(19, 2.0, dtype=np.float32),
        }

    def reset(self):
        return dict(self.obs), {
            "active_mask": {0: True, 1: True},
            "episode_end_reason": "running",
        }

    def get_global_state(self):
        return np.concatenate([self.obs[aid] for aid in self.agent_ids])

    def step(self, actions):
        self.action_key_history.append(set(actions.keys()))
        is_first = self.step_index == 0
        is_final = self.step_index == self.remaining_steps

        if is_first:
            self.obs = {
                0: np.zeros(19, dtype=np.float32),
                1: np.full(19, 3.0, dtype=np.float32),
            }
            # Emulate CARLA: returned observations were captured before agent 0
            # was despawned, while get_global_state() now sees post-cleanup slots.
            returned_obs = {
                0: np.full(19, 9.0, dtype=np.float32),
                1: np.full(19, 99.0, dtype=np.float32),
            }
        else:
            self.obs = {
                0: np.zeros(19, dtype=np.float32),
                1: np.full(19, 3.0 + self.step_index, dtype=np.float32),
            }
            returned_obs = dict(self.obs)

        dones = {0: True, 1: is_final}
        terminated = {0: True, 1: is_final}
        truncated = {0: False, 1: False}
        active_mask = {0: False, 1: not is_final}
        reason = "all_agents_reached" if is_final else "running"
        rewards = {0: 10.0 if is_first else 0.0, 1: 1.0}
        info = {
            "terminated": terminated,
            "truncated": truncated,
            "active_mask": active_mask,
            "episode_end_reason": reason,
        }
        self.step_index += 1
        return returned_obs, rewards, dones, info


class ActiveMaskRolloutTests(unittest.TestCase):
    def test_early_terminated_agent_has_no_extra_transitions(self):
        env = _EarlyTerminationEnv(remaining_steps=3)
        agent_ids = [0, 1]
        actors = {aid: _SpyActor() for aid in agent_ids}
        critic = _SpyCritic()
        buffer = RolloutBuffer(agent_ids)
        device = torch.device("cpu")

        obs, reset_info = env.reset()
        active_mask = validate_active_mask(
            reset_info["active_mask"], agent_ids, context="test reset"
        )
        active_decision_steps = {aid: 0 for aid in agent_ids}
        buffer_starts = {aid: len(buffer.get_agent_data(aid)) for aid in agent_ids}

        while True:
            action_dict, step_cache = build_active_step_cache(
                obs_dict=obs,
                active_mask=active_mask,
                global_state_np=env.get_global_state(),
                actors=actors,
                critic=critic,
                agent_ids=agent_ids,
                device=device,
                context=f"test step={env.step_index}",
            )
            for aid in step_cache:
                active_decision_steps[aid] += 1

            raw_next_obs, rewards, dones, info = env.step(action_dict)
            next_active, dones, terminated, truncated = validate_rollout_step_protocol(
                dones=dones,
                terminated=info["terminated"],
                truncated=info["truncated"],
                next_active_mask=info["active_mask"],
                episode_end_reason=info["episode_end_reason"],
                agent_ids=agent_ids,
                context=f"test step={env.step_index - 1}",
            )
            write_active_transitions(
                buffer=buffer,
                step_cache=step_cache,
                rewards=rewards,
                dones=dones,
                terminated=terminated,
                truncated=truncated,
                scenario_name="mock.json",
                context=f"test step={env.step_index - 1}",
            )
            next_obs = refresh_observations_after_active_change(
                env=env,
                next_obs=raw_next_obs,
                current_active_mask=active_mask,
                next_active_mask=next_active,
                episode_end_reason=info["episode_end_reason"],
                agent_ids=agent_ids,
                context=f"test step={env.step_index - 1}",
            )
            if env.step_index == 1:
                self.assertEqual(float(raw_next_obs[1][0]), 99.0)
                self.assertEqual(float(next_obs[1][0]), 3.0)
            obs = next_obs
            active_mask = next_active
            if info["episode_end_reason"] != "running":
                break

        buffer_counts = validate_episode_transition_counts(
            buffer=buffer,
            buffer_starts=buffer_starts,
            active_decision_steps=active_decision_steps,
            agent_ids=agent_ids,
            context="test episode",
        )

        # One real terminal decision for agent 0; agent 1 continues three steps.
        self.assertEqual(active_decision_steps, {0: 1, 1: 4})
        self.assertEqual(buffer_counts, active_decision_steps)
        self.assertEqual(env.action_key_history, [{0, 1}, {1}, {1}, {1}])
        self.assertEqual(actors[0].calls, 1)
        self.assertEqual(actors[1].calls, 4)
        self.assertEqual([aid for aid, _shape in critic.calls], [0, 1, 1, 1, 1])

        terminal_transition = buffer.get_agent_data(0)[0]
        self.assertEqual(terminal_transition.done, 1.0)
        self.assertEqual(terminal_transition.terminated, 1.0)
        self.assertEqual(terminal_transition.reward, 10.0)
        self.assertTrue(np.all(terminal_transition.obs == 1.0))

        all_transitions = [
            transition
            for aid in agent_ids
            for transition in buffer.get_agent_data(aid)
        ]
        self.assertEqual(len(all_transitions), 5)
        self.assertTrue(all(t.obs.shape == (19,) for t in all_transitions))
        self.assertTrue(all(t.global_state.shape == (38,) for t in all_transitions))
        self.assertTrue(all(t.action_z.shape == (2,) for t in all_transitions))
        self.assertTrue(all(shape == (1, 19) for shape in actors[0].obs_shapes))
        self.assertTrue(all(shape == (1, 19) for shape in actors[1].obs_shapes))
        self.assertTrue(all(shape == (1, 38) for _aid, shape in critic.calls))
        critic_batch = np.stack([t.global_state for t in all_transitions])
        self.assertEqual(critic_batch.shape, (5, 38))

        agent1_data = buffer.get_agent_data(1)
        self.assertEqual([t.done for t in agent1_data], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(
            [t.terminated for t in agent1_data], [0.0, 0.0, 0.0, 1.0]
        )
        self.assertEqual([float(t.obs[0]) for t in agent1_data], [2.0, 3.0, 4.0, 5.0])
        for transition in agent1_data[1:]:
            self.assertTrue(np.all(transition.global_state[:19] == 0.0))
            self.assertTrue(np.all(transition.global_state[19:] != 0.0))

    def test_global_reason_and_next_mask_must_agree(self):
        with self.assertRaisesRegex(RuntimeError, "no active agent remains"):
            validate_rollout_step_protocol(
                dones={0: True, 1: True},
                terminated={0: True, 1: True},
                truncated={0: False, 1: False},
                next_active_mask={0: False, 1: False},
                episode_end_reason="running",
                agent_ids=[0, 1],
                context="running-without-active",
            )

        with self.assertRaisesRegex(RuntimeError, "active agents remain"):
            validate_rollout_step_protocol(
                dones={0: True, 1: False},
                terminated={0: True, 1: False},
                truncated={0: False, 1: False},
                next_active_mask={0: False, 1: True},
                episode_end_reason="all_agents_reached",
                agent_ids=[0, 1],
                context="ended-with-active",
            )

        with self.assertRaisesRegex(RuntimeError, "must equal not done"):
            validate_rollout_step_protocol(
                dones={0: True, 1: False},
                terminated={0: True, 1: False},
                truncated={0: False, 1: False},
                next_active_mask={0: True, 1: True},
                episode_end_reason="running",
                agent_ids=[0, 1],
                context="done-but-active",
            )

        with self.assertRaisesRegex(ValueError, "unknown episode_end_reason"):
            validate_rollout_step_protocol(
                dones={0: True, 1: True},
                terminated={0: True, 1: True},
                truncated={0: False, 1: False},
                next_active_mask={0: False, 1: False},
                episode_end_reason="all_agent_reched",
                agent_ids=[0, 1],
                context="misspelled-reason",
            )

    def test_active_mask_and_observation_keys_are_strict(self):
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            validate_active_mask({0: True}, [0, 1], context="missing-key")

        with self.assertRaisesRegex(RuntimeError, "missing rollout inputs"):
            build_active_step_cache(
                obs_dict={0: np.zeros(19, dtype=np.float32)},
                active_mask={0: True, 1: True},
                global_state_np=np.zeros(38, dtype=np.float32),
                actors={0: _SpyActor(), 1: _SpyActor()},
                critic=_SpyCritic(),
                agent_ids=[0, 1],
                device=torch.device("cpu"),
                context="missing-active-observation",
            )


if __name__ == "__main__":
    unittest.main()
