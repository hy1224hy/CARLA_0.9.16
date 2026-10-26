"""Small result-critical checks; no CARLA connection or world ticks."""
import copy
import random
import unittest

import numpy as np
import torch

from models.actor_critic import Actor
from training.mut_mappo_learning import (
    GateState, TeamLearner, actor_states_hash, student_teacher_kl,
)
from training.train_no_rllib_mappo_robust_v2 import RolloutBuffer, Transition


def initial_actors():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return {role: Actor(20, 2, hidden=16).state_dict() for role in (0, 1)}


def learner(name="A", method="off", beta=0.01):
    return TeamLearner(name, 32, initial_actors(), {
        "method": method, "beta": beta, "hidden": 16, "ppo_epochs": 2,
        "minibatch_size": 8, "shuffle": False, "target_kl": 0.0,
    })


def buffer_for(team):
    buffer = RolloutBuffer([2, 3])
    for step in range(3):
        obs = {aid: np.full(20, step * 0.1 + aid, np.float32) for aid in range(4)}
        state = np.concatenate([obs[aid] for aid in range(4)])
        _, cache = team.sample(obs, state, {aid: True for aid in range(4)})
        for aid in (2, 3):
            row = cache[aid]
            buffer.add(aid, Transition(row["obs"], row["global_state"], row["action_z"],
                row["logp"], float(step + aid), float(step == 2), float(step == 2),
                row["value"], "following", "success"))
    for aid in (2, 3):
        buffer.mark_episode_end(aid)
        buffer.add_episode_bootstrap(aid, None)
    return buffer


class MutLearningTests(unittest.TestCase):
    def test_beta_zero_reuses_exact_ppo_and_four_car_critic(self):
        baseline, mutual_zero = learner(), learner("B", "plain", 0)
        buffers = buffer_for(baseline), buffer_for(mutual_zero)
        baseline.update(buffers[0])
        mutual_zero.update(buffers[1])
        self.assertEqual(baseline.critic.net[0].in_features, 84)
        for aid in (2, 3):
            for key, value in baseline.actors[aid].state_dict().items():
                torch.testing.assert_close(value, mutual_zero.actors[aid].state_dict()[key], rtol=0, atol=0)
        for key, value in baseline.critic.state_dict().items():
            torch.testing.assert_close(value, mutual_zero.critic.state_dict()[key], rtol=0, atol=0)

    def test_rng_restored_and_resumed_sampling_matches(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        team = learner()
        buffer_for(team)
        self.assertEqual(random.getstate(), expected[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], expected[1][1]))
        torch.testing.assert_close(torch.get_rng_state(), expected[2], rtol=0, atol=0)
        checkpoint = team.state_dict()
        resumed = learner()
        resumed.load_state_dict(checkpoint)
        kwargs = ({aid: np.zeros(20, np.float32) for aid in range(4)},
                  np.zeros(80, np.float32), {0: True, 1: True, 2: True, 3: False})
        _, first = team.sample(*kwargs)
        _, second = resumed.sample(*kwargs)
        np.testing.assert_array_equal(first[2]["action_z"], second[2]["action_z"])
        self.assertEqual(set(first), {2})
        self.assertEqual(team.next_episode_seed(), resumed.next_episode_seed())

    def test_gated_kl_keeps_zero_gate_rows_and_teacher_has_no_gradient(self):
        student = Actor(20, 2, hidden=16)
        teacher = copy.deepcopy(student)
        with torch.no_grad():
            teacher.mu_head.bias.add_(1.0)
        obs = torch.zeros(4, 20)
        full = student_teacher_kl(student, teacher, obs)
        half = student_teacher_kl(student, teacher, obs, torch.tensor([1, 0, 1, 0]))
        torch.testing.assert_close(half, full * 0.5)
        half.backward()
        self.assertTrue(any(p.grad is not None for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_teacher_is_fixed_and_full_state_resume_matches_next_update(self):
        team, teacher = learner("A", "selective"), learner("B")
        snapshot = teacher.snapshot(1)
        team.set_peer(snapshot, {"following": 1})
        first_buffer = buffer_for(team)
        teacher.update(buffer_for(teacher))
        self.assertEqual(actor_states_hash(team.peer_snapshot["actors"]), snapshot["hash"])
        metrics = team.update(first_buffer)
        self.assertEqual(metrics["gate_fraction_2"], 1.0)
        self.assertTrue(all(p.grad is None for actor in team.peer.values() for p in actor.parameters()))
        restored = learner("A", "selective")
        restored.load_state_dict(team.state_dict())
        team.update(buffer_for(team))
        restored.update(buffer_for(restored))
        self.assertEqual(actor_states_hash(team.export_actors()), actor_states_hash(restored.export_actors()))
        for key, value in team.critic.state_dict().items():
            torch.testing.assert_close(value, restored.critic.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(team.value_normalizer.statistics(), restored.value_normalizer.statistics())

    def test_gate_uses_all_episode_counts_and_checks_snapshot_version(self):
        gate = GateState()
        hashes = {"A": "a" * 64, "B": "b" * 64}
        stats = {"A": {"following": {"n": 8, "eligible_response_count": 1, "violation_count": 2}},
                 "B": {"following": {"n": 8, "eligible_response_count": 3, "violation_count": 1}}}
        result = gate.compute(stats, hashes, version=1, completed_round=5)
        self.assertEqual(result, {"A": {"following": 1}, "B": {"following": 0}})
        restored = GateState()
        restored.load_state_dict(gate.state_dict())
        self.assertEqual(restored.gates_for("A", hashes["B"]), {"following": 1})
        with self.assertRaises(ValueError):
            restored.gates_for("A", "wrong")
        stats["B"]["following"]["version"] = 1
        with self.assertRaises(ValueError):
            gate.compute(stats, hashes, version=2, completed_round=15)


if __name__ == "__main__":
    unittest.main()
