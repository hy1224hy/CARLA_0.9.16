"""B2 mechanism checks through the real PPO update; no simulator calls."""
import copy
import unittest

import torch

from tests.test_mut_mappo_learning import buffer_for, learner
from training.mut_mappo_learning import GateState, actor_states_hash


class GateAuditTests(unittest.TestCase):
    def peer(self):
        teacher = learner("B")
        with torch.no_grad():
            for actor in teacher.actors.values():
                actor.mu_head.bias.add_(0.4)
        return teacher.snapshot(1)

    def assert_same_update(self, left, right):
        self.assertEqual(actor_states_hash(left.export_actors()),
                         actor_states_hash(right.export_actors()))
        for key, tensor in left.critic.state_dict().items():
            torch.testing.assert_close(tensor, right.critic.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(left.value_normalizer.statistics(), right.value_normalizer.statistics())

    def test_all_closed_selective_matches_b0_update(self):
        baseline, selective = learner(), learner(method="selective")
        selective.set_peer(self.peer(), {"following": 0})
        baseline.update(buffer_for(baseline))
        metrics = selective.update(buffer_for(selective))
        self.assert_same_update(baseline, selective)
        for aid in (2, 3):
            self.assertGreater(metrics[f"mutual_kl_{aid}"], 0)
            self.assertEqual(metrics[f"mutual_gated_kl_{aid}"], 0)
            self.assertEqual(metrics[f"gate_fraction_{aid}"], 0)

    def test_all_open_selective_matches_plain_b1_update(self):
        plain, selective = learner(method="plain"), learner(method="selective")
        peer = self.peer()
        plain.set_peer(peer)
        selective.set_peer(peer, {"following": 1})
        first = plain.update(buffer_for(plain))
        second = selective.update(buffer_for(selective))
        self.assert_same_update(plain, selective)
        for aid in (2, 3):
            self.assertGreater(first[f"mutual_gated_kl_{aid}"], 0)
            self.assertEqual(first[f"mutual_gated_kl_{aid}"], second[f"mutual_gated_kl_{aid}"])
            self.assertTrue(all(p.grad is None for p in selective.peer[aid].parameters()))

    @staticmethod
    def evidence():
        return {team: {"blocking": dict(n_all=8, q_count=q, b_count=0,
                    case_keys=list(range(8)), teacher_hash=team, version=1)}
                for team, q in (("A", 0), ("B", 2))}

    def test_missing_duplicate_or_wrong_length_pair_keys_rejected(self):
        for keys in (None, [0] * 8, list(range(7))):
            with self.subTest(keys=keys):
                stats = self.evidence()
                for families in stats.values():
                    if keys is None:
                        del families["blocking"]["case_keys"]
                    else:
                        families["blocking"]["case_keys"] = copy.deepcopy(keys)
                with self.assertRaises(ValueError):
                    GateState().refresh({"A": "A", "B": "B"}, stats,
                                        version=1, completed_round=1)

    def test_invalid_efficiency_gap_rejected(self):
        for margin in (-0.1, float("nan"), float("inf")):
            stats = self.evidence()
            for families in stats.values():
                families["blocking"]["utility_sum"] = 0.0
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                GateState({"quality_metric": "efficiency", "min_efficiency_gap": margin}).refresh(
                    {"A": "A", "B": "B"}, stats, version=1, completed_round=1)


if __name__ == "__main__":
    unittest.main()
