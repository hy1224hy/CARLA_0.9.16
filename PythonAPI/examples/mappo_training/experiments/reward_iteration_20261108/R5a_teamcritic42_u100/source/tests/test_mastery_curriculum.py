import unittest

from training.mastery_curriculum import MasteryGate, MasteryWindow


def _window(update, *, agent1_speed=0.4):
    return MasteryWindow(
        update=update,
        episode_count=16,
        agent_mean_speed_mps={0: 0.4, 1: agent1_speed},
        agent_mean_completion={0: 0.3, 1: 0.3},
        agent_brake_mode_rate={0: 0.2, 1: 0.2},
        team_success_rate=0.2,
        team_timeout_rate=0.8,
        team_collision_rate=0.0,
        team_offroad_rate=0.0,
    )


class MasteryGateTests(unittest.TestCase):
    def test_update_count_alone_never_promotes(self):
        gate = MasteryGate(stage_count=3, agent_ids=[0, 1])
        self.assertEqual(gate.stage_index, 0)
        self.assertEqual(gate.last_update, 0)

    def test_any_agent_failure_resets_streak_and_holds_stage(self):
        gate = MasteryGate(stage_count=3, agent_ids=[0, 1])
        evidence = gate.evaluate(_window(1, agent1_speed=0.0))
        self.assertFalse(evidence["passed"])
        self.assertIn("agent1.mean_speed", evidence["failures"])
        self.assertEqual(gate.stage_index, 0)
        self.assertEqual(gate.consecutive_passes, 0)

    def test_two_consecutive_passing_windows_promote_exactly_one_stage(self):
        gate = MasteryGate(stage_count=3, agent_ids=[0, 1])
        self.assertFalse(gate.evaluate(_window(1))["promoted"])
        self.assertTrue(gate.evaluate(_window(2))["promoted"])
        self.assertEqual(gate.stage_index, 1)
        self.assertEqual(gate.consecutive_passes, 0)

    def test_state_restore_preserves_pending_mastery_streak(self):
        source = MasteryGate(stage_count=3, agent_ids=[0, 1])
        source.evaluate(_window(4))
        restored = MasteryGate(stage_count=3, agent_ids=[0, 1])
        restored.load_state_dict(source.state_dict())
        self.assertTrue(restored.evaluate(_window(5))["promoted"])
        self.assertEqual(restored.stage_index, 1)

    def test_final_stage_pass_streak_is_capped_and_resumable(self):
        gate = MasteryGate(stage_count=1, agent_ids=[0, 1])
        for update in range(1, 6):
            evidence = gate.evaluate(_window(update))
            self.assertFalse(evidence["promoted"])
        self.assertEqual(gate.consecutive_passes, 2)
        restored = MasteryGate(stage_count=1, agent_ids=[0, 1])
        restored.load_state_dict(gate.state_dict())
        self.assertEqual(restored.consecutive_passes, 2)
        self.assertFalse(restored.evaluate(_window(6))["promoted"])


if __name__ == "__main__":
    unittest.main()
