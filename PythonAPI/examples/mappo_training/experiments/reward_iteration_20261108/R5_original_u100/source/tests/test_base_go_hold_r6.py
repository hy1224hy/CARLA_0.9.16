import unittest

from evaluation.base_go_hold_protocol import (
    TOWN05_R6_GO_HOLD_PROTOCOL_VERSION,
    evaluate_behavioral_health,
)


def _contract():
    return {
        "protocol_version": TOWN05_R6_GO_HOLD_PROTOCOL_VERSION,
        "bindings": {"agent_ids": [0, 1]},
        "behavioral_health": {
            "assessment_milestones": [5, 10, 20],
            "warning_from_update": 5,
            "hold_from_update": 10,
            "required_consecutive_milestones": 2,
            "collapse_scope": "any_agent",
            "zero_success_required": True,
            "deterministic_brake_mode_rate_at_or_above": 0.95,
            "mean_speed_mps_at_or_below": 0.1,
            "mean_progress_m_per_episode_at_or_below": 3.0,
            "timeout_rate_at_or_above": 0.95,
            "previous_milestone_report_required_for_hold": True,
        },
    }


def _summary(collapsed_agents):
    agents = {}
    for agent_id in (0, 1):
        collapsed = agent_id in set(collapsed_agents)
        agents[str(agent_id)] = {
            "brake_mode_rate": 1.0 if collapsed else 0.1,
            "mean_speed_mps": 0.01 if collapsed else 2.0,
            "mean_progress_m_per_episode": 0.0 if collapsed else 20.0,
        }
    return {
        "outcome_counts": {"success": 0},
        "behavior": {"timeout_rate": 1.0, "agents": agents},
    }


class R6SingleAgentCollapseTests(unittest.TestCase):
    def test_same_single_agent_collapsing_twice_holds(self):
        result = evaluate_behavioral_health(
            update=10,
            c0_summary=_summary([0]),
            previous_c0_summary=_summary([0]),
            contract=_contract(),
        )
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["persistent_collapsed_agents"], ["0"])
        self.assertEqual(result["reason_codes"], ["BEHAVIORAL_COLLAPSE"])

    def test_different_agents_at_adjacent_milestones_do_not_fake_persistence(self):
        result = evaluate_behavioral_health(
            update=10,
            c0_summary=_summary([1]),
            previous_c0_summary=_summary([0]),
            contract=_contract(),
        )
        self.assertEqual(result["status"], "WARNING")
        self.assertFalse(result["persistent_collapse"])
        self.assertEqual(result["persistent_collapsed_agents"], [])

    def test_healthy_peer_cannot_mask_persistently_collapsed_agent(self):
        result = evaluate_behavioral_health(
            update=10,
            c0_summary=_summary([1]),
            previous_c0_summary=_summary([1]),
            contract=_contract(),
        )
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["observed"]["current"]["agents"]["0"]["collapsed"])
        self.assertTrue(result["observed"]["current"]["agents"]["1"]["collapsed"])


if __name__ == "__main__":
    unittest.main()
