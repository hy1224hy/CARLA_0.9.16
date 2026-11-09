import unittest

from evaluation.smoke_test_success_logic import (
    SuccessLogicAudit,
    SuccessProtocolError,
    build_near_goal_scenario,
    validate_reset_protocol,
)


AGENTS = (0, 1)


def _step_payload(
    *,
    reached,
    dones,
    terminated,
    truncated,
    active,
    reason,
    success_rewards,
    success_events,
    collisions=None,
):
    collisions = collisions or {0: False, 1: False}
    return {
        "rewards": {0: float(success_rewards[0]), 1: float(success_rewards[1])},
        "dones": dones,
        "reached_target": reached,
        "info": {
            "reward_details": {
                aid: {"success": float(success_rewards[aid]), "time_bonus": 1.0}
                for aid in AGENTS
            },
            "reward_diagnostics": {
                aid: {"success_event": float(success_events[aid])} for aid in AGENTS
            },
            "collisions": collisions,
            "terminated": terminated,
            "truncated": truncated,
            "active_mask": active,
            "episode_end_reason": reason,
        },
    }


def _first_success_payload():
    return _step_payload(
        reached={0: True, 1: False},
        dones={0: True, 1: False},
        terminated={0: True, 1: False},
        truncated={0: False, 1: False},
        active={0: False, 1: True},
        reason="running",
        success_rewards={0: 100.0, 1: 0.0},
        success_events={0: 1.0, 1: 0.0},
    )


def _all_success_payload(*, collisions=None):
    return _step_payload(
        reached={0: True, 1: True},
        dones={0: True, 1: True},
        terminated={0: True, 1: True},
        truncated={0: False, 1: False},
        active={0: False, 1: False},
        reason="all_agents_reached",
        success_rewards={0: 0.0, 1: 100.0},
        success_events={0: 0.0, 1: 1.0},
        collisions=collisions,
    )


class NearGoalScenarioTests(unittest.TestCase):
    def test_builds_starts_from_goal_boundary_for_arbitrary_yaw(self):
        scenario = build_near_goal_scenario(
            {
                "description": "base",
                "goal_zone": {
                    "x": 10.0,
                    "y": 20.0,
                    "z": 0.5,
                    "yaw": 0.0,
                    "half_length": 3.0,
                    "half_width": 1.75,
                },
                "agents": {"agent0": {"x": 0.0, "y": 0.0}},
            },
            lead_distance_m=5.0,
            following_gap_m=10.0,
        )

        self.assertAlmostEqual(scenario["agents"]["agent0"]["x"], 2.0)
        self.assertAlmostEqual(scenario["agents"]["agent0"]["y"], 20.0)
        self.assertAlmostEqual(scenario["agents"]["agent1"]["x"], -8.0)
        self.assertAlmostEqual(scenario["agents"]["agent1"]["y"], 20.0)
        self.assertEqual(scenario["agents"]["agent0"]["initial_speed_mps"], 0.0)

    def test_rejects_gap_that_can_overlap_spawned_vehicles(self):
        config = {
            "goal_zone": {
                "x": 0.0,
                "y": 0.0,
                "yaw": 90.0,
                "half_length": 3.0,
                "half_width": 1.75,
            }
        }
        with self.assertRaisesRegex(ValueError, ">= 6m"):
            build_near_goal_scenario(config, following_gap_m=5.9)


class SuccessLogicAuditTests(unittest.TestCase):
    def test_reset_requires_fully_active_nonterminal_agents(self):
        validate_reset_protocol(
            {
                "active_mask": {0: True, 1: True},
                "terminated": {0: False, 1: False},
                "truncated": {0: False, 1: False},
                "episode_end_reason": "running",
            },
            AGENTS,
        )

        with self.assertRaisesRegex(SuccessProtocolError, "all agents must be active"):
            validate_reset_protocol(
                {
                    "active_mask": {0: True, 1: False},
                    "terminated": {0: False, 1: False},
                    "truncated": {0: False, 1: False},
                    "episode_end_reason": "running",
                },
                AGENTS,
            )

    def test_two_agent_success_sequence_passes(self):
        audit = SuccessLogicAudit(AGENTS)
        first = _first_success_payload()
        audit.observe(step=1, **first)
        final = _all_success_payload()
        audit.observe(step=2, **final)

        result = audit.finalize()

        self.assertTrue(result["success_protocol_pass"])
        self.assertTrue(result["clean_run"])
        self.assertEqual(result["success_evidence"][0]["step"], 1)
        self.assertEqual(result["success_evidence"][1]["step"], 2)

    def test_success_requires_positive_reward_and_event_marker(self):
        audit = SuccessLogicAudit(AGENTS)
        payload = _first_success_payload()
        payload["info"]["reward_details"][0]["success"] = 0.0

        with self.assertRaisesRegex(SuccessProtocolError, "positive success reward"):
            audit.observe(step=1, **payload)

        audit = SuccessLogicAudit(AGENTS)
        payload = _first_success_payload()
        payload["info"]["reward_diagnostics"][0]["success_event"] = 0.0
        with self.assertRaisesRegex(SuccessProtocolError, "success_event=0.0"):
            audit.observe(step=1, **payload)

    def test_success_cannot_be_truncated_or_remain_active(self):
        payload = _first_success_payload()
        payload["info"]["terminated"][0] = False
        payload["info"]["truncated"][0] = True
        with self.assertRaisesRegex(SuccessProtocolError, "success must yield"):
            SuccessLogicAudit(AGENTS).observe(step=1, **payload)

        payload = _first_success_payload()
        payload["info"]["active_mask"][0] = True
        with self.assertRaisesRegex(SuccessProtocolError, "remains active"):
            SuccessLogicAudit(AGENTS).observe(step=1, **payload)

    def test_wrong_global_end_reason_is_rejected(self):
        audit = SuccessLogicAudit(AGENTS)
        audit.observe(step=1, **_first_success_payload())
        payload = _all_success_payload()
        payload["info"]["episode_end_reason"] = "all_agents_terminated"

        with self.assertRaisesRegex(SuccessProtocolError, "ended before clean cooperative success"):
            audit.observe(step=2, **payload)

    def test_collision_contamination_fails_clean_run(self):
        audit = SuccessLogicAudit(AGENTS)
        audit.observe(step=1, **_first_success_payload())
        audit.observe(
            step=2,
            **_all_success_payload(collisions={0: False, 1: True}),
        )

        with self.assertRaisesRegex(SuccessProtocolError, "contaminated"):
            audit.finalize()


if __name__ == "__main__":
    unittest.main()
