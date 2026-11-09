import unittest

from environment.road_legality_protocol import (
    DEFAULT_ROAD_LEGALITY_CONFIG,
    RoadLegalityObservation,
    RoadLegalityState,
    road_legality_protocol_spec,
    update_road_legality,
)


def _observation(**overrides):
    values = {
        "current_lane": (44, -1),
        "exact_driving": True,
        "same_reference_lane": True,
        "adjacent_to_reference": False,
        "lane_change_permitted": False,
        "current_to_reference_cosine": 1.0,
        "vehicle_to_current_cosine": 1.0,
        "speed_mps": 4.0,
        "in_junction": False,
        "broken_marking_event": False,
    }
    values.update(overrides)
    return RoadLegalityObservation(**values)


class RoadLegalityProtocolTests(unittest.TestCase):
    def test_initial_reference_and_stable_lane_are_success_eligible(self):
        initial = update_road_legality(RoadLegalityState(), _observation())
        self.assertEqual(initial.state.reference_lane, (44, -1))
        self.assertTrue(initial.success_eligible)
        stable = update_road_legality(initial.state, _observation())
        self.assertEqual(stable.classification, "reference_lane")
        self.assertFalse(stable.violation_raw)

    def test_legal_same_direction_lane_change_updates_reference(self):
        state = RoadLegalityState(reference_lane=(44, -1))
        decision = update_road_legality(
            state,
            _observation(
                current_lane=(44, -2),
                same_reference_lane=False,
                adjacent_to_reference=True,
                lane_change_permitted=True,
                broken_marking_event=True,
                current_to_reference_cosine=1.0,
            ),
        )
        self.assertEqual(decision.classification, "legal_lane_change")
        self.assertTrue(decision.legal_lane_change)
        self.assertTrue(decision.lane_deviation_exempted)
        self.assertEqual(decision.state.reference_lane, (44, -2))

    def test_broken_line_cannot_legalize_opposing_lane(self):
        state = RoadLegalityState(reference_lane=(44, -1))
        decision = update_road_legality(
            state,
            _observation(
                current_lane=(44, 1),
                same_reference_lane=False,
                adjacent_to_reference=True,
                lane_change_permitted=True,
                broken_marking_event=True,
                current_to_reference_cosine=-1.0,
            ),
        )
        self.assertEqual(decision.classification, "opposing_lane")
        self.assertTrue(decision.violation_raw)
        self.assertFalse(decision.lane_deviation_exempted)
        self.assertFalse(decision.success_eligible)

    def test_illegal_lane_change_requires_persistence(self):
        state = RoadLegalityState(reference_lane=(44, -1))
        obs = _observation(
            current_lane=(44, -2),
            same_reference_lane=False,
            adjacent_to_reference=True,
            lane_change_permitted=False,
        )
        for tick in range(1, DEFAULT_ROAD_LEGALITY_CONFIG.violation_persistence_ticks + 1):
            decision = update_road_legality(state, obs)
            state = decision.state
            self.assertEqual(state.consecutive_violation_ticks, tick)
            self.assertEqual(
                decision.violation_confirmed,
                tick >= DEFAULT_ROAD_LEGALITY_CONFIG.violation_persistence_ticks,
            )

    def test_wrong_way_requires_motion_and_persists_by_reason(self):
        state = RoadLegalityState(reference_lane=(44, -1))
        stationary = update_road_legality(
            state,
            _observation(vehicle_to_current_cosine=-1.0, speed_mps=0.0),
        )
        self.assertFalse(stationary.violation_raw)
        self.assertFalse(stationary.success_eligible)
        self.assertEqual(
            stationary.classification, "wrong_way_heading_stationary"
        )
        moving = update_road_legality(
            stationary.state,
            _observation(vehicle_to_current_cosine=-1.0, speed_mps=2.0),
        )
        self.assertEqual(moving.classification, "wrong_way_heading")
        self.assertEqual(moving.state.consecutive_violation_ticks, 1)

    def test_junction_is_not_killed_but_cannot_claim_success(self):
        state = RoadLegalityState(reference_lane=(44, -1))
        decision = update_road_legality(
            state,
            _observation(
                current_lane=(5, -3),
                same_reference_lane=False,
                current_to_reference_cosine=-1.0,
                in_junction=True,
            ),
        )
        self.assertEqual(decision.classification, "junction_ambiguous")
        self.assertFalse(decision.violation_raw)
        self.assertFalse(decision.success_eligible)
        self.assertEqual(decision.state.reference_lane, (44, -1))

    def test_offroad_is_delegated_to_existing_footprint_protocol(self):
        state = RoadLegalityState(reference_lane=(44, -1))
        decision = update_road_legality(
            state,
            _observation(current_lane=None, exact_driving=False),
        )
        self.assertEqual(decision.classification, "offroad_delegated")
        self.assertFalse(decision.violation_raw)
        self.assertFalse(decision.success_eligible)

    def test_protocol_explicitly_freezes_no_observation_change(self):
        self.assertFalse(road_legality_protocol_spec()["observation_space_change"])


if __name__ == "__main__":
    unittest.main()
