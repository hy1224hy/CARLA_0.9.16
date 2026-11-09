import unittest

from environment.offroad_protocol import (
    FOOTPRINT_INSET_FRACTION,
    OFFROAD_PERSISTENCE_TICKS,
    OffroadProtocolError,
    classify_offroad_tick,
    footprint_local_offsets,
    offroad_protocol_spec,
)


class FootprintSamplingTests(unittest.TestCase):
    def test_center_and_eight_inset_boundary_points_are_frozen(self):
        points = footprint_local_offsets(2.0, 1.0)

        self.assertEqual(len(points), 9)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertIn((2.0 * FOOTPRINT_INSET_FRACTION, 0.0), points)
        self.assertIn(
            (
                -2.0 * FOOTPRINT_INSET_FRACTION,
                -1.0 * FOOTPRINT_INSET_FRACTION,
            ),
            points,
        )
        self.assertEqual(len(set(points)), 9)

    def test_invalid_bounding_box_is_rejected(self):
        for extents in ((0.0, 1.0), (1.0, -1.0), (float("nan"), 1.0)):
            with self.subTest(extents=extents):
                with self.assertRaises(OffroadProtocolError):
                    footprint_local_offsets(*extents)


class OffroadPersistenceTests(unittest.TestCase):
    def test_all_driving_including_adjacent_lane_merge_is_legal(self):
        decision = classify_offroad_tick(
            driving_center_valid=True,
            boundary_driving_valid=[True] * 8,
            previous_consecutive_ticks=2,
        )

        self.assertFalse(decision.offroad_raw)
        self.assertEqual(decision.offroad_consecutive_ticks, 0)
        self.assertFalse(decision.offroad_confirmed)

    def test_center_outside_is_raw_even_when_boundary_is_mostly_driving(self):
        decision = classify_offroad_tick(
            driving_center_valid=False,
            boundary_driving_valid=[True] * 8,
            previous_consecutive_ticks=0,
        )

        self.assertTrue(decision.offroad_raw)
        self.assertEqual(decision.footprint_outside_count, 0)

    def test_three_of_eight_boundary_points_cross_the_material_threshold(self):
        decision = classify_offroad_tick(
            driving_center_valid=True,
            boundary_driving_valid=[False, False, False, True, True, True, True, True],
            previous_consecutive_ticks=0,
        )

        self.assertTrue(decision.offroad_raw)
        self.assertEqual(decision.footprint_outside_count, 3)
        self.assertEqual(decision.footprint_outside_fraction, 3 / 8)

    def test_one_or_two_tick_excursion_does_not_terminate_and_recovery_resets(self):
        count = 0
        for _ in range(OFFROAD_PERSISTENCE_TICKS - 1):
            decision = classify_offroad_tick(
                driving_center_valid=True,
                boundary_driving_valid=[False, False, False] + [True] * 5,
                previous_consecutive_ticks=count,
            )
            count = decision.offroad_consecutive_ticks
            self.assertFalse(decision.offroad_confirmed)

        recovered = classify_offroad_tick(
            driving_center_valid=True,
            boundary_driving_valid=[True] * 8,
            previous_consecutive_ticks=count,
        )
        self.assertEqual(recovered.offroad_consecutive_ticks, 0)

    def test_third_consecutive_tick_confirms_offroad(self):
        count = 0
        for index in range(1, OFFROAD_PERSISTENCE_TICKS + 1):
            decision = classify_offroad_tick(
                driving_center_valid=True,
                boundary_driving_valid=[False, False, False] + [True] * 5,
                previous_consecutive_ticks=count,
            )
            count = decision.offroad_consecutive_ticks
            self.assertEqual(decision.offroad_confirmed, index == 3)

    def test_machine_readable_spec_contains_all_frozen_thresholds(self):
        spec = offroad_protocol_spec()

        self.assertEqual(spec["sample_count"], 9)
        self.assertEqual(spec["outside_count_threshold"], 3)
        self.assertEqual(spec["persistence_ticks"], 3)
        self.assertEqual(spec["terminal_penalty"], -20.0)


if __name__ == "__main__":
    unittest.main()
