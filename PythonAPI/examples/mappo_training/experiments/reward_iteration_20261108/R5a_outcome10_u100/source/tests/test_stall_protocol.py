import unittest

from environment.stall_protocol import (
    StallState,
    is_complex_interaction_family,
    stall_protocol_spec,
    update_stall_diagnostic,
)


class StallDiagnosticTests(unittest.TestCase):
    def _at_step(self, *, blocker=False, complex_interaction=False):
        return update_stall_diagnostic(
            StallState(best_completion=0.1, last_progress_step=0, low_speed_ticks=99),
            current_step=100,
            speed_mps=0.0,
            route_completion=0.1,
            has_front_blocker=blocker,
            complex_interaction=complex_interaction,
        )

    def test_persistent_unblocked_stall_is_reportable_but_never_actionable(self):
        decision = self._at_step()
        self.assertTrue(decision.raw)
        self.assertTrue(decision.reportable)
        self.assertEqual(stall_protocol_spec()["effects"], {
            "reward": False,
            "termination": False,
            "curriculum": False,
        })

    def test_front_blocker_and_complex_interaction_exempt_reasonable_yielding(self):
        self.assertFalse(self._at_step(blocker=True).reportable)
        self.assertTrue(self._at_step(blocker=True).front_blocker_exempt)
        self.assertFalse(self._at_step(complex_interaction=True).reportable)
        self.assertTrue(
            self._at_step(complex_interaction=True).complex_interaction_exempt
        )

    def test_progress_resets_no_progress_clock(self):
        decision = update_stall_diagnostic(
            StallState(best_completion=0.1, last_progress_step=0, low_speed_ticks=80),
            current_step=150,
            speed_mps=0.0,
            route_completion=0.2,
            has_front_blocker=False,
            complex_interaction=False,
        )
        self.assertEqual(decision.no_progress_ticks, 0)
        self.assertFalse(decision.raw)

    def test_recent_stop_after_long_no_progress_is_not_a_persistent_stall(self):
        decision = update_stall_diagnostic(
            StallState(best_completion=0.1, last_progress_step=0, low_speed_ticks=0),
            current_step=150,
            speed_mps=0.0,
            route_completion=0.1,
            has_front_blocker=False,
            complex_interaction=False,
        )
        self.assertEqual(decision.no_progress_ticks, 150)
        self.assertEqual(decision.state.low_speed_ticks, 1)
        self.assertFalse(decision.raw)

    def test_registered_complex_families_cover_merge_and_following(self):
        self.assertTrue(is_complex_interaction_family("required_merge"))
        self.assertTrue(is_complex_interaction_family("car_following_near_gap"))
        self.assertFalse(is_complex_interaction_family("free_flow"))


if __name__ == "__main__":
    unittest.main()
