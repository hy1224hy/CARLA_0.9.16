"""Check that the file audit never invents applied terminal steering."""
import unittest

from diagnostics import audit_reward_v6 as audit


class V6AuditEvidenceTests(unittest.TestCase):
    def function(self):
        function = getattr(audit, "steer_cost_evidence", None)
        self.assertIsNotNone(function, "audit must distinguish applied controls from requested actions")
        return function

    def test_applied_readback_overrides_the_requested_action(self):
        row = {"post": {"0": {"control": [0, .1, 0]}},
               "environment_action": {"0": [0, .9, 0]}}
        self.assertEqual(self.function()(row, "0", .1, .1), (0, 0, 0, "applied_readback"))

    def test_missing_terminal_control_returns_bounds_not_a_fabricated_value(self):
        row = {"post": {"0": None}, "info": {
            "active_mask": {"0": False}, "reward_diagnostics": {
                "0": {"steer_rate_normalized": 1}}}}
        old, lower, upper, source = self.function()(row, "0", .1, .1)
        self.assertAlmostEqual(old, -.001)
        self.assertAlmostEqual(lower, -.1)
        self.assertAlmostEqual(upper, -.001)
        self.assertEqual(source, "terminal_clipped_risk_bounds")

    def test_unsaturated_terminal_risk_can_be_rescored_exactly(self):
        row = {"post": {"0": None}, "info": {
            "active_mask": {"0": False}, "reward_diagnostics": {
                "0": {"steer_rate_normalized": .25}}}}
        old, lower, upper, source = self.function()(row, "0", .1, .1)
        self.assertAlmostEqual(lower, -.00025)
        self.assertEqual(lower, upper)
        self.assertEqual(source, "terminal_unsaturated_risk")
