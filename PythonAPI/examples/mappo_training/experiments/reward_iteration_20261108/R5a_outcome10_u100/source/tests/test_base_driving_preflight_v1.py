import builtins
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.build_base_driving_scenarios_v1 import DEFAULT_OUTPUT_DIR, build_payloads
from utils.preflight_base_driving_scenarios_v1 import (
    EXPECTED_LANES,
    envelopes_overlap,
    require_geometry_preflight,
    run_preflight,
    validate_map_geometry,
)


class _FakeMap:
    """Only a straight-line map query stub; no server/clock/spawn interfaces."""
    name = "Town05"

    def __init__(self, payload, *, wrong_lane=False, junction=False):
        self.payload, self.wrong_lane, self.junction = payload, wrong_lane, junction

    def get_waypoint(self, point, *, project_to_road, lane_type):
        self.assert_read_only(project_to_road)
        if lane_type != 1:
            raise AssertionError("exact road footprint must search Driving, excluding shoulder centres")
        starts = self.payload["agents"]
        yaw = math.radians(starts["agent0"]["yaw"])
        forward, right = (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))
        nearest = min(starts, key=lambda aid: abs((point.x - starts[aid]["x"]) * right[0] + (point.y - starts[aid]["y"]) * right[1]))
        start = starts[nearest]
        along = (point.x - start["x"]) * forward[0] + (point.y - start["y"]) * forward[1]
        location = SimpleNamespace(x=start["x"] + along * forward[0], y=start["y"] + along * forward[1])
        role = self.payload["agent_roles"][nearest].split("_", 1)[0]
        lane = EXPECTED_LANES[self.payload["direction"]][role]
        return SimpleNamespace(
            road_id=44, lane_id=99 if self.wrong_lane else lane, lane_type=1,
            is_junction=self.junction,
            transform=SimpleNamespace(location=location, rotation=SimpleNamespace(yaw=start["yaw"])),
        )

    @staticmethod
    def assert_read_only(project):
        if project:
            raise AssertionError("off-road points must not be projected back to Driving")


_FAKE_API = SimpleNamespace(Location=lambda **kwargs: SimpleNamespace(**kwargs), LaneType=SimpleNamespace(Any=0, Driving=1))


class BaseDrivingPreflightTests(unittest.TestCase):
    def test_westbound_lane_roles_match_measured_town05_map(self):
        # Map probe 2026-10-20: physical left y=-0.434 => Road3/-1;
        # physical right y=-3.934 => Road3/-2. Historical tags were reversed.
        self.assertEqual(EXPECTED_LANES["westbound"], {"left": -1, "right": -2})

    def test_offline_default_never_imports_carla_and_is_not_launch_qualified(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "carla" or name.startswith("carla."):
                raise AssertionError("offline validation must not import CARLA")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            report = run_preflight(DEFAULT_OUTPUT_DIR)
        self.assertEqual(report["status"], "OFFLINE_VALID_LIVE_PENDING")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["clock_operations"], 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preflight.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid live geometry"):
                require_geometry_preflight(DEFAULT_OUTPUT_DIR, path)

    def test_configured_vehicle_envelope_detects_overlap_without_spawning(self):
        payload = next(iter(build_payloads().values()))
        first, second = payload["agents"].values()
        self.assertFalse(envelopes_overlap(first, second))
        self.assertTrue(envelopes_overlap(first, first))
        near = copy.deepcopy(first)
        near["x"] += 0.5
        self.assertTrue(envelopes_overlap(first, near))

    def test_map_query_covers_routes_and_rejects_wrong_lane_or_junction(self):
        name, payload = next((name, payload) for name, payload in build_payloads().items() if payload["direction"] == "northbound")
        report = validate_map_geometry(_FakeMap(payload), _FAKE_API, {name: payload})
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["scenarios"][name]["agents"]["agent0"]["point_queries"], 37)
        for parameters in ({"wrong_lane": True}, {"junction": True}):
            broken = validate_map_geometry(_FakeMap(payload, **parameters), _FAKE_API, {name: payload})
            self.assertTrue(broken["errors"])

    def test_gate_rejects_partial_geometry_and_changed_manifest_binding(self):
        report = run_preflight(DEFAULT_OUTPUT_DIR)
        report.update(status="PASS_GEOMETRY_ONLY", live_requested=True, geometry={"scenarios": {}, "errors": []})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preflight.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete_geometry_scenario_evidence"):
                require_geometry_preflight(DEFAULT_OUTPUT_DIR, path)
            report["manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                require_geometry_preflight(DEFAULT_OUTPUT_DIR, path)


if __name__ == "__main__":
    unittest.main()
