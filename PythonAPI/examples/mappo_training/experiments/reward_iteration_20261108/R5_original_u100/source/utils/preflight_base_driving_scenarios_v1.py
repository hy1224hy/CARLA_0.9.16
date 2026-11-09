"""Read-only map preflight for the new base-driving pools.

The default is strictly offline and never imports CARLA. ``--live`` acquires
the shared CARLA owner lock, checks an idle 0.9.16 server, and queries the
existing Town05 map. It never spawns actors, advances the clock, loads a map,
or changes world settings. Actual vehicle bounding boxes and initial-speed
commit still require a separate serial environment smoke before long runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from training.base_driving_scenarios_v1 import load_base_driving_manifest
from utils.build_base_driving_scenarios_v1 import (
    DEFAULT_OUTPUT_DIR,
    ROUTE_LENGTH_M,
    build_payloads,
    validate_payloads,
)


PREFLIGHT_PROTOCOL = "town05_base_driving_read_only_geometry_v1"
EXPECTED_ROAD = {"northbound": 44, "southbound": 44, "eastbound": 3, "westbound": 3}
# Physical left/right in CARLA, rather than the inconsistent historical tags.
EXPECTED_LANES = {
    "northbound": {"left": 1, "right": 2},
    "southbound": {"left": -1, "right": -2},
    "eastbound": {"left": 1, "right": 2},
    "westbound": {"left": -1, "right": -2},
}
ENVELOPE_HALF_LENGTH_M = 3.0
ENVELOPE_HALF_WIDTH_M = 1.3
ENVELOPE_CLEARANCE_M = 0.1


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def offline_validate(scenario_dir: Path) -> tuple[dict, dict[str, dict]]:
    """Fail on generated-pool, complete-pair, geometry, or manifest drift."""
    root = Path(scenario_dir)
    manifest = load_base_driving_manifest(root)
    payloads = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in manifest["scenario_sha256"]
    }
    validate_payloads(payloads)
    expected = build_payloads(split=manifest["split"])
    if payloads != expected:
        raise ValueError("pool differs from the frozen generated base-driving matrix")
    if manifest["scenario_count"] != len(payloads):
        raise ValueError("manifest scenario_count mismatch")
    return manifest, payloads


def envelope_corners(start: Mapping[str, float]) -> list[tuple[float, float]]:
    """Conservative *configured* OBB, not an observed actor bounding box."""
    yaw = math.radians(float(start["yaw"]))
    forward = (math.cos(yaw), math.sin(yaw))
    right = (-forward[1], forward[0])
    return [
        (
            float(start["x"]) + longitudinal * forward[0] + lateral * right[0],
            float(start["y"]) + longitudinal * forward[1] + lateral * right[1],
        )
        for longitudinal in (-ENVELOPE_HALF_LENGTH_M, ENVELOPE_HALF_LENGTH_M)
        for lateral in (-ENVELOPE_HALF_WIDTH_M, ENVELOPE_HALF_WIDTH_M)
    ]


def envelopes_overlap(first: Mapping[str, float], second: Mapping[str, float]) -> bool:
    """Separating-axis check including a small reserved clearance on each OBB."""
    boxes = (envelope_corners(first), envelope_corners(second))
    for start in (first, second):
        yaw = math.radians(float(start["yaw"]))
        for axis in ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))):
            projections = [[x * axis[0] + y * axis[1] for x, y in box] for box in boxes]
            if max(projections[0]) + ENVELOPE_CLEARANCE_M <= min(projections[1]) - ENVELOPE_CLEARANCE_M:
                return False
            if max(projections[1]) + ENVELOPE_CLEARANCE_M <= min(projections[0]) - ENVELOPE_CLEARANCE_M:
                return False
    return True


def _heading_error(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def validate_map_geometry(road_map, carla_api, payloads: Mapping[str, dict]) -> dict:
    """Query all start/route/goal centers and conservative envelope corners."""
    if str(road_map.name).replace("\\", "/").rsplit("/", 1)[-1] != "Town05":
        raise ValueError(f"preflight requires existing Town05, found {road_map.name!r}")
    rows, errors = {}, []
    for name, payload in payloads.items():
        scenario_errors, per_agent = [], {}
        expected_road = EXPECTED_ROAD[payload["direction"]]
        agents = payload["agents"]
        if envelopes_overlap(agents["agent0"], agents["agent1"]):
            scenario_errors.append("initial conservative vehicle envelopes overlap")
        for aid in ("agent0", "agent1"):
            start, goal = agents[aid], payload["agent_goal_zones"][aid]
            lane_role = str(payload["agent_roles"][aid]).split("_", 1)[0]
            expected_lane = EXPECTED_LANES[payload["direction"]][lane_role]
            max_center_error = max_heading_error = 0.0
            sampled = 0
            agent_errors = []

            def check_point(x: float, y: float, *, context: str, centered: bool):
                nonlocal max_center_error, max_heading_error, sampled
                point = carla_api.Location(x=x, y=y, z=float(start["z"]))
                # Restrict the nearest-lane search to Driving, as the runtime
                # road-occupancy protocol does. With LaneType.Any, CARLA may
                # choose the centre of a narrow shoulder outside the point,
                # then return None even while this point is inside Driving.
                # Keep exact membership: never project an off-road point back.
                waypoint = road_map.get_waypoint(point, project_to_road=False, lane_type=carla_api.LaneType.Driving)
                sampled += 1
                if waypoint is None:
                    agent_errors.append(f"{context}: outside any road lane")
                    return
                if waypoint.lane_type != carla_api.LaneType.Driving or waypoint.is_junction:
                    agent_errors.append(f"{context}: requires non-junction Driving lane")
                if (int(waypoint.road_id), int(waypoint.lane_id)) != (expected_road, expected_lane):
                    agent_errors.append(f"{context}: road/lane {waypoint.road_id}/{waypoint.lane_id} != {expected_road}/{expected_lane}")
                heading_error = _heading_error(start["yaw"], waypoint.transform.rotation.yaw)
                max_heading_error = max(max_heading_error, heading_error)
                if heading_error > 2.0:
                    agent_errors.append(f"{context}: heading error {heading_error:.3f}deg > 2deg")
                if centered:
                    location = waypoint.transform.location
                    lateral_error = math.hypot(float(location.x) - x, float(location.y) - y)
                    max_center_error = max(max_center_error, lateral_error)
                    if lateral_error > 0.20:
                        agent_errors.append(f"{context}: lane-center distance {lateral_error:.3f}m > 0.20m")

            route_length = math.hypot(goal["x"] - start["x"], goal["y"] - start["y"])
            if not math.isclose(route_length, ROUTE_LENGTH_M, abs_tol=1e-6):
                agent_errors.append(f"individual route length {route_length}m != {ROUTE_LENGTH_M}m")
            for index in range(33):
                alpha = index / 32.0
                check_point(start["x"] + alpha * (goal["x"] - start["x"]), start["y"] + alpha * (goal["y"] - start["y"]), context=f"route_{index:02d}", centered=True)
            for index, (x, y) in enumerate(envelope_corners(start)):
                check_point(x, y, context=f"start_envelope_corner_{index}", centered=False)
            per_agent[aid] = {
                "road_id": expected_road, "lane_id": expected_lane,
                "route_length_m": route_length, "point_queries": sampled,
                "max_lane_center_error_m": max_center_error,
                "max_heading_error_deg": max_heading_error,
                "errors": agent_errors,
            }
            scenario_errors.extend(f"{aid}: {error}" for error in agent_errors)
        rows[name] = {"agents": per_agent, "configured_spawn_envelopes_nonoverlap": not envelopes_overlap(agents["agent0"], agents["agent1"]), "errors": scenario_errors}
        errors.extend(f"{name}: {error}" for error in scenario_errors)
    return {"scenarios": rows, "errors": errors, "map_name": str(road_map.name)}


def run_preflight(scenario_dir: Path, *, live: bool = False, host: str = "127.0.0.1", port: int = 2000) -> dict:
    """Return an evidence record; only explicit live=True imports CARLA."""
    root = Path(scenario_dir).resolve()
    report = {
        "protocol_version": PREFLIGHT_PROTOCOL,
        "status": "FAILED", "started_at": _now(), "scenario_dir": str(root),
        "manifest_sha256": None, "preflight_source_sha256": _sha(Path(__file__)),
        "live_requested": bool(live), "errors": [],
        "clock_operations": 0, "spawn_operations": 0,
        "actor_model": "vehicle.tesla.model3",
        "spawn_envelope": {
            "half_length_m": ENVELOPE_HALF_LENGTH_M, "half_width_m": ENVELOPE_HALF_WIDTH_M,
            "reserved_clearance_m": ENVELOPE_CLEARANCE_M,
            "evidence_type": "configured_conservative_envelope_not_actual_actor_bbox",
            "assumption": "actual model footprint including local bbox offset fits inside this centered envelope",
        },
        "not_verified": ["actual actor spawn/bounding_box", "committed initial speed", "full-episode feasibility", "learned driving competence"],
    }
    try:
        report["manifest_sha256"] = _sha(root / "protocol" / "manifest.json")
        manifest, payloads = offline_validate(root)
        report["split"] = manifest["split"]
        report["scenario_count"] = len(payloads)
        report["offline_invariants"] = "PASS"
        if live:
            # No simulator/client imports or process audit occur on the offline path.
            import carla
            from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
            from utils.mut_mappo_session import CarlaOwner

            with CarlaOwner(host=host, port=port, purpose="base_driving_read_only_geometry_preflight"):
                report["idle_world"] = _preflight_idle_world(host=host, port=port, expected_version="0.9.16")
                client = carla.Client(host, port)
                client.set_timeout(10.0)
                world = client.get_world()
                report["geometry"] = validate_map_geometry(world.get_map(), carla, payloads)
                report["errors"].extend(report["geometry"]["errors"])
                report["idle_world_after"] = _preflight_idle_world(host=host, port=port, expected_version="0.9.16")
            if not report["errors"]:
                report["status"] = "PASS_GEOMETRY_ONLY"
        else:
            report["status"] = "OFFLINE_VALID_LIVE_PENDING"
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["ended_at"] = _now()
    return report


def require_geometry_preflight(scenario_dir: Path, report_path: Path) -> dict:
    """Pure-file launch guard: pending, stale and partial reports are rejected."""
    manifest, payloads = offline_validate(Path(scenario_dir))
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    expected = {
        "protocol_version": PREFLIGHT_PROTOCOL,
        "status": "PASS_GEOMETRY_ONLY", "live_requested": True,
        "manifest_sha256": _sha(Path(scenario_dir) / "protocol" / "manifest.json"),
        "preflight_source_sha256": _sha(Path(__file__)),
        "clock_operations": 0, "spawn_operations": 0,
        "scenario_count": manifest["scenario_count"],
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    geometry = report.get("geometry", {})
    rows = geometry.get("scenarios", {})
    if set(rows) != set(payloads) or geometry.get("errors"):
        mismatches.append("complete_geometry_scenario_evidence")
    for name, row in rows.items():
        if row.get("errors") or row.get("configured_spawn_envelopes_nonoverlap") is not True:
            mismatches.append(f"geometry.{name}")
        for aid in ("agent0", "agent1"):
            evidence = row.get("agents", {}).get(aid, {})
            if evidence.get("errors") or evidence.get("point_queries") != 37:
                mismatches.append(f"geometry.{name}.{aid}.point_evidence")
    if report.get("errors") or mismatches:
        raise ValueError(f"valid live geometry preflight required; mismatch={mismatches}, errors={report.get('errors')}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, required=True, help="Separate evidence JSON, outside the frozen scenario directory")
    parser.add_argument("--live", action="store_true", help="Query existing idle CARLA under its exclusive lock; no spawn or tick")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    root, destination = args.scenario_dir.resolve(), args.output.resolve()
    if root == destination or root in destination.parents:
        parser.error("--output must be outside the frozen scenario directory")
    report = run_preflight(root, live=args.live, host=args.host, port=args.port)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"{report['status']}: {destination}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
