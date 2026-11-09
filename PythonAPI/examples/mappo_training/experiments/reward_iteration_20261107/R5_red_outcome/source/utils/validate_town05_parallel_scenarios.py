"""Offline geometry and balance checks for the isolated Town05 pilot pool.

Run from ``PythonAPI/examples/mappo_training``::

    python -m utils.validate_town05_parallel_scenarios

This validator is intentionally separate from the Town02 validator because a
legal side-by-side start has only one lane width of centre-to-centre spacing.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import carla


DIRECTION_TAGS = {"northbound", "southbound", "eastbound", "westbound"}


def _location(config: Dict) -> carla.Location:
    return carla.Location(
        x=float(config["x"]),
        y=float(config["y"]),
        z=float(config.get("z", 0.5)),
    )


def _distance(left: carla.Location, right: carla.Location) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _yaw_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _inside_goal(location: carla.Location, goal: Dict, margin: float = 0.05) -> bool:
    yaw = math.radians(float(goal["yaw"]))
    dx = location.x - float(goal["x"])
    dy = location.y - float(goal["y"])
    along = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return (
        abs(along) <= float(goal["half_length"]) + margin
        and abs(lateral) <= float(goal["half_width"]) + margin
    )


def _legal_lateral_neighbours(waypoint: carla.Waypoint) -> Iterable[carla.Waypoint]:
    candidates: List[Tuple[carla.LaneChange, carla.Waypoint]] = []
    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    if left is not None:
        candidates.append((carla.LaneChange.Left, left))
    if right is not None:
        candidates.append((carla.LaneChange.Right, right))
    for required_change, candidate in candidates:
        if not (waypoint.lane_change & required_change):
            continue
        if candidate.lane_type != carla.LaneType.Driving:
            continue
        if _yaw_error(
            waypoint.transform.rotation.yaw,
            candidate.transform.rotation.yaw,
        ) > 10.0:
            continue
        yield candidate


def _route_reaches_zone(start: carla.Waypoint, goal: Dict) -> bool:
    """Search forward topology and legal same-direction lane changes."""
    frontier = deque([(start, 0.0, 0)])
    visited = set()
    while frontier:
        current, travelled, lane_changes = frontier.popleft()
        if travelled > 90.0:
            continue
        if _inside_goal(current.transform.location, goal):
            return True
        key = (
            current.road_id,
            current.section_id,
            current.lane_id,
            round(current.s, 1),
            lane_changes,
        )
        if key in visited:
            continue
        visited.add(key)
        for candidate in current.next(1.0):
            frontier.append((candidate, travelled + 1.0, lane_changes))
        if lane_changes < 1:
            for candidate in _legal_lateral_neighbours(current):
                frontier.append((candidate, travelled, lane_changes + 1))
    return False


def _waypoint(road_map: carla.Map, config: Dict) -> carla.Waypoint:
    return road_map.get_waypoint(
        _location(config),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )


def validate_pool(
    scenario_dir: Path,
    xodr_path: Path,
    *,
    expected_count: int | None = None,
    expected_difficulty_count: int | None = None,
    expected_direction_count: int | None = None,
    expected_family_counts: Dict[str, int] | None = None,
    require_validation_holdout: bool | None = None,
) -> List[str]:
    errors: List[str] = []
    road_map = carla.Map("Town05", xodr_path.read_text(encoding="utf-8"))
    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        return [f"No scenario JSON files in {scenario_dir}"]

    difficulty_counts: Counter = Counter()
    family_counts: Counter = Counter()
    direction_counts: Counter = Counter()
    role_counts: Counter = Counter()

    for path in files:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{path.name}: invalid JSON: {error}")
            continue
        prefix = path.name
        if config.get("schema_version") != 3:
            errors.append(f"{prefix}: schema_version must be 3")
        if config.get("town") != "Town05":
            errors.append(f"{prefix}: town must be Town05")

        try:
            difficulty_counts[int(config["difficulty"])] += 1
        except (KeyError, TypeError, ValueError):
            errors.append(f"{prefix}: invalid difficulty")
        family = config.get("family")
        goal_mode = config.get("goal_mode")
        if not isinstance(family, str) or not family:
            errors.append(f"{prefix}: missing family")
        else:
            family_counts[family] += 1

        tags = config.get("tags", [])
        directions = set(tags) & DIRECTION_TAGS if isinstance(tags, list) else set()
        if len(directions) != 1:
            errors.append(f"{prefix}: exactly one direction tag is required")
        else:
            direction_counts[next(iter(directions))] += 1
        for role_tag in (
            "agent0_left", "agent1_left", "agent0_rear", "agent1_rear",
            "agent0_merge", "agent1_merge",
        ):
            if role_tag in tags:
                role_counts[role_tag] += 1

        if require_validation_holdout is True and "validation_holdout" not in tags:
            errors.append(f"{prefix}: held-out pool requires validation_holdout tag")
        if require_validation_holdout is False and "validation_holdout" in tags:
            errors.append(f"{prefix}: training pool cannot contain validation_holdout tag")

        goal = config.get("goal_zone")
        agents = config.get("agents")
        if not isinstance(goal, dict):
            errors.append(f"{prefix}: missing goal_zone")
            continue
        if not isinstance(agents, dict) or set(agents) != {"agent0", "agent1"}:
            errors.append(f"{prefix}: agents must contain exactly agent0 and agent1")
            continue
        if float(goal.get("half_length", 0.0)) <= 0.0 or float(goal.get("half_width", 0.0)) <= 0.0:
            errors.append(f"{prefix}: goal dimensions must be positive")

        starts: Dict[str, carla.Location] = {}
        waypoints: Dict[str, carla.Waypoint] = {}
        for agent_name, agent in agents.items():
            try:
                speed = float(agent["initial_speed_mps"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{prefix}/{agent_name}: invalid initial_speed_mps")
                continue
            if not math.isfinite(speed) or not 0.0 <= speed <= 15.0:
                errors.append(f"{prefix}/{agent_name}: initial speed outside [0, 15] m/s")
            location = _location(agent)
            waypoint = _waypoint(road_map, agent)
            starts[agent_name] = location
            waypoints[agent_name] = waypoint
            offset = _distance(location, waypoint.transform.location)
            if offset > 0.20:
                errors.append(f"{prefix}/{agent_name}: lane-centre offset {offset:.3f}m > 0.20m")
            yaw_error = _yaw_error(
                float(agent.get("yaw", waypoint.transform.rotation.yaw)),
                waypoint.transform.rotation.yaw,
            )
            if yaw_error > 2.0:
                errors.append(f"{prefix}/{agent_name}: yaw error {yaw_error:.2f}deg > 2deg")
            forward = waypoint.transform.get_forward_vector()
            to_goal = (
                float(goal["x"]) - location.x,
                float(goal["y"]) - location.y,
            )
            if to_goal[0] * forward.x + to_goal[1] * forward.y <= 0.0:
                errors.append(f"{prefix}/{agent_name}: goal is not ahead")
            if not _route_reaches_zone(waypoint, goal):
                errors.append(f"{prefix}/{agent_name}: no legal route reaches goal zone")

        if set(waypoints) != {"agent0", "agent1"}:
            continue
        wp0, wp1 = waypoints["agent0"], waypoints["agent1"]
        if wp0.road_id != wp1.road_id:
            errors.append(f"{prefix}: both starts must remain on one road segment")
        if wp0.lane_id == wp1.lane_id:
            errors.append(f"{prefix}: agents must start in different lanes")
        if _yaw_error(wp0.transform.rotation.yaw, wp1.transform.rotation.yaw) > 2.0:
            errors.append(f"{prefix}: lanes are not same-direction")
        separation = _distance(starts["agent0"], starts["agent1"])
        if separation < 3.20:
            errors.append(f"{prefix}: unsafe centre separation {separation:.2f}m < 3.20m")

        if family == "required_merge":
            if goal_mode is not None:
                errors.append(f"{prefix}: required_merge must use the shared target lane")
            if "required_merge" not in tags:
                errors.append(f"{prefix}: required_merge family is missing its tag")
            if float(goal["half_width"]) > 1.75:
                errors.append(f"{prefix}: merge goal is too wide to require one target lane")
        elif float(goal["half_width"]) < 3.25:
            errors.append(f"{prefix}: non-merge goal does not cover both lane centres")
        elif goal_mode != "preserve_start_lane":
            errors.append(f"{prefix}: non-merge family must preserve each start lane")

    if expected_count is not None and len(files) != expected_count:
        errors.append(f"scenario count mismatch: {len(files)} != {expected_count}")
    if expected_difficulty_count is None:
        expected_difficulty_count = len(files) // 4
    expected_difficulties = {
        difficulty: expected_difficulty_count for difficulty in (1, 2, 3, 4)
    }
    if dict(difficulty_counts) != expected_difficulties:
        errors.append(f"difficulty balance mismatch: {dict(difficulty_counts)}")
    if expected_family_counts is None:
        expected_family_counts = {
            "parallel_stationary": 2,
            "parallel_rolling": 2,
            "parallel_stagger": 2,
            "required_merge": 2,
        }
    if dict(family_counts) != expected_family_counts:
        errors.append(f"family balance mismatch: {dict(family_counts)}")
    if expected_direction_count is None:
        expected_direction_count = len(files) // 4
    expected_directions = {
        direction: expected_direction_count
        for direction in ("northbound", "southbound", "eastbound", "westbound")
        if expected_direction_count > 0
    }
    if len(files) == 8 and set(direction_counts) == {"northbound", "southbound"}:
        expected_directions = {"northbound": 4, "southbound": 4}
    if dict(direction_counts) != expected_directions:
        errors.append(f"direction balance mismatch: {dict(direction_counts)}")
    for left, right in (
        ("agent0_left", "agent1_left"),
        ("agent0_rear", "agent1_rear"),
        ("agent0_merge", "agent1_merge"),
    ):
        if role_counts[left] != role_counts[right]:
            errors.append(f"role mirror mismatch: {left}={role_counts[left]}, {right}={role_counts[right]}")

    print(
        f"Town05 pilot: files={len(files)}, "
        f"difficulty={dict(sorted(difficulty_counts.items()))}, "
        f"family={dict(sorted(family_counts.items()))}, "
        f"direction={dict(sorted(direction_counts.items()))}, "
        f"roles={dict(sorted(role_counts.items()))}"
    )
    return errors


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[4]
    training_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=training_root / "data" / "training_scenarios_town05_v1",
    )
    parser.add_argument(
        "--xodr",
        type=Path,
        default=repo_root / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / "Town05.xodr",
    )
    parser.add_argument("--expected-count", type=int, default=16)
    parser.add_argument("--validation-pool", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.expected_count == 16:
        family_counts = {
            "parallel_stationary": 4,
            "parallel_rolling": 4,
            "stagger_rear_closing": 4,
            "required_merge": 4,
        }
    elif args.expected_count == 8 and args.validation_pool:
        family_counts = {
            "parallel_stationary": 2,
            "parallel_rolling": 2,
            "stagger_rear_closing": 2,
            "required_merge": 2,
        }
    else:
        family_counts = None
    errors = validate_pool(
        args.scenario_dir,
        args.xodr,
        expected_count=args.expected_count,
        expected_difficulty_count=args.expected_count // 4,
        expected_direction_count=args.expected_count // 4,
        expected_family_counts=family_counts,
        require_validation_holdout=args.validation_pool,
    )
    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASSED: Town05 parallel pilot geometry and balance checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
