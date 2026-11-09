"""Build the comprehensive, non-adversarial Town05 base-policy scenario pool.

The generated pool keeps the frozen Town05 v1 lateral-interaction tasks and
adds longitudinal fundamentals on the same four training directions.  It is a
new versioned artifact; the v1 directory is never modified.

Run from ``PythonAPI/examples/mappo_training``:

    python -m utils.build_town05_base_scenarios_v2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import carla


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_V1_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_v1"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_v2"
DEFAULT_XODR = (
    REPOSITORY_ROOT
    / "CarlaUE4"
    / "Content"
    / "Carla"
    / "Maps"
    / "OpenDrive"
    / "Town05.xodr"
)


@dataclass(frozen=True)
class Route:
    direction: str
    road_id: int
    lane_ids: tuple[int, int]
    rear_s: float
    goal_s: float


ROUTES = (
    Route("northbound", 44, (2, 1), 53.0, 8.0),
    Route("southbound", 44, (-1, -2), 8.0, 53.0),
    Route("eastbound", 3, (1, 2), 50.0, 8.0),
    Route("westbound", 3, (-1, -2), 5.0, 50.0),
)


FAMILY_DIFFICULTY = {
    "free_flow": 1,
    "parallel_stationary": 1,
    "parallel_rolling": 2,
    "car_following_steady": 2,
    "car_following_closing": 3,
    "stagger_rear_closing": 3,
    "car_following_near_gap": 4,
    "required_merge": 4,
}


def _waypoint_index(road_map: carla.Map) -> Dict[tuple[int, int], list[carla.Waypoint]]:
    grouped: Dict[tuple[int, int], list[carla.Waypoint]] = {}
    for waypoint in road_map.generate_waypoints(0.25):
        if waypoint.lane_type != carla.LaneType.Driving:
            continue
        grouped.setdefault((waypoint.road_id, waypoint.lane_id), []).append(waypoint)
    return grouped


def _waypoint_at(
    index: Dict[tuple[int, int], list[carla.Waypoint]],
    route: Route,
    lane_id: int,
    s: float,
) -> carla.Waypoint:
    candidates = index[(route.road_id, lane_id)]
    waypoint = min(candidates, key=lambda item: abs(float(item.s) - float(s)))
    if abs(float(waypoint.s) - float(s)) > 0.2:
        raise RuntimeError(
            f"Road {route.road_id} lane {lane_id} has no waypoint near s={s}"
        )
    return waypoint


def _agent(waypoint: carla.Waypoint, speed_mps: float) -> Dict[str, float]:
    transform = waypoint.transform
    return {
        "x": round(float(transform.location.x), 2),
        "y": round(float(transform.location.y), 2),
        "z": 0.5,
        "yaw": round(float(transform.rotation.yaw), 2),
        "initial_speed_mps": float(speed_mps),
    }


def _goal(waypoint: carla.Waypoint) -> Dict[str, float]:
    transform = waypoint.transform
    return {
        "x": round(float(transform.location.x), 2),
        "y": round(float(transform.location.y), 2),
        "z": 0.5,
        "yaw": round(float(transform.rotation.yaw), 2),
        "half_length": 3.0,
        "half_width": 1.6,
    }


def _scenario(
    *,
    family: str,
    description: str,
    tags: list[str],
    agent0: Dict[str, float],
    agent1: Dict[str, float],
    goal_zone: Dict[str, float],
    agent_goal_zones: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "schema_version": 3,
        "town": "Town05",
        "difficulty": FAMILY_DIFFICULTY[family],
        "family": family,
        "goal_mode": "preserve_start_lane",
        "description": description,
        "tags": tags,
        "goal_zone": goal_zone,
        "agents": {"agent0": agent0, "agent1": agent1},
    }
    if agent_goal_zones is not None:
        payload["agent_goal_zones"] = agent_goal_zones
    return payload


def _build_added_scenarios(road_map: carla.Map) -> list[tuple[str, Dict[str, object]]]:
    index = _waypoint_index(road_map)
    output: list[tuple[str, Dict[str, object]]] = []

    for direction_index, route in enumerate(ROUTES):
        first_lane, second_lane = route.lane_ids
        # Alternate lane and agent roles across directions.  This removes the
        # shortcuts "agent0 is always the follower" and "following is always
        # tested in the left lane" without multiplying the pool unnecessarily.
        task_lane = first_lane if direction_index % 2 == 0 else second_lane
        free_lane0 = first_lane
        free_lane1 = second_lane
        agent0_is_rear = direction_index % 2 == 0

        rear_wp = _waypoint_at(index, route, task_lane, route.rear_s)
        goal_wp = _waypoint_at(index, route, task_lane, route.goal_s)
        free_wp0 = _waypoint_at(index, route, free_lane0, route.rear_s)
        free_wp1 = _waypoint_at(index, route, free_lane1, route.rear_s)

        # D1: independent two-lane cruising at different, moderate initial
        # speeds.  Agents share no forced interaction but still populate both
        # lanes so both policies receive useful rollout data.
        free_agent0 = _agent(free_wp0, 2.0 if direction_index % 2 == 0 else 4.0)
        free_agent1 = _agent(free_wp1, 4.0 if direction_index % 2 == 0 else 2.0)
        free_goal0 = _goal(_waypoint_at(index, route, free_lane0, route.goal_s))
        free_goal1 = _goal(_waypoint_at(index, route, free_lane1, route.goal_s))
        output.append((
            f"scenario_{direction_index + 17:02d}_{route.direction}_free_flow.json",
            _scenario(
                family="free_flow",
                description=f"Road {route.road_id} {route.direction} independent free-flow cruising",
                tags=[
                    route.direction,
                    "two_lane",
                    "free_flow",
                    "rolling_start",
                    "different_speed",
                    "non_adversarial",
                ],
                agent0=free_agent0,
                agent1=free_agent1,
                goal_zone=_goal(goal_wp),
                agent_goal_zones={
                    "agent0": free_goal0,
                    "agent1": free_goal1,
                },
            ),
        ))

        family_specs = (
            ("car_following_steady", 10.0, 3.0, 3.0, "steady_gap"),
            ("car_following_closing", 12.0, 5.0, 3.0, "rear_faster"),
            ("car_following_near_gap", 8.0, 3.0, 3.0, "near_gap"),
        )
        for family_offset, (family, gap_m, rear_speed, front_speed, profile_tag) in enumerate(family_specs):
            travel_sign = -1.0 if first_lane > 0 else 1.0
            front_s = route.rear_s + travel_sign * gap_m
            front_wp = _waypoint_at(index, route, task_lane, front_s)
            rear_agent = _agent(rear_wp, rear_speed)
            front_agent = _agent(front_wp, front_speed)
            agents = (
                (rear_agent, front_agent)
                if agent0_is_rear
                else (front_agent, rear_agent)
            )
            role_tag = "agent0_rear" if agent0_is_rear else "agent1_rear"
            scenario_number = 21 + family_offset * 4 + direction_index
            output.append((
                f"scenario_{scenario_number:02d}_{route.direction}_{family}_{role_tag}.json",
                _scenario(
                    family=family,
                    description=(
                        f"Road {route.road_id} {route.direction} same-lane {family}; "
                        f"initial center gap {gap_m:.0f} m"
                    ),
                    tags=[
                        route.direction,
                        "single_lane_interaction",
                        "car_following",
                        "rolling_start",
                        profile_tag,
                        role_tag,
                        "non_adversarial",
                    ],
                    agent0=agents[0],
                    agent1=agents[1],
                    goal_zone=_goal(goal_wp),
                ),
            ))

    return output


def build(v1_dir: Path, output_dir: Path, xodr_path: Path) -> None:
    v1_files = sorted(v1_dir.glob("scenario_*.json"))
    if len(v1_files) != 16:
        raise RuntimeError(f"Expected 16 frozen v1 scenarios, found {len(v1_files)}")
    road_map = carla.Map("Town05", xodr_path.read_text(encoding="utf-8"))
    added = _build_added_scenarios(road_map)

    expected_names = {path.name for path in v1_files} | {name for name, _ in added}
    existing_names = {path.name for path in output_dir.glob("*.json")}
    stale_names = sorted(existing_names - expected_names)
    if stale_names:
        raise RuntimeError(
            "Refusing to overwrite a v2 directory with unexpected JSON files: "
            + ", ".join(stale_names)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in v1_files:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload["family"] != "required_merge":
            payload["goal_mode"] = "preserve_start_lane"
        destination = output_dir / source.name
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for name, payload in added:
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    family_counts: Dict[str, int] = {}
    difficulty_counts: Dict[int, int] = {}
    for path in sorted(output_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        family = str(payload["family"])
        difficulty = int(payload["difficulty"])
        family_counts[family] = family_counts.get(family, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    if len(family_counts) != 8 or set(family_counts.values()) != {4}:
        raise RuntimeError(f"Unbalanced v2 family matrix: {family_counts}")
    if difficulty_counts != {1: 8, 2: 8, 3: 8, 4: 8}:
        raise RuntimeError(f"Unbalanced v2 difficulty matrix: {difficulty_counts}")
    print(
        f"Built {sum(family_counts.values())} scenarios in {output_dir}\n"
        f"families={family_counts}\n"
        f"difficulties={difficulty_counts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    args = parser.parse_args()
    build(args.v1_dir.resolve(), args.output_dir.resolve(), args.xodr.resolve())


if __name__ == "__main__":
    main()
