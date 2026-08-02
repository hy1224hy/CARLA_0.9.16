"""离线校验训练场景是否贴合 CARLA OpenDRIVE 可行驶车道。

从 ``mappo_training`` 目录运行：

    python -m utils.validate_training_scenarios
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import carla


def _yaw_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _planar_distance(left: carla.Location, right: carla.Location) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _as_location(config: Dict) -> carla.Location:
    return carla.Location(
        x=float(config["x"]),
        y=float(config["y"]),
        z=float(config.get("z", 0.5)),
    )


def _route_reaches_goal(
    start: carla.Waypoint,
    goal: carla.Location,
    max_distance: float = 180.0,
) -> bool:
    frontier = [(start, 0.0)]
    visited = set()
    while frontier:
        current, travelled = frontier.pop(0)
        if travelled > max_distance:
            continue
        if _planar_distance(current.transform.location, goal) <= 3.0:
            return True
        key = (
            current.road_id,
            current.section_id,
            current.lane_id,
            round(current.s, 1),
        )
        if key in visited:
            continue
        visited.add(key)
        frontier.extend(
            (candidate, travelled + 2.0)
            for candidate in current.next(2.0)
        )
    return False


def validate_scenario_pool(
    scenario_dir: Path,
    xodr_path: Path,
    expected_town: str = "Town02",
) -> List[str]:
    errors: List[str] = []
    road_map = carla.Map(
        expected_town, xodr_path.read_text(encoding="utf-8")
    )
    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        return [f"场景目录没有 JSON: {scenario_dir}"]

    difficulty_counts: Counter = Counter()
    rear_role_counts: Counter = Counter()
    for path in files:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{path.name}: JSON 无法读取: {error}")
            continue

        prefix = path.name
        if config.get("schema_version") != 2:
            errors.append(f"{prefix}: schema_version 必须为 2")
        if config.get("town") != expected_town:
            errors.append(f"{prefix}: town 必须为 {expected_town}")
        try:
            difficulty = int(config["difficulty"])
            difficulty_counts[difficulty] += 1
        except Exception:
            errors.append(f"{prefix}: 缺少合法 difficulty")

        goal_config = config.get("goal_zone")
        agents = config.get("agents")
        if not isinstance(goal_config, dict):
            errors.append(f"{prefix}: 缺少 goal_zone")
            continue
        if not isinstance(agents, dict) or set(agents) != {"agent0", "agent1"}:
            errors.append(f"{prefix}: agents 必须且只能包含 agent0/agent1")
            continue

        goal = _as_location(goal_config)
        goal_wp = road_map.get_waypoint(
            goal, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if _planar_distance(goal, goal_wp.transform.location) > 0.25:
            errors.append(f"{prefix}: goal_zone 未落在车道中心附近")
        if _yaw_error(
            float(goal_config["yaw"]), goal_wp.transform.rotation.yaw
        ) > 5.0:
            errors.append(f"{prefix}: goal_zone yaw 与车道方向不一致")

        start_locations: Dict[str, carla.Location] = {}
        distances_to_goal: Dict[str, float] = {}
        for agent_name, agent_config in agents.items():
            location = _as_location(agent_config)
            start_locations[agent_name] = location
            distances_to_goal[agent_name] = _planar_distance(location, goal)
            waypoint = road_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            lateral_offset = _planar_distance(
                location, waypoint.transform.location
            )
            if lateral_offset > 0.75:
                errors.append(
                    f"{prefix}/{agent_name}: 横向偏移 {lateral_offset:.2f}m > 0.75m"
                )
            yaw = float(agent_config.get("yaw", waypoint.transform.rotation.yaw))
            if _yaw_error(yaw, waypoint.transform.rotation.yaw) > 5.0:
                errors.append(
                    f"{prefix}/{agent_name}: yaw 与行驶车道方向不一致"
                )
            forward = waypoint.transform.get_forward_vector()
            to_goal_x = goal.x - location.x
            to_goal_y = goal.y - location.y
            if to_goal_x * forward.x + to_goal_y * forward.y <= 0.0:
                errors.append(f"{prefix}/{agent_name}: 终点不在车辆前方")
            if not _route_reaches_goal(waypoint, goal):
                errors.append(f"{prefix}/{agent_name}: 沿车道拓扑无法到达终点")

        separation = _planar_distance(
            start_locations["agent0"], start_locations["agent1"]
        )
        if separation < 8.0:
            errors.append(f"{prefix}: 初始中心距 {separation:.2f}m < 8m")
        rear = max(distances_to_goal, key=distances_to_goal.get)
        rear_role_counts[rear] += 1

    if len(difficulty_counts) > 1 and len(set(difficulty_counts.values())) != 1:
        errors.append(f"difficulty 场景数不均衡: {dict(difficulty_counts)}")
    if rear_role_counts.get("agent0", 0) != rear_role_counts.get("agent1", 0):
        errors.append(f"前后角色不均衡: {dict(rear_role_counts)}")

    print(
        f"场景数={len(files)}, difficulty={dict(sorted(difficulty_counts.items()))}, "
        f"后车角色={dict(sorted(rear_role_counts.items()))}"
    )
    return errors


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[4]
    training_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=training_root / "data" / "training_scenarios_town02_v2",
    )
    parser.add_argument(
        "--xodr",
        type=Path,
        default=(
            repo_root
            / "CarlaUE4"
            / "Content"
            / "Carla"
            / "Maps"
            / "OpenDrive"
            / "Town02.xodr"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors = validate_scenario_pool(
        args.scenario_dir.resolve(), args.xodr.resolve()
    )
    if errors:
        print("校验失败:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Town02 场景池校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
