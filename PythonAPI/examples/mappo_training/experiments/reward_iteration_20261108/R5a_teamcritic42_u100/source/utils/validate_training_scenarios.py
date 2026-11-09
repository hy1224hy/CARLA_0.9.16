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
    expected_count: int | None = None,
    require_balanced_pool: bool = True,
) -> List[str]:
    errors: List[str] = []
    road_map = carla.Map(
        expected_town, xodr_path.read_text(encoding="utf-8")
    )
    files = sorted(scenario_dir.glob("*.json"))
    if not files:
        return [f"场景目录没有 JSON: {scenario_dir}"]
    if expected_count is not None and len(files) != expected_count:
        errors.append(
            f"场景数量必须为 {expected_count}，实际为 {len(files)}"
        )

    difficulty_counts: Counter = Counter()
    family_counts: Counter = Counter()
    direction_counts: Counter = Counter()
    longitudinal_rear_role_counts: Counter = Counter()
    speed_profile_counts: Counter = Counter()
    schema_versions: Counter = Counter()
    for path in files:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{path.name}: JSON 无法读取: {error}")
            continue

        prefix = path.name
        schema_version = config.get("schema_version")
        if schema_version not in {2, 3}:
            errors.append(f"{prefix}: schema_version 必须为 2 或 3")
        else:
            schema_versions[int(schema_version)] += 1
        if config.get("town") != expected_town:
            errors.append(f"{prefix}: town 必须为 {expected_town}")
        try:
            difficulty = int(config["difficulty"])
            difficulty_counts[difficulty] += 1
        except Exception:
            errors.append(f"{prefix}: 缺少合法 difficulty")

        if "episode_max_steps" in config:
            episode_max_steps = config["episode_max_steps"]
            if (
                isinstance(episode_max_steps, bool)
                or not isinstance(episode_max_steps, int)
                or episode_max_steps <= 0
            ):
                errors.append(
                    f"{prefix}: episode_max_steps 必须为正整数"
                )

        family = config.get("family")
        if not isinstance(family, str) or not family.strip():
            errors.append(f"{prefix}: 缺少合法 family")
            family = ""
        else:
            family = family.strip()
            family_counts[family] += 1

        tags = config.get("tags")
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) for tag in tags
        ):
            errors.append(f"{prefix}: tags 必须为字符串列表")
            tags = []
        direction_tags = set(tags) & {
            "eastbound", "westbound", "northbound", "southbound"
        }
        if len(direction_tags) != 1:
            errors.append(f"{prefix}: 必须且只能有一个方向 tag")
        else:
            direction_counts[next(iter(direction_tags))] += 1

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
        uses_lane_projected_goals = config.get("goal_mode") == "preserve_start_lane"
        if (
            not uses_lane_projected_goals
            and _planar_distance(goal, goal_wp.transform.location) > 0.25
        ):
            errors.append(f"{prefix}: goal_zone 未落在车道中心附近")
        if _yaw_error(
            float(goal_config["yaw"]), goal_wp.transform.rotation.yaw
        ) > 5.0:
            errors.append(f"{prefix}: goal_zone yaw 与车道方向不一致")

        start_locations: Dict[str, carla.Location] = {}
        distances_to_goal: Dict[str, float] = {}
        initial_speeds: Dict[str, float] = {}
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
            if (
                not uses_lane_projected_goals
                and family != "required_merge"
                and not _route_reaches_goal(waypoint, goal)
            ):
                errors.append(f"{prefix}/{agent_name}: 沿车道拓扑无法到达终点")

            try:
                initial_speed = float(agent_config.get("initial_speed_mps", 0.0))
            except (TypeError, ValueError):
                initial_speed = math.nan
            initial_speeds[agent_name] = initial_speed
            if not math.isfinite(initial_speed) or not 0.0 <= initial_speed <= 15.0:
                errors.append(
                    f"{prefix}/{agent_name}: initial_speed_mps 必须在 [0, 15]"
                )
            if schema_version == 3 and "initial_speed_mps" not in agent_config:
                errors.append(
                    f"{prefix}/{agent_name}: schema v3 必须显式给出 initial_speed_mps"
                )

        separation = _planar_distance(
            start_locations["agent0"], start_locations["agent1"]
        )
        same_lane = (
            road_map.get_waypoint(
                start_locations["agent0"],
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            ).lane_id
            == road_map.get_waypoint(
                start_locations["agent1"],
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            ).lane_id
        )
        minimum_separation = 8.0 if same_lane else 3.4
        if separation < minimum_separation:
            errors.append(
                f"{prefix}: 初始中心距 {separation:.2f}m < "
                f"{minimum_separation:.1f}m"
            )
        rear = max(distances_to_goal, key=distances_to_goal.get)
        if family.startswith("car_following_"):
            longitudinal_rear_role_counts[rear] += 1
        front = "agent1" if rear == "agent0" else "agent0"
        rear_speed = initial_speeds[rear]
        front_speed = initial_speeds[front]
        if rear_speed == 0.0 and front_speed == 0.0:
            speed_profile_counts["stationary"] += 1
        elif math.isclose(rear_speed, front_speed, abs_tol=1e-6):
            speed_profile_counts["same_speed_rolling"] += 1
        elif rear_speed > front_speed:
            speed_profile_counts["rear_closing"] += 1
        else:
            speed_profile_counts["front_faster"] += 1

        if "stationary_start" in tags and (rear_speed != 0.0 or front_speed != 0.0):
            errors.append(f"{prefix}: stationary_start 与初始速度不一致")
        if "rolling_start" in tags and max(rear_speed, front_speed) <= 0.0:
            errors.append(f"{prefix}: rolling_start 与初始速度不一致")
        if "same_speed" in tags and not math.isclose(
            rear_speed, front_speed, abs_tol=1e-6
        ):
            errors.append(f"{prefix}: same_speed 与初始速度不一致")
        if "rear_faster" in tags and rear_speed <= front_speed:
            errors.append(f"{prefix}: rear_faster 与前后车初始速度不一致")

    if (
        require_balanced_pool
        and len(difficulty_counts) > 1
        and len(set(difficulty_counts.values())) != 1
    ):
        errors.append(f"difficulty 场景数不均衡: {dict(difficulty_counts)}")
    if (
        longitudinal_rear_role_counts
        and longitudinal_rear_role_counts.get("agent0", 0)
        != longitudinal_rear_role_counts.get("agent1", 0)
    ):
        errors.append(
            f"跟车前后角色不均衡: {dict(longitudinal_rear_role_counts)}"
        )
    if require_balanced_pool and schema_versions and set(schema_versions) == {3}:
        if len(set(family_counts.values())) != 1:
            errors.append(f"family 场景数不均衡: {dict(family_counts)}")
        if len(direction_counts) != 4 or len(set(direction_counts.values())) != 1:
            errors.append(f"道路方向不均衡: {dict(direction_counts)}")

    print(
        f"场景数={len(files)}, difficulty={dict(sorted(difficulty_counts.items()))}, "
        f"family={dict(sorted(family_counts.items()))}, "
        f"方向={dict(sorted(direction_counts.items()))}, "
        f"跟车后车角色={dict(sorted(longitudinal_rear_role_counts.items()))}, "
        f"速度档={dict(sorted(speed_profile_counts.items()))}"
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
    parser.add_argument("--town", default="Town02")
    parser.add_argument("--expected-count", type=int, default=None)
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
    parser.add_argument(
        "--require-family",
        action="append",
        default=[],
        help="必须存在且四方向平衡的 family；可重复给出",
    )
    parser.add_argument(
        "--allow-curriculum-imbalance",
        action="store_true",
        help="允许版本化微课程的 stage/family/direction 数量不等；几何仍严格校验",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors = validate_scenario_pool(
        args.scenario_dir.resolve(),
        args.xodr.resolve(),
        expected_town=args.town,
        expected_count=args.expected_count,
        require_balanced_pool=not args.allow_curriculum_imbalance,
    )
    if not errors and args.require_family:
        family_counts: Counter = Counter()
        family_directions: Dict[str, Counter] = {}
        for path in args.scenario_dir.resolve().glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            family = str(payload.get("family", ""))
            family_counts[family] += 1
            direction = next(
                (
                    tag
                    for tag in payload.get("tags", [])
                    if tag in {
                        "eastbound",
                        "westbound",
                        "northbound",
                        "southbound",
                    }
                ),
                None,
            )
            if direction is not None:
                family_directions.setdefault(family, Counter())[direction] += 1
        for family in args.require_family:
            if family_counts[family] != 4:
                errors.append(
                    f"family {family!r} 必须有4个方向场景，实际为 {family_counts[family]}"
                )
            elif set(family_directions.get(family, {})) != {
                "eastbound",
                "westbound",
                "northbound",
                "southbound",
            }:
                errors.append(
                    f"family {family!r} 四方向不完整: "
                    f"{dict(family_directions.get(family, {}))}"
                )
    if errors:
        print("校验失败:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"{args.town} 场景池校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
