"""在运行中的 CARLA 上逐个验证训练场景的出生位置和初始速度。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from environment.carla_multi_agent_env import CarlaMultiAgentEnv


def _speed_mps(vehicle) -> float:
    velocity = vehicle.get_velocity()
    # 场景中的 initial_speed_mps 定义的是路面平面内的行驶速度；
    # 不应将车辆刚体的微小竖直震动计入验证值。
    return math.hypot(velocity.x, velocity.y)


def run_live_smoke(args: argparse.Namespace) -> int:
    files = sorted(args.scenario_dir.resolve().glob("*.json"))
    if not files:
        raise RuntimeError(f"场景目录没有 JSON: {args.scenario_dir}")

    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=2,
        max_steps=1,
        spawn_vehicles=False,
        disable_tensorboard=True,
        seed=args.seed,
        follow_spectator=args.follow_spectator,
        lidar_range_m=args.lidar_range_m,
        ttc_max_range_m=args.ttc_max_range_m,
    )
    failures = []
    try:
        for path in files:
            config = json.loads(path.read_text(encoding="utf-8"))
            env.set_scenario(config)
            observations_by_agent, info = env.reset()
            observations = []
            for agent_id, vehicle in enumerate(env.vehicles):
                expected = config["agents"][f"agent{agent_id}"]
                transform = vehicle.get_transform()
                location_error = math.hypot(
                    transform.location.x - float(expected["x"]),
                    transform.location.y - float(expected["y"]),
                )
                actual_speed = _speed_mps(vehicle)
                expected_speed = float(expected.get("initial_speed_mps", 0.0))
                speed_error = abs(actual_speed - expected_speed)
                observations.append(
                    f"a{agent_id}:pos_err={location_error:.2f}m,"
                    f"speed={actual_speed:.2f}/{expected_speed:.2f}m/s"
                )
                occupancy = info.get("road_occupancy", {}).get(agent_id)
                if not isinstance(occupancy, dict):
                    failures.append(
                        f"{path.name}/agent{agent_id}: missing road occupancy evidence"
                    )
                else:
                    observations[-1] += (
                        f",outside={occupancy['footprint_outside_count']}/8,"
                        f"raw={int(bool(occupancy['offroad_raw']))}"
                    )
                    if not occupancy.get("driving_center_valid", False):
                        failures.append(
                            f"{path.name}/agent{agent_id}: center is not exact Driving"
                        )
                    if occupancy.get("offroad_raw", True):
                        failures.append(
                            f"{path.name}/agent{agent_id}: spawn footprint is offroad"
                        )
                if location_error > args.position_tolerance:
                    failures.append(
                        f"{path.name}/agent{agent_id}: position error "
                        f"{location_error:.2f}m"
                    )
                if speed_error > args.speed_tolerance:
                    failures.append(
                        f"{path.name}/agent{agent_id}: speed error "
                        f"{speed_error:.2f}m/s"
                    )
            if "car_following" in config.get("tags", []):
                rear_tag = next(
                    tag
                    for tag in config["tags"]
                    if tag in {"agent0_rear", "agent1_rear"}
                )
                rear_id = int(rear_tag[len("agent")])
                safety = env.get_longitudinal_safety_state()[rear_id]
                gap_m = float(safety["front_gap_m"])
                lidar_min = float(observations_by_agent[rear_id][8:].min())
                observations.append(
                    f"rear=a{rear_id},gap={gap_m:.2f}m,lidar_min={lidar_min:.3f}"
                )
                if safety["has_front_vehicle"] != 1.0:
                    failures.append(
                        f"{path.name}/agent{rear_id}: no same-lane front vehicle"
                    )
                if not math.isfinite(gap_m) or gap_m > args.ttc_max_range_m:
                    failures.append(
                        f"{path.name}/agent{rear_id}: front gap {gap_m:.2f}m "
                        f"outside TTC range {args.ttc_max_range_m:.2f}m"
                    )
                if lidar_min >= 1.0:
                    failures.append(
                        f"{path.name}/agent{rear_id}: 20m lidar did not detect "
                        "the configured front vehicle"
                    )
            print(f"CHECK {path.name}: " + "; ".join(observations))
    finally:
        env.close()

    if failures:
        print("Live smoke 失败:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"CARLA live smoke 通过: {len(files)} 个场景")
    return 0


def build_parser() -> argparse.ArgumentParser:
    training_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town05")
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=training_root / "data" / "training_scenarios_town05_v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-tolerance", type=float, default=0.25)
    parser.add_argument("--speed-tolerance", type=float, default=0.25)
    parser.add_argument("--follow-spectator", action="store_true")
    parser.add_argument("--lidar-range-m", type=float, default=20.0)
    parser.add_argument("--ttc-max-range-m", type=float, default=20.0)
    return parser


def main() -> int:
    return run_live_smoke(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
