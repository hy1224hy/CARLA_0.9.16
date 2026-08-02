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
    )
    failures = []
    try:
        for path in files:
            config = json.loads(path.read_text(encoding="utf-8"))
            env.set_scenario(config)
            env.reset()
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
            print(f"PASS {path.name}: " + "; ".join(observations))
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
    parser.add_argument("--town", default="Town02")
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=training_root / "data" / "training_scenarios_town02_v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-tolerance", type=float, default=0.25)
    parser.add_argument("--speed-tolerance", type=float, default=0.25)
    return parser


def main() -> int:
    return run_live_smoke(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
