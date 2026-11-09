"""Real-CARLA positive/negative checks for the frozen offroad protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, Iterable, Optional, Tuple

import carla
import numpy as np

from environment.base_reward_v3 import is_legal_lane_change_marking
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.offroad_protocol import (
    OFFROAD_PERSISTENCE_TICKS,
    OFFROAD_TERMINAL_PENALTY,
    offroad_protocol_spec,
)


DIRECTION_SCENARIOS = {
    "northbound": "scenario_17_northbound_free_flow.json",
    "southbound": "scenario_18_southbound_free_flow.json",
    "eastbound": "scenario_19_eastbound_free_flow.json",
    "westbound": "scenario_20_westbound_free_flow.json",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _brake_actions(agent_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    return {
        int(agent_id): np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        for agent_id in agent_ids
    }


def _reset_tick_events(env: CarlaMultiAgentEnv) -> None:
    for agent_id in range(env.num_agents):
        env.collision_events[agent_id] = False
        env.crosswalk_invasion_events[agent_id] = False
        env.lane_invasion_events[agent_id] = False
        env.offroad_consecutive_ticks[agent_id] = 0
        env.offroad_confirmed_events[agent_id] = False
        env.offroad_terminal_events[agent_id] = False
        env.first_offroad_evidence[agent_id] = None
        env.max_offroad_consecutive_ticks[agent_id] = 0
        env.max_footprint_outside_fraction[agent_id] = 0.0
    env.last_road_occupancy = {}


def _teleport_and_settle(
    env: CarlaMultiAgentEnv,
    agent_id: int,
    transform: carla.Transform,
) -> None:
    vehicle = env.vehicles[agent_id]
    vehicle.set_target_velocity(carla.Vector3D())
    vehicle.set_target_angular_velocity(carla.Vector3D())
    vehicle.set_transform(transform)
    env.world.tick()
    vehicle.set_target_velocity(carla.Vector3D())
    vehicle.set_target_angular_velocity(carla.Vector3D())
    _reset_tick_events(env)


def _same_direction(first, second) -> bool:
    a = first.transform.get_forward_vector()
    b = second.transform.get_forward_vector()
    return a.x * b.x + a.y * b.y > 0.9


def _legal_adjacent_waypoint(origin) -> Tuple[object, str]:
    candidates = (
        (origin.get_left_lane(), origin.left_lane_marking),
        (origin.get_right_lane(), origin.right_lane_marking),
    )
    for candidate, marking in candidates:
        if candidate is None or candidate.lane_type != carla.LaneType.Driving:
            continue
        if not _same_direction(origin, candidate):
            continue
        marking_name = str(marking.type).rsplit(".", 1)[-1]
        if not is_legal_lane_change_marking([marking_name]):
            continue
        ahead = candidate.next(8.0)
        return (ahead[0] if ahead else candidate), marking_name
    raise RuntimeError("no same-direction adjacent Driving lane across Broken marking")


def _actor_transform_at_waypoint(vehicle, waypoint) -> carla.Transform:
    current = vehicle.get_transform()
    target = waypoint.transform
    return carla.Transform(
        carla.Location(
            x=float(target.location.x),
            y=float(target.location.y),
            z=float(current.location.z),
        ),
        carla.Rotation(
            pitch=float(target.rotation.pitch),
            yaw=float(target.rotation.yaw),
            roll=float(target.rotation.roll),
        ),
    )


def _teleport_to_stable_non_driving(
    env: CarlaMultiAgentEnv,
    agent_id: int,
):
    """Find a point that remains offroad after CARLA commits the teleport.

    Testing the candidate before ``world.tick()`` is insufficient: ``set_transform``
    is applied on the next synchronous tick, and a point just outside one lane can
    settle onto an opposite-direction Driving lane.  The live oracle is therefore
    the actor's post-tick footprint, not the proposed centre location.
    """
    vehicle = env.vehicles[agent_id]
    current = vehicle.get_transform()
    origin = env._get_driving_waypoint(current.location)
    if origin is None:
        raise RuntimeError("cannot project the legal origin to a Driving lane")
    right = origin.transform.get_right_vector()
    lane_width = float(origin.lane_width)
    attempted = []
    # Start beyond the neighbouring carriageway.  Smaller offsets can be outside
    # the origin lane yet still lie in another perfectly legal Driving lane.
    for multiplier in (4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
        for sign in (-1.0, 1.0):
            location = carla.Location(
                x=float(origin.transform.location.x + sign * multiplier * lane_width * right.x),
                y=float(origin.transform.location.y + sign * multiplier * lane_width * right.y),
                z=float(current.location.z),
            )
            transform = carla.Transform(
                location,
                carla.Rotation(
                    pitch=float(current.rotation.pitch),
                    yaw=float(current.rotation.yaw),
                    roll=float(current.rotation.roll),
                ),
            )
            _teleport_and_settle(env, agent_id, transform)
            # Collision/lane sensors deliver callbacks asynchronously after the
            # world frame.  Flush a few non-rollout frames and reject a location
            # that keeps colliding with a wall, prop, or other static geometry.
            collision_flush = []
            for _ in range(3):
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                env.world.tick()
                time.sleep(0.05)
                collision_flush.append(bool(env.collision_events.get(agent_id, False)))
                env.collision_events[agent_id] = False
                env.crosswalk_invasion_events[agent_id] = False
                env.lane_invasion_events[agent_id] = False
            actual_transform = vehicle.get_transform()
            occupancy = env._inspect_vehicle_road_occupancy(agent_id)
            attempted.append(
                {
                    "requested_lateral_offset_m": sign * multiplier * lane_width,
                    "actual_location": {
                        "x": float(actual_transform.location.x),
                        "y": float(actual_transform.location.y),
                        "z": float(actual_transform.location.z),
                    },
                    "offroad_raw": bool(occupancy["offroad_raw"]),
                    "outside_count": int(occupancy["footprint_outside_count"]),
                    "collision_flush": collision_flush,
                }
            )
            if occupancy["offroad_raw"] and not collision_flush[-1]:
                _reset_tick_events(env)
                return (
                    actual_transform,
                    sign * multiplier * lane_width,
                    occupancy,
                    attempted,
                )
    raise RuntimeError(
        "could not find a stable post-tick non-Driving point; "
        f"attempts={attempted}"
    )


def _assert_reset_legal(info: Dict, *, context: str) -> Dict[str, object]:
    occupancy = info.get("road_occupancy")
    if not isinstance(occupancy, dict) or set(occupancy) != {0, 1}:
        raise RuntimeError(f"{context}: reset lacks exact road-occupancy evidence")
    result = {}
    for agent_id in (0, 1):
        evidence = occupancy[agent_id]
        if not evidence["driving_center_valid"] or evidence["offroad_raw"]:
            raise RuntimeError(f"{context}: agent {agent_id} spawn is offroad")
        result[str(agent_id)] = evidence
    return result


def _run_legal_adjacent_case(
    env: CarlaMultiAgentEnv,
    config: Dict,
    *,
    context: str,
) -> Dict[str, object]:
    env.set_scenario(config)
    _, info = env.reset()
    reset_evidence = _assert_reset_legal(info, context=context)
    origin = env._get_driving_waypoint(env.vehicles[0].get_transform().location)
    adjacent, marking = _legal_adjacent_waypoint(origin)
    _teleport_and_settle(env, 0, _actor_transform_at_waypoint(env.vehicles[0], adjacent))
    env._reset_offroad_tracking()

    ticks = []
    for _ in range(OFFROAD_PERSISTENCE_TICKS):
        _, _, dones, info = env.step(_brake_actions((0, 1)))
        evidence = info["road_occupancy"][0]
        ticks.append(evidence)
        if evidence["offroad_raw"] or info["offroad_terminated"][0]:
            raise RuntimeError(f"{context}: legal adjacent Driving lane was rejected")
        if dones[0] or info["episode_end_reason"] != "running":
            raise RuntimeError(f"{context}: legal adjacent lane ended the episode")
    return {
        "marking": marking,
        "reset": reset_evidence,
        "ticks": ticks,
    }


def _run_offroad_goal_case(
    env: CarlaMultiAgentEnv,
    config: Dict,
    *,
    context: str,
) -> Dict[str, object]:
    env.set_scenario(config)
    _, info = env.reset()
    _assert_reset_legal(info, context=context)
    (
        offroad_transform,
        lateral_offset,
        immediate,
        search_attempts,
    ) = _teleport_to_stable_non_driving(env, 0)

    env.agent_goal_zones[0] = env._build_goal_zone(
        {
            "x": offroad_transform.location.x,
            "y": offroad_transform.location.y,
            "z": offroad_transform.location.z,
            "yaw": offroad_transform.rotation.yaw,
            "half_length": 3.0,
            "half_width": 3.0,
        }
    )

    ticks = []
    for tick in range(1, OFFROAD_PERSISTENCE_TICKS + 1):
        _, _, dones, info = env.step(_brake_actions((0, 1)))
        details = info["reward_details"][0]
        evidence = info["road_occupancy"][0]
        ticks.append(
            {
                "tick": tick,
                "dones": dict(dones),
                "terminated": dict(info["terminated"]),
                "truncated": dict(info["truncated"]),
                "episode_end_reason": info["episode_end_reason"],
                "offroad_terminated": dict(info["offroad_terminated"]),
                "reward_offroad": float(details["offroad"]),
                "reward_success": float(details["success"]),
                "occupancy": evidence,
            }
        )
        if details["success"] != 0.0:
            raise RuntimeError(f"{context}: offroad goal shortcut received success")
        if tick < OFFROAD_PERSISTENCE_TICKS:
            if dones[0] or info["episode_end_reason"] != "running":
                raise RuntimeError(f"{context}: offroad debounce terminated too early")
            if details["offroad"] != 0.0:
                raise RuntimeError(f"{context}: offroad penalty fired before confirmation")
        else:
            if info["episode_end_reason"] != "cooperative_offroad":
                raise RuntimeError(f"{context}: confirmed offroad reason is incorrect")
            if not info["terminated"][0] or not info["truncated"][1]:
                raise RuntimeError(f"{context}: offroad T/X lifecycle is incorrect")
            if details["offroad"] != OFFROAD_TERMINAL_PENALTY:
                raise RuntimeError(f"{context}: offroad terminal cost is incorrect")
            if info["first_offroad_evidence"][0] is None:
                raise RuntimeError(f"{context}: first offroad evidence is missing")
    return {
        "lateral_offset_m": lateral_offset,
        "search_attempts": search_attempts,
        "immediate": immediate,
        "ticks": ticks,
    }


def run(args: argparse.Namespace) -> int:
    scenario_dir = args.scenario_dir.resolve()
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town="Town05",
        num_agents=2,
        max_steps=20,
        spawn_vehicles=False,
        disable_tensorboard=True,
        seed=args.seed,
        reward_version="v3",
        reward_gamma=0.997,
        lidar_range_m=20.0,
        ttc_max_range_m=20.0,
    )
    report = {
        "report_version": "offroad_live_validation_v1",
        "created_at": _utc_now_iso(),
        "protocol": offroad_protocol_spec(),
        "directions": {},
    }
    failure: Optional[BaseException] = None
    try:
        for direction, filename in DIRECTION_SCENARIOS.items():
            path = scenario_dir / filename
            config = json.loads(path.read_text(encoding="utf-8"))
            context = f"{direction}/{filename}"
            report["directions"][direction] = {
                "legal_adjacent_driving": _run_legal_adjacent_case(
                    env, config, context=context
                ),
                "offroad_goal_shortcut": _run_offroad_goal_case(
                    env, config, context=context
                ),
            }
            print(f"PASS {context}")
        report["status"] = "PASS"
    except BaseException as error:
        failure = error
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        env.close()
        settings = env.world.get_settings()
        actors = env.world.get_actors()
        report["cleanup"] = {
            "synchronous_mode": bool(settings.synchronous_mode),
            "fixed_delta_seconds": settings.fixed_delta_seconds,
            "vehicle_count": len(actors.filter("vehicle.*")),
            "sensor_count": len(actors.filter("sensor.*")),
        }
        if (
            settings.synchronous_mode
            or settings.fixed_delta_seconds is not None
            or report["cleanup"]["vehicle_count"] != 0
            or report["cleanup"]["sensor_count"] != 0
        ):
            report["status"] = "FAIL"
            report["cleanup_error"] = "CARLA cleanup invariant failed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"report={args.output.resolve()} status={report['status']}")
    if failure is not None:
        raise failure
    return 0 if report["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=project_dir / "data" / "training_scenarios_town05_v2",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
