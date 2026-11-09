"""Four-direction real-CARLA smoke for the frozen P0-T05-10/R7 road rules.

This is a mechanism validator, not a learned-policy evaluation.  It uses
scripted controls and waypoint transforms to prove that the environment makes
the same distinction in every registered Town05 direction: legal lane
following/change/new-lane arrival, opposing-lane occupancy, moving wrong-way,
exact offroad occupancy, and an illegal pose inside a goal region.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import carla
import numpy as np

from environment.base_reward_v5 import reward_v5_spec
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.offroad_protocol import OFFROAD_PERSISTENCE_TICKS
from environment.road_legality_protocol import (
    DEFAULT_ROAD_LEGALITY_CONFIG,
    road_legality_protocol_spec,
)
from utils.validate_offroad_live import (
    _actor_transform_at_waypoint,
    _same_direction,
    _teleport_and_settle,
    _teleport_to_stable_non_driving,
)


DIRECTION_SCENARIOS = {
    "northbound": "scenario_17_northbound_free_flow.json",
    "southbound": "scenario_18_southbound_free_flow.json",
    "eastbound": "scenario_19_eastbound_free_flow.json",
    "westbound": "scenario_20_westbound_free_flow.json",
}
ROAD_TICKS = DEFAULT_ROAD_LEGALITY_CONFIG.violation_persistence_ticks


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _actions(
    agent_ids: Iterable[int], *, throttle: float = 0.0, brake: float = 1.0
) -> Dict[int, np.ndarray]:
    return {
        int(agent_id): np.asarray([throttle, 0.0, brake], dtype=np.float32)
        for agent_id in agent_ids
    }


def _lane_key(waypoint) -> Tuple[int, int]:
    return int(waypoint.road_id), int(waypoint.lane_id)


def _marking_name(marking) -> str:
    return str(getattr(marking, "type", "Unknown")).rsplit(".", 1)[-1]


def _load_scenario(directory: Path, filename: str) -> Dict[str, object]:
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def _assert_reward_conservation(rewards: Mapping, info: Mapping, *, context: str) -> None:
    for agent_id, reward in rewards.items():
        reconstructed = math.fsum(
            float(value) for value in info["reward_details"][agent_id].values()
        )
        if abs(float(reward) - reconstructed) > 1e-6:
            raise RuntimeError(
                f"{context}: agent={agent_id} reward residual="
                f"{float(reward) - reconstructed}"
            )


def _reset(env: CarlaMultiAgentEnv, config: Dict, *, context: str) -> Dict:
    env.set_scenario(config)
    _, info = env.reset()
    if info.get("episode_end_reason") != "running":
        raise RuntimeError(f"{context}: reset did not start a running episode")
    protocol = info.get("road_legality_protocol")
    if protocol != road_legality_protocol_spec():
        raise RuntimeError(f"{context}: reset road-legality protocol mismatch")
    for agent_id in (0, 1):
        evidence = info["road_legality"][agent_id]
        if (
            evidence["classification"] != "reference_initialized"
            or evidence["violation_raw"]
            or not evidence["success_eligible"]
        ):
            raise RuntimeError(f"{context}: illegal initial road state for agent={agent_id}")
    return info


def _move_peer_clear(env: CarlaMultiAgentEnv) -> None:
    peer = env.vehicles[1]
    origin = env._get_exact_driving_waypoint(peer.get_transform().location)
    candidates = origin.next(30.0) if origin is not None else []
    if not candidates:
        raise RuntimeError("cannot move peer to a clear same-lane waypoint")
    _teleport_and_settle(
        env, 1, _actor_transform_at_waypoint(peer, candidates[0])
    )


def _permitted_adjacent_waypoint(env: CarlaMultiAgentEnv, origin):
    for direction, getter_name, marking_name in (
        ("Left", "get_left_lane", "left_lane_marking"),
        ("Right", "get_right_lane", "right_lane_marking"),
    ):
        getter = getattr(origin, getter_name)
        candidate = getter()
        if (
            candidate is None
            or candidate.lane_type != carla.LaneType.Driving
            or not _same_direction(origin, candidate)
            or not env._lane_change_permission_contains(origin, direction)
        ):
            continue
        ahead = candidate.next(8.0)
        return (
            ahead[0] if ahead else candidate,
            direction,
            _marking_name(getattr(origin, marking_name)),
        )
    raise RuntimeError("no topology-adjacent same-direction permitted Driving lane")


def _opposing_waypoint(env: CarlaMultiAgentEnv, origin):
    world_map = env.world.get_map()
    right = origin.transform.get_right_vector()
    width = float(origin.lane_width)

    def _stable_same_road_candidates(candidate):
        # Scenario 19 starts close to a Road 3 boundary.  Blindly advancing an
        # opposing waypoint can therefore enter a junction connector even
        # though the lateral projection itself is valid.  Prefer moving
        # backwards along the opposing lane (forward along the scenario
        # direction), but re-check every returned waypoint against the exact
        # road/direction semantics before using it as a live fixture.
        probes = []
        for distance in (8.0, 12.0, 4.0, 16.0):
            probes.extend(candidate.previous(distance))
        probes.append(candidate)
        for distance in (4.0, 8.0, 12.0, 16.0):
            probes.extend(candidate.next(distance))
        return [
            probe
            for probe in probes
            if probe is not None
            and probe.lane_type == carla.LaneType.Driving
            and not probe.is_junction
            and int(probe.road_id) == int(origin.road_id)
            and env._road_direction_cosine(probe, origin) <= -0.5
        ]

    for multiplier in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        for sign in (-1.0, 1.0):
            requested = carla.Location(
                x=float(origin.transform.location.x + sign * multiplier * width * right.x),
                y=float(origin.transform.location.y + sign * multiplier * width * right.y),
                z=float(origin.transform.location.z),
            )
            candidate = world_map.get_waypoint(
                requested,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if (
                candidate is None
                or candidate.is_junction
                or _lane_key(candidate) == _lane_key(origin)
                or int(candidate.road_id) != int(origin.road_id)
                or env._road_direction_cosine(candidate, origin) > -0.5
            ):
                continue
            stable = _stable_same_road_candidates(candidate)
            if stable:
                return stable[0], sign * multiplier * width
    raise RuntimeError("no same-road opposing Driving lane found near legal origin")


def _step(
    env: CarlaMultiAgentEnv,
    actions: Dict[int, np.ndarray],
    *,
    context: str,
):
    _, rewards, dones, info = env.step(actions)
    _assert_reward_conservation(rewards, info, context=context)
    return rewards, dones, info


def _legal_straight(
    env: CarlaMultiAgentEnv, config: Dict, *, context: str
) -> Dict[str, object]:
    reset = _reset(env, config, context=context)
    _, dones, info = _step(
        env, _actions((0, 1)), context=f"{context}/legal-straight"
    )
    evidence = info["road_legality"][0]
    if (
        evidence["classification"] != "reference_lane"
        or evidence["violation_raw"]
        or not evidence["success_eligible"]
        or dones[0]
    ):
        raise RuntimeError(f"{context}: legal straight lane was rejected")
    return {"reset": reset["road_legality"][0], "tick": evidence}


def _legal_change_and_new_lane_goal(
    env: CarlaMultiAgentEnv, config: Dict, *, context: str
) -> Dict[str, object]:
    _reset(env, config, context=context)
    _move_peer_clear(env)
    vehicle = env.vehicles[0]
    origin = env._get_exact_driving_waypoint(vehicle.get_transform().location)
    adjacent, direction, marking = _permitted_adjacent_waypoint(env, origin)
    _teleport_and_settle(env, 0, _actor_transform_at_waypoint(vehicle, adjacent))

    ticks = []
    for tick in range(1, 4):
        _, dones, info = _step(
            env,
            _actions((0, 1)),
            context=f"{context}/legal-change-{tick}",
        )
        evidence = info["road_legality"][0]
        ticks.append(evidence)
        expected = "legal_lane_change" if tick == 1 else "reference_lane"
        if (
            evidence["classification"] != expected
            or evidence["violation_raw"]
            or not evidence["success_eligible"]
            or dones[0]
        ):
            raise RuntimeError(
                f"{context}: legal change/new-lane stabilization failed at tick={tick}"
            )
        if tick == 1 and (
            not evidence["lane_change_permitted"]
            or not evidence["legal_lane_change"]
            or info["reward_details"][0]["lane_invasion"] != 0.0
        ):
            raise RuntimeError(f"{context}: permitted change received an event cost")

    transform = vehicle.get_transform()
    env.agent_goal_zones[0] = env._build_goal_zone({
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "yaw": transform.rotation.yaw,
        "half_length": 3.0,
        "half_width": 1.75,
    })
    _, dones, info = _step(
        env, _actions((0, 1)), context=f"{context}/legal-new-lane-goal"
    )
    if (
        not dones[0]
        or not env.reached_target[0]
        or info["reward_details"][0]["success"] <= 0.0
        or info["reward_details"][0]["road_rule"] != 0.0
    ):
        raise RuntimeError(f"{context}: legal new-lane goal was not rewarded")
    return {
        "direction": direction,
        "crossed_marking": marking,
        "origin_lane": list(_lane_key(origin)),
        "new_lane": list(_lane_key(adjacent)),
        "ticks": ticks,
        "goal_success_reward": info["reward_details"][0]["success"],
    }


def _opposing_illegal_goal(
    env: CarlaMultiAgentEnv, config: Dict, *, context: str
) -> Dict[str, object]:
    _reset(env, config, context=context)
    _move_peer_clear(env)
    vehicle = env.vehicles[0]
    origin = env._get_exact_driving_waypoint(vehicle.get_transform().location)
    opposing, lateral_offset = _opposing_waypoint(env, origin)
    _teleport_and_settle(env, 0, _actor_transform_at_waypoint(vehicle, opposing))
    transform = vehicle.get_transform()
    env.agent_goal_zones[0] = env._build_goal_zone({
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "yaw": transform.rotation.yaw,
        "half_length": 3.0,
        "half_width": 2.0,
    })

    ticks = []
    for tick in range(1, ROAD_TICKS + 1):
        _, dones, info = _step(
            env, _actions((0, 1)), context=f"{context}/opposing-{tick}"
        )
        evidence = info["road_legality"][0]
        ticks.append({
            "tick": tick,
            "classification": evidence["classification"],
            "violation_ticks": evidence["consecutive_violation_ticks"],
            "success_reward": info["reward_details"][0]["success"],
            "road_rule_reward": info["reward_details"][0]["road_rule"],
            "offroad_raw": info["road_occupancy"][0]["offroad_raw"],
            "reason": info["episode_end_reason"],
        })
        if (
            evidence["classification"] != "opposing_lane"
            or evidence["success_eligible"]
            or info["reward_details"][0]["success"] != 0.0
            or info["road_occupancy"][0]["offroad_raw"]
        ):
            failure = {
                "tick": tick,
                "classification": evidence["classification"],
                "success_eligible": evidence["success_eligible"],
                "success_reward": info["reward_details"][0]["success"],
                "offroad_raw": info["road_occupancy"][0]["offroad_raw"],
                "collision_event": info["reward_diagnostics"][0]["collision_event"],
                "episode_end_reason": info["episode_end_reason"],
                "origin_lane": list(_lane_key(origin)),
                "candidate_lane": list(_lane_key(opposing)),
                "requested_lateral_offset_m": lateral_offset,
            }
            raise RuntimeError(
                f"{context}: opposing-lane illegal-goal semantics failed: "
                f"{json.dumps(failure, sort_keys=True)}"
            )
        if tick < ROAD_TICKS and dones[0]:
            raise RuntimeError(f"{context}: road-rule debounce terminated too early")
    if (
        info["episode_end_reason"] != "cooperative_road_violation"
        or not info["road_rule_terminated"][0]
        or not info["terminated"][0]
        or not info["truncated"][1]
        or info["reward_details"][0]["road_rule"] >= 0.0
    ):
        raise RuntimeError(f"{context}: opposing-lane terminal semantics failed")
    return {
        "origin_lane": list(_lane_key(origin)),
        "opposing_lane": list(_lane_key(opposing)),
        "requested_lateral_offset_m": lateral_offset,
        "ticks": ticks,
    }


def _moving_wrong_way(
    env: CarlaMultiAgentEnv, config: Dict, *, context: str
) -> Dict[str, object]:
    _reset(env, config, context=context)
    _move_peer_clear(env)
    vehicle = env.vehicles[0]
    origin = env._get_exact_driving_waypoint(vehicle.get_transform().location)
    ahead = origin.next(8.0)
    waypoint = ahead[0] if ahead else origin
    transform = _actor_transform_at_waypoint(vehicle, waypoint)
    transform.rotation.yaw += 180.0
    _teleport_and_settle(env, 0, transform)
    reversed_forward = vehicle.get_transform().get_forward_vector()
    vehicle.set_target_velocity(
        carla.Vector3D(2.0 * reversed_forward.x, 2.0 * reversed_forward.y, 0.0)
    )

    ticks = []
    for tick in range(1, ROAD_TICKS + 1):
        _, dones, info = _step(
            env,
            _actions((0, 1), throttle=0.35, brake=0.0),
            context=f"{context}/wrong-way-{tick}",
        )
        evidence = info["road_legality"][0]
        ticks.append({
            "tick": tick,
            "classification": evidence["classification"],
            "heading_cosine": evidence["vehicle_to_current_cosine"],
            "speed_mps": info["reward_diagnostics"][0]["speed_mps"],
            "reason": info["episode_end_reason"],
        })
        if evidence["classification"] != "wrong_way_heading":
            raise RuntimeError(f"{context}: moving wrong-way was not classified")
        if tick < ROAD_TICKS and dones[0]:
            raise RuntimeError(f"{context}: wrong-way debounce terminated too early")
    if info["episode_end_reason"] != "cooperative_road_violation":
        raise RuntimeError(f"{context}: moving wrong-way did not terminate as road violation")
    return {"lane": list(_lane_key(origin)), "ticks": ticks}


def _offroad_delegation(
    env: CarlaMultiAgentEnv, config: Dict, *, context: str
) -> Dict[str, object]:
    _reset(env, config, context=context)
    _move_peer_clear(env)
    _, lateral_offset, immediate, search_attempts = _teleport_to_stable_non_driving(
        env, 0
    )
    ticks = []
    for tick in range(1, OFFROAD_PERSISTENCE_TICKS + 1):
        _, dones, info = _step(
            env, _actions((0, 1)), context=f"{context}/offroad-{tick}"
        )
        evidence = info["road_legality"][0]
        ticks.append({
            "tick": tick,
            "classification": evidence["classification"],
            "road_rule_reward": info["reward_details"][0]["road_rule"],
            "offroad_reward": info["reward_details"][0]["offroad"],
            "reason": info["episode_end_reason"],
        })
        if (
            evidence["classification"] != "offroad_delegated"
            or evidence["violation_raw"]
            or info["reward_details"][0]["road_rule"] != 0.0
        ):
            raise RuntimeError(f"{context}: offroad was double-counted as road violation")
        if tick < OFFROAD_PERSISTENCE_TICKS and dones[0]:
            raise RuntimeError(f"{context}: offroad debounce terminated too early")
    if (
        info["episode_end_reason"] != "cooperative_offroad"
        or not info["offroad_terminated"][0]
        or info["road_rule_terminated"][0]
    ):
        raise RuntimeError(f"{context}: offroad delegation terminal semantics failed")
    return {
        "requested_lateral_offset_m": lateral_offset,
        "immediate": immediate,
        "search_attempts": search_attempts,
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
        reward_version="v5",
        reward_gamma=0.999,
        lidar_range_m=20.0,
        ttc_max_range_m=20.0,
    )
    report = {
        "report_version": "p0_t05_10_r7_road_legality_live_v1",
        "created_at": _utc_now_iso(),
        "road_legality_protocol": road_legality_protocol_spec(),
        "reward_protocol": reward_v5_spec(),
        "directions": {},
        "status": "RUNNING",
    }
    failure: Optional[BaseException] = None
    try:
        for direction, filename in DIRECTION_SCENARIOS.items():
            config = _load_scenario(scenario_dir, filename)
            context = f"{direction}/{filename}"
            report["directions"][direction] = {
                "legal_straight": _legal_straight(env, config, context=context),
                "legal_change_and_new_lane_goal": _legal_change_and_new_lane_goal(
                    env, config, context=context
                ),
                "opposing_lane_illegal_goal": _opposing_illegal_goal(
                    env, config, context=context
                ),
                "moving_wrong_way": _moving_wrong_way(env, config, context=context),
                "offroad_delegation": _offroad_delegation(
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
