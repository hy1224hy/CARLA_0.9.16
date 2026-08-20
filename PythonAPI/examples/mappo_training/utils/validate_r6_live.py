"""Real-CARLA mechanism smoke for the P0-T05-09/R6 reward/lifecycle stack."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import carla
import numpy as np

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from utils.validate_offroad_live import (
    _actor_transform_at_waypoint,
    _legal_adjacent_waypoint,
    _reset_tick_events,
    _teleport_and_settle,
)


SCENARIOS = {
    "stationary": "scenario_01_north_parallel_stationary_a0_left.json",
    "required_merge": "scenario_13_north_required_merge_a0_merge.json",
    "near_gap": "scenario_29_northbound_car_following_near_gap_agent0_rear.json",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _actions(agent_ids: Iterable[int], *, brake: float) -> Dict[int, np.ndarray]:
    return {
        int(agent_id): np.asarray([0.0, 0.0, brake], dtype=np.float32)
        for agent_id in agent_ids
    }


def _load_scenario(directory: Path, name: str) -> Dict[str, object]:
    return json.loads((directory / SCENARIOS[name]).read_text(encoding="utf-8"))


def _assert_reward_conservation(rewards: Dict, info: Dict, *, context: str) -> float:
    maximum = 0.0
    for agent_id, reward in rewards.items():
        details = info["reward_details"][agent_id]
        residual = abs(float(reward) - math.fsum(float(value) for value in details.values()))
        maximum = max(maximum, residual)
        if residual > 1e-6:
            raise RuntimeError(f"{context}: agent {agent_id} reward residual={residual}")
    return maximum


def _assert_time_observation(
    observations: Dict[int, np.ndarray], *, step: int, max_steps: int, context: str
) -> None:
    expected = max(0.0, (max_steps - step) / max_steps)
    for agent_id, observation in observations.items():
        if tuple(observation.shape) != (20,):
            raise RuntimeError(
                f"{context}: agent {agent_id} observation shape={observation.shape}"
            )
        if abs(float(observation[-1]) - expected) > 1e-6:
            raise RuntimeError(
                f"{context}: agent {agent_id} remaining time "
                f"{float(observation[-1])} != {expected}"
            )


def _run_timeout_and_unblocked_stall(
    env: CarlaMultiAgentEnv, scenario_dir: Path
) -> Dict[str, object]:
    env.set_scenario(_load_scenario(scenario_dir, "stationary"))
    observations, _ = env.reset()
    _assert_time_observation(
        observations, step=0, max_steps=env.max_steps, context="stationary/reset"
    )
    maximum_residual = 0.0
    stall_evidence = None
    final = None
    for tick in range(1, env.max_steps + 1):
        observations, rewards, dones, info = env.step(_actions((0, 1), brake=1.0))
        _assert_time_observation(
            observations,
            step=tick,
            max_steps=env.max_steps,
            context=f"stationary/tick-{tick}",
        )
        maximum_residual = max(
            maximum_residual,
            _assert_reward_conservation(rewards, info, context=f"stationary/tick-{tick}"),
        )
        diagnostics = info["reward_diagnostics"]
        for agent_id in (0, 1):
            if diagnostics[agent_id]["stall_reportable"] == 1.0:
                stall_evidence = {
                    "tick": tick,
                    "agent_id": agent_id,
                    "diagnostics": diagnostics[agent_id],
                }
        if tick < env.max_steps and any(dones.values()):
            raise RuntimeError(f"stationary: episode ended early at tick {tick}")
        final = (rewards, dones, info)
    if stall_evidence is None:
        raise RuntimeError("stationary: persistent unblocked stall was not reported")
    rewards, dones, info = final
    if info["episode_end_reason"] != "mission_timeout" or not all(dones.values()):
        raise RuntimeError("stationary: mission timeout did not close the episode")
    if not all(info["terminated"].values()) or any(info["truncated"].values()):
        raise RuntimeError("stationary: mission timeout T/X lifecycle is incorrect")
    for agent_id in (0, 1):
        details = info["reward_details"][agent_id]
        diagnostics = info["reward_diagnostics"][agent_id]
        if not details["mission_timeout"] < 0.0:
            raise RuntimeError("stationary: timeout failure reward is missing")
        if diagnostics["mission_timeout_event"] != 1.0:
            raise RuntimeError("stationary: timeout diagnostic is missing")
    return {
        "max_reward_residual": maximum_residual,
        "stall": stall_evidence,
        "final": {
            "reason": info["episode_end_reason"],
            "terminated": info["terminated"],
            "truncated": info["truncated"],
            "timeout_rewards": {
                str(agent_id): info["reward_details"][agent_id]["mission_timeout"]
                for agent_id in (0, 1)
            },
        },
    }


def _run_stall_exemption(
    env: CarlaMultiAgentEnv,
    scenario_dir: Path,
    *,
    scenario_name: str,
    diagnostic_name: str,
    agent_id: int | None = None,
) -> Dict[str, object]:
    env.set_scenario(_load_scenario(scenario_dir, scenario_name))
    env.reset()
    for tick in range(1, env.max_steps):
        _, rewards, dones, info = env.step(_actions((0, 1), brake=1.0))
        _assert_reward_conservation(rewards, info, context=f"{scenario_name}/tick-{tick}")
        candidates = (agent_id,) if agent_id is not None else (0, 1)
        for candidate in candidates:
            diagnostics = info["reward_diagnostics"][candidate]
            if diagnostics["stall_raw"] != 1.0:
                continue
            if diagnostics[diagnostic_name] != 1.0 or diagnostics["stall_reportable"] != 0.0:
                raise RuntimeError(
                    f"{scenario_name}: raw stall did not receive {diagnostic_name} exemption"
                )
            return {
                "tick": tick,
                "agent_id": candidate,
                "diagnostics": diagnostics,
            }
        if any(dones.values()):
            raise RuntimeError(f"{scenario_name}: episode ended before stall exemption")
    raise RuntimeError(f"{scenario_name}: no exempted raw stall was observed")


def _run_approach_and_projected_ttc(
    env: CarlaMultiAgentEnv, scenario_dir: Path
) -> Dict[str, object]:
    env.set_scenario(_load_scenario(scenario_dir, "near_gap"))
    env.reset()
    rear = env.vehicles[0]
    front = env.vehicles[1]
    forward = rear.get_transform().get_forward_vector()
    rear.set_target_velocity(carla.Vector3D(5.0 * forward.x, 5.0 * forward.y, 0.0))
    front.set_target_velocity(carla.Vector3D())
    env.world.tick()
    _reset_tick_events(env)
    for tick in range(1, 6):
        _, rewards, dones, info = env.step(_actions((0, 1), brake=0.0))
        _assert_reward_conservation(rewards, info, context=f"approach/tick-{tick}")
        diagnostics = info["reward_diagnostics"][0]
        if (
            diagnostics["lidar_approach_gate"] > 0.0
            and diagnostics["ttc_candidate_count"] >= 1.0
            and diagnostics["ttc_trigger_count"] >= 1.0
            and diagnostics["ttc_risk"] > 0.0
        ):
            return {"tick": tick, "diagnostics": diagnostics}
        if any(dones.values()):
            raise RuntimeError("approach: episode ended before safety evidence")
    raise RuntimeError("approach: LiDAR approach gate/projected TTC did not activate")


def _run_legal_lane_change_grace(
    env: CarlaMultiAgentEnv, scenario_dir: Path
) -> Dict[str, object]:
    env.set_scenario(_load_scenario(scenario_dir, "required_merge"))
    env.reset()
    # The adjacent waypoint selected below is eight metres ahead of agent0,
    # which is exactly agent1's frozen spawn in this required-merge case.  Move
    # the peer farther forward first so this protocol test cannot manufacture a
    # collision by teleporting two actors onto the same transform.
    peer_origin = env._get_driving_waypoint(env.vehicles[1].get_transform().location)
    peer_ahead = peer_origin.next(25.0)
    if not peer_ahead:
        raise RuntimeError("legal lane change: cannot place peer safely ahead")
    _teleport_and_settle(
        env,
        1,
        _actor_transform_at_waypoint(env.vehicles[1], peer_ahead[0]),
    )
    origin = env._get_driving_waypoint(env.vehicles[0].get_transform().location)
    adjacent, marking = _legal_adjacent_waypoint(origin)
    _teleport_and_settle(env, 0, _actor_transform_at_waypoint(env.vehicles[0], adjacent))
    env.lane_invasion_events[0] = True
    env.lane_invasion_details[0] = {
        "has_invasion": True,
        "crossed_markings": [marking],
        "marking_types": [marking],
    }
    _, rewards, dones, info = env.step(_actions((0, 1), brake=1.0))
    _assert_reward_conservation(rewards, info, context="legal-lane-change")
    diagnostics = info["reward_diagnostics"][0]
    details = info["reward_details"][0]
    if (
        diagnostics["legal_lane_change_event"] != 1.0
        or diagnostics["legal_lane_change_grace_remaining"] != 10.0
        or diagnostics["lane_deviation_exempted"] != 1.0
        or details["lane_invasion"] != 0.0
        or dones[0]
    ):
        raise RuntimeError("legal lane change did not receive the frozen R6 grace")
    return {"marking": marking, "diagnostics": diagnostics}


def run(args: argparse.Namespace) -> int:
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town="Town05",
        num_agents=2,
        max_steps=120,
        spawn_vehicles=False,
        disable_tensorboard=True,
        seed=args.seed,
        reward_version="v4",
        reward_gamma=0.999,
        lidar_range_m=20.0,
        ttc_max_range_m=20.0,
    )
    report = {
        "report_version": "p0_t05_09_r6_live_mechanism_smoke_v1",
        "created_at": _utc_now_iso(),
        "status": "RUNNING",
        "checks": {},
    }
    failure = None
    try:
        report["checks"]["timeout_time_reward_unblocked_stall"] = (
            _run_timeout_and_unblocked_stall(env, args.scenario_dir)
        )
        report["checks"]["complex_interaction_stall_exemption"] = (
            _run_stall_exemption(
                env,
                args.scenario_dir,
                scenario_name="required_merge",
                diagnostic_name="stall_complex_interaction_exempt",
            )
        )
        report["checks"]["front_blocker_stall_exemption"] = _run_stall_exemption(
            env,
            args.scenario_dir,
            scenario_name="near_gap",
            diagnostic_name="stall_front_blocker_exempt",
            agent_id=0,
        )
        report["checks"]["lidar_approach_and_projected_ttc"] = (
            _run_approach_and_projected_ttc(env, args.scenario_dir)
        )
        report["checks"]["legal_lane_change_grace"] = _run_legal_lane_change_grace(
            env, args.scenario_dir
        )
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
