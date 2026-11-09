"""Real-CARLA mechanism smoke for the frozen P0-T05-11/R8-A curriculum.

This validator uses scripted controls, not learned policies.  It proves spawn,
initial-speed, short-horizon, legal-goal, reward-conservation, terminal and
cleanup semantics for both mirrored S0-A and S0-B tasks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from environment.base_reward_v5 import reward_v5_spec
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from environment.road_legality_protocol import road_legality_protocol_spec
from training.r8a_curriculum import R8A_STAGE_NEW_SCENARIOS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(directory: Path, name: str) -> Dict[str, object]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _speed_mps(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _assert_reward_conservation(
    rewards: Mapping[int, float], info: Mapping[str, object], *, context: str
) -> None:
    details = info.get("reward_details", {})
    for agent_id, reward in rewards.items():
        reconstructed = math.fsum(
            float(value) for value in details[agent_id].values()
        )
        residual = float(reward) - reconstructed
        if abs(residual) > 1e-6:
            raise RuntimeError(
                f"{context}: reward residual for agent {agent_id} is {residual}"
            )


def _run_scripted_success(
    env: CarlaMultiAgentEnv,
    config: Dict[str, object],
    *,
    context: str,
) -> Dict[str, object]:
    env.set_scenario(config)
    _, info = env.reset()
    if env.max_steps != 200:
        raise RuntimeError(f"{context}: short horizon is {env.max_steps}, not 200")
    if info.get("episode_end_reason") != "running":
        raise RuntimeError(f"{context}: reset did not start a running episode")
    if info.get("road_legality_protocol") != road_legality_protocol_spec():
        raise RuntimeError(f"{context}: road-legality protocol mismatch")
    for agent_id in (0, 1):
        evidence = info["road_legality"][agent_id]
        if evidence["violation_raw"] or not evidence["success_eligible"]:
            raise RuntimeError(
                f"{context}: agent {agent_id} spawned in an illegal road state"
            )

    expected_speed = float(
        config["agents"]["agent0"]["initial_speed_mps"]
    )
    initial_speeds = {
        agent_id: _speed_mps(env.vehicles[agent_id]) for agent_id in (0, 1)
    }
    tolerance = 0.25 if expected_speed > 0.0 else 0.10
    for agent_id, actual in initial_speeds.items():
        if abs(actual - expected_speed) > tolerance:
            raise RuntimeError(
                f"{context}: agent {agent_id} initial speed {actual:.3f} differs "
                f"from {expected_speed:.3f}"
            )

    max_reward_residual = 0.0
    last_info = info
    for step in range(1, 201):
        active_mask = last_info["active_mask"]
        actions = {
            agent_id: np.asarray([0.35, 0.0, 0.0], dtype=np.float32)
            for agent_id in (0, 1)
            if active_mask[agent_id]
        }
        _, rewards, _, last_info = env.step(actions)
        _assert_reward_conservation(
            rewards, last_info, context=f"{context}/step-{step}"
        )
        for agent_id, reward in rewards.items():
            reconstructed = math.fsum(
                float(value)
                for value in last_info["reward_details"][agent_id].values()
            )
            max_reward_residual = max(
                max_reward_residual, abs(float(reward) - reconstructed)
            )
        if last_info["episode_end_reason"] != "running":
            break

    if last_info["episode_end_reason"] not in {
        "all_agents_reached", "cooperative_success"
    }:
        raise RuntimeError(
            f"{context}: scripted legal drive ended as "
            f"{last_info['episode_end_reason']!r}"
        )
    if not all(env.reached_target.values()):
        raise RuntimeError(f"{context}: success did not include both agents")
    if any(last_info["offroad_terminated"].values()):
        raise RuntimeError(f"{context}: scripted drive triggered offroad")
    if any(last_info["road_rule_terminated"].values()):
        raise RuntimeError(f"{context}: scripted drive triggered road violation")
    return {
        "expected_initial_speed_mps": expected_speed,
        "observed_initial_speed_mps": initial_speeds,
        "episode_max_steps": env.max_steps,
        "steps": step,
        "end_reason": last_info["episode_end_reason"],
        "max_abs_reward_residual": max_reward_residual,
    }


def run(args: argparse.Namespace) -> int:
    scenario_dir = args.scenario_dir.resolve()
    short_names = [
        name for stage in R8A_STAGE_NEW_SCENARIOS[:2] for name in stage
    ]
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town="Town05",
        num_agents=2,
        max_steps=500,
        spawn_vehicles=False,
        disable_tensorboard=True,
        seed=args.seed,
        reward_version="v5",
        reward_gamma=0.999,
        lidar_range_m=20.0,
        ttc_max_range_m=20.0,
    )
    report = {
        "report_version": "p0_t05_11_r8a_curriculum_live_v1",
        "created_at": _utc_now_iso(),
        "reward_protocol": reward_v5_spec(),
        "road_legality_protocol": road_legality_protocol_spec(),
        "scenarios": {},
        "status": "RUNNING",
    }
    failure: Optional[BaseException] = None
    try:
        long_config = _load(scenario_dir, R8A_STAGE_NEW_SCENARIOS[2][0])
        short_config = _load(scenario_dir, short_names[0])
        env.set_scenario(long_config)
        if env.max_steps != 500:
            raise RuntimeError("long scenario did not restore the contract horizon")
        env.set_scenario(short_config)
        if env.max_steps != 200:
            raise RuntimeError("short scenario did not restore its registered horizon")

        for name in short_names:
            result = _run_scripted_success(
                env, _load(scenario_dir, name), context=name
            )
            report["scenarios"][name] = result
            print(f"PASS {name}: {result['steps']} ticks")
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
        default=project_dir / "data/training_scenarios_town05_r8a_v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
