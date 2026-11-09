"""Real-CARLA mechanism smoke for reward-v6 free-flow pace shaping."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from environment.base_reward_v6 import PACE_COMPONENT, reward_v6_spec
from environment.carla_multi_agent_env import CarlaMultiAgentEnv


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_step(
    rewards: Mapping[int, float], info: Mapping[str, object], *, context: str
) -> None:
    details = info["reward_details"]
    diagnostics = info["reward_diagnostics"]
    for agent_id, reward in rewards.items():
        residual = float(reward) - math.fsum(
            float(value) for value in details[agent_id].values()
        )
        if abs(residual) > 1e-6:
            raise RuntimeError(
                f"{context}: reward residual agent={agent_id} value={residual}"
            )
        if PACE_COMPONENT not in details[agent_id]:
            raise RuntimeError(f"{context}: v6 pace component is missing")
        if set(("pace_deficit_raw", "pace_deficit_risk", "pace_deficit_eligible")) \
                - set(diagnostics[agent_id]):
            raise RuntimeError(f"{context}: v6 pace diagnostics are incomplete")


def _reset(env: CarlaMultiAgentEnv, config: Dict[str, object], *, context: str):
    env.set_scenario(config)
    _, info = env.reset()
    if info["episode_end_reason"] != "running":
        raise RuntimeError(f"{context}: reset did not start a running episode")
    if env.current_scenario_tags < {"free_flow", "non_interactive"}:
        raise RuntimeError(f"{context}: required pace tags were not registered")
    return info


def _actions(info: Mapping[str, object], values) -> Dict[int, np.ndarray]:
    return {
        agent_id: np.asarray(values, dtype=np.float32)
        for agent_id, active in info["active_mask"].items()
        if active
    }


def _moving_on_schedule(
    env: CarlaMultiAgentEnv, config: Dict[str, object]
) -> Dict[str, object]:
    info = _reset(env, config, context="moving_on_schedule")
    checked = 0
    maximum_abs_cost = 0.0
    final_completion = {0: 0.0, 1: 0.0}
    for step in range(1, 41):
        _, rewards, _, info = env.step(_actions(info, [0.35, 0.0, 0.0]))
        _assert_step(rewards, info, context=f"moving_on_schedule/step-{step}")
        for agent_id, diagnostics in info["reward_diagnostics"].items():
            final_completion[agent_id] = float(diagnostics["route_completion"])
            if step >= 20 and diagnostics["pace_deficit_eligible"]:
                checked += 1
                maximum_abs_cost = max(
                    maximum_abs_cost,
                    abs(float(info["reward_details"][agent_id][PACE_COMPONENT])),
                )
        if info["episode_end_reason"] != "running":
            break
    if checked == 0:
        raise RuntimeError("moving_on_schedule: no eligible post-grace transition")
    if maximum_abs_cost > 1e-9:
        raise RuntimeError(
            "moving_on_schedule: on-schedule trajectory received pace cost "
            f"{maximum_abs_cost}"
        )
    return {
        "steps": step,
        "checked_eligible_transitions": checked,
        "maximum_abs_pace_cost": maximum_abs_cost,
        "final_completion": final_completion,
        "end_reason": info["episode_end_reason"],
    }


def _stationary_lateness(
    env: CarlaMultiAgentEnv, config: Dict[str, object]
) -> Dict[str, object]:
    info = _reset(env, config, context="stationary_lateness")
    risk_series = {0: [], 1: []}
    reward_series = {0: [], 1: []}
    for step in range(1, 61):
        _, rewards, _, info = env.step(_actions(info, [0.0, 0.0, 1.0]))
        _assert_step(rewards, info, context=f"stationary_lateness/step-{step}")
        if info["episode_end_reason"] != "running":
            raise RuntimeError(
                "stationary_lateness: episode terminated during mechanism smoke"
            )
        for agent_id, diagnostics in info["reward_diagnostics"].items():
            if step >= 20:
                risk_series[agent_id].append(float(diagnostics["pace_deficit_risk"]))
                reward_series[agent_id].append(
                    float(info["reward_details"][agent_id][PACE_COMPONENT])
                )
    for agent_id in (0, 1):
        risks = risk_series[agent_id]
        if any(later + 1e-9 < earlier for earlier, later in zip(risks, risks[1:])):
            raise RuntimeError(
                f"stationary_lateness: risk is not monotonic for agent={agent_id}"
            )
        if reward_series[agent_id][-1] >= -0.001:
            raise RuntimeError(
                f"stationary_lateness: dense cost too small for agent={agent_id}"
            )
    return {
        "steps": 60,
        "final_risk": {agent: values[-1] for agent, values in risk_series.items()},
        "final_pace_reward": {
            agent: values[-1] for agent, values in reward_series.items()
        },
        "end_reason": info["episode_end_reason"],
    }


def _blocker_config(stationary: Dict[str, object]) -> Dict[str, object]:
    config = copy.deepcopy(stationary)
    config["family"] = "r8c1_blocker_free_flow_stationary"
    config["description"] = "R8-C1 live-only front-blocker exemption smoke"
    tags = set(config.get("tags", []))
    tags.update({"free_flow", "non_interactive", "live_smoke"})
    config["tags"] = sorted(tags)
    rear = config["agents"]["agent0"]
    front = config["agents"]["agent1"]
    front["x"] = rear["x"]
    front["y"] = float(rear["y"]) + 8.0
    front["yaw"] = rear["yaw"]
    front["initial_speed_mps"] = 0.0
    config["agent_goal_zones"]["agent1"] = copy.deepcopy(
        config["agent_goal_zones"]["agent0"]
    )
    return config


def _front_blocker_exemption(
    env: CarlaMultiAgentEnv, config: Dict[str, object]
) -> Dict[str, object]:
    info = _reset(env, config, context="front_blocker_exemption")
    initial_relative = env._get_relative_position_to_agents(0)
    initial_safety = env.get_longitudinal_safety_state()[0]
    final = None
    observed = None
    for step in range(1, 51):
        _, rewards, _, info = env.step(_actions(info, [0.0, 0.0, 1.0]))
        _assert_step(rewards, info, context=f"front_blocker_exemption/step-{step}")
        if info["episode_end_reason"] != "running":
            raise RuntimeError(
                "front_blocker_exemption: episode terminated during smoke"
            )
        final = info["reward_diagnostics"][0]
        pace_reward = float(info["reward_details"][0][PACE_COMPONENT])
        if (
            final["pace_front_blocker_exempt"]
            and final["pace_deficit_raw"] > 0.0
        ):
            if final["pace_deficit_eligible"] or abs(pace_reward) > 1e-12:
                raise RuntimeError(
                    "front_blocker_exemption: exempt rear agent was penalized"
                )
            observed = {
                "step": step,
                "raw_deficit": float(final["pace_deficit_raw"]),
                "pace_reward": pace_reward,
            }
    pace_reward = float(info["reward_details"][0][PACE_COMPONENT])
    final_relative = env._get_relative_position_to_agents(0)
    final_safety = env.get_longitudinal_safety_state()[0]
    if observed is None:
        raise RuntimeError(
            "front_blocker_exemption: no blocked-and-late transition was observed; "
            f"initial_relative={initial_relative!r}, "
            f"initial_has_front_vehicle={initial_safety['has_front_vehicle']!r}, "
            f"final_relative={final_relative!r}, "
            f"final_has_front_vehicle={final_safety['has_front_vehicle']!r}"
        )
    return {
        "steps": 50,
        "first_observed": observed,
        "initial_relative": initial_relative,
        "final_relative": final_relative,
        "rear_raw_deficit": float(final["pace_deficit_raw"]),
        "rear_blocker_exempt": bool(final["pace_front_blocker_exempt"]),
        "rear_pace_reward": pace_reward,
        "end_reason": info["episode_end_reason"],
    }


def run(args: argparse.Namespace) -> int:
    rolling = _load(args.scenario_dir / "scenario_01_north_short_rolling_a0_left.json")
    stationary = _load(
        args.scenario_dir / "scenario_03_north_short_stationary_a0_left.json"
    )
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town="Town05",
        num_agents=2,
        max_steps=500,
        spawn_vehicles=False,
        disable_tensorboard=True,
        seed=42,
        reward_version="v6",
        reward_gamma=0.999,
        follow_spectator=True,
        lidar_range_m=20.0,
        ttc_max_range_m=20.0,
    )
    report = {
        "report_version": "p0_t05_r8c1_pace_live_v1",
        "created_at": _utc_now_iso(),
        "reward_protocol": reward_v6_spec(),
        "cases": {},
        "status": "RUNNING",
    }
    failure: Optional[BaseException] = None
    try:
        report["cases"]["moving_on_schedule"] = _moving_on_schedule(env, rolling)
        print("PASS moving_on_schedule")
        report["cases"]["stationary_lateness"] = _stationary_lateness(
            env, stationary
        )
        print("PASS stationary_lateness")
        report["cases"]["front_blocker_exemption"] = _front_blocker_exemption(
            env, _blocker_config(stationary)
        )
        print("PASS front_blocker_exemption")
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
