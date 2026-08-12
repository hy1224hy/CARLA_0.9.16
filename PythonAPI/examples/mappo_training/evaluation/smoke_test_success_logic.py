"""Deterministic real-CARLA smoke test for the success/lifecycle protocol.

This module deliberately does not load an Actor or a checkpoint.  It separates
the question "does the environment recognise and account for success?" from the
question "has MAPPO learned to reach the goal?".

Run from the mappo_training directory, for example::

    python -m evaluation.smoke_test_success_logic --mode teleport
    python -m evaluation.smoke_test_success_logic --mode scripted --follow-spectator

Both modes use the real CarlaMultiAgentEnv.  ``teleport`` moves one agent at a
time into the goal immediately before a normal env.step(), which isolates the
success/reward/lifecycle logic.  ``scripted`` starts both agents close to the
goal and applies a fixed straight-driving control, which additionally exercises
short-horizon vehicle motion.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

# All project entry points are intended to be run as modules from mappo_training.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from training.checkpoint_protocol import atomic_write_json, get_git_metadata


PROTOCOL_VERSION = "success-logic-smoke/v1"
DEFAULT_SCENARIO = PROJECT_DIR / "data" / "training_scenarios" / "scenario_01_basic.json"
DEFAULT_AGENT_IDS = (0, 1)


class SuccessProtocolError(AssertionError):
    """The real environment output does not satisfy the frozen success contract."""


def _as_bool_map(raw: Mapping[Any, Any], agent_ids: Sequence[int], field_name: str) -> Dict[int, bool]:
    if not isinstance(raw, Mapping):
        raise SuccessProtocolError(f"{field_name} must be a mapping, got {type(raw).__name__}")
    result: Dict[int, bool] = {}
    for agent_id in agent_ids:
        if agent_id in raw:
            value = raw[agent_id]
        elif str(agent_id) in raw:
            value = raw[str(agent_id)]
        else:
            raise SuccessProtocolError(f"{field_name} is missing agent {agent_id}")
        result[agent_id] = bool(value)
    return result


def _as_agent_mapping(raw: Any, agent_id: int, field_name: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise SuccessProtocolError(f"{field_name} must be a mapping")
    if agent_id in raw:
        value = raw[agent_id]
    elif str(agent_id) in raw:
        value = raw[str(agent_id)]
    else:
        raise SuccessProtocolError(f"{field_name} is missing agent {agent_id}")
    if not isinstance(value, Mapping):
        raise SuccessProtocolError(f"{field_name}[{agent_id}] must be a mapping")
    return value


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SuccessProtocolError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise SuccessProtocolError(f"{field_name} must be finite, got {number!r}")
    return number


def build_near_goal_scenario(
    base_config: Mapping[str, Any],
    *,
    agent_ids: Sequence[int] = DEFAULT_AGENT_IDS,
    lead_distance_m: float = 5.0,
    following_gap_m: float = 10.0,
) -> Dict[str, Any]:
    """Replace scenario starts with collision-separated positions before the goal.

    ``lead_distance_m`` is measured from the near boundary of the goal rectangle,
    not its centre.  Each following agent is another ``following_gap_m`` behind.
    The calculation is yaw-generic rather than hard-coded to Town02's +Y road.
    """

    if not isinstance(base_config, Mapping):
        raise ValueError("scenario JSON root must be an object")
    goal = base_config.get("goal_zone")
    if not isinstance(goal, Mapping):
        raise ValueError("scenario must define goal_zone")

    lead_distance_m = float(lead_distance_m)
    following_gap_m = float(following_gap_m)
    half_length = float(goal.get("half_length", 3.0))
    half_width = float(goal.get("half_width", 1.75))
    if not math.isfinite(lead_distance_m) or lead_distance_m <= 0.0:
        raise ValueError("lead_distance_m must be a positive finite value")
    if not math.isfinite(following_gap_m) or following_gap_m < 6.0:
        raise ValueError("following_gap_m must be finite and >= 6m to avoid spawn overlap")
    if not math.isfinite(half_length) or half_length <= 0.0:
        raise ValueError("goal_zone.half_length must be positive and finite")
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("goal_zone.half_width must be positive and finite")

    center_x = float(goal["x"])
    center_y = float(goal["y"])
    center_z = float(goal.get("z", 0.5))
    yaw = float(goal["yaw"])
    yaw_rad = math.radians(yaw)
    forward_x = math.cos(yaw_rad)
    forward_y = math.sin(yaw_rad)

    scenario = copy.deepcopy(dict(base_config))
    scenario["description"] = (
        f"success logic smoke: lead={lead_distance_m:.2f}m before goal boundary, "
        f"gap={following_gap_m:.2f}m"
    )
    scenario["agents"] = {}
    for order, agent_id in enumerate(agent_ids):
        distance_from_center = half_length + lead_distance_m + order * following_gap_m
        scenario["agents"][f"agent{agent_id}"] = {
            "x": center_x - forward_x * distance_from_center,
            "y": center_y - forward_y * distance_from_center,
            "z": center_z,
            "yaw": yaw,
            "initial_speed_mps": 0.0,
        }
    return scenario


def validate_reset_protocol(info: Mapping[str, Any], agent_ids: Sequence[int]) -> None:
    required = {"active_mask", "terminated", "truncated", "episode_end_reason"}
    missing = sorted(required.difference(info))
    if missing:
        raise SuccessProtocolError(f"reset info is missing fields: {missing}")
    active = _as_bool_map(info["active_mask"], agent_ids, "reset.active_mask")
    terminated = _as_bool_map(info["terminated"], agent_ids, "reset.terminated")
    truncated = _as_bool_map(info["truncated"], agent_ids, "reset.truncated")
    if not all(active.values()):
        raise SuccessProtocolError(f"all agents must be active after reset, got {active}")
    if any(terminated.values()) or any(truncated.values()):
        raise SuccessProtocolError(
            f"reset cannot be terminal: terminated={terminated}, truncated={truncated}"
        )
    if info["episode_end_reason"] != "running":
        raise SuccessProtocolError(
            f"reset episode_end_reason must be 'running', got {info['episode_end_reason']!r}"
        )


@dataclass
class SuccessLogicAudit:
    """Stateful checker for sticky success, reward, and Gymnasium lifecycle semantics."""

    agent_ids: Sequence[int] = DEFAULT_AGENT_IDS
    previous_reached: Dict[int, bool] = field(init=False)
    success_evidence: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    collision_steps: list[Dict[str, Any]] = field(default_factory=list)
    last_snapshot: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.agent_ids = tuple(int(agent_id) for agent_id in self.agent_ids)
        self.previous_reached = {agent_id: False for agent_id in self.agent_ids}

    def observe(
        self,
        *,
        step: int,
        rewards: Mapping[int, float],
        dones: Mapping[int, bool],
        info: Mapping[str, Any],
        reached_target: Mapping[int, bool],
    ) -> Dict[str, Any]:
        required = {
            "reward_details",
            "reward_diagnostics",
            "collisions",
            "active_mask",
            "terminated",
            "truncated",
            "episode_end_reason",
        }
        missing = sorted(required.difference(info))
        if missing:
            raise SuccessProtocolError(f"step {step}: info is missing fields: {missing}")

        done_map = _as_bool_map(dones, self.agent_ids, f"step {step}.dones")
        terminated = _as_bool_map(info["terminated"], self.agent_ids, f"step {step}.terminated")
        truncated = _as_bool_map(info["truncated"], self.agent_ids, f"step {step}.truncated")
        active = _as_bool_map(info["active_mask"], self.agent_ids, f"step {step}.active_mask")
        collisions = _as_bool_map(info["collisions"], self.agent_ids, f"step {step}.collisions")
        reached = _as_bool_map(reached_target, self.agent_ids, f"step {step}.reached_target")
        reason = str(info["episode_end_reason"])

        for agent_id in self.agent_ids:
            if terminated[agent_id] and truncated[agent_id]:
                raise SuccessProtocolError(
                    f"step {step}: agent {agent_id} is both terminated and truncated"
                )
            if done_map[agent_id] != (terminated[agent_id] or truncated[agent_id]):
                raise SuccessProtocolError(
                    f"step {step}: agent {agent_id} lifecycle mismatch: "
                    f"done={done_map[agent_id]}, terminated={terminated[agent_id]}, "
                    f"truncated={truncated[agent_id]}"
                )
            if self.previous_reached[agent_id] and not reached[agent_id]:
                raise SuccessProtocolError(
                    f"step {step}: reached_target is not sticky for agent {agent_id}"
                )

            newly_reached = reached[agent_id] and not self.previous_reached[agent_id]
            if newly_reached:
                details = _as_agent_mapping(info["reward_details"], agent_id, "reward_details")
                diagnostics = _as_agent_mapping(
                    info["reward_diagnostics"], agent_id, "reward_diagnostics"
                )
                success_reward = _finite_float(
                    details.get("success"), f"step {step}.reward_details[{agent_id}].success"
                )
                success_event = _finite_float(
                    diagnostics.get("success_event"),
                    f"step {step}.reward_diagnostics[{agent_id}].success_event",
                )
                total_reward = _finite_float(
                    rewards[agent_id], f"step {step}.rewards[{agent_id}]"
                )
                if success_reward <= 0.0:
                    raise SuccessProtocolError(
                        f"step {step}: agent {agent_id} reached the goal without a positive success reward"
                    )
                if success_event != 1.0:
                    raise SuccessProtocolError(
                        f"step {step}: agent {agent_id} reached the goal but success_event={success_event}"
                    )
                if not done_map[agent_id] or not terminated[agent_id] or truncated[agent_id]:
                    raise SuccessProtocolError(
                        f"step {step}: success must yield done=True, terminated=True, "
                        f"truncated=False for agent {agent_id}"
                    )
                if active[agent_id]:
                    raise SuccessProtocolError(
                        f"step {step}: successful agent {agent_id} remains active"
                    )
                self.success_evidence[agent_id] = {
                    "step": int(step),
                    "success_reward": success_reward,
                    "time_bonus": _finite_float(
                        details.get("time_bonus", 0.0),
                        f"step {step}.reward_details[{agent_id}].time_bonus",
                    ),
                    "total_reward": total_reward,
                    "success_event": success_event,
                }

            if reached[agent_id]:
                if not done_map[agent_id] or not terminated[agent_id] or active[agent_id]:
                    raise SuccessProtocolError(
                        f"step {step}: reached agent {agent_id} has inconsistent sticky lifecycle"
                    )
            elif reason == "running":
                if done_map[agent_id] or not active[agent_id]:
                    raise SuccessProtocolError(
                        f"step {step}: unreached agent {agent_id} is inactive/terminal while episode runs"
                    )

        if any(collisions.values()):
            self.collision_steps.append({"step": int(step), "agents": collisions})

        if reason == "running":
            if all(reached.values()):
                raise SuccessProtocolError(
                    f"step {step}: all agents reached but episode_end_reason is still 'running'"
                )
        else:
            if not all(done_map.values()) or any(active.values()):
                raise SuccessProtocolError(
                    f"step {step}: global end {reason!r} must close every agent"
                )
            if reason != "all_agents_reached":
                raise SuccessProtocolError(
                    f"step {step}: episode ended before clean cooperative success: {reason!r}"
                )
            if not all(reached.values()):
                raise SuccessProtocolError(
                    f"step {step}: reason is all_agents_reached but reached_target={reached}"
                )

        snapshot = {
            "step": int(step),
            "episode_end_reason": reason,
            "reached_target": reached,
            "dones": done_map,
            "terminated": terminated,
            "truncated": truncated,
            "active_mask": active,
            "collisions": collisions,
            "success_reward": {
                agent_id: _finite_float(
                    _as_agent_mapping(info["reward_details"], agent_id, "reward_details").get(
                        "success", 0.0
                    ),
                    f"step {step}.reward_details[{agent_id}].success",
                )
                for agent_id in self.agent_ids
            },
            "success_event": {
                agent_id: _finite_float(
                    _as_agent_mapping(
                        info["reward_diagnostics"], agent_id, "reward_diagnostics"
                    ).get("success_event", 0.0),
                    f"step {step}.reward_diagnostics[{agent_id}].success_event",
                )
                for agent_id in self.agent_ids
            },
            "rewards": {
                agent_id: _finite_float(rewards[agent_id], f"step {step}.rewards[{agent_id}]")
                for agent_id in self.agent_ids
            },
        }
        self.previous_reached = reached
        self.last_snapshot = snapshot
        return snapshot

    def finalize(self) -> Dict[str, Any]:
        missing = [agent_id for agent_id in self.agent_ids if agent_id not in self.success_evidence]
        if missing:
            raise SuccessProtocolError(f"agents never produced success evidence: {missing}")
        if self.last_snapshot is None:
            raise SuccessProtocolError("no environment step was observed")
        if self.last_snapshot["episode_end_reason"] != "all_agents_reached":
            raise SuccessProtocolError(
                "final episode_end_reason must be 'all_agents_reached', got "
                f"{self.last_snapshot['episode_end_reason']!r}"
            )
        if self.collision_steps:
            raise SuccessProtocolError(
                "success path was contaminated by collision/crosswalk events: "
                f"{self.collision_steps}"
            )
        return {
            "success_protocol_pass": True,
            "clean_run": True,
            "success_evidence": self.success_evidence,
            "final_snapshot": self.last_snapshot,
            "collision_steps": self.collision_steps,
        }


def _load_json_object(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"scenario file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"scenario JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"scenario JSON root must be an object: {path}")
    return payload


def _agent_positions(env: CarlaMultiAgentEnv, agent_ids: Sequence[int]) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {}
    for agent_id in agent_ids:
        vehicle = env.vehicles[agent_id]
        if vehicle is None or not vehicle.is_alive:
            raise SuccessProtocolError(f"agent {agent_id} has no live vehicle after reset")
        transform = vehicle.get_transform()
        result[agent_id] = {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "yaw": float(transform.rotation.yaw),
        }
    return result


def _validate_spawn_positions(
    actual: Mapping[int, Mapping[str, float]],
    scenario: Mapping[str, Any],
    agent_ids: Sequence[int],
    tolerance_m: float = 0.75,
) -> None:
    configured = scenario["agents"]
    for agent_id in agent_ids:
        expected = configured[f"agent{agent_id}"]
        dx = float(actual[agent_id]["x"]) - float(expected["x"])
        dy = float(actual[agent_id]["y"]) - float(expected["y"])
        error = math.hypot(dx, dy)
        if error > tolerance_m:
            raise SuccessProtocolError(
                f"agent {agent_id} spawned {error:.3f}m from configured start; "
                "CARLA may have used a collision retry offset"
            )


def _set_spectator(env: CarlaMultiAgentEnv, scenario: Mapping[str, Any]) -> None:
    import carla

    goal = scenario["goal_zone"]
    starts = scenario["agents"]
    mean_x = sum(float(value["x"]) for value in starts.values()) / len(starts)
    mean_y = sum(float(value["y"]) for value in starts.values()) / len(starts)
    camera_x = (mean_x + float(goal["x"])) * 0.5
    camera_y = (mean_y + float(goal["y"])) * 0.5
    env.world.get_spectator().set_transform(
        carla.Transform(
            carla.Location(x=camera_x, y=camera_y, z=float(goal.get("z", 0.5)) + 25.0),
            carla.Rotation(pitch=-75.0, yaw=float(goal["yaw"])),
        )
    )


def _teleport_next_agent_to_goal(
    env: CarlaMultiAgentEnv,
    scenario: Mapping[str, Any],
    agent_id: int,
) -> None:
    import carla

    vehicle = env.vehicles[agent_id]
    if vehicle is None or not vehicle.is_alive:
        raise SuccessProtocolError(f"cannot teleport inactive agent {agent_id}")
    goal = scenario["goal_zone"]
    current_z = float(vehicle.get_transform().location.z)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_transform(
        carla.Transform(
            carla.Location(x=float(goal["x"]), y=float(goal["y"]), z=current_z),
            carla.Rotation(yaw=float(goal["yaw"])),
        )
    )


def _write_jsonl_line(file_obj: Any, payload: Mapping[str, Any]) -> None:
    file_obj.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    file_obj.flush()


def run_real_carla_smoke(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    agent_ids = DEFAULT_AGENT_IDS
    scenario_path = Path(args.scenario).resolve()
    base_scenario = _load_json_object(scenario_path)
    scenario = build_near_goal_scenario(
        base_scenario,
        agent_ids=agent_ids,
        lead_distance_m=args.lead_distance,
        following_gap_m=args.following_gap,
    )
    atomic_write_json(scenario, output_dir / "effective_scenario.json")

    env: Optional[CarlaMultiAgentEnv] = None
    cleanup_error: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None
    environment_info: Dict[str, Any] = {}
    post_cleanup: Dict[str, Any] = {}
    steps_path = output_dir / "steps.jsonl"
    audit = SuccessLogicAudit(agent_ids)
    try:
        env = CarlaMultiAgentEnv(
            host=args.host,
            port=args.port,
            town=args.town,
            num_agents=len(agent_ids),
            timestep=args.timestep,
            max_steps=args.max_steps,
            spawn_vehicles=False,
            adversarial_agent_ids=[],
            reward_version="v2",
            disable_tensorboard=True,
            seed=args.seed,
        )
        env.apply_scenario_config(scenario)
        _, reset_info = env.reset()
        validate_reset_protocol(reset_info, agent_ids)
        world_settings = env.world.get_settings()
        environment_info = {
            "client_version": str(env.client.get_client_version()),
            "server_version": str(env.client.get_server_version()),
            "map": str(env.map.name),
            "synchronous_mode_during_test": bool(world_settings.synchronous_mode),
            "fixed_delta_seconds_during_test": world_settings.fixed_delta_seconds,
        }
        initial_positions = _agent_positions(env, agent_ids)
        _validate_spawn_positions(initial_positions, scenario, agent_ids)
        if args.follow_spectator:
            _set_spectator(env, scenario)

        final_step = 0
        with steps_path.open("w", encoding="utf-8", newline="\n") as steps_file:
            for step in range(1, args.max_steps + 1):
                final_step = step
                active = dict(env.active_mask)
                if args.mode == "teleport":
                    pending = [
                        agent_id
                        for agent_id in agent_ids
                        if active.get(agent_id, False) and not env.reached_target.get(agent_id, False)
                    ]
                    if pending:
                        _teleport_next_agent_to_goal(env, scenario, pending[0])

                actions: Dict[int, np.ndarray] = {}
                for agent_id in agent_ids:
                    if not active.get(agent_id, False):
                        continue
                    if args.mode == "scripted":
                        actions[agent_id] = np.array(
                            [args.throttle, 0.0, 0.0], dtype=np.float32
                        )
                    else:
                        actions[agent_id] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                _, rewards, dones, info = env.step(actions)
                snapshot = audit.observe(
                    step=step,
                    rewards=rewards,
                    dones=dones,
                    info=info,
                    reached_target=dict(env.reached_target),
                )
                snapshot["positions"] = {
                    agent_id: {
                        "x": float(env.vehicles[agent_id].get_location().x),
                        "y": float(env.vehicles[agent_id].get_location().y),
                    }
                    for agent_id in agent_ids
                    if env.vehicles[agent_id] is not None and env.vehicles[agent_id].is_alive
                }
                _write_jsonl_line(steps_file, snapshot)

                if info["episode_end_reason"] != "running":
                    break
            else:
                raise SuccessProtocolError(
                    f"episode did not finish within max_steps={args.max_steps}"
                )

        audit_result = audit.finalize()
        result_payload = {
            **audit_result,
            "mode": args.mode,
            "steps": final_step,
            "scenario_source": str(scenario_path),
            "effective_scenario_file": str(output_dir / "effective_scenario.json"),
            "steps_file": str(steps_path),
            "initial_positions": initial_positions,
            "environment": environment_info,
            "post_cleanup": post_cleanup,
            "cleanup_error": cleanup_error,
        }
        return result_payload
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # preserve the primary failure while recording cleanup trouble
                cleanup_error = f"{type(exc).__name__}: {exc}"
                print(f"[WARN] CARLA cleanup failed: {cleanup_error}")
            else:
                settings = env.world.get_settings()
                actual_fixed_delta = settings.fixed_delta_seconds
                expected_fixed_delta = env._original_fixed_delta_seconds
                fixed_delta_restored = (
                    actual_fixed_delta is None and expected_fixed_delta is None
                ) or (
                    actual_fixed_delta is not None
                    and expected_fixed_delta is not None
                    and math.isclose(
                        float(actual_fixed_delta),
                        float(expected_fixed_delta),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                )
                post_cleanup = {
                    "vehicle_actor_count": len(env.world.get_actors().filter("vehicle.*")),
                    "sensor_actor_count": len(env.world.get_actors().filter("sensor.*")),
                    "synchronous_mode": bool(settings.synchronous_mode),
                    "fixed_delta_seconds": actual_fixed_delta,
                    "world_timing_restored": (
                        bool(settings.synchronous_mode) == env._original_synchronous_mode
                        and fixed_delta_restored
                    ),
                }
                cleanup_violations = []
                if post_cleanup["vehicle_actor_count"] != 0:
                    cleanup_violations.append(
                        f"vehicle actors remain={post_cleanup['vehicle_actor_count']}"
                    )
                if post_cleanup["sensor_actor_count"] != 0:
                    cleanup_violations.append(
                        f"sensor actors remain={post_cleanup['sensor_actor_count']}"
                    )
                if not post_cleanup["world_timing_restored"]:
                    cleanup_violations.append("world timing was not restored")
                if cleanup_violations:
                    cleanup_error = "; ".join(cleanup_violations)
            if result_payload is not None:
                result_payload["post_cleanup"] = post_cleanup
                result_payload["cleanup_error"] = cleanup_error


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return PROJECT_DIR / "logs" / f"success_logic_smoke_{timestamp}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify real-CARLA goal reward and lifecycle semantics without a trained policy"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("teleport", "scripted"),
        default="teleport",
        help="teleport isolates success logic; scripted also tests short straight driving",
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town05")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument(
        "--lead-distance",
        type=float,
        default=5.0,
        help="lead vehicle distance before the near boundary of the goal zone (metres)",
    )
    parser.add_argument(
        "--following-gap",
        type=float,
        default=10.0,
        help="longitudinal gap between consecutive vehicles (minimum 6m)",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.30,
        help="fixed throttle used only by scripted mode",
    )
    parser.add_argument(
        "--follow-spectator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="move CARLA spectator to a fixed overhead view",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory (default: logs/success_logic_smoke_<timestamp>)",
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    if not math.isfinite(args.timestep) or args.timestep <= 0.0:
        raise ValueError("timestep must be positive and finite")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not math.isfinite(args.throttle) or not 0.0 <= args.throttle <= 1.0:
        raise ValueError("throttle must be in [0, 1]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now().astimezone()
    result: MutableMapping[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "ERROR",
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "command": list(sys.argv if argv is None else ["smoke_test_success_logic", *argv]),
        "configuration": {
            "mode": args.mode,
            "host": args.host,
            "port": args.port,
            "town": args.town,
            "seed": args.seed,
            "timestep": args.timestep,
            "max_steps": args.max_steps,
            "lead_distance_m": args.lead_distance,
            "following_gap_m": args.following_gap,
            "throttle": args.throttle,
            "follow_spectator": args.follow_spectator,
        },
        "git": get_git_metadata(PROJECT_DIR),
        "output_dir": str(output_dir),
    }
    exit_code = 2
    try:
        _validate_cli(args)
        evidence = run_real_carla_smoke(args, output_dir)
        result.update(evidence)
        if evidence.get("cleanup_error"):
            raise RuntimeError(
                "success protocol passed but CARLA cleanup failed: "
                f"{evidence['cleanup_error']}"
            )
        result["status"] = "PASS"
        exit_code = 0
    except SuccessProtocolError as exc:
        result["status"] = "FAIL"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        exit_code = 1
    except KeyboardInterrupt:
        result["status"] = "INTERRUPTED"
        result["error_type"] = "KeyboardInterrupt"
        result["error"] = "test interrupted by user"
        exit_code = 130
    except Exception as exc:
        result["status"] = "ERROR"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        exit_code = 2
    finally:
        result["finished_at"] = datetime.now().astimezone().isoformat()
        result_path = output_dir / "result.json"
        atomic_write_json(result, result_path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"Result: {result_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
