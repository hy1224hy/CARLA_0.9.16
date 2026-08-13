"""Execute the frozen P0-BASE C0 checkpoint evaluation protocol.

Run as a module from ``mappo_training``.  Scientific choices come only from
the JSON contract; CLI options are limited to identity and runtime transport.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import carla

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from evaluation.base_c0_protocol import (
    AGENT_IDS,
    ACTION_C0_EPISODE_RECORD_VERSION,
    ACTION_C0_PROTOCOL_VERSION,
    C0_EPISODE_RECORD_VERSION,
    C0_PROTOCOL_VERSION,
    TOWN05_C0_PROTOCOL_VERSION,
    TOWN05_C0_EPISODE_RECORD_VERSION,
    TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
    TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION,
    TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
    TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION,
    C0_MANIFEST_VERSION,
    C0_RESOLVED_CONFIG_VERSION,
    LEGACY_C0_EPISODE_RECORD_VERSION,
    LEGACY_C0_PROTOCOL_VERSION,
    REWARD_C0_EPISODE_RECORD_VERSION,
    REWARD_C0_PROTOCOL_VERSION,
    C0ProtocolError,
    ScalarStats,
    append_jsonl,
    atomic_update_manifest,
    atomic_write_csv,
    build_summary,
    classify_outcome,
    ensure_completed_prefix,
    expand_profile_cases,
    load_and_validate_c0_checkpoint,
    load_c0_contract,
    load_strict_json,
    parse_eval_id,
    read_jsonl,
    sha256_file,
    summary_csv_rows,
    validate_episode_record,
)
from models.actor_critic import Actor
from training.checkpoint_protocol import (
    atomic_write_json,
    get_git_metadata,
    normalize_command_line,
    runtime_metadata,
    utc_now_iso,
)
from training.experiment_contract import verify_critical_sources_clean
from training.lifecycle_semantics import validate_lifecycle_flags
from training.metrics_protocol import (
    ACTION_RATE_NAMES,
    POLICY_ACTION_RATE_NAMES,
    REWARD_COMPONENTS,
    REWARD_DIAGNOSTICS,
    env_action_diagnostics,
    validate_active_reward_payload,
)
from training.action_protocol import (
    action_tanh_to_env,
    legacy_wta_action_tanh_to_env,
    policy_action_diagnostics,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONTRACT = PROJECT_DIR / "configs" / "evaluation" / "p0_base_c0_town05_v3.json"


def _set_seed(seed: int, env: CarlaMultiAgentEnv | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.traffic_manager.set_random_device_seed(int(seed))


def _strict_agent_bool_map(raw: Any, *, name: str) -> Dict[int, bool]:
    if not isinstance(raw, Mapping) or set(raw) != set(AGENT_IDS):
        raise C0ProtocolError(f"{name} must contain exactly agent ids {AGENT_IDS}")
    result: Dict[int, bool] = {}
    for agent_id in AGENT_IDS:
        value = raw[agent_id]
        if not isinstance(value, (bool, np.bool_)):
            raise C0ProtocolError(f"{name}[{agent_id}] must be boolean")
        result[agent_id] = bool(value)
    return result


def _observation(raw: Any, *, agent_id: int, context: str) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float32)
    if array.shape != (19,) or not np.isfinite(array).all():
        raise C0ProtocolError(
            f"{context}: observation agent={agent_id} must be finite shape (19,), got {array.shape}"
        )
    return array


class AgentEpisodeAccumulator:
    def __init__(
        self,
        *,
        reward_version: str,
        legacy_action: bool = False,
    ) -> None:
        self.legacy_action = bool(legacy_action)
        self.reward_version = str(reward_version)
        self.episode_return = 0.0
        self.active_steps = 0
        self.component_sums = {
            name: 0.0 for name in REWARD_COMPONENTS[self.reward_version]
        }
        self.residual_max = 0.0
        self.diagnostics = {
            name: ScalarStats() for name in REWARD_DIAGNOSTICS[self.reward_version]
        }
        self.action_sums = np.zeros(3, dtype=np.float64)
        self.action_rate_counts = {name: 0 for name in ACTION_RATE_NAMES}
        self.policy_action_sums = np.zeros(2, dtype=np.float64)
        self.policy_action_rate_counts = {
            name: 0 for name in POLICY_ACTION_RATE_NAMES
        }
        self.final_done = False
        self.final_terminated = False
        self.final_truncated = False

    def add(
        self,
        *,
        reward: float,
        components: Mapping[str, float],
        diagnostics: Mapping[str, float],
        residual: float,
        policy_action: np.ndarray,
        action: np.ndarray,
        context: str,
    ) -> None:
        self.episode_return += float(reward)
        self.active_steps += 1
        for name in self.component_sums:
            self.component_sums[name] += float(components[name])
        self.residual_max = max(self.residual_max, abs(float(residual)))
        for name, value in diagnostics.items():
            self.diagnostics[name].add(value, context=f"{context}.{name}")
        self.action_sums += np.asarray(action, dtype=np.float64)
        action_rates = env_action_diagnostics(
            action,
            context=f"{context}.action",
        )
        for name, value in action_rates.items():
            self.action_rate_counts[name] += int(value)
        if not self.legacy_action:
            policy_array = np.asarray(policy_action, dtype=np.float64)
            if policy_array.shape != (2,) or not np.all(np.isfinite(policy_array)):
                raise C0ProtocolError(
                    f"{context}: current policy action must be finite shape (2,)"
                )
            self.policy_action_sums += policy_array
            for name, value in policy_action_diagnostics(policy_array).items():
                self.policy_action_rate_counts[name] += int(value)

    def set_lifecycle(self, *, done: bool, terminated: bool, truncated: bool) -> None:
        self.final_done = done
        self.final_terminated = terminated
        self.final_truncated = truncated

    def as_dict(self, *, reached_target: bool) -> Dict[str, Any]:
        if self.active_steps:
            means = self.action_sums / self.active_steps
            action_stats: Dict[str, Any] = {
                "count": self.active_steps,
                "throttle_mean": float(means[0]),
                "steer_mean": float(means[1]),
                "brake_mean": float(means[2]),
                **{
                    f"{name}_rate": float(count / self.active_steps)
                    for name, count in self.action_rate_counts.items()
                },
            }
            if not self.legacy_action:
                policy_means = self.policy_action_sums / self.active_steps
                action_stats.update(
                    {
                        "policy_signed_longitudinal_mean": float(policy_means[0]),
                        "policy_steer_mean": float(policy_means[1]),
                        **{
                            f"{name}_rate": float(count / self.active_steps)
                            for name, count in self.policy_action_rate_counts.items()
                        },
                    }
                )
        else:
            action_stats = {
                "count": 0,
                "throttle_mean": None,
                "steer_mean": None,
                "brake_mean": None,
                **{f"{name}_rate": None for name in ACTION_RATE_NAMES},
            }
            if not self.legacy_action:
                action_stats.update(
                    {
                        "policy_signed_longitudinal_mean": None,
                        "policy_steer_mean": None,
                        **{
                            f"{name}_rate": None
                            for name in POLICY_ACTION_RATE_NAMES
                        },
                    }
                )
        diagnostic_payload = {name: stats.as_dict() for name, stats in self.diagnostics.items()}
        collision = (
            diagnostic_payload["collision_event"]["max"] is not None
            and diagnostic_payload["collision_event"]["max"] > 0.0
        )
        return {
            "episode_return": self.episode_return,
            "active_decision_steps": self.active_steps,
            "reached_target": bool(reached_target),
            "collision": bool(collision),
            "final_done": self.final_done,
            "final_terminated": self.final_terminated,
            "final_truncated": self.final_truncated,
            "reward_component_sums": dict(self.component_sums),
            "reward_residual_max_abs": self.residual_max,
            "diagnostics": diagnostic_payload,
            "actions": action_stats,
        }


def _build_actors(checkpoint: Mapping[str, Any]) -> Dict[int, Actor]:
    model_spec = checkpoint["model_spec"]
    actors: Dict[int, Actor] = {}
    for agent_id in AGENT_IDS:
        actor = Actor(
            obs_dim=int(model_spec["obs_dim"]),
            act_dim=int(model_spec["action_dim"]),
            hidden=int(model_spec["hidden_dim"]),
        )
        state = checkpoint["actors"].get(agent_id, checkpoint["actors"].get(str(agent_id)))
        if state is None:
            raise C0ProtocolError(f"checkpoint actor state missing for agent={agent_id}")
        actor.load_state_dict(state, strict=True)
        actor.eval()
        actors[agent_id] = actor
    return actors


def _run_case(
    *,
    env: CarlaMultiAgentEnv,
    actors: Mapping[int, Actor],
    case: Mapping[str, Any],
    contract: Mapping[str, Any],
    eval_id: str,
    profile: str,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    _set_seed(int(case["episode_seed"]), env)
    env.apply_scenario_config(dict(case["scenario_config"]))
    started_at = utc_now_iso()
    observations, info = env.reset()
    active_mask = _strict_agent_bool_map(info.get("active_mask"), name="reset.active_mask")
    if not all(active_mask.values()) or info.get("episode_end_reason") != "running":
        raise C0ProtocolError("C0 reset must start with both agents active and reason=running")
    legacy_action = contract["protocol_version"] == LEGACY_C0_PROTOCOL_VERSION
    reward_version = str(contract["environment"]["reward_version"])
    policy_action_dim = 3 if legacy_action else 2
    accumulators = {
        agent_id: AgentEpisodeAccumulator(
            reward_version=reward_version,
            legacy_action=legacy_action,
        )
        for agent_id in AGENT_IDS
    }
    final_reason = "running"
    final_info = info
    episode_length = 0
    lifecycle_transitions: Dict[int, Dict[str, bool] | None] = {
        agent_id: None for agent_id in AGENT_IDS
    }

    for step_index in range(1, int(contract["environment"]["max_steps"]) + 1):
        pre_active = dict(active_mask)
        active_ids = [agent_id for agent_id in AGENT_IDS if pre_active[agent_id]]
        if not active_ids:
            raise C0ProtocolError("no active agent while episode_end_reason is running")
        policy_actions: Dict[int, np.ndarray] = {}
        actions: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            for agent_id in active_ids:
                obs = _observation(observations[agent_id], agent_id=agent_id, context=f"step={step_index}")
                tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_tanh = actors[agent_id].get_deterministic_action(tensor).squeeze(0).cpu().numpy()
                if action_tanh.shape != (policy_action_dim,) or not np.isfinite(action_tanh).all():
                    raise C0ProtocolError(f"non-finite policy action for agent={agent_id}")
                policy_actions[agent_id] = action_tanh.copy()
                actions[agent_id] = (
                    legacy_wta_action_tanh_to_env(action_tanh)
                    if legacy_action
                    else action_tanh_to_env(action_tanh)
                )

        next_observations, raw_rewards, raw_dones, next_info = env.step(actions)
        final_info = next_info
        dones = _strict_agent_bool_map(raw_dones, name=f"step={step_index}.dones")
        terminated = _strict_agent_bool_map(next_info.get("terminated"), name=f"step={step_index}.terminated")
        truncated = _strict_agent_bool_map(next_info.get("truncated"), name=f"step={step_index}.truncated")
        next_active = _strict_agent_bool_map(next_info.get("active_mask"), name=f"step={step_index}.active_mask")
        normalized_rewards, details, diagnostics, residuals = validate_active_reward_payload(
            active_agent_ids=active_ids,
            rewards=raw_rewards,
            reward_details_by_agent=next_info.get("reward_details", {}),
            reward_diagnostics_by_agent=next_info.get("reward_diagnostics", {}),
            reward_version=reward_version,
            context=f"C0 case={case['case_id']} step={step_index}",
        )
        for agent_id in AGENT_IDS:
            done, term, trunc = validate_lifecycle_flags(
                dones[agent_id],
                terminated[agent_id],
                truncated[agent_id],
                context=f"C0 case={case['case_id']} step={step_index} agent={agent_id}",
            )
            if next_active[agent_id] != (pre_active[agent_id] and not done):
                raise C0ProtocolError(
                    f"active-mask transition mismatch at step={step_index}, agent={agent_id}"
                )
            if pre_active[agent_id] and not next_active[agent_id]:
                lifecycle_transitions[agent_id] = {
                    "pre_active": True,
                    "post_active": False,
                    "done": done,
                    "terminated": term,
                    "truncated": trunc,
                }
            accumulators[agent_id].set_lifecycle(done=done, terminated=term, truncated=trunc)
            if pre_active[agent_id]:
                accumulators[agent_id].add(
                    reward=normalized_rewards[agent_id],
                    components=details[agent_id],
                    diagnostics=diagnostics[agent_id],
                    residual=residuals[agent_id],
                    policy_action=policy_actions[agent_id],
                    action=actions[agent_id],
                    context=f"case={case['case_id']} step={step_index} agent={agent_id}",
                )
        observations = next_observations
        active_mask = next_active
        episode_length = step_index
        final_reason = str(next_info.get("episode_end_reason", ""))
        if final_reason != "running":
            if any(active_mask.values()):
                raise C0ProtocolError("global episode ended while an agent remains active")
            break
    else:
        raise C0ProtocolError("environment did not emit a terminal reason within max_steps")

    outcome = classify_outcome(final_reason, contract["outcomes"])
    agents_payload = {
        str(agent_id): accumulators[agent_id].as_dict(
            reached_target=bool(env.reached_target.get(agent_id, False))
        )
        for agent_id in AGENT_IDS
    }
    if contract["protocol_version"] in {
        TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
    }:
        offroad_terminated = _strict_agent_bool_map(
            final_info.get("offroad_terminated"),
            name="final.offroad_terminated",
        )
        first_evidence = final_info.get("first_offroad_evidence")
        max_ticks = final_info.get("max_offroad_consecutive_ticks")
        max_fraction = final_info.get("max_footprint_outside_fraction")
        for name, values in (
            ("first_offroad_evidence", first_evidence),
            ("max_offroad_consecutive_ticks", max_ticks),
            ("max_footprint_outside_fraction", max_fraction),
        ):
            if not isinstance(values, Mapping) or {
                int(key) for key in values
            } != set(AGENT_IDS):
                raise C0ProtocolError(
                    f"final.{name} must contain exactly agent ids {AGENT_IDS}"
                )
        for agent_id in AGENT_IDS:
            agents_payload[str(agent_id)].update(
                {
                    "offroad_terminated": offroad_terminated[agent_id],
                    "first_offroad_evidence": first_evidence[agent_id],
                    "max_offroad_consecutive_ticks": int(max_ticks[agent_id]),
                    "max_footprint_outside_fraction": float(
                        max_fraction[agent_id]
                    ),
                }
            )
            if (
                contract["protocol_version"]
                == TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION
            ):
                transition = lifecycle_transitions[agent_id]
                if transition is None:
                    raise C0ProtocolError(
                        f"missing lifecycle transition for agent={agent_id}"
                    )
                agents_payload[str(agent_id)]["lifecycle_transition"] = transition
    record = {
        "record_version": (
            LEGACY_C0_EPISODE_RECORD_VERSION
            if legacy_action
            else (
                ACTION_C0_EPISODE_RECORD_VERSION
                if contract["protocol_version"] == ACTION_C0_PROTOCOL_VERSION
                else (
                    REWARD_C0_EPISODE_RECORD_VERSION
                    if contract["protocol_version"] == REWARD_C0_PROTOCOL_VERSION
                    else (
                        TOWN05_C0_EPISODE_RECORD_VERSION
                        if contract["protocol_version"]
                        == TOWN05_C0_PROTOCOL_VERSION
                        else (
                            (
                                TOWN05_OUTCOME_REPEATABILITY_C0_EPISODE_RECORD_VERSION
                                if contract["protocol_version"]
                                == TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION
                                else TOWN05_OFFROAD_C0_EPISODE_RECORD_VERSION
                            )
                            if contract["protocol_version"] in {
                                TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
                                TOWN05_OUTCOME_REPEATABILITY_C0_PROTOCOL_VERSION,
                            }
                            else C0_EPISODE_RECORD_VERSION
                        )
                    )
                )
            )
        ),
        "eval_id": eval_id,
        "profile": profile,
        "case_id": case["case_id"],
        "case_ordinal": case["case_ordinal"],
        "scenario_name": case["scenario_name"],
        "scenario_sha256": case["scenario_sha256"],
        "scenario_family": case["scenario_family"],
        "scenario_difficulty": case["scenario_difficulty"],
        "episode_seed": case["episode_seed"],
        "checkpoint_sha256": checkpoint_sha256,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "episode_length": episode_length,
        "episode_end_reason": final_reason,
        "outcome": outcome,
        "agents": agents_payload,
        "team": {
            "mean_episode_return": sum(agents_payload[str(agent_id)]["episode_return"] for agent_id in AGENT_IDS) / 2.0,
            "total_active_decision_steps": sum(agents_payload[str(agent_id)]["active_decision_steps"] for agent_id in AGENT_IDS),
        },
    }
    validate_episode_record(
        record,
        eval_id=eval_id,
        profile=profile,
        checkpoint_sha256=checkpoint_sha256,
        expected_case=case,
    )
    return record


def _cleanup_environment(env: CarlaMultiAgentEnv | None) -> tuple[Dict[str, Any], str]:
    if env is None:
        return {}, "environment was not initialized"
    try:
        env.close()
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"
    settings = env.world.get_settings()
    actual_delta = settings.fixed_delta_seconds
    expected_delta = env._original_fixed_delta_seconds
    fixed_delta_restored = (
        actual_delta is None and expected_delta is None
    ) or (
        actual_delta is not None
        and expected_delta is not None
        and math.isclose(float(actual_delta), float(expected_delta), rel_tol=0.0, abs_tol=1e-9)
    )
    audit = {
        "vehicle_actor_count": len(env.world.get_actors().filter("vehicle.*")),
        "sensor_actor_count": len(env.world.get_actors().filter("sensor.*")),
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": actual_delta,
        "world_timing_restored": bool(settings.synchronous_mode) == env._original_synchronous_mode and fixed_delta_restored,
    }
    violations = []
    if audit["vehicle_actor_count"] != 0:
        violations.append(f"vehicle actors remain={audit['vehicle_actor_count']}")
    if audit["sensor_actor_count"] != 0:
        violations.append(f"sensor actors remain={audit['sensor_actor_count']}")
    if not audit["world_timing_restored"]:
        violations.append("world timing was not restored")
    return audit, "; ".join(violations)


def _prepare_output(
    *,
    args: argparse.Namespace,
    contract_record: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    cases: list[Dict[str, Any]],
) -> tuple[Path, list[Dict[str, Any]]]:
    contract = contract_record["payload"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.name != args.eval_id:
        raise C0ProtocolError("output directory basename must equal eval_id")
    outputs = contract["outputs"]
    evaluation_root = (PROJECT_DIR / outputs["evaluation_root"]).resolve()
    if output_dir.parent != evaluation_root:
        raise C0ProtocolError(
            f"C0 output directory must be a direct child of {evaluation_root}"
        )
    manifest_path = output_dir / outputs["manifest"]
    resolved_path = output_dir / outputs["resolved_config"]
    episodes_path = output_dir / outputs["episodes_jsonl"]
    if args.resume:
        if not output_dir.is_dir() or not manifest_path.is_file() or not resolved_path.is_file():
            raise C0ProtocolError("--resume requires an existing C0 manifest and resolved config")
        manifest = load_strict_json(manifest_path)
        resolved = load_strict_json(resolved_path)
        expected_identity = {
            "eval_id": args.eval_id,
            "profile": args.profile,
            "contract_sha256": contract_record["sha256"],
            "checkpoint_sha256": checkpoint_record["sha256"],
        }
        for name, expected in expected_identity.items():
            if manifest.get(name) != expected or resolved.get(name) != expected:
                raise C0ProtocolError(f"resume identity mismatch for {name}")
        records = read_jsonl(episodes_path) if episodes_path.is_file() else []
        ensure_completed_prefix(records, cases)
        for index, record in enumerate(records):
            validate_episode_record(
                record,
                eval_id=args.eval_id,
                profile=args.profile,
                checkpoint_sha256=checkpoint_record["sha256"],
                expected_case=cases[index],
            )
        if len(records) == len(cases):
            raise C0ProtocolError(
                "C0 evaluation is already complete; run the validator instead of --resume"
            )
        if (
            (output_dir / outputs["summary_json"]).exists()
            or (output_dir / outputs["summary_csv"]).exists()
        ):
            raise C0ProtocolError("incomplete resume evidence must not contain summary artifacts")
        return output_dir, records

    if output_dir.exists():
        raise C0ProtocolError("fresh C0 output directory already exists; use a new eval_id")
    output_dir.mkdir(parents=True)
    now = utc_now_iso()
    resolved = {
        "resolved_config_version": C0_RESOLVED_CONFIG_VERSION,
        "created_at": now,
        "eval_id": args.eval_id,
        "profile": args.profile,
        "contract_id": contract["contract_id"],
        "contract_path": contract_record["path"],
        "contract_sha256": contract_record["sha256"],
        "contract_canonical_sha256": contract_record["canonical_sha256"],
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "checkpoint_update": checkpoint_record["update"],
        "checkpoint_formal_eligible": checkpoint_record["formal_eligible"],
        "checkpoint_formal_ineligibility_reasons": checkpoint_record["formal_ineligibility_reasons"],
        "host": args.host,
        "port": args.port,
        "device": "cpu",
        "command": normalize_command_line(),
        "runtime": {
            **runtime_metadata(),
            "carla_python_module": str(getattr(carla, "__file__", "unknown")),
            "carla_client_version": "pending_connection",
            "carla_server_version": "pending_connection",
        },
        "case_count": len(cases),
        "cases": [{key: value for key, value in case.items() if key not in {"scenario_config"}} for case in cases],
        "git": get_git_metadata(REPOSITORY_ROOT),
    }
    atomic_write_json(resolved, resolved_path)
    atomic_write_json(
        {
            "manifest_version": C0_MANIFEST_VERSION,
            "eval_id": args.eval_id,
            "profile": args.profile,
            "contract_sha256": contract_record["sha256"],
            "resolved_config_sha256": sha256_file(resolved_path),
            "checkpoint_sha256": checkpoint_record["sha256"],
            "checkpoint_update": checkpoint_record["update"],
            "formal_eligible": checkpoint_record["formal_eligible"],
            "status": "PREPARED",
            "completed_cases": 0,
            "total_cases": len(cases),
            "created_at": now,
            "updated_at": now,
            "post_cleanup": {},
            "cleanup_error": "",
            "error": "",
        },
        manifest_path,
    )
    return output_dir, []


def _record_carla_runtime(
    *,
    env: CarlaMultiAgentEnv,
    resolved_path: Path,
    manifest_path: Path,
    expected_version: str,
) -> None:
    resolved = load_strict_json(resolved_path)
    runtime = dict(resolved["runtime"])
    client_version = str(env.client.get_client_version())
    server_version = str(env.client.get_server_version())
    if not client_version.startswith(expected_version) or not server_version.startswith(expected_version):
        raise C0ProtocolError(
            "CARLA client/server version differs from the frozen contract: "
            f"client={client_version!r}, server={server_version!r}, expected={expected_version!r}"
        )
    runtime.update(
        {
            "carla_client_version": client_version,
            "carla_server_version": server_version,
        }
    )
    resolved["runtime"] = runtime
    atomic_write_json(resolved, resolved_path)
    atomic_update_manifest(
        manifest_path,
        {
            "resolved_config_sha256": sha256_file(resolved_path),
            "updated_at": utc_now_iso(),
        },
    )


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    contract_record = load_c0_contract(args.eval_contract, project_dir=PROJECT_DIR)
    contract = contract_record["payload"]
    if (
        contract["protocol_version"] not in {
            C0_PROTOCOL_VERSION,
            TOWN05_C0_PROTOCOL_VERSION,
            TOWN05_OFFROAD_C0_PROTOCOL_VERSION,
        }
        and not args.allow_diagnostic_checkpoint
    ):
        raise C0ProtocolError(
            "historical C0 v1/v2/v3 is read-only evidence; new executions require "
            "a current Town02 v4 or Town05 contract, or an explicit diagnostic smoke invocation"
        )
    if args.allow_diagnostic_checkpoint and args.profile != "smoke":
        raise C0ProtocolError("diagnostic checkpoints are restricted to the smoke profile")
    checkpoint_record = load_and_validate_c0_checkpoint(
        args.checkpoint,
        contract=contract,
        allow_diagnostic_checkpoint=args.allow_diagnostic_checkpoint,
        map_location="cpu",
    )
    parse_eval_id(args.eval_id, profile=args.profile, checkpoint_update=checkpoint_record["update"])
    if checkpoint_record["formal_eligible"]:
        verify_critical_sources_clean(
            repository_root=REPOSITORY_ROOT,
            project_dir=PROJECT_DIR,
            relative_paths=contract["critical_source_paths"],
        )
    cases = expand_profile_cases(contract, args.profile, project_dir=PROJECT_DIR)
    output_dir, records = _prepare_output(
        args=args,
        contract_record=contract_record,
        checkpoint_record=checkpoint_record,
        cases=cases,
    )
    outputs = contract["outputs"]
    manifest_path = output_dir / outputs["manifest"]
    episodes_path = output_dir / outputs["episodes_jsonl"]
    env: CarlaMultiAgentEnv | None = None
    primary_error: BaseException | None = None
    cleanup_audit: Dict[str, Any] = {}
    cleanup_error = ""
    try:
        atomic_update_manifest(
            manifest_path,
            {"status": "RUNNING", "updated_at": utc_now_iso(), "error": ""},
        )
        actors = _build_actors(checkpoint_record["checkpoint"])
        env_config = contract["environment"]
        env = CarlaMultiAgentEnv(
            host=args.host,
            port=args.port,
            town=env_config["town"],
            num_agents=env_config["num_agents"],
            straight_road_length=env_config["straight_road_length"],
            timestep=env_config["timestep"],
            max_steps=env_config["max_steps"],
            spawn_vehicles=False,
            adversarial_agent_ids=[],
            reward_version=env_config["reward_version"],
            disable_tensorboard=True,
            seed=cases[0]["episode_seed"],
            reward_gamma=env_config["reward_gamma"],
            adversary_collision_ends_episode=True,
            lidar_range_m=env_config.get("lidar_range_m", 4.0),
            ttc_max_range_m=env_config.get("ttc_max_range_m", 10.0),
        )
        _record_carla_runtime(
            env=env,
            resolved_path=output_dir / outputs["resolved_config"],
            manifest_path=manifest_path,
            expected_version=env_config["carla_version"],
        )
        for case in cases[len(records) :]:
            record = _run_case(
                env=env,
                actors=actors,
                case=case,
                contract=contract,
                eval_id=args.eval_id,
                profile=args.profile,
                checkpoint_sha256=checkpoint_record["sha256"],
            )
            append_jsonl(episodes_path, record)
            records.append(record)
            atomic_update_manifest(
                manifest_path,
                {
                    "status": "RUNNING",
                    "completed_cases": len(records),
                    "updated_at": utc_now_iso(),
                },
            )
    except BaseException as error:
        primary_error = error
    finally:
        cleanup_audit, cleanup_error = _cleanup_environment(env)
        if cleanup_error:
            atomic_update_manifest(
                manifest_path,
                {
                    "status": "HOLD_CLEANUP",
                    "updated_at": utc_now_iso(),
                    "post_cleanup": cleanup_audit,
                    "cleanup_error": cleanup_error,
                },
            )

    if primary_error is not None:
        status = "INTERRUPTED" if isinstance(primary_error, KeyboardInterrupt) else "HOLD_ERROR"
        atomic_update_manifest(
            manifest_path,
            {
                "status": status,
                "completed_cases": len(records),
                "updated_at": utc_now_iso(),
                "post_cleanup": cleanup_audit,
                "cleanup_error": cleanup_error,
                "error": f"{type(primary_error).__name__}: {primary_error}",
            },
        )
        raise primary_error
    if cleanup_error:
        raise C0ProtocolError(f"C0 cleanup audit failed: {cleanup_error}")
    if len(records) != len(cases):
        raise C0ProtocolError("C0 evaluation ended without completing every frozen case")

    summary = build_summary(
        records,
        eval_id=args.eval_id,
        profile=args.profile,
        checkpoint_sha256=checkpoint_record["sha256"],
    )
    atomic_write_json(summary, output_dir / outputs["summary_json"])
    atomic_write_csv(output_dir / outputs["summary_csv"], summary_csv_rows(summary))
    completion_status = "PASS" if checkpoint_record["formal_eligible"] else "DIAGNOSTIC_PASS"
    atomic_update_manifest(
        manifest_path,
        {
            "status": completion_status,
            "completed_cases": len(records),
            "updated_at": utc_now_iso(),
            "post_cleanup": cleanup_audit,
            "cleanup_error": "",
            "error": "",
            "episodes_sha256": sha256_file(episodes_path),
            "summary_sha256": sha256_file(output_dir / outputs["summary_json"]),
            "summary_csv_sha256": sha256_file(output_dir / outputs["summary_csv"]),
        },
    )
    from evaluation.validate_base_c0_run import validate_c0_run

    report = validate_c0_run(
        output_dir,
        expected_contract=Path(args.eval_contract),
        expected_checkpoint=Path(args.checkpoint),
        expected_profile=args.profile,
        expected_eval_id=args.eval_id,
        write_report=True,
    )
    if report["status"] != "PASS":
        atomic_update_manifest(
            manifest_path,
            {"status": "HOLD_VALIDATION", "updated_at": utc_now_iso()},
        )
        raise C0ProtocolError("independent C0 artifact validation failed")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen P0-BASE C0 checkpoint evaluation")
    parser.add_argument("--eval-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "gate", "validation"), required=True)
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-diagnostic-checkpoint", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        report = run_evaluation(args)
    except KeyboardInterrupt:
        print("C0 evaluation interrupted; resume with the same identity and --resume", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        traceback.print_exc()
        print(f"C0 evaluation HOLD: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"C0 evaluation {report['status']}: {args.output_dir}")


if __name__ == "__main__":
    main()
