"""Reproducible real-CARLA mechanism diagnostic for R2 reward and R3 critic.

This is deliberately not a policy-performance pilot.  Fixed scripted controls
exercise reward-v3 outcomes, then an identical real-state return batch compares
raw-MSE Critic regression with the frozen R3 ValueNorm+Huber mechanism.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import carla

from environment.base_reward_v3 import REWARD_COMPONENTS, REWARD_DIAGNOSTICS
from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import AgentConditionedCritic
from training.value_normalization import ValueNormalizer, normalized_huber_loss


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCENARIO_PATH = (
    PROJECT_DIR
    / "data"
    / "training_scenarios_town02_v2"
    / "scenario_01_east_basic_launch_a0_rear.json"
)
GAMMA = 0.997
AGENT_IDS = (0, 1)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite diagnostic value: {value!r}")
    return result


def _control(mode: str, step: int, agent_id: int) -> np.ndarray:
    if mode == "stopped":
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    if mode == "smooth":
        throttle = min(0.65, 0.065 * (step + 1))
        return np.asarray([throttle, 0.0, 0.0], dtype=np.float32)
    if mode == "oscillatory":
        phase = (step // 5) % 2
        throttle = 0.75 if phase == 0 else 0.10
        steer = 0.18 if phase == 0 else -0.18
        return np.asarray([throttle, steer, 0.0], dtype=np.float32)
    if mode == "rear_collision":
        return np.asarray(
            [1.0, 0.0, 0.0] if agent_id == 0 else [0.0, 0.0, 1.0],
            dtype=np.float32,
        )
    raise ValueError(f"unknown scripted trajectory: {mode}")


def _discounted_return(rewards: Iterable[float]) -> float:
    result = 0.0
    for reward in reversed(list(rewards)):
        result = float(reward) + GAMMA * result
    return result


def _reward_to_go(rewards: Iterable[float]) -> list[float]:
    values: list[float] = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running = float(reward) + GAMMA * running
        values.append(running)
    return list(reversed(values))


def _run_trajectory(
    env: CarlaMultiAgentEnv,
    scenario: Mapping[str, object],
    *,
    mode: str,
    max_steps: int,
) -> tuple[Dict[str, object], list[Dict[str, object]]]:
    env.apply_scenario_config(dict(scenario))
    _, info = env.reset()
    active = {aid: bool(info["active_mask"][aid]) for aid in AGENT_IDS}
    if not all(active.values()) or info["episode_end_reason"] != "running":
        raise RuntimeError(f"{mode}: invalid reset lifecycle")

    rewards = {aid: [] for aid in AGENT_IDS}
    components = {
        aid: {name: 0.0 for name in REWARD_COMPONENTS} for aid in AGENT_IDS
    }
    completion_start = {aid: None for aid in AGENT_IDS}
    completion_end = {aid: None for aid in AGENT_IDS}
    residual_max = 0.0
    transitions: list[Dict[str, object]] = []
    end_reason = "diagnostic_horizon"

    for step in range(max_steps):
        pre_active = dict(active)
        state = env.get_global_state().astype(np.float32)
        actions = {
            aid: _control(mode, step, aid)
            for aid in AGENT_IDS
            if pre_active[aid]
        }
        _, step_rewards, _, next_info = env.step(actions)
        details_by_agent = next_info["reward_details"]
        diagnostics_by_agent = next_info["reward_diagnostics"]
        for aid in AGENT_IDS:
            if not pre_active[aid]:
                continue
            reward = _finite(step_rewards[aid])
            details = details_by_agent[aid]
            diagnostics = diagnostics_by_agent[aid]
            if set(details) != set(REWARD_COMPONENTS):
                raise RuntimeError(f"{mode} agent={aid}: reward component schema drift")
            if set(diagnostics) != set(REWARD_DIAGNOSTICS):
                raise RuntimeError(f"{mode} agent={aid}: diagnostic schema drift")
            reconstructed = math.fsum(_finite(details[name]) for name in REWARD_COMPONENTS)
            residual_max = max(residual_max, abs(reward - reconstructed))
            rewards[aid].append(reward)
            for name in REWARD_COMPONENTS:
                components[aid][name] += _finite(details[name])
            current_completion = _finite(diagnostics["route_completion"])
            if completion_start[aid] is None:
                completion_start[aid] = _finite(
                    diagnostics["route_completion_prev"]
                )
            completion_end[aid] = current_completion
            transitions.append(
                {
                    "state": state.copy(),
                    "agent_id": aid,
                    "reward": reward,
                    "trajectory": mode,
                    "agent_step": len(rewards[aid]) - 1,
                }
            )
        active = {aid: bool(next_info["active_mask"][aid]) for aid in AGENT_IDS}
        end_reason = str(next_info["episode_end_reason"])
        if end_reason != "running":
            break
    else:
        end_reason = "diagnostic_horizon"

    for aid in AGENT_IDS:
        returns = _reward_to_go(rewards[aid])
        row_indices = [
            index
            for index, row in enumerate(transitions)
            if row["agent_id"] == aid
        ]
        if len(row_indices) != len(returns):
            raise RuntimeError(f"{mode} agent={aid}: transition/return misalignment")
        for index, target in zip(row_indices, returns):
            transitions[index]["return_target"] = target

    per_agent = {}
    for aid in AGENT_IDS:
        per_agent[str(aid)] = {
            "active_steps": len(rewards[aid]),
            "undiscounted_return": math.fsum(rewards[aid]),
            "discounted_return": _discounted_return(rewards[aid]),
            "route_completion_start": completion_start[aid],
            "route_completion_end": completion_end[aid],
            "reward_component_sums": components[aid],
            "success_events": int(components[aid]["success"] > 0.0),
            "collision_events": int(components[aid]["collision"] < 0.0),
        }
    report = {
        "trajectory": mode,
        "configured_horizon": max_steps,
        "executed_steps": max(len(rewards[aid]) for aid in AGENT_IDS),
        "episode_end_reason": end_reason,
        "reward_conservation_max_abs_residual": residual_max,
        "agents": per_agent,
    }
    return report, transitions


def _explained_variance(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    variance = torch.var(targets, unbiased=False)
    if float(variance) <= 1e-12:
        return 0.0
    return float(
        (1.0 - torch.var(targets - predictions, unbiased=False) / variance).item()
    )


def _fit_critic_ab(transitions: list[Dict[str, object]]) -> Dict[str, object]:
    states = torch.as_tensor(
        np.stack([row["state"] for row in transitions]), dtype=torch.float32
    )
    agent_ids = torch.as_tensor(
        [row["agent_id"] for row in transitions], dtype=torch.long
    )
    targets = torch.as_tensor(
        [row["return_target"] for row in transitions], dtype=torch.float32
    )
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(transitions))
    split = max(1, int(0.8 * len(indices)))
    train_indices = torch.as_tensor(indices[:split], dtype=torch.long)
    test_indices = torch.as_tensor(indices[split:], dtype=torch.long)
    if test_indices.numel() == 0:
        test_indices = train_indices[-1:]

    torch.manual_seed(42)
    seed_model = AgentConditionedCritic(
        state_dim=states.shape[1], num_agents=2, hidden=128
    )
    initial_state = copy.deepcopy(seed_model.state_dict())

    def fit(kind: str) -> Dict[str, object]:
        model = AgentConditionedCritic(
            state_dim=states.shape[1], num_agents=2, hidden=128
        )
        model.load_state_dict(initial_state)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        normalizer = ValueNormalizer()
        if kind == "r3_valuenorm_huber":
            normalizer.update(targets[train_indices])
            train_targets = normalizer.normalize(targets[train_indices])
        else:
            train_targets = targets[train_indices]
        grad_norms = []
        losses = []
        for _ in range(300):
            predictions = model(states[train_indices], agent_ids[train_indices])
            if kind == "r3_valuenorm_huber":
                loss = normalized_huber_loss(predictions, train_targets, delta=10.0)
            else:
                loss = 0.5 * (train_targets - predictions).square().mean()
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), 0.5, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.item()))
            grad_norms.append(float(grad_norm))
        with torch.no_grad():
            raw_predictions = model(states[test_indices], agent_ids[test_indices])
            predictions = (
                normalizer.denormalize(raw_predictions)
                if kind == "r3_valuenorm_huber"
                else raw_predictions
            )
            test_targets = targets[test_indices]
        result = {
            "optimizer_steps": 300,
            "initial_training_loss": losses[0],
            "final_training_loss": losses[-1],
            "preclip_grad_norm_mean": float(np.mean(grad_norms)),
            "preclip_grad_norm_max": float(np.max(grad_norms)),
            "grad_clip_activation_rate": float(np.mean(np.asarray(grad_norms) > 0.5)),
            "heldout_raw_mse": float((test_targets - predictions).square().mean()),
            "heldout_raw_mae": float((test_targets - predictions).abs().mean()),
            "heldout_explained_variance": _explained_variance(
                test_targets, predictions
            ),
            "heldout_prediction_std": float(predictions.std(unbiased=False)),
            "heldout_target_std": float(test_targets.std(unbiased=False)),
        }
        if kind == "r3_valuenorm_huber":
            result["value_normalizer"] = normalizer.statistics()
            normalized_all = normalizer.normalize(targets[train_indices])
            result["normalized_train_target_mean"] = float(normalized_all.mean())
            result["normalized_train_target_std"] = float(
                normalized_all.std(unbiased=False)
            )
        return result

    baseline = fit("raw_mse_baseline")
    r3 = fit("r3_valuenorm_huber")
    return {
        "batch": {
            "transition_count": len(transitions),
            "train_count": int(train_indices.numel()),
            "heldout_count": int(test_indices.numel()),
            "raw_target_mean": float(targets.mean()),
            "raw_target_std": float(targets.std(unbiased=False)),
            "raw_target_min": float(targets.min()),
            "raw_target_max": float(targets.max()),
        },
        "raw_mse_baseline": baseline,
        "r3_valuenorm_huber": r3,
        "comparison": {
            "clip_rate_reduction": (
                baseline["grad_clip_activation_rate"]
                - r3["grad_clip_activation_rate"]
            ),
            "heldout_mse_ratio_r3_over_baseline": (
                r3["heldout_raw_mse"] / max(baseline["heldout_raw_mse"], 1e-12)
            ),
            "heldout_ev_gain": (
                r3["heldout_explained_variance"]
                - baseline["heldout_explained_variance"]
            ),
        },
    }


def run(args: argparse.Namespace) -> Path:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    client_version = client.get_client_version()
    server_version = client.get_server_version()
    before_world = client.get_world()
    before_settings = before_world.get_settings()
    before_vehicle_count = len(before_world.get_actors().filter("vehicle.*"))
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))

    env = None
    trajectory_reports = []
    transitions: list[Dict[str, object]] = []
    try:
        env = CarlaMultiAgentEnv(
            host=args.host,
            port=args.port,
            town="Town02",
            num_agents=2,
            straight_road_length=100.0,
            timestep=0.1,
            max_steps=400,
            spawn_vehicles=True,
            reward_version="v3",
            reward_gamma=GAMMA,
            disable_tensorboard=True,
            seed=42,
        )
        for mode, horizon in (
            ("stopped", 80),
            ("smooth", 320),
            ("oscillatory", 140),
            ("rear_collision", 240),
        ):
            report, rows = _run_trajectory(
                env, scenario, mode=mode, max_steps=horizon
            )
            trajectory_reports.append(report)
            transitions.extend(rows)
    finally:
        if env is not None:
            env.close()

    after_world = client.get_world()
    after_settings = after_world.get_settings()
    after_vehicle_count = len(after_world.get_actors().filter("vehicle.*"))
    critic_report = _fit_critic_ab(transitions)
    stopped = next(row for row in trajectory_reports if row["trajectory"] == "stopped")
    smooth = next(row for row in trajectory_reports if row["trajectory"] == "smooth")
    collision = next(
        row for row in trajectory_reports if row["trajectory"] == "rear_collision"
    )
    all_residuals = [
        row["reward_conservation_max_abs_residual"] for row in trajectory_reports
    ]
    stopped_max_abs = max(
        abs(agent["discounted_return"]) for agent in stopped["agents"].values()
    )
    smooth_mean_return = float(
        np.mean(
            [agent["discounted_return"] for agent in smooth["agents"].values()]
        )
    )
    collision_mean_return = float(
        np.mean(
            [agent["discounted_return"] for agent in collision["agents"].values()]
        )
    )
    acceptance = {
        "reward_conservation": max(all_residuals) <= 1e-6,
        "stopped_is_zero": stopped_max_abs <= 1e-6,
        "smooth_beats_stopped": smooth_mean_return > stopped_max_abs,
        "collision_below_stopped": collision_mean_return < -stopped_max_abs,
        "collision_event_observed": any(
            agent["collision_events"] > 0
            for agent in collision["agents"].values()
        ),
        "critic_normalized_targets_unit_scale": abs(
            critic_report["r3_valuenorm_huber"]["normalized_train_target_std"] - 1.0
        ) <= 1e-4,
        "critic_clip_pressure_reduced": critic_report["comparison"][
            "clip_rate_reduction"
        ] > 0.0,
        "critic_heldout_mse_improved": critic_report["comparison"][
            "heldout_mse_ratio_r3_over_baseline"
        ] < 1.0,
        "environment_timing_restored": (
            bool(after_settings.synchronous_mode)
            == bool(before_settings.synchronous_mode)
            and after_settings.fixed_delta_seconds
            == before_settings.fixed_delta_seconds
        ),
        "no_vehicle_actor_leak": after_vehicle_count <= before_vehicle_count,
    }
    report = {
        "diagnostic_protocol_version": "r2_reward_r3_critic_real_carla_v1",
        "created_at": _utc_now(),
        "purpose": "mechanism_validation_not_policy_selection",
        "runtime": {
            "host": args.host,
            "port": args.port,
            "carla_client_version": client_version,
            "carla_server_version": server_version,
            "scenario": str(SCENARIO_PATH.relative_to(PROJECT_DIR)),
            "seed": 42,
            "gamma": GAMMA,
            "before_vehicle_count": before_vehicle_count,
            "after_vehicle_count": after_vehicle_count,
            "before_synchronous_mode": bool(before_settings.synchronous_mode),
            "after_synchronous_mode": bool(after_settings.synchronous_mode),
            "before_fixed_delta_seconds": before_settings.fixed_delta_seconds,
            "after_fixed_delta_seconds": after_settings.fixed_delta_seconds,
        },
        "trajectories": trajectory_reports,
        "critic_ab": critic_report,
        "acceptance": acceptance,
        "overall_pass": all(acceptance.values()),
    }
    output_dir = (
        PROJECT_DIR
        / "diagnostics"
        / "r2_r3_real_carla"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(output_path), **acceptance}, ensure_ascii=False))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
