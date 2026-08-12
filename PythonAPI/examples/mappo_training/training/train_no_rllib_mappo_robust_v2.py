"""
基于 MAPPO 的鲁棒训练脚本 v2
改进项（详见 docs/robust_training_v2_changelog.md）：
1. 增大 rollout_episodes 默认值，降低 ppo_epochs
2. 修复 GAE 跨 episode 泄漏（按 episode 独立计算 + 截断 bootstrap）
3. 场景分层 + 加权采样（保证覆盖 + 困难场景上采样）
4. 课程软过渡（阶段切换时概率混合旧场景）
5. 每个 agent 在完整 rollout 上统一优势标准化（保留 episode 间相对信号）
6. RolloutBuffer 增强（episode 边界追踪 + 截断状态存储）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import copy
import json
import os
import re
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from visualization.mappo_visualizer import MAPPOVisualizer
from models.actor_critic import Actor, AgentConditionedCritic
from training.lifecycle_semantics import (
    bootstrap_state_for_boundary,
    validate_lifecycle_flags,
)
from training.action_protocol import (
    ACTION_PROTOCOL_VERSION,
    POLICY_ACTION_DIM,
    action_protocol_spec,
    action_tanh_to_env,
)
from training.metrics_protocol import METRICS_PROTOCOL_VERSION
from training.value_normalization import (
    VALUE_LOSS_PROTOCOL_VERSION,
    VALUE_NORMALIZATION_PROTOCOL_VERSION,
    ValueNormalizer,
    normalized_huber_loss,
)
from training.experiment_contract import (
    ExperimentContractError,
    apply_experiment_contract,
    enforce_formal_transition_minimum,
    initialize_experiment_evidence,
    update_experiment_evidence,
    validate_resume_experiment_evidence,
    verify_critical_sources_clean,
)
from utils.training_metrics_recorder import TrainingMetricsRecorder
from training.checkpoint_protocol import (
    CHECKPOINT_SCHEMA_VERSION,
    INTERRUPTED_CHECKPOINT_NAME,
    LATEST_CHECKPOINT_NAME,
    LIFECYCLE_PROTOCOL_VERSION,
    MANIFEST_NAME,
    OBSERVATION_PROTOCOL_VERSION,
    PERIODIC_DIR_NAME,
    TRAINING_PROTOCOL_VERSION,
    CheckpointProtocolError,
    atomic_copy_checkpoint,
    atomic_write_json,
    capture_rng_state,
    ensure_fresh_run_directory,
    files_fingerprint,
    get_git_metadata,
    infer_run_root,
    load_checkpoint_file,
    load_manifest,
    normalize_command_line,
    resolve_resume_path,
    restore_rng_state,
    runtime_metadata,
    save_checkpoint_file,
    scenario_pool_fingerprint,
    sha256_file,
    utc_now_iso,
    validate_checkpoint_header,
    validate_checkpoint_payload,
)


VALID_EPISODE_END_REASONS = frozenset({
    "running",
    "cooperative_success",
    "all_agents_reached",
    "cooperative_collision",
    "adversary_collision",
    "max_steps",
    "all_agents_terminated",
})
BASE_SUCCESS_END_REASONS = frozenset({"cooperative_success", "all_agents_reached"})
BASE_COLLISION_END_REASONS = frozenset(
    {"cooperative_collision", "adversary_collision"}
)

OBS_DIM = 19
ACTION_DIM = POLICY_ACTION_DIM


def classify_base_episode_outcome(
    episode_end_reason: str,
) -> Tuple[bool, bool, bool]:
    """Return mutually exclusive success/collision/timeout terminal outcomes."""
    reason = str(episode_end_reason)
    success = reason in BASE_SUCCESS_END_REASONS
    collision = reason in BASE_COLLISION_END_REASONS
    timeout = reason == "max_steps"
    if sum((success, collision, timeout)) != 1:
        raise RuntimeError(
            f"base episode ended with non-classifiable reason={reason!r}"
        )
    return success, collision, timeout

RESUME_IMMUTABLE_ARG_NAMES = (
    "town",
    "straight_road_length",
    "timestep",
    "max_steps",
    "rollout_episodes",
    "ppo_epochs",
    "minibatch_size",
    "min_transitions_per_update",
    "lr_actor",
    "lr_critic",
    "gamma",
    "gae_lambda",
    "clip_eps",
    "ent_coef",
    "max_grad_norm",
    "value_norm_beta",
    "value_norm_epsilon",
    "value_norm_variance_floor",
    "huber_delta",
    "hidden_dim",
    "target_kl",
    "lr_decay",
    "init_log_std",
    "scenario_sampling_mode",
    "warmup_updates",
    "full_scenario_updates",
    "curriculum_transition_window",
    "curriculum_mode",
    "reward_version",
    "save_interval",
    "log_interval",
    "tb_histogram_interval",
)


def set_seed(seed: int):
    # 修复 #6：完整播种
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def checkpoint_protocol_versions(reward_version: str) -> Dict[str, str]:
    return {
        "observation": OBSERVATION_PROTOCOL_VERSION,
        "action": ACTION_PROTOCOL_VERSION,
        "lifecycle": LIFECYCLE_PROTOCOL_VERSION,
        "reward": str(reward_version),
        "metrics": METRICS_PROTOCOL_VERSION,
        "value_normalization": VALUE_NORMALIZATION_PROTOCOL_VERSION,
        "value_loss": VALUE_LOSS_PROTOCOL_VERSION,
    }


def checkpoint_training_config(args) -> Dict[str, object]:
    return {
        name: getattr(args, name) for name in RESUME_IMMUTABLE_ARG_NAMES
    }


def checkpoint_model_spec(
    agent_ids: List[int], state_dim: int, hidden_dim: int
) -> Dict[str, object]:
    return {
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "state_dim": int(state_dim),
        "num_agents": len(agent_ids),
        "agent_ids": list(agent_ids),
        "hidden_dim": int(hidden_dim),
        "critic_type": "AgentConditionedCritic",
        "critic_output_semantics": "normalized_value",
    }


def _argument_was_explicit(args, name: str) -> bool:
    explicit = set(getattr(args, "_explicit_cli_options", set()))
    if name == "lr_decay":
        return bool({"--lr-decay", "--no-lr-decay"} & explicit)
    return f"--{name.replace('_', '-')}" in explicit


def apply_resume_configuration(args, checkpoint: Dict[str, object]) -> None:
    """Restore frozen run configuration while rejecting explicit drift."""
    saved_config = checkpoint["training_config"]
    for name in RESUME_IMMUTABLE_ARG_NAMES:
        if name not in saved_config:
            raise CheckpointProtocolError(
                f"saved training_config missing field: {name}"
            )
        saved_value = saved_config[name]
        current_value = getattr(args, name)
        if _argument_was_explicit(args, name) and current_value != saved_value:
            raise CheckpointProtocolError(
                f"resume cannot change --{name.replace('_', '-')}: "
                f"saved={saved_value!r}, requested={current_value!r}; "
                "use --pretrained-checkpoint to start a new run"
            )
        setattr(args, name, saved_value)

    saved_seed = int(checkpoint["run_metadata"]["seed"])
    if _argument_was_explicit(args, "seed") and int(args.seed) != saved_seed:
        raise CheckpointProtocolError(
            f"resume seed mismatch: saved={saved_seed}, requested={args.seed}"
        )
    args.seed = saved_seed

    saved_run_id = str(checkpoint["run_metadata"]["run_id"])
    if args.run_id and args.run_id != saved_run_id:
        raise CheckpointProtocolError(
            f"resume run_id mismatch: saved={saved_run_id!r}, "
            f"requested={args.run_id!r}"
        )
    args.run_id = saved_run_id

    saved_horizon = int(checkpoint["lr_schedule"]["decay_horizon"])
    if args.lr_decay_horizon not in (0, saved_horizon):
        raise CheckpointProtocolError(
            f"resume LR horizon mismatch: saved={saved_horizon}, "
            f"requested={args.lr_decay_horizon}"
        )
    args.lr_decay_horizon = saved_horizon


def _load_pretrained_weights(
    checkpoint: Dict[str, object],
    actors: Dict[int, Actor],
    critic: AgentConditionedCritic,
    value_normalizer: ValueNormalizer,
) -> Dict[str, object]:
    """Load current R3 weights only; progress/RNG deliberately remain fresh."""
    validate_checkpoint_header(checkpoint)
    if checkpoint.get("action_spec") != action_protocol_spec():
        raise CheckpointProtocolError(
            "pretrained checkpoint action_spec is not signed-longitudinal v2; "
            "legacy WTA weights are historical-diagnostic only"
        )
    model_spec = checkpoint.get("model_spec")
    if (
        not isinstance(model_spec, dict)
        or model_spec.get("action_dim") != ACTION_DIM
        or model_spec.get("agent_ids") != sorted(actors)
    ):
        raise CheckpointProtocolError(
            "pretrained checkpoint model_spec is incompatible with the "
            "current two-dimensional base Actor"
        )
    actor_states = checkpoint.get("actors")
    if not isinstance(actor_states, dict):
        raise CheckpointProtocolError(
            "pretrained checkpoint must contain an actors dictionary"
        )
    for aid, actor in actors.items():
        state = actor_states.get(aid, actor_states.get(str(aid)))
        if state is None:
            raise CheckpointProtocolError(
                f"pretrained checkpoint missing actor {aid}"
            )
        actor.load_state_dict(state, strict=True)

    critic_loaded = False
    critic_error = "missing critic"
    if "critic" in checkpoint and "value_normalizer" in checkpoint:
        critic_before = copy.deepcopy(critic.state_dict())
        normalizer_before = copy.deepcopy(value_normalizer.state_dict())
        try:
            critic.load_state_dict(checkpoint["critic"], strict=True)
            value_normalizer.load_state_dict(
                checkpoint["value_normalizer"], strict=True
            )
            value_normalizer.validate_state()
            critic_loaded = True
            critic_error = ""
        except (RuntimeError, ValueError) as error:
            critic.load_state_dict(critic_before, strict=True)
            value_normalizer.load_state_dict(normalizer_before, strict=True)
            critic_error = str(error)
    return {
        "actors_loaded": sorted(actors),
        "critic_loaded": critic_loaded,
        "critic_note": critic_error,
    }


def runtime_metadata_with_carla(env) -> Dict[str, object]:
    metadata = runtime_metadata()
    metadata["environment_reward_gamma"] = float(env.reward_gamma)
    try:
        metadata["carla_client"] = env.client.get_client_version()
        metadata["carla_server"] = env.client.get_server_version()
    except Exception as error:
        metadata["carla_version_error"] = str(error)
    return metadata


def build_training_environment_kwargs(args, *, num_agents: int) -> Dict[str, object]:
    """Build the base environment from the same frozen arguments used by PPO."""
    reward_gamma = float(args.gamma)
    if not 0.0 < reward_gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    return {
        "host": args.host,
        "port": args.port,
        "town": args.town,
        "num_agents": int(num_agents),
        "straight_road_length": args.straight_road_length,
        "timestep": args.timestep,
        "max_steps": args.max_steps,
        "spawn_vehicles": True,
        "reward_version": args.reward_version,
        "reward_gamma": reward_gamma,
        "disable_tensorboard": True,
        "seed": args.seed,
        "follow_spectator": args.follow_spectator,
    }


def validate_environment_reward_discount(env, args) -> None:
    """Fail closed if PBRS and PPO/GAE use different discount factors."""
    environment_gamma = float(getattr(env, "reward_gamma", float("nan")))
    ppo_gamma = float(args.gamma)
    if not np.isfinite(environment_gamma) or environment_gamma != ppo_gamma:
        raise CheckpointProtocolError(
            "reward discount mismatch: "
            f"environment PBRS gamma={environment_gamma!r}, "
            f"PPO/GAE gamma={ppo_gamma!r}"
        )


# ============================================================
# 数据结构和 Buffer
# ============================================================

@dataclass
class Transition:
    obs: np.ndarray
    global_state: np.ndarray
    action_z: np.ndarray          # 修复 #4：存 pre-tanh z
    log_prob: float
    reward: float
    done: float                   # 回合边界（切断 GAE trace）
    terminated: float             # 真正终止（切断 value bootstrap）；截断时为 0
    value: float                  # 该 agent 的 V(s, aid)
    scenario_name: str


class RolloutBuffer:
    """增强版 RolloutBuffer：支持 episode 边界追踪"""

    def __init__(self, agent_ids: List[int]):
        self.data: Dict[int, List[Transition]] = {aid: [] for aid in agent_ids}
        self.scenario_names: List[str] = []
        # 每个 agent 的 episode 起止索引 [(start, end), ...]
        self.episode_boundaries: Dict[int, List[Tuple[int, int]]] = {aid: [] for aid in agent_ids}
        # 每个 agent 每个 episode 的最后 global_state（截断时用于 GAE bootstrap）
        # 与 episode_boundaries[aid] 一一对应，None 表示自然终止
        self.episode_bootstrap_states: Dict[int, List[Optional[np.ndarray]]] = {
            aid: [] for aid in agent_ids
        }

    def add(self, agent_id: int, transition: Transition):
        self.data[agent_id].append(transition)

    def add_scenario(self, scenario_name: str):
        self.scenario_names.append(scenario_name)

    def mark_episode_end(self, agent_id: int):
        """在 episode 结束时调用，记录当前 episode 的数据边界"""
        current_len = len(self.data[agent_id])
        boundaries = self.episode_boundaries[agent_id]
        if not boundaries:
            start = 0
        else:
            start = boundaries[-1][1]
        if current_len > start:
            boundaries.append((start, current_len))

    def add_episode_bootstrap(self, agent_id: int, state: Optional[np.ndarray]):
        """记录 episode 的 GAE bootstrap state（截断时为最后 global_state，自然终止为 None）"""
        self.episode_bootstrap_states[agent_id].append(state)

    def get_agent_data(self, agent_id: int) -> List[Transition]:
        return self.data[agent_id]

    def total_size(self) -> int:
        return sum(len(v) for v in self.data.values())

    def clear(self):
        for k in self.data:
            self.data[k] = []
        self.scenario_names = []
        for k in self.episode_boundaries:
            self.episode_boundaries[k] = []
        for k in self.episode_bootstrap_states:
            self.episode_bootstrap_states[k] = []


# ============================================================
# 工具函数
# ============================================================

def validate_active_mask(
    raw_active_mask: Dict[int, bool],
    agent_ids: List[int],
    *,
    context: str,
) -> Dict[int, bool]:
    """Validate and copy the authoritative per-agent decision mask."""
    if not isinstance(raw_active_mask, dict):
        raise TypeError(f"{context}: active_mask must be a dict")

    expected = set(agent_ids)
    actual = set(raw_active_mask.keys())
    if actual != expected:
        raise ValueError(
            f"{context}: active_mask keys mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    active_mask: Dict[int, bool] = {}
    for aid in agent_ids:
        value = raw_active_mask[aid]
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"{context}: active_mask[{aid}] must be bool, got {type(value).__name__}"
            )
        active_mask[aid] = bool(value)
    return active_mask


def build_active_step_cache(
    obs_dict: Dict[int, np.ndarray],
    active_mask: Dict[int, bool],
    global_state_np: np.ndarray,
    actors: Dict[int, Actor],
    critic: AgentConditionedCritic,
    agent_ids: List[int],
    device: torch.device,
    *,
    context: str,
    value_normalizer: Optional[ValueNormalizer] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    """Sample exactly one decision for each currently active agent.

    ``active_mask`` is aligned with the state before ``env.step``. The cache is
    therefore also the authoritative set of transitions to write after the
    step, including an action that makes an agent terminate on that step.
    """
    current_active = validate_active_mask(active_mask, agent_ids, context=context)
    active_ids = [aid for aid in agent_ids if current_active[aid]]

    missing_obs = [aid for aid in active_ids if aid not in obs_dict]
    missing_actors = [aid for aid in active_ids if aid not in actors]
    if missing_obs or missing_actors:
        raise RuntimeError(
            f"{context}: active agents missing rollout inputs: "
            f"obs={missing_obs}, actors={missing_actors}"
        )

    global_state_np = np.asarray(global_state_np, dtype=np.float32)
    global_state_t = torch.as_tensor(
        global_state_np, dtype=torch.float32, device=device
    ).unsqueeze(0)
    action_dict: Dict[int, np.ndarray] = {}
    step_cache: Dict[int, Dict] = {}

    with torch.no_grad():
        for aid in active_ids:
            obs = np.asarray(obs_dict[aid], dtype=np.float32)
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            normalized_value = critic(global_state_t, aid)
            value_tensor = (
                value_normalizer.denormalize(normalized_value)
                if value_normalizer is not None
                else normalized_value
            )
            value = float(value_tensor.item())
            action_tanh_t, logp_t, z_t = actors[aid].sample_action_z(obs_t)

            action_tanh = action_tanh_t.squeeze(0).cpu().numpy()
            action_z = z_t.squeeze(0).cpu().numpy()
            action_dict[aid] = action_tanh_to_env(action_tanh)
            step_cache[aid] = {
                "obs": obs.copy(),
                "global_state": global_state_np.copy(),
                "action_tanh": action_tanh.copy(),
                "action_z": action_z.copy(),
                "logp": float(logp_t.item()),
                "value": value,
            }

    expected_active = set(active_ids)
    if set(action_dict) != expected_active or set(step_cache) != expected_active:
        raise RuntimeError(
            f"{context}: action/cache keys do not match active agents: "
            f"active={sorted(expected_active)}, actions={sorted(action_dict)}, "
            f"cache={sorted(step_cache)}"
        )
    return action_dict, step_cache


def validate_rollout_step_protocol(
    dones: Dict[int, bool],
    terminated: Dict[int, bool],
    truncated: Dict[int, bool],
    next_active_mask: Dict[int, bool],
    episode_end_reason: str,
    agent_ids: List[int],
    *,
    context: str,
) -> Tuple[Dict[int, bool], Dict[int, bool], Dict[int, bool], Dict[int, bool]]:
    """Fail fast when lifecycle flags, next mask, and global reason disagree."""
    expected = set(agent_ids)
    for name, mapping in (
        ("dones", dones),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if not isinstance(mapping, dict) or set(mapping.keys()) != expected:
            actual = set(mapping.keys()) if isinstance(mapping, dict) else set()
            raise ValueError(
                f"{context}: {name} keys mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )

    normalized_active = validate_active_mask(
        next_active_mask, agent_ids, context=f"{context} next"
    )
    normalized_dones: Dict[int, bool] = {}
    normalized_terminated: Dict[int, bool] = {}
    normalized_truncated: Dict[int, bool] = {}
    for aid in agent_ids:
        done, term, trunc = validate_lifecycle_flags(
            dones[aid], terminated[aid], truncated[aid],
            context=f"{context} agent={aid}",
        )
        normalized_dones[aid] = done
        normalized_terminated[aid] = term
        normalized_truncated[aid] = trunc
        if normalized_active[aid] != (not done):
            raise RuntimeError(
                f"{context} agent={aid}: next active={normalized_active[aid]} "
                f"must equal not done={not done}"
            )

    if episode_end_reason not in VALID_EPISODE_END_REASONS:
        raise ValueError(
            f"{context}: unknown episode_end_reason={episode_end_reason!r}; "
            f"expected one of {sorted(VALID_EPISODE_END_REASONS)}"
        )
    if episode_end_reason == "running":
        if not any(normalized_active.values()):
            raise RuntimeError(
                f"{context}: episode is running but no active agent remains"
            )
        local_truncations = [
            aid for aid in agent_ids if normalized_truncated[aid]
        ]
        if local_truncations:
            raise RuntimeError(
                f"{context}: local truncation while episode is running is unsupported; "
                f"agents={local_truncations}"
            )
    elif any(normalized_active.values()):
        raise RuntimeError(
            f"{context}: episode ended with reason={episode_end_reason!r} "
            f"but active agents remain="
            f"{[aid for aid in agent_ids if normalized_active[aid]]}"
        )

    return (
        normalized_active,
        normalized_dones,
        normalized_terminated,
        normalized_truncated,
    )


def refresh_observations_after_active_change(
    env,
    next_obs: Dict[int, np.ndarray],
    current_active_mask: Dict[int, bool],
    next_active_mask: Dict[int, bool],
    episode_end_reason: str,
    agent_ids: List[int],
    *,
    context: str,
) -> Dict[int, np.ndarray]:
    """Refresh next-decision observations after a local lifecycle change.

    The environment intentionally captures ``next_obs`` before lifecycle
    cleanup so that a global truncation can bootstrap from the same physical
    frame. If one agent terminates locally while the episode keeps running,
    however, reached/dead actors may be despawned after that snapshot. In that
    case only, rebuild the fixed-width observation slots from the post-cleanup
    global state before the next action. The caller must preserve the original
    ``next_obs`` separately for any boundary bootstrap.
    """
    current_active = validate_active_mask(
        current_active_mask, agent_ids, context=f"{context} current"
    )
    next_active = validate_active_mask(
        next_active_mask, agent_ids, context=f"{context} next"
    )
    if episode_end_reason != "running" or current_active == next_active:
        return next_obs

    expected = set(agent_ids)
    if not isinstance(next_obs, dict) or set(next_obs.keys()) != expected:
        actual = set(next_obs.keys()) if isinstance(next_obs, dict) else set()
        raise ValueError(
            f"{context}: next_obs keys mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    observation_sizes: Dict[int, int] = {}
    for aid in agent_ids:
        obs = np.asarray(next_obs[aid], dtype=np.float32)
        if obs.ndim != 1:
            raise ValueError(
                f"{context}: next_obs[{aid}] must be 1-D, got shape={obs.shape}"
            )
        observation_sizes[aid] = int(obs.shape[0])

    refreshed_global_state = np.asarray(
        env.get_global_state(), dtype=np.float32
    ).reshape(-1)
    expected_size = sum(observation_sizes.values())
    if refreshed_global_state.size != expected_size:
        raise RuntimeError(
            f"{context}: refreshed global state size={refreshed_global_state.size} "
            f"does not match fixed observation slots={expected_size}"
        )

    refreshed_obs: Dict[int, np.ndarray] = {}
    offset = 0
    for aid in agent_ids:
        size = observation_sizes[aid]
        refreshed_obs[aid] = refreshed_global_state[offset:offset + size].copy()
        offset += size
    return refreshed_obs


def write_active_transitions(
    buffer: RolloutBuffer,
    step_cache: Dict[int, Dict],
    rewards: Dict[int, float],
    dones: Dict[int, bool],
    terminated: Dict[int, bool],
    truncated: Dict[int, bool],
    scenario_name: str,
    *,
    context: str,
) -> Dict[int, float]:
    """Write only decisions present in the pre-step active cache."""
    written_rewards: Dict[int, float] = {}
    for aid, cached in step_cache.items():
        for name, mapping in (
            ("rewards", rewards),
            ("dones", dones),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if aid not in mapping:
                raise KeyError(f"{context} agent={aid}: missing {name} entry")

        done, term, _trunc = validate_lifecycle_flags(
            dones[aid], terminated[aid], truncated[aid],
            context=f"{context} agent={aid}",
        )
        reward = float(rewards[aid])
        buffer.add(
            aid,
            Transition(
                obs=cached["obs"],
                global_state=cached["global_state"],
                action_z=cached["action_z"],
                log_prob=cached["logp"],
                reward=reward,
                done=float(done),
                terminated=float(term),
                value=cached["value"],
                scenario_name=scenario_name,
            ),
        )
        written_rewards[aid] = reward
    return written_rewards


def validate_episode_transition_counts(
    buffer: RolloutBuffer,
    buffer_starts: Dict[int, int],
    active_decision_steps: Dict[int, int],
    agent_ids: List[int],
    *,
    context: str,
) -> Dict[int, int]:
    """Require one and only one buffer row for every active decision."""
    transition_counts: Dict[int, int] = {}
    for aid in agent_ids:
        if aid not in buffer_starts or aid not in active_decision_steps:
            raise KeyError(f"{context} agent={aid}: missing transition counter")
        transition_count = len(buffer.get_agent_data(aid)) - buffer_starts[aid]
        if transition_count != active_decision_steps[aid]:
            raise RuntimeError(
                f"{context} agent={aid}: "
                f"active_decision_steps={active_decision_steps[aid]} != "
                f"buffer_transition_count={transition_count}"
            )
        transition_counts[aid] = transition_count
    return transition_counts


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    terminateds: np.ndarray,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float = 0.0,
    *,
    context: str = "compute_gae",
) -> Tuple[np.ndarray, np.ndarray]:
    """在单个完整 episode 内计算未标准化 GAE。

    修复 #2：用 terminated（而非 done）控制 value bootstrap。
    env 在 max_steps 时也会置 done=True，若用 done 当 mask 会把截断 episode
    的 bootstrap_value 一并抹掉（旧实现的隐藏 bug）。这里：
        mask_value = 1 - terminated   # 真终止才切断 bootstrap
        mask_trace = 1 - done         # 回合边界切断 trace
    Args:
        bootstrap_value: 截断 episode 的 V(s_T)，自然终止时使用 0.0
    """
    vectors = {}
    for name, values_in in (
        ("rewards", rewards),
        ("values", values),
        ("dones", dones),
        ("terminateds", terminateds),
    ):
        values_np = np.asarray(values_in, dtype=np.float32)
        if values_np.ndim != 1:
            raise ValueError(
                f"{context}: {name} must be 1-D, got shape={values_np.shape}"
            )
        if not np.all(np.isfinite(values_np)):
            raise FloatingPointError(f"{context}: {name} contains NaN/Inf")
        vectors[name] = values_np

    rewards = vectors["rewards"]
    values = vectors["values"]
    dones = vectors["dones"]
    terminateds = vectors["terminateds"]
    n_steps = len(rewards)
    lengths = {
        "rewards": n_steps,
        "values": len(values),
        "dones": len(dones),
        "terminateds": len(terminateds),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{context}: GAE length mismatch: {lengths}")
    if n_steps == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty.copy()

    for name, flags in (("dones", dones), ("terminateds", terminateds)):
        if not np.all((flags == 0.0) | (flags == 1.0)):
            raise ValueError(f"{context}: {name} must contain only 0/1")
    if np.any(terminateds > dones):
        raise ValueError(f"{context}: terminated requires done=1")
    if np.any(dones[:-1] != 0.0) or dones[-1] != 1.0:
        raise ValueError(
            f"{context}: input must contain exactly one complete episode "
            "with done=1 only at the final transition"
        )
    if not np.isfinite(bootstrap_value):
        raise FloatingPointError(f"{context}: bootstrap_value is NaN/Inf")

    advantages = np.zeros_like(rewards)
    gae = 0.0
    next_value = float(bootstrap_value)

    for t in reversed(range(len(rewards))):
        mask_value = 1.0 - terminateds[t]
        mask_trace = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask_value - values[t]
        gae = delta + gamma * gae_lambda * mask_trace * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    if not np.all(np.isfinite(advantages)) or not np.all(np.isfinite(returns)):
        raise FloatingPointError(f"{context}: GAE produced NaN/Inf")
    return advantages, returns


def compute_raw_gae_per_episode(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    terminateds: np.ndarray,
    boundaries: List[Tuple[int, int]],
    gamma: float,
    gae_lambda: float,
    bootstrap_values: Optional[List[float]] = None,
    *,
    context: str = "compute_raw_gae_per_episode",
) -> Tuple[np.ndarray, np.ndarray]:
    """按严格 episode 边界计算原始 GAE，不做任何优势标准化。

    返回数组与输入 transition 保持同一时间顺序。边界必须连续、无重叠地
    覆盖全部样本；bootstrap_values 必须和 boundaries 一一对应。
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    terminateds = np.asarray(terminateds, dtype=np.float32)
    arrays = {
        "rewards": rewards,
        "values": values,
        "dones": dones,
        "terminateds": terminateds,
    }
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(
                f"{context}: {name} must be 1-D, got shape={array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{context}: {name} contains NaN/Inf")
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{context}: rollout length mismatch: {lengths}")

    if bootstrap_values is None:
        bootstrap_values = [0.0] * len(boundaries)
    if len(bootstrap_values) != len(boundaries):
        raise ValueError(
            f"{context}: bootstrap/boundary count mismatch: "
            f"{len(bootstrap_values)} != {len(boundaries)}"
        )

    n_steps = len(rewards)
    if n_steps == 0:
        if boundaries:
            raise ValueError(f"{context}: empty rollout cannot have boundaries")
        empty = np.array([], dtype=np.float32)
        return empty, empty.copy()
    if not boundaries:
        raise ValueError(f"{context}: non-empty rollout has no episode boundaries")

    all_advantages = np.empty(n_steps, dtype=np.float32)
    all_returns = np.empty(n_steps, dtype=np.float32)
    expected_start = 0
    for i, (start, end) in enumerate(boundaries):
        if not isinstance(start, (int, np.integer)) or not isinstance(
            end, (int, np.integer)
        ):
            raise TypeError(f"{context}: boundary {i} must contain integer indices")
        start, end = int(start), int(end)
        if start != expected_start:
            raise ValueError(
                f"{context}: boundary {i} starts at {start}, "
                f"expected contiguous start {expected_start}"
            )
        if end <= start:
            raise ValueError(f"{context}: boundary {i} is empty/reversed: {(start, end)}")
        if end > n_steps:
            raise ValueError(
                f"{context}: boundary {i} ends at {end}, rollout length={n_steps}"
            )
        ep_rewards = rewards[start:end]
        ep_values = values[start:end]
        ep_dones = dones[start:end]
        ep_terminateds = terminateds[start:end]
        bv = float(bootstrap_values[i])
        if ep_terminateds[-1] == 1.0 and abs(bv) > 1e-6:
            raise ValueError(
                f"{context}: terminated episode {i} must use zero bootstrap, "
                f"got {bv}"
            )
        adv, ret = compute_gae(
            ep_rewards,
            ep_values,
            ep_dones,
            ep_terminateds,
            gamma,
            gae_lambda,
            bootstrap_value=bv,
            context=f"{context} episode={i} range=({start},{end})",
        )
        all_advantages[start:end] = adv
        all_returns[start:end] = ret
        expected_start = end

    if expected_start != n_steps:
        raise ValueError(
            f"{context}: boundaries cover [0,{expected_start}), "
            f"rollout length={n_steps}"
        )
    return all_advantages, all_returns


def normalize_advantages_over_rollout(
    raw_advantages: np.ndarray,
    *,
    min_std: float = 1e-6,
    context: str = "normalize_advantages_over_rollout",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Normalize one agent's complete rollout once.

    Degenerate batches (fewer than two rows or near-zero variance) are kept
    unchanged so a valid policy signal is not replaced by NaN or amplified
    floating-point noise.
    """
    raw_advantages = np.asarray(raw_advantages, dtype=np.float32)
    if raw_advantages.ndim != 1:
        raise ValueError(
            f"{context}: advantages must be 1-D, got shape={raw_advantages.shape}"
        )
    if raw_advantages.size == 0:
        raise ValueError(f"{context}: cannot normalize an empty rollout")
    if not np.all(np.isfinite(raw_advantages)):
        raise FloatingPointError(f"{context}: raw advantages contain NaN/Inf")
    if not np.isfinite(min_std) or min_std <= 0:
        raise ValueError(f"{context}: min_std must be finite and positive")

    raw_mean = float(np.mean(raw_advantages, dtype=np.float64))
    raw_std = float(np.std(raw_advantages, dtype=np.float64))
    applied = raw_advantages.size >= 2 and raw_std > min_std
    if applied:
        normalized = (
            (raw_advantages.astype(np.float64) - raw_mean) / raw_std
        ).astype(np.float32)
    else:
        normalized = raw_advantages.copy()

    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError(f"{context}: normalization produced NaN/Inf")
    stats = {
        "count": float(raw_advantages.size),
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "normalized_mean": float(np.mean(normalized, dtype=np.float64)),
        "normalized_std": float(np.std(normalized, dtype=np.float64)),
        "applied": 1.0 if applied else 0.0,
    }
    return normalized, stats


def compute_gae_per_episode(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    terminateds: np.ndarray,
    boundaries: List[Tuple[int, int]],
    gamma: float,
    gae_lambda: float,
    bootstrap_values: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility path for not-yet-migrated adversarial/PSRO entrypoints.

    P0-00D intentionally moves ``robust_v2`` to
    :func:`compute_raw_gae_per_episode` plus one per-agent rollout
    normalization. This wrapper preserves the previous per-episode
    normalization for imports outside the current base-policy scope.
    """
    raw_advantages, returns = compute_raw_gae_per_episode(
        rewards=rewards,
        values=values,
        dones=dones,
        terminateds=terminateds,
        boundaries=boundaries,
        gamma=gamma,
        gae_lambda=gae_lambda,
        bootstrap_values=bootstrap_values,
        context="legacy compute_gae_per_episode",
    )
    legacy_advantages = raw_advantages.copy()
    for start, end in boundaries:
        ep_advantages = legacy_advantages[start:end]
        ep_std = float(np.std(ep_advantages))
        if ep_std > 1e-6:
            legacy_advantages[start:end] = (
                ep_advantages - float(np.mean(ep_advantages))
            ) / (ep_std + 1e-8)
    return legacy_advantages, returns


def ensure_finite_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    context: str,
) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{context}: {name} contains NaN/Inf")


def ensure_finite_module_gradients(module: nn.Module, *, context: str) -> None:
    found_gradient = False
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        found_gradient = True
        ensure_finite_tensor(
            parameter.grad, name=f"gradient[{name}]", context=context
        )
    if not found_gradient:
        raise RuntimeError(f"{context}: module has no gradients after backward")


def ensure_finite_module_parameters(module: nn.Module, *, context: str) -> None:
    for name, parameter in module.named_parameters():
        ensure_finite_tensor(parameter, name=f"parameter[{name}]", context=context)


@torch.no_grad()
def validate_initial_policy_ratio(
    actor: Actor,
    observations: torch.Tensor,
    action_z: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    tolerance: float = 1e-6,
    context: str,
) -> float:
    """Verify on-policy sample/log-prob alignment before any actor update."""
    batch_sizes = {
        "observations": int(observations.shape[0]),
        "action_z": int(action_z.shape[0]),
        "old_log_probs": int(old_log_probs.shape[0]),
    }
    if len(set(batch_sizes.values())) != 1 or batch_sizes["observations"] == 0:
        raise ValueError(f"{context}: PPO batch size mismatch: {batch_sizes}")
    if old_log_probs.ndim != 1:
        raise ValueError(
            f"{context}: old_log_probs must be 1-D, got {old_log_probs.shape}"
        )
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"{context}: tolerance must be finite and positive")

    ensure_finite_tensor(observations, name="observations", context=context)
    ensure_finite_tensor(action_z, name="action_z", context=context)
    ensure_finite_tensor(old_log_probs, name="old_log_probs", context=context)
    # Rollout inference calls the actor once per active agent with batch size 1.
    # Reproduce that geometry here: a large batched GEMM can differ from the
    # batch-1 forward by several float32 ulps even with identical parameters,
    # producing a false stale-policy failure at the strict 1e-6 threshold.
    # PPO optimization remains minibatched; only this preflight check is
    # transition-wise so it compares like with like.
    log_prob_parts = []
    entropy_parts = []
    for index in range(batch_sizes["observations"]):
        log_prob_part, entropy_part = actor.evaluate_actions_z(
            observations[index:index + 1],
            action_z[index:index + 1],
        )
        log_prob_parts.append(log_prob_part)
        entropy_parts.append(entropy_part)
    new_log_probs = torch.cat(log_prob_parts, dim=0)
    entropy = torch.cat(entropy_parts, dim=0)
    ensure_finite_tensor(new_log_probs, name="new_log_probs", context=context)
    ensure_finite_tensor(entropy, name="entropy", context=context)
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError(
            f"{context}: log-prob shape mismatch: "
            f"new={new_log_probs.shape}, old={old_log_probs.shape}"
        )

    ratios = torch.exp(new_log_probs - old_log_probs)
    ensure_finite_tensor(ratios, name="ratios", context=context)
    max_abs_error = float(torch.max(torch.abs(ratios - 1.0)).item())
    if max_abs_error >= tolerance:
        raise RuntimeError(
            f"{context}: pre-update PPO ratio mismatch: "
            f"max_abs(ratio-1)={max_abs_error:.9g} >= {tolerance:.9g}"
        )
    return max_abs_error


def compute_ppo_diagnostics(
    logratio: torch.Tensor,
    ratio: torch.Tensor,
    clip_eps: float,
    *,
    context: str,
) -> Tuple[float, float]:
    """Return finite Joschu approximate KL and clip fraction."""
    ensure_finite_tensor(logratio, name="logratio", context=context)
    ensure_finite_tensor(ratio, name="ratio", context=context)
    if logratio.shape != ratio.shape:
        raise ValueError(
            f"{context}: ratio shape mismatch: "
            f"logratio={logratio.shape}, ratio={ratio.shape}"
        )
    if not np.isfinite(clip_eps) or clip_eps <= 0:
        raise ValueError(f"{context}: clip_eps must be finite and positive")

    approx_kl = float(((ratio - 1.0) - logratio).mean().item())
    clip_fraction = float(
        ((ratio - 1.0).abs() > clip_eps).float().mean().item()
    )
    if not np.isfinite(approx_kl) or approx_kl < -1e-6:
        raise FloatingPointError(
            f"{context}: invalid approximate KL={approx_kl}"
        )
    if not np.isfinite(clip_fraction) or not 0.0 <= clip_fraction <= 1.0:
        raise FloatingPointError(
            f"{context}: invalid clip fraction={clip_fraction}"
        )
    return approx_kl, clip_fraction


@dataclass
class PreparedMAPPOUpdateBatch:
    """Validated tensors consumed by the production MAPPO update."""

    per_agent_tensors: Dict[int, Dict[str, torch.Tensor]]
    advantage_stats: Dict[int, Dict[str, float]]
    critic_states: torch.Tensor
    critic_returns: torch.Tensor
    critic_agent_ids: torch.Tensor
    transition_counts: Dict[int, int]


@dataclass
class MAPPOUpdateResult:
    metrics: Dict[str, object]
    agent_stop_epoch: Dict[int, int]
    preupdate_ratio_errors: Dict[int, float]


def _stack_transition_vectors(
    data: List[Transition],
    field_name: str,
    expected_dim: int,
    *,
    context: str,
) -> np.ndarray:
    vectors = []
    for index, transition in enumerate(data):
        vector = np.asarray(getattr(transition, field_name), dtype=np.float32)
        if vector.shape != (expected_dim,):
            raise ValueError(
                f"{context}: {field_name}[{index}] shape={vector.shape}, "
                f"expected ({expected_dim},)"
            )
        if not np.all(np.isfinite(vector)):
            raise FloatingPointError(
                f"{context}: {field_name}[{index}] contains NaN/Inf"
            )
        vectors.append(vector)
    return np.stack(vectors, axis=0)


def validate_prepared_mappo_batch(
    prepared: PreparedMAPPOUpdateBatch,
    agent_ids: List[int],
    *,
    obs_dim: int,
    state_dim: int,
    action_dim: int,
    context: str,
) -> None:
    """Validate ownership, dimensions, ordering, and finiteness before update."""
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError(f"{context}: duplicate agent ids: {agent_ids}")
    expected_agents = set(agent_ids)
    batch_agents = set(prepared.per_agent_tensors)
    if not batch_agents.issubset(expected_agents):
        raise ValueError(
            f"{context}: unexpected agents in prepared batch: "
            f"{sorted(batch_agents - expected_agents)}"
        )
    if set(prepared.advantage_stats) != batch_agents:
        raise ValueError(f"{context}: advantage stats do not match actor batches")
    if set(prepared.transition_counts) != expected_agents:
        raise ValueError(f"{context}: transition counts do not cover all agents")

    required_fields = {
        "obs",
        "act",
        "old_logp",
        "adv",
        "raw_adv",
        "old_value",
        "state",
        "ret",
    }
    expected_critic_states = []
    expected_critic_returns = []
    expected_critic_agent_ids = []
    for aid in agent_ids:
        if aid not in batch_agents:
            if prepared.transition_counts[aid] != 0:
                raise ValueError(
                    f"{context}: agent {aid} has no batch but transition_count="
                    f"{prepared.transition_counts[aid]}"
                )
            continue
        tensors = prepared.per_agent_tensors[aid]
        if set(tensors) != required_fields:
            raise ValueError(
                f"{context}: agent {aid} tensor fields mismatch: "
                f"missing={sorted(required_fields - set(tensors))}, "
                f"extra={sorted(set(tensors) - required_fields)}"
            )
        n_rows = int(tensors["obs"].shape[0])
        if n_rows <= 0 or prepared.transition_counts[aid] != n_rows:
            raise ValueError(
                f"{context}: agent {aid} invalid row count: "
                f"batch={n_rows}, transitions={prepared.transition_counts[aid]}"
            )
        expected_shapes = {
            "obs": (n_rows, obs_dim),
            "act": (n_rows, action_dim),
            "old_logp": (n_rows,),
            "adv": (n_rows,),
            "raw_adv": (n_rows,),
            "old_value": (n_rows,),
            "state": (n_rows, state_dim),
            "ret": (n_rows,),
        }
        for name, expected_shape in expected_shapes.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{context}: agent {aid} {name} shape={tuple(tensor.shape)}, "
                    f"expected={expected_shape}"
                )
            ensure_finite_tensor(
                tensor, name=f"agent_{aid}.{name}", context=context
            )
        if not torch.allclose(
            tensors["ret"],
            tensors["raw_adv"] + tensors["old_value"],
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{context}: agent {aid} critic return is not "
                "raw_advantage + old_value"
            )
        expected_critic_states.append(tensors["state"])
        expected_critic_returns.append(tensors["ret"])
        expected_critic_agent_ids.append(
            torch.full(
                (n_rows,), aid, dtype=torch.long, device=tensors["state"].device
            )
        )

    total_rows = sum(prepared.transition_counts.values())
    critic_shapes = {
        "states": tuple(prepared.critic_states.shape),
        "returns": tuple(prepared.critic_returns.shape),
        "agent_ids": tuple(prepared.critic_agent_ids.shape),
    }
    expected_critic_shapes = {
        "states": (total_rows, state_dim),
        "returns": (total_rows,),
        "agent_ids": (total_rows,),
    }
    if critic_shapes != expected_critic_shapes:
        raise ValueError(
            f"{context}: centralized critic shapes mismatch: "
            f"actual={critic_shapes}, expected={expected_critic_shapes}"
        )
    ensure_finite_tensor(
        prepared.critic_states, name="critic_states", context=context
    )
    ensure_finite_tensor(
        prepared.critic_returns, name="critic_returns", context=context
    )
    if prepared.critic_agent_ids.dtype != torch.long:
        raise TypeError(f"{context}: critic_agent_ids must use torch.long")

    if total_rows > 0:
        expected_states = torch.cat(expected_critic_states, dim=0)
        expected_returns = torch.cat(expected_critic_returns, dim=0)
        expected_agent_ids = torch.cat(expected_critic_agent_ids, dim=0)
        if not torch.equal(prepared.critic_states, expected_states):
            raise RuntimeError(f"{context}: critic states lost agent/time ordering")
        if not torch.equal(prepared.critic_returns, expected_returns):
            raise RuntimeError(f"{context}: critic returns lost agent/time ordering")
        if not torch.equal(prepared.critic_agent_ids, expected_agent_ids):
            raise RuntimeError(f"{context}: critic agent ids are misaligned")


def prepare_mappo_update_batch(
    buffer: RolloutBuffer,
    critic: AgentConditionedCritic,
    agent_ids: List[int],
    device: torch.device,
    *,
    gamma: float,
    gae_lambda: float,
    obs_dim: int,
    state_dim: int,
    action_dim: int,
    context: str,
    value_normalizer: Optional[ValueNormalizer] = None,
) -> PreparedMAPPOUpdateBatch:
    """Build the exact actor/critic batches used by the production update."""
    for name, dim in (
        ("obs_dim", obs_dim),
        ("state_dim", state_dim),
        ("action_dim", action_dim),
    ):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{context}: {name} must be a positive integer")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"{context}: gamma must be in [0,1]")
    if not np.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"{context}: gae_lambda must be in [0,1]")

    per_agent_tensors: Dict[int, Dict[str, torch.Tensor]] = {}
    advantage_stats: Dict[int, Dict[str, float]] = {}
    transition_counts: Dict[int, int] = {}
    critic_states = []
    critic_returns = []
    critic_agent_ids = []

    for aid in agent_ids:
        data = buffer.get_agent_data(aid)
        transition_counts[aid] = len(data)
        if not data:
            if buffer.episode_boundaries.get(aid, []):
                raise ValueError(
                    f"{context}: empty agent {aid} data has episode boundaries"
                )
            if buffer.episode_bootstrap_states.get(aid, []):
                raise ValueError(
                    f"{context}: empty agent {aid} data has bootstrap states"
                )
            continue

        agent_context = f"{context} agent={aid}"
        obs_arr = _stack_transition_vectors(
            data, "obs", obs_dim, context=agent_context
        )
        state_arr = _stack_transition_vectors(
            data, "global_state", state_dim, context=agent_context
        )
        act_arr = _stack_transition_vectors(
            data, "action_z", action_dim, context=agent_context
        )
        scalar_fields = {}
        for field_name in (
            "log_prob",
            "reward",
            "done",
            "terminated",
            "value",
        ):
            array = np.asarray(
                [getattr(transition, field_name) for transition in data],
                dtype=np.float32,
            )
            if array.shape != (len(data),):
                raise ValueError(
                    f"{agent_context}: {field_name} shape={array.shape}, "
                    f"expected ({len(data)},)"
                )
            if not np.all(np.isfinite(array)):
                raise FloatingPointError(
                    f"{agent_context}: {field_name} contains NaN/Inf"
                )
            scalar_fields[field_name] = array

        boundaries = buffer.episode_boundaries.get(aid, [])
        bootstrap_states = buffer.episode_bootstrap_states.get(aid, [])
        if len(bootstrap_states) != len(boundaries):
            raise ValueError(
                f"{agent_context}: bootstrap/boundary count mismatch: "
                f"{len(bootstrap_states)} != {len(boundaries)}"
            )
        bootstrap_values = []
        with torch.no_grad():
            for episode_index, bootstrap_state in enumerate(bootstrap_states):
                if bootstrap_state is None:
                    bootstrap_values.append(0.0)
                    continue
                bootstrap_np = np.asarray(bootstrap_state, dtype=np.float32)
                if bootstrap_np.shape != (state_dim,):
                    raise ValueError(
                        f"{agent_context}: bootstrap state {episode_index} "
                        f"shape={bootstrap_np.shape}, expected ({state_dim},)"
                    )
                if not np.all(np.isfinite(bootstrap_np)):
                    raise FloatingPointError(
                        f"{agent_context}: bootstrap state {episode_index} "
                        "contains NaN/Inf"
                    )
                bootstrap_tensor = torch.as_tensor(
                    bootstrap_np, dtype=torch.float32, device=device
                ).unsqueeze(0)
                normalized_prediction = critic(bootstrap_tensor, aid)
                bootstrap_prediction = (
                    value_normalizer.denormalize(normalized_prediction)
                    if value_normalizer is not None
                    else normalized_prediction
                )
                ensure_finite_tensor(
                    bootstrap_prediction,
                    name=f"bootstrap_value[{episode_index}]",
                    context=agent_context,
                )
                bootstrap_values.append(float(bootstrap_prediction.item()))

        raw_adv_arr, ret_arr = compute_raw_gae_per_episode(
            rewards=scalar_fields["reward"],
            values=scalar_fields["value"],
            dones=scalar_fields["done"],
            terminateds=scalar_fields["terminated"],
            boundaries=boundaries,
            gamma=gamma,
            gae_lambda=gae_lambda,
            bootstrap_values=bootstrap_values,
            context=agent_context,
        )
        adv_arr, agent_advantage_stats = normalize_advantages_over_rollout(
            raw_adv_arr, context=f"{agent_context} advantage normalization"
        )
        advantage_stats[aid] = agent_advantage_stats

        expected_returns = raw_adv_arr + scalar_fields["value"]
        if not np.allclose(ret_arr, expected_returns, rtol=1e-6, atol=1e-6):
            raise RuntimeError(
                f"{agent_context}: critic return is not "
                "raw_advantage + old_value"
            )
        per_agent_tensors[aid] = {
            "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
            "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
            "old_logp": torch.as_tensor(
                scalar_fields["log_prob"], dtype=torch.float32, device=device
            ),
            "adv": torch.as_tensor(adv_arr, dtype=torch.float32, device=device),
            "raw_adv": torch.as_tensor(
                raw_adv_arr, dtype=torch.float32, device=device
            ),
            "old_value": torch.as_tensor(
                scalar_fields["value"], dtype=torch.float32, device=device
            ),
            "state": torch.as_tensor(
                state_arr, dtype=torch.float32, device=device
            ),
            "ret": torch.as_tensor(ret_arr, dtype=torch.float32, device=device),
        }
        critic_states.append(state_arr)
        critic_returns.append(ret_arr)
        critic_agent_ids.append(np.full(len(data), aid, dtype=np.int64))

    if critic_states:
        critic_state_tensor = torch.as_tensor(
            np.concatenate(critic_states, axis=0),
            dtype=torch.float32,
            device=device,
        )
        critic_return_tensor = torch.as_tensor(
            np.concatenate(critic_returns, axis=0),
            dtype=torch.float32,
            device=device,
        )
        critic_agent_id_tensor = torch.as_tensor(
            np.concatenate(critic_agent_ids, axis=0),
            dtype=torch.long,
            device=device,
        )
    else:
        critic_state_tensor = torch.empty(
            (0, state_dim), dtype=torch.float32, device=device
        )
        critic_return_tensor = torch.empty((0,), dtype=torch.float32, device=device)
        critic_agent_id_tensor = torch.empty((0,), dtype=torch.long, device=device)

    prepared = PreparedMAPPOUpdateBatch(
        per_agent_tensors=per_agent_tensors,
        advantage_stats=advantage_stats,
        critic_states=critic_state_tensor,
        critic_returns=critic_return_tensor,
        critic_agent_ids=critic_agent_id_tensor,
        transition_counts=transition_counts,
    )
    validate_prepared_mappo_batch(
        prepared,
        agent_ids,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        context=context,
    )
    return prepared


def compute_clipped_actor_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    *,
    clip_eps: float,
    ent_coef: float,
    context: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the production PPO clipped objective and diagnostics."""
    shapes = {
        "new_log_probs": tuple(new_log_probs.shape),
        "old_log_probs": tuple(old_log_probs.shape),
        "advantages": tuple(advantages.shape),
        "entropy": tuple(entropy.shape),
    }
    if len(set(shapes.values())) != 1 or new_log_probs.ndim != 1:
        raise ValueError(f"{context}: actor loss shape mismatch: {shapes}")
    for name, tensor in (
        ("new_log_probs", new_log_probs),
        ("old_log_probs", old_log_probs),
        ("advantages", advantages),
        ("entropy", entropy),
    ):
        ensure_finite_tensor(tensor, name=name, context=context)
    if not np.isfinite(ent_coef) or ent_coef < 0:
        raise ValueError(f"{context}: ent_coef must be finite and non-negative")

    logratio = new_log_probs - old_log_probs
    ratio = torch.exp(logratio)
    approx_kl, clip_fraction = compute_ppo_diagnostics(
        logratio, ratio, clip_eps, context=context
    )
    surrogate_unclipped = ratio * advantages
    surrogate_clipped = (
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    )
    surrogate_loss = -torch.min(
        surrogate_unclipped, surrogate_clipped
    ).mean()
    entropy_bonus = ent_coef * entropy.mean()
    actor_loss = surrogate_loss - entropy_bonus
    ensure_finite_tensor(actor_loss, name="actor_loss", context=context)
    return actor_loss, {
        "surrogate_loss": float(surrogate_loss.item()),
        "entropy": float(entropy.mean().item()),
        "entropy_bonus": float(entropy_bonus.item()),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "ratio_mean": float(ratio.mean().item()),
        "ratio_std": float(ratio.std(unbiased=False).item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_max": float(ratio.max().item()),
    }


def compute_critic_value_loss(
    value_predictions: torch.Tensor,
    returns: torch.Tensor,
    *,
    context: str,
    huber_delta: float = 10.0,
) -> torch.Tensor:
    if value_predictions.ndim != 1 or value_predictions.shape != returns.shape:
        raise ValueError(
            f"{context}: critic loss shape mismatch: "
            f"predictions={tuple(value_predictions.shape)}, "
            f"returns={tuple(returns.shape)}"
        )
    ensure_finite_tensor(
        value_predictions, name="value_predictions", context=context
    )
    ensure_finite_tensor(returns, name="returns", context=context)
    loss = normalized_huber_loss(
        value_predictions, returns, delta=huber_delta
    )
    ensure_finite_tensor(loss, name="critic_loss", context=context)
    return loss


def should_stop_for_kl(approx_kl: float, target_kl: float) -> bool:
    if not np.isfinite(approx_kl) or approx_kl < -1e-6:
        raise ValueError(f"invalid approx_kl={approx_kl}")
    if not np.isfinite(target_kl) or target_kl < 0:
        raise ValueError(f"invalid target_kl={target_kl}")
    return target_kl > 0 and approx_kl > 1.5 * target_kl


def _validate_optimizer_owns_module(
    optimizer: optim.Optimizer,
    module: nn.Module,
    *,
    context: str,
) -> None:
    """Reject crossed Actor/Critic optimizer wiring before any parameter update."""
    if not isinstance(optimizer, optim.Optimizer):
        raise TypeError(f"{context}: optimizer must be torch.optim.Optimizer")
    module_parameter_ids = {id(parameter) for parameter in module.parameters()}
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_parameter_ids != module_parameter_ids:
        raise ValueError(
            f"{context}: optimizer parameters do not exactly match module parameters"
        )


def run_mappo_ppo_update(
    prepared: PreparedMAPPOUpdateBatch,
    actors: Dict[int, Actor],
    actor_optims: Dict[int, optim.Optimizer],
    critic: AgentConditionedCritic,
    critic_optim: optim.Optimizer,
    agent_ids: List[int],
    *,
    obs_dim: int,
    state_dim: int,
    action_dim: int,
    ppo_epochs: int,
    minibatch_size: int,
    clip_eps: float,
    ent_coef: float,
    max_grad_norm: float,
    target_kl: float,
    value_normalizer: Optional[ValueNormalizer] = None,
    huber_delta: float = 10.0,
    shuffle: bool = True,
    context: str,
) -> MAPPOUpdateResult:
    """Execute the exact production MAPPO optimizer path on a prepared batch."""
    validate_prepared_mappo_batch(
        prepared,
        agent_ids,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        context=f"{context} preflight",
    )
    if prepared.critic_states.shape[0] == 0:
        raise ValueError(f"{context}: cannot update from an empty batch")
    if not isinstance(ppo_epochs, int) or ppo_epochs <= 0:
        raise ValueError(f"{context}: ppo_epochs must be a positive integer")
    if not isinstance(minibatch_size, int) or minibatch_size <= 0:
        raise ValueError(f"{context}: minibatch_size must be a positive integer")
    if not np.isfinite(clip_eps) or clip_eps <= 0:
        raise ValueError(f"{context}: clip_eps must be finite and positive")
    if not np.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError(f"{context}: max_grad_norm must be finite and positive")
    if not np.isfinite(ent_coef) or ent_coef < 0:
        raise ValueError(f"{context}: ent_coef must be finite and non-negative")
    if not np.isfinite(target_kl) or target_kl < 0:
        raise ValueError(f"{context}: target_kl must be finite and non-negative")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError(f"{context}: huber_delta must be finite and positive")
    batch_agents = set(prepared.per_agent_tensors)
    for name, mapping in (("actors", actors), ("actor_optims", actor_optims)):
        missing = batch_agents - set(mapping)
        if missing:
            raise KeyError(f"{context}: {name} missing agents={sorted(missing)}")
    if critic.state_dim != state_dim:
        raise ValueError(
            f"{context}: critic state_dim={critic.state_dim}, expected={state_dim}"
        )
    invalid_critic_agent_ids = [
        aid
        for aid in agent_ids
        if not isinstance(aid, int) or not 0 <= aid < critic.num_agents
    ]
    if invalid_critic_agent_ids:
        raise ValueError(
            f"{context}: agent ids incompatible with critic: "
            f"{invalid_critic_agent_ids}"
        )

    _validate_optimizer_owns_module(
        critic_optim, critic, context=f"{context} critic optimizer preflight"
    )
    ensure_finite_module_parameters(critic, context=f"{context} critic preflight")
    with torch.no_grad():
        critic_preflight_values = critic(
            prepared.critic_states, prepared.critic_agent_ids
        )
    ensure_finite_tensor(
        critic_preflight_values,
        name="critic_preflight_values",
        context=f"{context} critic preflight",
    )
    preupdate_ratio_errors: Dict[int, float] = {}
    for aid, tensors in prepared.per_agent_tensors.items():
        _validate_optimizer_owns_module(
            actor_optims[aid],
            actors[aid],
            context=f"{context} actor={aid} optimizer preflight",
        )
        ensure_finite_module_parameters(
            actors[aid], context=f"{context} actor={aid} preflight"
        )
        preupdate_ratio_errors[aid] = validate_initial_policy_ratio(
            actor=actors[aid],
            observations=tensors["obs"],
            action_z=tensors["act"],
            old_log_probs=tensors["old_logp"],
            tolerance=1e-6,
            context=f"{context} actor={aid} pre-update ratio",
        )

    metrics: Dict[str, object] = {
        "critic_loss": [],
        "grad_norm_critic": [],
        "critic_samples_seen": 0,
        "explained_variance": 0.0,
    }
    for aid in agent_ids:
        metrics.update({
            f"actor_loss_{aid}": [],
            f"surrogate_loss_{aid}": [],
            f"entropy_{aid}": [],
            f"entropy_bonus_{aid}": [],
            f"approx_kl_{aid}": [],
            f"clip_frac_{aid}": [],
            f"ratio_mean_{aid}": [],
            f"ratio_std_{aid}": [],
            f"ratio_min_{aid}": [],
            f"ratio_max_{aid}": [],
            f"grad_norm_actor_{aid}": [],
            f"actor_samples_seen_{aid}": 0,
        })

    agent_stopped = {aid: False for aid in agent_ids}
    agent_stop_epoch = {aid: ppo_epochs for aid in agent_ids}
    critic_state_t = prepared.critic_states
    critic_ret_t = prepared.critic_returns
    critic_aid_t = prepared.critic_agent_ids
    if value_normalizer is not None:
        value_normalizer.update(critic_ret_t)
        critic_target_t = value_normalizer.normalize(critic_ret_t)
        metrics.update({
            f"value_normalizer_{name}": value
            for name, value in value_normalizer.statistics().items()
        })
    else:
        critic_target_t = critic_ret_t

    for epoch in range(ppo_epochs):
        critic_indices = np.arange(critic_state_t.shape[0])
        if shuffle:
            np.random.shuffle(critic_indices)
        epoch_critic_losses = []
        for start in range(0, len(critic_indices), minibatch_size):
            mb = critic_indices[start:start + minibatch_size]
            critic_context = (
                f"{context} critic epoch={epoch} minibatch_start={start}"
            )
            value_predictions = critic(critic_state_t[mb], critic_aid_t[mb])
            critic_loss = compute_critic_value_loss(
                value_predictions,
                critic_target_t[mb],
                context=critic_context,
                huber_delta=huber_delta,
            )
            critic_optim.zero_grad()
            critic_loss.backward()
            ensure_finite_module_gradients(critic, context=critic_context)
            critic_grad_norm = nn.utils.clip_grad_norm_(
                critic.parameters(), max_grad_norm, error_if_nonfinite=True
            )
            ensure_finite_tensor(
                critic_grad_norm, name="critic_grad_norm", context=critic_context
            )
            critic_optim.step()
            ensure_finite_module_parameters(critic, context=critic_context)
            epoch_critic_losses.append(float(critic_loss.item()))
            metrics["grad_norm_critic"].append(float(critic_grad_norm))
            metrics["critic_samples_seen"] += int(len(mb))
        metrics["critic_loss"].append(float(np.mean(epoch_critic_losses)))

        for aid in agent_ids:
            if aid not in prepared.per_agent_tensors or agent_stopped[aid]:
                continue
            tensors = prepared.per_agent_tensors[aid]
            actor_indices = np.arange(tensors["obs"].shape[0])
            if shuffle:
                np.random.shuffle(actor_indices)
            epoch_actor_losses = []
            epoch_surrogate_losses = []
            epoch_entropies = []
            epoch_entropy_bonuses = []
            epoch_approx_kls = []
            epoch_clip_fracs = []
            epoch_ratio_means = []
            epoch_ratio_stds = []
            epoch_ratio_mins = []
            epoch_ratio_maxs = []
            epoch_actor_grad_norms = []
            for start in range(0, len(actor_indices), minibatch_size):
                mb = actor_indices[start:start + minibatch_size]
                actor_context = (
                    f"{context} actor={aid} epoch={epoch} "
                    f"minibatch_start={start}"
                )
                new_log_probs, entropy = actors[aid].evaluate_actions_z(
                    tensors["obs"][mb], tensors["act"][mb]
                )
                actor_loss, diagnostics = compute_clipped_actor_loss(
                    new_log_probs,
                    tensors["old_logp"][mb],
                    tensors["adv"][mb],
                    entropy,
                    clip_eps=clip_eps,
                    ent_coef=ent_coef,
                    context=actor_context,
                )
                actor_optims[aid].zero_grad()
                actor_loss.backward()
                ensure_finite_module_gradients(
                    actors[aid], context=actor_context
                )
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    actors[aid].parameters(),
                    max_grad_norm,
                    error_if_nonfinite=True,
                )
                ensure_finite_tensor(
                    actor_grad_norm,
                    name="actor_grad_norm",
                    context=actor_context,
                )
                actor_optims[aid].step()
                ensure_finite_module_parameters(
                    actors[aid], context=actor_context
                )
                epoch_actor_losses.append(float(actor_loss.item()))
                epoch_surrogate_losses.append(diagnostics["surrogate_loss"])
                epoch_entropies.append(diagnostics["entropy"])
                epoch_entropy_bonuses.append(diagnostics["entropy_bonus"])
                epoch_approx_kls.append(diagnostics["approx_kl"])
                epoch_clip_fracs.append(diagnostics["clip_fraction"])
                epoch_ratio_means.append(diagnostics["ratio_mean"])
                epoch_ratio_stds.append(diagnostics["ratio_std"])
                epoch_ratio_mins.append(diagnostics["ratio_min"])
                epoch_ratio_maxs.append(diagnostics["ratio_max"])
                epoch_actor_grad_norms.append(float(actor_grad_norm))
                metrics[f"actor_samples_seen_{aid}"] += int(len(mb))

            metrics[f"actor_loss_{aid}"].append(
                float(np.mean(epoch_actor_losses))
            )
            metrics[f"surrogate_loss_{aid}"].append(
                float(np.mean(epoch_surrogate_losses))
            )
            metrics[f"entropy_{aid}"].append(
                float(np.mean(epoch_entropies))
            )
            metrics[f"entropy_bonus_{aid}"].append(
                float(np.mean(epoch_entropy_bonuses))
            )
            mean_epoch_kl = float(np.mean(epoch_approx_kls))
            metrics[f"approx_kl_{aid}"].append(mean_epoch_kl)
            metrics[f"clip_frac_{aid}"].append(
                float(np.mean(epoch_clip_fracs))
            )
            metrics[f"ratio_mean_{aid}"].append(float(np.mean(epoch_ratio_means)))
            metrics[f"ratio_std_{aid}"].append(float(np.mean(epoch_ratio_stds)))
            metrics[f"ratio_min_{aid}"].append(float(np.min(epoch_ratio_mins)))
            metrics[f"ratio_max_{aid}"].append(float(np.max(epoch_ratio_maxs)))
            metrics[f"grad_norm_actor_{aid}"].append(
                float(np.mean(epoch_actor_grad_norms))
            )
            if should_stop_for_kl(mean_epoch_kl, target_kl):
                agent_stopped[aid] = True
                agent_stop_epoch[aid] = epoch + 1

    with torch.no_grad():
        normalized_value_predictions = critic(critic_state_t, critic_aid_t)
        value_predictions = (
            value_normalizer.denormalize(normalized_value_predictions)
            if value_normalizer is not None
            else normalized_value_predictions
        )
        ensure_finite_tensor(
            value_predictions,
            name="explained_variance_value_predictions",
            context=context,
        )
        residual_variance = torch.var(
            critic_ret_t - value_predictions, unbiased=False
        )
        return_variance = torch.var(critic_ret_t, unbiased=False)
        explained_variance = 1.0 - residual_variance / (return_variance + 1e-8)
        ensure_finite_tensor(
            explained_variance, name="explained_variance", context=context
        )
        metrics["explained_variance"] = float(explained_variance.item())
        value_error = critic_ret_t - value_predictions
        metrics["value_prediction_mean"] = float(value_predictions.mean().item())
        metrics["value_prediction_std"] = float(
            value_predictions.std(unbiased=False).item()
        )
        metrics["return_target_mean"] = float(critic_ret_t.mean().item())
        metrics["return_target_std"] = float(
            critic_ret_t.std(unbiased=False).item()
        )
        metrics["value_error_abs_mean"] = float(value_error.abs().mean().item())
        metrics["normalized_value_prediction_mean"] = float(
            normalized_value_predictions.mean().item()
        )
        metrics["normalized_value_prediction_std"] = float(
            normalized_value_predictions.std(unbiased=False).item()
        )
        metrics["normalized_return_target_mean"] = float(
            critic_target_t.mean().item()
        )
        metrics["normalized_return_target_std"] = float(
            critic_target_t.std(unbiased=False).item()
        )

    return MAPPOUpdateResult(
        metrics=metrics,
        agent_stop_epoch=agent_stop_epoch,
        preupdate_ratio_errors=preupdate_ratio_errors,
    )


# ============================================================
# 场景采样策略
# ============================================================

def compute_scenario_sampling_priorities(
    available_scenarios: List[str],
    success_rates: Optional[Dict[str, float]] = None,
    sample_counts: Optional[Dict[str, int]] = None,
    external_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """计算可解释且有界的自适应场景优先级。

    最近成功率使用 Beta(2, 2) 先验平滑，未采样场景按 50% 成功率处理，
    避免旧实现把“未知”误当成“100% 失败”。最终困难度因子为
    ``0.25 + posterior_failure``，再乘课程阶段权重。
    """
    priorities: Dict[str, float] = {}
    for scenario in available_scenarios:
        curriculum_weight = (
            1.0 if external_weights is None
            else float(external_weights.get(scenario, 0.0))
        )
        if not np.isfinite(curriculum_weight) or curriculum_weight < 0.0:
            raise ValueError(
                f"invalid external weight for {scenario}: {curriculum_weight}"
            )

        raw_rate = 0.5
        if success_rates is not None:
            raw_rate = float(success_rates.get(scenario, 0.5))
            if not np.isfinite(raw_rate):
                raise ValueError(
                    f"non-finite success rate for {scenario}: {raw_rate}"
                )
            raw_rate = float(np.clip(raw_rate, 0.0, 1.0))

        if sample_counts is None:
            posterior_success = raw_rate
        else:
            count = max(0, int(sample_counts.get(scenario, 0)))
            if count == 0:
                posterior_success = 0.5
            else:
                posterior_success = (raw_rate * count + 2.0) / (count + 4.0)

        posterior_failure = 1.0 - posterior_success
        priorities[scenario] = curriculum_weight * (
            0.25 + posterior_failure
        )
    return priorities


def sample_scenarios_stratified(
    available_scenarios: List[str],
    n_episodes: int,
    success_rates: Optional[Dict[str, float]] = None,
    sample_counts: Optional[Dict[str, int]] = None,
    coverage_counts: Optional[Dict[str, int]] = None,
    mode: str = "weighted",
    external_weights: Optional[Dict[str, float]] = None,
    coverage_groups: Optional[Dict[str, str]] = None,
) -> List[str]:
    """从可用场景中采样 n_episodes 个场景。

    Args:
        available_scenarios: 可用场景文件名列表
        n_episodes: 需要采样的 episode 数
        success_rates: 各场景的历史成功率（weighted 模式需要）
        sample_counts: 各成功率对应的样本数，用于 Beta 先验平滑
        coverage_counts: 各场景累计采样数，用于 family 内长期公平轮换
        mode: "random" | "stratified" | "weighted"
        external_weights: 外部权重（如课程软过渡权重），与 success_rates 相乘
        coverage_groups: 可选的场景覆盖组（通常取 JSON family）。提供后，
            weighted 模式优先覆盖成熟的场景族；族内优先选择历史采样最少
            的场景，再把剩余配额按失败率分配。

    Returns:
        长度为 n_episodes 的场景文件名列表
    """
    if n_episodes < 0:
        raise ValueError("n_episodes must be non-negative")
    if n_episodes == 0:
        return []
    if not available_scenarios:
        raise ValueError("available_scenarios must not be empty")
    if mode not in {"random", "stratified", "weighted"}:
        raise ValueError(f"unsupported scenario sampling mode: {mode!r}")

    if mode == "random":
        return [np.random.choice(available_scenarios) for _ in range(n_episodes)]

    priorities = compute_scenario_sampling_priorities(
        available_scenarios=available_scenarios,
        success_rates=success_rates,
        sample_counts=sample_counts,
        external_weights=external_weights,
    )
    if mode == "weighted":
        active_scenarios = [s for s in available_scenarios if priorities[s] > 0.0]
        if not active_scenarios:
            raise ValueError("weighted sampling has no positive-weight scenarios")
    else:
        active_scenarios = list(available_scenarios)

    n_scenarios = len(active_scenarios)
    selected = []

    if mode == "stratified":
        # 分层轮询：每个场景至少分配 1 个 episode
        rounds = n_episodes // n_scenarios
        remainder = n_episodes % n_scenarios
        for _ in range(rounds):
            selected.extend(active_scenarios)
        if remainder > 0:
            selected.extend(np.random.choice(active_scenarios, size=remainder, replace=False).tolist())

    elif mode == "weighted":
        # 课程权重达到 0.5 后才进入强制覆盖集；刚引入的新场景仍可按
        # 小权重被抽中，但不会因“每场景一次”让软过渡瞬间跳变。
        mandatory_scenarios = [
            scenario for scenario in active_scenarios
            if external_weights is None
            or external_weights.get(scenario, 0.0) >= 0.5
        ]
        if coverage_groups is not None:
            grouped: Dict[str, List[str]] = {}
            for scenario in mandatory_scenarios:
                group = str(coverage_groups.get(scenario, scenario))
                grouped.setdefault(group, []).append(scenario)

            group_names = sorted(grouped)
            if n_episodes < len(group_names):
                group_priorities = np.asarray(
                    [
                        sum(priorities[scenario] for scenario in grouped[group])
                        for group in group_names
                    ],
                    dtype=np.float64,
                )
                group_priorities /= group_priorities.sum()
                selected_groups = np.random.choice(
                    group_names,
                    size=n_episodes,
                    replace=False,
                    p=group_priorities,
                ).tolist()
            else:
                selected_groups = group_names

            for group in selected_groups:
                candidates = list(grouped[group])
                rotation_counts = (
                    coverage_counts
                    if coverage_counts is not None
                    else sample_counts
                )
                if rotation_counts is not None and candidates:
                    least_count = min(
                        int(rotation_counts.get(scenario, 0))
                        for scenario in candidates
                    )
                    candidates = [
                        scenario for scenario in candidates
                        if int(rotation_counts.get(scenario, 0)) == least_count
                    ]
                probs = np.asarray(
                    [priorities[scenario] for scenario in candidates],
                    dtype=np.float64,
                )
                probs /= probs.sum()
                selected.append(str(np.random.choice(candidates, p=probs)))
            remaining = n_episodes - len(selected)
        elif n_episodes >= len(mandatory_scenarios):
            selected.extend(mandatory_scenarios)
            remaining = n_episodes - len(mandatory_scenarios)
        else:
            # 小 batch 也按优先级无放回抽样，避免退化成与难度无关的均匀随机。
            probs = np.asarray(
                [priorities[s] for s in mandatory_scenarios], dtype=np.float64
            )
            probs /= probs.sum()
            selected.extend(
                np.random.choice(
                    mandatory_scenarios,
                    size=n_episodes,
                    replace=False,
                    p=probs,
                ).tolist()
            )
            remaining = 0

        # 剩余配额按平滑失败率加权。设置批内上限，防止单一困难/不可解
        # 场景吞掉整个更新，同时仍允许它高于均匀配额。
        if remaining > 0:
            counts = {s: selected.count(s) for s in active_scenarios}
            uniform_quota = int(np.ceil(n_episodes / n_scenarios))
            max_per_scenario = uniform_quota + max(
                1, int(np.ceil(n_episodes * 0.10))
            )
            for _ in range(remaining):
                eligible = [
                    s for s in active_scenarios
                    if counts[s] < max_per_scenario
                ]
                if not eligible:
                    eligible = list(active_scenarios)
                probs = np.asarray(
                    [priorities[s] for s in eligible], dtype=np.float64
                )
                probs /= probs.sum()
                chosen = str(np.random.choice(eligible, p=probs))
                selected.append(chosen)
                counts[chosen] += 1

    np.random.shuffle(selected)
    return selected


# ============================================================
# 课程学习（增强版：软过渡）
# ============================================================

class CurriculumLearning:
    """课程学习管理器（v2：支持软过渡）"""

    def __init__(
        self,
        scenario_dir: str,
        curriculum_stages: List[List[str]] = None,
        warmup_updates: int = 10,
        full_scenario_updates: int = 40,
        transition_window: int = 10,
        mode: str = "classic",
    ):
        self.scenario_dir = scenario_dir
        self.warmup_updates = warmup_updates
        self.full_scenario_updates = full_scenario_updates
        self.transition_window = transition_window
        self.mode = mode

        all_scenario_files = []
        if os.path.exists(scenario_dir):
            all_scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])

        self.scenario_families: Dict[str, str] = {}
        for filename in all_scenario_files:
            path = os.path.join(scenario_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                family = str(config.get("family", filename)).strip()
            except (OSError, ValueError, TypeError):
                family = filename
            self.scenario_families[filename] = family or filename

        if curriculum_stages is None and mode == "metadata":
            self.curriculum_stages = self._build_metadata_stages(
                scenario_dir, all_scenario_files
            )
        elif curriculum_stages is None:
            # mode="classic" 强制走 basic/left|right/all 旧逻辑（回退距离课程，
            # 即使目录里存在 dist_<N>m 场景文件也忽略其距离分层）
            self.curriculum_stages = self._build_default_stages(
                all_scenario_files, use_distance=(mode == "distance")
            )
        else:
            self.curriculum_stages = curriculum_stages

        self.scenario_success_history: Dict[str, List[float]] = {f: [] for f in all_scenario_files}
        self.scenario_total_sample_counts: Dict[str, int] = {
            filename: 0 for filename in all_scenario_files
        }
        print(f"课程学习初始化(mode={mode})，共 {len(self.curriculum_stages)} 个阶段，{len(all_scenario_files)} 个场景")
        print(f"  阶段构成: {[len(s) for s in self.curriculum_stages]}（近→远 / 易→难）")
        print(f"  过渡窗口: {transition_window} updates")

    @staticmethod
    def _build_metadata_stages(
        scenario_dir: str, all_scenario_files: List[str]
    ) -> List[List[str]]:
        """按 JSON 的 difficulty 字段构建累积课程阶段。"""
        if not all_scenario_files:
            raise ValueError(f"场景目录为空: {scenario_dir}")

        difficulties: Dict[str, int] = {}
        for filename in all_scenario_files:
            path = os.path.join(scenario_dir, filename)
            with open(path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            if "difficulty" not in config:
                raise ValueError(
                    f"metadata curriculum requires difficulty: {path}"
                )
            difficulty = int(config["difficulty"])
            if difficulty < 1:
                raise ValueError(
                    f"scenario difficulty must be >= 1: {path}"
                )
            difficulties[filename] = difficulty

        levels = sorted(set(difficulties.values()))
        expected_levels = list(range(1, max(levels) + 1))
        if levels != expected_levels:
            raise ValueError(
                "scenario difficulty levels must be consecutive from 1: "
                f"found {levels}"
            )
        return [
            [
                filename for filename in all_scenario_files
                if difficulties[filename] <= level
            ]
            for level in levels
        ]

    @staticmethod
    def _build_default_stages(all_scenario_files: List[str], use_distance: bool = False) -> List[List[str]]:
        """构建默认课程阶段（近→远 / 易→难）。

        use_distance=True 且存在 `dist_<N>m` 命名的距离课程场景时，按距离升序
        分 3 阶段（让 agent 先在近 spawn 学会冲终点，再逐步拉远）。

        use_distance=False（默认/classic）：忽略距离分层，回退到原
        basic / left|right / all 三段逻辑。距离课程被验证会在短距离阶段
        诱发"原地不动"局部最优，故默认关闭。
        """
        def _dist_of(fname):
            m = re.search(r'dist_(\d+)m', fname)
            return int(m.group(1)) if m else None

        dist_files = [f for f in all_scenario_files if _dist_of(f) is not None]
        if use_distance and dist_files:
            dist_sorted = sorted(dist_files, key=_dist_of)
            other = [f for f in all_scenario_files if _dist_of(f) is None]
            n = len(dist_sorted)
            stage1 = dist_sorted[:max(1, n // 3)]                 # 最近 ~1/3
            stage2 = dist_sorted[:max(1, (2 * n) // 3)]           # 最近 ~2/3
            stage3 = dist_sorted + other                          # 全部（含原有远距/特殊场景）
            return [stage1, stage2, stage3]

        # 回退：原 basic / left|right / all 逻辑
        # classic 模式下排除 dist_<N>m 场景，使行为完全等同"加距离课程之前"的版本
        pool = [f for f in all_scenario_files if _dist_of(f) is None]
        if not pool:
            pool = list(all_scenario_files)
        stages = []
        basic_scenarios = [f for f in pool if 'basic' in f]
        if not basic_scenarios and pool:
            basic_scenarios = [pool[0]]
        stages.append(basic_scenarios)
        # stage2（中等难度）：基础 + 横向偏移(left/right) + 并排交互(parallel)
        stage2 = basic_scenarios + [
            f for f in pool
            if 'left' in f or 'right' in f or 'parallel' in f
        ]
        # Preserve the sorted scenario-pool order while removing duplicates.
        # list(set(...)) is process-hash dependent and made strict resume fail
        # when a fresh Python process rebuilt the same logical curriculum.
        stages.append(list(dict.fromkeys(stage2)))
        stages.append(pool)
        return stages

    def _get_stage_idx(self, current_update: int) -> int:
        """获取当前所处的阶段索引"""
        n_stages = len(self.curriculum_stages)
        if n_stages <= 1 or current_update < self.warmup_updates:
            return 0
        elif current_update >= self.full_scenario_updates:
            return n_stages - 1
        elif n_stages == 2:
            return 0
        else:
            # stage 1 在 warmup 边界立即开始；中间阶段均匀分布，最后阶段
            # 仍严格由 full_scenario_updates 开启。旧实现额外乘 n_stages，
            # 使默认 10-update warmup 实际延长到 update 20。
            middle_stage_count = n_stages - 2
            span = max(1, self.full_scenario_updates - self.warmup_updates)
            progress = (current_update - self.warmup_updates) / span
            return min(
                1 + int(progress * middle_stage_count), n_stages - 2
            )

    def _stage_start_update(self, stage_idx: int) -> float:
        """返回某阶段首次成为当前阶段的 update 边界。"""
        n_stages = len(self.curriculum_stages)
        if stage_idx <= 0:
            return 0.0
        if stage_idx >= n_stages - 1:
            return float(self.full_scenario_updates)
        middle_stage_count = n_stages - 2
        return float(self.warmup_updates) + (
            (stage_idx - 1)
            * (self.full_scenario_updates - self.warmup_updates)
            / max(1, middle_stage_count)
        )

    def _compute_transition_alpha(self, current_update: int) -> float:
        """计算软过渡系数 α ∈ [0, 1]。

        α=0 表示完全使用旧阶段场景，α=1 表示完全使用当前阶段场景。
        在阶段切换点附近线性过渡。
        """
        stage_idx = self._get_stage_idx(current_update)
        if stage_idx == 0:
            return 1.0  # 第一阶段，无旧场景

        n_stages = len(self.curriculum_stages)
        if n_stages <= 1:
            return 1.0
        if self.transition_window <= 0:
            return 1.0
        stage_start = self._stage_start_update(stage_idx)
        elapsed_updates = current_update - stage_start + 1.0
        return float(np.clip(
            elapsed_updates / self.transition_window, 0.0, 1.0
        ))

    def get_available_scenarios(self, current_update: int) -> List[str]:
        """获取当前阶段的场景列表（硬切换，向后兼容）"""
        stage_idx = self._get_stage_idx(current_update)
        return self.curriculum_stages[min(stage_idx, len(self.curriculum_stages) - 1)]

    def get_scenario_coverage_groups(
        self, scenarios: List[str]
    ) -> Dict[str, str]:
        """返回批内覆盖组；metadata 课程按 family，旧课程逐场景覆盖。"""
        if self.mode != "metadata":
            return {scenario: scenario for scenario in scenarios}
        return {
            scenario: self.scenario_families.get(scenario, scenario)
            for scenario in scenarios
        }

    def get_scenario_sampling_weights(self, current_update: int) -> Dict[str, float]:
        """获取场景采样权重（软过渡版本）。

        返回 {scenario_name: weight} 字典。
        在阶段过渡期，旧阶段场景以递减权重参与采样。
        """
        alpha = self._compute_transition_alpha(current_update)
        stage_idx = self._get_stage_idx(current_update)

        current_scenarios = self.curriculum_stages[
            min(stage_idx, len(self.curriculum_stages) - 1)
        ]
        if stage_idx == 0:
            return {scenario: 1.0 for scenario in current_scenarios}

        previous_scenarios = set(self.curriculum_stages[stage_idx - 1])
        # 阶段是累积集合：旧场景保持 1.0，仅本阶段新增场景从 alpha
        # 逐步升到 1.0。旧实现给整个当前累积集合 1.0，软过渡事实上无效。
        weights = {
            scenario: (1.0 if scenario in previous_scenarios else alpha)
            for scenario in current_scenarios
        }

        return weights

    def record_scenario_result(self, scenario_name: str, success: bool):
        success_float = 1.0 if success else 0.0
        if scenario_name not in self.scenario_success_history:
            self.scenario_success_history[scenario_name] = []
        self.scenario_total_sample_counts[scenario_name] = (
            int(self.scenario_total_sample_counts.get(scenario_name, 0)) + 1
        )
        self.scenario_success_history[scenario_name].append(success_float)

        if len(self.scenario_success_history[scenario_name]) > 20:
            self.scenario_success_history[scenario_name] = (
                self.scenario_success_history[scenario_name][-20:]
            )

    def get_scenario_success_rate(self, scenario_name: str) -> float:
        if scenario_name not in self.scenario_success_history or len(self.scenario_success_history[scenario_name]) == 0:
            return 0.5
        return float(np.mean(self.scenario_success_history[scenario_name]))

    def get_scenario_sample_count(self, scenario_name: str) -> int:
        return len(self.scenario_success_history.get(scenario_name, []))

    def get_scenario_total_sample_count(self, scenario_name: str) -> int:
        return int(self.scenario_total_sample_counts.get(scenario_name, 0))


# ============================================================
# 训练主函数
# ============================================================

def validate_p0_smoke_interrupt_configuration(args, *, check_rollout_bound=True):
    """Validate the deterministic P0 smoke interrupt without changing training data."""
    interrupt_after = int(args.p0_smoke_interrupt_after_episodes)
    if interrupt_after < 0:
        raise ValueError("--p0-smoke-interrupt-after-episodes must be non-negative")
    if interrupt_after > 0 and not args.resume:
        raise ValueError(
            "--p0-smoke-interrupt-after-episodes is only valid with --resume"
        )
    if check_rollout_bound and interrupt_after > int(args.rollout_episodes):
        raise ValueError(
            "--p0-smoke-interrupt-after-episodes cannot exceed "
            "--rollout-episodes"
        )
    return interrupt_after


def train(args):
    project_dir = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[4]
    if getattr(args, "experiment_contract", "") and not getattr(
        args, "_experiment_contract_metadata", None
    ):
        apply_experiment_contract(args, project_dir=project_dir)

    if args.resume and args.pretrained_checkpoint:
        raise ValueError(
            "--resume and --pretrained-checkpoint are mutually exclusive"
        )
    if args.updates <= 0:
        raise ValueError("--updates must be positive")
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be positive")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if args.tb_histogram_interval < 0:
        raise ValueError("--tb-histogram-interval must be non-negative")
    if args.lr_decay_horizon < 0:
        raise ValueError("--lr-decay-horizon must be non-negative")
    if not 0.0 <= args.value_norm_beta < 1.0:
        raise ValueError("--value-norm-beta must be in [0, 1)")
    if args.value_norm_epsilon <= 0.0:
        raise ValueError("--value-norm-epsilon must be positive")
    if args.value_norm_variance_floor <= 0.0:
        raise ValueError("--value-norm-variance-floor must be positive")
    if args.huber_delta <= 0.0:
        raise ValueError("--huber-delta must be positive")
    validate_p0_smoke_interrupt_configuration(
        args,
        check_rollout_bound=not bool(args.resume),
    )

    save_dir = os.path.abspath(args.save_dir)
    current_git = get_git_metadata(repository_root)
    fingerprint_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("action_protocol.py"),
        Path(__file__).resolve().parents[1]
        / "environment"
        / "base_reward_v3.py",
        Path(__file__).resolve().with_name("experiment_contract.py"),
        Path(__file__).resolve().with_name("checkpoint_protocol.py"),
        Path(__file__).resolve().with_name("lifecycle_semantics.py"),
        Path(__file__).resolve().with_name("metrics_protocol.py"),
        Path(__file__).resolve().with_name("value_normalization.py"),
        project_dir / "utils" / "training_metrics_recorder.py",
        project_dir / "models" / "actor_critic.py",
        project_dir / "environment" / "carla_multi_agent_env.py",
    ]
    experiment_contract_metadata = getattr(
        args, "_experiment_contract_metadata", None
    )
    if experiment_contract_metadata:
        fingerprint_paths.append(Path(experiment_contract_metadata["path"]))
        verify_critical_sources_clean(
            repository_root=repository_root,
            project_dir=project_dir,
            relative_paths=experiment_contract_metadata["payload"][
                "critical_source_paths"
            ],
        )
    code_fingerprint = files_fingerprint(fingerprint_paths)
    resume_checkpoint = None
    resume_digest = ""
    resume_path = None
    pretrained_checkpoint = None
    pretrained_digest = ""
    pretrained_path = None
    manifest = None

    if args.resume:
        resume_path = resolve_resume_path(args.resume, save_dir)
        if os.path.normcase(str(infer_run_root(resume_path))) != os.path.normcase(
            str(Path(save_dir).resolve())
        ):
            raise CheckpointProtocolError(
                "resume checkpoint must belong to --save-dir; use "
                "--pretrained-checkpoint to start a new run"
            )
        resume_checkpoint, resume_digest = load_checkpoint_file(
            resume_path, map_location="cpu"
        )
        validate_checkpoint_header(resume_checkpoint)
        if experiment_contract_metadata:
            saved_contract_sha = str(
                resume_checkpoint["run_metadata"].get(
                    "experiment_contract_sha256", ""
                )
            )
            if saved_contract_sha != experiment_contract_metadata["sha256"]:
                raise ExperimentContractError(
                    "resume checkpoint experiment contract SHA256 mismatch: "
                    f"{saved_contract_sha!r} != "
                    f"{experiment_contract_metadata['sha256']!r}"
                )
        apply_resume_configuration(args, resume_checkpoint)
        validate_p0_smoke_interrupt_configuration(args)

        manifest = load_manifest(Path(save_dir) / MANIFEST_NAME)
        if manifest.get("run_id") != args.run_id:
            raise CheckpointProtocolError(
                f"manifest run_id={manifest.get('run_id')!r} does not match "
                f"checkpoint run_id={args.run_id!r}"
            )
        latest_record = manifest.get("latest") or {}
        if int(latest_record.get("update", -1)) != int(
            resume_checkpoint["update"]
        ):
            raise CheckpointProtocolError(
                "resume checkpoint is not the manifest's latest safe update; "
                "rollback/fork resume is not implicit"
            )
        known_records = [
            manifest.get("latest"),
            manifest.get("interrupted"),
            *list(manifest.get("periodic", [])),
        ]
        known_digests = {
            record.get("sha256")
            for record in known_records
            if isinstance(record, dict)
        }
        if resume_digest not in known_digests:
            raise CheckpointProtocolError(
                "resume checkpoint SHA256 is not registered in the run manifest"
            )

        saved_git_sha = str(resume_checkpoint["run_metadata"]["last_git_sha"])
        current_git_sha = str(current_git["commit"])
        saved_code_fingerprint = str(
            resume_checkpoint["run_metadata"]["code_fingerprint"]
        )
        resume_code_drift = (
            saved_git_sha != current_git_sha
            or saved_code_fingerprint != code_fingerprint
        )
        if resume_code_drift and not args.allow_resume_code_drift:
            raise CheckpointProtocolError(
                "resume code identity mismatch: "
                f"git={saved_git_sha}/{current_git_sha}, "
                "source_fingerprint="
                f"{saved_code_fingerprint}/{code_fingerprint}; "
                "commit/use matching code or pass --allow-resume-code-drift "
                "for an explicitly marked non-bitwise continuation"
            )
    else:
        ensure_fresh_run_directory(save_dir)
        if not args.run_id:
            args.run_id = Path(save_dir).resolve().name
        if not args.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if args.lr_decay_horizon == 0:
            args.lr_decay_horizon = args.updates

        if args.config_path:
            config_path = Path(args.config_path).expanduser().resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"config path not found: {config_path}")
            args.config_path = str(config_path)

        if args.pretrained_checkpoint:
            pretrained_path = Path(
                args.pretrained_checkpoint
            ).expanduser().resolve()
            if not pretrained_path.is_file():
                raise FileNotFoundError(
                    f"pretrained checkpoint not found: {pretrained_path}"
                )
            pretrained_digest = sha256_file(pretrained_path)
            if Path(str(pretrained_path) + ".sha256").is_file():
                pretrained_checkpoint, verified_digest = load_checkpoint_file(
                    pretrained_path, map_location="cpu"
                )
                if verified_digest != pretrained_digest:
                    raise CheckpointProtocolError(
                        "pretrained checkpoint digest changed during load"
                    )
            else:
                pretrained_checkpoint = torch.load(
                    pretrained_path, map_location="cpu", weights_only=False
                )
                if not isinstance(pretrained_checkpoint, dict):
                    raise CheckpointProtocolError(
                        "pretrained checkpoint root must be a dictionary"
                    )

    set_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    scenario_dir = (
        os.path.abspath(args.scenario_dir)
        if args.scenario_dir
        else os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "training_scenarios_town05_v1",
        )
    )
    curriculum = CurriculumLearning(
        scenario_dir=scenario_dir,
        warmup_updates=args.warmup_updates,
        full_scenario_updates=args.full_scenario_updates,
        transition_window=args.curriculum_transition_window,
        mode=args.curriculum_mode,
    )

    # 当前冻结基础入口始终是 2 个 19 维红车观察的集中状态。
    # 先按协议构建/校验完整训练状态，确认可恢复后才连接 CARLA。
    agent_ids = [0, 1]
    state_dim = OBS_DIM * len(agent_ids)

    actors: Dict[int, Actor] = {
        aid: Actor(
            obs_dim=OBS_DIM,
            act_dim=ACTION_DIM,
            hidden=args.hidden_dim,
            init_log_std=args.init_log_std,
        ).to(device)
        for aid in agent_ids
    }
    actor_optims: Dict[int, optim.Optimizer] = {
        aid: optim.Adam(actors[aid].parameters(), lr=args.lr_actor)
        for aid in agent_ids
    }
    critic = AgentConditionedCritic(
        state_dim=state_dim,
        num_agents=len(agent_ids),
        hidden=args.hidden_dim,
    ).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)
    value_normalizer = ValueNormalizer(
        beta=args.value_norm_beta,
        epsilon=args.value_norm_epsilon,
        variance_floor=args.value_norm_variance_floor,
        device=device,
    )
    critic_spec = value_normalizer.spec()
    critic_spec["huber_delta"] = float(args.huber_delta)

    init_lr_actor = args.lr_actor
    init_lr_critic = args.lr_critic
    lr_decay_horizon = int(args.lr_decay_horizon)

    def _apply_lr_decay(cur_update: int):
        if not args.lr_decay:
            return init_lr_actor, init_lr_critic
        frac = max(
            0.05,
            1.0 - (cur_update - 1) / max(lr_decay_horizon - 1, 1),
        )
        new_lr_a = init_lr_actor * frac
        new_lr_c = init_lr_critic * frac
        for aid in agent_ids:
            for group in actor_optims[aid].param_groups:
                group["lr"] = new_lr_a
        for group in critic_optim.param_groups:
            group["lr"] = new_lr_c
        return new_lr_a, new_lr_c

    pretrained_load_report = None
    if pretrained_checkpoint is not None:
        pretrained_load_report = _load_pretrained_weights(
            pretrained_checkpoint, actors, critic, value_normalizer
        )
        print(f"加载预训练权重: {pretrained_path}")
        if not pretrained_load_report["critic_loaded"]:
            print(
                "  [预训练] Critic 未加载，将使用新初始化的 "
                f"AgentConditionedCritic: {pretrained_load_report['critic_note']}"
            )

    total_episodes = 0
    total_successes = 0
    total_collisions = 0
    total_timeouts = 0
    completed_update = 0
    start_update = 1

    protocol_versions = checkpoint_protocol_versions(args.reward_version)
    current_action_spec = action_protocol_spec()
    model_spec = checkpoint_model_spec(
        agent_ids, state_dim, args.hidden_dim
    )
    training_config = checkpoint_training_config(args)
    pool_fingerprint = scenario_pool_fingerprint(scenario_dir)
    if experiment_contract_metadata:
        if resume_checkpoint is not None:
            validate_resume_experiment_evidence(
                args=args, run_dir=Path(save_dir)
            )
        else:
            initialize_experiment_evidence(
                args=args,
                run_dir=Path(save_dir),
                git_metadata=current_git,
                code_fingerprint=code_fingerprint,
                scenario_pool_fingerprint=pool_fingerprint,
                training_config=training_config,
            )

    def _create_validated_env():
        created_env = CarlaMultiAgentEnv(
            **build_training_environment_kwargs(
                args, num_agents=len(agent_ids)
            )
        )
        try:
            validate_environment_reward_discount(created_env, args)
            actual_state_dim = int(created_env.get_global_state().shape[0])
            if (
                created_env.num_agents != len(agent_ids)
                or actual_state_dim != state_dim
            ):
                raise CheckpointProtocolError(
                    "runtime environment shape differs from frozen protocol: "
                    f"agents={created_env.num_agents}/{len(agent_ids)}, "
                    f"state_dim={actual_state_dim}/{state_dim}"
                )
        except Exception:
            created_env.close()
            raise
        print(f"全局状态维度: {actual_state_dim}")
        return created_env

    if resume_checkpoint is not None:
        validate_checkpoint_payload(
            resume_checkpoint,
            expected_protocol_versions=protocol_versions,
            expected_action_spec=current_action_spec,
            expected_model_spec=model_spec,
            expected_training_config=training_config,
            expected_critic_spec=critic_spec,
        )
        saved_curriculum = resume_checkpoint["curriculum_state"]
        if saved_curriculum.get("scenario_pool_fingerprint") != pool_fingerprint:
            raise CheckpointProtocolError(
                "scenario pool fingerprint differs from checkpoint"
            )
        if saved_curriculum.get("stages") != curriculum.curriculum_stages:
            raise CheckpointProtocolError(
                "curriculum stages differ from checkpoint"
            )

        try:
            for aid in agent_ids:
                actors[aid].load_state_dict(
                    resume_checkpoint["actors"][aid], strict=True
                )
                actor_optims[aid].load_state_dict(
                    resume_checkpoint["actor_optims"][aid]
                )
            critic.load_state_dict(
                resume_checkpoint["critic"], strict=True
            )
            critic_optim.load_state_dict(
                resume_checkpoint["critic_optim"]
            )
            value_normalizer.load_state_dict(
                resume_checkpoint["value_normalizer"], strict=True
            )
            value_normalizer.validate_state()
        except (KeyError, RuntimeError, ValueError) as error:
            raise CheckpointProtocolError(
                f"failed to restore complete model/optimizer state: {error}"
            ) from error

        completed_update = int(resume_checkpoint["update"])
        start_update = completed_update + 1
        total_episodes = int(resume_checkpoint["total_episodes"])
        total_successes = int(resume_checkpoint["total_successes"])
        total_collisions = int(resume_checkpoint["total_collisions"])
        total_timeouts = int(resume_checkpoint["total_timeouts"])
        curriculum.scenario_success_history = {
            str(name): list(values)
            for name, values in saved_curriculum["success_history"].items()
        }
        saved_total_counts = saved_curriculum.get("total_sample_counts", {})
        curriculum.scenario_total_sample_counts = {
            scenario: int(saved_total_counts.get(
                scenario,
                len(curriculum.scenario_success_history.get(scenario, [])),
            ))
            for scenario in curriculum.scenario_total_sample_counts
        }
        if start_update > args.updates:
            raise CheckpointProtocolError(
                f"checkpoint completed update={completed_update}, but "
                f"--updates={args.updates}; choose a larger stop update"
            )

        try:
            env = _create_validated_env()
        except Exception as error:
            update_experiment_evidence(
                args=args,
                run_dir=Path(save_dir),
                completed_update=completed_update,
                status="ERROR_PRE_TRAIN",
                notes=f"environment initialization failed: {error}",
            )
            raise
        run_metadata = copy.deepcopy(
            resume_checkpoint["run_metadata"]
        )
        code_drift_history = list(
            run_metadata.get("code_drift_history", [])
        )
        if resume_code_drift:
            code_drift_history.append(
                {
                    "resumed_at": utc_now_iso(),
                    "from_git_sha": saved_git_sha,
                    "to_git_sha": current_git_sha,
                    "from_code_fingerprint": saved_code_fingerprint,
                    "to_code_fingerprint": code_fingerprint,
                }
            )
        run_metadata.update(
            {
                "last_git_sha": current_git["commit"],
                "code_fingerprint": code_fingerprint,
                "last_git_dirty": current_git["dirty"],
                "last_git_dirty_paths": current_git["dirty_paths"],
                "last_resume_at": utc_now_iso(),
                "last_resume_checkpoint": str(resume_path),
                "last_resume_sha256": resume_digest,
                "resume_count": int(run_metadata.get("resume_count", 0)) + 1,
                "last_command": normalize_command_line(),
                "last_runtime": runtime_metadata_with_carla(env),
                "code_drift_override": bool(
                    run_metadata.get("code_drift_override", False)
                    or resume_code_drift
                ),
                "code_drift_history": code_drift_history,
                "experiment_contract_id": (
                    experiment_contract_metadata["contract_id"]
                    if experiment_contract_metadata
                    else run_metadata.get("experiment_contract_id", "")
                ),
                "experiment_contract_sha256": (
                    experiment_contract_metadata["sha256"]
                    if experiment_contract_metadata
                    else run_metadata.get("experiment_contract_sha256", "")
                ),
            }
        )
    else:
        try:
            env = _create_validated_env()
        except Exception as error:
            update_experiment_evidence(
                args=args,
                run_dir=Path(save_dir),
                completed_update=completed_update,
                status="ERROR_PRE_TRAIN",
                notes=f"environment initialization failed: {error}",
            )
            raise
        source_type = (
            "pretrained_weights"
            if pretrained_checkpoint is not None
            else "fresh"
        )
        run_metadata = {
            "run_id": args.run_id,
            "seed": int(args.seed),
            "created_at": utc_now_iso(),
            "origin_git_sha": current_git["commit"],
            "last_git_sha": current_git["commit"],
            "origin_code_fingerprint": code_fingerprint,
            "code_fingerprint": code_fingerprint,
            "git_dirty": current_git["dirty"],
            "git_dirty_paths": current_git["dirty_paths"],
            "config_path": args.config_path,
            "config_sha256": (
                sha256_file(args.config_path) if args.config_path else ""
            ),
            "command": normalize_command_line(),
            "runtime": runtime_metadata_with_carla(env),
            "source_type": source_type,
            "source_checkpoint": (
                str(pretrained_path) if pretrained_path else ""
            ),
            "source_sha256": pretrained_digest,
            "pretrained_load_report": pretrained_load_report,
            "resume_count": 0,
            "code_drift_override": False,
            "code_drift_history": [],
            "experiment_contract_id": (
                experiment_contract_metadata["contract_id"]
                if experiment_contract_metadata
                else ""
            ),
            "experiment_contract_sha256": (
                experiment_contract_metadata["sha256"]
                if experiment_contract_metadata
                else ""
            ),
            "experiment_contract_canonical_sha256": (
                experiment_contract_metadata["canonical_sha256"]
                if experiment_contract_metadata
                else ""
            ),
            "planned_total_updates": (
                experiment_contract_metadata["planned_total_updates"]
                if experiment_contract_metadata
                else int(args.updates)
            ),
        }
        manifest = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "training_protocol_version": TRAINING_PROTOCOL_VERSION,
            "run_id": args.run_id,
            "created_at": utc_now_iso(),
            "latest": None,
            "interrupted": None,
            "periodic": [],
        }

    visualizer = (
        MAPPOVisualizer(width=1200, height=800, fps=30)
        if args.render
        else None
    )
    last_metrics_payload = (
        copy.deepcopy(resume_checkpoint.get("last_metrics"))
        if resume_checkpoint is not None
        else None
    )
    try:
        metrics_recorder = TrainingMetricsRecorder(
            run_dir=save_dir,
            run_id=args.run_id,
            agent_ids=agent_ids,
            reward_version=args.reward_version,
            start_update=start_update,
            completed_update=completed_update,
            resume_metrics_payload=last_metrics_payload,
            histogram_interval=args.tb_histogram_interval,
            critic_protocol=critic_spec,
        )
        update_experiment_evidence(
            args=args,
            run_dir=Path(save_dir),
            completed_update=completed_update,
            status="RUNNING",
        )
    except Exception as error:
        try:
            update_experiment_evidence(
                args=args,
                run_dir=Path(save_dir),
                completed_update=completed_update,
                status="ERROR_PRE_TRAIN",
                notes=f"metrics recorder initialization failed: {error}",
            )
        except Exception as evidence_error:
            print(f"  [警告] 实验台账错误状态写入失败: {evidence_error}")
        if visualizer is not None:
            visualizer.close()
        if "metrics_recorder" in locals():
            metrics_recorder.close()
        env.close()
        raise
    print(f"RunID: {args.run_id}")
    print(f"TensorBoard 日志: {metrics_recorder.tb_dir}")
    print(f"结构化指标: {metrics_recorder.jsonl_path}")

    if resume_checkpoint is not None:
        rng_warnings = restore_rng_state(
            resume_checkpoint["rng_state"]
        )
        for warning in rng_warnings:
            print(f"  [恢复警告] {warning}")
        print(
            f"恢复训练: {resume_path} -> next_update={start_update}, "
            f"total_ep={total_episodes}, "
            f"cum_success={total_successes/max(total_episodes,1):.2%}, "
            f"lr_horizon={lr_decay_horizon}"
        )

    periodic_dir = Path(save_dir) / PERIODIC_DIR_NAME
    manifest_path = Path(save_dir) / MANIFEST_NAME
    latest_path = Path(save_dir) / LATEST_CHECKPOINT_NAME
    interrupted_path = Path(save_dir) / INTERRUPTED_CHECKPOINT_NAME

    def _serializable_args() -> Dict[str, object]:
        return {
            key: value
            for key, value in vars(args).items()
            if not key.startswith("_")
        }

    def _build_checkpoint_dict(kind: str) -> Dict[str, object]:
        current_actor_lrs = {
            aid: float(actor_optims[aid].param_groups[0]["lr"])
            for aid in agent_ids
        }
        current_critic_lr = float(
            critic_optim.param_groups[0]["lr"]
        )
        checkpoint_run_metadata = copy.deepcopy(run_metadata)
        checkpoint_run_metadata["last_saved_at"] = utc_now_iso()
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "training_protocol_version": TRAINING_PROTOCOL_VERSION,
            "protocol_versions": protocol_versions,
            "action_spec": current_action_spec,
            "checkpoint_kind": kind,
            "update": int(completed_update),
            "actors": {
                aid: actors[aid].state_dict() for aid in agent_ids
            },
            "critic": critic.state_dict(),
            "value_normalizer": value_normalizer.state_dict(),
            "critic_spec": critic_spec,
            "actor_optims": {
                aid: actor_optims[aid].state_dict()
                for aid in agent_ids
            },
            "critic_optim": critic_optim.state_dict(),
            "args": _serializable_args(),
            "model_spec": model_spec,
            "training_config": training_config,
            "lr_schedule": {
                "enabled": bool(args.lr_decay),
                "decay_horizon": lr_decay_horizon,
                "initial_actor_lr": init_lr_actor,
                "initial_critic_lr": init_lr_critic,
                "current_actor_lrs": current_actor_lrs,
                "current_critic_lr": current_critic_lr,
            },
            "run_metadata": checkpoint_run_metadata,
            "curriculum_state": {
                "config": {
                    "warmup_updates": args.warmup_updates,
                    "full_scenario_updates": args.full_scenario_updates,
                    "transition_window": (
                        args.curriculum_transition_window
                    ),
                    "mode": args.curriculum_mode,
                },
                "stages": curriculum.curriculum_stages,
                "scenario_pool_fingerprint": pool_fingerprint,
                "success_history": {
                    name: list(values)
                    for name, values in (
                        curriculum.scenario_success_history.items()
                    )
                },
                "total_sample_counts": {
                    name: int(value)
                    for name, value in (
                        curriculum.scenario_total_sample_counts.items()
                    )
                },
            },
            "rng_state": capture_rng_state(),
            "last_metrics": copy.deepcopy(last_metrics_payload),
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "total_collisions": int(total_collisions),
            "total_timeouts": int(total_timeouts),
            "curriculum_success_history": {
                name: list(values)
                for name, values in (
                    curriculum.scenario_success_history.items()
                )
            },
        }

    def _manifest_record(
        checkpoint_path: Path, digest: str
    ) -> Dict[str, object]:
        return {
            "path": str(checkpoint_path.relative_to(save_dir)),
            "update": int(completed_update),
            "sha256": digest,
            "recorded_at": utc_now_iso(),
        }

    def _save_safe_checkpoint(*, include_periodic: bool) -> None:
        if include_periodic:
            periodic_path = periodic_dir / (
                f"ckpt_update_{completed_update:06d}.pt"
            )
            periodic_digest = save_checkpoint_file(
                _build_checkpoint_dict("periodic"),
                periodic_path,
                immutable=True,
            )
            manifest["periodic"].append(
                _manifest_record(periodic_path, periodic_digest)
            )
            print(f"周期 checkpoint: {periodic_path}")

        latest_digest = save_checkpoint_file(
            _build_checkpoint_dict("latest"), latest_path
        )
        manifest["latest"] = _manifest_record(
            latest_path, latest_digest
        )
        atomic_write_json(manifest, manifest_path)
        print(f"最新 checkpoint: {latest_path}")

    def _save_interrupted_checkpoint() -> None:
        if not latest_path.is_file():
            return
        interrupted_digest = atomic_copy_checkpoint(
            latest_path, interrupted_path
        )
        manifest["interrupted"] = _manifest_record(
            interrupted_path, interrupted_digest
        )
        atomic_write_json(manifest, manifest_path)
        print(
            "安全中断 checkpoint（最近完整 update）: "
            f"{interrupted_path}"
        )

    experiment_status = "RUNNING"
    experiment_notes = ""
    try:
        if resume_checkpoint is None:
            _save_safe_checkpoint(include_periodic=False)
        for update in range(start_update, args.updates + 1):
            update_started_at = time.perf_counter()
            # 学习率线性衰减
            cur_lr_a, cur_lr_c = _apply_lr_decay(update)

            buffer = RolloutBuffer(agent_ids)

            # 当前 update 窗口统计
            window_episode_returns: List[float] = []
            window_episode_lengths: List[int] = []
            window_successes: List[float] = []          # 严格成功（all reached_target）
            window_collisions: List[float] = []         # 任一 agent 发生碰撞
            window_timeouts: List[float] = []           # max_steps 截断且未成功未碰撞

            # ── 场景采样（分层/加权） ──
            available_scenarios = curriculum.get_available_scenarios(update)

            # 获取成功率（weighted 模式需要）
            success_rates = None
            sample_counts = None
            coverage_counts = None
            soft_weights = None
            if args.scenario_sampling_mode == "weighted":
                soft_weights = curriculum.get_scenario_sampling_weights(update)
                weighted_scenarios = [
                    scenario for scenario, weight in soft_weights.items()
                    if weight > 0.0
                ]
                success_rates = {
                    s: curriculum.get_scenario_success_rate(s)
                    for s in weighted_scenarios
                }
                sample_counts = {
                    s: curriculum.get_scenario_sample_count(s)
                    for s in weighted_scenarios
                }
                coverage_counts = {
                    s: curriculum.get_scenario_total_sample_count(s)
                    for s in weighted_scenarios
                }
            else:
                weighted_scenarios = available_scenarios

            sampled_scenarios = sample_scenarios_stratified(
                available_scenarios=weighted_scenarios,
                n_episodes=args.rollout_episodes,
                success_rates=success_rates,
                sample_counts=sample_counts,
                coverage_counts=coverage_counts,
                mode=args.scenario_sampling_mode,
                external_weights=soft_weights,
                coverage_groups=(
                    curriculum.get_scenario_coverage_groups(weighted_scenarios)
                    if args.scenario_sampling_mode == "weighted"
                    else None
                ),
            )
            sampling_priorities = (
                compute_scenario_sampling_priorities(
                    available_scenarios=weighted_scenarios,
                    success_rates=success_rates,
                    sample_counts=sample_counts,
                    external_weights=soft_weights,
                )
                if args.scenario_sampling_mode == "weighted"
                else {}
            )

            print(f"\n更新 {update}/{args.updates}, 可用场景数: {len(available_scenarios)}, "
                  f"采样分布: {len(set(sampled_scenarios))} 种/{len(sampled_scenarios)} 个 episode")

            # ── 收集 rollout 数据 ──
            rollout_started_at = time.perf_counter()
            for rollout_episode in range(args.rollout_episodes):
                scenario_name = sampled_scenarios[rollout_episode]
                scenario_config = env.load_scenario_by_name(scenario_name, scenario_dir)
                if scenario_config is None:
                    scenario_config = env.load_random_scenario(scenario_dir, [scenario_name])

                if scenario_config:
                    env.set_scenario(scenario_config)

                obs_dict, info = env.reset()
                total_episodes += 1

                active_mask = validate_active_mask(
                    info['active_mask'], agent_ids,
                    context=f"robust_v2 reset episode={rollout_episode}",
                )
                episode_end_reason = info['episode_end_reason']
                if episode_end_reason != "running":
                    raise RuntimeError(
                        f"robust_v2 reset episode={rollout_episode}: "
                        f"unexpected episode_end_reason={episode_end_reason!r}"
                    )
                if not any(active_mask.values()):
                    raise RuntimeError(
                        f"robust_v2 reset episode={rollout_episode}: no active agent"
                    )

                episode_rewards = {aid: 0.0 for aid in agent_ids}
                active_decision_steps = {aid: 0 for aid in agent_ids}
                episode_buffer_starts = {
                    aid: len(buffer.get_agent_data(aid)) for aid in agent_ids
                }
                ep_length = 0
                final_dones: Dict[int, bool] = {aid: False for aid in agent_ids}
                final_terminated = dict(info['terminated'])
                final_truncated = dict(info['truncated'])
                final_global_state = env.get_global_state().astype(np.float32)

                for step in range(args.max_steps):
                    ep_length += 1
                    global_state_np = env.get_global_state().astype(np.float32)
                    action_dict, step_cache = build_active_step_cache(
                        obs_dict=obs_dict,
                        active_mask=active_mask,
                        global_state_np=global_state_np,
                        actors=actors,
                        critic=critic,
                        agent_ids=agent_ids,
                        device=device,
                        context=f"robust_v2 episode={rollout_episode} step={step}",
                        value_normalizer=value_normalizer,
                    )
                    for aid in step_cache:
                        active_decision_steps[aid] += 1

                    next_obs, rewards, dones, info = env.step(action_dict)
                    episode_end_reason = info['episode_end_reason']
                    (
                        next_active_mask,
                        final_dones,
                        final_terminated,
                        final_truncated,
                    ) = validate_rollout_step_protocol(
                        dones=dones,
                        terminated=info['terminated'],
                        truncated=info['truncated'],
                        next_active_mask=info['active_mask'],
                        episode_end_reason=episode_end_reason,
                        agent_ids=agent_ids,
                        context=f"robust_v2 episode={rollout_episode} step={step}",
                    )
                    # 直接从 env.step() 返回的 observation 固化该 transition 的 s_(t+1)，
                    # 避免后续 despawn 改变重新读取的全局状态。只有合法 X 边界会使用。
                    final_global_state = np.concatenate(
                        [np.asarray(next_obs[aid], dtype=np.float32) for aid in agent_ids]
                    )

                    written_rewards = write_active_transitions(
                        buffer=buffer,
                        step_cache=step_cache,
                        rewards=rewards,
                        dones=final_dones,
                        terminated=final_terminated,
                        truncated=final_truncated,
                        scenario_name=scenario_name,
                        context=f"robust_v2 episode={rollout_episode} step={step}",
                    )
                    for aid, reward in written_rewards.items():
                        episode_rewards[aid] += reward

                    metrics_recorder.record_env_step(
                        active_agent_ids=step_cache.keys(),
                        rewards=rewards,
                        reward_details=info.get('reward_details', {}) or {},
                        reward_diagnostics_by_agent=(
                            info.get('reward_diagnostics', {}) or {}
                        ),
                        policy_actions={
                            aid: step_cache[aid]["action_tanh"]
                            for aid in step_cache
                        },
                        environment_actions=action_dict,
                        context=(
                            f"robust_v2 update={update} "
                            f"episode={rollout_episode} step={step}"
                        ),
                    )

                    if visualizer is not None:
                        visualizer.handle_events()
                        if not visualizer.running:
                            raise KeyboardInterrupt

                    next_obs = refresh_observations_after_active_change(
                        env=env,
                        next_obs=next_obs,
                        current_active_mask=active_mask,
                        next_active_mask=next_active_mask,
                        episode_end_reason=episode_end_reason,
                        agent_ids=agent_ids,
                        context=f"robust_v2 episode={rollout_episode} step={step}",
                    )
                    obs_dict = next_obs
                    active_mask = next_active_mask

                    if episode_end_reason != "running":
                        break

                if episode_end_reason == "running":
                    raise RuntimeError(
                        f"robust_v2 episode={rollout_episode}: rollout exhausted "
                        f"{args.max_steps} steps without a global episode boundary"
                    )

                # P0-00C：逐 episode/agent 对账。网络输入特征维度固定，
                # 仅允许不同 agent 的有效 transition 数动态不同。
                episode_transition_counts = validate_episode_transition_counts(
                    buffer=buffer,
                    buffer_starts=episode_buffer_starts,
                    active_decision_steps=active_decision_steps,
                    agent_ids=agent_ids,
                    context=f"robust_v2 episode={rollout_episode}",
                )
                # 严格互斥三类终局只服从环境发布的全局 end reason。这样即使
                # success/collision sticky 信号在同一 tick 同时出现，也不会把一个
                # episode 同时计入两个 outcome；逐 agent 事件仍保留在 diagnostics。
                ep_success, ep_collision, ep_timeout = classify_base_episode_outcome(
                    episode_end_reason
                )

                if ep_success:
                    total_successes += 1
                elif ep_collision:
                    total_collisions += 1
                elif ep_timeout:
                    total_timeouts += 1

                # P0-00B：bootstrap 类型只服从环境发布的最终 T/X 协议，不再根据
                # reached_target 或碰撞 sticky 标志二次猜测。X 可来自 max_steps，
                # 也可来自同伴红车碰撞造成的 agent-centric 外部中断。
                episode_bootstrap_enabled = {aid: False for aid in agent_ids}
                for aid in agent_ids:
                    boundary_count = len(buffer.episode_boundaries[aid])
                    buffer.mark_episode_end(aid)
                    if len(buffer.episode_boundaries[aid]) == boundary_count:
                        continue
                    bootstrap_state = bootstrap_state_for_boundary(
                        final_dones.get(aid, False),
                        final_terminated.get(aid, False),
                        final_truncated.get(aid, False),
                        final_global_state,
                        context=f"robust_v2 agent={aid} episode={rollout_episode}",
                    )
                    buffer.add_episode_bootstrap(aid, bootstrap_state)
                    episode_bootstrap_enabled[aid] = bootstrap_state is not None

                # 记录 episode
                avg_ret = float(np.mean([episode_rewards[aid] for aid in agent_ids]))
                window_episode_returns.append(avg_ret)
                window_episode_lengths.append(ep_length)
                window_successes.append(1.0 if ep_success else 0.0)
                window_collisions.append(1.0 if ep_collision else 0.0)
                window_timeouts.append(1.0 if ep_timeout else 0.0)
                metrics_recorder.finish_episode(
                    update=update,
                    episode_index=rollout_episode,
                    episode_returns=episode_rewards,
                    episode_length=ep_length,
                    active_decision_steps=active_decision_steps,
                    transition_counts=episode_transition_counts,
                    dones=final_dones,
                    terminated=final_terminated,
                    truncated=final_truncated,
                    bootstrap_enabled=episode_bootstrap_enabled,
                    success=ep_success,
                    collision=ep_collision,
                    timeout=ep_timeout,
                    end_reason=episode_end_reason,
                    scenario_name=scenario_name,
                    context=(
                        f"robust_v2 update={update} episode={rollout_episode}"
                    ),
                )

                curriculum.record_scenario_result(scenario_name, ep_success)

                if (
                    args.p0_smoke_interrupt_after_episodes > 0
                    and rollout_episode + 1
                    == args.p0_smoke_interrupt_after_episodes
                ):
                    print(
                        "  [P0 smoke] controlled interrupt after "
                        f"update={update} episode={rollout_episode}; "
                        "the incomplete update must be discarded on resume"
                    )
                    raise KeyboardInterrupt

            rollout_seconds = time.perf_counter() - rollout_started_at

            # ===== PPO 更新 =====
            n_transitions = buffer.total_size()
            n_transitions_per_agent = {
                aid: len(buffer.get_agent_data(aid)) for aid in agent_ids
            }
            skip_update = n_transitions < args.min_transitions_per_update
            enforce_formal_transition_minimum(
                args=args, total_transitions=n_transitions
            )
            ppo_metrics = None
            advantage_stats: Dict[int, Dict[str, float]] = {}
            agent_stop_epoch: Dict[int, int] = {}
            preupdate_ratio_errors: Dict[int, float] = {}
            metrics_recorder.capture_model_snapshot(actors, critic)
            ppo_started_at = time.perf_counter()

            if skip_update:
                print(f"  [跳过] 总 transition 数 {n_transitions} < {args.min_transitions_per_update}，跳过 PPO 更新")
            else:
                prepared_batch = prepare_mappo_update_batch(
                    buffer=buffer,
                    critic=critic,
                    agent_ids=agent_ids,
                    device=device,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    obs_dim=19,
                    state_dim=state_dim,
                    action_dim=ACTION_DIM,
                    context=f"robust_v2 update={update}",
                    value_normalizer=value_normalizer,
                )
                advantage_stats = prepared_batch.advantage_stats

                update_result = run_mappo_ppo_update(
                    prepared=prepared_batch,
                    actors=actors,
                    actor_optims=actor_optims,
                    critic=critic,
                    critic_optim=critic_optim,
                    agent_ids=agent_ids,
                    obs_dim=19,
                    state_dim=state_dim,
                    action_dim=ACTION_DIM,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    clip_eps=args.clip_eps,
                    ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm,
                    target_kl=args.target_kl,
                    value_normalizer=value_normalizer,
                    huber_delta=args.huber_delta,
                    shuffle=True,
                    context=f"robust_v2 update={update}",
                )
                ppo_metrics = update_result.metrics
                agent_stop_epoch = update_result.agent_stop_epoch
                preupdate_ratio_errors = update_result.preupdate_ratio_errors
            ppo_seconds = time.perf_counter() - ppo_started_at
            metrics_recorder.record_model_update(actors, critic)

            # ===== 版本化、模块化指标记录 =====
            # 每个 update 都落盘；--log-interval 只控制控制台打印频率。
            actor_lrs = {
                aid: float(actor_optims[aid].param_groups[0]["lr"])
                for aid in agent_ids
            }
            critic_lr = float(critic_optim.param_groups[0]["lr"])
            metrics_recorder.record_ppo_update(
                ppo_metrics=ppo_metrics,
                advantage_stats=advantage_stats,
                preupdate_ratio_errors=preupdate_ratio_errors,
                agent_stop_epoch=agent_stop_epoch,
                transition_counts=n_transitions_per_agent,
                actor_lrs=actor_lrs,
                critic_lr=critic_lr,
                skipped=skip_update,
                max_grad_norm=args.max_grad_norm,
            )
            metrics_recorder.record_scalars(
                {
                    "episode/success_rate_cumulative": (
                        total_successes / max(total_episodes, 1)
                    ),
                    "episode/collision_rate_cumulative": (
                        total_collisions / max(total_episodes, 1)
                    ),
                    "episode/timeout_rate_cumulative": (
                        total_timeouts / max(total_episodes, 1)
                    ),
                    "train/total_episodes_cumulative": total_episodes,
                    "curriculum/available_scenario_count": len(available_scenarios),
                    "curriculum/sampled_unique_scenario_count": len(
                        set(sampled_scenarios)
                    ),
                    **{
                        f"sampling/{scenario}/batch_count": sampled_scenarios.count(
                            scenario
                        )
                        for scenario in weighted_scenarios
                    },
                    **{
                        f"sampling/{scenario}/priority": priority
                        for scenario, priority in sampling_priorities.items()
                    },
                    "timing/rollout_seconds": rollout_seconds,
                    "timing/ppo_seconds": ppo_seconds,
                    "timing/env_transitions_per_second": (
                        n_transitions / max(rollout_seconds, 1e-12)
                    ),
                    "timing/update_seconds_before_commit": (
                        time.perf_counter() - update_started_at
                    ),
                }
            )

            if update % args.log_interval == 0 and len(window_episode_returns) > 0:
                avg_r = float(np.mean(window_episode_returns))
                win_succ = float(np.mean(window_successes))
                win_coll = float(np.mean(window_collisions))
                win_to = float(np.mean(window_timeouts))

                def _mean_comp(name):
                    if name not in metrics_recorder.component_names:
                        return 0.0
                    return metrics_recorder.reward_component_mean(name)

                print(f"更新 {update}: avg_return={avg_r:.2f}, "
                      f"success={win_succ:.2%}, collision={win_coll:.2%}, timeout={win_to:.2%}, "
                      f"cum_success={total_successes/max(total_episodes,1):.2%}, "
                      f"avg_ep_len={float(np.mean(window_episode_lengths)):.0f}, "
                      f"n_trans={n_transitions_per_agent} (total={n_transitions}), "
                      f"lr_a={cur_lr_a:.2e}")
                if args.reward_version == "v3":
                    print(
                        "   reward-v3分量(均值/有效步): "
                        f"progress={_mean_comp('progress'):.4f}, "
                        f"overspeed={_mean_comp('overspeed'):.4f}, "
                        f"long_rate={_mean_comp('longitudinal_rate'):.4f}, "
                        f"steer_rate={_mean_comp('steer_rate'):.4f}, "
                        f"safety+ttc={(_mean_comp('safety_lidar') + _mean_comp('ttc')):.4f}, "
                        f"lane={(_mean_comp('lane_deviation') + _mean_comp('lane_invasion')):.4f}, "
                        f"collision={_mean_comp('collision'):.4f}, "
                        f"success={_mean_comp('success'):.4f}"
                    )
                else:
                    print(
                        "   reward-v2分量(均值/有效步): "
                        f"progress={_mean_comp('progress'):.3f}, "
                        f"approach={_mean_comp('approach'):.3f}, "
                        f"speed={_mean_comp('speed'):.3f}, "
                        f"overspeed={_mean_comp('overspeed'):.3f}, "
                        f"alive={_mean_comp('alive'):.3f}, "
                        f"safety+ttc={(_mean_comp('safety_lidar') + _mean_comp('ttc')):.3f}, "
                        f"lane={_mean_comp('lane_invasion'):.3f}, "
                        f"collision={_mean_comp('collision'):.3f}, "
                        f"step={_mean_comp('step'):.3f}, "
                        f"success+bonus={(_mean_comp('success') + _mean_comp('time_bonus')):.3f}"
                    )

            # 先冻结本 update 指标并写入 checkpoint，再写 TensorBoard/JSONL。
            # 若在两个动作之间中断，resume 会从 checkpoint 重放同一指标 payload。
            metrics_payload = metrics_recorder.finalize_update(update)
            completed_update = update
            last_metrics_payload = copy.deepcopy(metrics_payload)
            _save_safe_checkpoint(
                include_periodic=(update % args.save_interval == 0)
            )
            metrics_recorder.commit_payload(metrics_payload)
        experiment_status = "SEGMENT_COMPLETED"

    except KeyboardInterrupt:
        experiment_status = "INTERRUPTED"
        experiment_notes = "user interrupt; current incomplete update discarded"
        print("训练中断：丢弃当前未完成 update，保留最近完整边界")
        try:
            _save_interrupted_checkpoint()
        except Exception as save_error:
            print(f"  [警告] interrupted checkpoint 保存失败: {save_error}")
    except Exception as error:
        experiment_status = (
            "HOLD" if experiment_contract_metadata else "ERROR"
        )
        experiment_notes = f"{type(error).__name__}: {error}"
        try:
            _save_interrupted_checkpoint()
        except Exception as save_error:
            print(f"  [警告] 异常退出时保存 interrupted 失败: {save_error}")
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors = []
        if visualizer is not None:
            try:
                visualizer.close()
            except Exception as error:
                cleanup_errors.append(f"visualizer.close: {error}")
        try:
            env.close()
        except Exception as error:
            cleanup_errors.append(f"env.close: {error}")
        try:
            metrics_recorder.close()
        except Exception as error:
            cleanup_errors.append(f"metrics_recorder.close: {error}")
        if cleanup_errors:
            cleanup_note = "; ".join(cleanup_errors)
            experiment_notes = "; ".join(
                note for note in (experiment_notes, cleanup_note) if note
            )
            experiment_status = (
                "HOLD" if experiment_contract_metadata else "ERROR"
            )

        evidence_error = None
        try:
            update_experiment_evidence(
                args=args,
                run_dir=Path(save_dir),
                completed_update=completed_update,
                status=experiment_status,
                notes=experiment_notes,
            )
        except Exception as error:
            evidence_error = error
        if evidence_error is not None:
            if active_exception:
                print(f"  [警告] 实验台账最终状态写入失败: {evidence_error}")
            else:
                raise evidence_error
        if cleanup_errors and not active_exception:
            raise RuntimeError(
                "training cleanup failed: " + "; ".join(cleanup_errors)
            )


def build_parser():
    parser = argparse.ArgumentParser(description="鲁棒 MAPPO 训练脚本 v2（改进场景采样 + GAE修复 + 软过渡）")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town05")
    parser.add_argument("--straight-road-length", type=float, default=50.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 训练参数
    parser.add_argument("--updates", type=int, default=100, help="总更新次数")
    parser.add_argument("--rollout-episodes", type=int, default=16,
                        help="每次更新收集的 episode 数（v2: 4→16）")
    parser.add_argument("--ppo-epochs", type=int, default=5,
                        help="PPO epoch 数（v2: 10→5）")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--min-transitions-per-update", type=int, default=512,
                        help="最少 transition 数，低于此值跳过 PPO 更新")

    # 学习率
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997,
                        help="折扣因子（0.997 的有效视野约 330 步；reward v3 的路线势能"
                             "必须使用与 PPO 相同的 gamma 才能保持望远镜性质）")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.15,
                        help="PPO clip 系数（v2 修复：0.2→0.15，小批量训练更稳）")
    parser.add_argument("--ent-coef", type=float, default=0.03,
                        help="熵损失系数（v2 修复：0.01→0.03），防止 log_std 早期塌缩")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--value-norm-beta", type=float, default=0.99999,
        help="ValueNorm running-moment decay (official MAPPO default)",
    )
    parser.add_argument(
        "--value-norm-epsilon", type=float, default=1e-5,
        help="ValueNorm debiasing numerical floor",
    )
    parser.add_argument(
        "--value-norm-variance-floor", type=float, default=1e-2,
        help="Minimum ValueNorm variance (official MAPPO implementation)",
    )
    parser.add_argument(
        "--huber-delta", type=float, default=10.0,
        help="Huber threshold in normalized return units",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-kl", type=float, default=0.02,
                        help="PPO 单 agent 单 epoch approx_kl 超过 1.5×target-kl 时早停")
    parser.add_argument("--lr-decay", action="store_true", default=True,
                        help="启用 lr 线性衰减到 0（默认开启）")
    parser.add_argument("--no-lr-decay", dest="lr_decay", action="store_false",
                        help="关闭 lr 线性衰减")
    parser.add_argument(
        "--lr-decay-horizon",
        type=int,
        default=0,
        help=(
            "冻结的学习率衰减 update 长度；新训练为 0 时取 --updates，"
            "resume 时从 checkpoint 恢复且禁止隐式改变"
        ),
    )
    parser.add_argument("--init-log-std", type=float, default=0.0,
                        help="Actor log_std 初始值（std=1.0），配合 target-kl 早停，给足初期探索空间")

    # 场景采样
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default="",
        help=(
            "场景目录；默认 data/training_scenarios_town05_v1。"
            "Town02 历史实验必须显式传旧场景目录和地图"
        ),
    )
    parser.add_argument("--scenario-sampling-mode", type=str, default="weighted",
                        choices=["random", "stratified", "weighted"],
                        help="场景采样模式: random=随机, stratified=分层轮询, weighted=分层+失败率加权")

    # 课程学习
    parser.add_argument("--warmup-updates", type=int, default=10, help="基础场景预热次数")
    parser.add_argument("--full-scenario-updates", type=int, default=40, help="达到全场景的更新次数")
    parser.add_argument("--curriculum-transition-window", type=int, default=10,
                        help="阶段过渡窗口大小（v2 新增）")
    parser.add_argument("--curriculum-mode", type=str, default="metadata",
                        choices=["metadata", "classic", "distance"],
                        help="课程模式: metadata=读取 JSON difficulty(默认); "
                             "classic=basic/left|right/all 旧逻辑; "
                             "distance=按 dist_<N>m 距离分层(实验性)")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--follow-spectator",
        action="store_true",
        help="每次场景初始化后自动把 CARLA spectator 移到覆盖出生点与目标区的俯视镜头；不启用 pygame 可视化",
    )
    parser.add_argument("--pretrained-checkpoint", type=str, default="",
                        help="从权重启动新 Run；不恢复 optimizer/update/RNG")
    parser.add_argument("--resume", type=str, default="",
                        help=(
                            "严格恢复同一 Run 的完整状态；传 latest 加载 "
                            "save_dir/latest.pt。不存在或不兼容时直接失败"
                        ))
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="唯一实验 RunID；省略时使用 save_dir 的目录名",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="",
        help="可选：记录到 checkpoint 的冻结配置文件路径及 SHA256",
    )
    parser.add_argument(
        "--experiment-contract",
        type=str,
        default="",
        help=(
            "P0-BASE 正式实验契约 JSON；启用后科学参数仅由契约提供，"
            "CLI 必须显式给出 --updates/--run-id/--save-dir"
        ),
    )
    parser.add_argument(
        "--allow-resume-code-drift",
        action="store_true",
        help="显式允许 Git SHA/关键源码指纹不同的续训，并永久写入元数据",
    )
    parser.add_argument(
        "--p0-smoke-interrupt-after-episodes",
        type=int,
        default=0,
        help=(
            "仅用于 P0 smoke：resume rollout 写完指定数量的 episode 后触发受控中断；"
            "0 表示禁用，不改变 checkpoint 中冻结的训练配置"
        ),
    )

    parser.add_argument("--save-dir", type=str, default="./logs/mappo_robust_v2")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--tb-histogram-interval",
        type=int,
        default=10,
        help="参数直方图写入间隔；0 表示关闭，标量仍每个 update 记录",
    )
    parser.add_argument(
        "--reward-version",
        type=str,
        default="v3",
        choices=["v1", "v2", "v3"],
        help="奖励函数版本；当前基础策略使用 v3 路线完成度预算",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args._explicit_cli_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    train(args)
