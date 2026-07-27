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
import os
import re
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from visualization.mappo_visualizer import MAPPOVisualizer
from models.actor_critic import Actor, AgentConditionedCritic
from training.lifecycle_semantics import (
    bootstrap_state_for_boundary,
    validate_lifecycle_flags,
)
from training.checkpoint_protocol import (
    ACTION_PROTOCOL_VERSION,
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

OBS_DIM = 19
ACTION_DIM = 3

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
) -> Dict[str, object]:
    """Load weights only; optimizer/progress/RNG deliberately remain fresh."""
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
    if "critic" in checkpoint:
        try:
            critic.load_state_dict(checkpoint["critic"], strict=True)
            critic_loaded = True
            critic_error = ""
        except (RuntimeError, ValueError) as error:
            critic_error = str(error)
    return {
        "actors_loaded": sorted(actors),
        "critic_loaded": critic_loaded,
        "critic_note": critic_error,
    }


def runtime_metadata_with_carla(env) -> Dict[str, object]:
    metadata = runtime_metadata()
    try:
        metadata["carla_client"] = env.client.get_client_version()
        metadata["carla_server"] = env.client.get_server_version()
    except Exception as error:
        metadata["carla_version_error"] = str(error)
    return metadata


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

def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    """Convert tanh action [-1,1] to env action format [throttle, steer, brake]."""
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5

    if throttle_raw >= brake_raw:
        throttle = throttle_raw
        brake = 0.0
    else:
        throttle = 0.0
        brake = brake_raw

    return np.array([throttle, steer, brake], dtype=np.float32)


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
            value = float(critic(global_state_t, aid).item())
            action_tanh_t, logp_t, z_t = actors[aid].sample_action_z(obs_t)

            action_tanh = action_tanh_t.squeeze(0).cpu().numpy()
            action_z = z_t.squeeze(0).cpu().numpy()
            action_dict[aid] = action_tanh_to_env(action_tanh)
            step_cache[aid] = {
                "obs": obs.copy(),
                "global_state": global_state_np.copy(),
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
    new_log_probs, entropy = actor.evaluate_actions_z(observations, action_z)
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
                bootstrap_prediction = critic(bootstrap_tensor, aid)
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
    actor_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()
    actor_loss = actor_loss - ent_coef * entropy.mean()
    ensure_finite_tensor(actor_loss, name="actor_loss", context=context)
    return actor_loss, {
        "entropy": float(entropy.mean().item()),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def compute_critic_value_loss(
    value_predictions: torch.Tensor,
    returns: torch.Tensor,
    *,
    context: str,
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
    loss = 0.5 * (returns - value_predictions).pow(2).mean()
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
            f"entropy_{aid}": [],
            f"approx_kl_{aid}": [],
            f"clip_frac_{aid}": [],
            f"grad_norm_actor_{aid}": [],
            f"actor_samples_seen_{aid}": 0,
        })

    agent_stopped = {aid: False for aid in agent_ids}
    agent_stop_epoch = {aid: ppo_epochs for aid in agent_ids}
    critic_state_t = prepared.critic_states
    critic_ret_t = prepared.critic_returns
    critic_aid_t = prepared.critic_agent_ids

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
                value_predictions, critic_ret_t[mb], context=critic_context
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
            epoch_entropies = []
            epoch_approx_kls = []
            epoch_clip_fracs = []
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
                epoch_entropies.append(diagnostics["entropy"])
                epoch_approx_kls.append(diagnostics["approx_kl"])
                epoch_clip_fracs.append(diagnostics["clip_fraction"])
                epoch_actor_grad_norms.append(float(actor_grad_norm))
                metrics[f"actor_samples_seen_{aid}"] += int(len(mb))

            metrics[f"actor_loss_{aid}"].append(
                float(np.mean(epoch_actor_losses))
            )
            metrics[f"entropy_{aid}"].append(
                float(np.mean(epoch_entropies))
            )
            mean_epoch_kl = float(np.mean(epoch_approx_kls))
            metrics[f"approx_kl_{aid}"].append(mean_epoch_kl)
            metrics[f"clip_frac_{aid}"].append(
                float(np.mean(epoch_clip_fracs))
            )
            metrics[f"grad_norm_actor_{aid}"].append(
                float(np.mean(epoch_actor_grad_norms))
            )
            if should_stop_for_kl(mean_epoch_kl, target_kl):
                agent_stopped[aid] = True
                agent_stop_epoch[aid] = epoch + 1

    with torch.no_grad():
        value_predictions = critic(critic_state_t, critic_aid_t)
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

    return MAPPOUpdateResult(
        metrics=metrics,
        agent_stop_epoch=agent_stop_epoch,
        preupdate_ratio_errors=preupdate_ratio_errors,
    )


# ============================================================
# 场景采样策略
# ============================================================

def sample_scenarios_stratified(
    available_scenarios: List[str],
    n_episodes: int,
    success_rates: Optional[Dict[str, float]] = None,
    mode: str = "weighted",
    external_weights: Optional[Dict[str, float]] = None,
) -> List[str]:
    """从可用场景中采样 n_episodes 个场景。

    Args:
        available_scenarios: 可用场景文件名列表
        n_episodes: 需要采样的 episode 数
        success_rates: 各场景的历史成功率（weighted 模式需要）
        mode: "random" | "stratified" | "weighted"
        external_weights: 外部权重（如课程软过渡权重），与 success_rates 相乘

    Returns:
        长度为 n_episodes 的场景文件名列表
    """
    if mode == "random" or len(available_scenarios) == 0:
        return [np.random.choice(available_scenarios) for _ in range(n_episodes)]

    n_scenarios = len(available_scenarios)
    selected = []

    if mode == "stratified":
        # 分层轮询：每个场景至少分配 1 个 episode
        rounds = n_episodes // n_scenarios
        remainder = n_episodes % n_scenarios
        for _ in range(rounds):
            selected.extend(available_scenarios)
        if remainder > 0:
            selected.extend(np.random.choice(available_scenarios, size=remainder, replace=False).tolist())

    elif mode == "weighted":
        # 分层轮询保证覆盖
        if n_episodes >= n_scenarios:
            selected.extend(available_scenarios)
            remaining = n_episodes - n_scenarios
        else:
            # episode 数少于场景数，随机选 n_episodes 个各 1 次
            selected.extend(np.random.choice(available_scenarios, size=n_episodes, replace=False).tolist())
            remaining = 0

        # 剩余配额按失败率加权（合并外部权重如课程软过渡）
        if remaining > 0:
            weights = []
            for s in available_scenarios:
                w = 1.0
                if success_rates is not None:
                    rate = success_rates.get(s, 0.5)
                    w *= (1.0 - rate + 0.1)  # 失败率 + 最小权重
                if external_weights is not None:
                    w *= external_weights.get(s, 1.0)
                weights.append(w)
            total_w = sum(weights)
            if total_w > 0:
                probs = [w / total_w for w in weights]
                extra = np.random.choice(available_scenarios, size=remaining, p=probs, replace=True).tolist()
            else:
                extra = np.random.choice(available_scenarios, size=remaining, replace=True).tolist()
            selected.extend(extra)

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

        if curriculum_stages is None:
            # mode="classic" 强制走 basic/left|right/all 旧逻辑（回退距离课程，
            # 即使目录里存在 dist_<N>m 场景文件也忽略其距离分层）
            self.curriculum_stages = self._build_default_stages(
                all_scenario_files, use_distance=(mode == "distance")
            )
        else:
            self.curriculum_stages = curriculum_stages

        self.scenario_success_history: Dict[str, List[float]] = {f: [] for f in all_scenario_files}
        print(f"课程学习初始化(mode={mode})，共 {len(self.curriculum_stages)} 个阶段，{len(all_scenario_files)} 个场景")
        print(f"  阶段构成: {[len(s) for s in self.curriculum_stages]}（近→远 / 易→难）")
        print(f"  过渡窗口: {transition_window} updates")

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
        stages.append(list(set(stage2)))
        stages.append(pool)
        return stages

    def _get_stage_idx(self, current_update: int) -> int:
        """获取当前所处的阶段索引"""
        if current_update < self.warmup_updates:
            return 0
        elif current_update >= self.full_scenario_updates:
            return len(self.curriculum_stages) - 1
        else:
            progress = (current_update - self.warmup_updates) / (
                self.full_scenario_updates - self.warmup_updates
            )
            return min(int(progress * len(self.curriculum_stages)), len(self.curriculum_stages) - 2)

    def _compute_transition_alpha(self, current_update: int) -> float:
        """计算软过渡系数 α ∈ [0, 1]。

        α=0 表示完全使用旧阶段场景，α=1 表示完全使用当前阶段场景。
        在阶段切换点附近线性过渡。
        """
        stage_idx = self._get_stage_idx(current_update)
        if stage_idx == 0:
            return 1.0  # 第一阶段，无旧场景

        # 上一阶段切换到当前阶段的边界 update 数
        # 简化：阶段 i 的起点大约在 warmup_updates + (full_scenario_updates - warmup_updates) * i / n_stages
        n_stages = len(self.curriculum_stages)
        if n_stages <= 1:
            return 1.0

        # 与 _get_stage_idx 的实际过渡点对齐：
        # 非末阶段: int(progress * n) >= stage_idx → update >= warmup + stage_idx/n * (full - warmup)
        # 末阶段: update >= full_scenario_updates（_get_stage_idx 的显式检查）
        if stage_idx == n_stages - 1:
            stage_start = float(self.full_scenario_updates)
        else:
            stage_start = self.warmup_updates + (self.full_scenario_updates - self.warmup_updates) * stage_idx / n_stages
        half_window = self.transition_window / 2.0

        if current_update < stage_start - half_window:
            return 0.0  # 完全旧阶段（不应出现，因为 _get_stage_idx 已经判断）
        elif current_update > stage_start + half_window:
            return 1.0  # 完全当前阶段
        else:
            # 线性过渡
            return (current_update - (stage_start - half_window)) / self.transition_window

    def get_available_scenarios(self, current_update: int) -> List[str]:
        """获取当前阶段的场景列表（硬切换，向后兼容）"""
        stage_idx = self._get_stage_idx(current_update)
        return self.curriculum_stages[min(stage_idx, len(self.curriculum_stages) - 1)]

    def get_scenario_sampling_weights(self, current_update: int) -> Dict[str, float]:
        """获取场景采样权重（软过渡版本）。

        返回 {scenario_name: weight} 字典。
        在阶段过渡期，旧阶段场景以递减权重参与采样。
        """
        alpha = self._compute_transition_alpha(current_update)
        stage_idx = self._get_stage_idx(current_update)

        current_scenarios = self.curriculum_stages[min(stage_idx, len(self.curriculum_stages) - 1)]
        weights = {}

        # 当前阶段场景权重
        for s in current_scenarios:
            weights[s] = 1.0

        # 如果处于过渡期（α < 1），混入旧阶段场景
        if alpha < 1.0 and stage_idx > 0:
            prev_scenarios = self.curriculum_stages[stage_idx - 1]
            prev_weight = 1.0 - alpha
            for s in prev_scenarios:
                if s in weights:
                    weights[s] = max(weights[s], prev_weight)
                else:
                    weights[s] = prev_weight

        return weights

    def record_scenario_result(self, scenario_name: str, success: bool):
        success_float = 1.0 if success else 0.0
        if scenario_name not in self.scenario_success_history:
            self.scenario_success_history[scenario_name] = []
        self.scenario_success_history[scenario_name].append(success_float)

        if len(self.scenario_success_history[scenario_name]) > 20:
            self.scenario_success_history[scenario_name] = (
                self.scenario_success_history[scenario_name][-20:]
            )

    def get_scenario_success_rate(self, scenario_name: str) -> float:
        if scenario_name not in self.scenario_success_history or len(self.scenario_success_history[scenario_name]) == 0:
            return 0.0
        return float(np.mean(self.scenario_success_history[scenario_name]))


# ============================================================
# 训练主函数
# ============================================================

def train(args):
    if args.resume and args.pretrained_checkpoint:
        raise ValueError(
            "--resume and --pretrained-checkpoint are mutually exclusive"
        )
    if args.updates <= 0:
        raise ValueError("--updates must be positive")
    if args.save_interval <= 0:
        raise ValueError("--save-interval must be positive")
    if args.lr_decay_horizon < 0:
        raise ValueError("--lr-decay-horizon must be non-negative")

    save_dir = os.path.abspath(args.save_dir)
    repository_root = Path(__file__).resolve().parents[4]
    current_git = get_git_metadata(repository_root)
    code_fingerprint = files_fingerprint(
        [
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("checkpoint_protocol.py"),
            Path(__file__).resolve().with_name("lifecycle_semantics.py"),
            Path(__file__).resolve().parents[1] / "models" / "actor_critic.py",
            Path(__file__).resolve().parents[1]
            / "environment"
            / "carla_multi_agent_env.py",
        ]
    )
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
        apply_resume_configuration(args, resume_checkpoint)

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

    scenario_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "training_scenarios",
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
            pretrained_checkpoint, actors, critic
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
    model_spec = checkpoint_model_spec(
        agent_ids, state_dim, args.hidden_dim
    )
    training_config = checkpoint_training_config(args)
    pool_fingerprint = scenario_pool_fingerprint(scenario_dir)

    def _create_validated_env():
        created_env = CarlaMultiAgentEnv(
            host=args.host,
            port=args.port,
            town=args.town,
            num_agents=len(agent_ids),
            straight_road_length=args.straight_road_length,
            timestep=args.timestep,
            max_steps=args.max_steps,
            spawn_vehicles=True,
            reward_version=args.reward_version,
            disable_tensorboard=True,
            seed=args.seed,
        )
        actual_state_dim = int(created_env.get_global_state().shape[0])
        if (
            created_env.num_agents != len(agent_ids)
            or actual_state_dim != state_dim
        ):
            created_env.close()
            raise CheckpointProtocolError(
                "runtime environment shape differs from frozen protocol: "
                f"agents={created_env.num_agents}/{len(agent_ids)}, "
                f"state_dim={actual_state_dim}/{state_dim}"
            )
        print(f"全局状态维度: {actual_state_dim}")
        return created_env

    if resume_checkpoint is not None:
        validate_checkpoint_payload(
            resume_checkpoint,
            expected_protocol_versions=protocol_versions,
            expected_model_spec=model_spec,
            expected_training_config=training_config,
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
        if start_update > args.updates:
            raise CheckpointProtocolError(
                f"checkpoint completed update={completed_update}, but "
                f"--updates={args.updates}; choose a larger stop update"
            )

        env = _create_validated_env()
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
            }
        )
    else:
        env = _create_validated_env()
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
    tb_log_dir = os.path.join(save_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(
        tb_log_dir,
        purge_step=start_update if resume_checkpoint is not None else None,
    )
    print(f"RunID: {args.run_id}")
    print(f"TensorBoard 日志: {tb_log_dir}")

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
            "checkpoint_kind": kind,
            "update": int(completed_update),
            "actors": {
                aid: actors[aid].state_dict() for aid in agent_ids
            },
            "critic": critic.state_dict(),
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
            },
            "rng_state": capture_rng_state(),
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

    try:
        if resume_checkpoint is None:
            _save_safe_checkpoint(include_periodic=False)
        for update in range(start_update, args.updates + 1):
            # 学习率线性衰减
            cur_lr_a, cur_lr_c = _apply_lr_decay(update)

            buffer = RolloutBuffer(agent_ids)

            # 当前 update 窗口统计
            window_episode_returns: List[float] = []
            window_episode_lengths: List[int] = []
            window_successes: List[float] = []          # 严格成功（all reached_target）
            window_collisions: List[float] = []         # 任一 agent 发生碰撞
            window_timeouts: List[float] = []           # max_steps 截断且未成功未碰撞
            window_per_agent_returns: Dict[int, List[float]] = {aid: [] for aid in agent_ids}
            window_reward_components: Dict[str, List[float]] = {}

            # ── 场景采样（分层/加权） ──
            available_scenarios = curriculum.get_available_scenarios(update)

            # 获取成功率（weighted 模式需要）
            success_rates = None
            if args.scenario_sampling_mode == "weighted":
                success_rates = {
                    s: curriculum.get_scenario_success_rate(s)
                    for s in available_scenarios
                }

            # 软过渡权重
            soft_weights = None
            if args.scenario_sampling_mode != "random":
                soft_weights = curriculum.get_scenario_sampling_weights(update)
                weighted_scenarios = list(soft_weights.keys())
            else:
                weighted_scenarios = available_scenarios

            sampled_scenarios = sample_scenarios_stratified(
                available_scenarios=weighted_scenarios,
                n_episodes=args.rollout_episodes,
                success_rates=success_rates,
                mode=args.scenario_sampling_mode,
                external_weights=soft_weights,
            )

            print(f"\n更新 {update}/{args.updates}, 可用场景数: {len(available_scenarios)}, "
                  f"采样分布: {len(set(sampled_scenarios))} 种/{len(sampled_scenarios)} 个 episode")

            # ── 收集 rollout 数据 ──
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
                ep_success = False
                # Phase A2/A1: 必须在 rollout 循环里累积碰撞标志
                # env.collision_events / crosswalk_invasion_events 在每步末尾被环境清零
                # （见 carla_multi_agent_env.py:978-984），episode 结束后读是 0
                # 这里用 info['collisions'] 每步 OR 累积，episode 内 sticky
                ep_collision_sticky: Dict[int, bool] = {aid: False for aid in agent_ids}
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

                    # Phase A1: 每步累积"本回合是否撞过"（info['collisions'] 已含人行横道）
                    step_collisions = info.get('collisions', {}) or {}
                    for aid in agent_ids:
                        if step_collisions.get(aid, False):
                            ep_collision_sticky[aid] = True

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

                    if visualizer is not None:
                        visualizer.handle_events()
                        if not visualizer.running:
                            raise KeyboardInterrupt

                    if 'reward_details' in info:
                        for aid in step_cache:
                            details = info['reward_details'].get(aid, {})
                            for key, val in details.items():
                                comp_name = f"reward/{key}"
                                if comp_name not in window_reward_components:
                                    window_reward_components[comp_name] = []
                                window_reward_components[comp_name].append(float(val))

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
                for aid in agent_ids:
                    tb_writer.add_scalar(
                        f'rollout/active_decision_steps_agent_{aid}',
                        active_decision_steps[aid], total_episodes,
                    )
                    tb_writer.add_scalar(
                        f'rollout/buffer_transition_count_agent_{aid}',
                        episode_transition_counts[aid], total_episodes,
                    )

                # Phase A1: 严格三类终止分类
                #   ep_success_to_goal: 所有 agent 都进入 goal_zone（True success）
                #   ep_collision:       任一 agent 本回合撞过（含人行横道）
                #   ep_timeout:         max_steps 截断且既未成功也未碰撞
                # reached_target 是 sticky，可直接读 env；碰撞用循环里累积的 sticky 字典
                ep_success_to_goal = all(
                    env.reached_target.get(aid, False) for aid in agent_ids
                )
                ep_collision_per_agent = dict(ep_collision_sticky)
                ep_collision = any(ep_collision_per_agent.values())
                ep_timeout = (
                    ep_length >= args.max_steps
                    and not ep_success_to_goal
                    and not ep_collision
                )
                ep_success = ep_success_to_goal

                if ep_success:
                    total_successes += 1
                elif ep_collision:
                    total_collisions += 1
                elif ep_timeout:
                    total_timeouts += 1

                # P0-00B：bootstrap 类型只服从环境发布的最终 T/X 协议，不再根据
                # reached_target 或碰撞 sticky 标志二次猜测。X 可来自 max_steps，
                # 也可来自同伴红车碰撞造成的 agent-centric 外部中断。
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

                # 记录 episode
                avg_ret = float(np.mean([episode_rewards[aid] for aid in agent_ids]))
                window_episode_returns.append(avg_ret)
                window_episode_lengths.append(ep_length)
                window_successes.append(1.0 if ep_success else 0.0)
                window_collisions.append(1.0 if ep_collision else 0.0)
                window_timeouts.append(1.0 if ep_timeout else 0.0)
                for aid in agent_ids:
                    window_per_agent_returns[aid].append(episode_rewards[aid])

                curriculum.record_scenario_result(scenario_name, ep_success)

            # ===== PPO 更新 =====
            n_transitions = buffer.total_size()
            n_transitions_per_agent = {
                aid: len(buffer.get_agent_data(aid)) for aid in agent_ids
            }
            skip_update = n_transitions < args.min_transitions_per_update

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
                    action_dim=3,
                    context=f"robust_v2 update={update}",
                )
                per_agent_tensors = prepared_batch.per_agent_tensors
                advantage_stats = prepared_batch.advantage_stats
                critic_states = prepared_batch.critic_states

                update_result = run_mappo_ppo_update(
                    prepared=prepared_batch,
                    actors=actors,
                    actor_optims=actor_optims,
                    critic=critic,
                    critic_optim=critic_optim,
                    agent_ids=agent_ids,
                    obs_dim=19,
                    state_dim=state_dim,
                    action_dim=3,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch_size,
                    clip_eps=args.clip_eps,
                    ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm,
                    target_kl=args.target_kl,
                    shuffle=True,
                    context=f"robust_v2 update={update}",
                )
                ppo_metrics = update_result.metrics
                agent_stop_epoch = update_result.agent_stop_epoch
                preupdate_ratio_errors = update_result.preupdate_ratio_errors

            # ===== TensorBoard 记录 =====
            if update % args.log_interval == 0 and len(window_episode_returns) > 0:
                tb_writer.add_scalar('episode/avg_return',
                    float(np.mean(window_episode_returns)), update)
                tb_writer.add_scalar('episode/std_return',
                    float(np.std(window_episode_returns)), update)
                tb_writer.add_scalar('episode/avg_length',
                    float(np.mean(window_episode_lengths)), update)
                # Phase A1: 3 类终止分别记录
                tb_writer.add_scalar('episode/rate_success_to_goal',
                    float(np.mean(window_successes)), update)
                tb_writer.add_scalar('episode/rate_collision',
                    float(np.mean(window_collisions)), update)
                tb_writer.add_scalar('episode/rate_timeout',
                    float(np.mean(window_timeouts)), update)
                # 向后兼容：保留旧 key，但语义已改为严格 success
                tb_writer.add_scalar('episode/success_rate_window',
                    float(np.mean(window_successes)), update)
                tb_writer.add_scalar('episode/success_rate_cumulative',
                    total_successes / max(total_episodes, 1), update)
                tb_writer.add_scalar('episode/collision_rate_cumulative',
                    total_collisions / max(total_episodes, 1), update)
                tb_writer.add_scalar('episode/timeout_rate_cumulative',
                    total_timeouts / max(total_episodes, 1), update)
                tb_writer.add_scalar('episode/n_transitions',
                    n_transitions, update)
                for aid in agent_ids:
                    tb_writer.add_scalar(
                        f'episode/n_transitions_agent_{aid}',
                        n_transitions_per_agent[aid], update,
                    )

                # Phase A3: lr / log_std 监控
                tb_writer.add_scalar('train/lr_actor', cur_lr_a, update)
                tb_writer.add_scalar('train/lr_critic', cur_lr_c, update)
                for aid in agent_ids:
                    with torch.no_grad():
                        log_std_mean = float(actors[aid].log_std.mean().item())
                    tb_writer.add_scalar(f'train/log_std_agent{aid}', log_std_mean, update)

                for aid in agent_ids:
                    if window_per_agent_returns[aid]:
                        tb_writer.add_scalar(f'episode/return_agent_{aid}',
                            float(np.mean(window_per_agent_returns[aid])), update)

                avg_r = float(np.mean(window_episode_returns))
                win_succ = float(np.mean(window_successes))
                win_coll = float(np.mean(window_collisions))
                win_to = float(np.mean(window_timeouts))
                # Phase B2: 关键 reward 分量均值，验证是否还有"progress 主导"
                def _mean_comp(name):
                    vals = window_reward_components.get(f"reward/{name}", [])
                    return float(np.mean(vals)) if vals else 0.0
                comp_progress = _mean_comp("progress")
                comp_approach = _mean_comp("approach")
                comp_lane = _mean_comp("lane_invasion")
                comp_collision = _mean_comp("collision")
                comp_ttc = _mean_comp("safety_lidar") + _mean_comp("ttc")
                comp_success = _mean_comp("success") + _mean_comp("time_bonus")
                comp_alive = _mean_comp("alive")
                comp_step = _mean_comp("step")
                comp_speed = _mean_comp("speed")
                print(f"更新 {update}: avg_return={avg_r:.2f}, "
                      f"success={win_succ:.2%}, collision={win_coll:.2%}, timeout={win_to:.2%}, "
                      f"cum_success={total_successes/max(total_episodes,1):.2%}, "
                      f"avg_ep_len={float(np.mean(window_episode_lengths)):.0f}, "
                      f"n_trans={n_transitions_per_agent} (total={n_transitions}), "
                      f"lr_a={cur_lr_a:.2e}")
                print(f"   reward分量(均值/步): progress={comp_progress:.3f}, "
                      f"approach={comp_approach:.3f}, speed={comp_speed:.3f}, alive={comp_alive:.3f}, "
                      f"safety+ttc={comp_ttc:.3f}, lane={comp_lane:.3f}, collision={comp_collision:.3f}, "
                      f"step={comp_step:.3f}, success+bonus={comp_success:.3f}")

                if window_reward_components:
                    for comp_name, comp_values in window_reward_components.items():
                        if comp_values:
                            tb_writer.add_scalar(comp_name,
                                float(np.mean(comp_values)), update)

                if not skip_update and len(critic_states) > 0:
                    tb_writer.add_scalar('ppo/critic_loss',
                        float(np.mean(ppo_metrics["critic_loss"])), update)
                    tb_writer.add_scalar('ppo/critic_grad_norm',
                        float(np.mean(ppo_metrics["grad_norm_critic"])), update)
                    tb_writer.add_scalar('ppo/explained_variance',
                        ppo_metrics["explained_variance"], update)

                    for aid in agent_ids:
                        if aid in advantage_stats:
                            stats = advantage_stats[aid]
                            tb_writer.add_scalar(
                                f'ppo/adv_raw_mean_agent{aid}',
                                stats["raw_mean"], update,
                            )
                            tb_writer.add_scalar(
                                f'ppo/adv_raw_std_agent{aid}',
                                stats["raw_std"], update,
                            )
                            tb_writer.add_scalar(
                                f'ppo/adv_normalized_mean_agent{aid}',
                                stats["normalized_mean"], update,
                            )
                            tb_writer.add_scalar(
                                f'ppo/adv_normalized_std_agent{aid}',
                                stats["normalized_std"], update,
                            )
                            tb_writer.add_scalar(
                                f'ppo/adv_normalization_applied_agent{aid}',
                                stats["applied"], update,
                            )
                        if aid in preupdate_ratio_errors:
                            tb_writer.add_scalar(
                                f'ppo/preupdate_max_abs_ratio_error_agent{aid}',
                                preupdate_ratio_errors[aid], update,
                            )
                        if ppo_metrics[f"actor_loss_{aid}"]:
                            tb_writer.add_scalar(f'ppo/actor_loss_agent{aid}',
                                float(np.mean(ppo_metrics[f"actor_loss_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/entropy_agent{aid}',
                                float(np.mean(ppo_metrics[f"entropy_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/approx_kl_agent{aid}',
                                float(np.mean(ppo_metrics[f"approx_kl_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/clip_frac_agent{aid}',
                                float(np.mean(ppo_metrics[f"clip_frac_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/grad_norm_actor_agent{aid}',
                                float(np.mean(ppo_metrics[f"grad_norm_actor_{aid}"])), update)
                            # Phase A3: 该 agent 实际跑了多少个 PPO epoch（KL 早停诊断）
                            tb_writer.add_scalar(f'ppo/stop_epoch_agent{aid}',
                                agent_stop_epoch.get(aid, args.ppo_epochs), update)

                # 场景指标
                for scenario_name in curriculum.scenario_success_history:
                    history = curriculum.scenario_success_history[scenario_name]
                    if len(history) >= 3:
                        win_rate = float(np.mean(history))
                        safe_name = scenario_name.replace('.json', '').replace(' ', '_')[:40]
                        tb_writer.add_scalar(f'scenarios/{safe_name}/success_rate_window',
                            win_rate, update)
                        tb_writer.add_scalar(f'scenarios/{safe_name}/sample_count',
                            len(history), update)

            # 一个 update 只有在 rollout、PPO、统计与日志全部结束后才提交。
            # latest 每次覆盖；periodic 不可覆盖。训练过程不再选择 best。
            tb_writer.flush()
            completed_update = update
            _save_safe_checkpoint(
                include_periodic=(update % args.save_interval == 0)
            )

    except KeyboardInterrupt:
        print("训练中断：丢弃当前未完成 update，保留最近完整边界")
        try:
            _save_interrupted_checkpoint()
        except Exception as save_error:
            print(f"  [警告] interrupted checkpoint 保存失败: {save_error}")
    except Exception:
        try:
            _save_interrupted_checkpoint()
        except Exception as save_error:
            print(f"  [警告] 异常退出时保存 interrupted 失败: {save_error}")
        raise
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()
        tb_writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="鲁棒 MAPPO 训练脚本 v2（改进场景采样 + GAE修复 + 软过渡）")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
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
                        help="折扣因子（远距收敛修复：0.99→0.997，有效视野 ~100→~330 步，"
                             "让终点 +100 能反传到远 spawn 车的早期动作）")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.15,
                        help="PPO clip 系数（v2 修复：0.2→0.15，小批量训练更稳）")
    parser.add_argument("--ent-coef", type=float, default=0.03,
                        help="熵损失系数（v2 修复：0.01→0.03），防止 log_std 早期塌缩")
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
    parser.add_argument("--scenario-sampling-mode", type=str, default="weighted",
                        choices=["random", "stratified", "weighted"],
                        help="场景采样模式: random=随机, stratified=分层轮询, weighted=分层+失败率加权")

    # 课程学习
    parser.add_argument("--warmup-updates", type=int, default=10, help="基础场景预热次数")
    parser.add_argument("--full-scenario-updates", type=int, default=40, help="达到全场景的更新次数")
    parser.add_argument("--curriculum-transition-window", type=int, default=10,
                        help="阶段过渡窗口大小（v2 新增）")
    parser.add_argument("--curriculum-mode", type=str, default="classic",
                        choices=["classic", "distance"],
                        help="课程模式: classic=basic/left|right/all 旧逻辑(默认,已回退距离课程); "
                             "distance=按 dist_<N>m 距离分层(实验性,易诱发短距离原地不动)")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--render", action="store_true")
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
        "--allow-resume-code-drift",
        action="store_true",
        help="显式允许 Git SHA/关键源码指纹不同的续训，并永久写入元数据",
    )

    parser.add_argument("--save-dir", type=str, default="./logs/mappo_robust_v2")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--reward-version", type=str, default="v2",
                        choices=["v1", "v2"], help="奖励函数版本 (v1=原始, v2=优化)")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args._explicit_cli_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    train(args)
