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


VALID_EPISODE_END_REASONS = frozenset({
    "running",
    "cooperative_success",
    "all_agents_reached",
    "cooperative_collision",
    "adversary_collision",
    "max_steps",
    "all_agents_terminated",
})


def set_seed(seed: int):
    # 修复 #6：完整播种
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # 初始化环境
    env = CarlaMultiAgentEnv(
        host=args.host,
        port=args.port,
        town=args.town,
        num_agents=2,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep,
        max_steps=args.max_steps,
        spawn_vehicles=True,
        reward_version=args.reward_version,
        disable_tensorboard=True,
        seed=args.seed,
    )

    # 初始化课程学习
    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
    curriculum = CurriculumLearning(
        scenario_dir=scenario_dir,
        warmup_updates=args.warmup_updates,
        full_scenario_updates=args.full_scenario_updates,
        transition_window=args.curriculum_transition_window,
        mode=args.curriculum_mode,
    )

    agent_ids = list(range(env.num_agents))
    state_dim = env.get_global_state().shape[0]
    print(f"全局状态维度: {state_dim}")

    # 初始化模型
    actors: Dict[int, Actor] = {
        aid: Actor(
            obs_dim=19, act_dim=3, hidden=args.hidden_dim,
            init_log_std=args.init_log_std,
        ).to(device) for aid in agent_ids
    }
    actor_optims: Dict[int, optim.Optimizer] = {
        aid: optim.Adam(actors[aid].parameters(), lr=args.lr_actor) for aid in agent_ids
    }

    # 修复 #1：agent-id 条件化集中式 Critic
    critic = AgentConditionedCritic(state_dim=state_dim, num_agents=env.num_agents,
                                    hidden=args.hidden_dim).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)

    # 初始 lr 缓存（lr 线性衰减用）
    init_lr_actor = args.lr_actor
    init_lr_critic = args.lr_critic

    def _apply_lr_decay(cur_update: int):
        if not args.lr_decay:
            return init_lr_actor, init_lr_critic
        # 1.0 → 0.05（保留 5% 下限，避免后期完全停学）
        frac = max(0.05, 1.0 - (cur_update - 1) / max(args.updates - 1, 1))
        new_lr_a = init_lr_actor * frac
        new_lr_c = init_lr_critic * frac
        for aid in agent_ids:
            for g in actor_optims[aid].param_groups:
                g['lr'] = new_lr_a
        for g in critic_optim.param_groups:
            g['lr'] = new_lr_c
        return new_lr_a, new_lr_c

    # 加载预训练模型
    if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
        print(f"加载预训练模型: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
        for aid in agent_ids:
            if aid in checkpoint.get('actors', {}):
                actors[aid].load_state_dict(checkpoint['actors'][aid])
        # 旧 checkpoint 的 critic 是非条件化结构，不兼容 → 跳过
        if 'critic' in checkpoint:
            try:
                critic.load_state_dict(checkpoint['critic'])
            except Exception as e:
                print(f"[警告] critic 结构已变更（agent-id 条件化），跳过加载旧 critic: {e}")

    visualizer = MAPPOVisualizer(width=1200, height=800, fps=30) if args.render else None

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard：每次运行单独时间戳子目录，避免多次 run 的 event 文件混在一起
    # （之前同一 save_dir 反复跑会让 TB 把多次 run 的曲线合并成"花线"）
    import datetime as _dt
    run_tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
    tb_log_dir = os.path.join(save_dir, 'tb_logs', run_tag)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_dir)
    print(f"TensorBoard 日志: {tb_log_dir}")

    # 训练统计
    total_episodes = 0
    total_successes = 0          # 严格成功（all reached_target）
    total_collisions = 0
    total_timeouts = 0
    best_success_rate = -1.0     # 用于 best.pt 跟踪
    start_update = 1             # resume 时会更新

    # ─── 恢复训练（--resume）─── 必须放在 optim / curriculum 初始化之后
    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            cands = []
            if os.path.isdir(save_dir):
                for f in os.listdir(save_dir):
                    if f.startswith("mappo_robust_v2_update_") and f.endswith(".pt"):
                        try:
                            n = int(f.replace("mappo_robust_v2_update_", "").replace(".pt", ""))
                            cands.append((n, f))
                        except ValueError:
                            continue
            if not cands:
                print(f"⚠️ --resume latest: {save_dir} 下没找到 mappo_robust_v2_update_*.pt，从零开始")
                resume_path = ""
            else:
                cands.sort()
                resume_path = os.path.join(save_dir, cands[-1][1])

        if resume_path and os.path.exists(resume_path):
            print(f"恢复训练 from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            for aid in agent_ids:
                if aid in ckpt.get('actors', {}):
                    actors[aid].load_state_dict(ckpt['actors'][aid])
            # critic 结构升级为 agent-id 条件化；旧 checkpoint 不兼容时跳过 critic/critic_optim
            critic_loaded = False
            if 'critic' in ckpt:
                try:
                    critic.load_state_dict(ckpt['critic'])
                    critic_loaded = True
                except Exception as e:
                    print(f"[警告] critic 结构已变更，跳过加载旧 critic: {e}")
            if 'actor_optims' in ckpt:
                for aid in agent_ids:
                    if aid in ckpt['actor_optims']:
                        actor_optims[aid].load_state_dict(ckpt['actor_optims'][aid])
            if 'critic_optim' in ckpt and critic_loaded:
                critic_optim.load_state_dict(ckpt['critic_optim'])
            start_update = int(ckpt.get('update', 0)) + 1
            total_episodes = int(ckpt.get('total_episodes', 0))
            total_successes = int(ckpt.get('total_successes', 0))
            total_collisions = int(ckpt.get('total_collisions', 0))
            total_timeouts = int(ckpt.get('total_timeouts', 0))
            best_success_rate = float(ckpt.get('best_success_rate', -1.0))
            if 'curriculum_success_history' in ckpt:
                for k, v in ckpt['curriculum_success_history'].items():
                    curriculum.scenario_success_history[k] = list(v)
            print(f"  → 续训 update={start_update}, total_ep={total_episodes}, "
                  f"cum_success={total_successes/max(total_episodes,1):.2%}, "
                  f"best_succ={best_success_rate:.2%}")
            if start_update > args.updates:
                print(f"⚠️ 已恢复到 update={start_update-1}，但 --updates={args.updates} 不够，"
                      f"请加大 --updates 或者直接跳过本次训练")

    try:
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
                per_agent_tensors = {}
                advantage_stats: Dict[int, Dict[str, float]] = {}
                preupdate_ratio_errors: Dict[int, float] = {}
                critic_states = []
                critic_returns = []
                critic_agent_ids = []   # 修复 #1：每条样本对应的 agent id

                for aid in agent_ids:
                    data = buffer.get_agent_data(aid)
                    if len(data) == 0:
                        continue

                    obs_arr = np.array([t.obs for t in data], dtype=np.float32)
                    state_arr = np.array([t.global_state for t in data], dtype=np.float32)
                    act_arr = np.array([t.action_z for t in data], dtype=np.float32)
                    old_logp_arr = np.array([t.log_prob for t in data], dtype=np.float32)
                    reward_arr = np.array([t.reward for t in data], dtype=np.float32)
                    done_arr = np.array([t.done for t in data], dtype=np.float32)
                    terminated_arr = np.array([t.terminated for t in data], dtype=np.float32)
                    value_arr = np.array([t.value for t in data], dtype=np.float32)

                    # P0-00D：按 episode 独立计算原始 GAE；同一 agent 的全部
                    # episode 拼接后，只在整个 rollout 上统一标准化一次。
                    boundaries = buffer.episode_boundaries.get(aid, [])
                    # 修复 #1：bootstrap 值也要按 agent 条件化
                    bootstrap_states = buffer.episode_bootstrap_states.get(aid, [])
                    if len(bootstrap_states) != len(boundaries):
                        raise RuntimeError(
                            f"Agent {aid} bootstrap/boundary count mismatch: "
                            f"{len(bootstrap_states)} != {len(boundaries)}"
                        )
                    if not boundaries or boundaries[-1][1] != len(data):
                        raise RuntimeError(
                            f"Agent {aid} rollout has unclosed transitions: "
                            f"last_boundary={boundaries[-1][1] if boundaries else 0}, "
                            f"data={len(data)}"
                        )
                    bootstrap_values = []
                    with torch.no_grad():
                        for bs in bootstrap_states:
                            if bs is not None:
                                bs_t = torch.as_tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)
                                bootstrap_values.append(float(critic(bs_t, aid).item()))
                            else:
                                bootstrap_values.append(0.0)
                    raw_adv_arr, ret_arr = compute_raw_gae_per_episode(
                        rewards=reward_arr,
                        values=value_arr,
                        dones=done_arr,
                        terminateds=terminated_arr,
                        boundaries=boundaries,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                        bootstrap_values=bootstrap_values,
                        context=f"robust_v2 update={update} agent={aid}",
                    )
                    adv_arr, agent_advantage_stats = normalize_advantages_over_rollout(
                        raw_adv_arr,
                        context=(
                            f"robust_v2 advantage normalization "
                            f"update={update} agent={aid}"
                        ),
                    )
                    advantage_stats[aid] = agent_advantage_stats

                    array_lengths = {
                        "obs": len(obs_arr),
                        "state": len(state_arr),
                        "action_z": len(act_arr),
                        "old_logp": len(old_logp_arr),
                        "raw_advantage": len(raw_adv_arr),
                        "normalized_advantage": len(adv_arr),
                        "return": len(ret_arr),
                    }
                    if len(set(array_lengths.values())) != 1:
                        raise RuntimeError(
                            f"robust_v2 update={update} agent={aid}: "
                            f"PPO array length mismatch: {array_lengths}"
                        )
                    expected_returns = raw_adv_arr + value_arr
                    if not np.allclose(
                        ret_arr, expected_returns, rtol=1e-6, atol=1e-6
                    ):
                        raise RuntimeError(
                            f"robust_v2 update={update} agent={aid}: "
                            "critic return is not raw_advantage + old_value"
                        )

                    per_agent_tensors[aid] = {
                        "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
                        "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
                        "old_logp": torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device),
                        "adv": torch.as_tensor(adv_arr, dtype=torch.float32, device=device),
                        "raw_adv": torch.as_tensor(
                            raw_adv_arr, dtype=torch.float32, device=device
                        ),
                        "state": torch.as_tensor(state_arr, dtype=torch.float32, device=device),
                        "ret": torch.as_tensor(ret_arr, dtype=torch.float32, device=device),
                    }

                    critic_states.append(state_arr)
                    critic_returns.append(ret_arr)
                    critic_agent_ids.append(np.full(len(ret_arr), aid, dtype=np.int64))

                # 初始化 PPO 指标
                ppo_metrics = {
                    "critic_loss": [],
                    "explained_variance": 0.0,
                }
                for aid in agent_ids:
                    ppo_metrics.update({
                        f"actor_loss_{aid}": [],
                        f"entropy_{aid}": [],
                        f"approx_kl_{aid}": [],
                        f"clip_frac_{aid}": [],
                        f"grad_norm_actor_{aid}": [],
                    })
                ppo_metrics["grad_norm_critic"] = []

                # P0-00D：必须在任何 Actor optimizer.step() 之前验证采样策略
                # 与当前策略完全对齐。Critic 更新不影响 Actor，但仍在 epoch 前统一检查。
                for aid, tensors in per_agent_tensors.items():
                    preupdate_ratio_errors[aid] = validate_initial_policy_ratio(
                        actor=actors[aid],
                        observations=tensors["obs"],
                        action_z=tensors["act"],
                        old_log_probs=tensors["old_logp"],
                        tolerance=1e-6,
                        context=f"robust_v2 pre-update ratio update={update} agent={aid}",
                    )

                if len(critic_states) > 0:
                    critic_state_t = torch.as_tensor(
                        np.concatenate(critic_states, axis=0), dtype=torch.float32, device=device
                    )
                    critic_ret_t = torch.as_tensor(
                        np.concatenate(critic_returns, axis=0), dtype=torch.float32, device=device
                    )
                    critic_aid_t = torch.as_tensor(
                        np.concatenate(critic_agent_ids, axis=0), dtype=torch.long, device=device
                    )

                    # Phase A3: 每个 agent 在 epoch 间记录是否已 KL 早停
                    agent_stopped: Dict[int, bool] = {aid: False for aid in agent_ids}
                    agent_stop_epoch: Dict[int, int] = {aid: args.ppo_epochs for aid in agent_ids}

                    for epoch in range(args.ppo_epochs):
                        # --- Critic 更新 ---
                        c_n = critic_state_t.shape[0]
                        c_idx = np.arange(c_n)
                        np.random.shuffle(c_idx)
                        epoch_critic_losses = []
                        for start in range(0, c_n, args.minibatch_size):
                            mb = c_idx[start: start + args.minibatch_size]
                            value_pred = critic(critic_state_t[mb], critic_aid_t[mb])
                            critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()
                            critic_context = (
                                f"robust_v2 critic update={update} "
                                f"epoch={epoch} minibatch_start={start}"
                            )
                            ensure_finite_tensor(
                                value_pred, name="value_pred", context=critic_context
                            )
                            ensure_finite_tensor(
                                critic_ret_t[mb],
                                name="critic_return",
                                context=critic_context,
                            )
                            ensure_finite_tensor(
                                critic_loss, name="critic_loss", context=critic_context
                            )

                            critic_optim.zero_grad()
                            critic_loss.backward()
                            ensure_finite_module_gradients(
                                critic, context=critic_context
                            )
                            critic_grad_norm = nn.utils.clip_grad_norm_(
                                critic.parameters(),
                                args.max_grad_norm,
                                error_if_nonfinite=True,
                            )
                            ensure_finite_tensor(
                                critic_grad_norm,
                                name="critic_grad_norm",
                                context=critic_context,
                            )
                            critic_optim.step()
                            ensure_finite_module_parameters(
                                critic, context=critic_context
                            )

                            epoch_critic_losses.append(float(critic_loss.item()))
                            ppo_metrics["grad_norm_critic"].append(float(critic_grad_norm))

                        ppo_metrics["critic_loss"].append(float(np.mean(epoch_critic_losses)))

                        # --- Actor 更新 ---
                        for aid in agent_ids:
                            if aid not in per_agent_tensors:
                                continue
                            if agent_stopped[aid]:
                                continue  # Phase A3: 该 agent 已 KL 早停，跳过后续 epoch

                            agent_obs_t = per_agent_tensors[aid]["obs"]
                            agent_act_t = per_agent_tensors[aid]["act"]
                            agent_old_logp_t = per_agent_tensors[aid]["old_logp"]
                            agent_adv_t = per_agent_tensors[aid]["adv"]

                            a_n = agent_obs_t.shape[0]
                            a_idx = np.arange(a_n)
                            np.random.shuffle(a_idx)

                            epoch_actor_losses = []
                            epoch_entropies = []
                            epoch_approx_kls = []
                            epoch_clip_fracs = []
                            epoch_actor_grad_norms = []

                            for start in range(0, a_n, args.minibatch_size):
                                mb = a_idx[start: start + args.minibatch_size]
                                actor_context = (
                                    f"robust_v2 actor update={update} agent={aid} "
                                    f"epoch={epoch} minibatch_start={start}"
                                )
                                # 修复 #4：用存储的 pre-tanh z 重算 log_prob
                                new_logp, entropy = actors[aid].evaluate_actions_z(agent_obs_t[mb], agent_act_t[mb])

                                logratio = new_logp - agent_old_logp_t[mb]
                                ratio = torch.exp(logratio)
                                ensure_finite_tensor(
                                    new_logp, name="new_logp", context=actor_context
                                )
                                ensure_finite_tensor(
                                    entropy, name="entropy", context=actor_context
                                )
                                ensure_finite_tensor(
                                    agent_adv_t[mb],
                                    name="normalized_advantage",
                                    context=actor_context,
                                )
                                approx_kl, clip_fraction = compute_ppo_diagnostics(
                                    logratio,
                                    ratio,
                                    args.clip_eps,
                                    context=actor_context,
                                )
                                surr1 = ratio * agent_adv_t[mb]
                                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * agent_adv_t[mb]
                                actor_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy.mean()
                                ensure_finite_tensor(
                                    actor_loss, name="actor_loss", context=actor_context
                                )

                                actor_optims[aid].zero_grad()
                                actor_loss.backward()
                                ensure_finite_module_gradients(
                                    actors[aid], context=actor_context
                                )
                                actor_grad_norm = nn.utils.clip_grad_norm_(
                                    actors[aid].parameters(),
                                    args.max_grad_norm,
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
                                epoch_entropies.append(float(entropy.mean().item()))
                                epoch_approx_kls.append(approx_kl)
                                epoch_clip_fracs.append(clip_fraction)
                                epoch_actor_grad_norms.append(float(actor_grad_norm))

                            ppo_metrics[f"actor_loss_{aid}"].append(float(np.mean(epoch_actor_losses)))
                            ppo_metrics[f"entropy_{aid}"].append(float(np.mean(epoch_entropies)))
                            mean_epoch_kl = float(np.mean(epoch_approx_kls))
                            ppo_metrics[f"approx_kl_{aid}"].append(mean_epoch_kl)

                            # Phase A3: KL 早停 — 单 agent 单 epoch KL 超阈值就跳过后续 epoch
                            if args.target_kl > 0 and mean_epoch_kl > 1.5 * args.target_kl:
                                agent_stopped[aid] = True
                                agent_stop_epoch[aid] = epoch + 1
                            ppo_metrics[f"clip_frac_{aid}"].append(float(np.mean(epoch_clip_fracs)))
                            ppo_metrics[f"grad_norm_actor_{aid}"].append(
                                float(np.mean(epoch_actor_grad_norms))
                            )

                    # explained variance
                    with torch.no_grad():
                        all_value_preds = critic(critic_state_t, critic_aid_t)
                        ensure_finite_tensor(
                            all_value_preds,
                            name="explained_variance_value_predictions",
                            context=f"robust_v2 update={update}",
                        )
                        var_residual = torch.var(
                            critic_ret_t - all_value_preds, unbiased=False
                        )
                        var_total = torch.var(critic_ret_t, unbiased=False)
                        explained_var_t = 1.0 - var_residual / (var_total + 1e-8)
                        ensure_finite_tensor(
                            explained_var_t,
                            name="explained_variance",
                            context=f"robust_v2 update={update}",
                        )
                        explained_var = float(explained_var_t.item())
                        ppo_metrics["explained_variance"] = explained_var

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

            # 公用：组装可 resume 的完整 state
            def _build_ckpt_dict():
                return {
                    "update": update,
                    "actors": {aid: actors[aid].state_dict() for aid in agent_ids},
                    "critic": critic.state_dict(),
                    "actor_optims": {aid: actor_optims[aid].state_dict() for aid in agent_ids},
                    "critic_optim": critic_optim.state_dict(),
                    "args": vars(args),
                    # resume 需要的训练统计
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                    "total_collisions": total_collisions,
                    "total_timeouts": total_timeouts,
                    "best_success_rate": best_success_rate,
                    "curriculum_success_history": {
                        k: list(v) for k, v in curriculum.scenario_success_history.items()
                    },
                }

            # 保存 checkpoint
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"mappo_robust_v2_update_{update}.pt")
                torch.save(_build_ckpt_dict(), ckpt_path)
                print(f"模型保存: {ckpt_path}")

            # Phase A4: best.pt 跟踪（按窗口严格成功率取 max）
            if len(window_successes) > 0:
                cur_win_success = float(np.mean(window_successes))
                if cur_win_success > best_success_rate:
                    best_success_rate = cur_win_success
                    best_ckpt_path = os.path.join(save_dir, "best.pt")
                    torch.save(_build_ckpt_dict(), best_ckpt_path)
                    print(f"  [best] 窗口成功率 {best_success_rate:.2%}，保存 {best_ckpt_path}")

    except KeyboardInterrupt:
        print("训练中断")
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
                        help="预训练模型路径（仅加载 actor/critic 权重，从 update 1 开始）")
    parser.add_argument("--resume", type=str, default="",
                        help="恢复训练的 checkpoint 路径（加载权重 + optimizer + update 计数 + 训练统计 + 课程历史）。"
                             "传 'latest' 自动选 save_dir 下 update 编号最大的 ckpt。"
                             "与 --pretrained-checkpoint 同时给时优先用 --resume。")

    parser.add_argument("--save-dir", type=str, default="./logs/mappo_robust_v2")
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--reward-version", type=str, default="v2",
                        choices=["v1", "v2"], help="奖励函数版本 (v1=原始, v2=优化)")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
