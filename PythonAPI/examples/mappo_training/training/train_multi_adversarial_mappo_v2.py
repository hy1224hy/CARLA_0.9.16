"""
多对抗智能体 MAPPO 训练脚本 v2
================================
以 train_no_rllib_mappo_robust_v2.py 为模板重写，修复
train_multi_adversarial_mappo.py（v1）的若干缺陷：

- 合作团队 (Agents 0..N-1): 加载预训练 MAPPO 策略并冻结，deterministic 推理
- 对抗团队 (Agents N..N+M-1): MAPPO (CTDE) 训练
    · 共享一个 CentralizedCritic（输入全局状态 get_global_state，看到全员）
    · 每个对抗 agent 各自一个 Actor
    · per-agent advantage：各 agent 用自己的 reward 序列算 GAE（标准 MAPPO 信用分配）

相对 v1 的修复：
1. Critic 输入 get_global_state()（全员 obs），不再只看对抗 obs（修偏置）
2. per-agent advantage 取代 joint advantage（恢复团队内分工信号）
3. 零和项参数化 --zero-sum-coef，取代写死的 ×0.01；课程 ramp 作用其上
4. per-episode GAE + 截断 bootstrap + 逐 episode 优势标准化（移植 robust v2）
5. 场景分层 / 加权采样 + 课程软过渡（移植 robust v2，修训练/测试分布不一致）
6. 完整 PPO 诊断（KL 早停 / clip_frac / explained_variance / grad_norm）
7. 训练期记录合作团队成功率 / 碰撞率（coop 视角的攻击有效性监控）

注意：必须使用 reward_version='v2'，对抗 agent 才会获得 cut_in/blocking 奖励
（v1 奖励路径会把对抗奖励清零，见 carla_multi_agent_env.py:1200-1214）。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import datetime as _dt
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, CentralizedCritic

try:
    from visualization.mappo_visualizer import MAPPOVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据结构和 Buffer（移植自 robust v2）
# ============================================================

@dataclass
class Transition:
    obs: np.ndarray
    global_state: np.ndarray
    action_tanh: np.ndarray
    log_prob: float
    reward: float
    done: float
    value: float
    scenario_name: str


class RolloutBuffer:
    """增强版 RolloutBuffer：支持 episode 边界追踪（仅对抗 agent）"""

    def __init__(self, agent_ids: List[int]):
        self.data: Dict[int, List[Transition]] = {aid: [] for aid in agent_ids}
        self.episode_boundaries: Dict[int, List[Tuple[int, int]]] = {aid: [] for aid in agent_ids}
        self.episode_bootstrap_states: Dict[int, List[Optional[np.ndarray]]] = {
            aid: [] for aid in agent_ids
        }

    def add(self, agent_id: int, transition: Transition):
        self.data[agent_id].append(transition)

    def mark_episode_end(self, agent_id: int):
        current_len = len(self.data[agent_id])
        boundaries = self.episode_boundaries[agent_id]
        start = 0 if not boundaries else boundaries[-1][1]
        if current_len > start:
            boundaries.append((start, current_len))

    def add_episode_bootstrap(self, agent_id: int, state: Optional[np.ndarray]):
        self.episode_bootstrap_states[agent_id].append(state)

    def get_agent_data(self, agent_id: int) -> List[Transition]:
        return self.data[agent_id]

    def total_size(self) -> int:
        return sum(len(v) for v in self.data.values())


# ============================================================
# 工具函数（移植自 robust v2）
# ============================================================

def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    """Convert tanh action [-1,1] to env action format [throttle, steer, brake]."""
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5
    if throttle_raw >= brake_raw:
        throttle, brake = throttle_raw, 0.0
    else:
        throttle, brake = 0.0, brake_raw
    return np.array([throttle, steer, brake], dtype=np.float32)


def _scenario_agents(config: Dict) -> Dict[str, Dict]:
    """从场景配置里提取 agent 起点，兼容嵌套 "agents" 与扁平（顶层 agentX）两种格式。"""
    src = config.get("agents", config)
    return {k: dict(v) for k, v in src.items()
            if isinstance(k, str) and k.startswith("agent") and isinstance(v, dict)}


def _load_config_file(path_or_name: str, scenario_dir: str) -> Optional[Dict]:
    """加载场景/位置配置：先按绝对/相对路径找，找不到再当作 scenario_dir 下的文件名。"""
    import json
    cand = path_or_name
    if not os.path.exists(cand):
        cand = os.path.join(scenario_dir, path_or_name)
    if not os.path.exists(cand):
        return None
    try:
        with open(cand, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def make_adversarial_positions(scenario_config, cooperative_ids, adversarial_ids, margin_ratio=0.0):
    """把蓝色对抗车安置在红色合作车之间，排成 红-蓝…蓝-红（确定性，可复现）。

    - 合作车(红)保留场景配置中的原始位置不动，作为车队首尾；
    - 对抗车(蓝)全部落在「最后一辆红车 ~ 最前一辆红车」的区间内，沿车道方向
      均匀分布（n 辆蓝车 → 区间 n+1 等分点）。因此没有任何蓝车跑到最前红车
      之前，蓝车永远不会比红车更靠近终点；
    - margin_ratio：蓝车分布区间相对红车两端再内缩的比例（0=纯均匀等分，
      >0 让蓝车更聚向中间、离红车更远）。

    返回完整的 agents dict（原红车 + 新蓝车），写回 scenario_config["agents"] 即可。
    场景配置只定义红车时，蓝车位置由本函数补全，因此无需改动任何场景文件。
    """
    agents = _scenario_agents(scenario_config)
    coop = [(cid, agents[f"agent{cid}"]) for cid in cooperative_ids if f"agent{cid}" in agents]
    if not coop:
        # 没有红车基准就不动（退回环境默认 waypoint 分配）
        return agents

    # 用第一辆红车的 yaw 推车道前进方向，按前进方向投影排序红车（从后到前）
    yaw = float(coop[0][1].get("yaw", 90.0))
    fx, fy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    prog = lambda p: float(p["x"]) * fx + float(p["y"]) * fy
    coop.sort(key=lambda t: prog(t[1]))

    back = coop[0][1]                       # 最后一辆红车（车队尾）
    s_back, s_front = prog(coop[0][1]), prog(coop[-1][1])
    span = s_front - s_back
    if span < 1e-3:                         # 红车重合/仅一辆 → 在其前方一段距离内排布
        span = 14.0
        s_front = s_back + span
    lo = s_back + span * margin_ratio
    hi = s_front - span * margin_ratio
    if hi <= lo:
        lo, hi = s_back, s_front

    new_agents = {f"agent{cid}": p for cid, p in coop}
    z = float(back.get("z", 0.5))
    n = len(adversarial_ids)
    for i, aid in enumerate(adversarial_ids):
        frac = (i + 1) / (n + 1)            # 区间 n+1 等分点，红车守两端
        off = (lo - s_back) + (hi - lo) * frac   # 相对车队尾红车的前向偏移
        new_agents[f"agent{aid}"] = {
            "x": round(float(back["x"]) + fx * off, 3),
            "y": round(float(back["y"]) + fy * off, 3),
            "z": z,
            "yaw": yaw,
        }
    return new_agents


def compute_gae(rewards, values, dones, gamma, gae_lambda, bootstrap_value=0.0):
    """在单个 episode 内计算 GAE（不应跨 episode 调用）。"""
    advantages = np.zeros_like(rewards)
    gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def compute_gae_per_episode(rewards, values, dones, boundaries, gamma, gae_lambda,
                            bootstrap_values=None):
    """按 episode 边界独立计算 GAE + 逐 episode 优势标准化 + 截断 bootstrap。"""
    if bootstrap_values is None:
        bootstrap_values = [0.0] * len(boundaries)
    all_advantages, all_returns = [], []
    for i, (start, end) in enumerate(boundaries):
        if end <= start:
            continue
        ep_rewards = rewards[start:end]
        ep_values = values[start:end]
        ep_dones = dones[start:end]
        bv = bootstrap_values[i] if i < len(bootstrap_values) else 0.0
        adv, ret = compute_gae(ep_rewards, ep_values, ep_dones, gamma, gae_lambda, bootstrap_value=bv)
        ep_std = adv.std()
        if ep_std > 1e-6:
            adv = (adv - adv.mean()) / (ep_std + 1e-8)
        all_advantages.append(adv)
        all_returns.append(ret)
    if len(all_advantages) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(all_advantages), np.concatenate(all_returns)


def sample_scenarios_stratified(available_scenarios, n_episodes, success_rates=None,
                                mode="weighted", external_weights=None):
    """从可用场景中采样 n_episodes 个场景（分层 / 加权）。"""
    if mode == "random" or len(available_scenarios) == 0:
        return [np.random.choice(available_scenarios) for _ in range(n_episodes)]

    n_scenarios = len(available_scenarios)
    selected = []
    if mode == "stratified":
        rounds = n_episodes // n_scenarios
        remainder = n_episodes % n_scenarios
        for _ in range(rounds):
            selected.extend(available_scenarios)
        if remainder > 0:
            selected.extend(np.random.choice(available_scenarios, size=remainder, replace=False).tolist())
    elif mode == "weighted":
        if n_episodes >= n_scenarios:
            selected.extend(available_scenarios)
            remaining = n_episodes - n_scenarios
        else:
            selected.extend(np.random.choice(available_scenarios, size=n_episodes, replace=False).tolist())
            remaining = 0
        if remaining > 0:
            weights = []
            for s in available_scenarios:
                w = 1.0
                if success_rates is not None:
                    rate = success_rates.get(s, 0.5)
                    w *= (1.0 - rate + 0.1)
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
# 课程学习（场景分层 + 软过渡，移植自 robust v2）
# ============================================================

class CurriculumLearning:
    """场景课程学习管理器（支持软过渡）。

    注意：这里的 success 记录的是 **对抗团队的成功**（即合作团队未全部到达），
    weighted 采样会对 "对抗失败" 的场景上采样 —— 即合作团队还顶得住的场景，
    正好是对抗策略需要加练的地方。
    """

    def __init__(self, scenario_dir, warmup_updates=10, full_scenario_updates=40,
                 transition_window=10):
        self.scenario_dir = scenario_dir
        self.warmup_updates = warmup_updates
        self.full_scenario_updates = full_scenario_updates
        self.transition_window = transition_window

        all_scenario_files = []
        if os.path.exists(scenario_dir):
            all_scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])

        self.curriculum_stages = self._build_default_stages(all_scenario_files)

        self.scenario_success_history: Dict[str, List[float]] = {f: [] for f in all_scenario_files}
        print(f"场景课程初始化，共 {len(self.curriculum_stages)} 个阶段，{len(all_scenario_files)} 个场景")

    @staticmethod
    def _build_default_stages(all_scenario_files: List[str]) -> List[List[str]]:
        """构建默认课程阶段（近→远）。存在 dist_<N>m 场景时按距离升序分 3 段，
        否则回退 basic / left|right / all（与 robust v2 保持一致）。"""
        def _dist_of(fname):
            m = re.search(r'dist_(\d+)m', fname)
            return int(m.group(1)) if m else None

        dist_files = [f for f in all_scenario_files if _dist_of(f) is not None]
        if dist_files:
            dist_sorted = sorted(dist_files, key=_dist_of)
            other = [f for f in all_scenario_files if _dist_of(f) is None]
            n = len(dist_sorted)
            return [dist_sorted[:max(1, n // 3)],
                    dist_sorted[:max(1, (2 * n) // 3)],
                    dist_sorted + other]
        stages = []
        basic_scenarios = [f for f in all_scenario_files if 'basic' in f]
        if not basic_scenarios and all_scenario_files:
            basic_scenarios = [all_scenario_files[0]]
        stages.append(basic_scenarios)
        stage2 = basic_scenarios + [f for f in all_scenario_files if 'left' in f or 'right' in f]
        stages.append(list(set(stage2)))
        stages.append(all_scenario_files)
        return stages

    def _get_stage_idx(self, current_update: int) -> int:
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
        stage_idx = self._get_stage_idx(current_update)
        if stage_idx == 0:
            return 1.0
        n_stages = len(self.curriculum_stages)
        if n_stages <= 1:
            return 1.0
        if stage_idx == n_stages - 1:
            stage_start = float(self.full_scenario_updates)
        else:
            stage_start = self.warmup_updates + (self.full_scenario_updates - self.warmup_updates) * stage_idx / n_stages
        half_window = self.transition_window / 2.0
        if current_update < stage_start - half_window:
            return 0.0
        elif current_update > stage_start + half_window:
            return 1.0
        else:
            return (current_update - (stage_start - half_window)) / max(self.transition_window, 1e-8)

    def get_available_scenarios(self, current_update: int) -> List[str]:
        stage_idx = self._get_stage_idx(current_update)
        return self.curriculum_stages[min(stage_idx, len(self.curriculum_stages) - 1)]

    def get_scenario_sampling_weights(self, current_update: int) -> Dict[str, float]:
        alpha = self._compute_transition_alpha(current_update)
        stage_idx = self._get_stage_idx(current_update)
        current_scenarios = self.curriculum_stages[min(stage_idx, len(self.curriculum_stages) - 1)]
        weights = {s: 1.0 for s in current_scenarios}
        if alpha < 1.0 and stage_idx > 0:
            prev_scenarios = self.curriculum_stages[stage_idx - 1]
            prev_weight = 1.0 - alpha
            for s in prev_scenarios:
                weights[s] = max(weights.get(s, 0.0), prev_weight)
        return weights

    def record_scenario_result(self, scenario_name: str, adv_success: bool):
        val = 1.0 if adv_success else 0.0
        if scenario_name not in self.scenario_success_history:
            self.scenario_success_history[scenario_name] = []
        self.scenario_success_history[scenario_name].append(val)
        if len(self.scenario_success_history[scenario_name]) > 20:
            self.scenario_success_history[scenario_name] = self.scenario_success_history[scenario_name][-20:]

    def get_scenario_success_rate(self, scenario_name: str) -> float:
        hist = self.scenario_success_history.get(scenario_name, [])
        return float(np.mean(hist)) if hist else 0.0


# ============================================================
# 对抗系数课程（warmup → ramp-up，沿用 v1 语义，作用于 zero-sum-coef）
# ============================================================

def compute_zero_sum_coef(update, args):
    """计算当前 update 的零和系数（课程 ramp）。"""
    if not args.curriculum_learning:
        return args.zero_sum_coef
    if update < args.warmup_updates:
        return 0.0
    elif update < args.warmup_updates + args.ramp_up_updates:
        progress = (update - args.warmup_updates) / max(args.ramp_up_updates, 1)
        return args.zero_sum_coef * progress
    else:
        return args.zero_sum_coef


# ============================================================
# 训练主函数
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    total_agents = args.num_cooperative + args.num_adversarial
    cooperative_ids = list(range(args.num_cooperative))
    adversarial_ids = list(range(args.num_cooperative, total_agents))

    print(f"多对抗 MAPPO v2 训练配置:")
    print(f"  总智能体数: {total_agents}")
    print(f"  合作团队 (冻结): {cooperative_ids}")
    print(f"  对抗团队 (训练): {adversarial_ids}")

    if args.reward_version != "v2":
        print(f"⚠️ reward_version={args.reward_version}：v1 路径不发放 cut_in/blocking 对抗奖励，"
              f"对抗 agent 将只剩零和项可学，建议用 v2。")

    # 环境（标记对抗 agent，使其走对抗奖励通道）
    env = CarlaMultiAgentEnv(
        host=args.host, port=args.port, town=args.town,
        num_agents=total_agents,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep, max_steps=args.max_steps,
        spawn_vehicles=True,
        adversarial_agent_ids=adversarial_ids,
        reward_version=args.reward_version,
        disable_tensorboard=True,
    )

    # 场景课程
    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
    curriculum = CurriculumLearning(
        scenario_dir=scenario_dir,
        warmup_updates=args.scenario_warmup_updates,
        full_scenario_updates=args.scenario_full_updates,
        transition_window=args.curriculum_transition_window,
    )

    # 红车(合作)固定起点：从配置文件读取，覆盖采样场景里红车位置
    # （避免课程早期采样到近距场景 dist_30m，红车离终点太近）
    coop_start_positions: Optional[Dict[str, Dict]] = None
    if args.coop_start_config and args.coop_start_config.lower() not in ("none", ""):
        coop_cfg = _load_config_file(args.coop_start_config, scenario_dir)
        if coop_cfg is None:
            print(f"⚠️ coop-start-config 加载失败: {args.coop_start_config}，改用采样场景自带的红车起点")
        else:
            src = _scenario_agents(coop_cfg)
            coop_start_positions = {f"agent{cid}": src[f"agent{cid}"]
                                    for cid in cooperative_ids if f"agent{cid}" in src}
            missing = [cid for cid in cooperative_ids if f"agent{cid}" not in src]
            if missing:
                print(f"⚠️ coop-start-config {args.coop_start_config} 缺少 agent{missing} 起点，这些红车将用采样场景起点")
            print(f"红车(合作)固定起点读取自 {args.coop_start_config}: "
                  f"{ {k: (round(v['x'],2), round(v['y'],2)) for k, v in coop_start_positions.items()} }")

    # 全局状态维度（全员 obs 拼接）
    state_dim = env.get_global_state().shape[0]
    print(f"全局状态维度 (CentralizedCritic 输入): {state_dim}")

    # ── 加载预训练 checkpoint ──
    pretrained_path = args.pretrained_ckpt
    while pretrained_path.startswith('\\') or pretrained_path.startswith('/'):
        pretrained_path = pretrained_path[1:]
    pretrained_path = pretrained_path.replace('\\', os.sep).replace('/', os.sep)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {pretrained_path}")
    print(f"加载预训练 checkpoint: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=device)
    pretrained_actors_state = ckpt.get("actors", {})

    # 合作团队：冻结
    cooperative_actors: Dict[int, Actor] = {}
    for aid in cooperative_ids:
        if aid not in pretrained_actors_state:
            raise KeyError(f"Checkpoint 缺少 agent {aid} 的权重（合作团队需要 {cooperative_ids}）")
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(pretrained_actors_state[aid])
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        cooperative_actors[aid] = actor
    print(f"已加载冻结合作团队: {list(cooperative_actors.keys())}")

    # 对抗团队：用预训练权重初始化
    adversarial_actors: Dict[int, Actor] = {}
    adversarial_actor_optims: Dict[int, optim.Optimizer] = {}
    for aid in adversarial_ids:
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim,
                      init_log_std=args.init_log_std).to(device)
        if aid in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[aid])
            print(f"对抗 agent {aid} 用其预训练权重初始化")
        elif cooperative_ids and cooperative_ids[0] in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[cooperative_ids[0]])
            print(f"对抗 agent {aid} 用 agent {cooperative_ids[0]} 权重初始化")
        adversarial_actors[aid] = actor
        adversarial_actor_optims[aid] = optim.Adam(actor.parameters(), lr=args.lr_actor)

    # 共享集中式 Critic（输入全局状态）
    critic = CentralizedCritic(state_dim=state_dim, hidden=args.hidden_dim).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)

    init_lr_actor, init_lr_critic = args.lr_actor, args.lr_critic

    def _apply_lr_decay(cur_update: int):
        if not args.lr_decay:
            return init_lr_actor, init_lr_critic
        frac = max(0.05, 1.0 - (cur_update - 1) / max(args.updates - 1, 1))
        new_lr_a, new_lr_c = init_lr_actor * frac, init_lr_critic * frac
        for aid in adversarial_ids:
            for g in adversarial_actor_optims[aid].param_groups:
                g['lr'] = new_lr_a
        for g in critic_optim.param_groups:
            g['lr'] = new_lr_c
        return new_lr_a, new_lr_c

    # 可视化（可选）
    visualizer = None
    if args.render and not args.no_viz and HAS_VISUALIZER:
        visualizer = MAPPOVisualizer(width=1200, height=800, fps=30)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    run_tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
    tb_log_dir = os.path.join(save_dir, 'tb_logs', run_tag)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_dir)
    print(f"TensorBoard 日志: {tb_log_dir}")

    # 训练统计
    total_episodes = 0
    total_coop_success = 0          # 合作团队全部到达
    total_coop_collision = 0        # 任一合作 agent 碰撞
    best_adv_effectiveness = -1.0   # best.pt 跟踪（窗口 1 - coop_success 最大）
    start_update = 1

    # ── 恢复训练（--resume）──
    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            cands = []
            if os.path.isdir(save_dir):
                for f in os.listdir(save_dir):
                    if f.startswith("multi_adv_v2_update_") and f.endswith(".pt"):
                        try:
                            n = int(f.replace("multi_adv_v2_update_", "").replace(".pt", ""))
                            cands.append((n, f))
                        except ValueError:
                            continue
            if not cands:
                print(f"⚠️ --resume latest: {save_dir} 下没有 multi_adv_v2_update_*.pt，从零开始")
                resume_path = ""
            else:
                cands.sort()
                resume_path = os.path.join(save_dir, cands[-1][1])
        if resume_path and os.path.exists(resume_path):
            print(f"恢复训练 from: {resume_path}")
            rck = torch.load(resume_path, map_location=device)
            for aid in adversarial_ids:
                if aid in rck.get('adversarial_actors', {}):
                    adversarial_actors[aid].load_state_dict(rck['adversarial_actors'][aid])
            if 'critic' in rck:
                critic.load_state_dict(rck['critic'])
            if 'adversarial_actor_optims' in rck:
                for aid in adversarial_ids:
                    if aid in rck['adversarial_actor_optims']:
                        adversarial_actor_optims[aid].load_state_dict(rck['adversarial_actor_optims'][aid])
            if 'critic_optim' in rck:
                critic_optim.load_state_dict(rck['critic_optim'])
            start_update = int(rck.get('update', 0)) + 1
            total_episodes = int(rck.get('total_episodes', 0))
            total_coop_success = int(rck.get('total_coop_success', 0))
            total_coop_collision = int(rck.get('total_coop_collision', 0))
            best_adv_effectiveness = float(rck.get('best_adv_effectiveness', -1.0))
            if 'curriculum_success_history' in rck:
                for k, v in rck['curriculum_success_history'].items():
                    curriculum.scenario_success_history[k] = list(v)
            print(f"  → 续训 update={start_update}, total_ep={total_episodes}")

    print(f"对抗训练开始 (MAPPO CTDE, per-agent advantage, global critic)...")

    def _build_ckpt_dict(update):
        return {
            "update": update,
            "adversarial_actors": {aid: adversarial_actors[aid].state_dict() for aid in adversarial_ids},
            "critic": critic.state_dict(),
            "adversarial_actor_optims": {aid: adversarial_actor_optims[aid].state_dict() for aid in adversarial_ids},
            "critic_optim": critic_optim.state_dict(),
            "cooperative_ids": cooperative_ids,
            "adversarial_ids": adversarial_ids,
            "pretrained_ckpt": args.pretrained_ckpt,
            "args": vars(args),
            "total_episodes": total_episodes,
            "total_coop_success": total_coop_success,
            "total_coop_collision": total_coop_collision,
            "best_adv_effectiveness": best_adv_effectiveness,
            "curriculum_success_history": {
                k: list(v) for k, v in curriculum.scenario_success_history.items()
            },
        }

    try:
        for update in range(start_update, args.updates + 1):
            cur_lr_a, cur_lr_c = _apply_lr_decay(update)
            cur_zero_sum_coef = compute_zero_sum_coef(update, args)

            buffer = RolloutBuffer(adversarial_ids)

            window_coop_success: List[float] = []
            window_coop_collision: List[float] = []
            window_coop_returns: List[float] = []
            window_adv_returns: Dict[int, List[float]] = {aid: [] for aid in adversarial_ids}
            window_episode_lengths: List[int] = []
            comp_adv_base: List[float] = []
            comp_zero_sum: List[float] = []

            # 场景采样
            available_scenarios = curriculum.get_available_scenarios(update)
            success_rates = None
            if args.scenario_sampling_mode == "weighted":
                success_rates = {s: curriculum.get_scenario_success_rate(s) for s in available_scenarios}
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

            print(f"\n更新 {update}/{args.updates}, zero_sum_coef={cur_zero_sum_coef:.3f}, "
                  f"可用场景数: {len(available_scenarios)}, "
                  f"采样: {len(set(sampled_scenarios))} 种/{len(sampled_scenarios)} ep")

            # ── 收集 rollout ──
            for rollout_episode in range(args.rollout_episodes):
                scenario_name = sampled_scenarios[rollout_episode]
                scenario_config = env.load_scenario_by_name(scenario_name, scenario_dir)
                if scenario_config is None:
                    scenario_config = env.load_random_scenario(scenario_dir, [scenario_name])
                if scenario_config:
                    scenario_config = dict(scenario_config)
                    # 红车起点：固定从 coop-start-config 覆盖（保留采样场景的 goal_zone）
                    if coop_start_positions is not None:
                        agents = _scenario_agents(scenario_config)
                        agents.update(coop_start_positions)
                        scenario_config["agents"] = agents
                    if args.auto_adv_positions:
                        # 蓝(对抗)车夹在红(合作)车之间，排成 红-蓝…蓝-红（确定性）
                        scenario_config["agents"] = make_adversarial_positions(
                            scenario_config, cooperative_ids, adversarial_ids,
                            margin_ratio=args.adv_margin_ratio,
                        )
                    env.set_scenario(scenario_config)

                obs_dict, info = env.reset()
                total_episodes += 1

                coop_ep_reward = 0.0
                adv_ep_rewards = {aid: 0.0 for aid in adversarial_ids}
                ep_length = 0
                collision_sticky: Dict[int, bool] = {i: False for i in range(total_agents)}

                for step in range(args.max_steps):
                    ep_length += 1
                    action_dict: Dict[int, np.ndarray] = {}
                    step_cache: Dict[int, Dict] = {}

                    global_state_np = env.get_global_state().astype(np.float32)
                    global_state_t = torch.as_tensor(global_state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        global_value = float(critic(global_state_t).item())

                    with torch.no_grad():
                        # 合作团队：deterministic frozen
                        for cid in cooperative_ids:
                            if cid not in obs_dict:
                                continue
                            obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32, device=device).unsqueeze(0)
                            a_tanh = cooperative_actors[cid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                            action_dict[cid] = action_tanh_to_env(a_tanh)
                        # 对抗团队：stochastic
                        for aid in adversarial_ids:
                            if aid not in obs_dict:
                                continue
                            obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device).unsqueeze(0)
                            a_tanh_t, logp_t = adversarial_actors[aid].sample_action(obs_t)
                            a_tanh = a_tanh_t.squeeze(0).cpu().numpy()
                            action_dict[aid] = action_tanh_to_env(a_tanh)
                            step_cache[aid] = {
                                "obs": obs_dict[aid].copy(),
                                "global_state": global_state_np,
                                "action_tanh": a_tanh,
                                "logp": float(logp_t.item()),
                                "value": global_value,
                            }

                    next_obs, rewards, dones, info = env.step(action_dict)

                    step_collisions = info.get('collisions', {}) or {}
                    for i in range(total_agents):
                        if step_collisions.get(i, False):
                            collision_sticky[i] = True

                    # 合作团队回报（仅监控）
                    coop_step_rewards = [rewards.get(cid, 0.0) for cid in cooperative_ids if cid in rewards]
                    avg_coop_reward = float(np.mean(coop_step_rewards)) if coop_step_rewards else 0.0
                    coop_ep_reward += avg_coop_reward

                    # 对抗团队：组合奖励 = 对抗环境奖励 + zero_sum_coef × (-avg_coop)
                    for aid in step_cache.keys():
                        base_adv_reward = float(rewards.get(aid, 0.0))
                        zero_sum_term = cur_zero_sum_coef * (-avg_coop_reward)
                        combined = base_adv_reward + zero_sum_term
                        adv_ep_rewards[aid] += combined
                        comp_adv_base.append(base_adv_reward)
                        comp_zero_sum.append(zero_sum_term)
                        buffer.add(aid, Transition(
                            obs=step_cache[aid]["obs"],
                            global_state=step_cache[aid]["global_state"],
                            action_tanh=step_cache[aid]["action_tanh"],
                            log_prob=step_cache[aid]["logp"],
                            reward=combined,
                            done=float(dones.get(aid, False)),
                            value=step_cache[aid]["value"],
                            scenario_name=scenario_name,
                        ))

                    if visualizer is not None:
                        visualizer.handle_events()
                        if not visualizer.running:
                            raise KeyboardInterrupt

                    obs_dict = next_obs
                    if all(dones.values()):
                        break

                # episode 终止分类（合作团队视角）
                coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
                coop_collision = any(collision_sticky[cid] for cid in cooperative_ids)
                # 对抗成功 = 合作团队没能全部到达（攻击有效）
                adv_success = not coop_success
                if coop_success:
                    total_coop_success += 1
                if coop_collision:
                    total_coop_collision += 1

                # per-agent 截断 bootstrap
                last_gs = env.get_global_state().astype(np.float32)
                for aid in adversarial_ids:
                    buffer.mark_episode_end(aid)
                    own_reached = env.reached_target.get(aid, False)
                    own_crashed = collision_sticky[aid]
                    naturally_terminal = own_reached or own_crashed
                    buffer.add_episode_bootstrap(aid, None if naturally_terminal else last_gs)

                window_coop_success.append(1.0 if coop_success else 0.0)
                window_coop_collision.append(1.0 if coop_collision else 0.0)
                window_coop_returns.append(coop_ep_reward)
                window_episode_lengths.append(ep_length)
                for aid in adversarial_ids:
                    window_adv_returns[aid].append(adv_ep_rewards[aid])
                curriculum.record_scenario_result(scenario_name, adv_success)

            # ===== MAPPO 更新（对抗团队） =====
            n_transitions = buffer.total_size()
            skip_update = n_transitions < args.min_transitions_per_update
            ppo_metrics = {"critic_loss": [], "explained_variance": 0.0, "grad_norm_critic": []}
            for aid in adversarial_ids:
                ppo_metrics.update({
                    f"actor_loss_{aid}": [], f"entropy_{aid}": [],
                    f"approx_kl_{aid}": [], f"clip_frac_{aid}": [], f"grad_norm_actor_{aid}": [],
                })
            agent_stop_epoch: Dict[int, int] = {aid: args.ppo_epochs for aid in adversarial_ids}

            if skip_update:
                print(f"  [跳过] transition 数 {n_transitions} < {args.min_transitions_per_update}")
            else:
                per_agent_tensors = {}
                critic_states, critic_returns = [], []
                for aid in adversarial_ids:
                    data = buffer.get_agent_data(aid)
                    if len(data) == 0:
                        continue
                    obs_arr = np.array([t.obs for t in data], dtype=np.float32)
                    state_arr = np.array([t.global_state for t in data], dtype=np.float32)
                    act_arr = np.array([t.action_tanh for t in data], dtype=np.float32)
                    old_logp_arr = np.array([t.log_prob for t in data], dtype=np.float32)
                    reward_arr = np.array([t.reward for t in data], dtype=np.float32)
                    done_arr = np.array([t.done for t in data], dtype=np.float32)
                    value_arr = np.array([t.value for t in data], dtype=np.float32)

                    boundaries = buffer.episode_boundaries.get(aid, [])
                    bootstrap_states = buffer.episode_bootstrap_states.get(aid, [])
                    bootstrap_values = []
                    with torch.no_grad():
                        for bs in bootstrap_states:
                            if bs is not None:
                                bs_t = torch.as_tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)
                                bootstrap_values.append(float(critic(bs_t).item()))
                            else:
                                bootstrap_values.append(0.0)
                    adv_arr, ret_arr = compute_gae_per_episode(
                        rewards=reward_arr, values=value_arr, dones=done_arr,
                        boundaries=boundaries, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        bootstrap_values=bootstrap_values,
                    )
                    per_agent_tensors[aid] = {
                        "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
                        "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
                        "old_logp": torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device),
                        "adv": torch.as_tensor(adv_arr, dtype=torch.float32, device=device),
                    }
                    critic_states.append(state_arr)
                    critic_returns.append(ret_arr)

                if len(critic_states) > 0:
                    critic_state_t = torch.as_tensor(np.concatenate(critic_states, axis=0), dtype=torch.float32, device=device)
                    critic_ret_t = torch.as_tensor(np.concatenate(critic_returns, axis=0), dtype=torch.float32, device=device)

                    agent_stopped: Dict[int, bool] = {aid: False for aid in adversarial_ids}
                    for epoch in range(args.ppo_epochs):
                        # Critic
                        c_n = critic_state_t.shape[0]
                        c_idx = np.arange(c_n)
                        np.random.shuffle(c_idx)
                        epoch_critic_losses = []
                        for start in range(0, c_n, args.minibatch_size):
                            mb = c_idx[start: start + args.minibatch_size]
                            value_pred = critic(critic_state_t[mb])
                            critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()
                            critic_optim.zero_grad()
                            critic_loss.backward()
                            cgn = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                            critic_optim.step()
                            epoch_critic_losses.append(float(critic_loss.item()))
                            ppo_metrics["grad_norm_critic"].append(float(cgn))
                        ppo_metrics["critic_loss"].append(float(np.mean(epoch_critic_losses)))

                        # Actor（per-agent）
                        for aid in adversarial_ids:
                            if aid not in per_agent_tensors or agent_stopped[aid]:
                                continue
                            a_obs = per_agent_tensors[aid]["obs"]
                            a_act = per_agent_tensors[aid]["act"]
                            a_oldlogp = per_agent_tensors[aid]["old_logp"]
                            a_adv = per_agent_tensors[aid]["adv"]
                            a_n = a_obs.shape[0]
                            a_idx = np.arange(a_n)
                            np.random.shuffle(a_idx)
                            ep_losses, ep_ent, ep_kl, ep_cf = [], [], [], []
                            for start in range(0, a_n, args.minibatch_size):
                                mb = a_idx[start: start + args.minibatch_size]
                                new_logp, entropy = adversarial_actors[aid].evaluate_actions(a_obs[mb], a_act[mb])
                                logratio = new_logp - a_oldlogp[mb]
                                ratio = torch.exp(logratio)
                                surr1 = ratio * a_adv[mb]
                                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * a_adv[mb]
                                actor_loss = -torch.min(surr1, surr2).mean() - args.ent_coef * entropy.mean()
                                adversarial_actor_optims[aid].zero_grad()
                                actor_loss.backward()
                                agn = nn.utils.clip_grad_norm_(adversarial_actors[aid].parameters(), args.max_grad_norm)
                                adversarial_actor_optims[aid].step()
                                ep_losses.append(float(actor_loss.item()))
                                ep_ent.append(float(entropy.mean().item()))
                                with torch.no_grad():
                                    ep_kl.append(float(((ratio - 1.0) - logratio).mean().item()))
                                    ep_cf.append(float(((ratio - 1.0).abs() > args.clip_eps).float().mean().item()))
                            ppo_metrics[f"actor_loss_{aid}"].append(float(np.mean(ep_losses)))
                            ppo_metrics[f"entropy_{aid}"].append(float(np.mean(ep_ent)))
                            mean_kl = float(np.mean(ep_kl))
                            ppo_metrics[f"approx_kl_{aid}"].append(mean_kl)
                            ppo_metrics[f"clip_frac_{aid}"].append(float(np.mean(ep_cf)))
                            ppo_metrics[f"grad_norm_actor_{aid}"].append(float(agn))
                            if args.target_kl > 0 and mean_kl > 1.5 * args.target_kl:
                                agent_stopped[aid] = True
                                agent_stop_epoch[aid] = epoch + 1

                    with torch.no_grad():
                        all_vp = critic(critic_state_t)
                        var_res = torch.var(critic_ret_t - all_vp)
                        var_tot = torch.var(critic_ret_t)
                        ppo_metrics["explained_variance"] = float(1.0 - var_res / (var_tot + 1e-8))

            # ===== TensorBoard =====
            if update % args.log_interval == 0 and len(window_coop_success) > 0:
                win_coop_succ = float(np.mean(window_coop_success))
                win_coop_coll = float(np.mean(window_coop_collision))
                adv_eff = 1.0 - win_coop_succ
                tb_writer.add_scalar('coop/rate_success', win_coop_succ, update)
                tb_writer.add_scalar('coop/rate_collision', win_coop_coll, update)
                tb_writer.add_scalar('coop/avg_return', float(np.mean(window_coop_returns)), update)
                tb_writer.add_scalar('coop/success_rate_cumulative', total_coop_success / max(total_episodes, 1), update)
                tb_writer.add_scalar('adv/effectiveness_window', adv_eff, update)
                tb_writer.add_scalar('adv/zero_sum_coef', cur_zero_sum_coef, update)
                tb_writer.add_scalar('episode/avg_length', float(np.mean(window_episode_lengths)), update)
                tb_writer.add_scalar('episode/n_transitions', n_transitions, update)
                tb_writer.add_scalar('reward/adv_base', float(np.mean(comp_adv_base)) if comp_adv_base else 0.0, update)
                tb_writer.add_scalar('reward/zero_sum_term', float(np.mean(comp_zero_sum)) if comp_zero_sum else 0.0, update)
                tb_writer.add_scalar('train/lr_actor', cur_lr_a, update)
                tb_writer.add_scalar('train/lr_critic', cur_lr_c, update)
                adv_return_means = []
                for aid in adversarial_ids:
                    if window_adv_returns[aid]:
                        m = float(np.mean(window_adv_returns[aid]))
                        adv_return_means.append(m)
                        tb_writer.add_scalar(f'adv/return_agent_{aid}', m, update)
                        with torch.no_grad():
                            tb_writer.add_scalar(f'train/log_std_agent{aid}',
                                                 float(adversarial_actors[aid].log_std.mean().item()), update)
                tb_writer.add_scalar('adv/avg_return', float(np.mean(adv_return_means)) if adv_return_means else 0.0, update)

                if not skip_update and ppo_metrics["critic_loss"]:
                    tb_writer.add_scalar('ppo/critic_loss', float(np.mean(ppo_metrics["critic_loss"])), update)
                    tb_writer.add_scalar('ppo/critic_grad_norm', float(np.mean(ppo_metrics["grad_norm_critic"])), update)
                    tb_writer.add_scalar('ppo/explained_variance', ppo_metrics["explained_variance"], update)
                    for aid in adversarial_ids:
                        if ppo_metrics[f"actor_loss_{aid}"]:
                            tb_writer.add_scalar(f'ppo/actor_loss_agent{aid}', float(np.mean(ppo_metrics[f"actor_loss_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/entropy_agent{aid}', float(np.mean(ppo_metrics[f"entropy_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/approx_kl_agent{aid}', float(np.mean(ppo_metrics[f"approx_kl_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/clip_frac_agent{aid}', float(np.mean(ppo_metrics[f"clip_frac_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/grad_norm_actor_agent{aid}', float(np.mean(ppo_metrics[f"grad_norm_actor_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/stop_epoch_agent{aid}', agent_stop_epoch.get(aid, args.ppo_epochs), update)

                for sname in curriculum.scenario_success_history:
                    hist = curriculum.scenario_success_history[sname]
                    if len(hist) >= 3:
                        safe = sname.replace('.json', '').replace(' ', '_')[:40]
                        tb_writer.add_scalar(f'scenarios/{safe}/adv_success_rate', float(np.mean(hist)), update)

                print(f"更新 {update}: adv_effectiveness={adv_eff:.2%} "
                      f"(coop_success={win_coop_succ:.2%}, coop_collision={win_coop_coll:.2%}), "
                      f"adv_return={float(np.mean(adv_return_means)) if adv_return_means else 0.0:.2f}, "
                      f"adv_base={float(np.mean(comp_adv_base)) if comp_adv_base else 0.0:.3f}, "
                      f"zero_sum={float(np.mean(comp_zero_sum)) if comp_zero_sum else 0.0:.3f}, "
                      f"n_trans={n_transitions}, lr_a={cur_lr_a:.2e}")

            # 保存 checkpoint
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"multi_adv_v2_update_{update}.pt")
                torch.save(_build_ckpt_dict(update), ckpt_path)
                print(f"模型保存: {ckpt_path}")

            # best.pt：窗口对抗有效性最大
            if len(window_coop_success) > 0:
                cur_eff = 1.0 - float(np.mean(window_coop_success))
                if cur_eff > best_adv_effectiveness:
                    best_adv_effectiveness = cur_eff
                    best_path = os.path.join(save_dir, "best.pt")
                    torch.save(_build_ckpt_dict(update), best_path)
                    print(f"  [best] 对抗有效性 {best_adv_effectiveness:.2%}，保存 {best_path}")

    except KeyboardInterrupt:
        print("训练中断")
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()
        tb_writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="多对抗 MAPPO 训练 v2 (CTDE, per-agent adv, global critic)")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    # 团队配置
    parser.add_argument("--pretrained-ckpt", type=str, required=True, help="预训练合作 MAPPO checkpoint")
    parser.add_argument("--num-cooperative", type=int, default=2)
    parser.add_argument("--num-adversarial", type=int, default=2)

    # 初始位置：红(合作)-蓝(对抗) 排布
    parser.add_argument("--coop-start-config", type=str, default="scenario_03_far.json",
                        help="红车(合作)起点固定读取的配置文件（scenario_dir 下的文件名或绝对路径）。"
                             "避免课程早期红车离终点太近（默认 scenario_03_far.json，红车 y=160/220）。"
                             "设为 none/'' 则改用每个采样场景自带的红车起点。")
    parser.add_argument("--auto-adv-positions", action="store_true", default=True,
                        help="对抗车(蓝)确定性安置在合作车(红)之间，排成 红-蓝…蓝-红，"
                             "蓝车不会比红车更靠近终点（默认开）")
    parser.add_argument("--no-auto-adv-positions", dest="auto_adv_positions", action="store_false",
                        help="关闭自动排布，退回环境默认 waypoint 分配（红车扎堆在前、蓝车在后）")
    parser.add_argument("--adv-margin-ratio", type=float, default=0.0,
                        help="蓝车分布区间相对红车两端的内缩比例（0=红车之间均匀等分，"
                             ">0 让蓝车更聚向中间、离红车更远）")

    # 训练参数
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--rollout-episodes", type=int, default=16,
                        help="每次更新收集的 episode 数（episode 制 rollout）")
    parser.add_argument("--ppo-epochs", type=int, default=5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--min-transitions-per-update", type=int, default=512)

    # 学习率 / PPO
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997,
                        help="折扣因子（远距收敛修复：0.99→0.997，长直道长视野）")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.15)
    parser.add_argument("--ent-coef", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--lr-decay", action="store_true", default=True)
    parser.add_argument("--no-lr-decay", dest="lr_decay", action="store_false")
    parser.add_argument("--init-log-std", type=float, default=0.0)

    # 零和项 + 对抗课程（系数 ramp）
    parser.add_argument("--zero-sum-coef", type=float, default=1.0,
                        help="零和项权重：combined = adv_env + zero_sum_coef × (-avg_coop)。取代 v1 写死的 ×0.01")
    parser.add_argument("--curriculum-learning", action="store_true",
                        help="启用对抗系数课程（warmup → ramp-up 作用于 zero-sum-coef）")
    parser.add_argument("--warmup-updates", type=int, default=50, help="对抗系数预热（系数=0）")
    parser.add_argument("--ramp-up-updates", type=int, default=100, help="对抗系数渐增轮数")

    # 场景采样 + 场景课程
    parser.add_argument("--scenario-sampling-mode", type=str, default="weighted",
                        choices=["random", "stratified", "weighted"])
    parser.add_argument("--scenario-warmup-updates", type=int, default=10, help="基础场景预热（场景课程）")
    parser.add_argument("--scenario-full-updates", type=int, default=40, help="达到全场景的更新次数")
    parser.add_argument("--curriculum-transition-window", type=int, default=10, help="场景阶段过渡窗口")
    parser.add_argument("--reward-version", type=str, default="v2", choices=["v1", "v2"],
                        help="奖励版本（必须 v2 才发放对抗 cut_in/blocking 奖励）")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-viz", action="store_true", help="不启动可视化（默认即无）")
    parser.add_argument("--resume", type=str, default="",
                        help="恢复训练 ckpt 路径，'latest' 自动选 save_dir 下最大 update")
    parser.add_argument("--save-dir", type=str, default="./logs/multi_adv_v2")
    parser.add_argument("--save-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=1)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
