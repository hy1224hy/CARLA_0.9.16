"""
对抗 MAPPO 训练（KL-anchored + 风险增量奖励）
=============================================
基于 train_multi_adversarial_mappo_v2.py 的设计意图重写，融合以下论文做法：

- STRIVE (CVPR 2022) / Robustifying Imitation (Waymo 2023)：
    actor loss 加 KL 锚定到预训练正常驾驶策略，让 BLUE 始终"像正常司机"。
- AdvSim (ICCV 2021) / Causal Influence (ICML 2019)：
    奖励改为"BLUE 对 RED 施加的 TTC 风险"，归因到 BLUE 实际造成的威胁，
    而不是 BLUE 自己的几何位置。
- PID-Lagrangian (ICML 2020) 简化版：
    BLUE 自碰撞惩罚自适应调整 —— 自碰率超出 ε 就加大，反之放松。

设计文档：docs/对抗训练_文献调研与新方案.md

复用 v2 的：RolloutBuffer / CurriculumLearning / GAE / 场景采样 / 位置布置。

注意：必须 reward_version='v2'，对抗 reward 路径才会发放 base_driving 明细。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import copy
import datetime as _dt
import math
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, AgentConditionedCritic

from training.train_multi_adversarial_mappo_v2 import (
    set_seed,
    Transition,
    RolloutBuffer,
    action_tanh_to_env,
    _scenario_agents,
    _load_config_file,
    make_adversarial_positions,
    compute_gae_per_episode,
    sample_scenarios_stratified,
    CurriculumLearning,
    compute_zero_sum_coef,
)

try:
    from visualization.mappo_visualizer import MAPPOVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


# ============================================================
# 新组件 1：BLUE 对 RED 的威胁评分（风险增量奖励的核心）
# ============================================================

def compute_blue_threat_scores(env: CarlaMultiAgentEnv,
                               cooperative_ids: List[int],
                               adversarial_ids: List[int],
                               threat_radius: float = 15.0,
                               threat_lateral: float = 2.5,
                               threat_tau: float = 1.5,
                               threat_near: float = 3.0,
                               red_desired_speed: float = 4.0,
                               kinematics: Optional[Dict[int, object]] = None,
                               ) -> Dict[int, float]:
    """对每个 BLUE，计算它当前步对所有 RED 施加的总威胁分。

    ── 优化 A（安全挡道，2026-06-24）──
    旧实现 threat = exp(−TTC/τ) 在 TTC→0（即将相撞）时最大，**直接奖励贴到撞距离** →
    蓝车学会怼上去把红车撞停（自毁式）。新实现把"贴脸即撞"的激励换成"**安全距离逼停**"：

      威胁分 = band_kernel(纵向距离) × impede(红车被压慢的程度)

      - band_kernel：在 [threat_near, threat_radius] 内的钟形核，峰在偏近但**安全**的距离
        d_peak（≈ threat_near + 0.3·跨度）；**纵向 < threat_near（撞距离）直接 0 分**，
        于是"怼上去"不再有正回报（撞了还要吃自碰罚）。
      - impede = clamp((red_desired_speed − red_前向速度)/red_desired_speed, 0, 1)：
        红车被压得越慢，挡道越有效；红车自由巡航（没被挡慢）则≈0。
        这奖励"**持续把红车压慢**"而非"逼近瞬间"，方向对齐"拦住红车"的真实目标。

    必要前置：BLUE 在 RED 正前方（纵向>0）、横向对齐（|lat|≤threat_lateral）、RED 未到点。

    返回：{blue_id: total_threat_this_step}（非负）。
    threat_tau 保留以兼容旧签名（新核不再用 TTC）。
    """
    scores: Dict[int, float] = {aid: 0.0 for aid in adversarial_ids}

    # 缓存所有车的位姿/速度，避免重复 API 调用。碰撞帧可由调用方传入
    # info['pre_collision_kinematics'] 还原出的 AgentKinematics，避免 actor 已 despawn。
    cache: Dict[int, Dict] = {}
    if kinematics is not None:
        for aid in cooperative_ids + adversarial_ids:
            kin = kinematics.get(aid)
            if kin is None or not getattr(kin, "alive", True):
                continue
            cache[aid] = {
                "loc": kin.loc,
                "fwd": kin.fwd,
                "right": kin.right,
                "fwd_speed": kin.fwd_speed,
                "reached": bool(getattr(kin, "reached", False)),
            }
    else:
        for aid in cooperative_ids + adversarial_ids:
            if aid >= len(env.vehicles):
                continue
            v = env.vehicles[aid]
            if v is None or not v.is_alive:
                continue
            tr = v.get_transform()
            vel = v.get_velocity()
            yaw_rad = math.radians(tr.rotation.yaw)
            fwd = (math.cos(yaw_rad), math.sin(yaw_rad))
            right = (math.sin(yaw_rad), -math.cos(yaw_rad))
            fwd_speed = vel.x * fwd[0] + vel.y * fwd[1]
            cache[aid] = {
                "loc": (tr.location.x, tr.location.y),
                "fwd": fwd,
                "right": right,
                "fwd_speed": fwd_speed,
                "reached": bool(env.reached_target.get(aid, False)),
            }

    span = max(threat_radius - threat_near, 1e-3)
    d_peak = threat_near + 0.3 * span        # 安全挡道甜点（偏近但不进撞距离）
    width = 0.6 * span                       # 钟形核宽度
    desired = max(red_desired_speed, 1e-3)

    for red_id in cooperative_ids:
        if red_id not in cache:
            continue
        rc = cache[red_id]
        if rc["reached"]:
            continue   # RED 已到点，BLUE 再威胁也没用
        # 阻滞因子：红车被压得越慢越高（自由巡航≈0）
        impede = (desired - rc["fwd_speed"]) / desired
        impede = min(1.0, max(0.0, impede))
        if impede <= 0.0:
            continue   # 红车没被压慢 → 没在实际挡路
        for blue_id in adversarial_ids:
            if blue_id not in cache:
                continue
            bc = cache[blue_id]

            # blue 相对 red 的纵向 / 横向位置（red 坐标系）
            rel = (bc["loc"][0] - rc["loc"][0], bc["loc"][1] - rc["loc"][1])
            longitudinal = rel[0] * rc["fwd"][0] + rel[1] * rc["fwd"][1]
            lateral = rel[0] * rc["right"][0] + rel[1] * rc["right"][1]

            # 必须在 RED 前方【安全挡道带】内：纵向 < threat_near（撞距离）不给分
            if longitudinal < threat_near or longitudinal > threat_radius:
                continue
            if abs(lateral) > threat_lateral:
                continue

            band = math.exp(-((longitudinal - d_peak) / width) ** 2)
            scores[blue_id] += band * impede

    return scores


def extract_base_driving_safe(reward_details: Dict, adv_id: int) -> float:
    """从环境 reward_details 中提取"安全驾驶"基础奖励，剔除目标导向项。

    剔除：progress / success / time_bonus（这些项推车去 goal，与对抗目标冲突）
    保留：speed / centering / yaw / jerk / steer_rate / safety_lidar / ttc / lane_invasion / step

    注意 collision 项单独由自适应 self_collision_coef 控制，这里也剔除。
    """
    if adv_id not in reward_details:
        return 0.0
    d = reward_details[adv_id]
    # _compute_adversarial_reward 在 v2 路径里会把 base_details 包成新的 dict（含 base_reward），
    # 但 base_driving 分量在 base 计算阶段就已经写入 reward_details 了。
    # 这里直接按 _compute_rewards_v2 输出的 keys 取。
    keys_safe = [
        ("speed", 0.3),
        ("centering", 0.2),
        ("yaw", 0.1),
        ("jerk", 0.3),
        ("steer_rate", 0.3),
        ("safety_lidar", 0.3),
        ("ttc", 0.5),
        ("lane_invasion", 0.5),
        ("step", 0.1),
    ]
    total = 0.0
    for k, w in keys_safe:
        total += float(d.get(k, 0.0)) * w
    return total


def extract_collision_component(reward_details: Dict, adv_id: int) -> float:
    """提取 collision 分量原值（负数或 0），供自适应惩罚使用。"""
    if adv_id not in reward_details:
        return 0.0
    return float(reward_details[adv_id].get("collision", 0.0))


# ============================================================
# 新组件 2：对角高斯 KL（pre-tanh）
# ============================================================

def gaussian_kl_diag(mu_p: torch.Tensor, log_std_p: torch.Tensor,
                     mu_q: torch.Tensor, log_std_q: torch.Tensor) -> torch.Tensor:
    """KL(p ‖ q) for diagonal Gaussians, summed over dim, mean over batch.

    KL = sum_i [ log(σ_q/σ_p) + (σ_p² + (μ_p−μ_q)²) / (2σ_q²) − 0.5 ]
    """
    var_p = (2.0 * log_std_p).exp()
    var_q = (2.0 * log_std_q).exp()
    kl = log_std_q - log_std_p + (var_p + (mu_p - mu_q).pow(2)) / (2.0 * var_q) - 0.5
    return kl.sum(dim=-1).mean()


# ============================================================
# 训练主函数
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    total_agents = args.num_cooperative + args.num_adversarial
    cooperative_ids = list(range(args.num_cooperative))
    adversarial_ids = list(range(args.num_cooperative, total_agents))

    print(f"KL-anchored + 风险增量奖励 对抗训练:")
    print(f"  合作团队 (冻结): {cooperative_ids}")
    print(f"  对抗团队 (训练): {adversarial_ids}")
    print(f"  β_KL = {args.beta_kl}, threat_coef = {args.threat_coef}")
    print(f"  target_blue_collision_rate ε = {args.target_blue_collision_rate}")

    env = CarlaMultiAgentEnv(
        host=args.host, port=args.port, town=args.town,
        num_agents=total_agents,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep, max_steps=args.max_steps,
        spawn_vehicles=True,
        adversarial_agent_ids=adversarial_ids,
        reward_version="v2",
        disable_tensorboard=True,
        seed=args.seed,
    )

    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
    curriculum = CurriculumLearning(
        scenario_dir=scenario_dir,
        warmup_updates=args.scenario_warmup_updates,
        full_scenario_updates=args.scenario_full_updates,
        transition_window=args.curriculum_transition_window,
    )

    coop_start_positions: Optional[Dict[str, Dict]] = None
    if args.coop_start_config and args.coop_start_config.lower() not in ("none", ""):
        coop_cfg = _load_config_file(args.coop_start_config, scenario_dir)
        if coop_cfg is not None:
            src = _scenario_agents(coop_cfg)
            coop_start_positions = {f"agent{cid}": src[f"agent{cid}"]
                                    for cid in cooperative_ids if f"agent{cid}" in src}
            print(f"红车固定起点: {args.coop_start_config}")

    state_dim = env.get_global_state().shape[0]

    # ── 加载预训练 checkpoint ──
    pretrained_path = args.pretrained_ckpt
    while pretrained_path.startswith('\\') or pretrained_path.startswith('/'):
        pretrained_path = pretrained_path[1:]
    pretrained_path = pretrained_path.replace('\\', os.sep).replace('/', os.sep)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {pretrained_path}")
    print(f"加载预训练: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_actors_state = ckpt.get("actors", {})

    # 合作团队：冻结，eval 推理
    cooperative_actors: Dict[int, Actor] = {}
    for aid in cooperative_ids:
        if aid not in pretrained_actors_state:
            raise KeyError(f"checkpoint 缺少 agent {aid} 权重")
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(device)
        actor.load_state_dict(pretrained_actors_state[aid])
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        cooperative_actors[aid] = actor

    # 对抗团队：用预训练权重初始化（PPO 训练）
    adversarial_actors: Dict[int, Actor] = {}
    base_actors_frozen: Dict[int, Actor] = {}   # 新增：KL 锚定用的冻结副本
    adversarial_actor_optims: Dict[int, optim.Optimizer] = {}
    for aid in adversarial_ids:
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim,
                      init_log_std=args.init_log_std).to(device)
        if aid in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[aid])
            print(f"adv {aid} 用其预训练权重初始化")
        elif cooperative_ids and cooperative_ids[0] in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[cooperative_ids[0]])
            print(f"adv {aid} 用 coop {cooperative_ids[0]} 权重初始化")
        adversarial_actors[aid] = actor
        adversarial_actor_optims[aid] = optim.Adam(actor.parameters(), lr=args.lr_actor)

        # KL 锚定：deepcopy 一份冻结
        anchor = copy.deepcopy(actor)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad = False
        base_actors_frozen[aid] = anchor

    adv_index = {aid: i for i, aid in enumerate(adversarial_ids)}
    critic = AgentConditionedCritic(state_dim=state_dim, num_agents=len(adversarial_ids),
                                    hidden=args.hidden_dim).to(device)
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

    # 可视化
    visualizer = None
    if args.render and not args.no_viz and HAS_VISUALIZER:
        visualizer = MAPPOVisualizer(width=1200, height=800, fps=30)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    run_tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
    tb_log_dir = os.path.join(save_dir, 'tb_logs', run_tag)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_log_dir)
    print(f"TensorBoard: {tb_log_dir}")

    # 自适应自碰撞系数（PID-Lagrangian 简版，只有 I 项）
    self_collision_coef = args.self_collision_coef_init

    # 训练统计
    total_episodes = 0
    best_adv_effectiveness = -1.0
    start_update = 1

    # ── 续训 ──
    if args.resume:
        resume_path = args.resume
        if resume_path == "latest":
            cands = []
            if os.path.isdir(save_dir):
                for f in os.listdir(save_dir):
                    if f.startswith("kl_risk_update_") and f.endswith(".pt"):
                        try:
                            n = int(f.replace("kl_risk_update_", "").replace(".pt", ""))
                            cands.append((n, f))
                        except ValueError:
                            continue
            if cands:
                cands.sort()
                resume_path = os.path.join(save_dir, cands[-1][1])
            else:
                resume_path = ""
        if resume_path and os.path.exists(resume_path):
            print(f"恢复训练 from: {resume_path}")
            rck = torch.load(resume_path, map_location=device, weights_only=False)
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
            best_adv_effectiveness = float(rck.get('best_adv_effectiveness', -1.0))
            self_collision_coef = float(rck.get('self_collision_coef', self_collision_coef))
            # 课程学习历史：恢复场景成功率统计，否则加权采样从零开始重新摸索
            if 'curriculum_success_history' in rck:
                for k, v in rck['curriculum_success_history'].items():
                    curriculum.scenario_success_history[k] = list(v)
            # RNG 状态：尽力恢复（torch/numpy/python），保证 rollout 采样可复现
            try:
                if 'rng_torch' in rck:
                    torch.set_rng_state(rck['rng_torch'])
                if 'rng_torch_cuda' in rck and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rck['rng_torch_cuda'])
                if 'rng_numpy' in rck:
                    np.random.set_state(rck['rng_numpy'])
                if 'rng_python' in rck:
                    import random as _rand
                    _rand.setstate(rck['rng_python'])
            except Exception as e:
                print(f"  ⚠️ RNG 状态恢复失败（不影响训练）: {e}")
            print(f"  → 续训 update={start_update}, total_ep={total_episodes}, "
                  f"self_coll_coef={self_collision_coef:.2f}")

    def _build_ckpt_dict(update):
        import random as _rand
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
            "best_adv_effectiveness": best_adv_effectiveness,
            "self_collision_coef": self_collision_coef,
            "curriculum_success_history": {
                k: list(v) for k, v in curriculum.scenario_success_history.items()
            },
            "rng_torch": torch.get_rng_state(),
            "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_python": _rand.getstate(),
        }

    try:
        for update in range(start_update, args.updates + 1):
            cur_lr_a, cur_lr_c = _apply_lr_decay(update)
            cur_zero_sum_coef = compute_zero_sum_coef(update, args)

            buffer = RolloutBuffer(adversarial_ids)

            window_coop_success: List[float] = []
            window_coop_collision: List[float] = []
            window_blue_collision: List[float] = []
            window_episode_lengths: List[int] = []
            window_adv_returns: Dict[int, List[float]] = {aid: [] for aid in adversarial_ids}
            comp_base_safe: List[float] = []
            comp_threat: List[float] = []
            comp_self_coll: List[float] = []
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

            print(f"\n更新 {update}/{args.updates}, β_KL={args.beta_kl:.3f}, "
                  f"threat_coef={args.threat_coef:.2f}, self_coll_coef={self_collision_coef:.2f}, "
                  f"zero_sum={cur_zero_sum_coef:.2f}")

            # ── 收集 rollout ──
            for rollout_episode in range(args.rollout_episodes):
                scenario_name = sampled_scenarios[rollout_episode]
                scenario_config = env.load_scenario_by_name(scenario_name, scenario_dir)
                if scenario_config is None:
                    scenario_config = env.load_random_scenario(scenario_dir, [scenario_name])
                if scenario_config:
                    scenario_config = dict(scenario_config)
                    if coop_start_positions is not None:
                        agents = _scenario_agents(scenario_config)
                        agents.update(coop_start_positions)
                        scenario_config["agents"] = agents
                    if args.auto_adv_positions:
                        scenario_config["agents"] = make_adversarial_positions(
                            scenario_config, cooperative_ids, adversarial_ids,
                            margin_ratio=args.adv_margin_ratio,
                        )
                    env.set_scenario(scenario_config)

                obs_dict, info = env.reset()
                total_episodes += 1

                adv_ep_rewards = {aid: 0.0 for aid in adversarial_ids}
                ep_length = 0
                collision_sticky: Dict[int, bool] = {i: False for i in range(total_agents)}

                for step in range(args.max_steps):
                    ep_length += 1
                    action_dict: Dict[int, np.ndarray] = {}
                    step_cache: Dict[int, Dict] = {}

                    global_state_np = env.get_global_state().astype(np.float32)
                    global_state_t = torch.as_tensor(
                        global_state_np, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        agent_values = {
                            aid: float(critic(global_state_t, adv_index[aid]).item())
                            for aid in adversarial_ids
                        }

                    with torch.no_grad():
                        for cid in cooperative_ids:
                            if cid not in obs_dict:
                                continue
                            obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32,
                                                    device=device).unsqueeze(0)
                            a_tanh = cooperative_actors[cid].get_deterministic_action(obs_t)\
                                .squeeze(0).cpu().numpy()
                            action_dict[cid] = action_tanh_to_env(a_tanh)
                        for aid in adversarial_ids:
                            if aid not in obs_dict:
                                continue
                            obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32,
                                                    device=device).unsqueeze(0)
                            a_tanh_t, logp_t, z_t = adversarial_actors[aid].sample_action_z(obs_t)
                            a_tanh = a_tanh_t.squeeze(0).cpu().numpy()
                            a_z = z_t.squeeze(0).cpu().numpy()
                            action_dict[aid] = action_tanh_to_env(a_tanh)
                            step_cache[aid] = {
                                "obs": obs_dict[aid].copy(),
                                "global_state": global_state_np,
                                "action_z": a_z,
                                "logp": float(logp_t.item()),
                                "value": agent_values[aid],
                            }

                    next_obs, rewards, dones, info = env.step(action_dict)
                    terminated_info = info.get('terminated', {})
                    reward_details = info.get('reward_details', {}) or {}

                    step_collisions = info.get('collisions', {}) or {}
                    for i in range(total_agents):
                        if step_collisions.get(i, False):
                            collision_sticky[i] = True

                    # 合作团队 avg reward（零和项用）
                    coop_step_rewards = [rewards.get(cid, 0.0) for cid in cooperative_ids if cid in rewards]
                    avg_coop_reward = float(np.mean(coop_step_rewards)) if coop_step_rewards else 0.0

                    # 风险增量奖励（每个 BLUE 对 RED 的威胁分）
                    threat_scores = compute_blue_threat_scores(
                        env, cooperative_ids, adversarial_ids,
                        threat_radius=args.threat_radius,
                        threat_lateral=args.threat_lateral,
                        threat_tau=args.threat_tau,
                    )

                    for aid in step_cache.keys():
                        base_safe = extract_base_driving_safe(reward_details, aid)
                        threat = threat_scores.get(aid, 0.0)
                        collision_raw = extract_collision_component(reward_details, aid)
                        # collision_raw 已是负数（如 -300），乘 self_collision_coef 控制幅度
                        # 原始 _compute_adversarial_reward 中 collision 系数为 1.0，这里替换为自适应
                        self_coll_term = self_collision_coef * collision_raw
                        zero_sum_term = cur_zero_sum_coef * (-avg_coop_reward)

                        combined = (
                            base_safe
                            + args.threat_coef * threat
                            + self_coll_term
                            + zero_sum_term
                        )
                        adv_ep_rewards[aid] += combined
                        comp_base_safe.append(base_safe)
                        comp_threat.append(threat)
                        comp_self_coll.append(self_coll_term)
                        comp_zero_sum.append(zero_sum_term)
                        buffer.add(aid, Transition(
                            obs=step_cache[aid]["obs"],
                            global_state=step_cache[aid]["global_state"],
                            action_z=step_cache[aid]["action_z"],
                            log_prob=step_cache[aid]["logp"],
                            reward=combined,
                            done=float(dones.get(aid, False)),
                            terminated=float(terminated_info.get(aid, dones.get(aid, False))),
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

                coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
                coop_collision = any(collision_sticky[cid] for cid in cooperative_ids)
                blue_collision = any(collision_sticky[bid] for bid in adversarial_ids)
                adv_success = not coop_success

                last_gs = env.get_global_state().astype(np.float32)
                for aid in adversarial_ids:
                    buffer.mark_episode_end(aid)
                    own_reached = env.reached_target.get(aid, False)
                    own_crashed = collision_sticky[aid]
                    naturally_terminal = own_reached or own_crashed
                    buffer.add_episode_bootstrap(aid, None if naturally_terminal else last_gs)

                window_coop_success.append(1.0 if coop_success else 0.0)
                window_coop_collision.append(1.0 if coop_collision else 0.0)
                window_blue_collision.append(1.0 if blue_collision else 0.0)
                window_episode_lengths.append(ep_length)
                for aid in adversarial_ids:
                    window_adv_returns[aid].append(adv_ep_rewards[aid])
                curriculum.record_scenario_result(scenario_name, adv_success)

            # ===== 自适应自碰撞系数（PID-Lagrangian I 项）=====
            blue_coll_rate = float(np.mean(window_blue_collision)) if window_blue_collision else 0.0
            if blue_coll_rate > args.target_blue_collision_rate:
                self_collision_coef *= 1.2
            else:
                self_collision_coef *= 0.95
            self_collision_coef = float(np.clip(self_collision_coef, 0.1, 10.0))

            # ===== MAPPO + KL 锚定 更新 =====
            n_transitions = buffer.total_size()
            skip_update = n_transitions < args.min_transitions_per_update
            ppo_metrics = {"critic_loss": [], "explained_variance": 0.0, "grad_norm_critic": []}
            for aid in adversarial_ids:
                ppo_metrics.update({
                    f"actor_loss_{aid}": [], f"entropy_{aid}": [], f"kl_anchor_{aid}": [],
                    f"approx_kl_{aid}": [], f"clip_frac_{aid}": [], f"grad_norm_actor_{aid}": [],
                })
            agent_stop_epoch: Dict[int, int] = {aid: args.ppo_epochs for aid in adversarial_ids}

            if skip_update:
                print(f"  [跳过] transition 数 {n_transitions} < {args.min_transitions_per_update}")
            else:
                per_agent_tensors = {}
                critic_states, critic_returns, critic_agent_ids = [], [], []
                for aid in adversarial_ids:
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

                    boundaries = buffer.episode_boundaries.get(aid, [])
                    bootstrap_states = buffer.episode_bootstrap_states.get(aid, [])
                    bootstrap_values = []
                    with torch.no_grad():
                        for bs in bootstrap_states:
                            if bs is not None:
                                bs_t = torch.as_tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)
                                bootstrap_values.append(float(critic(bs_t, adv_index[aid]).item()))
                            else:
                                bootstrap_values.append(0.0)
                    adv_arr, ret_arr = compute_gae_per_episode(
                        rewards=reward_arr, values=value_arr, dones=done_arr,
                        terminateds=terminated_arr, boundaries=boundaries,
                        gamma=args.gamma, gae_lambda=args.gae_lambda,
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
                    critic_agent_ids.append(np.full(len(ret_arr), adv_index[aid], dtype=np.int64))

                if len(critic_states) > 0:
                    critic_state_t = torch.as_tensor(np.concatenate(critic_states, axis=0),
                                                     dtype=torch.float32, device=device)
                    critic_ret_t = torch.as_tensor(np.concatenate(critic_returns, axis=0),
                                                   dtype=torch.float32, device=device)
                    critic_aid_t = torch.as_tensor(np.concatenate(critic_agent_ids, axis=0),
                                                   dtype=torch.long, device=device)

                    agent_stopped: Dict[int, bool] = {aid: False for aid in adversarial_ids}
                    for epoch in range(args.ppo_epochs):
                        # Critic 更新
                        c_n = critic_state_t.shape[0]
                        c_idx = np.arange(c_n); np.random.shuffle(c_idx)
                        epoch_critic_losses = []
                        for start in range(0, c_n, args.minibatch_size):
                            mb = c_idx[start: start + args.minibatch_size]
                            value_pred = critic(critic_state_t[mb], critic_aid_t[mb])
                            critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()
                            critic_optim.zero_grad()
                            critic_loss.backward()
                            cgn = nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                            critic_optim.step()
                            epoch_critic_losses.append(float(critic_loss.item()))
                            ppo_metrics["grad_norm_critic"].append(float(cgn))
                        ppo_metrics["critic_loss"].append(float(np.mean(epoch_critic_losses)))

                        # Actor 更新（per-agent，加 KL 锚定）
                        for aid in adversarial_ids:
                            if aid not in per_agent_tensors or agent_stopped[aid]:
                                continue
                            a_obs = per_agent_tensors[aid]["obs"]
                            a_act = per_agent_tensors[aid]["act"]
                            a_oldlogp = per_agent_tensors[aid]["old_logp"]
                            a_adv = per_agent_tensors[aid]["adv"]
                            a_n = a_obs.shape[0]
                            a_idx = np.arange(a_n); np.random.shuffle(a_idx)
                            ep_losses, ep_ent, ep_kl_p, ep_cf, ep_kl_anc = [], [], [], [], []
                            for start in range(0, a_n, args.minibatch_size):
                                mb = a_idx[start: start + args.minibatch_size]
                                new_logp, entropy = adversarial_actors[aid].evaluate_actions_z(
                                    a_obs[mb], a_act[mb]
                                )
                                logratio = new_logp - a_oldlogp[mb]
                                ratio = torch.exp(logratio)
                                surr1 = ratio * a_adv[mb]
                                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps,
                                                    1.0 + args.clip_eps) * a_adv[mb]
                                ppo_loss = -torch.min(surr1, surr2).mean()

                                # ─── KL 锚定到 frozen base actor ───
                                mu_adv, logstd_adv = adversarial_actors[aid].forward(a_obs[mb])
                                with torch.no_grad():
                                    mu_base, logstd_base = base_actors_frozen[aid].forward(a_obs[mb])
                                kl_anchor = gaussian_kl_diag(mu_adv, logstd_adv, mu_base, logstd_base)

                                actor_loss = (
                                    ppo_loss
                                    - args.ent_coef * entropy.mean()
                                    + args.beta_kl * kl_anchor
                                )

                                adversarial_actor_optims[aid].zero_grad()
                                actor_loss.backward()
                                agn = nn.utils.clip_grad_norm_(adversarial_actors[aid].parameters(),
                                                               args.max_grad_norm)
                                adversarial_actor_optims[aid].step()
                                ep_losses.append(float(actor_loss.item()))
                                ep_ent.append(float(entropy.mean().item()))
                                ep_kl_anc.append(float(kl_anchor.item()))
                                with torch.no_grad():
                                    ep_kl_p.append(float(((ratio - 1.0) - logratio).mean().item()))
                                    ep_cf.append(
                                        float(((ratio - 1.0).abs() > args.clip_eps).float().mean().item())
                                    )
                            ppo_metrics[f"actor_loss_{aid}"].append(float(np.mean(ep_losses)))
                            ppo_metrics[f"entropy_{aid}"].append(float(np.mean(ep_ent)))
                            ppo_metrics[f"kl_anchor_{aid}"].append(float(np.mean(ep_kl_anc)))
                            mean_kl = float(np.mean(ep_kl_p))
                            ppo_metrics[f"approx_kl_{aid}"].append(mean_kl)
                            ppo_metrics[f"clip_frac_{aid}"].append(float(np.mean(ep_cf)))
                            ppo_metrics[f"grad_norm_actor_{aid}"].append(float(agn))
                            if args.target_kl > 0 and mean_kl > 1.5 * args.target_kl:
                                agent_stopped[aid] = True
                                agent_stop_epoch[aid] = epoch + 1

                    with torch.no_grad():
                        all_vp = critic(critic_state_t, critic_aid_t)
                        var_res = torch.var(critic_ret_t - all_vp)
                        var_tot = torch.var(critic_ret_t)
                        ppo_metrics["explained_variance"] = float(1.0 - var_res / (var_tot + 1e-8))

            # ===== TensorBoard 日志 =====
            if update % args.log_interval == 0 and len(window_coop_success) > 0:
                win_coop_succ = float(np.mean(window_coop_success))
                win_coop_coll = float(np.mean(window_coop_collision))
                win_blue_coll = float(np.mean(window_blue_collision))
                adv_eff = 1.0 - win_coop_succ
                tb_writer.add_scalar('coop/rate_success', win_coop_succ, update)
                tb_writer.add_scalar('coop/rate_collision', win_coop_coll, update)
                tb_writer.add_scalar('blue/rate_collision', win_blue_coll, update)
                tb_writer.add_scalar('blue/self_collision_coef', self_collision_coef, update)
                tb_writer.add_scalar('adv/effectiveness_window', adv_eff, update)
                tb_writer.add_scalar('adv/zero_sum_coef', cur_zero_sum_coef, update)
                tb_writer.add_scalar('episode/avg_length',
                                     float(np.mean(window_episode_lengths)), update)
                tb_writer.add_scalar('episode/n_transitions', n_transitions, update)
                tb_writer.add_scalar('reward/base_safe',
                                     float(np.mean(comp_base_safe)) if comp_base_safe else 0.0, update)
                tb_writer.add_scalar('reward/threat',
                                     float(np.mean(comp_threat)) if comp_threat else 0.0, update)
                tb_writer.add_scalar('reward/self_collision',
                                     float(np.mean(comp_self_coll)) if comp_self_coll else 0.0, update)
                tb_writer.add_scalar('reward/zero_sum',
                                     float(np.mean(comp_zero_sum)) if comp_zero_sum else 0.0, update)
                tb_writer.add_scalar('train/lr_actor', cur_lr_a, update)
                tb_writer.add_scalar('train/beta_kl', args.beta_kl, update)
                adv_return_means = []
                for aid in adversarial_ids:
                    if window_adv_returns[aid]:
                        m = float(np.mean(window_adv_returns[aid]))
                        adv_return_means.append(m)
                        tb_writer.add_scalar(f'adv/return_agent_{aid}', m, update)
                        tb_writer.add_scalar(f'train/log_std_agent{aid}',
                                             float(adversarial_actors[aid].log_std.mean().item()), update)
                tb_writer.add_scalar('adv/avg_return',
                                     float(np.mean(adv_return_means)) if adv_return_means else 0.0, update)

                if not skip_update and ppo_metrics["critic_loss"]:
                    tb_writer.add_scalar('ppo/critic_loss',
                                         float(np.mean(ppo_metrics["critic_loss"])), update)
                    tb_writer.add_scalar('ppo/explained_variance',
                                         ppo_metrics["explained_variance"], update)
                    for aid in adversarial_ids:
                        if ppo_metrics[f"actor_loss_{aid}"]:
                            tb_writer.add_scalar(f'ppo/actor_loss_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"actor_loss_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/entropy_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"entropy_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/kl_anchor_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"kl_anchor_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/approx_kl_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"approx_kl_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/clip_frac_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"clip_frac_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/stop_epoch_agent{aid}',
                                                 agent_stop_epoch.get(aid, args.ppo_epochs), update)

                print(f"更新 {update}: adv_eff={adv_eff:.2%} "
                      f"(coop_succ={win_coop_succ:.2%}, coop_coll={win_coop_coll:.2%}, "
                      f"blue_coll={win_blue_coll:.2%}), "
                      f"adv_R={float(np.mean(adv_return_means)) if adv_return_means else 0.0:.2f}, "
                      f"base={float(np.mean(comp_base_safe)) if comp_base_safe else 0.0:.3f}, "
                      f"threat={float(np.mean(comp_threat)) if comp_threat else 0.0:.3f}, "
                      f"self_coll={float(np.mean(comp_self_coll)) if comp_self_coll else 0.0:.3f}, "
                      f"n_trans={n_transitions}, lr_a={cur_lr_a:.2e}")

            # 保存 checkpoint
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"kl_risk_update_{update}.pt")
                torch.save(_build_ckpt_dict(update), ckpt_path)
                print(f"保存: {ckpt_path}")

            if len(window_coop_success) > 0:
                cur_eff = 1.0 - float(np.mean(window_coop_success))
                if cur_eff > best_adv_effectiveness:
                    best_adv_effectiveness = cur_eff
                    best_path = os.path.join(save_dir, "best.pt")
                    torch.save(_build_ckpt_dict(update), best_path)
                    print(f"  [best] 对抗有效性 {best_adv_effectiveness:.2%}")

    except KeyboardInterrupt:
        print("训练中断")
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()
        tb_writer.close()


def build_parser():
    p = argparse.ArgumentParser(description="对抗 MAPPO (KL-anchored + 风险增量奖励)")

    # CARLA
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town02")
    p.add_argument("--straight-road-length", type=float, default=100.0)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=1000)

    # 团队
    p.add_argument("--pretrained-ckpt", type=str, required=True)
    p.add_argument("--num-cooperative", type=int, default=2)
    p.add_argument("--num-adversarial", type=int, default=2)

    # 初始位置
    p.add_argument("--coop-start-config", type=str, default="scenario_03_far.json")
    p.add_argument("--auto-adv-positions", action="store_true", default=True)
    p.add_argument("--no-auto-adv-positions", dest="auto_adv_positions", action="store_false")
    p.add_argument("--adv-margin-ratio", type=float, default=0.0)

    # PPO / 训练
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--rollout-episodes", type=int, default=16)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--min-transitions-per-update", type=int, default=512)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--lr-decay", action="store_true", default=True)
    p.add_argument("--no-lr-decay", dest="lr_decay", action="store_false")
    p.add_argument("--init-log-std", type=float, default=-0.5)

    # ─── 新方案核心参数 ───
    p.add_argument("--beta-kl", type=float, default=0.1,
                   help="KL 锚定权重：β_KL·KL(π_adv ‖ π_base_frozen) 加到 actor loss。"
                        "大→更像正常驾驶，小→对抗自由度大。推荐 0.05~0.3")
    p.add_argument("--threat-coef", type=float, default=2.0,
                   help="风险增量奖励权重 α，乘到 exp(−TTC_red/τ) 求和上")
    p.add_argument("--threat-radius", type=float, default=15.0,
                   help="威胁判定纵向距离上限（米）")
    p.add_argument("--threat-lateral", type=float, default=2.5,
                   help="威胁判定横向窗口（米）")
    p.add_argument("--threat-tau", type=float, default=1.5,
                   help="TTC exp 衰减常数（秒），越大威胁衰减越慢")
    p.add_argument("--target-blue-collision-rate", type=float, default=0.05,
                   help="自适应自碰撞惩罚的目标值 ε。BLUE 自碰率超此值则加大惩罚系数")
    p.add_argument("--self-collision-coef-init", type=float, default=1.0,
                   help="自适应自碰撞惩罚系数初始值（乘到 env 的 collision_raw 上，已是负数）")

    # 零和项（默认关）
    p.add_argument("--zero-sum-coef", type=float, default=0.0,
                   help="零和项权重，默认 0（KL 锚定 + 风险奖励已足够，避免重叠信号）")
    p.add_argument("--curriculum-learning", action="store_true")
    p.add_argument("--warmup-updates", type=int, default=50)
    p.add_argument("--ramp-up-updates", type=int, default=100)

    # 场景采样
    p.add_argument("--scenario-sampling-mode", type=str, default="weighted",
                   choices=["random", "stratified", "weighted"])
    p.add_argument("--scenario-warmup-updates", type=int, default=10)
    p.add_argument("--scenario-full-updates", type=int, default=40)
    p.add_argument("--curriculum-transition-window", type=int, default=10)

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-dir", type=str, default="./logs/multi_adv_kl_risk")
    p.add_argument("--save-interval", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
