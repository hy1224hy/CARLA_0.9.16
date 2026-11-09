"""
对抗有效性快速验证 — 固定场景训练
====================================
为初步验证 KL-anchored + 风险增量奖励方法的有效性而设计的极简训练脚本。

关键简化：
  - 不引入课程学习（无 stage / 无场景采样 / 无 ramp）
  - 固定一个场景反复跑，所有 update 用同样的红/蓝起点与 goal
  - 沿用 KL anchor + threat reward + 自适应自碰惩罚（与 train_multi_adversarial_kl_risk 一致）
  - 增强对抗有效性指标，便于一眼判断方法是否生效

对照基线：
  BLUE 初始权重 = 预训练 RED 权重 → 第 1 次 update 收集的 rollout 就是"BLUE 不学习时"
  的对抗效果。后续 update 中各指标相对 update 1 的变化 = 训练带来的纯对抗增益。

判读指标（TensorBoard 上看趋势）：
  eval/coop_success_rate            ── 应下降（RED 越来越难到点）
  eval/blue_in_threat_window_pct    ── 应上升（BLUE 更多时间处于威胁位）
  eval/threat_reward_per_step       ── 应上升（BLUE 学到威胁 RED）
  eval/red_avg_speed_under_threat   ── 应低于 free（BLUE 真的让 RED 减速）
  eval/blue_collision_rate          ── 应保持 < 15%（KL 锚定生效，BLUE 不自毁）
  ppo/kl_anchor_agentX              ── 应在 0.05~0.3 区间稳定（锚定强度合适）

用法：
  python -m training.train_adversarial_fixed_scenario \
      --pretrained-ckpt logs/robust_v2/best.pt \
      --scenario scenario_01_basic.json \
      --updates 60
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import copy
import datetime as _dt
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, AgentConditionedCritic

# 复用既有组件（RolloutBuffer / Transition / GAE / 蓝车摆位）
from training.train_multi_adversarial_mappo_v2 import (
    set_seed,
    Transition,
    RolloutBuffer,
    action_tanh_to_env,
    _scenario_agents,
    _load_config_file,
    make_adversarial_positions,
    compute_gae_per_episode,
)

# 复用新方案专用组件（威胁评分 / KL / base_safe）
from training.train_multi_adversarial_kl_risk import (
    compute_blue_threat_scores,
    extract_base_driving_safe,
    extract_collision_component,
    gaussian_kl_diag,
)

try:
    from visualization.mappo_visualizer import MAPPOVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


# ============================================================
# 对抗有效性辅助指标
# ============================================================

def is_any_red_under_threat(env, cooperative_ids, adversarial_ids,
                            threat_radius: float, threat_lateral: float) -> bool:
    """判断当前步是否有任何 RED 处于任何 BLUE 的威胁窗口内。

    与 compute_blue_threat_scores 同口径（前方窗口 + 横向偏离），但不算 TTC，
    仅看几何位置 —— 用于"BLUE 是否在场上施压"的占比统计。
    """
    for red_id in cooperative_ids:
        if red_id >= len(env.vehicles):
            continue
        red_v = env.vehicles[red_id]
        if not red_v.is_alive or env.reached_target.get(red_id, False):
            continue
        rt = red_v.get_transform()
        yaw_rad = math.radians(rt.rotation.yaw)
        rf = (math.cos(yaw_rad), math.sin(yaw_rad))
        rr = (math.sin(yaw_rad), -math.cos(yaw_rad))
        rx, ry = rt.location.x, rt.location.y
        for blue_id in adversarial_ids:
            if blue_id >= len(env.vehicles):
                continue
            bv = env.vehicles[blue_id]
            if not bv.is_alive:
                continue
            bl = bv.get_location()
            rel = (bl.x - rx, bl.y - ry)
            longitudinal = rel[0] * rf[0] + rel[1] * rf[1]
            lateral = rel[0] * rr[0] + rel[1] * rr[1]
            if 0.5 <= longitudinal <= threat_radius and abs(lateral) <= threat_lateral:
                return True
    return False


def avg_red_speed(env, cooperative_ids) -> Optional[float]:
    """所有未到点的 RED 平均速度（m/s）。无可计 RED 时返回 None。"""
    speeds = []
    for red_id in cooperative_ids:
        if red_id >= len(env.vehicles):
            continue
        red_v = env.vehicles[red_id]
        if not red_v.is_alive or env.reached_target.get(red_id, False):
            continue
        vel = red_v.get_velocity()
        speeds.append(math.sqrt(vel.x ** 2 + vel.y ** 2))
    return float(np.mean(speeds)) if speeds else None


# ============================================================
# 固定场景配置预处理
# ============================================================

def prepare_fixed_scenario_config(scenario_file: str, scenario_dir: str,
                                  cooperative_ids: List[int], adversarial_ids: List[int],
                                  coop_start_config: Optional[str],
                                  auto_adv_positions: bool, adv_margin_ratio: float) -> Dict:
    """加载场景文件 + 可选 coop 起点覆盖 + 自动蓝车摆位，返回最终的 scenario_config。"""
    scenario_config = _load_config_file(scenario_file, scenario_dir)
    if scenario_config is None:
        raise FileNotFoundError(f"场景文件加载失败: {scenario_file}")
    scenario_config = dict(scenario_config)

    if coop_start_config and coop_start_config.lower() not in ("none", ""):
        coop_cfg = _load_config_file(coop_start_config, scenario_dir)
        if coop_cfg is not None:
            src = _scenario_agents(coop_cfg)
            coop_pos = {f"agent{cid}": src[f"agent{cid}"]
                        for cid in cooperative_ids if f"agent{cid}" in src}
            agents = _scenario_agents(scenario_config)
            agents.update(coop_pos)
            scenario_config["agents"] = agents
            print(f"红车起点从 {coop_start_config} 覆盖")

    if auto_adv_positions:
        scenario_config["agents"] = make_adversarial_positions(
            scenario_config, cooperative_ids, adversarial_ids,
            margin_ratio=adv_margin_ratio,
        )

    # 打印最终布置，方便核对
    agents_final = _scenario_agents(scenario_config)
    print(f"固定场景: {scenario_file}")
    print(f"  goal_zone: {scenario_config.get('goal_zone', '<未声明，env 回退默认>')}")
    for aid_str, pos in sorted(agents_final.items()):
        role = "BLUE" if int(aid_str.replace("agent", "")) in adversarial_ids else "RED"
        print(f"  {aid_str} ({role}): x={pos.get('x'):.2f}, y={pos.get('y'):.2f}, "
              f"yaw={pos.get('yaw', 'auto')}")
    return scenario_config


# ============================================================
# 训练主函数
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    total_agents = args.num_cooperative + args.num_adversarial
    cooperative_ids = list(range(args.num_cooperative))
    adversarial_ids = list(range(args.num_cooperative, total_agents))

    print(f"固定场景对抗训练（KL-anchored + 风险增量）")
    print(f"  合作团队 (冻结): {cooperative_ids}")
    print(f"  对抗团队 (训练): {adversarial_ids}")
    print(f"  β_KL = {args.beta_kl}, threat_coef = {args.threat_coef}, "
          f"target ε = {args.target_blue_collision_rate}")

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

    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'training_scenarios')

    # 固定场景：训练循环外只 set 一次
    scenario_config = prepare_fixed_scenario_config(
        args.scenario, scenario_dir,
        cooperative_ids, adversarial_ids,
        args.coop_start_config, args.auto_adv_positions, args.adv_margin_ratio,
    )
    env.set_scenario(scenario_config)

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

    # 合作团队：冻结
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

    # 对抗团队：用预训练权重初始化
    adversarial_actors: Dict[int, Actor] = {}
    base_actors_frozen: Dict[int, Actor] = {}
    adversarial_actor_optims: Dict[int, optim.Optimizer] = {}
    for aid in adversarial_ids:
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim,
                      init_log_std=args.init_log_std).to(device)
        if aid in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[aid])
        elif cooperative_ids and cooperative_ids[0] in pretrained_actors_state:
            actor.load_state_dict(pretrained_actors_state[cooperative_ids[0]])
        adversarial_actors[aid] = actor
        adversarial_actor_optims[aid] = optim.Adam(actor.parameters(), lr=args.lr_actor)

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
    print(f"提示：第 1 次 update 的指标 ≈ 训练前基线（BLUE 还没学到对抗）")

    self_collision_coef = args.self_collision_coef_init
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
                    if f.startswith("fixed_scn_update_") and f.endswith(".pt"):
                        try:
                            n = int(f.replace("fixed_scn_update_", "").replace(".pt", ""))
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
            try:
                if 'rng_torch' in rck:
                    torch.set_rng_state(rck['rng_torch'])
                if 'rng_numpy' in rck:
                    np.random.set_state(rck['rng_numpy'])
                if 'rng_python' in rck:
                    import random as _rand
                    _rand.setstate(rck['rng_python'])
            except Exception as e:
                print(f"  ⚠️ RNG 恢复失败: {e}")
            print(f"  → 续训 update={start_update}")

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
            "scenario": args.scenario,
            "args": vars(args),
            "total_episodes": total_episodes,
            "best_adv_effectiveness": best_adv_effectiveness,
            "self_collision_coef": self_collision_coef,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            "rng_python": _rand.getstate(),
        }

    try:
        for update in range(start_update, args.updates + 1):
            cur_lr_a, cur_lr_c = _apply_lr_decay(update)

            buffer = RolloutBuffer(adversarial_ids)

            # 窗口统计
            win_coop_success: List[float] = []
            win_coop_collision: List[float] = []
            win_blue_collision: List[float] = []
            win_ep_lengths: List[int] = []
            win_adv_returns: Dict[int, List[float]] = {aid: [] for aid in adversarial_ids}

            # 奖励分量统计
            comp_base_safe: List[float] = []
            comp_threat: List[float] = []
            comp_self_coll: List[float] = []

            # 对抗有效性指标
            win_blue_threat_window_pct: List[float] = []   # 每 ep "BLUE 在场上施压"的步数占比
            win_red_speed_threat: List[float] = []         # 每 ep RED 被威胁时的平均速度
            win_red_speed_free: List[float] = []           # 每 ep RED 未被威胁时的平均速度
            win_avg_threat_score: List[float] = []         # 每 ep 平均威胁分（sum over blues）

            print(f"\n更新 {update}/{args.updates}, β_KL={args.beta_kl:.3f}, "
                  f"threat_coef={args.threat_coef:.2f}, self_coll={self_collision_coef:.2f}, "
                  f"lr_a={cur_lr_a:.2e}")

            # ── 收集 rollout ──
            for rollout_episode in range(args.rollout_episodes):
                obs_dict, info = env.reset()
                total_episodes += 1

                adv_ep_rewards = {aid: 0.0 for aid in adversarial_ids}
                ep_length = 0
                collision_sticky: Dict[int, bool] = {i: False for i in range(total_agents)}
                ep_threat_active_steps = 0
                ep_red_speeds_threat: List[float] = []
                ep_red_speeds_free: List[float] = []
                ep_total_threat: List[float] = []

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

                    # ── 对抗有效性指标采样 ──
                    threat_active = is_any_red_under_threat(
                        env, cooperative_ids, adversarial_ids,
                        args.threat_radius, args.threat_lateral,
                    )
                    if threat_active:
                        ep_threat_active_steps += 1
                    red_spd = avg_red_speed(env, cooperative_ids)
                    if red_spd is not None:
                        (ep_red_speeds_threat if threat_active else ep_red_speeds_free).append(red_spd)

                    # 风险增量奖励
                    threat_scores = compute_blue_threat_scores(
                        env, cooperative_ids, adversarial_ids,
                        threat_radius=args.threat_radius,
                        threat_lateral=args.threat_lateral,
                        threat_tau=args.threat_tau,
                    )
                    ep_total_threat.append(sum(threat_scores.values()))

                    for aid in step_cache.keys():
                        base_safe = extract_base_driving_safe(reward_details, aid)
                        threat = threat_scores.get(aid, 0.0)
                        collision_raw = extract_collision_component(reward_details, aid)
                        self_coll_term = self_collision_coef * collision_raw

                        combined = (
                            base_safe
                            + args.threat_coef * threat
                            + self_coll_term
                        )
                        adv_ep_rewards[aid] += combined
                        comp_base_safe.append(base_safe)
                        comp_threat.append(threat)
                        comp_self_coll.append(self_coll_term)
                        buffer.add(aid, Transition(
                            obs=step_cache[aid]["obs"],
                            global_state=step_cache[aid]["global_state"],
                            action_z=step_cache[aid]["action_z"],
                            log_prob=step_cache[aid]["logp"],
                            reward=combined,
                            done=float(dones.get(aid, False)),
                            terminated=float(terminated_info.get(aid, dones.get(aid, False))),
                            value=step_cache[aid]["value"],
                            scenario_name=args.scenario,
                        ))

                    if visualizer is not None:
                        visualizer.handle_events()
                        if not visualizer.running:
                            raise KeyboardInterrupt

                    obs_dict = next_obs
                    if all(dones.values()):
                        break

                # episode 终止统计
                coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
                coop_collision = any(collision_sticky[cid] for cid in cooperative_ids)
                blue_collision = any(collision_sticky[bid] for bid in adversarial_ids)

                last_gs = env.get_global_state().astype(np.float32)
                for aid in adversarial_ids:
                    buffer.mark_episode_end(aid)
                    own_reached = env.reached_target.get(aid, False)
                    own_crashed = collision_sticky[aid]
                    naturally_terminal = own_reached or own_crashed
                    buffer.add_episode_bootstrap(aid, None if naturally_terminal else last_gs)

                win_coop_success.append(1.0 if coop_success else 0.0)
                win_coop_collision.append(1.0 if coop_collision else 0.0)
                win_blue_collision.append(1.0 if blue_collision else 0.0)
                win_ep_lengths.append(ep_length)
                for aid in adversarial_ids:
                    win_adv_returns[aid].append(adv_ep_rewards[aid])

                # 对抗有效性 episode 级聚合
                win_blue_threat_window_pct.append(ep_threat_active_steps / max(ep_length, 1))
                if ep_red_speeds_threat:
                    win_red_speed_threat.append(float(np.mean(ep_red_speeds_threat)))
                if ep_red_speeds_free:
                    win_red_speed_free.append(float(np.mean(ep_red_speeds_free)))
                win_avg_threat_score.append(float(np.mean(ep_total_threat)) if ep_total_threat else 0.0)

            # ── 自适应自碰系数 ──
            blue_coll_rate = float(np.mean(win_blue_collision)) if win_blue_collision else 0.0
            if blue_coll_rate > args.target_blue_collision_rate:
                self_collision_coef *= 1.2
            else:
                self_collision_coef *= 0.95
            self_collision_coef = float(np.clip(self_collision_coef, 0.1, 10.0))

            # ── MAPPO + KL 锚定 更新 ──
            n_transitions = buffer.total_size()
            skip_update = n_transitions < args.min_transitions_per_update
            ppo_metrics = {"critic_loss": [], "explained_variance": 0.0}
            for aid in adversarial_ids:
                ppo_metrics.update({
                    f"actor_loss_{aid}": [], f"entropy_{aid}": [], f"kl_anchor_{aid}": [],
                    f"approx_kl_{aid}": [], f"clip_frac_{aid}": [],
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
                        c_n = critic_state_t.shape[0]
                        c_idx = np.arange(c_n); np.random.shuffle(c_idx)
                        epoch_critic_losses = []
                        for start in range(0, c_n, args.minibatch_size):
                            mb = c_idx[start: start + args.minibatch_size]
                            value_pred = critic(critic_state_t[mb], critic_aid_t[mb])
                            critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()
                            critic_optim.zero_grad()
                            critic_loss.backward()
                            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                            critic_optim.step()
                            epoch_critic_losses.append(float(critic_loss.item()))
                        ppo_metrics["critic_loss"].append(float(np.mean(epoch_critic_losses)))

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
                                nn.utils.clip_grad_norm_(adversarial_actors[aid].parameters(),
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
                            if args.target_kl > 0 and mean_kl > 1.5 * args.target_kl:
                                agent_stopped[aid] = True
                                agent_stop_epoch[aid] = epoch + 1

                    with torch.no_grad():
                        all_vp = critic(critic_state_t, critic_aid_t)
                        var_res = torch.var(critic_ret_t - all_vp)
                        var_tot = torch.var(critic_ret_t)
                        ppo_metrics["explained_variance"] = float(1.0 - var_res / (var_tot + 1e-8))

            # ===== TensorBoard 日志 =====
            if update % args.log_interval == 0 and len(win_coop_success) > 0:
                win_succ = float(np.mean(win_coop_success))
                win_coll = float(np.mean(win_coop_collision))
                win_b_coll = float(np.mean(win_blue_collision))
                win_threat_pct = float(np.mean(win_blue_threat_window_pct))
                win_threat_score = float(np.mean(win_avg_threat_score))
                adv_eff = 1.0 - win_succ
                red_spd_t = float(np.mean(win_red_speed_threat)) if win_red_speed_threat else 0.0
                red_spd_f = float(np.mean(win_red_speed_free)) if win_red_speed_free else 0.0
                slowdown = ((red_spd_f - red_spd_t) / red_spd_f) if red_spd_f > 0.1 else 0.0

                # ─── 对抗有效性核心指标 ───
                tb_writer.add_scalar('eval/coop_success_rate', win_succ, update)
                tb_writer.add_scalar('eval/coop_collision_rate', win_coll, update)
                tb_writer.add_scalar('eval/blue_collision_rate', win_b_coll, update)
                tb_writer.add_scalar('eval/adv_effectiveness', adv_eff, update)
                tb_writer.add_scalar('eval/blue_in_threat_window_pct', win_threat_pct, update)
                tb_writer.add_scalar('eval/avg_threat_score_per_ep', win_threat_score, update)
                tb_writer.add_scalar('eval/red_avg_speed_under_threat', red_spd_t, update)
                tb_writer.add_scalar('eval/red_avg_speed_free', red_spd_f, update)
                tb_writer.add_scalar('eval/red_relative_slowdown', slowdown, update)
                tb_writer.add_scalar('eval/avg_ep_length', float(np.mean(win_ep_lengths)), update)

                # 训练侧
                tb_writer.add_scalar('blue/self_collision_coef', self_collision_coef, update)
                tb_writer.add_scalar('episode/n_transitions', n_transitions, update)
                tb_writer.add_scalar('reward/base_safe',
                                     float(np.mean(comp_base_safe)) if comp_base_safe else 0.0, update)
                tb_writer.add_scalar('reward/threat',
                                     float(np.mean(comp_threat)) if comp_threat else 0.0, update)
                tb_writer.add_scalar('reward/self_collision',
                                     float(np.mean(comp_self_coll)) if comp_self_coll else 0.0, update)
                tb_writer.add_scalar('train/lr_actor', cur_lr_a, update)
                tb_writer.add_scalar('train/beta_kl', args.beta_kl, update)

                adv_return_means = []
                for aid in adversarial_ids:
                    if win_adv_returns[aid]:
                        m = float(np.mean(win_adv_returns[aid]))
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
                            tb_writer.add_scalar(f'ppo/kl_anchor_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"kl_anchor_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/approx_kl_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"approx_kl_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/clip_frac_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"clip_frac_{aid}"])), update)
                            tb_writer.add_scalar(f'ppo/entropy_agent{aid}',
                                                 float(np.mean(ppo_metrics[f"entropy_{aid}"])), update)

                print(f"更新 {update}: ⚔️ adv_eff={adv_eff:.2%} "
                      f"(coop_succ={win_succ:.2%}, coop_coll={win_coll:.2%}, blue_coll={win_b_coll:.2%})")
                print(f"          🎯 blue 施压步占比={win_threat_pct:.2%}, "
                      f"威胁分/ep={win_threat_score:.2f}, "
                      f"RED 减速={slowdown:.2%} ({red_spd_f:.1f}→{red_spd_t:.1f}m/s)")
                print(f"          reward: base={float(np.mean(comp_base_safe)) if comp_base_safe else 0:.3f}, "
                      f"threat={float(np.mean(comp_threat)) if comp_threat else 0:.3f}, "
                      f"self_coll={float(np.mean(comp_self_coll)) if comp_self_coll else 0:.3f}, "
                      f"ep_len={float(np.mean(win_ep_lengths)):.0f}, n_trans={n_transitions}")

            # 保存
            if update % args.save_interval == 0:
                ckpt_path = os.path.join(save_dir, f"fixed_scn_update_{update}.pt")
                torch.save(_build_ckpt_dict(update), ckpt_path)
                print(f"保存: {ckpt_path}")

            if len(win_coop_success) > 0:
                cur_eff = 1.0 - float(np.mean(win_coop_success))
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
    p = argparse.ArgumentParser(description="固定场景对抗有效性快速验证（KL anchored + risk reward）")

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

    # ─── 核心：固定场景 ───
    p.add_argument("--scenario", type=str, default="scenario_01_basic.json",
                   help="训练用的单一场景文件名（位于 data/training_scenarios/）")
    p.add_argument("--coop-start-config", type=str, default="none",
                   help="可选：用另一个场景的红车起点覆盖。默认 none，让 --scenario 自己说了算")
    p.add_argument("--auto-adv-positions", action="store_true", default=True)
    p.add_argument("--no-auto-adv-positions", dest="auto_adv_positions", action="store_false")
    p.add_argument("--adv-margin-ratio", type=float, default=0.15,
                   help="蓝车分布相对红车两端的内缩比例（0=均匀夹中，0.15 推荐避免太贴 leader）")

    # PPO / 训练
    p.add_argument("--updates", type=int, default=60,
                   help="60 update 足够看出趋势；要看完整收敛 → 150+")
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

    # KL anchor + 风险增量 + 自适应自碰
    p.add_argument("--beta-kl", type=float, default=0.1)
    p.add_argument("--threat-coef", type=float, default=2.0)
    p.add_argument("--threat-radius", type=float, default=15.0)
    p.add_argument("--threat-lateral", type=float, default=2.5)
    p.add_argument("--threat-tau", type=float, default=1.5)
    p.add_argument("--target-blue-collision-rate", type=float, default=0.05)
    p.add_argument("--self-collision-coef-init", type=float, default=1.0)

    # 其他
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-dir", type=str, default="./logs/adv_fixed_scenario")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=1)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
