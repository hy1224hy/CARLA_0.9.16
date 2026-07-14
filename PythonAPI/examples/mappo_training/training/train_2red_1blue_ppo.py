"""
2红车 + 1蓝车 对抗训练（PPO，§2.3 奖励初步验证）
====================================================
目标定位：这是通向最终 **2红+2蓝 MAPPO**（train_multi_adversarial_kl_risk.py）的踏脚石。
把 num_adversarial 从 1 改成 2 即得 2+2；结构、奖励、埋点全部对齐，方便平滑升级。

  - 红车（cooperative_ids，默认 [0,1]）：从预训练 ckpt 加载，**冻结**，正常驾驶受害者。
  - 蓝车（adversarial_ids，默认 [2]）：单车 **PPO** 训练的对抗者。
  - 奖励：全部走 `training/adversarial_reward.AdversarialRewardComputer`（§2.5 单一真相源），
          即 §2.3 的 r_impact + 门控 threat + base_safe − λ_t·c_self；KL 锚定在 actor_loss。
  - 埋点：对抗有效性 + PPO 健康值 + reward-hacking 仪表盘（§3）。

CTDE：AgentConditionedCritic（单蓝车时 num_agents=1，2+2 时 num_agents=2）。

运行（须先启动 CARLA 且预训练 ckpt 就绪）：
  cd PythonAPI/examples/mappo_training
  python -m training.train_2red_1blue_ppo --pretrained-ckpt <base_policy.pt> \
      --scenario scenario_03_far.json --updates 120 --render

设计文档：docs/adversarial_metrics_reward_design_and_plan_20260705.md §2.3/§2.5/§3
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import copy
import datetime as _dt
import os
import random as _rand
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, AgentConditionedCritic

from training.train_multi_adversarial_mappo_v2 import (
    set_seed, Transition, RolloutBuffer, action_tanh_to_env,
    _load_config_file, make_adversarial_positions, compute_gae_per_episode,
)
from training.train_multi_adversarial_kl_risk import gaussian_kl_diag
from training.train_adversarial_fixed_scenario import (
    is_any_red_under_threat, avg_red_speed, prepare_fixed_scenario_config,
)
from training.adversarial_reward import AdversarialRewardComputer, RewardConfig, IMPACT_KEYS

try:
    from visualization.mappo_visualizer import MAPPOVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2红+1蓝 对抗 PPO（§2.3 奖励）")
    # 环境
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town02")
    p.add_argument("--straight-road-length", type=float, default=100.0)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=1000)
    # 团队 / 场景
    p.add_argument("--pretrained-ckpt", type=str, required=True,
                   help="红车基础策略 ckpt（含 actors 权重）")
    p.add_argument("--num-cooperative", type=int, default=2, help="红车数（冻结）")
    p.add_argument("--num-adversarial", type=int, default=1,
                   help="蓝车数（PPO 训练）；改成 2 即 2+2")
    p.add_argument("--scenario", type=str, default="scenario_03_far.json")
    p.add_argument("--coop-start-config", type=str, default=None)
    p.add_argument("--auto-adv-positions", action="store_true", default=True)
    p.add_argument("--no-auto-adv-positions", dest="auto_adv_positions", action="store_false")
    p.add_argument("--adv-margin-ratio", type=float, default=0.0)
    # PPO
    p.add_argument("--updates", type=int, default=120)
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
    # KL 锚定
    p.add_argument("--beta-kl", type=float, default=0.1,
                   help="actor_loss 中 KL(π_blue‖π_anchor) 的权重")
    # 奖励权重（§2.3，映射到 RewardConfig）
    p.add_argument("--w-impact", type=float, default=1.0)
    p.add_argument("--w-threat", type=float, default=1.0)
    p.add_argument("--w-base-safe", type=float, default=0.3)
    p.add_argument("--threat-cap-per-episode", type=float, default=20.0)
    p.add_argument("--threat-radius", type=float, default=15.0)
    p.add_argument("--threat-lateral", type=float, default=2.5)
    p.add_argument("--threat-tau", type=float, default=1.5)
    p.add_argument("--threat-near", type=float, default=3.0)
    p.add_argument("--birth-protect-secs", type=float, default=3.0)
    # 自适应自碰约束
    p.add_argument("--self-collision-coef-init", type=float, default=1.0)
    p.add_argument("--self-collision-coef-max", type=float, default=10.0)
    p.add_argument("--target-blue-collision-rate", type=float, default=0.05)
    # 杂项
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save-dir", type=str, default="./logs/train_2red_1blue_ppo")
    p.add_argument("--save-interval", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)
    return p


def make_reward_config(args) -> RewardConfig:
    """CLI → RewardConfig（§2.3 权重/阈值单一来源）。"""
    return RewardConfig(
        w_impact=args.w_impact, w_threat=args.w_threat, w_base_safe=args.w_base_safe,
        threat_cap_per_episode=args.threat_cap_per_episode,
        threat_radius=args.threat_radius, threat_lateral=args.threat_lateral,
        threat_tau=args.threat_tau, threat_near=args.threat_near,
        timestep=args.timestep, birth_protect_secs=args.birth_protect_secs,
        self_collision_coef_init=args.self_collision_coef_init,
        self_collision_coef_max=args.self_collision_coef_max,
        target_blue_collision_rate=args.target_blue_collision_rate,
    )


def train(args):
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    total_agents = args.num_cooperative + args.num_adversarial
    cooperative_ids = list(range(args.num_cooperative))
    adversarial_ids = list(range(args.num_cooperative, total_agents))
    adv_index = {aid: i for i, aid in enumerate(adversarial_ids)}

    print(f"2红+{args.num_adversarial}蓝 对抗 PPO")
    print(f"  红车(冻结): {cooperative_ids}   蓝车(PPO): {adversarial_ids}")
    print(f"  β_KL={args.beta_kl}, w_impact={args.w_impact}, w_threat={args.w_threat}, "
          f"w_base_safe={args.w_base_safe}, threat_cap={args.threat_cap_per_episode}")

    env = CarlaMultiAgentEnv(
        host=args.host, port=args.port, town=args.town,
        num_agents=total_agents, straight_road_length=args.straight_road_length,
        timestep=args.timestep, max_steps=args.max_steps, spawn_vehicles=True,
        adversarial_agent_ids=adversarial_ids, reward_version="v2",
        disable_tensorboard=True, seed=args.seed,
        adversary_collision_ends_episode=False,
    )

    scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'training_scenarios')
    scenario_config = prepare_fixed_scenario_config(
        args.scenario, scenario_dir, cooperative_ids, adversarial_ids,
        args.coop_start_config, args.auto_adv_positions, args.adv_margin_ratio,
    )
    env.set_scenario(scenario_config)
    state_dim = env.get_global_state().shape[0]

    # ── 加载预训练红车策略 ──
    pretrained_path = args.pretrained_ckpt
    while pretrained_path and pretrained_path[0] in ('\\', '/'):
        pretrained_path = pretrained_path[1:]
    pretrained_path = pretrained_path.replace('\\', os.sep).replace('/', os.sep)
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {pretrained_path}")
    print(f"加载预训练: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    pretrained_actors_state = ckpt.get("actors", {})

    # 红车冻结
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

    # 蓝车：预训练权重初始化 mu，显式恢复 init_log_std 保留初始探索
    adversarial_actors: Dict[int, Actor] = {}
    base_actors_frozen: Dict[int, Actor] = {}
    adversarial_actor_optims: Dict[int, optim.Optimizer] = {}
    for aid in adversarial_ids:
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim,
                      init_log_std=args.init_log_std).to(device)
        src = aid if aid in pretrained_actors_state else cooperative_ids[0]
        actor.load_state_dict(pretrained_actors_state[src])
        with torch.no_grad():
            actor.log_std.fill_(float(args.init_log_std))
        adversarial_actors[aid] = actor
        adversarial_actor_optims[aid] = optim.Adam(actor.parameters(), lr=args.lr_actor)
        anchor = copy.deepcopy(actor)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad = False
        base_actors_frozen[aid] = anchor

    critic = AgentConditionedCritic(state_dim=state_dim, num_agents=len(adversarial_ids),
                                    hidden=args.hidden_dim).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)
    init_lr_actor, init_lr_critic = args.lr_actor, args.lr_critic

    def _apply_lr_decay(cur_update: int):
        if not args.lr_decay:
            return init_lr_actor, init_lr_critic
        frac = max(0.05, 1.0 - (cur_update - 1) / max(args.updates - 1, 1))
        na, nc = init_lr_actor * frac, init_lr_critic * frac
        for aid in adversarial_ids:
            for g in adversarial_actor_optims[aid].param_groups:
                g['lr'] = na
        for g in critic_optim.param_groups:
            g['lr'] = nc
        return na, nc

    # ── 奖励计算器（§2.5 单一真相源）──
    reward_computer = AdversarialRewardComputer(
        cooperative_ids, adversarial_ids, make_reward_config(args))

    # ── 断点续训：恢复蓝车 actor / critic / optimizer / λ_t / 起始 update ──
    start_update = 0
    best_adv_effectiveness = -1.0
    if args.resume:
        resume_path = args.resume.replace('\\', os.sep).replace('/', os.sep)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"续训 checkpoint 不存在: {resume_path}")
        print(f"断点续训: {resume_path}")
        rc = torch.load(resume_path, map_location=device, weights_only=False)
        saved_adv = rc.get("adversarial_ids", adversarial_ids)
        if list(saved_adv) != list(adversarial_ids):
            raise ValueError(
                f"续训 ckpt 的蓝车 {saved_adv} 与当前 {adversarial_ids} 不一致；"
                f"检查 --num-cooperative/--num-adversarial 是否与训练时相同")
        adv_state = rc.get("adversarial_actors", {})
        adv_opt_state = rc.get("adversarial_actor_optims", {})
        for aid in adversarial_ids:
            adversarial_actors[aid].load_state_dict(adv_state[aid])
            if aid in adv_opt_state:
                adversarial_actor_optims[aid].load_state_dict(adv_opt_state[aid])
        critic.load_state_dict(rc["critic"])
        if "critic_optim" in rc:
            critic_optim.load_state_dict(rc["critic_optim"])
        reward_computer.self_collision_coef = float(
            rc.get("self_collision_coef", reward_computer.self_collision_coef))
        start_update = int(rc.get("update", 0))
        best_adv_effectiveness = float(rc.get("best_adv_effectiveness", -1.0))
        print(f"  已恢复：从 update {start_update} 继续 → 目标 {args.updates}"
              f"（λ_t={reward_computer.self_collision_coef:.2f}, "
              f"best_adv_eff={best_adv_effectiveness:.2%}）")
        if start_update >= args.updates:
            print(f"  ⚠ 起始 update {start_update} ≥ --updates {args.updates}，"
                  f"不会再训练；请调大 --updates")

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
    print("提示：update 1 的指标 ≈ 训练前基线（蓝车还没学到对抗）")

    _run_training_loop(args, env, device, cooperative_ids, adversarial_ids, adv_index,
                       cooperative_actors, adversarial_actors, base_actors_frozen,
                       adversarial_actor_optims, critic, critic_optim,
                       _apply_lr_decay, reward_computer, visualizer, tb_writer, save_dir,
                       start_update=start_update,
                       best_adv_effectiveness=best_adv_effectiveness)


def _rollout(args, env, device, cooperative_ids, adversarial_ids, adv_index,
             cooperative_actors, adversarial_actors, critic, reward_computer,
             visualizer, buffer, acc):
    """收集一个 rollout 窗口，写入 buffer + 累积统计到 acc（dict of lists）。"""
    total_agents = len(cooperative_ids) + len(adversarial_ids)
    for _ in range(args.rollout_episodes):
        obs_dict, info = env.reset()
        active_mask = dict(info['active_mask'])
        episode_end_reason = info['episode_end_reason']
        reward_computer.reset_episode()
        adv_ep_rewards = {aid: 0.0 for aid in adversarial_ids}
        ep_len = 0
        collision_sticky: Dict[int, bool] = {i: False for i in range(total_agents)}
        ep_threat_steps = 0
        ep_red_spd_threat: List[float] = []
        ep_red_spd_free: List[float] = []
        # 该回合每辆蓝车在 buffer 中的最后一条 transition 下标（供 timeout bonus 回填）
        last_idx: Dict[int, int] = {aid: -1 for aid in adversarial_ids}

        for step in range(args.max_steps):
            ep_len += 1
            action_dict: Dict[int, np.ndarray] = {}
            step_cache: Dict[int, Dict] = {}
            global_state_np = env.get_global_state().astype(np.float32)
            global_state_t = torch.as_tensor(global_state_np, dtype=torch.float32,
                                             device=device).unsqueeze(0)
            with torch.no_grad():
                agent_values = {aid: float(critic(global_state_t, adv_index[aid]).item())
                                for aid in adversarial_ids if active_mask.get(aid, False)}
                for cid in cooperative_ids:
                    if not active_mask.get(cid, False) or cid not in obs_dict:
                        continue
                    obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    a = cooperative_actors[cid].get_deterministic_action(obs_t)\
                        .squeeze(0).cpu().numpy()
                    action_dict[cid] = action_tanh_to_env(a)
                for aid in adversarial_ids:
                    if not active_mask.get(aid, False) or aid not in obs_dict:
                        continue
                    obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    a_t, logp_t, z_t = adversarial_actors[aid].sample_action_z(obs_t)
                    action_dict[aid] = action_tanh_to_env(a_t.squeeze(0).cpu().numpy())
                    step_cache[aid] = {
                        "obs": obs_dict[aid].copy(), "global_state": global_state_np,
                        "action_z": z_t.squeeze(0).cpu().numpy(),
                        "logp": float(logp_t.item()), "value": agent_values[aid],
                    }

            next_obs, rewards, dones, info = env.step(action_dict)
            terminated_info = info.get('terminated', {})
            reward_details = info.get('reward_details', {}) or {}
            step_collisions = info.get('collisions', {}) or {}
            next_active_mask = info['active_mask']
            episode_end_reason = info['episode_end_reason']
            pre_collision_kinematics = info.get('pre_collision_kinematics', {}) or {}
            for i in range(total_agents):
                if step_collisions.get(i, False):
                    collision_sticky[i] = True

            # 对抗有效性采样（几何口径，供 TB 减速代理）
            threat_active = is_any_red_under_threat(
                env, cooperative_ids, adversarial_ids, args.threat_radius, args.threat_lateral)
            if threat_active:
                ep_threat_steps += 1
            red_spd = avg_red_speed(env, cooperative_ids)
            if red_spd is not None:
                (ep_red_spd_threat if threat_active else ep_red_spd_free).append(red_spd)

            # ── §2.3 奖励（单一真相源）──
            combined_rewards, details = reward_computer.step(
                env, reward_details, action_dict, step, step_collisions,
                kinematic_snapshots=pre_collision_kinematics)

            for aid in step_cache.keys():
                combined = combined_rewards.get(aid, 0.0)
                d = details.get(aid, {})
                adv_ep_rewards[aid] += combined
                acc["base_safe"].append(d.get("base_safe", 0.0))
                acc["threat_raw"].append(d.get("threat_raw", 0.0))
                acc["threat_gated"].append(d.get("threat_gated", 0.0))
                acc["self_coll"].append(d.get("self_coll_term", 0.0))
                acc["impact"].append(d.get("impact", 0.0))
                if d.get("n_reds_in_window", 0) > 0:
                    acc["gate_on_when_in_window"].append(d.get("threat_gate_on", 0.0))
                for k in IMPACT_KEYS:
                    acc[f"impact_{k}"].append(d.get(f"impact_{k}", 0.0))
                buffer.add(aid, Transition(
                    obs=step_cache[aid]["obs"], global_state=step_cache[aid]["global_state"],
                    action_z=step_cache[aid]["action_z"], log_prob=step_cache[aid]["logp"],
                    reward=combined, done=float(dones.get(aid, False)),
                    terminated=float(terminated_info.get(aid, dones.get(aid, False))),
                    value=step_cache[aid]["value"], scenario_name=args.scenario,
                ))
                last_idx[aid] = len(buffer.data[aid]) - 1

            if visualizer is not None:
                visualizer.handle_events()
                if not visualizer.running:
                    raise KeyboardInterrupt
            obs_dict = next_obs
            active_mask = dict(next_active_mask)
            if episode_end_reason != "running":
                break

        # 回合结束
        coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
        coop_collision = any(collision_sticky[cid] for cid in cooperative_ids)
        blue_collision = any(collision_sticky[bid] for bid in adversarial_ids)
        timed_out = episode_end_reason == "max_steps"

        # 归因超时 bonus 回填到该回合蓝车最后一步
        bonus, _binfo = reward_computer.finalize_episode(timed_out)
        for aid in adversarial_ids:
            b = bonus.get(aid, 0.0)
            if b != 0.0 and last_idx[aid] >= 0:
                buffer.data[aid][last_idx[aid]].reward += b
                adv_ep_rewards[aid] += b
            acc[f"impact_timeout"].append(b)

        last_gs = env.get_global_state().astype(np.float32)
        for aid in adversarial_ids:
            buffer.mark_episode_end(aid)
            naturally_terminal = env.reached_target.get(aid, False) or collision_sticky[aid]
            buffer.add_episode_bootstrap(aid, None if naturally_terminal else last_gs)
            acc["adv_return"][aid].append(adv_ep_rewards[aid])

        acc["coop_success"].append(1.0 if coop_success else 0.0)
        acc["coop_collision"].append(1.0 if coop_collision else 0.0)
        acc["blue_collision"].append(1.0 if blue_collision else 0.0)
        acc["red_timeout"].append(1.0 if timed_out else 0.0)
        acc["ep_len"].append(ep_len)
        acc["threat_window_pct"].append(ep_threat_steps / max(ep_len, 1))
        if ep_red_spd_threat:
            acc["red_spd_threat"].append(float(np.mean(ep_red_spd_threat)))
        if ep_red_spd_free:
            acc["red_spd_free"].append(float(np.mean(ep_red_spd_free)))


def _ppo_update(args, device, adversarial_ids, adv_index, adversarial_actors,
                base_actors_frozen, adversarial_actor_optims, critic, critic_optim,
                buffer):
    """PPO + KL 锚定更新。返回 (ppo_metrics, n_transitions, skipped)。"""
    ppo_metrics: Dict = {"critic_loss": [], "explained_variance": 0.0}
    for aid in adversarial_ids:
        ppo_metrics.update({f"actor_loss_{aid}": [], f"entropy_{aid}": [],
                            f"kl_anchor_{aid}": [], f"approx_kl_{aid}": [],
                            f"clip_frac_{aid}": []})
    n_transitions = buffer.total_size()
    if n_transitions < args.min_transitions_per_update:
        return ppo_metrics, n_transitions, True

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
            rewards=reward_arr, values=value_arr, dones=done_arr, terminateds=terminated_arr,
            boundaries=boundaries, gamma=args.gamma, gae_lambda=args.gae_lambda,
            bootstrap_values=bootstrap_values)
        per_agent_tensors[aid] = {
            "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
            "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
            "old_logp": torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device),
            "adv": torch.as_tensor(adv_arr, dtype=torch.float32, device=device),
        }
        critic_states.append(state_arr)
        critic_returns.append(ret_arr)
        critic_agent_ids.append(np.full(len(ret_arr), adv_index[aid], dtype=np.int64))

    if len(critic_states) == 0:
        return ppo_metrics, n_transitions, True
    critic_state_t = torch.as_tensor(np.concatenate(critic_states, 0), dtype=torch.float32, device=device)
    critic_ret_t = torch.as_tensor(np.concatenate(critic_returns, 0), dtype=torch.float32, device=device)
    critic_aid_t = torch.as_tensor(np.concatenate(critic_agent_ids, 0), dtype=torch.long, device=device)

    agent_stopped = {aid: False for aid in adversarial_ids}
    for epoch in range(args.ppo_epochs):
        c_n = critic_state_t.shape[0]
        c_idx = np.arange(c_n); np.random.shuffle(c_idx)
        for start in range(0, c_n, args.minibatch_size):
            mb = c_idx[start: start + args.minibatch_size]
            value_pred = critic(critic_state_t[mb], critic_aid_t[mb])
            critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()
            critic_optim.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optim.step()
            ppo_metrics["critic_loss"].append(float(critic_loss.item()))

        for aid in adversarial_ids:
            if aid not in per_agent_tensors or agent_stopped[aid]:
                continue
            a_obs = per_agent_tensors[aid]["obs"]; a_act = per_agent_tensors[aid]["act"]
            a_oldlogp = per_agent_tensors[aid]["old_logp"]; a_adv = per_agent_tensors[aid]["adv"]
            a_n = a_obs.shape[0]; a_idx = np.arange(a_n); np.random.shuffle(a_idx)
            ep_kl_p, ep_losses, ep_ent, ep_cf, ep_kl_anc = [], [], [], [], []
            for start in range(0, a_n, args.minibatch_size):
                mb = a_idx[start: start + args.minibatch_size]
                new_logp, entropy = adversarial_actors[aid].evaluate_actions_z(a_obs[mb], a_act[mb])
                logratio = new_logp - a_oldlogp[mb]; ratio = torch.exp(logratio)
                surr1 = ratio * a_adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * a_adv[mb]
                ppo_loss = -torch.min(surr1, surr2).mean()
                mu_a, ls_a = adversarial_actors[aid].forward(a_obs[mb])
                with torch.no_grad():
                    mu_b, ls_b = base_actors_frozen[aid].forward(a_obs[mb])
                kl_anchor = gaussian_kl_diag(mu_a, ls_a, mu_b, ls_b)
                actor_loss = ppo_loss - args.ent_coef * entropy.mean() + args.beta_kl * kl_anchor
                adversarial_actor_optims[aid].zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(adversarial_actors[aid].parameters(), args.max_grad_norm)
                adversarial_actor_optims[aid].step()
                ep_losses.append(float(actor_loss.item())); ep_ent.append(float(entropy.mean().item()))
                ep_kl_anc.append(float(kl_anchor.item()))
                with torch.no_grad():
                    ep_kl_p.append(float(((ratio - 1.0) - logratio).mean().item()))
                    ep_cf.append(float(((ratio - 1.0).abs() > args.clip_eps).float().mean().item()))
            ppo_metrics[f"actor_loss_{aid}"].append(float(np.mean(ep_losses)))
            ppo_metrics[f"entropy_{aid}"].append(float(np.mean(ep_ent)))
            ppo_metrics[f"kl_anchor_{aid}"].append(float(np.mean(ep_kl_anc)))
            mean_kl = float(np.mean(ep_kl_p))
            ppo_metrics[f"approx_kl_{aid}"].append(mean_kl)
            ppo_metrics[f"clip_frac_{aid}"].append(float(np.mean(ep_cf)))
            if args.target_kl > 0 and mean_kl > 1.5 * args.target_kl:
                agent_stopped[aid] = True

    with torch.no_grad():
        all_vp = critic(critic_state_t, critic_aid_t)
        var_res = torch.var(critic_ret_t - all_vp); var_tot = torch.var(critic_ret_t)
        ppo_metrics["explained_variance"] = float(1.0 - var_res / (var_tot + 1e-8))
    return ppo_metrics, n_transitions, False


def _fresh_acc(adversarial_ids) -> Dict:
    acc = {k: [] for k in [
        "base_safe", "threat_raw", "threat_gated", "self_coll", "impact",
        "gate_on_when_in_window", "coop_success", "coop_collision", "blue_collision",
        "red_timeout", "ep_len", "threat_window_pct", "red_spd_threat", "red_spd_free"]}
    for k in IMPACT_KEYS:
        acc[f"impact_{k}"] = []
    acc["adv_return"] = {aid: [] for aid in adversarial_ids}
    return acc


def _log_tb(tb_writer, update, args, acc, ppo_metrics, n_transitions, skipped,
            adversarial_ids, adversarial_actors, self_collision_coef, cur_lr_a):
    def m(key):
        return float(np.mean(acc[key])) if acc[key] else 0.0
    win_succ = m("coop_success")
    red_spd_t = m("red_spd_threat"); red_spd_f = m("red_spd_free")
    slowdown = ((red_spd_f - red_spd_t) / red_spd_f) if red_spd_f > 0.1 else 0.0
    threat_pct = m("threat_window_pct")

    # ── 对抗有效性 ──
    tb_writer.add_scalar('eval/coop_success_rate', win_succ, update)
    tb_writer.add_scalar('eval/coop_collision_rate', m("coop_collision"), update)
    tb_writer.add_scalar('eval/blue_collision_rate', m("blue_collision"), update)
    tb_writer.add_scalar('eval/red_timeout_rate', m("red_timeout"), update)
    tb_writer.add_scalar('eval/adv_effectiveness', 1.0 - win_succ, update)
    tb_writer.add_scalar('eval/blue_in_threat_window_pct', threat_pct, update)
    tb_writer.add_scalar('eval/red_avg_speed_under_threat', red_spd_t, update)
    tb_writer.add_scalar('eval/red_avg_speed_free', red_spd_f, update)
    tb_writer.add_scalar('eval/red_relative_slowdown', slowdown, update)
    tb_writer.add_scalar('eval/avg_ep_length', m("ep_len"), update)

    # ── 奖励分项 ──
    tb_writer.add_scalar('reward/base_safe', m("base_safe"), update)
    tb_writer.add_scalar('reward/threat_raw', m("threat_raw"), update)
    tb_writer.add_scalar('reward/threat_gated', m("threat_gated"), update)
    tb_writer.add_scalar('reward/impact', m("impact"), update)
    tb_writer.add_scalar('reward/self_collision', m("self_coll"), update)
    tb_writer.add_scalar('train/self_collision_coef', self_collision_coef, update)
    tb_writer.add_scalar('train/lr_actor', cur_lr_a, update)
    tb_writer.add_scalar('train/beta_kl', args.beta_kl, update)
    tb_writer.add_scalar('episode/n_transitions', n_transitions, update)
    for aid in adversarial_ids:
        if acc["adv_return"][aid]:
            tb_writer.add_scalar(f'adv/return_agent_{aid}',
                                 float(np.mean(acc["adv_return"][aid])), update)
        tb_writer.add_scalar(f'train/log_std_agent{aid}',
                             float(adversarial_actors[aid].log_std.mean().item()), update)

    # ── reward-hacking 仪表盘（§3）──
    # 事件计数（每 update 求和）：kamikaze/induced/slowdown/swerve/timeout
    for k in IMPACT_KEYS:
        tb_writer.add_scalar(f'hack/impact_{k}_per_update',
                             float(np.sum(acc[f"impact_{k}"])), update)
    ep_count = max(len(acc["coop_success"]), 1)
    tb_writer.add_scalar('hack/kamikaze_rate',
                         float(np.sum(acc["impact_kamikaze"])) / ep_count, update)
    # H2 占位刷分：在窗口内的步里门控真正打开的比例（低 = 占位多、红车没反应）
    tb_writer.add_scalar('hack/threat_gate_on_ratio', m("gate_on_when_in_window"), update)
    # H2 交叉验证：施压占比高但 RED 没减速 = 可疑
    tb_writer.add_scalar('hack/threat_pct_minus_slowdown', threat_pct - slowdown, update)

    # ── PPO 健康值 ──
    if not skipped and ppo_metrics["critic_loss"]:
        tb_writer.add_scalar('ppo/critic_loss', float(np.mean(ppo_metrics["critic_loss"])), update)
        tb_writer.add_scalar('ppo/explained_variance', ppo_metrics["explained_variance"], update)
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

    print(f"更新 {update}: ⚔️ adv_eff={1.0 - win_succ:.2%} "
          f"(coop_succ={win_succ:.2%}, blue_coll={m('blue_collision'):.2%}, "
          f"red_timeout={m('red_timeout'):.2%})")
    print(f"          🎯 施压步占比={threat_pct:.2%}, 门控开启比={m('gate_on_when_in_window'):.2%}, "
          f"RED 减速={slowdown:.2%} ({red_spd_f:.1f}→{red_spd_t:.1f}m/s), "
          f"kamikaze/ep={float(np.sum(acc['impact_kamikaze'])) / ep_count:.2f}")
    print(f"          reward: base={m('base_safe'):.3f}, threat_gated={m('threat_gated'):.3f}, "
          f"impact={m('impact'):.3f}, self_coll={m('self_coll'):.3f}, n_trans={n_transitions}")


def _run_training_loop(args, env, device, cooperative_ids, adversarial_ids, adv_index,
                       cooperative_actors, adversarial_actors, base_actors_frozen,
                       adversarial_actor_optims, critic, critic_optim,
                       _apply_lr_decay, reward_computer, visualizer, tb_writer, save_dir,
                       start_update=0, best_adv_effectiveness=-1.0):

    def _save(update, tag):
        path = os.path.join(save_dir, f"{tag}.pt")
        torch.save({
            "update": update,
            "adversarial_actors": {aid: adversarial_actors[aid].state_dict() for aid in adversarial_ids},
            "critic": critic.state_dict(),
            "adversarial_actor_optims": {aid: adversarial_actor_optims[aid].state_dict() for aid in adversarial_ids},
            "critic_optim": critic_optim.state_dict(),
            "cooperative_ids": cooperative_ids, "adversarial_ids": adversarial_ids,
            "self_collision_coef": reward_computer.self_collision_coef,
            "best_adv_effectiveness": best_adv_effectiveness,
            "args": vars(args),
        }, path)
        return path

    try:
        for update in range(start_update + 1, args.updates + 1):
            cur_lr_a, cur_lr_c = _apply_lr_decay(update)
            buffer = RolloutBuffer(adversarial_ids)
            acc = _fresh_acc(adversarial_ids)

            print(f"\n更新 {update}/{args.updates}, β_KL={args.beta_kl:.3f}, "
                  f"self_coll={reward_computer.self_collision_coef:.2f}, lr_a={cur_lr_a:.2e}")

            _rollout(args, env, device, cooperative_ids, adversarial_ids, adv_index,
                     cooperative_actors, adversarial_actors, critic, reward_computer,
                     visualizer, buffer, acc)

            # 自适应自碰约束（对偶更新）
            blue_coll_rate = float(np.mean(acc["blue_collision"])) if acc["blue_collision"] else 0.0
            self_collision_coef = reward_computer.update_dual(blue_coll_rate)

            ppo_metrics, n_transitions, skipped = _ppo_update(
                args, device, adversarial_ids, adv_index, adversarial_actors,
                base_actors_frozen, adversarial_actor_optims, critic, critic_optim, buffer)
            if skipped:
                print(f"  [跳过 PPO] transition 数 {n_transitions} < {args.min_transitions_per_update}")

            if update % args.log_interval == 0 and acc["coop_success"]:
                _log_tb(tb_writer, update, args, acc, ppo_metrics, n_transitions, skipped,
                        adversarial_ids, adversarial_actors, self_collision_coef, cur_lr_a)

            adv_eff = 1.0 - (float(np.mean(acc["coop_success"])) if acc["coop_success"] else 0.0)
            if adv_eff > best_adv_effectiveness:
                best_adv_effectiveness = adv_eff
                _save(update, "best")
            if update % args.save_interval == 0:
                _save(update, f"ckpt_{update}")
        _save(args.updates, "final")
        print(f"\n训练完成。best_adv_effectiveness={best_adv_effectiveness:.2%}")
    except KeyboardInterrupt:
        interrupted_at = locals().get("update", start_update)
        print(f"\n中断，保存当前进度（update={interrupted_at}）…")
        _save(interrupted_at, "interrupted")
    finally:
        tb_writer.close()
        try:
            env.close()
        except Exception:
            pass


def main():
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()





