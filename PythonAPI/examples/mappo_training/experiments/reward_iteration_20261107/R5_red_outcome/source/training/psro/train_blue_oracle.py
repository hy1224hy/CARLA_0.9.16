"""
PSRO 内层 BLUE oracle 训练（MAPPO + KL 锚定 + 风险增量奖励）
==============================================================
基于 train_adversarial_fixed_scenario.py 改造，关键差异：

  ① 不再加载单个 pretrained ckpt → 接受 RedTeamPool + σ_red，每个 episode 采样 RED
  ② 不暴露独立 CLI → 作为 PSRO runner 的内层调用，函数 train_blue_oracle(...)
  ③ 返回 BLUE 团队 ckpt dict 而不仅保存到磁盘（便于 runner 立刻评估）

保留：
  - MAPPO + CTDE（AgentConditionedCritic）
  - KL anchor 到 base_actors_frozen
  - 风险增量奖励 + 自适应自碰惩罚（PID-I）
  - target_kl 早停 + lr 衰减 + pre-tanh z PPO

设计文档：docs/对抗训练_MAPPO_PSRO_设计计划.md §4.1
"""
import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))

import copy
import math
import os
import random as _rand
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor, AgentConditionedCritic

from training.train_multi_adversarial_mappo_v2 import (
    Transition,
    RolloutBuffer,
    action_tanh_to_env,
    _scenario_agents,
    _load_config_file,
    make_adversarial_positions,
    compute_gae_per_episode,
)
from training.train_multi_adversarial_kl_risk import (
    compute_blue_threat_scores,
    extract_base_driving_safe,
    extract_collision_component,
    gaussian_kl_diag,
)
from training.train_adversarial_fixed_scenario import (
    is_any_red_under_threat,
    avg_red_speed,
    prepare_fixed_scenario_config,
)
from utils.adversarial_metrics import extract_kinematics

from training.psro.population import RedTeamPool


def _build_blue_team_ckpt(adversarial_actors, critic, adv_actor_optims, critic_optim,
                          adversarial_ids, sigma_red, psro_iter, args_vars,
                          stats: Optional[Dict] = None) -> Dict:
    """组装 BLUE 团队 ckpt（保存格式与 BlueTeamPool.get_actors 兼容）。"""
    return {
        "adversarial_actors": {
            int(bid): adversarial_actors[bid].state_dict() for bid in adversarial_ids
        },
        "critic": critic.state_dict(),
        "adversarial_actor_optims": {
            int(bid): adv_actor_optims[bid].state_dict() for bid in adversarial_ids
        },
        "critic_optim": critic_optim.state_dict(),
        "log_std": {
            int(bid): float(adversarial_actors[bid].log_std.detach().mean().item())
            for bid in adversarial_ids
        },
        "adversarial_ids": list(adversarial_ids),
        "trained_against_sigma_red": [float(x) for x in np.asarray(sigma_red).flatten()],
        "psro_iter": int(psro_iter),
        "args": dict(args_vars),
        "stats": stats or {},
    }


def train_blue_oracle(
    env: CarlaMultiAgentEnv,
    red_pool: RedTeamPool,
    sigma_red: np.ndarray,
    adversarial_ids: List[int],
    cooperative_ids: List[int],
    scenario_config: Dict,
    args,
    psro_iter: int,
    base_actor_init_state: Optional[Dict[int, dict]] = None,
    tb_writer=None,
    tb_step_offset: int = 0,
) -> Dict:
    """训练单个 BLUE MAPPO oracle 对抗 RED 池上的 σ_red 混合分布。

    参数:
      env                    : 已配置好的 CarlaMultiAgentEnv（共享给所有 PSRO 轮）
      red_pool               : RedTeamPool
      sigma_red              : np.ndarray, 概率分布，shape (len(red_pool),)
      adversarial_ids        : 蓝车 agent id 列表，例如 [2, 3]
      cooperative_ids        : 红车 agent id 列表，例如 [0, 1]
      scenario_config        : 已经过 make_adversarial_positions 处理的场景配置 dict
      args                   : 训练超参 Namespace（含 PPO/KL/threat 等所有字段）
      psro_iter              : 当前 PSRO 轮次，记入 ckpt + tb tag
      base_actor_init_state  : 可选，BLUE actor 初始化权重（默认用 red_pool[0] 的合作策略）
      tb_writer              : 可选 SummaryWriter
      tb_step_offset         : tb 全局 step 偏移（让多轮 PSRO 在同一 TB 里连续）

    返回:
      blue_team_ckpt: dict — BLUE 团队权重+元信息，由调用方(psro_runner)负责落盘
                       （本函数只构造并返回，不写磁盘；保存见 BlueTeamPool.add）
    """
    device = next(iter(red_pool.get_actors(0).values())).net[0].weight.device

    if sigma_red is None or len(sigma_red) != len(red_pool):
        raise ValueError(f"sigma_red 长度 {None if sigma_red is None else len(sigma_red)} "
                         f"必须等于 RED 池大小 {len(red_pool)}")
    sigma_red = np.asarray(sigma_red, dtype=np.float64)
    sigma_red = sigma_red / sigma_red.sum()

    total_agents = len(cooperative_ids) + len(adversarial_ids)

    # ── BLUE actor 初始化：用 red_pool[0] 的合作策略（或显式传入）──
    if base_actor_init_state is None:
        sample_actors = red_pool.get_actors(0)
        # 取合作 id 0 的 state 作为蓝车 init
        base_state_for_blue = sample_actors[cooperative_ids[0]].state_dict()
    else:
        base_state_for_blue = base_actor_init_state[cooperative_ids[0]]

    adversarial_actors: Dict[int, Actor] = {}
    base_actors_frozen: Dict[int, Actor] = {}
    adversarial_actor_optims: Dict[int, optim.Optimizer] = {}
    for bid in adversarial_ids:
        actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim,
                      init_log_std=args.init_log_std).to(device)
        actor.load_state_dict(base_state_for_blue)
        # base_state_for_blue 携带 RED 收敛后的 log_std（通常很小，探索几乎为 0）。
        # load 会覆盖 Actor(init_log_std=...) 的设定，使 --init-log-std 失效、BLUE 起步
        # 几乎无探索 → 难以发现对抗行为。这里显式恢复 init_log_std，让 BLUE 保留
        # "自然驾驶均值 mu + 受控初始探索 std"。anchor 在其后 deepcopy，故 KL 锚点
        # 的 std 也同步为 init_log_std（锚均值仍是自然驾驶策略）。
        with torch.no_grad():
            actor.log_std.fill_(float(args.init_log_std))
        adversarial_actors[bid] = actor
        adversarial_actor_optims[bid] = optim.Adam(actor.parameters(), lr=args.lr_actor)

        anchor = copy.deepcopy(actor)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad = False
        base_actors_frozen[bid] = anchor

    # CTDE critic（仅 BLUE 团队）
    state_dim = env.get_global_state().shape[0]
    adv_index = {bid: i for i, bid in enumerate(adversarial_ids)}
    critic = AgentConditionedCritic(state_dim=state_dim, num_agents=len(adversarial_ids),
                                    hidden=args.hidden_dim).to(device)
    critic_optim = optim.Adam(critic.parameters(), lr=args.lr_critic)

    init_lr_actor, init_lr_critic = args.lr_actor, args.lr_critic

    def _apply_lr_decay(cur_update: int):
        if not getattr(args, 'lr_decay', True):
            return init_lr_actor, init_lr_critic
        frac = max(0.05, 1.0 - (cur_update - 1) / max(args.inner_updates - 1, 1))
        new_lr_a, new_lr_c = init_lr_actor * frac, init_lr_critic * frac
        for bid in adversarial_ids:
            for g in adversarial_actor_optims[bid].param_groups:
                g['lr'] = new_lr_a
        for g in critic_optim.param_groups:
            g['lr'] = new_lr_c
        return new_lr_a, new_lr_c

    # 自适应自碰系数（PID-I）
    self_collision_coef = float(getattr(args, 'self_collision_coef_init', 1.0))
    target_blue_collision_rate = float(getattr(args, 'target_blue_collision_rate', 0.05))

    env.set_scenario(scenario_config)

    # ── 训练主循环 ──
    last_window_coop_success = []
    last_window_blue_collision = []
    last_window_threat_pct = []

    for update in range(1, args.inner_updates + 1):
        cur_lr_a, cur_lr_c = _apply_lr_decay(update)

        buffer = RolloutBuffer(adversarial_ids)

        win_coop_success: List[float] = []
        win_blue_collision: List[float] = []
        win_ep_lengths: List[int] = []
        win_adv_returns: Dict[int, List[float]] = {bid: [] for bid in adversarial_ids}
        win_blue_threat_pct: List[float] = []
        win_avg_threat_score: List[float] = []
        # 红车结局拆分（补齐 TB 缺口：原来只有 coop_success + blue_collision）
        win_red_collision: List[float] = []
        win_red_timeout: List[float] = []
        # 减速代理：按"是否在威胁窗口内"分桶收集红车速度
        win_red_speed_threat: List[float] = []
        win_red_speed_free: List[float] = []

        comp_base_safe, comp_threat, comp_self_coll = [], [], []
        # PPO 健康值（补齐 TB 缺口：原来算了但被丢弃）
        ppo_metrics: Dict[str, List[float]] = {
            "critic_loss": [],
            "explained_variance": 0.0,
        }
        for bid in adversarial_ids:
            ppo_metrics[f"actor_loss_{bid}"] = []
            ppo_metrics[f"entropy_{bid}"] = []
            ppo_metrics[f"kl_anchor_{bid}"] = []
            ppo_metrics[f"approx_kl_{bid}"] = []
            ppo_metrics[f"clip_frac_{bid}"] = []

        # ── 收集 rollout，每个 episode 采样一个 RED 团队 ──
        for rollout_ep in range(args.rollout_episodes):
            # 从 σ_red 采样这一 episode 用的 RED
            red_idx = int(np.random.choice(len(red_pool), p=sigma_red))
            current_cooperative_actors = red_pool.get_actors(red_idx)

            obs_dict, info = env.reset()
            active_mask = dict(info['active_mask'])
            episode_end_reason = info['episode_end_reason']
            adv_ep_rewards = {bid: 0.0 for bid in adversarial_ids}
            ep_length = 0
            collision_sticky: Dict[int, bool] = {i: False for i in range(total_agents)}
            ep_threat_active_steps = 0
            ep_total_threat_scores: List[float] = []

            for step in range(args.max_steps):
                ep_length += 1
                action_dict: Dict[int, np.ndarray] = {}
                step_cache: Dict[int, Dict] = {}

                global_state_np = env.get_global_state().astype(np.float32)
                global_state_t = torch.as_tensor(global_state_np, dtype=torch.float32,
                                                 device=device).unsqueeze(0)
                with torch.no_grad():
                    agent_values = {
                        bid: float(critic(global_state_t, adv_index[bid]).item())
                        for bid in adversarial_ids if active_mask.get(bid, False)
                    }

                with torch.no_grad():
                    for cid in cooperative_ids:
                        if not active_mask.get(cid, False) or cid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32,
                                                device=device).unsqueeze(0)
                        a = current_cooperative_actors[cid].get_deterministic_action(obs_t)\
                            .squeeze(0).cpu().numpy()
                        action_dict[cid] = action_tanh_to_env(a)
                    for bid in adversarial_ids:
                        if not active_mask.get(bid, False) or bid not in obs_dict:
                            continue
                        obs_t = torch.as_tensor(obs_dict[bid], dtype=torch.float32,
                                                device=device).unsqueeze(0)
                        a_tanh_t, logp_t, z_t = adversarial_actors[bid].sample_action_z(obs_t)
                        a = a_tanh_t.squeeze(0).cpu().numpy()
                        z = z_t.squeeze(0).cpu().numpy()
                        action_dict[bid] = action_tanh_to_env(a)
                        step_cache[bid] = {
                            "obs": obs_dict[bid].copy(),
                            "global_state": global_state_np,
                            "action_z": z,
                            "logp": float(logp_t.item()),
                            "value": agent_values[bid],
                        }

                next_obs, rewards, dones, info = env.step(action_dict)
                terminated_info = info.get('terminated', {})
                reward_details = info.get('reward_details', {}) or {}
                next_active_mask = info['active_mask']
                episode_end_reason = info['episode_end_reason']
                pre_collision_kinematics = info.get('pre_collision_kinematics', {}) or {}

                step_collisions = info.get('collisions', {}) or {}
                for i in range(total_agents):
                    if step_collisions.get(i, False):
                        collision_sticky[i] = True

                # 几何/速度指标
                threat_active = is_any_red_under_threat(
                    env, cooperative_ids, adversarial_ids,
                    args.threat_radius, args.threat_lateral,
                )
                if threat_active:
                    ep_threat_active_steps += 1

                # 减速代理：按威胁桶收集红车平均速度（与 fixed_scenario 同口径）
                red_spd = avg_red_speed(env, cooperative_ids)
                if red_spd is not None:
                    if threat_active:
                        win_red_speed_threat.append(red_spd)
                    else:
                        win_red_speed_free.append(red_spd)

                step_kinematics = extract_kinematics(
                    env, cooperative_ids + adversarial_ids, action_dict,
                    pre_collision_kinematics,
                )
                threat_scores = compute_blue_threat_scores(
                    env, cooperative_ids, adversarial_ids,
                    threat_radius=args.threat_radius,
                    threat_lateral=args.threat_lateral,
                    threat_tau=args.threat_tau,
                    threat_near=getattr(args, 'threat_near', 3.0),
                    kinematics=step_kinematics,
                )
                ep_total_threat_scores.append(sum(threat_scores.values()))

                # 奖励组装
                for bid in step_cache.keys():
                    base_safe = extract_base_driving_safe(reward_details, bid)
                    threat = threat_scores.get(bid, 0.0)
                    collision_raw = extract_collision_component(reward_details, bid)
                    self_coll_term = self_collision_coef * collision_raw

                    combined = base_safe + args.threat_coef * threat + self_coll_term
                    adv_ep_rewards[bid] += combined
                    comp_base_safe.append(base_safe)
                    comp_threat.append(threat)
                    comp_self_coll.append(self_coll_term)

                    buffer.add(bid, Transition(
                        obs=step_cache[bid]["obs"],
                        global_state=step_cache[bid]["global_state"],
                        action_z=step_cache[bid]["action_z"],
                        log_prob=step_cache[bid]["logp"],
                        reward=combined,
                        done=float(dones.get(bid, False)),
                        terminated=float(terminated_info.get(bid, dones.get(bid, False))),
                        value=step_cache[bid]["value"],
                        scenario_name=f"psro{psro_iter}_red{red_idx}",
                    ))

                obs_dict = next_obs
                active_mask = dict(next_active_mask)
                # 蓝车全部死亡时 reason 仍为 running，红车继续到成功/碰撞/真超时。
                if episode_end_reason != "running":
                    break

            # episode 终止统计
            coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
            blue_collision = any(collision_sticky[bid] for bid in adversarial_ids)
            # 红车结局拆分：碰撞（任一红车撞了）/ 超时（没成功也没撞，跑满步数）
            red_collision = any(collision_sticky[cid] for cid in cooperative_ids)
            red_timeout = episode_end_reason == "max_steps"

            last_gs = env.get_global_state().astype(np.float32)
            for bid in adversarial_ids:
                buffer.mark_episode_end(bid)
                own_reached = env.reached_target.get(bid, False)
                own_crashed = collision_sticky[bid]
                naturally_terminal = own_reached or own_crashed
                buffer.add_episode_bootstrap(bid, None if naturally_terminal else last_gs)

            win_coop_success.append(1.0 if coop_success else 0.0)
            win_blue_collision.append(1.0 if blue_collision else 0.0)
            win_red_collision.append(1.0 if red_collision else 0.0)
            win_red_timeout.append(1.0 if red_timeout else 0.0)
            win_ep_lengths.append(ep_length)
            for bid in adversarial_ids:
                win_adv_returns[bid].append(adv_ep_rewards[bid])
            win_blue_threat_pct.append(ep_threat_active_steps / max(ep_length, 1))
            win_avg_threat_score.append(
                float(np.mean(ep_total_threat_scores)) if ep_total_threat_scores else 0.0
            )

        # ── 自适应自碰 PID-I 更新 ──
        blue_coll_rate = float(np.mean(win_blue_collision)) if win_blue_collision else 0.0
        if blue_coll_rate > target_blue_collision_rate:
            self_collision_coef *= 1.2
        else:
            self_collision_coef *= 0.95
        self_collision_coef = float(np.clip(
            self_collision_coef, 0.1, getattr(args, 'self_collision_coef_max', 10.0)))

        # ── PPO 更新 ──
        n_transitions = buffer.total_size()
        if n_transitions < args.min_transitions_per_update:
            print(f"  [PSRO {psro_iter} update {update}] 跳过 PPO: n_trans={n_transitions}")
            continue

        per_agent_tensors = {}
        critic_states, critic_returns, critic_agent_ids = [], [], []
        for bid in adversarial_ids:
            data = buffer.get_agent_data(bid)
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

            boundaries = buffer.episode_boundaries.get(bid, [])
            bootstrap_states = buffer.episode_bootstrap_states.get(bid, [])
            bootstrap_values = []
            with torch.no_grad():
                for bs in bootstrap_states:
                    if bs is not None:
                        bs_t = torch.as_tensor(bs, dtype=torch.float32, device=device).unsqueeze(0)
                        bootstrap_values.append(float(critic(bs_t, adv_index[bid]).item()))
                    else:
                        bootstrap_values.append(0.0)
            adv_arr, ret_arr = compute_gae_per_episode(
                rewards=reward_arr, values=value_arr, dones=done_arr,
                terminateds=terminated_arr, boundaries=boundaries,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                bootstrap_values=bootstrap_values,
            )
            per_agent_tensors[bid] = {
                "obs": torch.as_tensor(obs_arr, dtype=torch.float32, device=device),
                "act": torch.as_tensor(act_arr, dtype=torch.float32, device=device),
                "old_logp": torch.as_tensor(old_logp_arr, dtype=torch.float32, device=device),
                "adv": torch.as_tensor(adv_arr, dtype=torch.float32, device=device),
            }
            critic_states.append(state_arr)
            critic_returns.append(ret_arr)
            critic_agent_ids.append(np.full(len(ret_arr), adv_index[bid], dtype=np.int64))

        if len(critic_states) == 0:
            continue
        critic_state_t = torch.as_tensor(np.concatenate(critic_states, axis=0),
                                         dtype=torch.float32, device=device)
        critic_ret_t = torch.as_tensor(np.concatenate(critic_returns, axis=0),
                                       dtype=torch.float32, device=device)
        critic_aid_t = torch.as_tensor(np.concatenate(critic_agent_ids, axis=0),
                                       dtype=torch.long, device=device)

        agent_stopped: Dict[int, bool] = {bid: False for bid in adversarial_ids}
        for epoch in range(args.ppo_epochs):
            c_n = critic_state_t.shape[0]
            c_idx = np.arange(c_n); np.random.shuffle(c_idx)
            for start in range(0, c_n, args.minibatch_size):
                mb = c_idx[start: start + args.minibatch_size]
                value_pred = critic(critic_state_t[mb], critic_aid_t[mb])
                critic_loss = 0.5 * (critic_ret_t[mb] - value_pred).pow(2).mean()
                critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                critic_optim.step()
                ppo_metrics["critic_loss"].append(float(critic_loss.item()))

            for bid in adversarial_ids:
                if bid not in per_agent_tensors or agent_stopped[bid]:
                    continue
                a_obs = per_agent_tensors[bid]["obs"]
                a_act = per_agent_tensors[bid]["act"]
                a_oldlogp = per_agent_tensors[bid]["old_logp"]
                a_adv = per_agent_tensors[bid]["adv"]
                a_n = a_obs.shape[0]
                a_idx = np.arange(a_n); np.random.shuffle(a_idx)
                ep_kl_proxy, ep_losses, ep_ent, ep_cf, ep_kl_anc = [], [], [], [], []
                for start in range(0, a_n, args.minibatch_size):
                    mb = a_idx[start: start + args.minibatch_size]
                    new_logp, entropy = adversarial_actors[bid].evaluate_actions_z(
                        a_obs[mb], a_act[mb]
                    )
                    logratio = new_logp - a_oldlogp[mb]
                    ratio = torch.exp(logratio)
                    surr1 = ratio * a_adv[mb]
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps,
                                        1.0 + args.clip_eps) * a_adv[mb]
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    # KL 锚定到 frozen base actor
                    mu_adv, logstd_adv = adversarial_actors[bid].forward(a_obs[mb])
                    with torch.no_grad():
                        mu_base, logstd_base = base_actors_frozen[bid].forward(a_obs[mb])
                    kl_anchor = gaussian_kl_diag(mu_adv, logstd_adv, mu_base, logstd_base)

                    actor_loss = (
                        ppo_loss
                        - args.ent_coef * entropy.mean()
                        + args.beta_kl * kl_anchor
                    )

                    adversarial_actor_optims[bid].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(adversarial_actors[bid].parameters(),
                                             args.max_grad_norm)
                    adversarial_actor_optims[bid].step()
                    ep_losses.append(float(actor_loss.item()))
                    ep_ent.append(float(entropy.mean().item()))
                    ep_kl_anc.append(float(kl_anchor.item()))
                    with torch.no_grad():
                        ep_kl_proxy.append(float(((ratio - 1.0) - logratio).mean().item()))
                        ep_cf.append(
                            float(((ratio - 1.0).abs() > args.clip_eps).float().mean().item())
                        )
                mean_kl = float(np.mean(ep_kl_proxy)) if ep_kl_proxy else 0.0
                # 补齐 TB 缺口：把已算的 PPO 健康值收集起来（原来算了就丢）
                if ep_losses:
                    ppo_metrics[f"actor_loss_{bid}"].append(float(np.mean(ep_losses)))
                    ppo_metrics[f"entropy_{bid}"].append(float(np.mean(ep_ent)))
                    ppo_metrics[f"kl_anchor_{bid}"].append(float(np.mean(ep_kl_anc)))
                    ppo_metrics[f"approx_kl_{bid}"].append(mean_kl)
                    ppo_metrics[f"clip_frac_{bid}"].append(float(np.mean(ep_cf)))
                if args.target_kl > 0 and mean_kl > 1.5 * args.target_kl:
                    agent_stopped[bid] = True

        # explained_variance：critic 拟合优度（补齐 TB 缺口）
        with torch.no_grad():
            all_vp = critic(critic_state_t, critic_aid_t)
            var_res = torch.var(critic_ret_t - all_vp)
            var_tot = torch.var(critic_ret_t)
            ppo_metrics["explained_variance"] = float(1.0 - var_res / (var_tot + 1e-8))

        # ── TB 日志 ──
        if tb_writer is not None:
            tb_step = tb_step_offset + update
            win_succ = float(np.mean(win_coop_success))
            win_b_coll = float(np.mean(win_blue_collision))
            win_t_pct = float(np.mean(win_blue_threat_pct))
            tb_writer.add_scalar(f'psro{psro_iter}/coop_success_rate', win_succ, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/blue_collision_rate', win_b_coll, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/blue_threat_window_pct', win_t_pct, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/avg_threat_score',
                                 float(np.mean(win_avg_threat_score)), tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/self_collision_coef',
                                 self_collision_coef, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/reward_base_safe',
                                 float(np.mean(comp_base_safe)) if comp_base_safe else 0.0, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/reward_threat',
                                 float(np.mean(comp_threat)) if comp_threat else 0.0, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/reward_self_collision',
                                 float(np.mean(comp_self_coll)) if comp_self_coll else 0.0, tb_step)
            tb_writer.add_scalar(f'psro{psro_iter}/lr_actor', cur_lr_a, tb_step)

            # ── 红车结局拆分（对抗有效性归因）──
            pre = f'psro{psro_iter}'
            tb_writer.add_scalar(f'{pre}/red_collision_rate',
                                 float(np.mean(win_red_collision)) if win_red_collision else 0.0, tb_step)
            tb_writer.add_scalar(f'{pre}/red_timeout_rate',
                                 float(np.mean(win_red_timeout)) if win_red_timeout else 0.0, tb_step)
            # adv_effectiveness = 1 − coop_success（红车没全到点即算对抗生效，绝对量，非反事实）
            tb_writer.add_scalar(f'{pre}/adv_effectiveness', 1.0 - win_succ, tb_step)

            # ── 减速代理 ──
            red_spd_t = float(np.mean(win_red_speed_threat)) if win_red_speed_threat else 0.0
            red_spd_f = float(np.mean(win_red_speed_free)) if win_red_speed_free else 0.0
            slowdown = ((red_spd_f - red_spd_t) / red_spd_f) if red_spd_f > 0.1 else 0.0
            tb_writer.add_scalar(f'{pre}/red_avg_speed_under_threat', red_spd_t, tb_step)
            tb_writer.add_scalar(f'{pre}/red_avg_speed_free', red_spd_f, tb_step)
            tb_writer.add_scalar(f'{pre}/red_relative_slowdown', slowdown, tb_step)
            tb_writer.add_scalar(f'{pre}/avg_ep_length',
                                 float(np.mean(win_ep_lengths)) if win_ep_lengths else 0.0, tb_step)

            # ── PPO 健康值（原来算了就丢，现补记）──
            if ppo_metrics["critic_loss"]:
                tb_writer.add_scalar(f'{pre}/ppo_critic_loss',
                                     float(np.mean(ppo_metrics["critic_loss"])), tb_step)
                tb_writer.add_scalar(f'{pre}/ppo_explained_variance',
                                     ppo_metrics["explained_variance"], tb_step)
            for bid in adversarial_ids:
                if ppo_metrics[f"actor_loss_{bid}"]:
                    tb_writer.add_scalar(f'{pre}/ppo_actor_loss_b{bid}',
                                         float(np.mean(ppo_metrics[f"actor_loss_{bid}"])), tb_step)
                    tb_writer.add_scalar(f'{pre}/ppo_entropy_b{bid}',
                                         float(np.mean(ppo_metrics[f"entropy_{bid}"])), tb_step)
                    tb_writer.add_scalar(f'{pre}/ppo_kl_anchor_b{bid}',
                                         float(np.mean(ppo_metrics[f"kl_anchor_{bid}"])), tb_step)
                    tb_writer.add_scalar(f'{pre}/ppo_approx_kl_b{bid}',
                                         float(np.mean(ppo_metrics[f"approx_kl_{bid}"])), tb_step)
                    tb_writer.add_scalar(f'{pre}/ppo_clip_frac_b{bid}',
                                         float(np.mean(ppo_metrics[f"clip_frac_{bid}"])), tb_step)
                # 蓝车探索水平
                tb_writer.add_scalar(f'{pre}/log_std_b{bid}',
                                     float(adversarial_actors[bid].log_std.mean().item()), tb_step)

        win_succ_f = float(np.mean(win_coop_success))
        print(f"  [PSRO {psro_iter} update {update:3d}/{args.inner_updates}] "
              f"coop_succ={win_succ_f:.0%}, blue_coll={float(np.mean(win_blue_collision)):.0%}, "
              f"threat_pct={float(np.mean(win_blue_threat_pct)):.0%}, "
              f"self_coll={self_collision_coef:.2f}, lr_a={cur_lr_a:.1e}")

        last_window_coop_success = win_coop_success
        last_window_blue_collision = win_blue_collision
        last_window_threat_pct = win_blue_threat_pct

    # ── 训练完成，组装 ckpt 返回 ──
    stats = {
        "final_coop_success_rate": float(np.mean(last_window_coop_success))
            if last_window_coop_success else float('nan'),
        "final_blue_collision_rate": float(np.mean(last_window_blue_collision))
            if last_window_blue_collision else float('nan'),
        "final_blue_threat_window_pct": float(np.mean(last_window_threat_pct))
            if last_window_threat_pct else float('nan'),
        "final_self_collision_coef": float(self_collision_coef),
    }

    blue_team_ckpt = _build_blue_team_ckpt(
        adversarial_actors, critic, adversarial_actor_optims, critic_optim,
        adversarial_ids, sigma_red, psro_iter, vars(args), stats=stats,
    )
    return blue_team_ckpt
