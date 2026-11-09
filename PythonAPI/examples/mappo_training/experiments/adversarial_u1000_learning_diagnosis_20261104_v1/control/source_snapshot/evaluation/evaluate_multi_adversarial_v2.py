"""
增强版多对抗评估脚本 v2
========================
为 KL-anchored + 风险增量奖励训练（train_multi_adversarial_kl_risk /
train_adversarial_fixed_scenario）配套的对抗有效性评估。

相对原 evaluate_multi_adversarial.py 的修复（参见
docs/对抗评估_检查与修复.md）：

  M1 蓝车摆位与训练对齐
      默认调用 make_adversarial_positions 把蓝车摆在红车之间，
      可选 --coop-start-config 覆盖红车起点
  M2 丰富对抗有效性指标
      - blue_collision_rate              (BLUE 自碰率)
      - blue_threat_window_pct           (BLUE 在威胁几何窗口内的步数占比)
      - red_speed_under_threat / free    (RED 被威胁 vs 未被威胁时的均速)
      - red_relative_slowdown            (因果减速比)
      - avg_threat_score_per_ep          (平均威胁分)
  M3 缺 goal_zone 场景显式 warn（依赖 env apply_scenario_config 修复）
  M4 可选 --policy {deterministic, stochastic}

对照模式（baseline vs attacked）：
  baseline : BLUE 用合作策略（= BLUE 训练前的初始权重）
  attacked : BLUE 用训练后的对抗策略

核心指标:
  adversary_effectiveness = baseline_coop_success - attacked_coop_success

用法:
  # 单场景快验
  python -m evaluation.evaluate_multi_adversarial_v2 \
      --pretrained-ckpt logs/robust_v2/best.pt \
      --adversarial-ckpt logs/adv_fixed_scenario/best.pt \
      --scenarios scenario_01_basic.json \
      --num-episodes-per-scenario 10

  # 多场景全量
  python -m evaluation.evaluate_multi_adversarial_v2 \
      --pretrained-ckpt logs/robust_v2/best.pt \
      --adversarial-ckpt logs/multi_adv_kl_risk/best.pt \
      --num-episodes-per-scenario 5 \
      --output-file ./logs/eval/kl_risk_attacked.json
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import json
import datetime
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor

# 复用训练侧组件：蓝车摆位 / 场景配置工具 / 威胁评分
from training.train_multi_adversarial_mappo_v2 import (
    make_adversarial_positions, _scenario_agents, _load_config_file,
)
from training.train_multi_adversarial_kl_risk import compute_blue_threat_scores


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


# ============================================================
# M2 复用：与 train_adversarial_fixed_scenario 同口径的几何指标
# ============================================================

def is_any_red_under_threat(env, cooperative_ids, adversarial_ids,
                            threat_radius: float, threat_lateral: float) -> bool:
    """是否有任何 RED 处于任何 BLUE 的威胁几何窗口内（不算 TTC，只看位置）。"""
    for red_id in cooperative_ids:
        if red_id >= len(env.vehicles):
            continue
        rv = env.vehicles[red_id]
        if not rv.is_alive or env.reached_target.get(red_id, False):
            continue
        rt = rv.get_transform()
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
    """所有未到点 RED 的均速 (m/s)。无可计 RED 返回 None。"""
    speeds = []
    for red_id in cooperative_ids:
        if red_id >= len(env.vehicles):
            continue
        rv = env.vehicles[red_id]
        if not rv.is_alive or env.reached_target.get(red_id, False):
            continue
        vel = rv.get_velocity()
        speeds.append(math.sqrt(vel.x ** 2 + vel.y ** 2))
    return float(np.mean(speeds)) if speeds else None


# ============================================================
# Stats 容器
# ============================================================

@dataclass
class ModeStats:
    """单一模式（baseline / attacked）在某场景上的聚合统计"""
    n_episodes: int = 0
    coop_success: int = 0
    coop_collision: int = 0
    blue_collision: int = 0
    completion_times: List[float] = field(default_factory=list)
    threat_window_pcts: List[float] = field(default_factory=list)
    red_speeds_threat: List[float] = field(default_factory=list)
    red_speeds_free: List[float] = field(default_factory=list)
    avg_threat_scores: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    @property
    def coop_success_rate(self) -> float:
        return self.coop_success / max(self.n_episodes, 1)

    @property
    def coop_collision_rate(self) -> float:
        return self.coop_collision / max(self.n_episodes, 1)

    @property
    def blue_collision_rate(self) -> float:
        return self.blue_collision / max(self.n_episodes, 1)

    @property
    def avg_completion_time(self) -> float:
        return float(np.mean(self.completion_times)) if self.completion_times else float('nan')

    @property
    def avg_threat_window_pct(self) -> float:
        return float(np.mean(self.threat_window_pcts)) if self.threat_window_pcts else 0.0

    @property
    def avg_red_speed_under_threat(self) -> float:
        return float(np.mean(self.red_speeds_threat)) if self.red_speeds_threat else float('nan')

    @property
    def avg_red_speed_free(self) -> float:
        return float(np.mean(self.red_speeds_free)) if self.red_speeds_free else float('nan')

    @property
    def red_relative_slowdown(self) -> float:
        """(free - threat) / free，>0 表示 BLUE 让 RED 减速。free=0 或缺失时返回 0。"""
        free = self.avg_red_speed_free
        thrt = self.avg_red_speed_under_threat
        if math.isnan(free) or free < 0.1 or math.isnan(thrt):
            return 0.0
        return (free - thrt) / free

    @property
    def avg_threat_score(self) -> float:
        return float(np.mean(self.avg_threat_scores)) if self.avg_threat_scores else 0.0

    @property
    def avg_episode_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    def to_dict(self) -> Dict:
        return {
            "n_episodes": self.n_episodes,
            "coop_success_rate": self.coop_success_rate,
            "coop_collision_rate": self.coop_collision_rate,
            "blue_collision_rate": self.blue_collision_rate,
            "avg_completion_time": self.avg_completion_time,
            "avg_threat_window_pct": self.avg_threat_window_pct,
            "avg_red_speed_under_threat": self.avg_red_speed_under_threat,
            "avg_red_speed_free": self.avg_red_speed_free,
            "red_relative_slowdown": self.red_relative_slowdown,
            "avg_threat_score": self.avg_threat_score,
            "avg_episode_length": self.avg_episode_length,
        }

    def merge_from(self, other: 'ModeStats'):
        self.n_episodes += other.n_episodes
        self.coop_success += other.coop_success
        self.coop_collision += other.coop_collision
        self.blue_collision += other.blue_collision
        self.completion_times.extend(other.completion_times)
        self.threat_window_pcts.extend(other.threat_window_pcts)
        self.red_speeds_threat.extend(other.red_speeds_threat)
        self.red_speeds_free.extend(other.red_speeds_free)
        self.avg_threat_scores.extend(other.avg_threat_scores)
        self.episode_lengths.extend(other.episode_lengths)


# ============================================================
# Evaluator
# ============================================================

class MultiAdversarialEvaluatorV2:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

        # 加载对抗 ckpt
        adv_path = self._normalize_path(args.adversarial_ckpt)
        if not os.path.exists(adv_path):
            raise FileNotFoundError(f"对抗 checkpoint 不存在: {adv_path}")
        adv_ckpt = torch.load(adv_path, map_location=self.device, weights_only=False)
        self.cooperative_ids: List[int] = adv_ckpt.get("cooperative_ids")
        self.adversarial_ids: List[int] = adv_ckpt.get("adversarial_ids")
        if self.cooperative_ids is None or self.adversarial_ids is None:
            raise KeyError("对抗 ckpt 缺少 cooperative_ids / adversarial_ids "
                           "（请用 v2 / kl_risk / fixed_scenario 训练脚本生成）")
        self.total_agents = len(self.cooperative_ids) + len(self.adversarial_ids)
        print(f"团队划分: 合作 {self.cooperative_ids} | 对抗 {self.adversarial_ids}")

        # 对抗 actors
        adv_actors_state = adv_ckpt.get("adversarial_actors")
        if not adv_actors_state:
            raise KeyError("对抗 ckpt 缺少 adversarial_actors（复数）")
        self.adversarial_actors: Dict[int, Actor] = {}
        for aid in self.adversarial_ids:
            if aid not in adv_actors_state:
                raise KeyError(f"对抗 ckpt 缺少 agent {aid} 权重")
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(self.device)
            actor.load_state_dict(adv_actors_state[aid])
            actor.eval()
            self.adversarial_actors[aid] = actor

        # 合作 actors
        coop_path = self._normalize_path(args.pretrained_ckpt)
        if not os.path.exists(coop_path):
            raise FileNotFoundError(f"合作 checkpoint 不存在: {coop_path}")
        coop_ckpt = torch.load(coop_path, map_location=self.device, weights_only=False)
        coop_state = coop_ckpt.get("actors", {})
        self.cooperative_actors: Dict[int, Actor] = {}
        for cid in self.cooperative_ids:
            if cid not in coop_state:
                raise KeyError(f"合作 ckpt 缺少 agent {cid} 权重")
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(self.device)
            actor.load_state_dict(coop_state[cid])
            actor.eval()
            self.cooperative_actors[cid] = actor

        # baseline 模式下 BLUE 使用的"正常驾驶"策略
        # 训练侧 BLUE init 时即用合作 agent 0 权重 → baseline 即 BLUE 未训练状态
        self.normal_policy = self.cooperative_actors[self.cooperative_ids[0]]
        print(f"已加载 actors，合作 {list(self.cooperative_actors.keys())}, "
              f"对抗 {list(self.adversarial_actors.keys())}")
        print(f"policy 模式: {self.args.policy}, 摆位对齐: {self.args.match_training_layout}, "
              f"coop_start_config: {self.args.coop_start_config or 'none'}")

    @staticmethod
    def _normalize_path(p: str) -> str:
        while p.startswith('\\') or p.startswith('/'):
            p = p[1:]
        return p.replace('\\', os.sep).replace('/', os.sep)

    def _make_env(self) -> CarlaMultiAgentEnv:
        return CarlaMultiAgentEnv(
            host=self.args.host, port=self.args.port, town=self.args.town,
            num_agents=self.total_agents,
            straight_road_length=self.args.straight_road_length,
            timestep=self.args.timestep, max_steps=self.args.max_steps,
            spawn_vehicles=True,
            adversarial_agent_ids=self.adversarial_ids,
            reward_version="v2",
            disable_tensorboard=True,
            seed=self.args.seed,
        )

    def _select_action(self, actor: Actor, obs_t: torch.Tensor) -> np.ndarray:
        """M4: deterministic 或 stochastic 采样动作。"""
        if self.args.policy == "stochastic":
            a_tanh_t, _ = actor.sample_action(obs_t)
            return a_tanh_t.squeeze(0).cpu().numpy()
        return actor.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()

    def _run_episode(self, env: CarlaMultiAgentEnv, has_adversary: bool) -> Dict:
        """跑一个 episode，返回单 ep 的 success/collision + M2 所有指标。"""
        obs_dict, info = env.reset()
        collision_sticky = {i: False for i in range(self.total_agents)}
        reach_step = {i: None for i in range(self.total_agents)}
        steps = 0

        # M2 累积器
        ep_threat_steps = 0
        ep_red_spd_threat: List[float] = []
        ep_red_spd_free: List[float] = []
        ep_threat_scores: List[float] = []

        for step in range(self.args.max_steps):
            steps += 1
            action_dict: Dict[int, np.ndarray] = {}
            with torch.no_grad():
                for cid in self.cooperative_ids:
                    if cid not in obs_dict:
                        continue
                    obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                    a = self._select_action(self.cooperative_actors[cid], obs_t)
                    action_dict[cid] = action_tanh_to_env(a)
                for aid in self.adversarial_ids:
                    if aid not in obs_dict:
                        continue
                    policy = self.adversarial_actors[aid] if has_adversary else self.normal_policy
                    obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                    a = self._select_action(policy, obs_t)
                    action_dict[aid] = action_tanh_to_env(a)

            next_obs, rewards, dones, info = env.step(action_dict)

            step_collisions = info.get('collisions', {}) or {}
            for i in range(self.total_agents):
                if step_collisions.get(i, False):
                    collision_sticky[i] = True
                if reach_step[i] is None and env.reached_target.get(i, False):
                    reach_step[i] = steps

            # ── M2 几何指标采样 ──
            threat_active = is_any_red_under_threat(
                env, self.cooperative_ids, self.adversarial_ids,
                self.args.threat_radius, self.args.threat_lateral,
            )
            if threat_active:
                ep_threat_steps += 1
            rs = avg_red_speed(env, self.cooperative_ids)
            if rs is not None:
                (ep_red_spd_threat if threat_active else ep_red_spd_free).append(rs)
            threat_scores = compute_blue_threat_scores(
                env, self.cooperative_ids, self.adversarial_ids,
                threat_radius=self.args.threat_radius,
                threat_lateral=self.args.threat_lateral,
                threat_tau=self.args.threat_tau,
            )
            ep_threat_scores.append(sum(threat_scores.values()))

            obs_dict = next_obs
            if all(dones.values()):
                break

        coop_success = all(env.reached_target.get(cid, False) for cid in self.cooperative_ids)
        coop_collision = any(collision_sticky[cid] for cid in self.cooperative_ids)
        blue_collision = any(collision_sticky[bid] for bid in self.adversarial_ids)

        completion_time = None
        if coop_success:
            last = max(reach_step[cid] for cid in self.cooperative_ids)
            completion_time = last * self.args.timestep

        return {
            "coop_success": coop_success,
            "coop_collision": coop_collision,
            "blue_collision": blue_collision,
            "completion_time": completion_time,
            "steps": steps,
            "threat_window_pct": ep_threat_steps / max(steps, 1),
            "red_speed_threat": float(np.mean(ep_red_spd_threat)) if ep_red_spd_threat else None,
            "red_speed_free": float(np.mean(ep_red_spd_free)) if ep_red_spd_free else None,
            "avg_threat_score": float(np.mean(ep_threat_scores)) if ep_threat_scores else 0.0,
        }

    def _prepare_scenario_config(self, env, scenario_name, scenario_dir) -> Optional[Dict]:
        """M1：加载场景 + 可选 coop_start_config 覆盖红车 + 蓝车自动摆位。"""
        config = env.load_scenario_by_name(scenario_name, scenario_dir)
        if config is None:
            config = env.load_random_scenario(scenario_dir, [scenario_name])
        if config is None:
            return None
        config = dict(config)

        # 可选：红车起点覆盖（与训练脚本对齐）
        if self.args.coop_start_config and self.args.coop_start_config.lower() not in ("none", ""):
            coop_cfg = _load_config_file(self.args.coop_start_config, scenario_dir)
            if coop_cfg is not None:
                src = _scenario_agents(coop_cfg)
                coop_pos = {f"agent{cid}": src[f"agent{cid}"]
                            for cid in self.cooperative_ids if f"agent{cid}" in src}
                agents = _scenario_agents(config)
                agents.update(coop_pos)
                config["agents"] = agents

        # M1 关键：蓝车摆位对齐训练
        if self.args.match_training_layout:
            config["agents"] = make_adversarial_positions(
                config, self.cooperative_ids, self.adversarial_ids,
                margin_ratio=self.args.adv_margin_ratio,
            )

        # M3 警示：缺 goal_zone 时提醒（env 侧 fail-loud 完成后此处可移除）
        if "goal_zone" not in config:
            print(f"  ⚠️ 场景 {scenario_name} 未声明 goal_zone，env 将沿用上次或回退 waypoint")

        return config

    def _eval_scenario(self, env, scenario_name, scenario_dir) -> Dict[str, ModeStats]:
        config = self._prepare_scenario_config(env, scenario_name, scenario_dir)
        if config is not None:
            env.set_scenario(config)
            agents = _scenario_agents(config)
            print(f"  场景几何: {[(k, round(v.get('x',0),1), round(v.get('y',0),1)) for k,v in sorted(agents.items())]}")

        stats = {"baseline": ModeStats(), "attacked": ModeStats()}
        for mode, has_adv in [("baseline", False), ("attacked", True)]:
            st = stats[mode]
            for ep_i in range(self.args.num_episodes_per_scenario):
                r = self._run_episode(env, has_adversary=has_adv)
                st.n_episodes += 1
                st.coop_success += 1 if r["coop_success"] else 0
                st.coop_collision += 1 if r["coop_collision"] else 0
                st.blue_collision += 1 if r["blue_collision"] else 0
                st.episode_lengths.append(r["steps"])
                st.threat_window_pcts.append(r["threat_window_pct"])
                st.avg_threat_scores.append(r["avg_threat_score"])
                if r["completion_time"] is not None:
                    st.completion_times.append(r["completion_time"])
                if r["red_speed_threat"] is not None:
                    st.red_speeds_threat.append(r["red_speed_threat"])
                if r["red_speed_free"] is not None:
                    st.red_speeds_free.append(r["red_speed_free"])
        return stats

    def evaluate(self) -> Dict:
        scenario_dir = self.args.test_scenario_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
        scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])
        if self.args.scenarios:
            wanted = set(self.args.scenarios.split(','))
            scenario_files = [f for f in scenario_files if f in wanted]
        if not scenario_files:
            raise ValueError("没有匹配到任何场景文件")
        print(f"评估场景目录: {scenario_dir}，共 {len(scenario_files)} 个场景")

        env = self._make_env()
        per_scenario: Dict[str, Dict] = {}
        agg = {"baseline": ModeStats(), "attacked": ModeStats()}
        try:
            for sname in scenario_files:
                print(f"\n>>> 场景 {sname}")
                stats = self._eval_scenario(env, sname, scenario_dir)
                eff = stats["baseline"].coop_success_rate - stats["attacked"].coop_success_rate
                per_scenario[sname] = {
                    "baseline": stats["baseline"].to_dict(),
                    "attacked": stats["attacked"].to_dict(),
                    "effectiveness": eff,
                }
                for mode in ("baseline", "attacked"):
                    agg[mode].merge_from(stats[mode])
                # 每场景小结
                b, a = stats["baseline"], stats["attacked"]
                print(f"    [baseline] coop_succ={b.coop_success_rate:.0%}, "
                      f"coop_coll={b.coop_collision_rate:.0%}, "
                      f"blue_coll={b.blue_collision_rate:.0%}, "
                      f"ep_len={b.avg_episode_length:.0f}")
                print(f"    [attacked] coop_succ={a.coop_success_rate:.0%}, "
                      f"coop_coll={a.coop_collision_rate:.0%}, "
                      f"blue_coll={a.blue_collision_rate:.0%}, "
                      f"ep_len={a.avg_episode_length:.0f}")
                print(f"    [对抗有效性] effectiveness={eff:+.2%}, "
                      f"BLUE 施压步占比 {b.avg_threat_window_pct:.0%}→{a.avg_threat_window_pct:.0%}, "
                      f"RED 减速 {a.red_relative_slowdown:+.2%}")
        finally:
            env.close()

        overall_eff = agg["baseline"].coop_success_rate - agg["attacked"].coop_success_rate
        result = {
            "pretrained_ckpt": self.args.pretrained_ckpt,
            "adversarial_ckpt": self.args.adversarial_ckpt,
            "cooperative_ids": self.cooperative_ids,
            "adversarial_ids": self.adversarial_ids,
            "num_episodes_per_scenario": self.args.num_episodes_per_scenario,
            "policy_mode": self.args.policy,
            "match_training_layout": self.args.match_training_layout,
            "coop_start_config": self.args.coop_start_config,
            "adv_margin_ratio": self.args.adv_margin_ratio,
            "threat_params": {
                "radius": self.args.threat_radius,
                "lateral": self.args.threat_lateral,
                "tau": self.args.threat_tau,
            },
            "overall": {
                "baseline": agg["baseline"].to_dict(),
                "attacked": agg["attacked"].to_dict(),
                "effectiveness": overall_eff,
            },
            "per_scenario": per_scenario,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict):
        b = result["overall"]["baseline"]
        a = result["overall"]["attacked"]
        eff = result["overall"]["effectiveness"]
        print("\n" + "=" * 70)
        print("增强版多对抗评估汇总（M1+M2+M4 修复）")
        print("=" * 70)
        print(f"{'':24} {'baseline':>14} {'attacked':>14} {'差值':>10}")
        print(f"{'-' * 70}")
        rows = [
            ("coop_success_rate", "coop_success_rate", "%"),
            ("coop_collision_rate", "coop_collision_rate", "%"),
            ("blue_collision_rate", "blue_collision_rate", "%"),
            ("avg_completion_time", "avg_completion_time", "s"),
            ("avg_episode_length", "avg_episode_length", "步"),
            ("avg_threat_window_pct", "avg_threat_window_pct", "%"),
            ("avg_red_speed_under_threat", "avg_red_speed_under_threat", "m/s"),
            ("avg_red_speed_free", "avg_red_speed_free", "m/s"),
            ("red_relative_slowdown", "red_relative_slowdown", "%"),
            ("avg_threat_score", "avg_threat_score", ""),
        ]
        for label, key, unit in rows:
            vb = b.get(key, float('nan'))
            va = a.get(key, float('nan'))
            try:
                if unit == "%":
                    print(f"{label:<24} {vb*100:>13.1f}% {va*100:>13.1f}% {(va-vb)*100:>+9.1f}%")
                elif unit in ("s", "m/s", "步"):
                    print(f"{label:<24} {vb:>13.2f}{unit:<1} {va:>13.2f}{unit:<1} {va-vb:>+9.2f}")
                else:
                    print(f"{label:<24} {vb:>13.2f}  {va:>13.2f}  {va-vb:>+9.2f}")
            except Exception:
                print(f"{label:<24} {str(vb):>14} {str(va):>14}")
        print(f"{'-' * 70}")
        print(f"  ★ adversary_effectiveness = baseline_coop_success - attacked_coop_success "
              f"= {eff:+.2%}")
        print(f"  解读：BLUE 让 RED 成功率下降 {eff*100:.1f} 个百分点，"
              f"自身碰撞率 {a['blue_collision_rate']:.1%}")
        print("=" * 70)


def build_parser():
    parser = argparse.ArgumentParser(
        description="增强版多对抗 MAPPO 评估（M1+M2+M4 修复）"
    )

    parser.add_argument("--pretrained-ckpt", type=str, required=True,
                        help="合作团队预训练 ckpt")
    parser.add_argument("--adversarial-ckpt", type=str, required=True,
                        help="对抗 ckpt（v2 / kl_risk / fixed_scenario 任一）")
    parser.add_argument("--num-episodes-per-scenario", type=int, default=5)
    parser.add_argument("--test-scenario-dir", type=str, default="",
                        help="评估场景目录（默认 data/training_scenarios）")
    parser.add_argument("--scenarios", type=str, default="",
                        help="逗号分隔的场景文件名子集，留空=全部")
    parser.add_argument("--output-file", type=str, default="",
                        help="结果 JSON 输出路径，留空不写")

    # M1：摆位对齐
    parser.add_argument("--match-training-layout", action="store_true", default=True,
                        help="评估时也用 make_adversarial_positions 摆位蓝车（默认开）")
    parser.add_argument("--no-match-training-layout", dest="match_training_layout",
                        action="store_false")
    parser.add_argument("--coop-start-config", type=str, default="none",
                        help="（可选）红车起点固定配置，与训练参数对齐。默认 none")
    parser.add_argument("--adv-margin-ratio", type=float, default=0.15,
                        help="蓝车分布内缩比例，与训练对齐（默认 0.15）")

    # M2：威胁几何参数（与训练默认对齐）
    parser.add_argument("--threat-radius", type=float, default=15.0)
    parser.add_argument("--threat-lateral", type=float, default=2.5)
    parser.add_argument("--threat-tau", type=float, default=1.5)

    # M4：policy 模式
    parser.add_argument("--policy", type=str, default="deterministic",
                        choices=["deterministic", "stochastic"],
                        help="deterministic=mu→tanh; stochastic=sample（提高统计 power）")

    # 环境
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    set_seed(args.seed)
    evaluator = MultiAdversarialEvaluatorV2(args)
    result = evaluator.evaluate()
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output_file}")
