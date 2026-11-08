"""
对抗奖励单一真相源（§2.5）
==========================
把 §2.3 的完整对抗奖励收敛成一个类 `AdversarialRewardComputer`，供三处共同 import：

  - train_adversarial_fixed_scenario.py  (Phase 1 快验代理)
  - training/train_2red_1blue_ppo.py     (2+1 初步验证，本轮新增)
  - training/psro/train_blue_oracle.py   (Phase 4 PSRO 蓝 oracle 真正落点)
  - train_multi_adversarial_kl_risk.py   (Phase 3 双蓝车 MAPPO / 最终 2+2)

奖励结构（§2.3）:
    r_blue(t) = w_I · r_impact(t)        # 稀疏·归因事件（主目标）
              + w_T · r_threat_gated(t)  # 稠密·门控引导（脚手架，每回合封顶）
              + w_B · base_safe(t)       # 小权重·基础驾驶能力
              − λ_t · c_self(t)          # 自适应约束（蓝车自身安全，对偶更新）
    （KL 锚定项 β_KL·KL(π‖π_anchor) 属于 actor_loss，不在本模块，见训练脚本）

复用：
  - compute_blue_threat_scores / extract_base_driving_safe / extract_collision_component
    ← training.train_multi_adversarial_kl_risk（几何威胁分 + 基础安全分 + 碰撞分）
  - SlowdownDetector / SteerResponseDetector / classify_collision / in_threat_window /
    birth_protection_active / red_is_responding / extract_kinematics
    ← utils.adversarial_metrics（归因工具）

设计文档：docs/adversarial_metrics_reward_design_and_plan_20260705.md §2.3 / §2.5 / §3
"""
from __future__ import annotations

import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils.adversarial_metrics import (
    AgentKinematics,
    extract_kinematics,
    in_threat_window,
    SlowdownDetector,
    SteerResponseDetector,
    birth_protection_active,
    red_is_responding,
    classify_collision,
    COLLISION_KAMIKAZE,
    COLLISION_INDUCED,
    COLLISION_COLLATERAL,
)
from training.train_multi_adversarial_kl_risk import (
    compute_blue_threat_scores,
    extract_base_driving_safe,
    extract_collision_component,
)

@dataclass
class RewardConfig:
    """§2.3 对抗奖励全部超参。默认值为初值（§2.4 定标后在文档变更表追加修订）。"""
    # ── 三大项权重 ──
    w_impact: float = 1.0             # 稀疏归因事件（主目标）
    w_threat: float = 1.0             # 门控稠密威胁引导
    w_base_safe: float = 0.3          # 基础安全驾驶（小权重）

    # ── r_impact 各事件奖励 + 每回合上限（§2.3 表）──
    reward_slowdown: float = 5.0
    cap_slowdown: int = 3
    reward_swerve: float = 3.0
    cap_swerve: int = 3
    reward_induced_collision: float = 30.0
    cap_induced_collision: int = 1
    reward_timeout: float = 15.0      # 归因超时（回合末结算）
    # kamikaze: 固定 0 分，不设参数

    # ── r_threat_gated 每回合累计封顶（≈ 单次诱导碰撞的 2/3）──
    threat_cap_per_episode: float = 20.0

    # ── 门控 / 归因阈值 ──
    threat_radius: float = 15.0
    threat_lateral: float = 2.5
    threat_tau: float = 1.5
    threat_near: float = 3.0
    slowdown_window_secs: float = 2.0
    slowdown_drop_thresh: float = 2.0
    swerve_window_secs: float = 0.5
    swerve_jump_thresh: float = 0.15
    timestep: float = 0.1
    birth_protect_secs: float = 3.0
    timeout_threat_pct_min: float = 0.30   # 归因超时要求：威胁窗口占比 ≥ 此值

    # ── c_self 自适应约束（PID-I 对偶）──
    self_collision_coef_init: float = 1.0
    self_collision_coef_max: float = 10.0
    target_blue_collision_rate: float = 0.05
    dual_lr_up: float = 1.2
    dual_lr_down: float = 0.95


# r_impact 事件类型键（供 TB / hack 仪表盘统计）
IMPACT_KEYS = ("slowdown", "swerve", "induced_collision", "timeout", "kamikaze")


class AdversarialRewardComputer:
    """§2.3 对抗奖励计算器（单一真相源）。

    生命周期:
      computer = AdversarialRewardComputer(coop_ids, adv_ids, cfg)
      for episode:
          computer.reset_episode()
          for step:
              r, details = computer.step(env, reward_details, control_dict, step_idx, collisions)
          bonus, bonus_details = computer.finalize_episode(timed_out)
          # bonus 加到该回合最后一步的蓝车奖励上（见训练脚本）
      # 每个 rollout 窗口结束：
      computer.update_dual(blue_collision_rate)   # 自适应 λ_t

    线程/进程内单实例，非线程安全（与训练循环同步调用）。
    """

    def __init__(self, cooperative_ids: List[int], adversarial_ids: List[int],
                 cfg: Optional[RewardConfig] = None):
        self.coop_ids = list(cooperative_ids)
        self.adv_ids = list(adversarial_ids)
        self.cfg = cfg or RewardConfig()
        # 对偶变量（跨回合持续，仅 update_dual 改动）
        self.self_collision_coef = float(self.cfg.self_collision_coef_init)
        # 归因检测器（跨回合复用，reset_episode 清空历史）
        self._slowdown = SlowdownDetector(
            window_secs=self.cfg.slowdown_window_secs, timestep=self.cfg.timestep,
            drop_thresh=self.cfg.slowdown_drop_thresh)
        self._steer = SteerResponseDetector(
            window_secs=self.cfg.swerve_window_secs, timestep=self.cfg.timestep,
            jump_thresh=self.cfg.swerve_jump_thresh)
        self.reset_episode()

    def reset_episode(self) -> None:
        """回合初始化：清空封顶计数、事件计数、检测器历史、威胁窗口占比统计。"""
        self._slowdown.reset()
        self._steer.reset()
        # 每回合封顶计数（per blue）
        self._impact_counts: Dict[int, Dict[str, int]] = {
            b: {k: 0 for k in IMPACT_KEYS} for b in self.adv_ids}
        self._threat_gated_acc: Dict[int, float] = {b: 0.0 for b in self.adv_ids}
        # 威胁窗口占比统计（供归因超时判定 + TB）
        self._threat_steps: Dict[int, int] = {b: 0 for b in self.adv_ids}
        self._ep_steps: int = 0
        # 碰撞去重（sticky）：一辆蓝车一回合只结算一次碰撞事件
        self._collision_settled: Dict[int, bool] = {b: False for b in self.adv_ids}
        # 边沿触发：归因事件只在"从否变是"的瞬间计一次，避免持续状态每步刷分
        self._red_slowing_prev: Dict[int, bool] = {r: False for r in self.coop_ids}
        self._red_swerve_prev: Dict[int, bool] = {r: False for r in self.coop_ids}

    # ---- 内部：受每回合上限约束地累加一次 impact 事件 ----
    def _try_impact(self, bid: int, key: str, value: float, cap: int) -> float:
        if self._impact_counts[bid][key] >= cap:
            return 0.0
        self._impact_counts[bid][key] += 1
        return value

    def step(self, env, reward_details: Dict, control_dict: Dict,
             step_idx: int, collisions: Dict[int, bool],
             kinematic_snapshots: Optional[Dict[int, Dict]] = None,
             ) -> "tuple[Dict[int,float], Dict[int,Dict]]":
        """计算这一步每辆蓝车的对抗奖励 + 分项明细。

        参数:
          env            : CarlaMultiAgentEnv
          reward_details : env.step 返回 info['reward_details']（须 reward_version='v2'）
          control_dict   : 本步下发的 {id: carla.VehicleControl}（取红车 steer 用）
          step_idx       : 回合内步序（从 0 起），用于出生保护期
          collisions     : 本步 {id: bool} 碰撞标志（info['collisions']）
          kinematic_snapshots: info['pre_collision_kinematics']；碰撞帧优先于 actor，
                               保证已 despawn 蓝车仍能结算终局奖励

        返回: (rewards, details)
          rewards[bid]   = w_I·impact + w_T·threat_gated + w_B·base_safe − λ·c_self
          details[bid]   = 各分项 dict（供 TB / hack 仪表盘）
        """
        cfg = self.cfg
        self._ep_steps += 1
        kins = extract_kinematics(
            env, self.coop_ids + self.adv_ids, control_dict, kinematic_snapshots
        )
        self._slowdown.update({r: kins[r] for r in self.coop_ids if r in kins})
        self._steer.update({r: kins[r] for r in self.coop_ids if r in kins})

        threat_scores = compute_blue_threat_scores(
            env, self.coop_ids, self.adv_ids,
            threat_radius=cfg.threat_radius, threat_lateral=cfg.threat_lateral,
            threat_tau=cfg.threat_tau, threat_near=cfg.threat_near,
            kinematics=kins,
        )
        birth_lock = birth_protection_active(step_idx, cfg.timestep, cfg.birth_protect_secs)

        rewards: Dict[int, float] = {}
        details: Dict[int, Dict] = {}
        for bid in self.adv_ids:
            if bid not in kins:
                continue
            r, d = self._step_one_blue(
                bid, kins, reward_details, threat_scores, collisions, birth_lock)
            rewards[bid] = r
            details[bid] = d

        # 更新红车边沿状态（放在蓝车结算之后，供下一步 rising-edge 判定）
        for rid in self.coop_ids:
            self._red_slowing_prev[rid] = self._slowdown.is_slowing(rid)
            self._red_swerve_prev[rid] = self._steer.is_swerving(rid)
        return rewards, details

    def _step_one_blue(self, bid: int, kins: Dict[int, AgentKinematics],
                       reward_details: Dict, threat_scores: Dict[int, float],
                       collisions: Dict[int, bool], birth_lock: bool):
        cfg = self.cfg
        blue_k = kins[bid]

        # 该蓝车此步威胁/影响到的红车集合（在其威胁窗口内、且红车未到点）
        reds_in_window = [
            rid for rid in self.coop_ids
            if rid in kins and not kins[rid].reached
            and in_threat_window(kins[rid], blue_k,
                                 cfg.threat_radius, cfg.threat_lateral, near=cfg.threat_near)
        ]
        if reds_in_window:
            self._threat_steps[bid] += 1

        # ── (1) base_safe：基础安全驾驶（剔除目标导向项）──
        base_safe = extract_base_driving_safe(reward_details, bid)

        # ── (2) r_threat_gated：几何威胁分 × 门控(红车正在响应) × 每回合封顶 ──
        raw_threat = float(threat_scores.get(bid, 0.0))
        gate = any(red_is_responding(rid, self._slowdown, self._steer)
                   for rid in reds_in_window)
        gated_threat = raw_threat if gate else 0.0
        room = max(0.0, cfg.threat_cap_per_episode - self._threat_gated_acc[bid])
        gated_threat = min(gated_threat, room)
        self._threat_gated_acc[bid] += gated_threat

        # ── (3) r_impact：稀疏归因事件（出生保护期内一律不给分）──
        impact = 0.0
        impact_fired: Dict[str, float] = {}
        if not birth_lock:
            impact += self._impact_slowdown_swerve(bid, reds_in_window, impact_fired)
        # 碰撞事件（碰撞发生就结算，不受出生保护期约束——撞了就是撞了）
        impact += self._impact_collision(bid, kins, collisions, impact_fired)

        # ── (4) c_self：自身碰撞/违规约束（负值），λ_t 自适应 ──
        c_self = extract_collision_component(reward_details, bid)   # ≤ 0
        self_coll_term = self.self_collision_coef * c_self

        combined = (cfg.w_impact * impact
                    + cfg.w_threat * gated_threat
                    + cfg.w_base_safe * base_safe
                    + self_coll_term)
        d = {
            "combined": combined,
            "base_safe": base_safe,
            "threat_raw": raw_threat,
            "threat_gated": gated_threat,
            "threat_gate_on": 1.0 if gate else 0.0,
            "impact": impact,
            "self_coll_term": self_coll_term,
            "c_self": c_self,
            "n_reds_in_window": len(reds_in_window),
        }
        for k in IMPACT_KEYS:
            d[f"impact_{k}"] = impact_fired.get(k, 0.0)
        return combined, d

    def _impact_slowdown_swerve(self, bid: int, reds_in_window: List[int],
                                fired: Dict[str, float]) -> float:
        """归因减速 / 归因避让：红车在本蓝车窗口内且出现 rising-edge 反应时计一次。"""
        cfg = self.cfg
        total = 0.0
        for rid in reds_in_window:
            now_slow = self._slowdown.is_slowing(rid)
            if now_slow and not self._red_slowing_prev[rid]:
                v = self._try_impact(bid, "slowdown", cfg.reward_slowdown, cfg.cap_slowdown)
                total += v
                fired["slowdown"] = fired.get("slowdown", 0.0) + v
            now_swerve = self._steer.is_swerving(rid)
            if now_swerve and not self._red_swerve_prev[rid]:
                v = self._try_impact(bid, "swerve", cfg.reward_swerve, cfg.cap_swerve)
                total += v
                fired["swerve"] = fired.get("swerve", 0.0) + v
        return total

    def _impact_collision(self, bid: int, kins: Dict[int, AgentKinematics],
                          collisions: Dict[int, bool], fired: Dict[str, float]) -> float:
        """碰撞归因：本蓝车这步碰撞（一回合只结算一次）。

        induced/collateral → +reward_induced_collision（封顶）；kamikaze → 0（c_self 照罚）。
        与哪辆红车碰撞未知，取"最近的未到点红车"做几何归因（初步近似）。
        """
        if self._collision_settled[bid] or not collisions.get(bid, False):
            return 0.0
        self._collision_settled[bid] = True
        blue_k = kins[bid]
        # 选最近的未到点红车做归因
        nearest_rid, nearest_d = None, float("inf")
        for rid in self.coop_ids:
            if rid not in kins or kins[rid].reached:
                continue
            dx = kins[rid].loc[0] - blue_k.loc[0]
            dy = kins[rid].loc[1] - blue_k.loc[1]
            dist = dx * dx + dy * dy
            if dist < nearest_d:
                nearest_d, nearest_rid = dist, rid
        if nearest_rid is None:
            # 找不到红车（都到点/despawn）→ 视为连带，不奖励主动撞击
            fired["kamikaze"] = fired.get("kamikaze", 0.0)
            return 0.0
        kind = classify_collision(blue_k, kins[nearest_rid])
        if kind == COLLISION_KAMIKAZE:
            fired["kamikaze"] = fired.get("kamikaze", 0.0) + 1.0   # 计数用，奖励 0
            return 0.0
        # induced / collateral → 有效诱导碰撞
        v = self._try_impact(bid, "induced_collision",
                             self.cfg.reward_induced_collision,
                             self.cfg.cap_induced_collision)
        fired["induced_collision"] = fired.get("induced_collision", 0.0) + v
        return v

    def finalize_episode(self, timed_out: bool) -> "tuple[Dict[int,float], Dict[int,Dict]]":
        """回合末结算归因超时事件。返回 {bid: bonus}，由训练脚本加到该回合最后一步奖励上。

        归因超时：回合超时 且 本蓝车威胁窗口占比 ≥ timeout_threat_pct_min。
        """
        cfg = self.cfg
        bonus: Dict[int, float] = {}
        binfo: Dict[int, Dict] = {}
        for bid in self.adv_ids:
            pct = self._threat_steps[bid] / max(self._ep_steps, 1)
            b = 0.0
            if timed_out and pct >= cfg.timeout_threat_pct_min:
                b = self._try_impact(bid, "timeout", cfg.reward_timeout, 1)
                b *= cfg.w_impact
            bonus[bid] = b
            binfo[bid] = {"threat_window_pct": pct, "timeout_bonus": b}
        return bonus, binfo

    def update_dual(self, blue_collision_rate: float) -> float:
        """PID-I 对偶更新 λ_t（self_collision_coef）：自碰率超预算就加大，反之放松。"""
        cfg = self.cfg
        if blue_collision_rate > cfg.target_blue_collision_rate:
            self.self_collision_coef *= cfg.dual_lr_up
        else:
            self.self_collision_coef *= cfg.dual_lr_down
        self.self_collision_coef = float(np.clip(
            self.self_collision_coef, 0.1, cfg.self_collision_coef_max))
        return self.self_collision_coef

    def episode_threat_pct(self, bid: int) -> float:
        """本回合该蓝车威胁窗口占比（供 TB / 归因超时诊断）。"""
        return self._threat_steps[bid] / max(self._ep_steps, 1)
