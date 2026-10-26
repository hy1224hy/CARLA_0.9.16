"""CARLA-independent candidate Mut-MAPPO evidence and reward contract.

The old attribution geometry is reused; physical-contact blame heuristics and
the old global-timeout bonus are deliberately not used. Candidate constants
must be frozen after the pilot before making research comparisons.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
import math
from typing import Mapping

from utils.adversarial_metrics import AgentKinematics, SteerResponseDetector, in_threat_window, relative_long_lat


SAFETY_KEYS = ("collision", "offroad", "road_rule", "task_violation")
BLUE_BASE_COMPONENTS = ("progress", "success", "longitudinal_rate", "steer_rate")
BLUE_BASE_RISK_COMPONENTS = ("safety_lidar", "ttc")


@dataclass(frozen=True)
class MutRewardConfig:
    timestep: float = 0.1
    birth_protect_secs: float = 3.0
    slowdown_window_secs: float = 2.0
    slowdown_drop_thresh: float = 2.0
    slowdown_persistence_secs: float = 0.5
    threat_near: float = 3.0
    threat_radius: float = 15.0
    threat_lateral: float = 2.5
    reward_slowdown: float = 5.0
    cap_slowdown: int = 3
    reward_timeout: float = 15.0
    threat_cap_per_episode: float = 20.0
    threat_weight: float = 1.0
    base_driving_weight: float = 0.3
    safety_lambda: float = 1.0
    collision_cost: float = 50.0
    offroad_cost: float = 50.0
    road_rule_cost: float = 50.0
    task_violation_cost: float = 20.0
    overspeed_start_mps: float = 15.0 / 3.6
    overspeed_weight: float = 0.005
    free_flow_target_mps: float = 3.0
    free_flow_low_speed_weight: float = 0.01

    @classmethod
    def from_dict(cls, values: Mapping | None = None, *, timestep: float = 0.1):
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown Mut reward fields: {sorted(unknown)}")
        values.setdefault("timestep", timestep)
        result = cls(**values)
        for name in known:
            value = getattr(result, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if result.timestep <= 0 or result.slowdown_window_secs <= 0:
            raise ValueError("timestep and slowdown window must be positive")
        if isinstance(result.cap_slowdown, bool) or int(result.cap_slowdown) != result.cap_slowdown:
            raise ValueError("cap_slowdown must be an integer")
        if result.threat_radius <= result.threat_near:
            raise ValueError("threat_radius must exceed threat_near")
        return result


class MutEpisodeEvents:
    """One shared detector per four-car episode, with red-centric de-duplication."""

    def __init__(self, config: MutRewardConfig | None = None,
                 red_ids=(0, 1), blue_ids=(2, 3)):
        self.config = config or MutRewardConfig.from_dict()
        self.red_ids, self.blue_ids = tuple(red_ids), tuple(blue_ids)
        self.reset()

    def reset(self):
        n = max(2, int(math.ceil(self.config.slowdown_window_secs / self.config.timestep)) + 1)
        self._speed = {r: deque(maxlen=n) for r in self.red_ids}
        self._persistent = {r: 0 for r in self.red_ids}
        self._responding = {r: False for r in self.red_ids}
        self._event_counts = {r: 0 for r in self.red_ids}
        self.safety = {b: {key: False for key in SAFETY_KEYS} for b in self.blue_ids}
        self.response_events = []
        self.threat_total = 0.0
        self._timeout_settled = False
        self._last_step = -1
        self._steer = SteerResponseDetector(timestep=self.config.timestep)
        self.steer_diagnostics = []
        self._steer_prev = {r: False for r in self.red_ids}

    @property
    def blue_eligible(self):
        return not any(any(v.values()) for v in self.safety.values())

    def step(self, *, step: int, kinematics: Mapping[int, AgentKinematics],
             safety_flags: Mapping[int, Mapping[str, bool]] | None = None,
             near_goal: Mapping[int, bool] | None = None,
             frame: int = -1):
        if step <= self._last_step:
            raise ValueError("event detector requires strictly increasing decision steps")
        self._last_step = step
        cfg = self.config
        for bid in self.blue_ids:
            for name in SAFETY_KEYS:
                self.safety[bid][name] |= bool((safety_flags or {}).get(bid, {}).get(name, False))
        eligible = self.blue_eligible
        protected = step * cfg.timestep <= cfg.birth_protect_secs + 1e-9
        team_impact, team_threat = 0.0, 0.0
        current_events = []
        self._steer.update({rid: kinematics[rid] for rid in self.red_ids if rid in kinematics})
        for rid in self.red_ids:
            red = kinematics.get(rid)
            swerving = self._steer.is_swerving(rid)
            if swerving and not self._steer_prev[rid]:
                self.steer_diagnostics.append({"red_id": rid, "step": step, "frame": frame,
                                               "kind": "steer_jump_diagnostic", "attack_credit": 0.0})
            self._steer_prev[rid] = swerving
            if (protected or red is None or not red.alive or red.reached
                    or (near_goal or {}).get(rid, False)):
                self._speed[rid].clear()
                self._persistent[rid] = 0
                self._responding[rid] = False
                continue
            self._speed[rid].append(float(red.fwd_speed))
            blues = [bid for bid in self.blue_ids if bid in kinematics
                     and kinematics[bid].alive and not kinematics[bid].reached
                     and not any(self.safety[bid].values())
                     and in_threat_window(red, kinematics[bid],
                         threat_radius=cfg.threat_radius,
                         threat_lateral=cfg.threat_lateral, near=cfg.threat_near)]
            drop = max(self._speed[rid], default=red.fwd_speed) - red.fwd_speed
            raw = eligible and bool(blues) and drop >= cfg.slowdown_drop_thresh
            self._persistent[rid] = self._persistent[rid] + 1 if raw else 0
            response = (self._persistent[rid] * cfg.timestep
                        + 1e-9 >= cfg.slowdown_persistence_secs) and raw
            if response and not self._responding[rid] and self._event_counts[rid] < cfg.cap_slowdown:
                event = {"kind": "slowdown", "red_id": rid, "blue_ids": blues,
                         "step": step, "frame": frame, "seconds": step * cfg.timestep,
                         "speed_drop_mps": float(drop),
                         "persistent_seconds": self._persistent[rid] * cfg.timestep,
                         "blame": "unknown", "proxy_only": True}
                self._event_counts[rid] += 1
                self.response_events.append(event)
                current_events.append(event)
                team_impact += cfg.reward_slowdown
            if response:
                # One maximum influence per red, even when both blues overlap.
                score = max((1.0 - (relative_long_lat(red, kinematics[b])[0] - cfg.threat_near)
                             / (cfg.threat_radius - cfg.threat_near)) for b in blues)
                proposed = max(0.0, score) * cfg.threat_weight * cfg.timestep
                increment = min(proposed, max(0.0, cfg.threat_cap_per_episode - self.threat_total))
                self.threat_total += increment
                team_threat += increment
            self._responding[rid] = response
        return {"team_impact": team_impact, "team_threat": team_threat,
                "events": current_events, "birth_protected": protected,
                "blue_eligible": eligible}

    def finalize(self, *, end_reason: str, unfinished_red_ids=(), step: int):
        """Only a genuine deadline linked to prior response evidence earns a bonus."""
        if self._timeout_settled:
            return 0.0
        self._timeout_settled = True
        linked = any(e["red_id"] in unfinished_red_ids for e in self.response_events)
        if (end_reason == "mission_timeout" and linked and self.blue_eligible
                and step * self.config.timestep > self.config.birth_protect_secs + 1e-9):
            return self.config.reward_timeout
        return 0.0

    def metrics(self):
        safety = {bid: dict(values) for bid, values in self.safety.items()}
        return {"blue_safe": self.blue_eligible,
                "blue_eligible": self.blue_eligible,
                # Raw response evidence remains visible when a later unsafe
                # blue event makes the episode ineligible for q/teacher use.
                "response": bool(self.response_events),
                "qualified_response": bool(self.response_events) and self.blue_eligible,
                "response_events": list(self.response_events),
                "response_event_count": len(self.response_events),
                "steer_diagnostics": list(self.steer_diagnostics),
                "safety": {"per_agent": safety,
                           **{key: any(v[key] for v in safety.values()) for key in SAFETY_KEYS}},
                "threat_reward_total": self.threat_total}


def blue_driving_reward(*, base_details: Mapping[str, float], speed_mps: float,
                        route_speed_mps: float, front_blocker: bool, near_goal: bool,
                        terminal: bool, safety_flags: Mapping[str, bool],
                        config: MutRewardConfig):
    """Return route/comfort reward and explicit, nonnegative safety costs.

    Blue pace uses present speed only, and overspeed grows without a plateau.
    No private red return is subtracted. Historical blue collision bonuses are
    absent, including the induced/collateral collision categories.
    """
    drive = config.base_driving_weight * math.fsum(
        float(base_details.get(name, 0.0)) for name in
        BLUE_BASE_COMPONENTS)
    if not (front_blocker or near_goal or terminal):
        deficit = max(0.0, config.free_flow_target_mps - route_speed_mps)
        drive -= config.free_flow_low_speed_weight * deficit * config.timestep
    costs = {name: float(bool(safety_flags.get(name, False))) * getattr(config, name + "_cost")
             for name in SAFETY_KEYS}
    costs["overspeed"] = config.overspeed_weight * max(0.0, speed_mps - config.overspeed_start_mps) ** 2
    costs["lidar"] = max(0.0, -float(base_details.get("safety_lidar", 0.0)))
    costs["ttc"] = max(0.0, -float(base_details.get("ttc", 0.0)))
    if not math.isfinite(drive) or any(not math.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError("blue reward contains invalid or negative costs")
    return drive, costs
