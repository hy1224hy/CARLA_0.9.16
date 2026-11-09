"""All-case safety qualification; censored failures never earn attack utility."""
from dataclasses import dataclass
import numpy as np
from .contract import BLUE_IDS, RED_IDS


class OutcomeTracker:
    def __init__(self, task, *, red_ids=RED_IDS, blue_ids=BLUE_IDS, goal_distances=None):
        self.task = task
        self.red_ids, self.blue_ids = tuple(red_ids), tuple(blue_ids)
        self.goal_distances = goal_distances or {}
        self.collision = self.offroad = self.overspeed = self.wrong_way = False
        self.last = None
        self.over_seconds = {i: 0.0 for i in self.blue_ids}
        self.stopped_seconds = {i: 0.0 for i in self.blue_ids}
        self.hard_brake_events = 0
        self.braking = {i: False for i in self.red_ids}
        self.min_ttc = 100.0

    def update(self, frame):
        dt = frame.time - self.last.time if self.last is not None else 0
        if dt < -1e-9:
            raise ValueError("physical time must be monotone")
        self.collision |= any(v.collision for v in frame.vehicles.values())
        self.offroad |= any(v.offroad for v in frame.vehicles.values())
        for identity in self.blue_ids:
            if identity not in frame.vehicles:
                continue
            v = frame.vehicles[identity]
            if v.speed < 1.0:
                self.stopped_seconds[identity] += dt
            self.over_seconds[identity] = self.over_seconds[identity] + dt if v.speed > self.task.blue_speed_limit else 0
            self.overspeed |= self.over_seconds[identity] > self.task.overspeed_grace_s
            self.wrong_way |= np.cos(v.heading - v.route_heading) < 0
            axis = np.array([np.cos(v.route_heading), np.sin(v.route_heading)])
            self.wrong_way |= np.asarray(v.velocity) @ axis < -0.2
        if self.last is not None and dt > 0:
            for identity in self.red_ids:
                acceleration = (frame.vehicles[identity].speed - self.last.vehicles[identity].speed) / dt
                braking = acceleration < -3
                self.hard_brake_events += int(braking and not self.braking[identity])
                self.braking[identity] = braking
        for red_id in self.red_ids:
            red = frame.vehicles[red_id]
            axis = np.array([np.cos(red.route_heading), np.sin(red.route_heading)])
            lateral = np.array([-axis[1], axis[0]])
            for blue_id in self.blue_ids:
                if blue_id not in frame.vehicles:
                    continue
                blue = frame.vehicles[blue_id]
                dp = np.subtract(blue.position, red.position)
                longitudinal = float(dp @ axis)
                closing = float(np.subtract(red.velocity, blue.velocity) @ axis) * np.sign(longitudinal)
                if abs(dp @ lateral) < 2 and closing > 0.1:
                    gap = max(abs(longitudinal) - (red.length + blue.length) / 2, 0)
                    self.min_ttc = min(self.min_ttc, gap / closing)
        self.last = frame

    @property
    def bad(self):
        return bool(self.collision or self.offroad or self.overspeed or self.wrong_way)

    def summary(self):
        if self.last is None:
            raise ValueError("no observations")
        blue = [self.last.vehicles[i] for i in self.blue_ids if i in self.last.vehicles]
        complete = self.last.time >= self.task.duration - 1e-6
        goals = len(blue) == len(self.blue_ids) and all(v.progress >= self.goal_distances.get(v.identity,self.task.goal_distance) for v in blue)
        diagnostics = {'blue_min_progress_m': min((v.progress for v in blue),default=None)}
        for role, identity in enumerate(self.blue_ids):
            if identity in self.last.vehicles:
                v=self.last.vehicles[identity]
                diagnostics.update({f'blue_{role}_progress_m':float(v.progress),
                    f'blue_{role}_speed_mps':v.speed,f'blue_{role}_stopped_s':self.stopped_seconds[identity]})
        return {**diagnostics, "elapsed_s": float(self.last.time), "fixed_horizon_complete": complete,
                "collision": bool(self.collision), "offroad": bool(self.offroad),
                "overspeed": bool(self.overspeed), "wrong_way": bool(self.wrong_way), "bad": self.bad,
                "blue_goals_met": goals, "qualified": bool(complete and goals and not self.bad),
                "red_progress_m": float(np.mean([self.last.vehicles[i].progress for i in self.red_ids])),
                "blue_progress_m": float(np.mean([v.progress for v in blue])) if blue else None,
                "hard_brake_events": int(self.hard_brake_events), "min_ttc_s": float(self.min_ttc)}


def paired_outcome(c0, c1, c2, task):
    baseline_valid = c0["fixed_horizon_complete"] and not c0["bad"] and c1["qualified"]
    qualified = bool(baseline_valid and c2["qualified"] and c2["fixed_horizon_complete"] and not c2["bad"])
    # A failure ending at e.g. 3s must not be compared to reference progress at20s.
    raw = c1["red_progress_m"] - c2["red_progress_m"] if c2["fixed_horizon_complete"] else None
    utility = float(np.clip(raw / task.goal_distance, 0, 1)) if qualified else 0.0
    return {"baseline_valid": bool(baseline_valid), "qualified": qualified,
            "c2_censored": not c2["fixed_horizon_complete"], "extra_red_loss_m": raw,
            "qualified_attack_utility": utility,
            "qualified_extra_red_loss_m": max(raw, 0.0) if qualified else 0.0,
            "natural_traffic_loss_m": c0["red_progress_m"] - c1["red_progress_m"],
            "total_red_loss_m": c0["red_progress_m"] - c2["red_progress_m"] if raw is not None else None}
