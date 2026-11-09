"""Per-vehicle driving measurements shared by baseline and attack evaluation."""
from __future__ import annotations

import math
from collections import defaultdict
import numpy as np


def average(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


class DrivingMetrics:
    def __init__(self, dt=0.1, initial_speeds=None):
        self.dt = dt
        self.initial_speeds = dict(initial_speeds or {})
        self.data = defaultdict(lambda: {"speed": [], "lateral": [], "cruise": [], "low": [],
            "ttc": [], "gap": [], "headway": [], "accel": [], "jerk": [], "steer_rate": [],
            "control_samples": 0, "steps": 0, "low_run": 0, "low_max": 0,
            "unblocked_elapsed": 0.0, "first_motion": None, "previous_v": None,
            "previous_a": None, "previous_steer": None})

    def update(self, active, snapshots, diagnostics, safety, headings=None):
        headings = headings or {}
        for aid, enabled in active.items():
            if not enabled:
                continue
            snap = snapshots.get(aid, snapshots.get(str(aid), {}))
            d = diagnostics.get(aid, diagnostics.get(str(aid), {}))
            item = self.data[aid]
            item["steps"] += 1
            if not snap or "speed" not in snap:
                continue
            speed = float(snap["speed"])
            item["speed"].append(speed)
            yaw = math.radians(headings.get(aid, snap["rotation"]["yaw"]))
            velocity = snap.get("velocity", {})
            route_v = float(velocity.get("x", 0)) * math.cos(yaw) + float(velocity.get("y", 0)) * math.sin(yaw)
            if item["previous_v"] is not None:
                a = (route_v - item["previous_v"]) / self.dt
                item["accel"].append(abs(a))
                if item["previous_a"] is not None:
                    item["jerk"].append(abs(a - item["previous_a"]) / self.dt)
                item["previous_a"] = a
            item["previous_v"] = route_v
            blocked = d.get("pace_front_blocker_exempt")
            if blocked is not None and not blocked and item["first_motion"] is None:
                item["unblocked_elapsed"] += self.dt
                if speed >= 0.5:
                    item["first_motion"] = item["unblocked_elapsed"]
            free = (blocked is not None and not blocked and item["steps"] * self.dt > 2.0
                    and d.get("distance_to_goal_m", 0) > 5.0)
            if "lateral_distance_m" in d:
                item["lateral"].append(abs(float(d["lateral_distance_m"])))
            if free:
                item["cruise"].append(speed)
                item["low"].append(speed < 0.2)
            item["low_run"] = item["low_run"] + 1 if free and speed < 0.2 else 0
            item["low_max"] = max(item["low_max"], item["low_run"])
            control = snap.get("control")
            if control:
                item["control_samples"] += 1
                steer = float(control["steer"])
                if item["previous_steer"] is not None:
                    item["steer_rate"].append(abs(steer - item["previous_steer"]) / self.dt)
                item["previous_steer"] = steer
            s = safety.get(aid, safety.get(str(aid), {}))
            gap, ttc = s.get("front_gap_m"), s.get("ttc_s")
            if gap is not None:
                item["gap"].append(float(gap))
                if route_v > 0.2:
                    item["headway"].append(max(0.0, float(gap)) / route_v)
            if ttc is not None and math.isfinite(float(ttc)):
                item["ttc"].append(float(ttc))

    def summary(self):
        def quantile(values, q):
            return float(np.quantile(values, q)) if values else None
        result = {}
        for aid, item in self.data.items():
            speed, cruise, n = item["speed"], item["cruise"], item["steps"]
            stationary = self.initial_speeds.get(aid, self.initial_speeds.get(str(aid)))
            stationary = stationary < 0.5 if stationary is not None else None
            result[str(aid)] = {"active_steps": n, "mean_speed_mps": average(speed),
                "peak_speed_mps": max(speed) if speed else None, "startup_applicable": stationary,
                "measured_state_fraction": len(speed) / n,
                "unblocked_first_motion_seconds": item["first_motion"] if stationary else None,
                "preferred_speed_fraction": average([2 <= v <= 15 / 3.6 for v in cruise]),
                "unblocked_low_speed_fraction": average(item["low"]),
                "unblocked_low_speed_longest_seconds": item["low_max"] * self.dt,
                "above_preferred_speed_fraction": average([v > 15 / 3.6 for v in speed]),
                "reference_lateral_p95_m": quantile(item["lateral"], .95),
                "reference_lateral_peak_m": max(item["lateral"]) if item["lateral"] else None,
                "actual_control_fraction": item["control_samples"] / n,
                "actual_steer_rate_p95_per_second": quantile(item["steer_rate"], .95),
                "abs_acceleration_p95_mps2": quantile(item["accel"], .95),
                "abs_jerk_p95_mps3": quantile(item["jerk"], .95),
                "minimum_net_gap_m": min(item["gap"]) if item["gap"] else None,
                "minimum_closing_ttc_s": min(item["ttc"]) if item["ttc"] else None,
                "closing_ttc_below_3s_fraction": average([v < 3 for v in item["ttc"]]),
                "minimum_moving_headway_s": min(item["headway"]) if item["headway"] else None,
                "longitudinal_exposure_samples": len(item["gap"]),
                "comfort_definition": "raw 0.1s road-tangent finite differences; no calibrated comfort pass threshold"}
        return result


def summarize_driving(records):
    """Aggregate measured episode statistics, retaining missing exposure as None."""
    result = {}
    for aid in ("0", "1", "2", "3"):
        rows = [row["driving_metrics"][aid] for row in records
                if row.get("valid") and aid in row.get("driving_metrics", {})]
        if not rows:
            continue
        numeric = {k for row in rows for k, value in row.items() if isinstance(value, (int, float)) or value is None}
        result[aid] = {k + "_episode_mean": average([row.get(k) for row in rows]) for k in numeric}
        result[aid]["episodes"] = len(rows)
    return result


def summarize_base_evaluation(directory):
    """Add comfort measurements to the existing frozen base qualification output."""
    import json
    from pathlib import Path
    from training.checkpoint_protocol import atomic_write_json
    directory = Path(directory)
    measured = []
    for path in sorted((directory / "traces").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        initial = rows[0].get("kinematics", {})
        meter = DrivingMetrics(.1, {aid: value["speed"] for aid, value in initial.items() if value})
        active = rows[0]["info"]["active_mask"]
        headings = {aid: value["yaw"] for aid, value in initial.items() if value}
        for row in rows[1:]:
            snapshots = {aid: {"speed": value["speed"], "rotation": {"yaw": value["yaw"]},
                              "velocity": {"x": value["vx"], "y": value["vy"]}, "control": value.get("control")}
                         for aid, value in row.get("post", {}).items() if value}
            meter.update(active, snapshots, row["info"].get("reward_diagnostics", {}),
                         row["info"].get("longitudinal_safety", {}), headings)
            active = row["info"]["active_mask"]
        measured.append({"case": rows[0]["case"], "valid": True, "driving_metrics": meter.summary()})
    result = {"cases": measured, "summary": summarize_driving(measured),
              "scope": "additional measured comfort/efficiency; original qualification thresholds unchanged; unobserved TTC remains None"}
    atomic_write_json(result, directory / "driving_measurements.json")
    return result
