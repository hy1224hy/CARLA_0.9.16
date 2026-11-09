"""C1-relative progress reward; no simulator, optimizer or learned credit model."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

PROFILE = "c1_progress_team_v1"
CRITICAL_PROFILES = frozenset(("critical_interaction_team_v1", "normal_driving_critical_team_v2", "individual_driving_critical_v3", "individual_three_dimension_risk_v4", "individual_red_outcome_v5"))
TEAM_PROFILES = CRITICAL_PROFILES | {PROFILE}
ROOT = Path(__file__).resolve().parents[1]


def reference_identity(config):
    task = {k: v for k, v in config["task"].items()
            if k not in {"team_reward", "follow_spectator", "cameras_enabled"}}
    return {"red_sha256": config["red_checkpoint"]["sha256"],
            "case_manifest_sha256": config["_root_validation"]["case_manifest_sha256"],
            "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()}


class ReferenceStore:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        if not self.directory.is_absolute():
            self.directory = ROOT / self.directory
        header = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if header["identity"] != identity:
            raise ValueError("C1 reference belongs to a different red policy, task or case list")
        self.records = header["cases"]
        self.cache = {}

    def curve(self, case_key):
        if case_key not in self.cache:
            # Case keys are registered manifest IDs; digest prevents path interpretation.
            path = self.directory / (hashlib.sha256(case_key.encode()).hexdigest() + ".json")
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.records[case_key]["sha256"]:
                raise ValueError("C1 reference curve changed after preparation")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["case_key"] != case_key:
                raise ValueError("reference case identity differs")
            self.cache[case_key] = dict(record["progress"])
            if "response_states" in record:
                self.cache[case_key]["response_states"] = record["response_states"]
            if "outcome_reference" in record:
                self.cache[case_key]["outcome_reference"] = record["outcome_reference"]
        return self.cache[case_key]


class TeamEfficiencyReward:
    def __init__(self, settings=None, *, dt=0.1, horizon=500, gamma=0.999):
        self.settings = {"attack_weight": 10.0, "driving_weight": 1.0,
                         "collision_cost": 50.0, "offroad_cost": 50.0,
                         "road_rule_cost": 50.0, "task_violation_cost": 20.0,
                         **dict(settings or {})}
        self.dt, self.horizon, self.gamma = dt, horizon, gamma
        self.reset()

    def reset(self):
        self.previous_blue = {2: 0.0, 3: 0.0}
        self.seen_bad = set()
        self.attack_integral = self.utility_total = self.cost_total = 0.0
        self.components_total = {}

    def step(self, *, step, progress, reached, eligible, safety, costs, base_details, reference):
        # Successful completion is an absorbing task fact, not zero-speed failure.
        progress = {i: (1.0 if reached.get(i, False) else float(p)) for i, p in progress.items()}
        attack = 0.0
        if reference is not None and eligible:
            for rid in (0, 1):
                curve = reference[str(rid)]
                target = float(curve[min(step, len(curve) - 1)])
                attack += target - progress[rid]
            attack /= 2 * self.horizon
        absorbing_tail, discounted_tail = 0.0, 0.0
        if reference is not None and eligible and all(reached.get(r, False) for r in (0, 1)):
            # A faster completion still includes its known completed-task tail.
            # No physical observations or actions are invented after departure.
            for future in range(step + 1, self.horizon + 1):
                value = sum(float(reference[str(r)][min(future, len(reference[str(r)]) - 1)]) - 1.0
                            for r in (0, 1)) / (2 * self.horizon)
                absorbing_tail += value
                discounted_tail += self.gamma ** (future - step) * value
        drive = sum(self.gamma * progress.get(b, self.previous_blue[b]) - self.previous_blue[b]
                    for b in (2, 3)) / 2
        for b in (2, 3):
            self.previous_blue[b] = progress.get(b, self.previous_blue[b])
        event_cost = 0.0
        for name in ("collision", "offroad", "road_rule", "task_violation"):
            if name not in self.seen_bad and any(flags.get(name, False) for flags in safety.values()):
                event_cost += self.settings[name + "_cost"]
                self.seen_bad.add(name)
        # First team event per category; no repeated charge for latched flags.
        risk = sum(sum(float(c.get(k, 0)) for k in ("overspeed", "lidar", "ttc")) for c in costs.values()) / 2
        comfort = sum(max(0.0, -float(d.get("longitudinal_rate", 0)) - float(d.get("steer_rate", 0)))
                      for d in base_details.values()) / 2
        components = {"attack": self.settings["attack_weight"] * attack,
                      "completed_task_tail": self.settings["attack_weight"] * discounted_tail,
                      "driving": self.settings["driving_weight"] * drive,
                      "risk": -risk, "comfort": -comfort, "safety_event": -event_cost}
        utility = math.fsum(components.values())
        if not math.isfinite(utility):
            raise FloatingPointError("nonfinite team reward")
        self.attack_integral += attack + absorbing_tail
        self.utility_total += utility
        self.cost_total += event_cost + risk + comfort
        for key, value in components.items():
            self.components_total[key] = self.components_total.get(key, 0.0) + value
        return utility, components

    def metrics(self, eligible):
        return {"attack_integral": self.attack_integral,
                "safe_efficiency_utility": self.attack_integral if eligible else 0.0,
                "team_utility_total": self.utility_total, "team_cost_total": self.cost_total,
                "team_reward_components": dict(self.components_total)}


def settle_team_tail(buffer, starts, timeline, gamma):
    """Fold the true post-departure team reward into the last real action.

    This preserves each role's discounted full-team return without generating
    inactive actions. The environment reward log remains the actual step reward.
    """
    audit = {}
    for aid in (2, 3):
        rows = buffer.data[aid][starts[aid]:]
        if not rows:
            continue
        n = len(rows)
        tail = math.fsum((gamma ** (step - n + 1)) * reward
                         for step, reward in enumerate(timeline[n:], start=n))
        rows[-1].reward += tail
        audit[aid] = {"real_action_rows": n, "team_steps": len(timeline), "discounted_tail": tail}
    return audit
