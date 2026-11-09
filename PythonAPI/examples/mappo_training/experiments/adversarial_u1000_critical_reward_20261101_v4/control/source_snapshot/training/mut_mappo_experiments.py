"""Explicit algorithm/ablation registry over one environment and evaluator.

Algorithm names describe the actual critic and learner topology, not just
plot labels. Configs are candidates; generation never launches training.
"""
from __future__ import annotations
import copy

GROUPS = {
    "MAPPO": {"algorithm": "mappo", "team_count": 1, "method": "off", "gate": "evidence",
              "question": "single-team centralized-value performance baseline"},
    "IPPO": {"algorithm": "ippo", "team_count": 1, "method": "off", "gate": "evidence",
             "question": "independent local-value baseline with per-role optimizer/ValueNorm"},
    "B0": {"algorithm": "mappo", "team_count": 2, "method": "off", "gate": "evidence",
           "question": "dual independent MAPPO; matches training networks and selection opportunities"},
    "B1": {"algorithm": "mappo", "team_count": 2, "method": "plain", "gate": "evidence",
           "question": "ordinary policy distribution mutual distillation"},
    "B2": {"algorithm": "mappo", "team_count": 2, "method": "selective", "gate": "evidence",
           "question": "scene-quality/safety evidence selection"},
    "G_RANDOM": {"algorithm": "mappo", "team_count": 2, "method": "selective", "gate": "random",
                 "question": "shuffle this update's evidence weights, preserving each role's enabled count"},
    "G_STRENGTH": {"algorithm": "mappo", "team_count": 2, "method": "selective", "gate": "strength",
                   "question": "uniform distillation weight equal to this update's evidence fraction"},
}
SUITES = {"core": ("B0", "B1", "B2"),
          "comparisons": ("MAPPO", "IPPO", "B0", "B1", "B2"),
          "controls": ("G_RANDOM", "G_STRENGTH"), "all": tuple(GROUPS)}


def experiment_config(base, group):
    if group not in GROUPS:
        raise ValueError(f"unsupported implemented algorithm group {group}")
    result = copy.deepcopy(base)
    spec = GROUPS[group]
    result.update(algorithm=spec["algorithm"], team_count=spec["team_count"], method=spec["method"])
    distillation = result["learning"].setdefault("distillation", {})
    distillation.setdefault("direction", "student_to_teacher")
    distillation.setdefault("reduction", "all_valid")
    distillation["gate_variant"] = spec["gate"]
    # Common actual training-cost clock across single/dual learners.  Refresh
    # occurs at a complete boundary after the threshold; report quantization.
    result["run"]["schedule_clock"] = "train_ticks"
    result["run"]["stop_rule"] = "train_ticks"
    result["mutual"].update(schedule_clock="train_ticks", warmup_train_ticks=10000,
                              refresh_train_ticks=20000)
    return result
