"""Frozen teachers; gates use paired gate cases, never final test outcomes."""
import copy
import numpy as np


def distillation_weights(method, families, evidence, *, seed, round_id, team):
    if method == "off":
        return np.zeros(len(families), np.float32)
    if method == "plain":
        return np.ones(len(families), np.float32)
    weights = np.asarray([evidence.get(f, 0) for f in families], np.float32)
    if method == "random":
        rng = np.random.default_rng(seed + round_id * 1009 + (0 if team == "A" else 1))
        return weights[rng.permutation(len(weights))]
    if method == "strength":
        return np.full_like(weights, float(weights.mean()))
    if method != "selective":
        raise ValueError("unknown mutual-learning control")
    return weights


def snapshots(models):
    result = {name: copy.deepcopy(model).eval() for name, model in models.items()}
    for model in result.values():
        model.requires_grad_(False)
    return result


def compute_gates(reports, config):
    gates = {"A": {}, "B": {}}
    for student, teacher in (("A", "B"), ("B", "A")):
        for family in reports[student]["families"]:
            own, peer = (reports[k]["families"][family] for k in (student, teacher))
            if own["case_keys"] != peer["case_keys"]:
                raise ValueError("gate cases must be paired")
            if own.get('case_initial_hashes')!=peer.get('case_initial_hashes'):
                raise ValueError('gate cases have different initial states/controllers')
            gates[student][family] = float(
                peer["episodes"] >= config.gate_cases_per_family
                and peer["bad_rate"] <= config.gate_max_bad_rate
                and peer["bad_rate"] <= own["bad_rate"]
                and peer["quality"] - own["quality"] >= config.gate_margin
                and (config.task.research_protocol == "legacy_v2" or
                     (peer["success_rate"] >= config.task.selection_min_qualified
                      and peer.get("baseline_invalid_count", 0) == 0)))
    return gates
