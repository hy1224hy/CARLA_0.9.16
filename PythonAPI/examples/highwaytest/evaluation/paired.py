"""C0/C1/C2 are independent matched trials; the final test is never used for gates."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
import torch
from ..common.contract import PROTOCOL, inputs
from ..common.outcomes import paired_outcome
from ..environment.scenarios import cases, FAMILIES


@torch.no_grad()
def run_trial(backend, model, task, *, intervention="normal", drop_probability=0,
              seed=0, capture=False):
    frames = []
    prior_obs = None
    rng = np.random.default_rng(seed + 11_000_000)
    initial = {str(i): [*v.position, *v.velocity, v.heading] for i, v in backend.frame.vehicles.items()}
    try:
        while backend.frame.time < task.duration - 1e-6 and not backend.tracker.bad:
            actions = None
            if model is not None:
                obs, _, links = backend.policy_inputs() if hasattr(backend,'policy_inputs') else inputs(backend.frame, task)
                if intervention == "drop":
                    links *= rng.random(links.shape) >= drop_probability
                distribution = model.distribution(torch.as_tensor(obs), torch.as_tensor(links),
                    intervention=intervention,
                    stale_obs=torch.as_tensor(obs if prior_obs is None else prior_obs))
                actions = distribution.loc.tanh().numpy()
                prior_obs = obs.copy()
            backend.step(actions)
            if capture:
                frames.append(backend.render())
        result = backend.tracker.summary()
        result["return"] = float(backend.total_return) if backend.total_return is not None else None
        result["initial_state"] = initial
        if hasattr(backend,'initial_contract'):
            result['initial_contract']=backend.initial_contract
        return result, frames
    finally:
        backend.close()


def aggregate(rows):
    fields = ("success", "bad", "collision", "offroad", "overspeed", "wrong_way", "quality",
              "qualified_attack_utility", "qualified_extra_red_loss_m", "baseline_valid",
              "blue_progress_m", "red_delay_m", "min_ttc_s", "hard_brakes", "return", "episode_steps",
              "safe_event", "qualified_event", "c2_censored", 'blue_min_progress_m',
              'blue_0_progress_m','blue_1_progress_m','blue_0_speed_mps','blue_1_speed_mps',
              'blue_0_stopped_s','blue_1_stopped_s','blue_min_progress_retention',
              'blue_0_progress_retention','blue_1_progress_retention')
    rates = {"success", "bad", "collision", "offroad", "overspeed", "wrong_way", "safe_event", "qualified_event", "baseline_valid", "c2_censored"}
    result = {}
    for key in fields:
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key + ("_rate" if key in rates else "")] = float(np.mean(values)) if values else None
    result.update(episodes=len(rows), case_keys=[r["case_key"] for r in rows],
                  case_initial_hashes=[r.get('initial_state_sha256') for r in rows],
                  baseline_invalid_count=sum(not r["baseline_valid"] for r in rows),
                  # red_delay_m diagnostic excludes censored cases; denominator explicit.
                  red_delay_observed_count=sum(r["red_delay_m"] is not None for r in rows))
    return result


def evaluate_paired(model, task, split, per_family, *, gamma=0.99, normal_model=None,
                    intervention="normal", drop_probability=0, capture=False, backend_factory=None, backend_contract=None,
                    case_list=None, red_ids=(0,1), blue_ids=(2,3)):
    if task.research_protocol not in (PROTOCOL,'bottleneck_paired_attack_v4'):
        raise ValueError("paired protocol requires a newly trained v3 policy, not v2 weights")
    if intervention not in ("normal", "zero", "stale", "drop") or not 0 <= drop_probability <= 1:
        raise ValueError("invalid communication intervention")
    backend_contract = backend_contract or {"backend": "highway" if backend_factory is None else "external"}
    if backend_factory is None:
        if task.research_protocol != PROTOCOL:
            raise ValueError('v4 requires its explicit three-car backend')
        from ..environment.conditions import HighwayConditionBackend
        backend_factory = lambda condition, family, seed: HighwayConditionBackend(
            task, condition, family, seed, gamma=gamma, learned_normal=normal_model is not None)
    rows, preview = [], []
    case_list=list(cases(split,per_family) if case_list is None else case_list)
    for family, seed in case_list:
        trials = {}
        for condition, policy in (("C0", None), ("C1", normal_model), ("C2", model)):
            trials[condition], images = run_trial(backend_factory(condition, family, seed), policy, task,
                intervention=intervention if condition == "C2" else "normal",
                drop_probability=drop_probability if condition == "C2" else 0,
                seed=seed, capture=capture and not rows and condition == "C2")
            if images:
                preview = images
        for identity in map(str,red_ids):
            if any(not np.allclose(trials["C0"]["initial_state"][identity], trials[c]["initial_state"][identity], atol=1e-3, rtol=0)
                   for c in ("C1", "C2")):
                raise ValueError("C0/C1/C2 red initial states are not paired")
        for identity in map(str,blue_ids):
            if not np.allclose(trials["C1"]["initial_state"][identity], trials["C2"]["initial_state"][identity], atol=1e-3, rtol=0):
                raise ValueError("normal/attack blue initial states are not paired")
        if any('initial_contract' in r for r in trials.values()):
            if any(r.get('initial_contract')!=trials['C0'].get('initial_contract') for r in trials.values()):
                raise ValueError('paired controller/geometry initial contract mismatch')
        paired = paired_outcome(trials["C0"], trials["C1"], trials["C2"], task)
        c2 = trials["C2"]
        rows.append({"case_key": f"{family}:{seed}", "family": family,
            'initial_state_sha256':hashlib.sha256(json.dumps({'state':c2['initial_state'],
                'contract':c2.get('initial_contract')},sort_keys=True).encode()).hexdigest(),
            **{k:v for k,v in c2.items() if k.startswith('blue_') and k != 'blue_goals_met'},
            "success": c2["qualified"], "bad": c2["bad"], "collision": c2["collision"],
            "offroad": c2["offroad"], "overspeed": c2["overspeed"], "wrong_way": c2["wrong_way"],
            "quality": (float(c2["qualified"]) if task.attack_weight == 0 else paired["qualified_attack_utility"]),
            "blue_progress_m": c2["blue_progress_m"], "red_delay_m": paired["extra_red_loss_m"],
            "min_ttc_s": c2["min_ttc_s"], "hard_brakes": c2["hard_brake_events"], "return": c2["return"],
            "episode_steps": int(round(c2["elapsed_s"] * task.policy_frequency)),
            "safe_event": bool(not c2["bad"] and c2["min_ttc_s"] < task.ttc_threshold),
            "qualified_event": bool(c2["qualified"] and c2["min_ttc_s"] < task.ttc_threshold),
            **paired, "conditions": trials})
    contract = {"protocol": task.research_protocol, "task": asdict(task), "gamma": gamma, "backend": backend_contract,
                "normal_controller": "frozen_policy" if normal_model is not None else
                    ("fixed_IDM" if backend_contract["backend"] == "highway" else "external_frozen"),
                "primary": "all-case qualified positive C1-minus-C2 progress loss / goal distance",
                "censoring": "unsafe early C2 yields zero primary and null raw distance difference"}
    return {"protocol": task.research_protocol, "split": split, "intervention": intervention,
            "drop_probability": drop_probability, "evaluation_contract": contract,
            "evaluation_contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            "overall": aggregate(rows), "families": {f: aggregate([r for r in rows if r["family"] == f]) for f in dict.fromkeys(f for f,_ in case_list)},
            "episodes": rows}, preview
