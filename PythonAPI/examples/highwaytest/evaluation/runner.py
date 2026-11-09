import numpy as np
import torch
from ..environment.factory import make_env
from ..environment.scenarios import cases, FAMILIES
from ..training.ppo import tensor


def summarize(rows):
    keys = ("success", "bad", "collision", "offroad", "red_delay_m", "blue_progress_m",
            "safe_event", "qualified_event", "quality", "min_ttc_s", "hard_brakes", "episode_steps", "return")
    result = {k + ("_rate" if k in ("success", "bad", "collision", "offroad", "safe_event", "qualified_event") else ""):
              float(np.mean([r[k] for r in rows])) for k in keys}
    result["episodes"] = len(rows)
    result["case_keys"] = [r["case_key"] for r in rows]
    return result


@torch.no_grad()
def evaluate(model, task, split, per_family, intervention="normal", drop_probability=0.0,
             capture=False, gamma=0.99):
    if task.research_protocol=='bottleneck_paired_attack_v4':
        from ..bottleneck.evaluation import evaluate as evaluate_bottleneck
        return evaluate_bottleneck(model,task,split,per_family,intervention=intervention,
            drop_probability=drop_probability,capture=capture,gamma=gamma)
    if task.research_protocol != "legacy_v2":
        from .paired import evaluate_paired
        return evaluate_paired(model, task, split, per_family, gamma=gamma,
                               intervention=intervention, drop_probability=drop_probability, capture=capture)
    if intervention not in ("normal", "zero", "stale", "drop"):
        raise ValueError("unknown message intervention")
    if not 0 <= drop_probability <= 1:
        raise ValueError("drop probability must lie in [0,1]")
    env = make_env(task, gamma)
    rows, frames = [], []
    try:
        for family, seed in cases(split, per_family):
            obs, info = env.reset(seed=seed, options={"family": family})
            stale = obs.copy()  # Same-role unrelated previous-step message source, first step unchanged.
            rng = np.random.default_rng(seed + 11_000_000)
            total = 0.0
            while True:
                links = info["links"].copy()
                if intervention == "drop":
                    links *= rng.random(links.shape) >= drop_probability
                dist = model.distribution(tensor(obs), tensor(links), intervention=intervention,
                                          stale_obs=tensor(stale))
                action = dist.loc.tanh().numpy()
                stale = obs.copy()
                obs, reward, term, trunc, info = env.step(action)
                total += float(reward.mean())
                if capture and len(rows) == 0:
                    frames.append(env.render())
                if term or trunc:
                    break
            rows.append({k: v for k, v in info.items() if k not in ("state", "links")} |
                        {"case_key": f"{family}:{seed}", "return": total})
    finally:
        env.close()
    return {"split": split, "intervention": intervention, "drop_probability": drop_probability,
            "overall": summarize(rows),
            "families": {f: summarize([r for r in rows if r["family"] == f]) for f in FAMILIES},
            "episodes": rows}, frames
