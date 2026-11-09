"""Exercise every group, checkpoint selection, interventions and scripted feasibility."""
from datetime import datetime
import json
import numpy as np
import torch
from ..config import Config, Task, ROOT, GROUPS
from ..environment.highway import HighwayTeamEnv
from ..environment.scenarios import cases
from ..evaluation.runner import evaluate
from ..models.team import Team
from ..training.runner import run
from ..training.checkpoint import read
from ..experiments import comparison_contract


def feasibility():
    """Use an IDM copy for blue controls as a task probe, never as learned results."""
    rows = []
    env = HighwayTeamEnv(Task())
    try:
        for family, seed in cases("gate", 4):
            _, info = env.reset(seed=seed, options={"family": family})
            env.road.vehicles[2:] = [env._red(env.road, v.position[0], v.position[1], v.speed)
                                     for v in env.road.vehicles[2:]]
            while True:
                # IDMVehicle.act ignores external commands and recomputes IDM.
                # This probe checks task feasibility, not policy/action execution.
                _, _, term, trunc, info = env.step(np.zeros((2, 2)))
                if term or trunc:
                    break
            rows.append({"family": family, "seed": seed, "success": info["success"],
                         "bad": info["bad"], "blue_progress_m": info["blue_progress_m"]})
    finally:
        env.close()
    return {"controller": "scripted IDM probe, NOT learned MAPPO", "episodes": rows,
            "success_rate": float(np.mean([r["success"] for r in rows]))}


def main():
    root = ROOT / "runs" / f"smoke_{datetime.now():%Y%m%d_%H%M%S}"
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for group in list(GROUPS)[:6]:
        config = Config(group=group, total_train_steps=64, rollout_steps=16, hidden=32,
                        epochs=1, minibatch=16, gate_cases_per_family=1,
                        warmup_rounds=0, refresh_rounds=1,
                        task=Task(duration=1.0, attack_weight=0.3))
        results[group] = run(config, root / group)
        state = read(root / group / "selected.pt")
        model = Team(config).eval()
        model.load_state_dict(state["model"])
        for mode in ("normal", "zero", "stale", "drop"):
            report, frames = evaluate(model, config.task, "test", 1, mode, 0.3,
                                      capture=(group == "G4" and mode == "normal"))
            report.update(group=group, seed=config.seed, task=config.to_dict()["task"],
                          source_sha256=state["source_sha256"], source_changed=False,
                          experiment_contract=comparison_contract(config), initialization_sha256=None)
            target = root / group / "evaluations" / f"test_{mode}.json"
            target.parent.mkdir(exist_ok=True)
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if frames:
                from PIL import Image
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(root / "preview.gif", save_all=True, append_images=images[1:], duration=200, loop=0)
    receipt = {"purpose": "engineering smoke only; 64 decisions/group, not performance validation",
               "groups": results, "task_feasibility": feasibility()}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(root)
    print(json.dumps(receipt["task_feasibility"], indent=2))


if __name__ == "__main__":
    main()
