"""Aggregate one normal test report per independent (group, seed); never A/B as seeds."""
import csv
import json
from collections import defaultdict
import numpy as np


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    groups, identities, task_id = defaultdict(list), set(), None
    initialization_by_seed = {}
    for path in sorted(root.rglob("*.json")):
        if "source" in path.parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue  # batch_status.json and comparison.json are lists, not episode reports
        if data.get("split") != "test" or data.get("intervention") != "normal":
            continue
        identity = (data["group"], data["seed"])
        if identity in identities:
            raise ValueError(f"duplicate group/seed test evidence: {identity}")
        identities.add(identity)
        if data.get("source_changed"):
            raise ValueError(f"evaluation used changed source: {path}")
        if "experiment_contract" not in data:
            raise ValueError(f"missing full experiment contract; export a v2 evaluation: {path}")
        contract = json.dumps({"experiment": data["experiment_contract"],
                               "cases": data["overall"]["case_keys"],
                               "source": data["source_sha256"]}, sort_keys=True)
        if task_id is not None and contract != task_id:
            raise ValueError("cannot aggregate different budgets, hyperparameters, initialization modes, source or test cases")
        task_id = contract
        seed = data["seed"]
        warmstart = data.get("initialization_sha256")
        if seed in initialization_by_seed and initialization_by_seed[seed] != warmstart:
            raise ValueError("groups with the same seed used different pretraining checkpoints")
        initialization_by_seed[seed] = warmstart
        groups[data["group"]].append(data)
    fields = ("success_rate", "bad_rate", "safe_event_rate", "qualified_event_rate", "red_delay_m", "blue_progress_m", "quality")
    rows = []
    for group, records in sorted(groups.items()):
        row = {"group": group, "independent_seeds": len(records)}
        for key in fields:
            values = [r["overall"][key] for r in records]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for path in sorted(root.rglob("metrics.jsonl")):
        if "source" in path.parts:
            continue
        data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for name in ("A", "B"):
            subset = [r for r in data if r["team"] == name]
            ax.plot([r.get("round_end_steps", r["train_steps"]) for r in subset], [r["mean_step_reward"] for r in subset],
                    label=f"{path.parent.name}/{name}", alpha=0.7)
    ax.set(xlabel="A+B training policy decisions", ylabel="Mean training step reward")
    if ax.lines:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)
    print(f"Aggregated {len(identities)} independent runs into {output}")
