"""Versioned research study: paired attack effects, controls and independent seeds."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .config import Config, Task, ROOT, load
from .common.contract import PROTOCOL
from .training.checkpoint import read, source_digest
from .models.team import Team
from .evaluation.paired import evaluate_paired


def path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def write_json(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {target}")
    target.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def plan(output, steps=100000, attack_weight=1.0, seeds=(42, 43, 44), controls=True,
         red_lane_change=True):
    groups = ["G0", "G1", "G2", "G3", "G4", "G5"]
    if controls:
        groups += ["G6_random", "G7_strength"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("provide distinct training seeds")
    entries = []
    for objective, selected in (("normal", ["G0"]), ("attack", groups)):
        for seed in seeds:
            for group in selected:
                task = Task(research_protocol=PROTOCOL, attack_weight=0 if objective == "normal" else attack_weight,
                            red_lane_change=red_lane_change)
                cfg = Config(group=group, seed=seed, total_train_steps=steps, task=task).validate()
                target = output / objective / f"{group}_s{seed}.json"
                entries.append((target, cfg.to_dict()))
    if output.exists():
        raise FileExistsError("use a fresh study-config directory")
    for target, data in entries:
        write_json(target, data)
    write_json(output / "study.json", {"protocol": PROTOCOL, "seeds": list(seeds), "attack_groups": groups,
        "steps_per_run": steps, "runs": len(entries), "total_train_steps": len(entries) * steps,
        "attack_weight": attack_weight, "red_lane_change": red_lane_change,
        "primary": "qualified extra red progress loss versus C1, all-case denominator",
        "claims": "advantages are hypotheses; report null/negative outcomes",
        "controls": "random and strength match this run's evidence coverage, not cross-run realized KL"})
    return output


def load_policy(checkpoint, allow_source_change=False):
    state = read(checkpoint)
    if "model" not in state:
        raise ValueError("evaluate selected.pt, not latest.pt")
    raw = state["config"]
    config = Config(**(raw | {"task": Task(**raw["task"])})).validate()
    if config.task.research_protocol not in (PROTOCOL,'bottleneck_paired_attack_v4'):
        raise ValueError("old v2 policies use different observations/reward and cannot be relabeled v3")
    changed = state["source_sha256"] != source_digest()
    if changed and not allow_source_change:
        raise ValueError("source changed; use original source or explicitly --allow-source-change for diagnostics")
    torch.set_num_threads(config.threads)
    model = Team(config).eval()
    model.load_state_dict(state["model"])
    return config, model, state, changed


def attack_exposure(metrics, nominal):
    histogram, full = {}, 0
    for row in metrics:
        if 'sample_steps' not in row:
            continue
        weight=float(row.get('attack_weight',nominal))
        key=format(weight,'.12g')
        histogram[key]=histogram.get(key,0)+row['sample_steps']
        if weight==float(nominal):
            full+=row['sample_steps']
    return histogram,full if histogram else None


def evaluate_checkpoint(checkpoint, output, *, split="selection", per_family=8,
                        normal_checkpoint=None, intervention="normal", drop_probability=0.3,
                        allow_source_change=False, red_lane_change=None, gif=False):
    if output.exists():
        raise FileExistsError(output)
    config, model, state, changed = load_policy(checkpoint, allow_source_change)
    if config.backend != "highway":
        raise ValueError("evaluate CARLA via evaluate_paired with its explicit frozen backend_factory")
    task = replace(config.task, red_lane_change=config.task.red_lane_change if red_lane_change is None else red_lane_change)
    normal_model, normal_hash = None, None
    if normal_checkpoint:
        normal_config, normal_model, _, normal_changed = load_policy(normal_checkpoint, allow_source_change)
        if normal_config.backend != "highway":
            raise ValueError("normal and attack policies must use the same simulator backend")
        if normal_config.task.attack_weight != 0 or normal_config.communication:
            raise ValueError("normal baseline requires a no-communication normal-driving policy")
        left, right = asdict(normal_config.task), asdict(config.task)
        left.pop("attack_weight")
        right.pop("attack_weight")
        if left != right:
            raise ValueError("normal policy must share physical task and information protocol")
        changed |= normal_changed
        normal_hash = hashlib.sha256(normal_checkpoint.read_bytes()).hexdigest()
    evaluator=evaluate_paired
    if task.research_protocol=='bottleneck_paired_attack_v4':
        from .bottleneck.evaluation import evaluate as evaluator
    report, frames = evaluator(model, task, split, per_family, gamma=config.gamma,
        normal_model=normal_model, intervention=intervention,
        drop_probability=drop_probability if intervention == "drop" else 0, capture=gif)
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps({"communication": config.communication,
        "hidden": config.hidden, "message_dim": config.message_dim, "protocol": config.task.research_protocol}, sort_keys=True).encode())
    for key, value in sorted(state["model"].items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    metrics_path = checkpoint.parent / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()] if metrics_path.exists() else []
    chosen = [r for r in metrics if r["team"] == state["selection"]["team"] and r["round"] <= state["selection"]["round"]]
    participation = {"records": len(chosen),
        "nonzero_mutual_updates": sum(r.get("gate_fraction", 0) > 0 and r.get("mutual_kl", 0) > 1e-10 for r in chosen),
        "mean_gate_fraction_per_update": float(np.mean([r["gate_fraction"] for r in chosen])) if chosen else None,
        "mean_message_effect_per_update": float(np.mean([r["message_mean_effect"] for r in chosen])) if chosen else None}
    exposure, full_strength_steps = attack_exposure(metrics, config.task.attack_weight)
    report.update(group=config.group, seed=config.seed, training_config=config.to_dict(),
        selection=state["selection"], source_sha256=state["source_sha256"], source_changed=changed,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        normal_policy_sha256=normal_hash,
        training_segments=manifest.get("training_segments", [{"budget": config.total_train_steps,
                                                            "source_sha256": manifest["source_sha256"]}]),
        initialization=manifest.get("initialization"))
    report.update(selected_policy_sha256=digest.hexdigest(), selected_training_participation=participation)
    report.update(training_attack_weight_steps=exposure,
                  adaptive_curriculum=config.curriculum,
                  full_strength_train_steps=full_strength_steps)
    write_json(output, report)
    if frames:
        from PIL import Image
        images = [Image.fromarray(x) for x in frames]
        images[0].save(output.with_suffix(".gif"), save_all=True, append_images=images[1:], duration=200, loop=0)
        for item in images:
            item.close()
    return report


def report_study(root, output, split="test", *, protocol=PROTOCOL):
    from scipy.stats import t as student_t
    import csv
    data, identities, contract = [], set(), None
    normal_hashes, initial_hashes = {}, {}
    study_attack_weight = None
    for target in sorted(root.rglob("*.json")):
        if "source" in target.parts:
            continue
        record = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("protocol") != protocol or record.get("split") != split or record.get("intervention") != "normal":
            continue
        if record.get("source_changed"):
            raise ValueError("changed-source diagnostic results cannot enter the main table")
        if record["overall"]["baseline_invalid_count"]:
            raise ValueError(f"normal reference failed qualification: {target}; fix task, do not silently exclude cases")
        cfg = record["training_config"]
        if record["seed"] != cfg["seed"] or record["group"] != cfg["group"]:
            raise ValueError("report identity disagrees with the saved training configuration")
        if cfg["task"]["attack_weight"] > 0:
            if study_attack_weight is not None and study_attack_weight != cfg["task"]["attack_weight"]:
                raise ValueError("attack algorithms must share the same reward weight")
            study_attack_weight = cfg["task"]["attack_weight"]
        label = ("normal/" if cfg["task"]["attack_weight"] == 0 else "attack/") + record["group"]
        identity = label, record["seed"]
        if identity in identities:
            raise ValueError(f"duplicate training seed identity: {identity}")
        identities.add(identity)
        # Only predeclared experimental factors may differ across the main table.
        comparison = json.loads(json.dumps(cfg))
        comparison.pop("group")
        comparison.pop("seed")
        comparison["task"].pop("attack_weight")
        ev = json.loads(json.dumps(record["evaluation_contract"]))
        ev["task"].pop("attack_weight")
        key = {"training": comparison, "evaluation": ev,
               "cases": record["overall"]["case_keys"], "source": record["source_sha256"],
               "segments": record["training_segments"],
               "initialization_mode": (record.get("initialization") or {}).get("prior_train_steps", 0)}
        encoded = json.dumps(key, sort_keys=True)
        if contract is not None and contract != encoded:
            raise ValueError("different budgets, task, source, evaluation, baseline type or continuation schedules")
        contract = encoded
        seed = record["seed"]
        h = record.get("normal_policy_sha256")
        if seed in normal_hashes and normal_hashes[seed] != h:
            raise ValueError("same-seed groups used different C1 policy checkpoints")
        normal_hashes[seed] = h
        initial = (record.get("initialization") or {}).get("sha256")
        if seed in initial_hashes and initial_hashes[seed] != initial:
            raise ValueError("same-seed groups used different pretrained initialization")
        initial_hashes[seed] = initial
        data.append((label, record))
    if not data:
        raise ValueError("no paired reports found")
    labels = sorted({label for label, _ in data})
    fields = ("qualified_attack_utility", "success_rate", "bad_rate", "qualified_extra_red_loss_m")
    rows = []
    for label in labels:
        selected = [r for name, r in data if name == label]
        row = {"label": label, "independent_seeds": len(selected), "seeds": [r["seed"] for r in selected],
               "minimum_three_seeds": len(selected) >= 3,
               "all_selected_models_eligible": all(r["selection"].get("eligible", False) for r in selected)}
        row["selected_models_with_nonzero_mutual_updates"] = sum(r.get("selected_training_participation", {}).get("nonzero_mutual_updates", 0) > 0 for r in selected)
        for field in fields:
            values = [r["overall"][field] for r in selected]
            row[field + "_mean"] = float(np.mean(values))
            row[field + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    mapping = {(label, r["seed"]): r for label, r in data}
    contrasts = []
    pairs = [("attack/G0", "normal/G0"), ("attack/G2", "attack/G0"),
             ("attack/G2", "attack/G1"), ("attack/G3", "attack/G0"),
             ("attack/G4", "attack/G2"), ("attack/G4", "attack/G3"),
             ("attack/G2", "attack/G6_random"), ("attack/G2", "attack/G7_strength")]
    for treatment, baseline in pairs:
        a = {seed for label, seed in mapping if label == treatment}
        b = {seed for label, seed in mapping if label == baseline}
        if not a or not b:
            continue
        if a != b:
            raise ValueError(f"unmatched training seeds for {treatment} vs {baseline}")
        differences = [mapping[treatment, s]["overall"]["qualified_attack_utility"] -
                       mapping[baseline, s]["overall"]["qualified_attack_utility"] for s in sorted(a)]
        mean = float(np.mean(differences))
        half = float(student_t.ppf(0.975, len(a) - 1) * np.std(differences, ddof=1) / np.sqrt(len(a))) if len(a) >= 3 else None
        contrasts.append({"treatment": treatment, "baseline": baseline, "seeds": sorted(a),
            "per_seed_difference": differences, "mean_difference": mean,
            "paired_seed_t95_interval": [mean - half, mean + half] if half is not None else None,
            "interpretation": "exploratory; seed-level interval, not episode-level replication; no automatic superiority claim"})
    factorial = ("attack/G0", "attack/G2", "attack/G3", "attack/G4")
    seed_sets = [{s for label, s in mapping if label == name} for name in factorial]
    if all(seed_sets):
        if any(values != seed_sets[0] for values in seed_sets):
            raise ValueError("2x2 interaction requires identical independent seed sets")
        differences = [(mapping["attack/G4", s]["overall"]["qualified_attack_utility"] -
                        mapping["attack/G2", s]["overall"]["qualified_attack_utility"]) -
                       (mapping["attack/G3", s]["overall"]["qualified_attack_utility"] -
                        mapping["attack/G0", s]["overall"]["qualified_attack_utility"]) for s in sorted(seed_sets[0])]
        mean = float(np.mean(differences))
        n = len(differences)
        half = float(student_t.ppf(0.975, n - 1) * np.std(differences, ddof=1) / np.sqrt(n)) if n >= 3 else None
        contrasts.append({"treatment": "interaction: (G4-G2)-(G3-G0)", "baseline": "zero interaction",
            "seeds": sorted(seed_sets[0]), "per_seed_difference": differences, "mean_difference": mean,
            "paired_seed_t95_interval": [mean - half, mean + half] if half is not None else None,
            "interpretation": "exploratory interaction; confidence intervals are not multiplicity-adjusted"})
    output.mkdir(parents=True, exist_ok=True)
    policy_groups = {}
    for label, record in data:
        if record.get("selected_policy_sha256"):
            key = (record["seed"], record["selected_policy_sha256"])
            policy_groups.setdefault(key, []).append(label)
    duplicates = [{"seed": seed, "policy_sha256": h, "labels": names}
                  for (seed, h), names in policy_groups.items() if len(names) > 1]
    write_json(output / "paired_summary.json", {"split": split, "groups": rows, "contrasts": contrasts,
        "identical_selected_policies": duplicates,
        "contract": json.loads(contract), "independent_unit": "outer training seed; A/B and paired cases are not independent seeds"})
    with (output / "paired_summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details = []
    for label, record in data:
        for family, metrics in {"overall": record["overall"], **record.get("families", {})}.items():
            details.append({"label": label, "seed": record["seed"], "family": family,
                "selected_eligible": record["selection"].get("eligible", False),
                **{key: metrics[key] for key in fields}})
    with (output / "paired_seed_family.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    return rows, contrasts


def main():
    parser = argparse.ArgumentParser(description="Paired attack research protocol; no guaranteed algorithm advantage")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output", default="configs/research_v3")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--attack-weight", type=float, default=1.0)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--fixed-red-lanes", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--config", required=True)
    v.add_argument("--cases-per-family", type=int, default=8)
    v.add_argument("--output", default="runs/research_validation.json")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--normal-checkpoint")
    e.add_argument("--split", choices=("gate", "selection", "test"), default="selection")
    e.add_argument("--cases-per-family", type=int, default=24)
    e.add_argument("--intervention", choices=("normal", "zero", "stale", "drop"), default="normal")
    e.add_argument("--drop-probability", type=float, default=0.3)
    e.add_argument("--red-lanes", choices=("trained", "fixed", "change"), default="trained")
    e.add_argument("--allow-source-change", action="store_true")
    e.add_argument("--gif", action="store_true")
    e.add_argument("--output")
    r = sub.add_parser("report")
    r.add_argument("--runs", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--split", choices=("selection", "test"), default="test")
    args = parser.parse_args()
    if args.command == "plan":
        print(plan(path(args.output), args.steps, args.attack_weight, args.seeds, not args.no_controls, not args.fixed_red_lanes))
    elif args.command == "validate":
        from .environment.conditions import HighwayConditionBackend
        cfg = load(path(args.config))
        if cfg.task.research_protocol=='bottleneck_paired_attack_v4':
            raise ValueError('use python -m highwaytest.bottleneck calibrate for the three-car protocol')
        if cfg.backend != "highway":
            raise ValueError("CARLA calibration requires its explicit backend factory")
        torch.set_num_threads(cfg.threads)
        factory = lambda condition, family, seed: HighwayConditionBackend(cfg.task, "C1" if condition == "C2" else condition, family, seed)
        report, _ = evaluate_paired(None, cfg.task, "gate", args.cases_per_family, backend_factory=factory)
        write_json(path(args.output), report)
        print(json.dumps(report["overall"], indent=2))
        if report["overall"]["baseline_invalid_count"] or report["overall"]["success_rate"] < 1:
            raise RuntimeError("normal traffic failed calibration; do not start main experiments")
    elif args.command == "evaluate":
        checkpoint = path(args.checkpoint)
        lane = None if args.red_lanes == "trained" else args.red_lanes == "change"
        output = path(args.output) if args.output else checkpoint.parent / "paired_evaluations" / f"{args.split}_{args.intervention}_{args.red_lanes}.json"
        report = evaluate_checkpoint(checkpoint, output, split=args.split, per_family=args.cases_per_family,
            normal_checkpoint=path(args.normal_checkpoint) if args.normal_checkpoint else None,
            intervention=args.intervention, drop_probability=args.drop_probability,
            allow_source_change=args.allow_source_change, red_lane_change=lane, gif=args.gif)
        print(json.dumps(report["overall"], indent=2))
    else:
        rows, contrasts = report_study(path(args.runs), path(args.output), args.split)
        print(json.dumps({"groups": rows, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
