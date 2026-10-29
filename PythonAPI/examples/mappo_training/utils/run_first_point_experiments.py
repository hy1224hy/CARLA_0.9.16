"""Prepare/run the small first-point comparison: references, train, select, test.

Preparation is offline. All live stages share one CARLA server, serially.
Communication is intentionally disabled; Actor/learner interfaces stay separate.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from training.checkpoint_protocol import atomic_write_json, load_checkpoint_file, sha256_file
from training.first_point_scenarios import build_cases
from training.mut_mappo_config import ROOT, load_config, portable_config, resolve_path, validate_config
from training.mut_mappo_experiments import SUITES, experiment_config


def prepare(output, red_checkpoint, *, suite="core", seeds=(42, 43, 44), ticks=120000, device="cpu", scene_stage="full"):
    output, red_checkpoint = resolve_path(output), resolve_path(red_checkpoint)
    if output.exists():
        raise FileExistsError("experiment exists; reuse it with the live commands or choose a new directory")
    source, digest = load_checkpoint_file(red_checkpoint, map_location="cpu")
    base = portable_config(load_config("configs/mut_mappo/local_candidate_v1.json"))
    manifest = build_cases(parallel_only=scene_stage == "parallel")
    output.mkdir(parents=True)
    atomic_write_json(manifest, output / "cases.json")
    base.update(case_manifest=(output / "cases.json").relative_to(ROOT).as_posix(),
                device=device, research_status="candidate")
    base["red_checkpoint"] = {"path": str(red_checkpoint), "sha256": digest,
                              "capability_status": "requires_independent_driving_qualification"}
    base["learning"].update(hidden_dim=source["model_spec"]["hidden_dim"],
        gamma=float(source["training_config"]["gamma"]))
    for key in ("min_log_std", "max_log_std", "init_log_std"):
        if key in source["training_config"]:
            base["learning"][key] = source["training_config"][key]
    base["task"].update(reward_profile="c1_progress_team_v1", max_steps=500,
        team_reward={"reference_dir": (output / "references").relative_to(ROOT).as_posix(),
                     "gamma": base["learning"]["gamma"], "attack_weight": 10.0, "driving_weight": 1.0})
    base["mutual"].update(quality_metric="efficiency", min_efficiency_gap=0.02,
                           gate_min_cases=8, gate_max_bad_rate=0.125)
    base["evaluation"].update(selection_metric="safe_efficiency_utility")
    base["run"].update(train_tick_budget=int(ticks), rounds=1000, lr_horizon_rounds=1000,
                        episodes_per_team_update=4)
    if ticks <= 0 or len(set(seeds)) != len(seeds):
        raise ValueError("positive training ticks and unique seeds are required")
    atomic_write_json(base, output / "base_config.json")
    entries = []
    for seed in seeds:
        for group in SUITES[suite]:
            cfg = experiment_config(base, group)
            cfg["seed"] = int(seed)
            validate_config(cfg)
            run_id = f"{group}_s{seed}"
            path = output / "configs" / (run_id + ".json")
            atomic_write_json(cfg, path)
            entries.append({"run_id": run_id, "group": group, "seed": int(seed),
                            "config": str(path), "run_dir": str(output / "runs" / run_id)})
    experiment = {"schema": "first_point_experiment_v1", "suite": suite, "scene_stage": scene_stage,
        "base_config": str(output / "base_config.json"), "entries": entries,
        "train_tick_budget": ticks, "split_counts": {s: len(k) for s, k in manifest["splits"].items()},
        "execution": "one user-started CARLA, serial; A/B cost combined; complete-boundary soft cap",
        "reward": "shared C1-relative progress utility; team tail credit; no communication",
        "status": "PREPARED_NOT_RUN"}
    atomic_write_json(experiment, output / "experiment.json")
    return experiment


def load_experiment(directory, groups=None, seeds=None):
    root = resolve_path(directory)
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    entries = [row for row in experiment["entries"]
               if (not groups or row["group"] in groups) and (not seeds or row["seed"] in seeds)]
    if not entries:
        raise ValueError("no prepared runs match the requested groups/seeds")
    return root, experiment, entries


def reusable_evaluation(path, checkpoint_digest=None):
    """Reuse completed evidence; preserve failed attempts in their directories."""
    path = Path(path)
    attempt = 0
    while path.exists():
        summary = path / "summary.json"
        if summary.exists():
            report = json.loads(summary.read_text(encoding="utf-8"))
            if (report.get("metadata", {}).get("status") == "COMPLETED"
                    and report["metadata"].get("checkpoint_sha256") == checkpoint_digest
                    and (not report["metadata"].get("selection_requested") or "selection" in report)):
                return path, report
        attempt += 1
        path = path.parent / (path.name.split("_retry_")[0] + f"_retry_{attempt:03d}")
    return path, None


def red_screen(root, experiment):
    from evaluation.evaluate_mut_mappo import evaluate
    output, report = reusable_evaluation(root / "red_screen")
    if report is None:
        report = evaluate(experiment["base_config"], team="A", split="gate", output_dir=output, conditions=("C0", "C1"))
    conditions = report["attempts"]["conditions"]
    ready = all(conditions[c]["n_invalid"] == 0 and (conditions[c]["red_success_rate"] or 0) >= .95
                for c in ("C0", "C1")) and (conditions["C1"]["blue_eligible_rate"] or 0) >= .95
    result = {"ready_for_registered_attack_cases": ready, "report": str(output / "summary.json"),
              "scope": "C0/C1 rolling interaction screen; full stationary/base qualification is a separate existing evaluator"}
    atomic_write_json(result, root / "red_readiness.json")
    return result


def base_evaluate(root, experiment, *, geometry_report=None, validate_only=False):
    from evaluation.evaluate_base_driving_vnext import evaluate
    from evaluation.driving_policy_metrics import summarize_base_evaluation
    config = load_config(experiment["base_config"])
    source = resolve_path(config["red_checkpoint"]["path"])
    geometry = resolve_path(geometry_report) if geometry_report else ROOT / "diagnostics/base_driving_vnext_20261020/preflight_validation_live.json"
    if not geometry.exists() and not validate_only:
        raise FileNotFoundError("provide --geometry-report from the existing base scenario preflight")
    output = root / "base_screen"
    report = evaluate([source], output,
        ROOT / "data/training_scenarios_town05_base_vnext_v1_validation/protocol/manifest.json",
        geometry_report=geometry if geometry.exists() else None, validate_only=validate_only)
    if not validate_only:
        summarize_base_evaluation(output)
    return report


def train_entries(entries, *, resume=False, stop_round=None):
    from training.train_mut_mappo import train
    from utils.mut_mappo_session import CarlaOwner
    for entry in entries:
        cfg = validate_config(load_config(entry["config"]))
        run = Path(entry["run_dir"])
        status_path = run / "session/status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("training_budget_reached") and (run / "checkpoint_manifest.json").exists():
                print("TRAIN_ALREADY_COMPLETE", entry["run_id"], flush=True)
                continue
        with CarlaOwner(cfg["host"], cfg["port"], run_dir=run, purpose="first-point " + entry["run_id"]):
            train(cfg, run, resume=resume and (run / "checkpoint_manifest.json").exists(), stop_round=stop_round)


def evaluate_entries(root, experiment, entries, *, with_credit=False, allow_weak_red=False):
    from evaluation.evaluate_mut_mappo import evaluate
    from evaluation.evaluate_mut_mappo_credit import evaluate as evaluate_credit
    from evaluation.mut_mappo_reference import prepare as prepare_references
    for entry in entries:
        run = Path(entry["run_dir"])
        manifest = json.loads((run / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        checkpoint = run / manifest["checkpoint"]
        if manifest["completed_round"] < 1:
            raise RuntimeError("training produced no completed learned boundary")
        key = entry["run_id"] + f"_r{manifest['completed_round']:08d}"
        result_path = root / "results" / (key + ".json")
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else dict(entry)
        output, report = reusable_evaluation(root / "evaluations" / key / "selection", manifest["sha256"])
        if report is None:
            report = evaluate(entry["config"], checkpoint, split="selection", output_dir=output, select_best=True)
        result.update(source_checkpoint=str(checkpoint), source_checkpoint_sha256=manifest["sha256"],
                      completed_round=manifest["completed_round"], selection_dir=str(output),
                      selection_status=report["selection"]["status"])
        selected = report["selection"]["selected"]
        atomic_write_json(result, result_path)
        if selected is None:
            print("NO_QUALIFIED_BLUE", entry["run_id"], flush=True)
            continue
        # Test curves are prepared only after this candidate has been frozen by selection.
        prepare_references(experiment["base_config"], ("test",), allow_weak_red=allow_weak_red)
        frozen = output / "frozen_blue_team.pt"
        test_output, test_report = reusable_evaluation(root / "evaluations" / key / "test", sha256_file(frozen))
        if test_report is None:
            test_report = evaluate(entry["config"], frozen, split="test", output_dir=test_output)
        result.update(test_dir=str(test_output), selected_team=selected["team"])
        if with_credit:
            credit_dir = Path(result.get("credit_dir", root / "evaluations" / key / "credit"))
            if not (credit_dir / "summary.json").exists():
                if credit_dir.exists():
                    credit_dir = credit_dir.parent / ("credit_retry_" + str(len(list(credit_dir.parent.glob("credit*")))))
                evaluate_credit(entry["config"], frozen, credit_dir, split="test", team=selected["team"])
            result["credit_dir"] = str(credit_dir)
        atomic_write_json(result, result_path)


def summarize(root, experiment):
    rows = []
    latest = {}
    for path in sorted((root / "results").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["run_id"] not in latest or result["completed_round"] > latest[result["run_id"]]["completed_round"]:
            latest[result["run_id"]] = result
    for result in latest.values():
        row = {k: result.get(k) for k in ("run_id", "group", "seed", "completed_round", "selection_status", "selected_team")}
        if result.get("test_dir"):
            report = json.loads((Path(result["test_dir"]) / "summary.json").read_text(encoding="utf-8"))
            team = result["selected_team"]
            paired = report["paired"]["teams"][team]
            c2 = paired["conditions"]["C2"]
            row.update(safe_efficiency_utility=c2["safe_efficiency_utility"], red_success_rate=c2["red_success_rate"],
                       blue_bad_rate=c2["b"], AttackGain_safe=paired["AttackGain_safe"],
                       arrival_delay_s=paired["arrival_delay_C2_minus_C1_both_success_seconds_mean"],
                       test_common_cases=paired["n_common_valid"])
        if result.get("credit_dir"):
            credit = json.loads((Path(result["credit_dir"]) / "summary.json").read_text(encoding="utf-8"))
            row.update(credit["means"])
        from utils.mut_mappo_session import last_json_record
        audit = last_json_record(Path(result["run_dir"]) / "actual_cost_audit.jsonl")
        row["actual_train_ticks"] = audit.get("actual_counters", {}).get("train") if audit else None
        row["training_budget"] = experiment["train_tick_budget"]
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    base_config = load_config(experiment["base_config"])
    reference_path = resolve_path(base_config["task"]["team_reward"]["reference_dir"]) / "manifest.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {"cases": {}}
    weak_cases = sum(not r["qualified"] for r in reference["cases"].values())
    base_path = root / "base_screen/qualification.json"
    base_report = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    base_qualified = any(r.get("qualified") and r.get("checkpoint_sha256") == base_config["red_checkpoint"]["sha256"]
                         for r in base_report.get("reports", []))
    for row in rows:
        row["diagnostic_weak_red"] = not base_qualified or weak_cases > 0
    columns = sorted({key for row in rows for key in row}) or ["run_id"]
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # One row per run/outer seed, never one independent sample per A/B team.
    groups = {}
    for group in sorted({row["group"] for row in rows}):
        members = [row for row in rows if row["group"] == group]
        stats = {}
        for metric in ("safe_efficiency_utility", "blue_bad_rate", "red_success_rate", "AttackGain_safe", "interaction"):
            values = [row[metric] for row in members if row.get(metric) is not None]
            stats[metric] = {"n": len(values), "mean": statistics.fmean(values) if values else None,
                             "std": statistics.stdev(values) if len(values) > 1 else None}
        groups[group] = {"runs": len(members), "qualified_exports": sum(row.get("selected_team") is not None for row in members), "metrics": stats}
    summary = {"expected_runs": len(experiment["entries"]), "reported_boundaries": len(rows),
               "groups": groups, "rows": rows, "independent_unit": "outer training seed; no A/B pooling"}
    summary["diagnostic_weak_reference_cases"] = weak_cases
    summary["base_red_qualified"] = base_qualified
    summary["diagnostic_weak_red"] = not summary["base_red_qualified"] or summary["diagnostic_weak_reference_cases"] > 0
    atomic_write_json(summary, root / "comparison.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", default="experiments/first_point_v1")
    p.add_argument("--red-checkpoint", required=True)
    p.add_argument("--suite", choices=tuple(SUITES), default="core")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--train-ticks", type=int, default=120000)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--scene-stage", choices=("parallel", "full"), default="full")
    for name in ("base-eval", "red-eval", "references", "train", "evaluate", "run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--experiment", default="experiments/first_point_v1")
        p.add_argument("--groups", nargs="+")
        p.add_argument("--seeds", type=int, nargs="+")
        p.add_argument("--allow-weak-red", action="store_true")
        p.add_argument("--resume", action="store_true")
        p.add_argument("--with-credit", action="store_true")
        p.add_argument("--geometry-report")
        p.add_argument("--validate-only", action="store_true")
        p.add_argument("--stop-round", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.red_checkpoint, suite=args.suite, seeds=args.seeds,
                         ticks=args.train_ticks, device=args.device, scene_stage=args.scene_stage)
    else:
        root, experiment, entries = load_experiment(args.experiment, args.groups, args.seeds)
        if args.command == "base-eval":
            result = base_evaluate(root, experiment, geometry_report=args.geometry_report, validate_only=args.validate_only)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command == "run" and not args.allow_weak_red:
            base_path = root / "base_screen/qualification.json"
            if not base_path.exists():
                base_evaluate(root, experiment, geometry_report=args.geometry_report)
            base = json.loads(base_path.read_text(encoding="utf-8"))
            red_digest = load_config(experiment["base_config"])["red_checkpoint"]["sha256"]
            if not any(r.get("qualified") and r.get("checkpoint_sha256") == red_digest for r in base.get("reports", [])):
                raise RuntimeError("base driving qualification failed; --allow-weak-red is diagnostic only")
        if args.command in {"red-eval", "run"}:
            result = red_screen(root, experiment)
            if args.command == "run" and not result["ready_for_registered_attack_cases"] and not args.allow_weak_red:
                raise RuntimeError("C0/C1 red screen failed; inspect report, or explicitly choose diagnostic --allow-weak-red")
        if args.command in {"references", "run"}:
            from evaluation.mut_mappo_reference import prepare as prepare_references
            result = prepare_references(experiment["base_config"], allow_weak_red=args.allow_weak_red)
        if args.command in {"train", "run"}:
            train_entries(entries, resume=args.resume, stop_round=args.stop_round)
        if args.command in {"evaluate", "run"}:
            evaluate_entries(root, experiment, entries, with_credit=args.with_credit, allow_weak_red=args.allow_weak_red)
        result = summarize(root, experiment)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
