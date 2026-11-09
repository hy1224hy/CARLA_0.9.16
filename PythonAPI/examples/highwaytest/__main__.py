"""Run from PythonAPI/examples: python -m highwaytest --help."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import torch
from .config import ROOT, GROUPS, Config, Task, load
from .training.checkpoint import read, source_digest


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description="Highway-env MUT-MAPPO experiment manager")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train both A/B teams with a shared total budget")
    train.add_argument("--config")
    train.add_argument("--group", choices=GROUPS)
    train.add_argument("--seed", type=int)
    train.add_argument("--steps", type=int)
    train.add_argument("--attack-weight", type=float)
    train.add_argument("--output")
    train.add_argument("--resume", help="continue latest.pt with identical config/source")
    train.add_argument("--initialize-from", help="warm-start from a trusted normal-driving latest.pt")
    train.add_argument("--max-rounds", type=int, help="stop cleanly at a boundary, useful for staged runs")
    train.add_argument("--preview", action="store_true", help="publish actual training frames to this run's preview directory")
    train.add_argument("--preview-speed", type=float, default=0, help="0=unpaced; 1=simulation real time (requires --preview)")
    evaluate = commands.add_parser("evaluate", help="evaluate a validation-selected single team")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("gate", "selection", "test"), default="test")
    evaluate.add_argument("--cases-per-family", type=int, default=24)
    evaluate.add_argument("--intervention", choices=("normal", "zero", "stale", "drop"), default="normal")
    evaluate.add_argument("--drop-probability", type=float, default=0.3)
    evaluate.add_argument("--gif", action="store_true")
    evaluate.add_argument("--output")
    evaluate.add_argument("--allow-source-change", action="store_true", help="explicitly evaluate with changed Python source; recorded in report")
    plan = commands.add_parser("plan", help="write the five-group x three-seed matrix; does not launch training")
    plan.add_argument("--output", default="configs/matrix")
    plan.add_argument("--steps", type=int, default=100000)
    plan.add_argument("--attack-weight", type=float, default=0.3)
    plan.add_argument("--include-g5", action="store_true")
    report = commands.add_parser("report", help="aggregate independent-seed test reports and plot training curves")
    report.add_argument("--runs", default="runs")
    report.add_argument("--output", default="runs/comparison")
    check = commands.add_parser("check", help="headless dependency, configuration and matrix fairness preflight")
    check.add_argument("--configs", default="configs/server_pilot")
    batch = commands.add_parser("batch", help="run/resume a checked matrix, with per-run logs and bounded CPU workers")
    batch.add_argument("--configs", default="configs/server_pilot")
    batch.add_argument("--output", default="runs/server_pilot")
    batch.add_argument("--jobs", type=int, default=1)
    viewer = commands.add_parser("serve-preview", help="serve an actual training preview on localhost for SSH forwarding")
    viewer.add_argument("--run", required=True)
    viewer.add_argument("--port", type=int, default=8000)
    extension = commands.add_parser("extend", help="continue full state into a new run with a larger TOTAL step budget")
    extension.add_argument("--checkpoint", required=True, help="parent latest.pt")
    extension.add_argument("--steps", type=int, required=True, help="new total A+B budget, not additional steps")
    extension.add_argument("--output", required=True, help="new run directory")
    extension.add_argument("--allow-source-change", action="store_true", help="explicitly permit audited v2 source changes; recorded in lineage")
    extension.add_argument("--max-rounds", type=int)
    extension.add_argument("--preview", action="store_true")
    extension.add_argument("--preview-speed", type=float, default=0)
    args = parser.parse_args()
    if args.command == "train":
        from .training.runner import run
        if args.resume and args.initialize_from:
            parser.error("resume and initialize-from are mutually exclusive")
        if args.preview_speed and not args.preview:
            parser.error("preview-speed requires --preview")
        if args.resume:
            checkpoint = project_path(args.resume)
            data = read(checkpoint)["config"]
            config = Config(**(data | {"task": Task(**data["task"])}))
        else:
            config = load(project_path(args.config) if args.config else None)
        for field, value in (("group", args.group), ("seed", args.seed), ("total_train_steps", args.steps)):
            if value is not None:
                setattr(config, field, value)
        if args.attack_weight is not None:
            config.task.attack_weight = args.attack_weight
        output = project_path(args.output) if args.output else (project_path(args.resume).parent if args.resume else
                 ROOT / "runs" / f"{config.group}_s{config.seed}_{datetime.now():%Y%m%d_%H%M%S}")
        print(json.dumps(run(config, output, resume=project_path(args.resume) if args.resume else None,
              initialize_from=project_path(args.initialize_from) if args.initialize_from else None,
              max_rounds=args.max_rounds, preview=args.preview, preview_speed=args.preview_speed), indent=2))
    elif args.command == "evaluate":
        from .models.team import Team
        from .evaluation.runner import evaluate
        path = project_path(args.checkpoint)
        state = read(path)
        if state["source_sha256"] != source_digest() and not args.allow_source_change:
            parser.error("checkpoint source differs; use the saved source or explicitly --allow-source-change")
        if "model" not in state:
            parser.error("use selected.pt, chosen on selection cases; latest.pt is for resume")
        data = state["config"]
        config = Config(**(data | {"task": Task(**data["task"])})).validate()
        if config.backend != "highway":
            parser.error("CARLA evaluation requires an explicit frozen backend factory; do not run a Highway substitute")
        torch.set_num_threads(config.threads)
        model = Team(config).eval()
        model.load_state_dict(state["model"])
        result, frames = evaluate(model, config.task, args.split, args.cases_per_family,
                                  args.intervention, args.drop_probability if args.intervention == "drop" else 0,
                                  capture=args.gif, gamma=config.gamma)
        import hashlib
        from .experiments import comparison_contract
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        result.update(group=config.group, seed=config.seed, selection=state["selection"],
                      config_sha256=config.digest(), source_sha256=state["source_sha256"],
                      task=config.to_dict()["task"], source_changed=state["source_sha256"] != source_digest(),
                      evaluation_source_sha256=source_digest(),
                      experiment_contract=comparison_contract(config, manifest["initialization"]),
                      initialization_sha256=(manifest["initialization"] or {}).get("sha256"),
                      checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        segments = manifest.get("training_segments", [{"budget": config.total_train_steps,
                                                       "source_sha256": manifest["source_sha256"]}])
        result["experiment_contract"]["training_segments"] = segments
        output = project_path(args.output) if args.output else path.parent / "evaluations" / (
            f"{args.split}_{args.intervention}_{args.drop_probability if args.intervention == 'drop' else 0:g}.json")
        if output.exists():
            raise FileExistsError(f"evaluation already exists: {output}; use a different --output")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if frames:
            from PIL import Image
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(output.with_suffix(".gif"), save_all=True, append_images=images[1:],
                           duration=1000 // config.task.policy_frequency, loop=0)
        print(json.dumps(result["overall"], indent=2))
        print(output)
    elif args.command == "plan":
        output = project_path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        groups = list(GROUPS)[:6] if args.include_g5 else list(GROUPS)[:5]
        for group in groups:
            for seed in (42, 43, 44):
                config = Config(group=group, seed=seed, total_train_steps=args.steps,
                                task=Task(attack_weight=args.attack_weight)).validate()
                path = output / f"{group}_s{seed}.json"
                if path.exists():
                    raise FileExistsError(path)
                path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote {len(groups) * 3} configs to {output}")
    elif args.command == "report":
        from .evaluation.report import report
        report(project_path(args.runs), project_path(args.output))
    elif args.command == "check":
        from .experiments import preflight
        print(json.dumps(preflight(project_path(args.configs)), indent=2))
    elif args.command == "batch":
        from .experiments import batch
        batch(project_path(args.configs), project_path(args.output), args.jobs)
    elif args.command == "serve-preview":
        from .visualization.live import serve
        serve(project_path(args.run), args.port)
    elif args.command == "extend":
        from .training.continuation import extend
        if args.preview_speed and not args.preview:
            parser.error("preview-speed requires --preview")
        print(json.dumps(extend(project_path(args.checkpoint), project_path(args.output), args.steps,
                         allow_source_change=args.allow_source_change, max_rounds=args.max_rounds,
                         preview=args.preview, preview_speed=args.preview_speed), indent=2))


if __name__ == "__main__":
    main()
