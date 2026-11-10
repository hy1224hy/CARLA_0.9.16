import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from ..config import ROOT
from ..models.team import Team
from ..environment.factory import make_env
from ..evaluation.runner import evaluate
from .ppo import collect, update, tensor
from .mutual import snapshots, compute_gates, distillation_weights
from .checkpoint import PROTOCOL, provenance, source_digest, read, save, validate_runtime
from .telemetry import Telemetry, committed_logs
from .locking import run_lock
from .curriculum import Curriculum


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def run(config, output, *, resume=None, initialize_from=None, max_rounds=None, preview=False, preview_speed=0.0,
        environment_factory=None, evaluation_factory=None):
    with run_lock(Path(output).resolve()):
        return _run(config, output, resume=resume, initialize_from=initialize_from, max_rounds=max_rounds,
                    preview=preview, preview_speed=preview_speed,
                    environment_factory=environment_factory, evaluation_factory=evaluation_factory)


def _run(config, output, *, resume=None, initialize_from=None, max_rounds=None, preview=False, preview_speed=0.0,
         environment_factory=None, evaluation_factory=None):
    config.validate()
    if config.backend == "carla" and (environment_factory is None or evaluation_factory is None):
        raise ValueError("CARLA requires explicit scene/controller factories; it must not silently run Highway")
    evaluate_policy = evaluation_factory or evaluate
    if max_rounds is not None and (type(max_rounds) is not int or max_rounds < 1):
        raise ValueError("max_rounds must be a positive integer")
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("run outputs must be inside highwaytest/runs for managed source/artifact separation")
    torch.set_num_threads(config.threads)
    models, optimizers, generators, torch_states = {}, {}, {}, {}
    for index, name in enumerate(("A", "B")):
        torch.manual_seed(config.seed + index * 10007)
        models[name] = Team(config)
        optimizers[name] = torch.optim.Adam(models[name].parameters(), lr=config.lr, eps=1e-5)
        generators[name] = np.random.default_rng(config.seed + index * 10007)
        torch_states[name] = torch.get_rng_state()
    teachers = snapshots(models)
    gates = {"A": {}, "B": {}}
    train_steps = round_id = validation_steps = 0
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_selection = None
    best_model = None
    curriculum = Curriculum(config.curriculum)
    if resume:
        state = read(resume)
        if state["config"] != config.to_dict() or state["source_sha256"] != source_digest():
            raise ValueError("resume requires identical config and Python source")
        if Path(resume).resolve().parent != output or Path(resume).name != "latest.pt":
            raise ValueError("resume must use this output directory's latest.pt")
        manifest = json.loads((output / "manifest.json").read_text())
        validate_runtime(manifest["packages"])
        if manifest["config_sha256"] != config.digest():
            raise ValueError("manifest/config mismatch")
        for name in models:
            models[name].load_state_dict(state["models"][name])
            optimizers[name].load_state_dict(state["optimizers"][name])
            teachers[name].load_state_dict(state["teachers"][name])
            generators[name].bit_generator.state = state["numpy_rng"][name]
        torch_states = state["torch_rng"]
        gates = state["gates"]
        train_steps, round_id = state["train_steps"], state["round"]
        validation_steps, best_score = state["validation_steps"], state["best_score"]
        best_selection = state["best_selection"]
        best_model = state["best_model"]
        if config.curriculum is not None:
            if 'curriculum' not in state:
                raise ValueError('curriculum checkpoint is missing stage state')
            curriculum = Curriculum(config.curriculum, state['curriculum'])
        if best_model is not None:
            save(output / "selected.pt", {"protocol": PROTOCOL, "config": config.to_dict(),
                 "model": best_model, "selection": best_selection, "source_sha256": source_digest()})
        # A crash after logging but before checkpoint commit can leave orphan rows.
        # Trim only this run's uncommitted metric records to avoid duplicate budgets.
        committed_logs(output, round_id)
    else:
        initialization = None
        if initialize_from:
            initial = read(initialize_from)
            prior = initial["config"]
            if initial['source_sha256'] != source_digest():
                raise ValueError('warm-start source mismatch; use a new same-source initialization')
            if prior['seed'] != config.seed or prior.get('exploration') != config.exploration:
                raise ValueError('warm-start seed/exploration mismatch')
            if any(prior[k] != config.to_dict()[k] for k in ("hidden", "message_dim")):
                raise ValueError("warm-start architecture mismatch")
            if prior["task"]["attack_weight"] != 0:
                raise ValueError("warm-start must come from the normal-driving stage")
            prior_task = dict(prior["task"])
            next_task = config.to_dict()["task"].copy()
            prior_task.pop("attack_weight")
            next_task.pop("attack_weight")
            if prior_task != next_task:
                raise ValueError("warm-start must preserve the task except attack_weight")
            for name in models:
                models[name].load_state_dict(initial["models"][name])
            import hashlib
            initialization = {"path": str(Path(initialize_from).resolve()),
                              "sha256": hashlib.sha256(Path(initialize_from).read_bytes()).hexdigest(),
                              "prior_train_steps": initial["train_steps"], "optimizer_reset": True,
                              "driving_preparation": initial.get('driving_preparation')}
            teachers = snapshots(models)
        provenance(output, config, initialization)
    envs = {name: (environment_factory or make_env)(config.task, config.gamma) for name in models}
    telemetry = Telemetry(output, config)
    if preview:
        from ..visualization.live import LivePreview
        preview = LivePreview(output, speed=preview_speed)
    else:
        preview = None
    start = time.perf_counter()
    initial_round = round_id
    try:
        while train_steps < config.total_train_steps:
            if max_rounds is not None and round_id - initial_round >= max_rounds:
                break
            n = min(config.rollout_steps, (config.total_train_steps - train_steps) // 2)
            round_id += 1
            round_metrics, round_episodes, validation = [], [], None
            round_end_steps = train_steps + 2 * n
            effective_weight = curriculum.weight if config.curriculum else config.task.attack_weight
            if config.curriculum:
                for env in envs.values():
                    env.task = replace(config.task, attack_weight=effective_weight)
            for name, model in models.items():
                team_start = time.perf_counter()
                torch.set_rng_state(torch_states[name])
                if preview:
                    preview.context(name, round_id, train_steps)
                # Deliberate reset at round boundaries: no live simulator state is
                # required for exact checkpoint continuation. Tail GAE bootstraps.
                batch, families, episodes, _ = collect(model, envs[name], n, generators[name], None, preview=preview)
                if config.backend == "carla":
                    envs[name].close()  # Release A before loading B or any reference trial on one server.
                if preview:
                    preview.phase("PPO update")
                active = round_id > config.warmup_rounds and config.method != "off"
                weights = tensor(distillation_weights(config.method, families, gates[name], seed=config.seed,
                    round_id=round_id, team=name)) if active else torch.zeros(n)
                peer = teachers["B" if name == "A" else "A"] if active else None
                metrics = update(model, optimizers[name], batch, config, peer, weights)
                # The PPO-reported mutual_kl already includes sample weights.
                metrics['effective_beta'] = config.mutual_beta if active and bool(weights.any()) else 0.0
                metrics['mutual_loss'] = config.mutual_beta * metrics['mutual_kl']
                torch_states[name] = torch.get_rng_state()
                train_steps += n
                row = metrics | {"round": round_id, "team": name, "round_end_steps": round_end_steps,
                       "attack_weight": effective_weight, "curriculum_stage": curriculum.state['stage'],
                       "sample_steps": n,
                       "train_steps": train_steps, "team_steps": train_steps // 2 if name == "B" else (train_steps + n) // 2,
                       "mean_step_reward": float(batch["reward"].mean()), "completed_episodes": len(episodes),
                       "sample_update_steps_per_second": n / (time.perf_counter() - team_start),
                       "unfinished_fragment_steps": envs[name].steps if not envs[name].ended else 0,
                       "elapsed_session_s": time.perf_counter() - start}
                round_metrics.append(row)
                append(output / "metrics.jsonl", row)
                for episode in episodes:
                    row = episode | {"round": round_id, "team": name}
                    round_episodes.append(row)
                    append(output / "episodes.jsonl", row)
            if (round_id - 1) % config.refresh_rounds == 0 or train_steps == config.total_train_steps:
                if preview:
                    preview.phase("validation")
                teachers = snapshots(models)
                reports = {}
                for name in models:
                    report, _ = evaluate_policy(teachers[name], config.task, "gate", config.gate_cases_per_family, gamma=config.gamma)
                    reports[name] = report
                    validation_steps += sum(r["episode_steps"] for r in report["episodes"])
                gates = compute_gates(reports, config)
                selection = {}
                for name in models:
                    report, _ = evaluate_policy(models[name], config.task, "selection", config.gate_cases_per_family, gamma=config.gamma)
                    selection[name] = report
                    validation_steps += sum(r["episode_steps"] for r in report["episodes"])
                def rank(name):
                    metrics = selection[name]["overall"]
                    eligible = (config.task.research_protocol == "legacy_v2" or
                                (metrics["bad_rate"] <= config.gate_max_bad_rate
                                 and metrics["success_rate"] >= config.task.selection_min_qualified
                                 and metrics.get("baseline_invalid_count", 0) == 0
                                 and curriculum.full_strength))
                    primary = metrics["quality"] if eligible else -1 - metrics["bad_rate"]
                    return (primary, -metrics["bad_rate"], metrics["blue_progress_m"])
                winner = max(selection, key=rank)
                score = rank(winner)
                validation = {"round": round_id, "train_steps": train_steps,
                       "attack_weight": effective_weight, "curriculum_stage": curriculum.state['stage'],
                       "validation_steps": validation_steps, "gates": gates, "gate": reports,
                       "selection": selection, "winner": winner}
                append(output / "validation.jsonl", validation)
                if score > best_score:
                    import copy
                    best_score = score
                    best_model = copy.deepcopy(models[winner].state_dict())
                    best_selection = {"team": winner, "round": round_id,
                                      "train_steps": train_steps, "quality": selection[winner]["overall"]["quality"],
                                      "eligible": score[0] >= 0,
                                      "attack_weight": effective_weight, "curriculum_stage": curriculum.state['stage'],
                                      "rank": list(score)}
                    save(output / "selected.pt", {"protocol": PROTOCOL, "config": config.to_dict(),
                         "model": best_model, "selection": best_selection,
                         "source_sha256": source_digest()})
                curriculum.observe(reports, train_steps)
            state = {"protocol": PROTOCOL, "config": config.to_dict(), "source_sha256": source_digest(),
                     "models": {k: m.state_dict() for k, m in models.items()},
                     "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                     "teachers": {k: m.state_dict() for k, m in teachers.items()},
                     "numpy_rng": {k: g.bit_generator.state for k, g in generators.items()},
                     "torch_rng": torch_states, "gates": gates, "round": round_id,
                     "train_steps": train_steps, "validation_steps": validation_steps,
                     "best_score": best_score, "best_selection": best_selection, "best_model": best_model}
            if config.curriculum:
                state['curriculum'] = curriculum.state_dict()
            save(output / "latest.pt", state)
            telemetry.round(round_metrics, round_episodes, validation)
            (output / "status.json").write_text(json.dumps({"status": "running", "round": round_id,
                "train_steps": train_steps, "validation_steps": validation_steps}), encoding="utf-8")
            print(f"{config.group} seed={config.seed} round={round_id} A+B={train_steps}/{config.total_train_steps} "
                  f"selection={best_score[0]:.3f}", flush=True)
        summary = {"status": "complete" if train_steps == config.total_train_steps else "paused_at_boundary",
                   "train_steps": train_steps, "blue_agent_steps": 2 * train_steps,
                   "validation_steps": validation_steps, "round": round_id, "selection": best_selection}
        if config.curriculum:
            summary['curriculum'] = curriculum.state_dict() | {'next_attack_weight': curriculum.weight,
                'full_strength_reached': any(t['to'] == config.task.attack_weight for t in curriculum.state['transitions'])}
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        if preview:
            preview.phase("trainer stopped; check run summary/status for completion")
        telemetry.close()
        for env in envs.values():
            env.close()
