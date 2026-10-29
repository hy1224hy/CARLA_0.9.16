"""Prepare/check this review package on CPU; a CARLA Client is forbidden.

Run from mappo_training as:
python -m experiments.adversarial_u1000_pilot_20261028_v1.offline_audit --prepare
Use --tests to run the existing, simulator-independent regression suite too.
Preparation refuses to replace an existing configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import carla
import torch

from training.checkpoint_protocol import atomic_write_json, load_checkpoint_file, sha256_file
from training.first_point_scenarios import build_cases
from training.mut_mappo_config import ROOT, load_config, make_actors, portable_config, validate_config
from training.mut_mappo_experiments import experiment_config
from training.mut_mappo_learning import TeamLearner

OUT = Path(__file__).resolve().parent
CONTROL = ROOT / "diagnostics/base_driving_vnext_20261020/long_s42_u600_v1"
BASE_RUN = ROOT / "logs/BD_vNext_V6_scratch_s42_u600_20261020_v1"
SHA = {1000: "0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a",
       1200: "e6b1f93b770e2bb6cf7b40d3cba33b50bbc16e46ea4da2b4023f283498736163"}
CONFIG = OUT / "configs/B0_s42.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def protected_inputs():
    paths = [CONTROL / "plan.json", CONTROL / "status.json", BASE_RUN / "initialization.json",
             BASE_RUN / "metrics.jsonl", BASE_RUN / "episodes.jsonl",
             BASE_RUN / "session/launch.json", BASE_RUN / "session/exit.json"]
    paths += list(CONTROL.glob("extension*.json"))
    paths += list(CONTROL.glob("u*_summary.json"))
    paths += [CONTROL / f"u{u:06d}.pt" for u in SHA]
    for screen in CONTROL.glob("screen_u*"):
        for split in ("parallel", "validation"):
            for name in ("qualification.json", "episodes.jsonl", "manifest.json", "resolved_config.json"):
                if (screen / split / name).exists():
                    paths.append(screen / split / name)
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(set(paths)) if p.exists()}


def prepare():
    if any(p.exists() for p in (CONFIG, OUT / "base_config.json", OUT / "cases.json", OUT / "experiment.json")):
        raise FileExistsError("prepared experiment exists; verify it without --prepare")
    source, digest = load_checkpoint_file(CONTROL / "u001000.pt", map_location="cpu")
    assert digest == SHA[1000]
    cases = build_cases()  # Existing 24/16/16/16 complete-pair protocol.
    atomic_write_json(cases, OUT / "cases.json")
    base = portable_config(load_config("configs/mut_mappo/local_candidate_v1.json"))
    base.update(case_manifest=(OUT / "cases.json").relative_to(ROOT).as_posix(),
                seed=42, device="cpu", research_status="candidate")
    base["red_checkpoint"] = {"path": str(CONTROL / "u001000.pt"), "sha256": digest,
                              "capability_status": "NOT_QUALIFIED_diagnostic_candidate_only"}
    base["learning"].update(hidden_dim=source["model_spec"]["hidden_dim"],
                            gamma=float(source["training_config"]["gamma"]))
    for key in ("min_log_std", "max_log_std", "init_log_std"):
        if key in source["training_config"]:
            base["learning"][key] = source["training_config"][key]
    base["task"].update(reward_profile="c1_progress_team_v1", max_steps=500,
                         cameras_enabled=False, follow_spectator=False,
                         team_reward={"reference_dir": (OUT / "references").relative_to(ROOT).as_posix(),
                                      "gamma": base["learning"]["gamma"], "attack_weight": 10.0,
                                      "driving_weight": 1.0, "collision_cost": 50.0,
                                      "offroad_cost": 50.0, "road_rule_cost": 50.0,
                                      "task_violation_cost": 20.0})
    base["mutual"].update(quality_metric="efficiency", min_efficiency_gap=0.02,
                          gate_min_cases=8, gate_max_bad_rate=0.125)
    base["evaluation"].update(selection_metric="safe_efficiency_utility")
    base["run"].update(train_tick_budget=24000, rounds=1000, lr_horizon_rounds=1000,
                       episodes_per_team_update=4)
    cfg = experiment_config(base, "B0")
    base = copy.deepcopy(cfg)  # Same task and tick clock for screens/references.
    validate_config(cfg)
    atomic_write_json(base, OUT / "base_config.json")
    atomic_write_json(cfg, CONFIG)
    atomic_write_json({"schema": "first_point_experiment_v1", "suite": "B0_pilot_only",
        "scene_stage": "full", "base_config": str(OUT / "base_config.json"),
        "entries": [{"run_id": "B0_s42", "group": "B0", "seed": 42,
                     "config": str(CONFIG), "run_dir": str(OUT / "runs/B0_s42")}],
        "train_tick_budget": 24000,
        "split_counts": {k: len(v) for k, v in cases["splits"].items()},
        "execution": "serial only; A+B actual train ticks including setup; complete-boundary soft cap",
        "reward": "c1_progress_team_v1; team tail credit; no communication",
        "status": "PREPARED_NOT_RUN",
        "authorization": {"live_execution": False, "weak_red_override": False},
        "qualification": "red NOT_QUALIFIED; config preparation does not grant eligibility",
        "reference_repeats_planned": 2,
        "evaluation_route": "explicit evaluate_mut_mappo stages; wrapper evaluate uses repeats=1"},
        OUT / "experiment.json")


def baseline_evidence():
    result = {}
    for u in (400, 600, 800, 1000, 1200):
        screen = (CONTROL / f"reevaluation_u400_u600_20261027_v1/screen_u{u:06d}"
                  if u in (400, 600) else CONTROL / f"screen_u{u:06d}")
        item = {}
        for split in ("parallel", "validation"):
            path = screen / split / "qualification.json"
            q = read(path)
            r = q["reports"][0]
            checks, events, lane = {}, Counter(), {}
            for aid in ("0", "1"):
                checks[aid] = dict(Counter(k for c in r["cases"] for k in c["agents"][aid]["failed_checks"]))
                lane[aid] = sum(c["agents"][aid]["metrics"]["reference_lateral_p95_m"] for c in r["cases"]) / len(r["cases"])
            for c in r["cases"]:
                for kind in ("collision_event", "offroad_terminated", "road_violation_terminated", "mission_timeout_event"):
                    events[kind] += int(any(c["agents"][aid]["metrics"]["terminal_events"].get(kind, 0) for aid in ("0", "1")))
            item[split] = {"path": str(path), "sha256": sha256_file(path), "cases": r["case_count"],
                "both_reached": round(r["both_reached_fraction"] * r["case_count"]),
                "qualified": r["qualified"], "decision": r["qualification_decision"],
                "selection": q["selection"], "failed_checks_per_agent": checks,
                "terminal_event_episode_counts": dict(events), "mean_case_lane_p95_m": lane}
        result[str(u)] = item
    old = read(CONTROL / "screen_u001000/validation/qualification.json")["reports"][0]["cases"]
    new = read(CONTROL / "screen_u001200/validation/qualification.json")["reports"][0]["cases"]
    def paired(rows):
        return {(c["case"]["scenario"], c["case"]["seed"]): c["both_reached"] for c in rows}
    a, b = paired(old), paired(new)
    assert a.keys() == b.keys() and len(a) == 64
    result["u1000_to_u1200_pairs"] = dict(Counter(f"{int(a[k])}->{int(b[k])}" for k in a))
    metrics = [json.loads(s) for s in (BASE_RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    episodes = [json.loads(s) for s in (BASE_RUN / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    assert [m["update"] for m in metrics] == list(range(1, 1201))
    assert len(episodes) == 19200
    result["training_windows"] = {}
    for lo, hi in ((801, 1000), (1001, 1200)):
        rows = [e for e in episodes if lo <= e["update"] <= hi]
        result["training_windows"][f"{lo}-{hi}"] = {"episodes": len(rows),
            **{k: sum(bool(e[k]) for e in rows) / len(rows) for k in ("success", "collision", "offroad", "road_violation")},
            "critic_window_means": {k: sum(m["scalars"][k] for m in metrics if lo <= m["update"] <= hi) / (hi-lo+1)
                for k in ("critic/value_error_abs_mean", "critic/return_target_std", "critic/explained_variance")}}
    keys = ("policy/agent_0/preupdate_max_abs_ratio_error", "policy/agent_1/preupdate_max_abs_ratio_error",
            "reward_audit/max_abs_step_residual", "reward_audit/nonfinite_count", "train/update_skipped")
    result["integrity_maxima"] = {k: max(m["scalars"][k] for m in metrics) for k in keys}
    result["critic_nodes"] = {str(u): {k: metrics[u-1]["scalars"][k] for k in
        ("critic/value_error_abs_mean", "critic/return_target_std", "critic/explained_variance")} for u in (1000, 1200)}
    result["base_status"] = read(CONTROL / "status.json")
    result["u1200_completion_summary"] = read(CONTROL / "u001200_summary.json")
    launch, exit_record = read(BASE_RUN / "session/launch.json"), read(BASE_RUN / "session/exit.json")
    assert launch["launch_id"] == exit_record["launch_id"]
    assert launch["actual_worker_pid"] == exit_record["trainer_pid"] and exit_record["returncode"] == 0
    result["matching_u1200_exit"] = {"launch_id": launch["launch_id"],
                                    "actual_worker_pid": launch["actual_worker_pid"], "returncode": 0}
    return result


def policy_check(config):
    output = {}
    for u, digest in SHA.items():
        source, actual = load_checkpoint_file(CONTROL / f"u{u:06d}.pt", map_location="cpu")
        assert actual == digest
        red = make_actors(source["actors"], source["training_config"], (0, 1), "cpu")
        blue = make_actors({2: source["actors"][0], 3: source["actors"][1]}, source["training_config"], (2, 3), "cpu")
        learning = dict(config["learning"], method="off", algorithm="mappo")
        teams = {n: TeamLearner(n, 42 + offset, source["actors"], learning, "cpu")
                 for n, offset in (("A", 1000), ("B", 2000))}
        red_params = {id(p) for actor in red.values() for p in actor.parameters()}
        for team in teams.values():
            for aid in (2, 3):
                assert all(torch.equal(t, source["actors"][aid-2][k]) for k, t in team.actors[aid].state_dict().items())
                assert not any(id(p) in red_params for group in team.actor_optims[aid].param_groups for p in group["params"])
            assert int(team.value_normalizer.update_count) == 0
            assert not team.critic_optim.state and all(not o.state for o in team.actor_optims.values())
            assert next(team.critic.parameters()).shape[1] == 84
        assert all(not p.requires_grad for actor in (*red.values(), *blue.values()) for p in actor.parameters())
        assert all(torch.isfinite(actor.get_deterministic_action(torch.zeros(1, 20))).all() for actor in red.values())
        output[str(u)] = {"sha256": actual, "model_spec": source["model_spec"],
            "strict_roles_and_frozen_red": True, "blue_source_roles": {"2": 0, "3": 1},
            "fresh_critic_input": 84, "fresh_optimizers_and_valuenorm": True,
            "hidden_dim": config["learning"]["hidden_dim"], "gamma": config["learning"]["gamma"],
            "source_environment": {k: source["training_config"].get(k) for k in
                ("reward_version", "pace_profile", "base_driving_reward_profile", "max_steps", "lidar_range_m", "ttc_max_range_m")},
            "pilot_max_steps": config["task"]["max_steps"]}
    return output


def run_tests():
    files = sorted((ROOT / "tests").glob("test_mut_mappo_*.py"))
    files += [ROOT / "tests/test_first_point_experiment.py", ROOT / "tests/test_mutual_distillation_controls.py"]
    modules = ["tests." + f.stem for f in files]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (OUT / "checks/tests.log").write_text(log.getvalue(), encoding="utf-8")
    report = {"modules": modules, "run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful(),
              "scope": "CPU/synthetic environment/local OpenDRIVE only; carla.Client forbidden"}
    atomic_write_json(report, OUT / "checks/test_results.json")
    assert result.wasSuccessful(), log.getvalue()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--tests", action="store_true")
    args = p.parse_args()
    (OUT / "checks").mkdir(exist_ok=True)
    before = protected_inputs()
    with patch("carla.Client", side_effect=AssertionError("CARLA Client forbidden during offline preparation")):
        if args.prepare:
            prepare()
        cfg = validate_config(load_config(CONFIG))
        from evaluation.evaluate_mut_mappo import evaluate
        screens = {s: evaluate(CONFIG, split=s, team="A", conditions=("C0", "C1"), validate_only=True)
                   for s in ("train", "gate")}
        result = {"status": "OFFLINE_COMPATIBLE_NOT_LIVE_VALIDATED", "created_at": datetime.now(timezone.utc).isoformat(),
            "carla_client_forbidden": True, "carla_connected": False, "world_ticks": 0,
            "live_started": False, "weak_red_authorized": False,
            "configuration": cfg["_root_validation"], "checkpoint_checks": policy_check(cfg),
            "C0_C1_preflight": screens, "historical_base_evidence": baseline_evidence()}
        if args.tests:
            result["tests"] = run_tests()
        elif (OUT / "checks/test_results.json").exists():
            result["prior_test_report"] = read(OUT / "checks/test_results.json")
    after = protected_inputs()
    assert before == after, "historical inputs changed during offline audit"
    result["historical_input_hashes_unchanged"] = True
    atomic_write_json(before, OUT / "checks/protected_input_hashes.json")
    sources = sorted(set(ROOT.glob("training/*mut*mappo*.py")) | {
        ROOT / "utils/run_first_point_experiments.py", ROOT / "utils/mut_mappo_session.py",
        ROOT / "training/first_point_scenarios.py", ROOT / "training/mutual_distillation.py",
        ROOT / "training/train_no_rllib_mappo_robust_v2.py", ROOT / "training/checkpoint_protocol.py",
        ROOT / "training/value_normalization.py", ROOT / "utils/adversarial_metrics.py",
        ROOT / "training/action_protocol.py", ROOT / "environment/carla_multi_agent_env.py",
        ROOT / "environment/base_driving_env.py", ROOT / "environment/base_driving_reward.py",
        *ROOT.glob("environment/base_reward_v*.py"), *ROOT.glob("environment/instant_pace_reward*.py"),
        ROOT / "environment/route_pace_reward.py",
        ROOT / "models/actor_critic.py", ROOT / "models/mut_mappo_critics.py",
        *ROOT.glob("environment/mut_mappo*.py"), *ROOT.glob("evaluation/*mut*mappo*.py"),
        ROOT / "evaluation/driving_policy_metrics.py"})
    atomic_write_json({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources}, OUT / "checks/source_inventory.json")
    atomic_write_json(result, OUT / "checks/offline_audit.json")
    print(json.dumps({"status": result["status"], "directory": str(OUT),
        "split_counts": cfg["_root_validation"]["split_counts"],
        "C0_C1_planned_episodes": sum(s["planned_episodes"] for s in screens.values()),
        "tests": result.get("tests"), "carla_connected": False, "world_ticks": 0,
        "historical_input_hashes_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
