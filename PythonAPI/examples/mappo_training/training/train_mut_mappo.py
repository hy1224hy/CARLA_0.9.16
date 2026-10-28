"""Serial four-car Mut-MAPPO runner, reusing the existing MAPPO update core.

Candidate code/configuration is executable for explicit local development;
live geometry and scientific claims remain unverified until independent tests.
CARLA must already have been started by the user. No server is launched here.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from training.action_protocol import action_tanh_to_env
from training.checkpoint_protocol import (atomic_write_json, capture_rng_state,
    load_checkpoint_file, restore_rng_state)
from training.mut_mappo_config import (ROOT, MODEL_SPEC, cases_for, load_config,
    make_actors, portable_config, resolve_path, run_identity, validate_config, task_contract_digest, config_digest)
from training.mut_mappo_learning import GateState, TeamLearner, actor_states_hash
from training.mut_mappo_checkpoint import CheckpointStore, SCHEMA_VERSION, _validate_payload
from training.train_no_rllib_mappo_robust_v2 import (RolloutBuffer, build_active_step_cache,
    refresh_observations_after_active_change, validate_active_mask, write_active_transitions)
from utils.mut_mappo_session import CarlaOwner, last_json_record
from visualization.mut_mappo_monitor import MutMAPPOMonitor, add_monitor_arguments, options_from_args
from environment.mut_mappo_efficiency_reward import PROFILE as TEAM_PROFILE, ReferenceStore, reference_identity, settle_team_tail
from evaluation.driving_policy_metrics import DrivingMetrics


def now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(record), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def validate_checkpoint_for_config(config, path):
    payload, _digest = load_checkpoint_file(resolve_path(path), map_location="cpu")
    from evaluation.mut_mappo_selection import EXPORT_SCHEMA, EXPORT_PROTOCOL
    inference_only = payload.get("schema_version") == EXPORT_SCHEMA
    if inference_only:
        if payload.get("protocol") != EXPORT_PROTOCOL or payload.get("purpose") != "frozen_blue_inference_only_not_training_resume":
            raise ValueError("unregistered selected-policy inference protocol")
        if payload.get("team") not in {"A", "B"} or set(payload.get("actors", {})) != {2, 3}:
            raise ValueError("selected policy must explicitly identify both blue roles")
        if payload.get("actor_hash") != actor_states_hash(payload["actors"]):
            raise ValueError("selected actor content hash differs")
        if payload.get("completed_round", 0) < 1:
            raise ValueError("selected inference model has no completed trained boundary")
    else:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("C2 requires a current complete training checkpoint or selected inference export")
        _validate_payload(payload)
    identity = payload.get("identity", {})
    if identity.get("training_reference_hashes"):
        import hashlib
        from training.checkpoint_protocol import sha256_file
        directory = resolve_path(config["task"]["team_reward"]["reference_dir"])
        for key, digest in identity["training_reference_hashes"].items():
            if sha256_file(directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")) != digest:
                raise ValueError("training C1 reference changed since checkpoint creation")
    expected = config["_root_validation"]
    if identity.get("model_spec") != MODEL_SPEC or identity.get("red_sha256") != expected["red_sha256"]:
        raise ValueError("evaluation model/red identity differs")
    if identity.get("case_manifest_sha256") != expected["case_manifest_sha256"]:
        raise ValueError("evaluation case manifest differs from training identity")
    if identity.get("algorithm", config["algorithm"]) != config["algorithm"]:
        raise ValueError("evaluation configuration algorithm differs from source training identity")
    expected_task = identity.get("task_contract_sha256")
    if expected_task is not None:
        if expected_task != task_contract_digest(config):
            raise ValueError("evaluation task/reward/lifecycle differs from the training contract")
    elif identity.get("config_sha256") != config_digest(config):
        raise ValueError("legacy task identity requires the exact original configuration")
    if inference_only:
        make_actors(payload["actors"], payload["training_config"], (2, 3), "cpu")
    else:
        for name in payload["teams"]:
            team = payload["teams"][name]
            make_actors(team["actors"], team["config"], (2, 3), "cpu")
    return payload


class Runtime:
    def __init__(self, config, env, red_actors, blue_actors, condition="C2", team="A"):
        self.config, self.env, self.red_actors, self.blue_actors = config, env, red_actors, blue_actors
        self.condition, self.team = condition, team
        self.device = torch.device(config.get("device", "cpu"))

    def run_case(self, case, *, seed=None, deterministic=True, trace_path=None):
        return run_case(self, case, seed=seed, deterministic=deterministic, trace_path=trace_path)

    def close(self):
        try:
            monitor = getattr(self, "monitor", None)
            if monitor is not None:
                monitor.detach()
        finally:
            self.env.close()


def create_runtime(config, checkpoint=None, condition="C2", team="A", audit_callback=None):
    if "_red_payload" not in config:
        config = validate_config(config)
    if condition not in {"C0", "C1", "C2"} or team not in {"A", "B"}:
        raise ValueError("invalid runtime condition/team")
    source = config["_red_payload"]
    before = capture_rng_state()
    try:
        red = make_actors(source["actors"], source["training_config"], (0, 1), config["device"])
        blue = {}
        if condition == "C1" or checkpoint is None:
            states = {2: source["actors"][0], 3: source["actors"][1]}
            blue = make_actors(states, source["training_config"], (2, 3), config["device"])
        elif condition == "C2":
            payload = validate_checkpoint_for_config(config, checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
            if "teams" in payload:
                blue = make_actors(payload["teams"][team]["actors"], payload["teams"][team]["config"], (2, 3), config["device"])
            else:
                if team != payload["team"]:
                    raise ValueError("requested team differs from the selected inference team")
                blue = make_actors(payload["actors"], payload["training_config"], (2, 3), config["device"])
    finally:
        restore_rng_state(before)
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world, _restore_failed_initialization
    from environment.mut_mappo_env import MutMAPPOEnv
    idle = _preflight_idle_world(host=config["host"], port=config["port"], expected_version="0.9.16")
    red_config = dict(source["training_config"], reward_gamma=source["training_config"]["gamma"])
    reference = None
    if config["task"].get("reward_profile") == TEAM_PROFILE and condition == "C2" and not config.get("_building_reference"):
        reference = ReferenceStore(config["task"]["team_reward"]["reference_dir"], reference_identity(config))
    try:
        env = MutMAPPOEnv(red_checkpoint_config=red_config, task_config=config["task"],
                         host=config["host"], port=config["port"], seed=config["seed"],
                         num_agents=2 if condition == "C0" else 4, audit_callback=audit_callback,
                         reference_store=reference)
        env._original_synchronous_mode = idle["synchronous_mode"]
        env._original_fixed_delta_seconds = idle["fixed_delta_seconds"]
    except BaseException:
        _restore_failed_initialization(host=config["host"], port=config["port"], preflight=idle)
        raise
    return Runtime(config, env, red, blue, condition, team)


def policy_controls(actors, obs, active, device, deterministic=True):
    controls = {}
    with torch.no_grad():
        for aid, actor in actors.items():
            if not active.get(aid, False):
                continue
            tensor = torch.as_tensor(obs[aid], dtype=torch.float32, device=device).unsqueeze(0)
            action = actor.get_deterministic_action(tensor) if deterministic else actor.sample_action_z(tensor)[0]
            controls[aid] = action_tanh_to_env(action.squeeze(0).cpu().numpy())
    return controls


def _flags(env, info, dones):
    ids = list(range(env.num_agents))
    active = validate_active_mask(info["active_mask"], ids, context="MutMAPPO physical lifecycle")
    terminated = info.get("agent_terminated", info.get("terminated"))
    truncated = info.get("agent_truncated", info.get("truncated"))
    if set(dones) != set(ids) or set(terminated) != set(ids) or set(truncated) != set(ids):
        raise ValueError("physical lifecycle flags must have all fixed ID slots")
    for aid in ids:
        if bool(terminated[aid]) and bool(truncated[aid]) or bool(dones[aid]) != bool(terminated[aid] or truncated[aid]):
            raise ValueError("inconsistent terminated/truncated/done flags")
        if active[aid] != (not bool(dones[aid])):
            raise ValueError("active mask differs from physical lifecycle")
    return active, terminated, truncated


def _episode(runtime, case, *, learner=None, buffer=None, seed=None, deterministic=True, trace_path=None, step_callback=None):
    case = copy.deepcopy(case)
    if seed is not None:
        case["seed"] = int(seed)
    env = runtime.env
    ticks_before = dict(getattr(env, "_world_tick_counts", {}))
    monitor = getattr(runtime, "monitor", None)
    if monitor is not None:
        monitor.begin(runtime, case)
    obs, info = env.reset_case(case)
    if monitor is not None:
        monitor.ready()
    # Reset evidence is not repeated by every step and may be mutated during
    # actor cleanup. Preserve it before the first control, including the exact
    # realized initial state used for paired C0/C1/C2 comparisons.
    case_geometry = copy.deepcopy(info.get("case_geometry"))
    setup_contacts = copy.deepcopy(info.get("setup_contacts", {}))
    source_identity = runtime.config.get("_root_validation", {})
    base_scenario = case.get("base_scenario")
    initial_evidence_identity = {
        "case_key": case["case_key"], "family": case["family"],
        "seed": case.get("seed"), "condition": runtime.condition,
        "team": runtime.team,
        "case_manifest_sha256": source_identity.get("case_manifest_sha256"),
        "base_scenario_sha256": source_identity.get("scenario_hashes", {}).get(base_scenario)
            if isinstance(base_scenario, str) else None,
        "red_checkpoint_sha256": source_identity.get("red_sha256"),
    }
    sensor_timing = {"collision_callback_frame_barrier": "not_enforced",
                     "validation_status": "live_validation_pending"}
    physical_ids = list(range(env.num_agents))
    active = validate_active_mask(info["active_mask"], physical_ids, context="MutMAPPO reset")
    if trace_path:
        append_record(trace_path, {
            "record_kind": "case_initial_geometry", "step": 0,
            **initial_evidence_identity,
            "case_geometry": case_geometry, "setup_contacts": setup_contacts,
            "sensor_timing": sensor_timing, "obs": obs,
            "active_mask": active,
        })
    starts = {aid: len(buffer.data[aid]) for aid in (2, 3)} if buffer else {}
    returns = {aid: 0.0 for aid in physical_ids}
    control_counts = {aid: 0 for aid in physical_ids}
    brake_counts = {aid: 0 for aid in physical_ids}
    max_reward_residual = 0.0
    step = 0
    initial_states = (case_geometry or {}).get("realized_initial_states", {})
    driving = DrivingMetrics(float(runtime.config["task"].get("timestep", 0.1)),
                             {aid: state["speed"] for aid, state in initial_states.items() if "speed" in state})
    team_timeline = []
    while any(active.values()):
        controls = policy_controls(runtime.red_actors, obs, active, runtime.device, True)
        cache = {}
        if learner is None:
            controls.update(policy_controls(runtime.blue_actors, obs, active, runtime.device, deterministic))
        else:
            blue_mask = {aid: active[aid] for aid in (2, 3)}
            blue_controls, cache = learner.sample(obs, env.get_global_state(), blue_mask)
            controls.update(blue_controls)
        if monitor is not None:
            monitor.before_step(env)
        next_obs, rewards, dones, info = env.step(controls)
        if monitor is not None:
            monitor.after_step(env, step + 1, rewards, info)
        if "team_step_reward" in info:
            team_timeline.append(float(info["team_step_reward"]))
        from environment.base_driving_env import BaseDrivingEnv
        longitudinal = (BaseDrivingEnv.get_longitudinal_safety_state(env)
                        if hasattr(env, "vehicles") else {})
        references = getattr(env, "road_reference_waypoints", {})
        driving.update(active, info.get("post_tick_kinematics", {}), info.get("reward_diagnostics", {}),
                       longitudinal, {aid: wp.transform.rotation.yaw for aid, wp in references.items()})
        for aid, control in controls.items():
            control_counts[aid] += 1
            brake_counts[aid] += int(control[2] > 0.01 and control[0] <= 0.01)
        for audit in info.get("blue_reward_audit", {}).values():
            residual = audit.get("reward_residual")
            if residual is not None:
                max_reward_residual = max(max_reward_residual, abs(float(residual)))
        next_active, terminated, truncated = _flags(env, info, dones)
        if step_callback is not None:
            step_callback(info)
        step += 1
        pre_cleanup_state = np.asarray(info.get("bootstrap_global_state", np.concatenate(
            [np.asarray(next_obs[aid], dtype=np.float32) for aid in physical_ids])), dtype=np.float32).copy()
        if pre_cleanup_state.shape != (20 * len(physical_ids),) or not np.isfinite(pre_cleanup_state).all():
            raise ValueError("terminal bootstrap state must retain every physical slot")
        if buffer is not None:
            write_active_transitions(buffer, cache, rewards, dones, terminated, truncated, case["case_key"], context="MutMAPPO write")
        for aid, reward in rewards.items():
            returns[aid] += float(reward)
        if trace_path:
            append_record(trace_path, {"case_key": case["case_key"], "condition": runtime.condition,
                "team": runtime.team, "step": step, "obs": obs, "controls": controls, "rewards": rewards,
                "active_mask": active, "next_active_mask": next_active,
                "terminated": terminated, "truncated": truncated,
                "mut_metrics": info.get("mut_metrics", {}),
                "kinematics": info.get("post_tick_kinematics", info.get("kinematics", info.get("pre_collision_kinematics", {}))),
                "driving_diagnostics": info.get("reward_diagnostics", {}), "longitudinal_safety": longitudinal})
        end_reason = info.get("episode_end_reason", "running" if any(next_active.values()) else "ended")
        obs = refresh_observations_after_active_change(env, next_obs, active, next_active, end_reason,
            physical_ids, context="MutMAPPO physical refresh")
        active = next_active
        if step > runtime.config["task"]["max_steps"] + 1:
            raise RuntimeError("environment did not honor the frozen task limit; invalid episode")
    if buffer is not None:
        tail_audit = (settle_team_tail(buffer, starts, team_timeline, float(runtime.config["learning"]["gamma"]))
                      if team_timeline else {})
        for aid in (2, 3):
            if len(buffer.data[aid]) > starts[aid]:
                buffer.mark_episode_end(aid)
                buffer.add_episode_bootstrap(aid, pre_cleanup_state if truncated[aid] else None)
        buffer.add_scenario(case["case_key"])
    metric = dict(info.get("mut_metrics", {}))
    if team_timeline:
        metric["team_discounted_return"] = math.fsum(float(runtime.config["learning"]["gamma"]) ** i * r
                                                    for i, r in enumerate(team_timeline))
        metric["return_semantics"] = "returns=actual active step rewards; team_tail_credit folds post-departure reward for PPO"
    metric.update({"case_key": case["case_key"], "family": case["family"], "seed": case.get("seed"),
        "direction": case.get("direction"), "pair_id": case.get("pair_id"),
        "startup_mode": case.get("startup_mode"), "red0_role": case.get("red0_role"),
        "condition": runtime.condition, "team": runtime.team, "steps": step, "returns": returns,
        "case_geometry": case_geometry, "setup_contacts": setup_contacts,
        "initial_evidence_identity": initial_evidence_identity, "sensor_timing": sensor_timing,
        "execution_mode": "stochastic_training" if learner else "deterministic" if deterministic else "stochastic_evaluation",
        "record_kind": "training_episode" if learner else "independent_evaluation",
        "valid": bool(metric.get("valid", True)),
        "tick_counts": {key: value - ticks_before.get(key, 0) for key, value in info.get("world_tick_counts", {}).items()},
        "tick_count_scope": "per_case_explicit_ticks_including_reset",
        "brake_mode_rate": {aid: brake_counts[aid]/control_counts[aid] if control_counts[aid] else None for aid in physical_ids},
        "max_abs_reward_residual": max_reward_residual,
        "driving_metrics": driving.summary(),
        "team_tail_credit": tail_audit if buffer is not None else {},
        "agent_terminated": terminated, "agent_truncated": truncated, "created_at": now()})
    return json_value(metric)


def run_case(runtime, case, *, seed=None, deterministic=True, trace_path=None, step_callback=None):
    before = capture_rng_state()
    try:
        torch.manual_seed(int(seed if seed is not None else case.get("seed", runtime.config["seed"])))
        return _episode(runtime, case, seed=seed, deterministic=deterministic, trace_path=trace_path, step_callback=step_callback)
    finally:
        restore_rng_state(before)


class CostLedger:
    def __init__(self, store):
        self.store, self.phase = store, "train"
        self.counts = {"train": 0, "validation": 0, "train_decision": 0, "validation_decision": 0, "setup": 0, "total": 0}
        event = last_json_record(store.audit_path)
        if event and "actual_counters" in event:
            self.counts = event["actual_counters"]

    def tick(self, event):
        n = int(event.get("ticks", 1))
        kind = event.get("kind", "setup")
        self.counts[self.phase] += n
        self.counts["total"] += n
        if kind == "decision":
            self.counts[self.phase + "_decision"] += n
        else:
            self.counts["setup"] += n
        self.store.append_audit({"kind": "world_ticks", "phase": self.phase, "tick_kind": kind,
                                 "actual_counters": self.counts})


def refresh_peers(runtime, teams, gate, config, completed_round, ledger, run_dir):
    version = gate.version + 1
    snapshots = {name: team.snapshot(version) for name, team in teams.items()}
    stats = {name: {} for name in teams}
    ledger.phase = "validation"
    monitor = getattr(runtime, "monitor", None)
    if monitor is not None:
        monitor.context(operation="gate", round=completed_round)
        monitor.phase("validating", "正在评估互学习教师与选择性门控")
    try:
        for name, team in teams.items():
            runtime.team, runtime.blue_actors = name, team.frozen_actors()
            for case in cases_for(config, "gate"):
                result = run_case(runtime, case)
                append_record(Path(run_dir) / "gate_episodes.jsonl", dict(result, teacher_hash=snapshots[name]["hash"], version=version))
                if not result["valid"]:
                    raise RuntimeError("invalid gate case: refuse unmatched teacher statistics")
                record = stats[name].setdefault(case["family"], {"n_all": 0, "q_count": 0, "b_count": 0, "case_keys": []})
                record["n_all"] += 1
                record["q_count"] += int(bool(result.get("blue_eligible")) and bool(result.get("response")))
                record["b_count"] += int(not result.get("blue_eligible", False))
                record["case_keys"].append(case["case_key"])
                record.update(teacher_hash=snapshots[name]["hash"], version=version)
                if config["mutual"].get("quality_metric") == "efficiency":
                    record["utility_sum"] = record.get("utility_sum", 0.0) + float(result["safe_efficiency_utility"])
        gate.refresh({n: s["hash"] for n, s in snapshots.items()}, stats, version=version, completed_round=completed_round)
        for name, team in teams.items():
            peer = "B" if name == "A" else "A"
            team.set_peer(snapshots[peer], gate.gates_for(name, snapshots[peer]["hash"]))
        return snapshots
    finally:
        ledger.phase = "train"
        if monitor is not None:
            monitor.context(operation="train")


def train(config, run_dir, *, resume=False, stop_round=None, env_factory=create_runtime):
    run_dir = Path(run_dir).resolve()
    store = CheckpointStore(run_dir, run_identity(config))
    source = config["_red_payload"]
    initial = {2: source["actors"][0], 3: source["actors"][1]}
    learning = dict(config["learning"], method=config["method"], algorithm=config["algorithm"])
    topology = ("A",) if config["team_count"] == 1 else ("A", "B")
    teams = {name: TeamLearner(name, config["seed"] + offset, initial, learning, config["device"])
             for name, offset in (("A", 1000), ("B", 2000)) if name in topology}
    gate = GateState(config["mutual"])
    payload = {"teams": {n: t.state_dict() for n, t in teams.items()}, "teachers": {n: None for n in topology},
        "team_count": config["team_count"], "algorithm": config["algorithm"],
        "gate": gate.state_dict(), "completed_round": 0, "next_team": "A", "counters": {},
        "last_metrics": {"status": "INITIALIZED_NOT_TRAINED"}, "round_segments": {},
        "config": portable_config(config), "next_refresh_train_tick": config["mutual"].get("warmup_train_ticks", 0)}
    if resume:
        payload = store.load()
        for name, team in teams.items():
            team.load_state_dict(payload["teams"][name])
        gate.load_state_dict(payload["gate"])
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            permitted = {"session"}
            if any(path.name not in permitted for path in run_dir.iterdir()):
                raise FileExistsError("fresh run directory contains experiment evidence; use a new RunID or --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(portable_config(config), run_dir / "resolved_config.json")
        atomic_write_json(config["_case_manifest"], run_dir / "case_manifest.json")
        store.save(payload)
    if (run_dir / "STOP_REQUESTED").exists():
        raise RuntimeError("STOP_REQUESTED exists; explicitly archive it before starting another segment")
    ledger = CostLedger(store)
    runtime = None
    tick_stopping = config["run"].get("stop_rule") == "train_ticks"
    stop = stop_round if stop_round is not None else (
        payload["completed_round"] + max(1, config["run"]["train_tick_budget"] - ledger.counts["train"])
        if tick_stopping else config["run"]["rounds"])
    if not payload["completed_round"] < stop or (not tick_stopping and stop > config["run"]["lr_horizon_rounds"]):
        raise ValueError("stop round must advance; round-clock runs cannot exceed their frozen LR horizon")
    segment = uuid.uuid4().hex
    status_path = run_dir / "session/status.json"
    status = {"started_at": now(), "segment_id": segment, "status": "STARTING", "method": config["method"]}
    monitor = MutMAPPOMonitor(config, run_dir, planned_rounds=None if tick_stopping else stop)
    terminal, monitor_result = "error", {}
    try:
        monitor.start()
        runtime = env_factory(config, audit_callback=ledger.tick)
        runtime.monitor = monitor
        train_cases = cases_for(config, "train")
        family_map = {case["case_key"]: case["family"] for case in train_cases}
        teachers = payload["teachers"]
        for round_no in range(payload["completed_round"] + 1, stop + 1):
            if ledger.counts["train"] >= config["run"]["train_tick_budget"]:
                break
            round_metrics, episode_metrics = {}, {}
            monitor.context(round=round_no, update=round_no, operation="train",
                            train_tick_budget=config["run"]["train_tick_budget"])
            order = ("A",) if len(topology) == 1 else (payload["next_team"], "B" if payload["next_team"] == "A" else "A")
            for name in order:
                team = teams[name]
                runtime.team = name
                monitor.context(team=name)
                runtime.blue_actors = team.actors
                if hasattr(runtime.env, "set_cost_coefficient"):
                    runtime.env.set_cost_coefficient(team.cost_state["lambda"])
                buffer, records = RolloutBuffer([2, 3]), []
                with team.rng_context():
                    if config["_case_manifest"].get("sampling_protocol") == "first_point_complete_pairs_v1":
                        from training.first_point_scenarios import sample_pairs
                        collection_cases = sample_pairs(train_cases, round_no - 1, config["run"]["episodes_per_team_update"])
                    else:
                        collection_cases = [train_cases[((round_no-1)*config["run"]["episodes_per_team_update"]+ep) % len(train_cases)]
                                            for ep in range(config["run"]["episodes_per_team_update"])]
                    for case in collection_cases:
                        result = _episode(runtime, case, learner=team, buffer=buffer,
                                          seed=team.next_episode_seed(), deterministic=False)
                        if not result["valid"]:
                            raise RuntimeError("invalid training episode cannot enter PPO")
                        records.append(result)
                        append_record(run_dir / "episodes.jsonl", dict(result, round=round_no, segment_id=segment))
                progress = min(1.0, ledger.counts["train"]/config["run"]["train_tick_budget"]
                    if config["run"].get("schedule_clock") == "train_ticks"
                    else (round_no-1)/config["run"]["lr_horizon_rounds"])
                for key, optimizer in (("lr_actor", team.actor_optims), ("lr_critic", team.critic_optims)):
                    value = config["learning"][key] * max(0.1, 1-progress)
                    for opt in optimizer.values():
                        for group in opt.param_groups:
                            group["lr"] = value
                    team.schedule_state["actor_lr" if key == "lr_actor" else "critic_lr"] = value
                monitor.phase("optimizing", f"正在更新蓝方团队 {name} 的模型", team=name)
                round_metrics[name] = team.update(buffer, aux_context={"family_by_scenario": family_map})
                bad_rate = sum(not r.get("blue_eligible", False) for r in records)/len(records)
                coeff = (1.0 if config["task"].get("reward_profile") == TEAM_PROFILE else
                         team.cost_state["lambda"] * (config["run"]["cost_up"] if bad_rate > config["run"]["cost_target"] else config["run"]["cost_down"]))
                team.cost_state = {"lambda": max(config["run"]["cost_min"], min(config["run"]["cost_max"], coeff)), "updates": round_no}
                episode_metrics[name] = {"n": len(records), "q": sum(r.get("blue_eligible", False) and r.get("response", False) for r in records)/len(records),
                    "b": bad_rate, "red_success_rate": sum(r.get("red_success", False) for r in records)/len(records),
                    "scenario_coverage": {case_key: sum(r["case_key"] == case_key for r in records) for case_key in family_map},
                    "agents": {}, "max_abs_reward_residual": max(r.get("max_abs_reward_residual", 0) for r in records)}
                if config["task"].get("reward_profile") == TEAM_PROFILE:
                    episode_metrics[name]["safe_efficiency_utility"] = sum(r["safe_efficiency_utility"] for r in records) / len(records)
                for aid in range(4):
                    def measurements(key):
                        values = []
                        for record in records:
                            agents = record.get("agent_results", record.get("red_results", {}))
                            item = agents.get(str(aid), agents.get(aid, {}))
                            value = item.get(key)
                            if value is not None:
                                values.append(float(value))
                        return sum(values)/len(values) if values else None
                    brakes = [r["brake_mode_rate"][str(aid)] for r in records if r["brake_mode_rate"].get(str(aid)) is not None]
                    episode_metrics[name]["agents"][str(aid)] = {"mean_speed_mps": measurements("mean_decision_speed_mps"),
                        "mean_completion": measurements("route_completion"), "brake_mode_rate": sum(brakes)/len(brakes) if brakes else None}
                status.update(status="RUNNING_PARTIAL_AB", collecting_team=name, in_progress_round=round_no,
                              completed_round=payload["completed_round"], actual_counters=ledger.counts, at=now())
                atomic_write_json(status, status_path)
            warmup, interval = config["mutual"]["warmup_rounds"], config["mutual"]["refresh_rounds"]
            refresh = round_no >= warmup and (round_no-warmup) % interval == 0
            if config["mutual"].get("schedule_clock") == "train_ticks":
                refresh = ledger.counts["train"] >= payload["next_refresh_train_tick"]
            if len(topology) == 2 and refresh:
                teachers = refresh_peers(runtime, teams, gate, config, round_no, ledger, run_dir)
                if config["mutual"].get("schedule_clock") == "train_ticks":
                    payload["next_refresh_train_tick"] = ledger.counts["train"] + config["mutual"]["refresh_train_ticks"]
            candidate_payload = copy.deepcopy(payload)
            candidate_payload.update(teams={n: t.state_dict() for n, t in teams.items()}, teachers=teachers,
                gate=gate.state_dict(), completed_round=round_no,
                next_team="A" if len(topology) == 1 else "B" if order[0] == "A" else "A",
                counters=copy.deepcopy(ledger.counts))
            candidate_payload["round_segments"][str(round_no)] = segment
            metrics = {"completed_round": round_no, "segment_id": segment, "method": config["method"],
                "teams": round_metrics, "training_window": episode_metrics, "actual_counters": ledger.counts,
                "gate": gate.state_dict(), "source": "training_window_not_independent_evaluation", "created_at": now(),
                "algorithm": config["algorithm"], "team_count": config["team_count"],
                "budget_policy": "complete_boundary_soft_cap", "train_tick_budget": config["run"]["train_tick_budget"],
                "train_tick_overshoot": max(0, ledger.counts["train"]-config["run"]["train_tick_budget"]),
                "schedule_clock": config["run"].get("schedule_clock", "rounds")}
            metrics["stop_rule"] = config["run"].get("stop_rule", "rounds")
            metrics["training_parameter_counts"] = {name: {
                "actors": sum(p.numel() for actor in team.actors.values() for p in actor.parameters()),
                "critics": sum(p.numel() for critic in team.critics.values() for p in critic.parameters())}
                for name, team in teams.items()}
            monitor.phase("saving", "正在保存完整 A+B 训练边界")
            store.save(candidate_payload, metrics)
            candidate_payload["last_metrics"] = copy.deepcopy(metrics)
            payload = candidate_payload
            monitor.phase("saving", "完整 A+B 边界已保存", safe_round=payload["completed_round"])
            print(json.dumps({"safe_round": round_no, "actual_train_ticks": ledger.counts["train"],
                              "training_window": episode_metrics}, ensure_ascii=False), flush=True)
            status.update(status="SAFE_AB_BOUNDARY", completed_round=round_no, at=now())
            atomic_write_json(status, status_path)
            if (run_dir / "STOP_REQUESTED").exists():
                break
        status.update(status="SEGMENT_COMPLETED", completed_round=payload["completed_round"], ended_at=now())
        status.update(actual_counters=copy.deepcopy(ledger.counts),
            train_tick_budget=config["run"]["train_tick_budget"],
            train_tick_overshoot=max(0, ledger.counts["train"]-config["run"]["train_tick_budget"]),
            budget_policy="complete_boundary_soft_cap",
            training_budget_reached=ledger.counts["train"] >= config["run"]["train_tick_budget"],
            stop_rule=config["run"].get("stop_rule", "rounds"))
        atomic_write_json(status, status_path)
        terminal = "completed"
        monitor_result = {"segment_status": status["status"], "actual_counters": ledger.counts}
        return payload
    except BaseException as error:
        terminal = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        monitor_result = {"error": f"{type(error).__name__}: {error}"}
        # Manifest commit can succeed before latest.pt/metric replay fails.
        # Report the verified authority, never the uncommitted candidate state.
        try:
            safe_payload = store.load()
            safe_round = safe_payload["completed_round"]
        except Exception as verification_error:
            safe_round = payload["completed_round"]
            status["boundary_verification_error"] = str(verification_error)
        status.update(status="INTERRUPTED_SAFE_BOUNDARY_RETAINED", completed_round=payload["completed_round"],
                      error=f"{type(error).__name__}: {error}", at=now())
        status["completed_round"] = safe_round
        atomic_write_json(status, status_path)
        raise
    finally:
        monitor.phase("finalizing", "正在清理四车环境并保留最终画面")
        monitor.detach()
        try:
            if runtime is not None:
                runtime.close()
        except BaseException as cleanup_error:
            terminal = "error"
            monitor_result["cleanup_error"] = str(cleanup_error)
            raise
        finally:
            monitor.finish(terminal, result=monitor_result, message="训练阶段已结束，画面与安全保存结果已保留")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mut_mappo/local_candidate_v1.json")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-round", type=int)
    parser.add_argument("--mutual-mode", choices=["off", "plain", "selective"])
    parser.add_argument("--algorithm", choices=["mappo", "ippo"])
    parser.add_argument("--team-count", type=int, choices=[1, 2])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-candidate", action="store_true")
    add_monitor_arguments(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mutual_mode:
        config["method"] = args.mutual_mode
    if args.algorithm:
        config["algorithm"] = args.algorithm
    if args.team_count is not None:
        config["team_count"] = args.team_count
    if args.seed is not None:
        config["seed"] = args.seed
    config = validate_config(config)
    config["_monitor_options"] = options_from_args(args)
    if args.validate_only:
        print(json.dumps(config["_root_validation"], ensure_ascii=False, indent=2))
        return
    if not args.run_dir:
        parser.error("--run-dir is required for live training")
    if config["research_status"] != "frozen" and not args.allow_candidate:
        parser.error("candidate local development must be explicitly labeled with --allow-candidate; no formal eligibility")
    with CarlaOwner(config["host"], config["port"], args.run_dir, "training"):
        train(config, args.run_dir, resume=args.resume, stop_round=args.stop_round)


if __name__ == "__main__":
    main()
