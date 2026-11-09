"""Portable candidate configuration and policy identities; no CARLA connection."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from models.actor_critic import Actor
from environment.mut_mappo_efficiency_reward import TEAM_PROFILES, CRITICAL_PROFILES
from training.action_protocol import POLICY_ACTION_DIM, action_protocol_spec
from training.checkpoint_protocol import load_checkpoint_file, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPEC = {"obs_dim": 20, "action_dim": 2, "physical_ids": [0, 1, 2, 3],
              "learner_ids": [2, 3], "state_dim": 80, "critic_num_agents": 4}


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path):
    config = json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))
    config["_config_path"] = str(resolve_path(path))
    return config


def portable_config(config):
    return {k: copy.deepcopy(v) for k, v in config.items() if not k.startswith("_")}


def config_digest(config):
    data = json.dumps(portable_config(config), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def task_contract_digest(config):
    """Environment/reward/lifecycle identity shared by train and evaluation.

    Device/output/monitor choices do not redefine the task. Every other task
    field is retained, including the candidate stall rule and reward weights.
    """
    task = {key: value for key, value in config["task"].items()
            if key not in {"follow_spectator", "cameras_enabled"}}
    contract = {"protocol": config["protocol"], "town": config["town"], "task": task,
                "action_spec": action_protocol_spec(), "red_policy_mode": "deterministic"}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_config(config, config_path=None):
    config = copy.deepcopy(config)
    if config.get("schema_version") != "mut_mappo_config_v1":
        raise ValueError("unsupported Mut-MAPPO configuration schema")
    if config.get("model_spec") != MODEL_SPEC or POLICY_ACTION_DIM != 2:
        raise ValueError("four-car 20/2 physical/learner/critic model specification mismatch")
    if config.get("method") not in {"off", "plain", "selective"}:
        raise ValueError("method must be off, plain or selective")
    config.setdefault("algorithm", "mappo")
    config.setdefault("team_count", 2)
    if config["algorithm"] not in {"mappo", "ippo"}:
        raise ValueError("implemented algorithms are mappo and ippo")
    if isinstance(config["team_count"], bool) or config["team_count"] not in (1, 2):
        raise ValueError("team_count must declare single(1) or dual(2) learners")
    if config["method"] != "off" and (config["algorithm"] != "mappo" or config["team_count"] != 2):
        raise ValueError("current mutual method requires dual MAPPO; IPPO is the independent baseline")
    from training.mutual_distillation import validate_distillation_options
    distillation = validate_distillation_options(config["learning"].get("distillation"))
    config["learning"]["distillation"] = distillation
    if config["method"] != "selective" and distillation["gate_variant"] != "evidence":
        raise ValueError("matched gate controls require selective mutual mode")
    if config["run"].get("schedule_clock", "rounds") not in {"rounds", "train_ticks"}:
        raise ValueError("unknown learning-rate schedule clock")
    if config["run"].get("stop_rule", "rounds") not in {"rounds", "train_ticks"}:
        raise ValueError("unknown run stopping rule")
    if config["run"].get("stop_rule") == "train_ticks" and config["run"].get("schedule_clock") != "train_ticks":
        raise ValueError("tick-budget stopping requires tick-based learning-rate schedule")
    if config["mutual"].get("schedule_clock", "rounds") not in {"rounds", "train_ticks"}:
        raise ValueError("unknown teacher refresh clock")
    if config["mutual"].get("schedule_clock") == "train_ticks" and any(
            isinstance(config["mutual"].get(k), bool) or not isinstance(config["mutual"].get(k), int)
            or config["mutual"][k] <= 0 for k in ("warmup_train_ticks", "refresh_train_ticks")):
        raise ValueError("tick-based teacher schedule needs positive integer warmup/refresh limits")
    if config.get("host") not in {"127.0.0.1", "localhost"} or config.get("port") != 2000:
        raise ValueError("this experiment shares local CARLA port 2000")
    if config.get("town") != "Town05" or config["task"].get("timestep") != 0.1:
        raise ValueError("first implementation freezes Town05 and 0.1 second timestep")
    from environment.mut_mappo_events import MutRewardConfig
    MutRewardConfig.from_dict(config["task"].get("reward"), timestep=config["task"]["timestep"])
    if config['task'].get('c1_controller','frozen_actor') not in {'frozen_actor','lane_pid_v1'}:
        raise ValueError('unknown frozen normal-traffic controller')
    profile = config["task"].get("reward_profile", "legacy")
    if profile not in TEAM_PROFILES | {"legacy"}:
        raise ValueError("unknown Mut reward profile")
    if profile in TEAM_PROFILES:
        settings = config["task"]["team_reward"]
        if not settings.get("reference_dir"):
            raise ValueError("C1 progress reward requires its prepared reference directory")
        if any(not math.isfinite(float(settings.get(k, default))) or float(settings.get(k, default)) < 0
               for k, default in (("attack_weight", 10), ("driving_weight", 1))):
            raise ValueError("team reward weights must be finite and nonnegative")
    if profile in CRITICAL_PROFILES:
        if profile == 'individual_red_outcome_v5':
            from environment.mut_mappo_outcome_reward_v5 import validate_v5_settings
            validate_v5_settings(config['task']['team_reward'], horizon=config['task']['max_steps'], gamma=config['learning']['gamma'])
        elif profile == 'individual_three_dimension_risk_v4':
            from environment.mut_mappo_critical_reward_v4 import validate_v4_settings
            validate_v4_settings(config['task']['team_reward'], horizon=config['task']['max_steps'], gamma=config['learning']['gamma'])
        elif profile == 'individual_driving_critical_v3':
            from environment.mut_mappo_critical_reward_v3 import validate_v3_settings
            validate_v3_settings(config['task']['team_reward'], horizon=config['task']['max_steps'], gamma=config['learning']['gamma'])
        elif profile == 'normal_driving_critical_team_v2':
            from environment.mut_mappo_critical_reward_v2 import validate_v2_settings
            validate_v2_settings(config['task']['team_reward'], horizon=config['task']['max_steps'],
                                 gamma=config['learning']['gamma'])
        else:
            from environment.mut_mappo_critical_reward import validate_settings
            validate_settings(config['task']['team_reward'])
        if not math.isclose(float(config['task']['team_reward'].get('gamma', .999)),
                            float(config['learning']['gamma'])):
            raise ValueError("critical reward and learner gamma differ")
    if config['mutual'].get('quality_metric') not in {None, 'efficiency', 'critical'}:
        raise ValueError("unknown mutual quality metric")
    if config['mutual'].get('quality_metric') == 'critical' and profile not in CRITICAL_PROFILES:
        raise ValueError("critical gate requires measured online critical geometry")
    if config.get('evaluation', {}).get('selection_metric') == 'safe_critical_event' and profile not in CRITICAL_PROFILES:
        raise ValueError("critical selection requires measured online critical geometry")
    stall = config["task"].get("stall", {})
    if not isinstance(stall.get("enabled"), bool) or any(
            not math.isfinite(float(stall.get(key, -1))) or float(stall.get(key, -1)) <= 0
            for key in ("seconds", "speed_mps")):
        raise ValueError("candidate stall rule requires explicit enabled, positive seconds and speed")
    if config.get("communication", {}).get("enabled"):
        raise ValueError("communication is planned; do not label the current 20/2 model as communication-enabled")
    for value in (config["run"]["episodes_per_team_update"], config["run"]["rounds"],
                  config["run"]["lr_horizon_rounds"], config["run"]["train_tick_budget"],
                  config["task"]["max_steps"]):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("episode, round and tick limits must be positive integers")
    if "lr_horizon_train_ticks" in config["run"]:
        horizon = config["run"]["lr_horizon_train_ticks"]
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < config["run"]["train_tick_budget"]:
            raise ValueError("LR tick horizon must be an integer covering the training budget")
    if config["run"]["rounds"] > config["run"]["lr_horizon_rounds"]:
        raise ValueError("requested rounds exceed the frozen LR horizon")
    if config["mutual"].get("warmup_rounds", 0) < 1 or config["mutual"].get("refresh_rounds", 0) < 1:
        raise ValueError("warmup and refresh boundaries must be positive")
    for key in ("gamma", "gae_lambda", "lr_actor", "lr_critic", "beta", "anchor_beta"):
        value = config["learning"][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid learning coefficient {key}")
    checkpoint, digest = load_checkpoint_file(resolve_path(config["red_checkpoint"]["path"]), map_location="cpu")
    if digest != config["red_checkpoint"]["sha256"]:
        raise ValueError("red checkpoint identity differs from the candidate configuration")
    source_spec = checkpoint.get("model_spec", {})
    if source_spec.get("obs_dim") != 20 or source_spec.get("action_dim") != 2:
        raise ValueError("red candidate is not a compatible 20/2 checkpoint")
    if (not isinstance(config["learning"].get("hidden_dim"), int)
            or config["learning"]["hidden_dim"] != source_spec.get("hidden_dim")):
        raise ValueError("pretrained initialization requires the frozen source Actor hidden width; do not silently reshape weights")
    for aid in (0, 1):
        if aid not in checkpoint["actors"]:
            raise ValueError(f"red source role {aid} missing")
    make_actors(checkpoint["actors"], checkpoint["training_config"], (0, 1), "cpu")
    manifest_path = resolve_path(config["case_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != "mut_mappo_cases_v1":
        raise ValueError("unknown four-car case manifest")
    cases = manifest.get("cases", [])
    lookup = {case["case_key"]: case for case in cases}
    if len(lookup) != len(cases):
        raise ValueError("duplicate case keys")
    memberships = []
    for split in ("train", "gate", "selection", "test"):
        keys = manifest.get("splits", {}).get(split, [])
        if not keys or len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            raise ValueError(f"invalid or empty {split} case split")
        memberships.extend(keys)
    if len(memberships) != len(set(memberships)):
        raise ValueError("train/gate/selection/test must have disjoint case keys")
    physical_configs = []
    source_hashes = {}
    for case in cases:
        if case.get("family") not in {"blocking", "cut_in", "parallel", "staggered"}:
            raise ValueError("unknown candidate interaction family")
        scenario_path = resolve_path(case["base_scenario"])
        scenario = json.loads(scenario_path.read_text(encoding="utf-8-sig"))
        source_hashes[case["base_scenario"]] = sha256_file(scenario_path)
        physical = {k: v for k, v in case.items() if k not in {"case_key", "seed", "split", "description"}}
        physical_configs.append(hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest())
        if scenario.get("town") != "Town05" or set(case.get("blue_spawn_offsets", {})) != {"agent2", "agent3"}:
            raise ValueError("case does not declare Town05 and both blue spawn roles")
    if len(physical_configs) != len(set(physical_configs)):
        raise ValueError("changing only keys or seeds does not create independent physical case splits")
    if config.get("research_status") == "frozen" and (
            config["red_checkpoint"].get("capability_status") != "independently_qualified"
            or manifest.get("geometry_status") != "live_validated"):
        raise ValueError("formal configuration requires independent red qualification and live geometry evidence")
    config["_red_payload"] = checkpoint
    config["_case_manifest"] = manifest
    config["_root_validation"] = {"status": "OFFLINE_COMPATIBLE_NOT_LIVE_VALIDATED",
                                   "config_sha256": config_digest(config), "red_sha256": digest,
                                   "task_contract_sha256": task_contract_digest(config),
                                   "case_manifest_sha256": sha256_file(manifest_path),
                                   "scenario_hashes": source_hashes,
                                   "cases": len(cases),
                                   "split_counts": {k: len(v) for k, v in manifest["splits"].items()}}
    return config


def make_actors(states, source_config, ids, device="cpu"):
    actors = {}
    for aid in ids:
        actor = Actor(20, 2, hidden=int(source_config.get("hidden_dim", source_config.get("hidden", 128))),
                      init_log_std=float(source_config.get("init_log_std", -0.5)),
                      min_log_std=float(source_config.get("min_log_std", -2)),
                      max_log_std=source_config.get("max_log_std", 0)).to(device)
        actor.load_state_dict(states[aid], strict=True)
        if any(not torch.isfinite(tensor).all() for tensor in actor.state_dict().values()):
            raise ValueError("nonfinite policy parameters")
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        actors[aid] = actor
    return actors


def cases_for(config, split):
    manifest = config["_case_manifest"]
    lookup = {case["case_key"]: case for case in manifest["cases"]}
    return [copy.deepcopy(lookup[key]) for key in manifest["splits"][split]]


def run_identity(config):
    paths = [ROOT / "training" / name for name in ("train_mut_mappo.py", "mut_mappo_config.py",
             "mut_mappo_learning.py", "mut_mappo_checkpoint.py", "train_no_rllib_mappo_robust_v2.py",
             "action_protocol.py", "value_normalization.py", "checkpoint_protocol.py", "mutual_distillation.py")]
    paths += [ROOT / "environment" / name for name in ("carla_multi_agent_env.py", "mut_mappo_env.py", "mut_mappo_events.py",
                                                       "base_reward_v4.py", "base_reward_v5.py", "base_reward_v6.py", "route_pace_reward.py")]
    paths += [ROOT / "models/actor_critic.py", ROOT / "models/mut_mappo_critics.py", ROOT / "evaluation/mut_mappo_metrics.py",
              ROOT / "evaluation/evaluate_mut_mappo.py", ROOT / "utils/mut_mappo_session.py",
              ROOT / "utils/adversarial_metrics.py", ROOT / "evaluation/mut_mappo_selection.py"]
    paths += [ROOT / "environment/mut_mappo_efficiency_reward.py", ROOT / "training/first_point_scenarios.py",
              ROOT / "evaluation/driving_policy_metrics.py"]
    if config['task'].get('reward_profile') in CRITICAL_PROFILES:
        paths += [ROOT / 'environment/mut_mappo_critical_reward.py',
                  ROOT / 'evaluation/critical_interaction_metrics.py']
    if config['task'].get('reward_profile') == 'normal_driving_critical_team_v2':
        paths.append(ROOT / 'environment/mut_mappo_critical_reward_v2.py')
    if config['task'].get('reward_profile') == 'individual_driving_critical_v3':
        paths += [ROOT / 'environment/mut_mappo_critical_reward_v3.py', ROOT / 'environment/mut_mappo_critical_reward_v2.py', ROOT / 'environment/base_reward_v6.py', ROOT / 'environment/base_reward_v5.py']
    if config['task'].get('reward_profile') == 'individual_three_dimension_risk_v4':
        paths += [ROOT / 'environment/mut_mappo_critical_reward_v4.py', ROOT / 'environment/mut_mappo_critical_reward_v3.py', ROOT / 'environment/mut_mappo_critical_reward_v2.py', ROOT / 'evaluation/r4_risk_metrics.py', ROOT / 'evaluation/mut_mappo_reference.py']
    if config['task'].get('reward_profile') == 'individual_red_outcome_v5':
        paths += [ROOT / p for p in ('environment/mut_mappo_outcome_reward_v5.py', 'evaluation/r5_outcome_metrics.py', 'environment/mut_mappo_critical_reward_v4.py', 'environment/mut_mappo_critical_reward_v3.py', 'environment/mut_mappo_critical_reward_v2.py', 'evaluation/r4_risk_metrics.py', 'utils/prepare_r5_reference.py', 'evaluation/evaluate_r5_counterfactual.py')]
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in paths}
    if config['task'].get('c1_controller') == 'lane_pid_v1':
        sources['environment/frozen_lane_traffic.py']=sha256_file(ROOT/'environment/frozen_lane_traffic.py')
        for dependency in ('navigation/controller.py','tools/misc.py'):
            sources['@carla/agents/'+dependency]=sha256_file(ROOT.parents[1]/'carla/agents'/dependency)
    identity = {"protocol": config["protocol"], "config_sha256": config_digest(config),
            "task_contract_sha256": task_contract_digest(config),
            "algorithm": config["algorithm"], "team_count": config["team_count"],
            "model_spec": MODEL_SPEC, "red_sha256": config["_root_validation"]["red_sha256"],
            "case_manifest_sha256": config["_root_validation"]["case_manifest_sha256"],
            "scenario_hashes": config["_root_validation"]["scenario_hashes"], "source_hashes": sources}
    if config["task"].get("reward_profile") in TEAM_PROFILES:
        directory = resolve_path(config["task"]["team_reward"]["reference_dir"])
        identity["training_reference_hashes"] = {
            case["case_key"]: sha256_file(directory / (hashlib.sha256(case["case_key"].encode()).hexdigest() + ".json"))
            for split in ("train", "gate") for case in cases_for(config, split)}
    return identity
