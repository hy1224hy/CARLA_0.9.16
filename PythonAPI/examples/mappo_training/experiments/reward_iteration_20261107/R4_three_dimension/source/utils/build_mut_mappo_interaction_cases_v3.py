"""Build a small mirrored sequential/crossing pool, without a CARLA server.

Run as a module from mappo_training. Existing scenes/configs are never edited.
Crossing means a same-direction diagonal formation, not an intersection route
or a scripted lane change. The existing runtime checks legal adjacent lanes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from utils.build_mut_mappo_symmetric_cases_v2 import (
    DEFAULT_XODR, approximate_initial_states, serialized, write_payloads,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/mut_mappo/interaction_small_v3"
DEFAULT_EXPERIMENT = ROOT / "experiments/adversarial_u1000_interactions_20261029_v2"
SOURCE_CONFIG = ROOT / "experiments/adversarial_u1000_pilot_20261028_v1/base_config.json"
DIRECTIONS = ("northbound", "southbound", "eastbound", "westbound")
GAPS = {"train": 10.0, "gate": 11.0, "selection": 12.0, "test": 13.0}
RED_SPEED, FRONT_SPEED, REAR_SPEED = 2.5, 2.5, 3.5
START_ADVANCE, ROUTE_LENGTH = 8.0, 32.0
RED_SHA = "0ac714a5172621cb1b716604bd69dd3a32dfaa57d5e44678eae7815195d6981a"


def _walk(wp, distance):
    options = wp.next(distance) if distance > 0 else wp.previous(-distance)
    options = [w for w in options if w.road_id == wp.road_id and w.lane_id == wp.lane_id and not w.is_junction]
    if len(options) != 1:
        raise ValueError("scene geometry leaves its unique non-junction Driving lane")
    return options[0]


def _point(wp):
    p = wp.transform
    return {"x": float(p.location.x), "y": float(p.location.y), "z": 0.5,
            "yaw": float(p.rotation.yaw)}


def _bases(output, road_map):
    import carla
    bases = {}
    prefix = output.relative_to(ROOT).as_posix()
    for direction in DIRECTIONS:
        ref = f"data/training_scenarios_town05_base_vnext_v1/scenario_bd1_{direction}_rolling_z00_a0l.json"
        path = ROOT / ref
        original = json.loads(path.read_text(encoding="utf-8"))
        starts, goals = {}, {}
        for rid in (0, 1):
            a = original["agents"][f"agent{rid}"]
            wp = road_map.get_waypoint(carla.Location(x=a["x"], y=a["y"], z=a["z"]),
                                      project_to_road=False, lane_type=carla.LaneType.Driving)
            if wp is None or wp.is_junction:
                raise ValueError("source lacks a non-junction Driving lane")
            start = _walk(wp, START_ADVANCE)
            starts[rid] = dict(_point(start), initial_speed_mps=RED_SPEED)
            goals[rid] = dict(_point(_walk(start, ROUTE_LENGTH)), half_length=3.0, half_width=1.6)
        for red0_left in (True, False):
            base = copy.deepcopy(original)
            order = (0, 1) if red0_left else (1, 0)
            base["agents"] = {f"agent{i}": copy.deepcopy(starts[j]) for i, j in enumerate(order)}
            base["agent_goal_zones"] = {f"agent{i}": copy.deepcopy(goals[j]) for i, j in enumerate(order)}
            base["agent_roles"] = {f"agent{i}": ("left_parallel" if j == 0 else "right_parallel") for i, j in enumerate(order)}
            base["goal_zone"] = dict(base["goal_zone"],
                x=sum(g["x"] for g in goals.values())/2, y=sum(g["y"] for g in goals.values())/2)
            base.update(episode_max_steps=500, split="shared_interaction_base", route_length_m=ROUTE_LENGTH,
                description="small interaction v3: road-projected parallel red pair, complete slot exchange",
                validation_status="OFFLINE_GEOMETRY_ONLY_LIVE_INITIAL_STATE_PENDING",
                geometry_provenance={"source": ref, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "method": "local_OpenDRIVE_waypoint_projection", "start_advance_m": START_ADVANCE,
                    "red_route_m": ROUTE_LENGTH, "carla_client_created": False})
            key = f"{prefix}/bases/{direction}_red0_{'left' if red0_left else 'right'}.json"
            bases[key] = base
    return bases


def validate_manifest(manifest, bases):
    cases = manifest["cases"]
    lookup = {c["case_key"]: c for c in cases}
    keys = [k for group in manifest["splits"].values() for k in group]
    if len(lookup) != len(cases) or len(keys) != len(set(keys)) or set(keys) != set(lookup):
        raise ValueError("duplicate/missing case or split membership")
    # Compare physical states first: keys, seeds and descriptive metadata cannot
    # manufacture independent geometry. Identity-preserving role pairs remain.
    signatures = set()
    for c in cases:
        agents = approximate_initial_states(c, bases[c["base_scenario"]])
        signature = tuple(tuple(round(a[k], 5) for k in ("x", "y", "yaw", "initial_speed_mps")) for a in agents.values())
        if signature in signatures:
            raise ValueError("duplicate physical initial state across cases/splits")
        signatures.add(signature)
    orbit = {("sequential", "both", side) for side in (False, True)} | {
             ("crossing", front, side) for front in ("left", "right") for side in (False, True)}
    for split, membership in manifest["splits"].items():
        rows = [lookup[k] for k in membership]
        for direction in DIRECTIONS:
            own = [c for c in rows if c["direction"] == direction]
            if len(own) != 6 or {(c["layout"], c["front_side"], c["red0_left"]) for c in own} != orbit:
                raise ValueError("incomplete left/right and joint-slot symmetry orbit")
        for c in rows:
            if c["split"] != split or c["family"] != ("blocking" if c["layout"] == "sequential" else "cut_in"):
                raise ValueError("scene family/split differs from physical layout")
            for bid in (2, 3):
                item = c["blue_spawn_offsets"][f"agent{bid}"]
                left = c["red0_left"] if bid == 2 else not c["red0_left"]
                front = c["layout"] == "sequential" or left == (c["front_side"] == "left")
                expected = GAPS[split] * (1 if front else -1)
                if item != {"anchor_red_id": bid-2, "forward_m": expected,
                            "initial_speed_mps": FRONT_SPEED if front else REAR_SPEED}:
                    raise ValueError("scene geometry/velocity differs from its mirrored layout")
                if not front and (abs(expected) - (REAR_SPEED-RED_SPEED)*3 < 6 or
                        (abs(expected)-6)/(REAR_SPEED-RED_SPEED) >= (ROUTE_LENGTH-3)/RED_SPEED):
                    raise ValueError("nominal rear protection/interaction window invalid")
        pairs = defaultdict(list)
        for c in rows:
            pairs[c["pair_id"]].append(c)
        if any(len(p) != 2 or {c["red0_left"] for c in p} != {False, True} for p in pairs.values()):
            raise ValueError("incomplete joint actor-slot pair")


def build_payloads(output_dir=DEFAULT_OUTPUT, xodr=DEFAULT_XODR):
    import carla
    output = Path(output_dir).resolve()
    road_map = carla.Map("Town05", Path(xodr).read_text(encoding="utf-8"))
    bases = _bases(output, road_map)
    prefix = output.relative_to(ROOT).as_posix()
    cases, splits = [], {s: [] for s in GAPS}
    for split, gap in GAPS.items():
        for layout, fronts in (("sequential", ("both",)), ("crossing", ("left", "right"))):
            for direction, front_side, red0_left in itertools.product(DIRECTIONS, fronts, (True, False)):
                offsets = {}
                for bid in (2, 3):
                    left = red0_left if bid == 2 else not red0_left
                    front = layout == "sequential" or left == (front_side == "left")
                    offsets[f"agent{bid}"] = {"anchor_red_id": bid-2, "forward_m": gap if front else -gap,
                                              "initial_speed_mps": FRONT_SPEED if front else REAR_SPEED}
                key = f"{split}_{layout}_{direction}_front_{front_side}_r0{'l' if red0_left else 'r'}"
                case = {"case_key": key, "split": split, "family": "blocking" if layout == "sequential" else "cut_in",
                    "layout": layout, "front_side": front_side, "direction": direction, "red0_left": red0_left,
                    "startup_mode": "rolling", "red0_role": "left_parallel" if red0_left else "right_parallel",
                    "pair_id": f"{split}_{layout}_{direction}_front_{front_side}", "seed": 75000+len(cases),
                    "episode_max_steps": 500, "red_overrides": {}, "blue_spawn_offsets": offsets,
                    "base_scenario": f"{prefix}/bases/{direction}_red0_{'left' if red0_left else 'right'}.json",
                    "blue_goal_mode": "anchor_red_goal", "geometry_status": "candidate_not_live_validated"}
                cases.append(case)
                splits[split].append(key)
    manifest = {"schema_version": "mut_mappo_cases_v1", "case_set_version": "interaction_small_v3",
        "sampling_protocol": "first_point_complete_pairs_v1", "geometry_status": "candidate_not_live_validated",
        "scope": "same-direction two-lane sequential and diagonal crossing; four Town05 corridors",
        "symmetry": "complete lateral mirror and simultaneous (red0/red1, blue2/blue3) exchange; not independent color permutations",
        "split_rule": "one held-out signed-gap profile per split on shared corridors; no cross-map claim",
        "layout_semantics": "sequential=both blues ahead; crossing=one front and one side-rear; no scripted lane change",
        "preflight_subsets": {"smoke": [c["case_key"] for c in cases if c["split"] == "train"
                               and c["direction"] in ("northbound", "eastbound")]},
        "cases": cases, "splits": splits}
    validate_manifest(manifest, bases)
    coverage = {"case_set_version": manifest["case_set_version"], "train_templates": 24,
        "physical_layouts_per_direction": ["both_front", "left_front_right_rear", "left_rear_right_front"],
        "splits": {s: {"n": len(keys), "families": dict(Counter(c["family"] for c in cases if c["split"] == s)),
                       "signed_gap_m": gap} for s, (keys, gap) in ((s, (splits[s], GAPS[s])) for s in GAPS)},
        "red_initial_mps": RED_SPEED, "blue_front_initial_mps": FRONT_SPEED, "blue_rear_initial_mps": REAR_SPEED,
        "source_start_advance_m": START_ADVANCE, "red_route_m": ROUTE_LENGTH,
        "nominal_3s_min_same_lane_center_gap_m": min(GAPS.values())-(REAR_SPEED-RED_SPEED)*3,
        "nominal_rear_time_to_6m_gap_s": {s: (g-6)/(REAR_SPEED-RED_SPEED) for s, g in GAPS.items()},
        "smoke_templates": len(manifest["preflight_subsets"]["smoke"]),
        "sampling": "equal family exposure; 8 updates x 4 episodes cover all 24 train templates; sequential pairs repeat",
        "status": "OFFLINE_CANDIDATE_LIVE_C0_C1_REQUIRED",
        "limits": ["constant-speed windows are geometric screening, not live safety/reachability guarantees",
                   "red u1000 and ordinary blue are not driving-qualified; actual red may outrun rear blue",
                   "blue shares same-lane red goal and may retire earlier; actual interaction exposure must be measured",
                   "directions use real road geometry, not exact global-coordinate rotations; paired cases are dependent"]}
    files = {Path(ref).relative_to(prefix).as_posix(): value for ref, value in bases.items()}
    files.update({"cases.json": manifest, "coverage.json": coverage})
    return files


def validate_opendrive(manifest, bases, xodr=DEFAULT_XODR):
    import carla
    from environment.mut_mappo_env import MutMAPPOEnv
    env = MutMAPPOEnv.__new__(MutMAPPOEnv)
    env.map = carla.Map("Town05", Path(xodr).read_text(encoding="utf-8"))
    env.red_ids, env.blue_ids, env.num_agents, env.default_max_steps = (0, 1), (2, 3), 4, 500
    signatures, records = set(), []
    for original in manifest["cases"]:
        case = copy.deepcopy(original)
        case["base_scenario"] = bases[case["base_scenario"]]
        scenario, evidence = env._resolve_case(case)
        signature = tuple(tuple(round(a[k], 5) for k in ("x", "y", "yaw", "initial_speed_mps")) for a in scenario["agents"].values())
        if signature in signatures:
            raise ValueError("duplicate production-resolved physical state")
        signatures.add(signature)
        for key, a in scenario["agents"].items():
            start = env._get_exact_driving_waypoint(carla.Location(x=a["x"], y=a["y"], z=a["z"]))
            target = scenario["agent_goal_zones"][key]
            goal = env._get_exact_driving_waypoint(carla.Location(x=target["x"], y=target["y"], z=target["z"]))
            if goal is None or goal.is_junction or (goal.road_id, goal.lane_id) != (start.road_id, start.lane_id):
                raise ValueError("goal leaves initial Driving lane")
            distance = abs(goal.s-start.s)
            for step in [*range(2, int(distance), 2), distance]:
                _walk(start, step)
            endpoint = _walk(start, distance).transform.location
            if math.hypot(endpoint.x-target["x"], endpoint.y-target["y"]) > .1:
                raise ValueError("goal is not forward-reachable on initial lane")
        records.append({"case_key": original["case_key"], "waypoints": evidence["waypoints"],
                        "road_id": evidence["waypoints"]["agent0"]["road_id"]})
    return {"status": "OFFLINE_OPENDRIVE_PASS_LIVE_SPAWN_PENDING", "cases_checked": len(signatures),
        "crossing_cases_with_legal_adjacent_lane": sum(c["layout"] == "crossing" for c in manifest["cases"]),
        "carla_client_created": False, "world_ticks": 0,
        "xodr_sha256": hashlib.sha256(Path(xodr).read_bytes()).hexdigest(),
        "case_manifest_sha256": hashlib.sha256(serialized(manifest).encode()).hexdigest(),
        "base_scenario_sha256": {k: hashlib.sha256(serialized(v).encode()).hexdigest() for k, v in bases.items()},
        "checks": ["production_resolver", "legal_adjacent_markings", "initial_footprint_separation",
                   "all_four_forward_goals", "non_junction_path_every_2m", "distinct_physical_split_states"], "cases": records}


def candidate_config(source, output_dir=DEFAULT_OUTPUT, experiment_dir=DEFAULT_EXPERIMENT):
    cfg = copy.deepcopy(source)
    cfg["case_manifest"] = (Path(output_dir).resolve() / "cases.json").relative_to(ROOT).as_posix()
    cfg["research_status"] = "candidate"
    cfg["task"]["team_reward"]["reference_dir"] = (Path(experiment_dir).resolve()/"references").relative_to(ROOT).as_posix()
    if cfg["red_checkpoint"]["sha256"] != RED_SHA:
        raise ValueError("this scene pilot is registered to the reviewed u1000 identity")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--base-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--validate-map", action="store_true")
    parser.add_argument("--prepare-experiment", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output, experiment = args.output_dir.resolve(), args.experiment_dir.resolve()
    with patch("carla.Client", side_effect=AssertionError("scene preparation must remain offline")):
        files = build_payloads(output)
        prefix = output.relative_to(ROOT).as_posix()
        bases = {f"{prefix}/{k}": v for k, v in files.items() if k.startswith("bases/")}
        report = validate_opendrive(files["cases.json"], bases) if args.validate_map else None
        write_payloads(files, output, check=args.check)
        if report:
            path, content = output/"offline_map_validation.json", serialized(report)
            if args.check:
                if path.read_text(encoding="utf-8") != content:
                    raise ValueError("offline map evidence drift")
            else:
                path.write_text(content, encoding="utf-8", newline="\n")
        if args.prepare_experiment:
            from training.mut_mappo_config import validate_config
            cfg = candidate_config(json.loads(args.base_config.read_text(encoding="utf-8-sig")), output, experiment)
            checked = validate_config(cfg)["_root_validation"]
            entry = {"run_id": "B0_s42", "group": "B0", "seed": 42,
                     "config": str(experiment/"configs/B0_s42.json"), "run_dir": str(experiment/"runs/B0_s42")}
            payload = {"schema": "first_point_experiment_v1", "suite": "B0_pilot_only", "scene_stage": "sequential_crossing_v3",
                "base_config": str(experiment/"base_config.json"), "entries": [entry],
                "train_tick_budget": cfg["run"]["train_tick_budget"], "status": "PREPARED_NOT_RUN",
                "split_counts": checked["split_counts"], "reference_repeats_planned": 2,
                "authorization": {"live_execution": False, "weak_red_override": False},
                "qualification": "red NOT_QUALIFIED; offline geometry is not live qualification",
                "execution": "serial CARLA only; training reward unchanged; new case/reference identity"}
            for relative, value in {"base_config.json": cfg, "configs/B0_s42.json": cfg,
                                     "experiment.json": payload, "offline_compatibility.json": checked}.items():
                path, content = experiment/relative, serialized(value)
                if path.exists() and path.read_text(encoding="utf-8") != content:
                    raise FileExistsError("candidate evidence differs; choose a new experiment version")
                if not args.check:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8", newline="\n")
                elif not path.exists():
                    raise FileNotFoundError(path)
    print(json.dumps({"status": "PREPARED_NOT_RUN", "case_manifest": str(output/"cases.json"),
        "split_counts": {s: len(k) for s, k in files["cases.json"]["splits"].items()},
        "map_checked_cases": report["cases_checked"] if report else None,
        "physical_world_ticks": 0, "training_launched": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
