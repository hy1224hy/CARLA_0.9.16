"""Generate a separate, balanced four-car candidate pool; never tick CARLA.

Run from mappo_training with --validate-map to check the actual Mut reset
resolver against the local OpenDRIVE map, without creating a CARLA Client.
The original case list, training defaults, rewards and checkpoints are untouched.
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/mut_mappo/symmetric_v2"
def default_xodr():
    import os
    suffix = "CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr"
    candidates = ([Path(os.environ["CARLA_ROOT"]) / suffix] if os.environ.get("CARLA_ROOT") else [])
    candidates += [parent / suffix for parent in ROOT.parents]
    candidates.append(Path("/opt/carla/CARLA_0.9.16") / suffix)
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_XODR = default_xodr()
DIRECTIONS = ("north", "south", "east", "west")
FAMILIES = ("blocking", "cut_in")
SOURCES = {
    "north": "scenario_05_north_parallel_rolling_a1_left.json",
    "south": "scenario_06_south_parallel_rolling_a0_left.json",
    "east": "scenario_07_east_parallel_rolling_a0_left.json",
    "west": "scenario_08_west_parallel_rolling_a1_left.json",
}
# Each whole parameter orbit stays in exactly one split. Speeds travel with
# near/far positions when blue identities are exchanged, preserving symmetry.
# Fields: parameter ID, red speed, near gap, far gap, near speed, far speed.
PROFILES = {
    "train": (("t0", 3.0, 10.0, 16.0, 2.4, 2.55),
              ("t1", 3.3, 12.0, 19.0, 2.5, 2.65),
              ("t2", 3.6, 14.0, 22.0, 2.7, 2.85)),
    "gate": (("g0", 3.1, 11.0, 17.5, 2.45, 2.6),),
    "selection": (("s0", 3.4, 13.0, 20.5, 2.6, 2.75),),
    "test": (("e0", 3.7, 15.0, 23.0, 2.8, 2.95),),
}


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _left_position(agent, center):
    yaw = math.radians(float(agent["yaw"]))
    # CARLA left-handed world: physical left=(sin(yaw), -cos(yaw)).
    return ((agent["x"] - center[0]) * math.sin(yaw)
            - (agent["y"] - center[1]) * math.cos(yaw))


def base_scenario(direction, red0_left):
    source_ref = "data/training_scenarios_town05_v1/" + SOURCES[direction]
    scenario = json.loads((ROOT / source_ref).read_text(encoding="utf-8"))
    agents = list(scenario["agents"].values())
    center = tuple(sum(a[axis] for a in agents) / 2 for axis in ("x", "y"))
    left, right = sorted(agents, key=lambda a: _left_position(a, center), reverse=True)
    scenario["agents"] = copy.deepcopy({"agent0": left if red0_left else right,
                                         "agent1": right if red0_left else left})
    scenario["tags"] = [tag for tag in scenario["tags"]
                        if tag not in {"agent0_left", "agent1_left"}]
    scenario["tags"].append("agent0_left" if red0_left else "agent1_left")
    scenario.update(description=f"Mut symmetric v2 {direction}, physical red0 "
                    + ("left" if red0_left else "right"), source_scenario=source_ref)
    return scenario


def approximate_initial_states(case, base):
    """Straight-corridor arithmetic, not a substitute for map/spawn checks."""
    agents = copy.deepcopy(base["agents"])
    for key, override in case["red_overrides"].items():
        agents[key].update(override)
    for key, spec in case["blue_spawn_offsets"].items():
        anchor = agents[f"agent{spec['anchor_red_id']}"]
        yaw = math.radians(anchor["yaw"])
        agents[key] = dict(anchor, x=anchor["x"] + spec["forward_m"] * math.cos(yaw),
                           y=anchor["y"] + spec["forward_m"] * math.sin(yaw),
                           initial_speed_mps=spec["initial_speed_mps"])
    return agents


def validate_manifest(manifest, bases):
    lookup = {case["case_key"]: case for case in manifest["cases"]}
    if len(lookup) != len(manifest["cases"]):
        raise ValueError("duplicate case keys")
    all_keys, physical_states, orbit_splits = set(), set(), defaultdict(set)
    expected_axes = set(itertools.product(FAMILIES, DIRECTIONS, (False, True), (False, True)))
    for split, keys in manifest["splits"].items():
        if len(keys) != len(set(keys)) or all_keys.intersection(keys):
            raise ValueError("duplicate or overlapping split keys")
        all_keys.update(keys)
        orbits = defaultdict(list)
        for key in keys:
            case = lookup[key]
            orbit_splits[case["parameter_id"]].add(split)
            orbits[case["parameter_id"]].append(case)
            base = bases[case["base_scenario"]]
            agents = approximate_initial_states(case, base)
            # Deliberately ignore case names, family, seeds and all metadata.
            signature = tuple(tuple(round(float(a[field]), 6) for field in
                                    ("x", "y", "yaw", "initial_speed_mps"))
                              for a in agents.values())
            if signature in physical_states:
                raise ValueError("duplicate labeled four-car physical initial state")
            physical_states.add(signature)
            center = tuple(sum(a[axis] for a in base["agents"].values()) / 2 for axis in ("x", "y"))
            if (_left_position(base["agents"]["agent0"], center) > 0) != case["red0_left"]:
                raise ValueError("physical left/right role mislabeled")
            for spec in case["blue_spawn_offsets"].values():
                anchor = agents[f"agent{spec['anchor_red_id']}"]
                # Conservative constant-speed separation during 3s protection.
                gap = spec["forward_m"] - max(0, anchor["initial_speed_mps"]
                                              - spec["initial_speed_mps"]) * 3.0
                yaw = math.radians(anchor["yaw"])
                remaining = ((base["goal_zone"]["x"] - anchor["x"]) * math.cos(yaw)
                             + (base["goal_zone"]["y"] - anchor["y"]) * math.sin(yaw)
                             - spec["forward_m"])
                if gap < 5.0 or remaining <= 6.0:
                    raise ValueError("candidate lacks initial separation or route remaining")
        for cases in orbits.values():
            axes = {(c["family"], c["direction"], c["red0_left"], c["blue2_near"]) for c in cases}
            if axes != expected_axes or len(cases) != len(expected_axes):
                raise ValueError("incomplete symmetry orbit")
        # Existing runner takes consecutive groups of four. Balance each one.
        for start in range(0, len(keys), 4):
            batch = [lookup[key] for key in keys[start:start + 4]]
            if ({c["direction"] for c in batch} != set(DIRECTIONS)
                    or Counter(c["family"] for c in batch) != dict.fromkeys(FAMILIES, 2)
                    or sum(c["red0_left"] for c in batch) != 2
                    or sum(c["blue2_near"] for c in batch) != 2):
                raise ValueError("four-episode sampling block is unbalanced")
    if all_keys != set(lookup) or any(len(splits) != 1 for splits in orbit_splits.values()):
        raise ValueError("missing split membership or symmetry-orbit leakage")


def build_payloads(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir).resolve()
    relative_root = output_dir.relative_to(ROOT).as_posix()
    bases = {f"{relative_root}/bases/{direction}_red0_{'left' if left else 'right'}.json":
             base_scenario(direction, left) for direction in DIRECTIONS for left in (False, True)}
    cases, splits = [], {split: [] for split in PROFILES}
    for split, profiles in PROFILES.items():
        for parameter_id, red_speed, near, far, near_speed, far_speed in profiles:
            for family_phase, red_phase, blue_phase in itertools.product(range(2), repeat=3):
                for d, direction in enumerate(DIRECTIONS):
                    family = FAMILIES[(d + family_phase) % 2]
                    red0_left = bool((d // 2 + red_phase) % 2)
                    blue2_near = bool((d % 2 + blue_phase) % 2)
                    key = (f"{split}_{parameter_id}_{family}_{direction}_"
                           f"r0{'l' if red0_left else 'r'}_b2{'near' if blue2_near else 'far'}")
                    offsets = {}
                    for bid in (2, 3):
                        is_near = blue2_near if bid == 2 else not blue2_near
                        anchor = bid - 2 if family == "blocking" else 3 - bid
                        offsets[f"agent{bid}"] = {"anchor_red_id": anchor,
                            "forward_m": near if is_near else far,
                            "initial_speed_mps": near_speed if is_near else far_speed}
                    case = {"case_key": key, "family": family, "direction": direction,
                            "seed": 50000 + len(cases), "episode_max_steps": 500,
                            "base_scenario": f"{relative_root}/bases/{direction}_red0_{'left' if red0_left else 'right'}.json",
                            "red_overrides": {f"agent{rid}": {"initial_speed_mps": red_speed} for rid in (0, 1)},
                            "blue_spawn_offsets": offsets, "red0_left": red0_left,
                            "blue2_near": blue2_near, "parameter_id": parameter_id,
                            "geometry_status": "candidate_not_live_validated"}
                    cases.append(case)
                    splits[split].append(key)
    manifest = {"schema_version": "mut_mappo_cases_v1", "case_set_version": "symmetric_v2",
                "geometry_status": "candidate_not_live_validated",
                "scope": "two_interaction_families_four_corridors_complete_role_and_gap_symmetries",
                "generation": "deterministic_physical_parameter_orbits_not_seed_only_variants",
                "geometry_protocol": "existing MutMAPPOEnv resolver and runtime spawn checks remain mandatory",
                "split_rule": "whole 32-case parameter orbits; same-corridor parameter holdout only",
                "sampling": "existing sequential runner; each four-case block balances directions/families/red-side/blue-gap-order",
                "family_semantics": "blocking uses own-role lane anchors; cut_in exchanges anchors and requires legal adjacent lanes; policy actions are not scripted",
                "cases": cases, "splits": splits}
    validate_manifest(manifest, bases)
    coverage = {"case_set_version": "symmetric_v2", "n_cases": len(cases), "n_bases": len(bases),
                "axes_per_parameter_orbit": {"family": 2, "direction": 4, "red0_left": 2, "blue2_near": 2},
                "parameter_profiles": {split: [dict(zip(("id", "red_speed_mps", "near_gap_m", "far_gap_m",
                    "near_blue_speed_mps", "far_blue_speed_mps"), row)) for row in profiles] for split, profiles in PROFILES.items()},
                "splits": {split: {"n": len(keys), "families": dict(Counter(c["family"] for c in cases if c["case_key"] in keys))}
                           for split, keys in splits.items()},
                "training_rounds_for_one_complete_pass_at_four_episodes": len(splits["train"]) // 4,
                "coordinate_convention": "CARLA physical left=(sin(yaw),-cos(yaw)); source filename role tags are not trusted",
                "validation_status": "OFFLINE_INVARIANTS_PASS_LIVE_SPAWN_PENDING",
                "limits": ["four real corridors have different route lengths, not exact world-coordinate rotations",
                           "mirrored cases are correlated evidence, not independent training seeds",
                           "no new map, traffic density, weather or compulsory merge task",
                           "family labels do not guarantee that a learned policy actually changes lanes"]}
    files = {Path(ref).relative_to(relative_root).as_posix(): base for ref, base in bases.items()}
    files.update({"cases.json": manifest, "coverage.json": coverage})
    return files


def validate_opendrive(manifest, bases, xodr=DEFAULT_XODR):
    """Exercise production waypoint/marking/footprint checks without a server."""
    import carla
    from environment.mut_mappo_env import MutMAPPOEnv

    xodr = Path(xodr)
    env = MutMAPPOEnv.__new__(MutMAPPOEnv)
    env.map = carla.Map("Town05", xodr.read_text(encoding="utf-8"))
    env.red_ids, env.blue_ids, env.num_agents, env.default_max_steps = (0, 1), (2, 3), 4, 500
    signatures, roads = set(), Counter()
    for original in manifest["cases"]:
        case = copy.deepcopy(original)
        case["base_scenario"] = bases[case["base_scenario"]]
        scenario, evidence = env._resolve_case(case)
        signature = tuple(tuple(round(float(a[f]), 5) for f in
                                ("x", "y", "yaw", "initial_speed_mps")) for a in scenario["agents"].values())
        if signature in signatures:
            raise ValueError("duplicate map-resolved labeled initial state")
        signatures.add(signature)
        # Check each same-lane terminal plane is reachable in forward topology.
        goal = scenario["goal_zone"]
        for key, agent in scenario["agents"].items():
            start = env._get_exact_driving_waypoint(carla.Location(x=agent["x"], y=agent["y"], z=agent["z"]))
            yaw = math.radians(agent["yaw"])
            distance = ((goal["x"] - agent["x"]) * math.cos(yaw)
                        + (goal["y"] - agent["y"]) * math.sin(yaw))
            endpoints = [wp for wp in start.next(distance) if wp.road_id == start.road_id
                         and wp.lane_id == start.lane_id and not wp.is_junction]
            if distance <= 6 or len(endpoints) != 1:
                raise ValueError(f"{original['case_key']}/{key}: missing forward same-lane goal")
        roads[evidence["waypoints"]["agent0"]["road_id"]] += 1
    return {"status": "OFFLINE_OPENDRIVE_PASS_LIVE_SPAWN_PENDING", "cases_checked": len(signatures),
            "road_counts": dict(roads), "xodr_sha256": hashlib.sha256(xodr.read_bytes()).hexdigest(),
            "case_manifest_sha256": hashlib.sha256(serialized(manifest).encode("utf-8")).hexdigest(),
            "base_scenario_sha256": {ref: hashlib.sha256(serialized(base).encode("utf-8")).hexdigest()
                                     for ref, base in sorted(bases.items())},
            "carla_client_created": False, "world_ticks": 0,
            "checks": ["production_case_resolver", "legal_cut_in_markings", "initial_footprint_separation",
                       "distinct_physical_initial_states", "same_lane_forward_goal_reachability"]}


def write_payloads(files, output_dir=DEFAULT_OUTPUT, *, check=False):
    output_dir = Path(output_dir)
    expected = set(files)
    existing = {p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*.json")}
    if existing - expected - {"offline_map_validation.json"}:
        raise ValueError("unexpected generated files; refuse to overwrite")
    if check and expected != existing - {"offline_map_validation.json"}:
        raise ValueError("generated case file set drift")
    for name, payload in files.items():
        path, content = output_dir / name, serialized(payload)
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"generated case drift: {path}")
        elif path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"changed candidate exists; use a new versioned directory: {path}")
    if not check:
        for name, payload in files.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized(payload), encoding="utf-8", newline="\n")


def configuration_for_manifest(source, manifest_path):
    result = copy.deepcopy(source)
    result["case_manifest"] = Path(manifest_path).resolve().relative_to(ROOT).as_posix()
    result["research_status"] = "candidate"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate-map", action="store_true")
    parser.add_argument("--xodr", type=Path, default=DEFAULT_XODR)
    parser.add_argument("--base-config", type=Path, help="optional source configuration; its red policy and training settings are preserved")
    parser.add_argument("--config-output", type=Path, help="new separate candidate configuration; required with --base-config")
    args = parser.parse_args()
    if bool(args.base_config) != bool(args.config_output):
        parser.error("--base-config and --config-output must be supplied together")
    output = args.output_dir.resolve()
    files = build_payloads(output)
    report = None
    if args.validate_map:
        prefix = output.relative_to(ROOT).as_posix()
        bases = {f"{prefix}/{name}": value for name, value in files.items() if name.startswith("bases/")}
        report = validate_opendrive(files["cases.json"], bases, args.xodr)
    write_payloads(files, output, check=args.check)
    if report and not args.check:
        (output / "offline_map_validation.json").write_text(serialized(report), encoding="utf-8", newline="\n")
    if args.base_config:
        source = json.loads(args.base_config.read_text(encoding="utf-8-sig"))
        content = serialized(configuration_for_manifest(source, output / "cases.json"))
        destination = args.config_output
        if args.check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
                raise ValueError(f"generated configuration drift: {destination}")
        else:
            if destination.exists() and destination.read_text(encoding="utf-8") != content:
                raise FileExistsError("configuration exists; preserve it and use a new path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "case_manifest": str(output / "cases.json"),
        "split_counts": {s: len(k) for s, k in files["cases.json"]["splits"].items()},
        "map_validation": report, "candidate_config": str(args.config_output) if args.config_output else None,
        "training_launched": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
