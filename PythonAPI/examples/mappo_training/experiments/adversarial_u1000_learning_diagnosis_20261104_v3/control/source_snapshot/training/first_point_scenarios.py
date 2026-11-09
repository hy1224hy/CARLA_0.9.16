"""Small complete-pair scene pool and sampler for the first research point."""
from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "first_point_complete_pairs_v1"
DIRECTIONS = ("northbound", "southbound", "eastbound", "westbound")


def build_cases(parallel_only=False):
    train_dir = ROOT / "data/training_scenarios_town05_base_vnext_v1"
    val_dir = ROOT / "data/training_scenarios_town05_base_vnext_v1_validation"
    train = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(train_dir.glob("*.json"))]
    val = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(val_dir.glob("*.json"))]
    train = [(p, s) for p, s in train if s["startup_mode"] == "rolling"]
    parallel = [(p, s) for p, s in train if s["delta_s_m"] == 0]
    # Keep whole actor-swap pairs. Alternate the physical leading lane across directions.
    stagger = [(p, s) for p, s in val if s["startup_mode"] == "rolling"
               and s["delta_s_m"] * (1 if "a0l" in p.stem else -1)
               == (4 if DIRECTIONS.index(s["direction"]) % 2 == 0 else -4)]
    cases, splits = [], {}
    for split, pool, gap in (("train", train, 10.0), ("gate", parallel + stagger, 11.0),
                             ("selection", parallel + stagger, 12.0), ("test", parallel + stagger, 13.0)):
        splits[split] = []
        if split == "train" and parallel_only:
            pool = parallel
        for path, scenario in pool:
            key = split + "_" + path.stem.removeprefix("scenario_bd1_")
            family = "parallel" if scenario["delta_s_m"] == 0 else "staggered"
            case = {"case_key": key, "family": family, "direction": scenario["direction"],
                "startup_mode": "rolling", "red0_role": scenario["agent_roles"]["agent0"],
                "seed": 64000 + len(cases), "episode_max_steps": 500,
                "base_scenario": path.relative_to(ROOT).as_posix(), "red_overrides": {},
                "blue_spawn_offsets": {f"agent{b}": {"anchor_red_id": b - 2,
                    "forward_m": gap, "initial_speed_mps": 2.5} for b in (2, 3)},
                "blue_goal_mode": "anchor_red_goal", "pair_id": split + "_" + scenario["pair_id"],
                "geometry_status": "candidate_not_live_validated"}
            cases.append(case)
            splits[split].append(key)
    return {"schema_version": "mut_mappo_cases_v1", "case_set_version": "first_point_v1",
        "geometry_status": "candidate_not_live_validated", "sampling_protocol": PROTOCOL,
        "scope": f"rolling_only_base_policy_pairs; train{len(splits['train'])}_gate16_selection16_test16",
        "split_rule": "same corridors, disjoint four-car geometry via held-out gaps; actor-swap pairs stay together",
        "cases": cases, "splits": splits}


def sample_pairs(cases, update_index, episodes=4):
    """Four episodes = two complete pairs; directions rotate across updates."""
    if episodes < 2 or episodes % 2:
        raise ValueError("paired sampling needs a positive even episode count")
    groups = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(list)
    for case in cases:
        pairs[case["pair_id"]].append(case)
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("each scene requires its complete two-member actor swap")
        groups[pair[0]["family"]][pair[0]["direction"]].append(sorted(pair, key=lambda c: c["case_key"]))
    for directions in groups.values():
        for pairs_ in directions.values():
            pairs_.sort(key=lambda pair: pair[0]["pair_id"])
    families = sorted(groups)
    selected = []
    for slot in range(episodes // 2):
        direction = DIRECTIONS[(2 * update_index + slot) % 4]
        family = families[(slot + update_index // 2) % len(families)]
        options = groups[family][direction]
        pair = options[(update_index // 4 + slot // 4) % len(options)]
        selected.extend(pair if update_index % 2 == 0 else reversed(pair))
    return copy.deepcopy(selected)
