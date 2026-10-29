"""Offline scene/response audit for research design; never creates a Client."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import carla

from environment.mut_mappo_env import MutMAPPOEnv
from environment.mut_mappo_events import MutEpisodeEvents, MutRewardConfig
from utils.adversarial_metrics import AgentKinematics
from utils.build_mut_mappo_symmetric_cases_v2 import DEFAULT_OUTPUT, DEFAULT_XODR, ROOT


OUTPUT = ROOT / "docs/figures/mut_mappo_scope_20261026"


def kinematic(x, speed):
    return AgentKinematics(loc=(x, 0.0), fwd=(1.0, 0.0), right=(0.0, -1.0),
                           fwd_speed=speed, speed=abs(speed))


def response_trace(*, slow_speed=1.0, brake_in_protection=False):
    detector = MutEpisodeEvents(MutRewardConfig.from_dict())
    rows = []
    for step in range(1, 121):
        change = 20 if brake_in_protection else 51
        speed = 3.0 if step < change else slow_speed
        result = detector.step(step=step, kinematics={0: kinematic(0, speed), 2: kinematic(8, speed)})
        rows.append({"seconds": round(step * 0.1, 3), "red_speed_mps": speed,
                     "impact": result["team_impact"], "threat": result["team_threat"]})
    active = [row["seconds"] for row in rows if row["threat"] > 0]
    return {"assumptions": "synthetic detector input, constant 8m relative gap, no CARLA dynamics or safety events",
            "event_count": len(detector.response_events),
            "impact_total": sum(row["impact"] for row in rows),
            "threat_total": sum(row["threat"] for row in rows),
            "threat_positive_seconds": round(len(active) * 0.1, 3),
            "first_threat_time": min(active) if active else None,
            "last_threat_time": max(active) if active else None, "rows": rows}


def main():
    manifest_path = DEFAULT_OUTPUT / "cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    env = MutMAPPOEnv.__new__(MutMAPPOEnv)
    env.map = carla.Map("Town05", DEFAULT_XODR.read_text(encoding="utf-8"))
    env.red_ids, env.blue_ids, env.num_agents, env.default_max_steps = (0, 1), (2, 3), 4, 500
    membership = {key: split for split, keys in manifest["splits"].items() for key in keys}
    counts, state_sets = defaultdict(Counter), defaultdict(lambda: defaultdict(set))
    family_sets = defaultdict(set)
    point_groups, profiles = {}, {}
    split_states = defaultdict(set)
    for original in manifest["cases"]:
        case = copy.deepcopy(original)
        case["base_scenario"] = json.loads((ROOT / case["base_scenario"]).read_text(encoding="utf-8"))
        scenario, _ = env._resolve_case(case)
        states = [tuple(round(float(agent[f]), 5) for f in ("x", "y", "yaw", "initial_speed_mps"))
                  for agent in scenario["agents"].values()]
        signatures = {"labeled": tuple(states),
                      "blue_unlabeled": (tuple(states[:2]), tuple(sorted(states[2:]))),
                      "same_color_unlabeled": (tuple(sorted(states[:2])), tuple(sorted(states[2:])))}
        split = membership[case["case_key"]]
        counts[case["direction"]][split] += 1
        for name, signature in signatures.items():
            state_sets["all"][name].add(signature)
            state_sets[split][name].add(signature)
        family_sets[case["family"]].add(signatures["same_color_unlabeled"])
        split_states[split].add(signatures["same_color_unlabeled"])
        for aid, state in enumerate(states):
            color = "red" if aid < 2 else "blue"
            key = (color, *state)
            if key not in point_groups:
                point_groups[key] = {"color": color, "x": state[0], "y": state[1], "yaw": state[2],
                                     "speed_mps": state[3], "assignments": 0}
            point_groups[key]["assignments"] += 1
        if case["parameter_id"] not in profiles:
            blue = sorted(case["blue_spawn_offsets"].values(), key=lambda row: row["forward_m"])
            profiles[case["parameter_id"]] = {"id": case["parameter_id"], "split": split,
                "red_speed": case["red_overrides"]["agent0"]["initial_speed_mps"],
                "near_gap": blue[0]["forward_m"], "far_gap": blue[1]["forward_m"],
                "near_speed": blue[0]["initial_speed_mps"], "far_speed": blue[1]["initial_speed_mps"]}
    geometries = list(point_groups.values())
    extent = {axis: [min(p[axis] for p in geometries) - 12, max(p[axis] for p in geometries) + 12]
              for axis in ("x", "y")}
    road_groups = defaultdict(list)
    for wp in env.map.generate_waypoints(2.0):
        if wp.lane_type == carla.LaneType.Driving:
            road_groups[(wp.road_id, wp.section_id, wp.lane_id)].append(wp)
    roads = []
    for (road, section, lane), waypoints in road_groups.items():
        points = [[round(wp.transform.location.x, 2), round(wp.transform.location.y, 2)]
                  for wp in sorted(waypoints, key=lambda item: item.s)]
        if any(extent["x"][0] <= x <= extent["x"][1] and extent["y"][0] <= y <= extent["y"][1] for x, y in points):
            roads.append({"road": road, "section": section, "lane": lane, "points": points})
    overlap = family_sets["blocking"] & family_sets["cut_in"]
    result = {"schema": "mut_mappo_scope_audit_v1", "source_manifest": str(manifest_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "carla_client_created": False, "world_ticks": 0,
        "direction_counts": {direction: dict(values) for direction, values in counts.items()},
        "unique_states": {split: {name: len(values) for name, values in tiers.items()} for split, tiers in state_sets.items()},
        "family_geometry_overlap": {"blocking": len(family_sets["blocking"]), "cut_in": len(family_sets["cut_in"]),
                                    "intersection": len(overlap), "union": len(family_sets["blocking"] | family_sets["cut_in"])},
        "cross_split_unlabeled_overlap": {f"{a}/{b}": len(split_states[a] & split_states[b])
            for i, a in enumerate(split_states) for b in list(split_states)[i + 1:]},
        "dedup_semantics": "same-color permutation equivalence of positions/yaws/initial speeds; NOT equivalence of policies or rollout outcomes",
        "profiles": list(profiles.values()), "map_extent": extent, "roads": roads, "points": geometries,
        "response_audit": {"drop_2_mps": response_trace(), "drop_1_8_mps": response_trace(slow_speed=1.2),
                           "drop_during_protection": response_trace(slow_speed=1.0, brake_in_protection=True)}}
    assert result["unique_states"]["all"]["labeled"] == 192
    assert not any(result["cross_split_unlabeled_overlap"].values())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scope_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("unique_states", "family_geometry_overlap", "cross_split_unlabeled_overlap")}, indent=2))
    print(json.dumps({key: {name: value for name, value in trace.items() if name != "rows"}
                      for key, trace in result["response_audit"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
