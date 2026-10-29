"""File/CPU diagnostics of scene coverage and reward loopholes; never CARLA."""
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import carla

from environment.mut_mappo_efficiency_reward import TeamEfficiencyReward
from environment.mut_mappo_env import MutMAPPOEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def scene_audit():
    manifest = json.loads((ROOT / "experiments/adversarial_u1000_pilot_20261028_v1/cases.json").read_text())
    membership = {k: s for s, keys in manifest["splits"].items() for k in keys}
    rows = []
    for case in manifest["cases"]:
        source = json.loads((ROOT / case["base_scenario"]).read_text())
        agents = source["agents"]
        yaw = math.radians(agents["agent0"]["yaw"])
        axis = (math.cos(yaw), math.sin(yaw))
        projected_red = {i: agents[f"agent{i}"]["x"] * axis[0] + agents[f"agent{i}"]["y"] * axis[1] for i in (0, 1)}
        projected_blue, blue_remaining, red_remaining = [], [], []
        for bid in (2, 3):
            offset = case["blue_spawn_offsets"][f"agent{bid}"]
            rid = offset["anchor_red_id"]
            start, goal = agents[f"agent{rid}"], source["agent_goal_zones"][f"agent{rid}"]
            length = (goal["x"]-start["x"]) * axis[0] + (goal["y"]-start["y"]) * axis[1]
            projected_blue.append(projected_red[rid] + offset["forward_m"])
            red_remaining.append(length)
            blue_remaining.append(length-offset["forward_m"])
        rows.append({"case_key": case["case_key"], "split": membership[case["case_key"]],
                     "family": case["family"], "both_blue_ahead_of_all_red": min(projected_blue) > max(projected_red.values()),
                     "red_longitudinal_span_m": max(projected_red.values()) - min(projected_red.values()),
                     "blue_offsets_m": [x["forward_m"] for x in case["blue_spawn_offsets"].values()],
                     "red_remaining_center_m": red_remaining, "blue_remaining_center_m": blue_remaining})
    return {"cases": len(rows), "all_front_count": sum(r["both_blue_ahead_of_all_red"] for r in rows),
            "split_counts": dict(Counter(r["split"] for r in rows)), "rows": rows,
            "scope": "straight-axis arithmetic from saved templates; no realized spawn or trajectory evidence"}


def stationary_rule_probe(speed, goal_distance):
    env = MutMAPPOEnv.__new__(MutMAPPOEnv)
    env.blue_ids = (2, 3)
    env.task_config = {"stall": {"enabled": True, "seconds": 3.0, "speed_mps": 0.2}}
    env.mut_reward_config = SimpleNamespace(birth_protect_secs=3.0)
    env.timestep = 0.1
    env.active_mask = {2: True, 3: True}
    env._stall_seconds = {2: 0.0, 3: 0.0}
    env._task_violations = {2: False, 3: False}
    env._task_stall_evidence = {}
    env.vehicles = {i: SimpleNamespace(is_alive=True,
        get_location=lambda: carla.Location(x=0),
        get_velocity=lambda: carla.Vector3D(x=speed)) for i in (2, 3)}
    env.road_reference_waypoints = {i: SimpleNamespace(transform=SimpleNamespace(rotation=SimpleNamespace(yaw=0))) for i in (2, 3)}
    env._get_target_location = lambda aid: carla.Location(x=goal_distance)
    env._pace_has_front_blocker = lambda aid: False
    for step in range(31, 71):
        env.current_step = step
        env._update_blue_task_stall({i: SimpleNamespace(alive=True) for i in (2, 3)})
    return {"speed_mps": speed, "goal_distance_m": goal_distance, "task_violation_after_4s": env._task_violations[2],
            "evidence": env._task_stall_evidence[2],
            "scope": "isolated production rule, fixed synthetic kinematics, no front blocker; not a physical rollout"}


def main():
    with patch("carla.Client", side_effect=AssertionError("No CARLA Client in this diagnostic")):
        reward = TeamEfficiencyReward(horizon=500)
        for step in range(31, 51):
            reward.step(step=step, progress={0: .4, 1: .4, 2: 0., 3: 0.}, reached={}, eligible=True,
                        safety={}, costs={}, base_details={}, reference={"0": [.5] * 501, "1": [.5] * 501})
        probes = {"creep_away_from_goal": stationary_rule_probe(.21, 20.),
                  "stop_away_from_goal": stationary_rule_probe(0., 20.),
                  "stop_5m_before_goal": stationary_rule_probe(0., 5.)}
        assert not probes["creep_away_from_goal"]["task_violation_after_4s"]
        assert probes["stop_away_from_goal"]["task_violation_after_4s"]
        assert not probes["stop_5m_before_goal"]["task_violation_after_4s"]
        assert math.isclose(reward.components_total["attack"], .04, abs_tol=1e-12)
        report = {"status": "OFFLINE_DESIGN_DIAGNOSTICS", "physical_world_ticks": 0,
                  "scene_coverage": scene_audit(), "stationary_rule_probes": probes,
                  "reward_scale_example": {"red_progress_lag": .1, "duration_seconds": 2.,
                      "horizon_seconds": 50., "attack_weight": 10.,
                      "attack_reward": reward.components_total["attack"], "collision_event_cost": 50.,
                      "scope": "synthetic fixed progress inputs, other reward terms zero; not observed learning"}}
    (OUT / "offline_probes.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": report["scene_coverage"]["cases"],
                      "both_blue_ahead_of_all_red": report["scene_coverage"]["all_front_count"],
                      "stationary_rule_flags": {k: v["task_violation_after_4s"] for k, v in probes.items()},
                      "synthetic_attack_reward": report["reward_scale_example"]["attack_reward"],
                      "physical_world_ticks": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
