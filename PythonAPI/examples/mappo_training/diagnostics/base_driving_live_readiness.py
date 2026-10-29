"""Explicit bounded simulator setup/smoke, separate from learning evidence."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback

from training.checkpoint_protocol import atomic_write_json, utc_now_iso


def run(args):
    from utils.mut_mappo_session import CarlaOwner
    from evaluation.evaluate_base_policy_c0 import (
        _preflight_idle_world, _cleanup_environment, _restore_failed_initialization,
    )
    report = {"status": "RUNNING", "mode": args.mode, "started_at": utc_now_iso(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "meaning": "Mechanical smoke only; no learning or driving qualification", "cases": []}
    try:
        with CarlaOwner(run_dir=args.output.parent, purpose="explicit bounded base driving readiness"):
            idle = _preflight_idle_world(host="127.0.0.1", port=2000, expected_version="0.9.16")
            report["idle_before"] = idle
            import carla
            client = carla.Client("127.0.0.1", 2000)
            client.set_timeout(60)
            current = client.get_world().get_map().name.rsplit("/", 1)[-1]
            report["map_before"] = current
            if args.mode == "town":
                if current != "Town05":
                    client.load_world("Town05")
                report["map_after"] = client.get_world().get_map().name
                report["idle_after"] = _preflight_idle_world(host="127.0.0.1", port=2000, expected_version="0.9.16")
            elif args.mode == "probe":
                from utils.preflight_base_driving_scenarios_v1 import envelope_corners
                directory = Path(__file__).resolve().parents[1] / "data/training_scenarios_town05_base_vnext_v1"
                road_map = client.get_world().get_map()
                for path in sorted(directory.glob("*stationary_z00_a0l.json")):
                    scene = json.loads(path.read_text(encoding="utf-8"))
                    case = {"scenario": path.name, "agents": {}}
                    for aid, start in scene["agents"].items():
                        points = [(start["x"], start["y"])] + envelope_corners(start)
                        values = []
                        for x, y in points:
                            point = carla.Location(x=x,y=y,z=.5)
                            row = {"xy": [x,y]}
                            for key, project, kind in (("raw_any",False,carla.LaneType.Any), ("project_any",True,carla.LaneType.Any), ("project_driving",True,carla.LaneType.Driving)):
                                w = road_map.get_waypoint(point, project_to_road=project, lane_type=kind)
                                row[key] = None if w is None else {"road": w.road_id, "lane": w.lane_id, "width": w.lane_width,
                                    "xy": [w.transform.location.x,w.transform.location.y], "yaw": w.transform.rotation.yaw}
                            values.append(row)
                        case["agents"][aid] = values
                    report["cases"].append(case)
            else:
                if current != "Town05":
                    raise ValueError("smoke requires Town05")
                from utils.preflight_base_driving_scenarios_v1 import require_geometry_preflight
                from training.base_driving_contract import load_contract
                from environment.base_driving_env import BaseDrivingEnv
                root = Path(__file__).resolve().parents[1]
                contract = load_contract(args.contract, root)
                directory = Path(contract["scenario_dir"])
                require_geometry_preflight(directory, args.geometry)
                scenes = [(name, json.loads((directory / name).read_text(encoding="utf-8")))
                          for name in sorted(directory.glob("*.json"))]
                # Eight reset cases: all directions and stationary/rolling starts.
                selected = {}
                for name, scene in scenes:
                    if scene["curriculum_stage"] == 0:
                        selected.setdefault((scene["direction"], scene["startup_mode"]), (name, scene))
                if len(selected) != 8:
                    raise ValueError("smoke coverage must include all direction/startup strata")
                env = None
                try:
                    env = BaseDrivingEnv(host="127.0.0.1", port=2000, town="Town05", num_agents=2,
                        timestep=.1, max_steps=400, spawn_vehicles=False, disable_tensorboard=True,
                        reward_version="v6", pace_profile="instant_pace_v6", reward_gamma=.999,
                        lidar_range_m=20., ttc_max_range_m=20., follow_spectator=True,
                        base_driving_reward_profile=contract["record"]["reward_profile"])
                    env._original_synchronous_mode = idle["synchronous_mode"]
                    env._original_fixed_delta_seconds = idle["fixed_delta_seconds"]
                    for name, original in selected.values():
                        scene = copy.deepcopy(original)
                        env.apply_scenario_config(scene)
                        obs, info = env.reset()
                        row = {"scenario": Path(name).name, "initial": {}, "steps": 0}
                        for aid in (0, 1):
                            vehicle = env.vehicles[aid]
                            pose, velocity, box = vehicle.get_transform(), vehicle.get_velocity(), vehicle.bounding_box
                            expected = scene["agents"][f"agent{aid}"]
                            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                            position_error = math.hypot(pose.location.x-expected["x"], pose.location.y-expected["y"])
                            yaw_error = abs((pose.rotation.yaw-expected["yaw"]+180) % 360-180)
                            if position_error > .5 or yaw_error > 5 or abs(speed-expected["initial_speed_mps"]) > .5:
                                raise ValueError(f"initial physical state mismatch: {name} agent{aid}")
                            if box.extent.x+abs(box.location.x) > 3 or box.extent.y+abs(box.location.y) > 1.3:
                                raise ValueError("actual vehicle bbox exceeds preflight conservative envelope")
                            goal = env.agent_goal_zones[aid]
                            requested = scene["agent_goal_zones"][f"agent{aid}"]
                            if any(abs(getattr(goal["center"], k)-requested[k]) > 1e-3 for k in ("x", "y")):
                                raise ValueError("own explicit goal was overwritten")
                            row["initial"][str(aid)] = {"position_error_m": position_error, "yaw_error_deg": yaw_error,
                                "speed_mps": speed, "goal": {k: getattr(goal["center"], k) for k in ("x", "y")},
                                "bbox_half_length_m": box.extent.x, "bbox_half_width_m": box.extent.y}
                        for _ in range(6):
                            _, reward, _, info = env.step({aid: [0.2, 0., 0.] for aid in (0, 1) if info["active_mask"][aid]})
                            row["steps"] += 1
                            for aid, amount in reward.items():
                                if not math.isfinite(amount) or abs(amount-math.fsum(info["reward_details"][aid].values())) > 1e-6:
                                    raise ValueError("reward residual/nonfinite during smoke")
                            if not any(info["active_mask"].values()):
                                break
                        report["cases"].append(row)
                        print("SMOKE_CASE", row["scenario"], flush=True)
                finally:
                    audit, error = (_cleanup_environment(env) if env is not None else
                                    _restore_failed_initialization(host="127.0.0.1", port=2000, preflight=idle))
                    report["cleanup"] = {"audit": audit, "error": error}
                    if error:
                        raise RuntimeError(error)
            report["status"] = "PASS_MECHANICAL_ONLY"
    except BaseException:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["ended_at"] = utc_now_iso()
        atomic_write_json(report, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("town", "probe", "smoke"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--contract", type=Path, default=Path("configs/base_driving/base_driving_v6_v1.json"))
    run(parser.parse_args())
