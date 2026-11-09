"""Build the frozen Town05 R8-A four-stage scenario pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_v2"
OUTPUT_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_r8a_v1"


def _goal(x: float, y: float, yaw: float) -> Dict[str, float]:
    return {
        "x": x,
        "y": y,
        "z": 0.5,
        "yaw": yaw,
        "half_length": 3.0,
        "half_width": 1.6,
    }


def _short_scenario(*, direction: str, stationary: bool) -> Dict[str, object]:
    if direction == "northbound":
        yaw = 89.98
        starts = {
            "agent0": (-195.09, -64.58),
            "agent1": (-191.59, -64.58),
        }
        goals = {
            "agent0": (-195.09, -49.58),
            "agent1": (-191.59, -49.58),
        }
        shared_goal = (-193.34, -49.58)
        role_tag = "agent0_left"
    elif direction == "southbound":
        yaw = -90.02
        starts = {
            "agent0": (-188.08, -19.585),
            "agent1": (-184.58, -19.585),
        }
        goals = {
            "agent0": (-188.08, -34.585),
            "agent1": (-184.58, -34.585),
        }
        shared_goal = (-186.33, -34.585)
        role_tag = "agent1_left"
    else:
        raise ValueError(f"unsupported short direction: {direction!r}")

    speed = 0.0 if stationary else 2.0
    stage = "S0-B" if stationary else "S0-A"
    family = (
        "r8a_short_free_flow_stationary"
        if stationary else "r8a_short_free_flow_rolling"
    )
    start_tag = "stationary_start" if stationary else "rolling_start"
    return {
        "schema_version": 3,
        "town": "Town05",
        "difficulty": 2 if stationary else 1,
        "curriculum_stage": stage,
        "family": family,
        "goal_mode": "preserve_start_lane",
        "episode_max_steps": 200,
        "description": (
            f"R8-A {stage} Road 44 {direction} mirrored short free-flow"
        ),
        "tags": [
            direction,
            "two_lane",
            "free_flow",
            "short_route",
            "non_interactive",
            start_tag,
            "same_speed",
            role_tag,
            "r8a",
            stage.lower(),
        ],
        "goal_zone": {
            "x": shared_goal[0],
            "y": shared_goal[1],
            "z": 0.5,
            "yaw": yaw,
            "half_length": 3.0,
            "half_width": 3.4,
        },
        "agents": {
            agent: {
                "x": position[0],
                "y": position[1],
                "z": 0.5,
                "yaw": yaw,
                "initial_speed_mps": speed,
            }
            for agent, position in starts.items()
        },
        "agent_goal_zones": {
            agent: _goal(position[0], position[1], yaw)
            for agent, position in goals.items()
        },
    }


def build_payloads() -> Dict[str, Dict[str, object]]:
    payloads = {
        "scenario_01_north_short_rolling_a0_left.json": _short_scenario(
            direction="northbound", stationary=False
        ),
        "scenario_02_south_short_rolling_a1_left.json": _short_scenario(
            direction="southbound", stationary=False
        ),
        "scenario_03_north_short_stationary_a0_left.json": _short_scenario(
            direction="northbound", stationary=True
        ),
        "scenario_04_south_short_stationary_a1_left.json": _short_scenario(
            direction="southbound", stationary=True
        ),
    }
    copies = (
        ("scenario_17_northbound_free_flow.json", "scenario_05_northbound_free_flow.json", 3, "S1"),
        ("scenario_18_southbound_free_flow.json", "scenario_06_southbound_free_flow.json", 3, "S1"),
        ("scenario_19_eastbound_free_flow.json", "scenario_07_eastbound_free_flow.json", 3, "S1"),
        ("scenario_20_westbound_free_flow.json", "scenario_08_westbound_free_flow.json", 3, "S1"),
        ("scenario_01_north_parallel_stationary_a0_left.json", "scenario_09_north_parallel_stationary_a0_left.json", 4, "S2"),
        ("scenario_02_south_parallel_stationary_a1_left.json", "scenario_10_south_parallel_stationary_a1_left.json", 4, "S2"),
        ("scenario_03_east_parallel_stationary_a1_left.json", "scenario_11_east_parallel_stationary_a1_left.json", 4, "S2"),
        ("scenario_04_west_parallel_stationary_a0_left.json", "scenario_12_west_parallel_stationary_a0_left.json", 4, "S2"),
    )
    for source_name, destination_name, difficulty, stage in copies:
        payload = json.loads((SOURCE_DIR / source_name).read_text(encoding="utf-8"))
        payload["difficulty"] = difficulty
        payload["curriculum_stage"] = stage
        payload["episode_max_steps"] = 500
        payload["description"] = f"R8-A {stage}: {payload['description']}"
        payload["tags"] = [*payload.get("tags", []), "r8a", stage.lower()]
        payloads[destination_name] = payload
    return payloads


def _serialized(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_payloads(*, check: bool = False) -> None:
    payloads = build_payloads()
    if check:
        actual_names = sorted(path.name for path in OUTPUT_DIR.glob("*.json"))
        if actual_names != sorted(payloads):
            raise RuntimeError(
                f"R8-A scenario file set drift: {actual_names!r} != "
                f"{sorted(payloads)!r}"
            )
        drifted = [
            name for name, payload in payloads.items()
            if (OUTPUT_DIR / name).read_text(encoding="utf-8")
            != _serialized(payload)
        ]
        if drifted:
            raise RuntimeError(f"R8-A generated scenario drift: {drifted!r}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.json"):
        if path.name not in payloads:
            raise RuntimeError(f"unexpected R8-A scenario file: {path}")
    for name, payload in payloads.items():
        (OUTPUT_DIR / name).write_text(_serialized(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_payloads(check=args.check)
    print(
        "R8-A scenario generation PASS"
        if args.check else "R8-A scenarios generated"
    )


if __name__ == "__main__":
    main()
