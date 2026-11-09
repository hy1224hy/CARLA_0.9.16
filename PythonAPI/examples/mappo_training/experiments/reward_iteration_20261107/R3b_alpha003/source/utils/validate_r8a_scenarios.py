"""Strict offline validator for the frozen Town05 R8-A scenario pool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping

from training.r8a_curriculum import (
    R8A_STAGE_NAMES,
    R8A_STAGE_NEW_SCENARIOS,
    r8a_cumulative_stages,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = (
    PROJECT_DIR / "data" / "training_scenarios_town05_r8a_v1"
)
FORBIDDEN_TOKENS = {
    "required_merge",
    "blocker",
    "closing",
    "following",
    "yielding",
}


def _distance(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _load_payloads(scenario_dir: Path) -> Dict[str, Dict[str, object]]:
    payloads: Dict[str, Dict[str, object]] = {}
    for path in sorted(scenario_dir.glob("*.json")):
        payloads[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def validate_r8a_scenario_pool(scenario_dir: Path) -> List[str]:
    errors: List[str] = []
    payloads = _load_payloads(scenario_dir)
    expected_names = [
        name for additions in R8A_STAGE_NEW_SCENARIOS for name in additions
    ]
    if sorted(payloads) != sorted(expected_names):
        errors.append(
            f"scenario file set mismatch: {sorted(payloads)!r} != "
            f"{sorted(expected_names)!r}"
        )
        return errors

    for stage_index, (stage_name, additions) in enumerate(
        zip(R8A_STAGE_NAMES, R8A_STAGE_NEW_SCENARIOS), start=1
    ):
        for name in additions:
            payload = payloads[name]
            if payload.get("schema_version") != 3:
                errors.append(f"{name}: schema_version must be 3")
            if payload.get("town") != "Town05":
                errors.append(f"{name}: town must be Town05")
            if payload.get("difficulty") != stage_index:
                errors.append(
                    f"{name}: difficulty must be {stage_index}"
                )
            if payload.get("curriculum_stage") != stage_name:
                errors.append(
                    f"{name}: curriculum_stage must be {stage_name}"
                )
            episode_max_steps = payload.get("episode_max_steps")
            expected_limit = 200 if stage_index <= 2 else 500
            if episode_max_steps != expected_limit:
                errors.append(
                    f"{name}: episode_max_steps must be {expected_limit}"
                )
            serialized_tokens = {
                str(payload.get("family", "")).lower(),
                *(str(tag).lower() for tag in payload.get("tags", [])),
            }
            if any(
                forbidden in token
                for forbidden in FORBIDDEN_TOKENS
                for token in serialized_tokens
            ):
                errors.append(f"{name}: forbidden interaction token present")

    short_names = [
        name for additions in R8A_STAGE_NEW_SCENARIOS[:2] for name in additions
    ]
    for name in short_names:
        payload = payloads[name]
        agents = payload.get("agents", {})
        goals = payload.get("agent_goal_zones", {})
        if set(agents) != {"agent0", "agent1"}:
            errors.append(f"{name}: short scenario must contain two agents")
            continue
        if set(goals) != {"agent0", "agent1"}:
            errors.append(f"{name}: short scenario needs per-agent goals")
            continue
        if not math.isclose(
            _distance(agents["agent0"], agents["agent1"]),
            3.5,
            abs_tol=1e-9,
        ):
            errors.append(f"{name}: lane spacing must be exactly 3.5m")
        for agent_name in ("agent0", "agent1"):
            route_length = _distance(agents[agent_name], goals[agent_name])
            if not math.isclose(route_length, 15.0, abs_tol=1e-9):
                errors.append(
                    f"{name}/{agent_name}: route length {route_length} != 15m"
                )

    rolling = [payloads[name] for name in R8A_STAGE_NEW_SCENARIOS[0]]
    stationary = [payloads[name] for name in R8A_STAGE_NEW_SCENARIOS[1]]
    for payload in rolling:
        speeds = {
            float(agent["initial_speed_mps"])
            for agent in payload["agents"].values()
        }
        if speeds != {2.0}:
            errors.append("S0-A rolling speeds must be exactly 2.0m/s")
    for payload in stationary:
        speeds = {
            float(agent["initial_speed_mps"])
            for agent in payload["agents"].values()
        }
        if speeds != {0.0}:
            errors.append("S0-B stationary speeds must be exactly 0.0m/s")

    for rolling_payload, stationary_payload in zip(rolling, stationary):
        for agent_name in ("agent0", "agent1"):
            rolling_agent = dict(rolling_payload["agents"][agent_name])
            stationary_agent = dict(stationary_payload["agents"][agent_name])
            rolling_agent.pop("initial_speed_mps")
            stationary_agent.pop("initial_speed_mps")
            if rolling_agent != stationary_agent:
                errors.append(
                    f"{agent_name}: S0-A/S0-B start geometry drift"
                )
            if (
                rolling_payload["agent_goal_zones"][agent_name]
                != stationary_payload["agent_goal_zones"][agent_name]
            ):
                errors.append(
                    f"{agent_name}: S0-A/S0-B goal geometry drift"
                )

    north = rolling[0]
    south = rolling[1]
    north_lengths = sorted(
        _distance(north["agents"][agent], north["agent_goal_zones"][agent])
        for agent in ("agent0", "agent1")
    )
    south_lengths = sorted(
        _distance(south["agents"][agent], south["agent_goal_zones"][agent])
        for agent in ("agent0", "agent1")
    )
    if north_lengths != south_lengths:
        errors.append("north/south short route lengths are not mirrored")
    if "agent0_left" not in north["tags"] or "agent1_left" not in south["tags"]:
        errors.append("north/south mirror must exchange agent/lane roles")

    difficulty_stages = []
    for difficulty in range(1, 5):
        difficulty_stages.append([
            name for name in expected_names
            if int(payloads[name]["difficulty"]) <= difficulty
        ])
    if difficulty_stages != r8a_cumulative_stages():
        errors.append("difficulty metadata does not reconstruct frozen stages")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    args = parser.parse_args()
    errors = validate_r8a_scenario_pool(args.scenario_dir.resolve())
    if errors:
        raise SystemExit("R8-A scenario validation failed:\n- " + "\n- ".join(errors))
    print("R8-A scenario validation PASS: 12 scenarios, stages 2/4/8/12")


if __name__ == "__main__":
    main()
