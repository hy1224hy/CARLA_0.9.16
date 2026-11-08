"""Generate a separate symmetric free-driving matrix without importing CARLA.

Geometry reuses the four frozen Town05 v2 straight corridors. Arithmetic
checks are offline evidence only; every new initial state and route still
requires a serial CARLA preflight before training. Old scenario pools remain
unchanged. Run as ``python -m utils.build_base_driving_scenarios_v1 --check``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from training.base_driving_scenarios_v1 import (
    BASE_DRIVING_DIRECTIONS,
    BASE_DRIVING_POOL_PROTOCOL,
    BASE_DRIVING_SAMPLING_PROTOCOL,
    scenario_axes,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_v2"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_base_vnext_v1"
DEFAULT_VALIDATION_DIR = PROJECT_DIR / "data" / "training_scenarios_town05_base_vnext_v1_validation"
ROUTE_LENGTH_M = 32.0
SOURCE_START_ADVANCE_M = 5.0
LANE_WIDTH_M = 3.5


def _serialize(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _signed(value: int) -> str:
    return "z00" if not value else f"{'p' if value > 0 else 'm'}{abs(value):02d}"


def _name(direction: str, startup: str, delta: int, a0_left: bool) -> str:
    return f"scenario_bd1_{direction}_{startup}_{_signed(delta)}_{'a0l' if a0_left else 'a0r'}.json"


def _source(direction: str) -> tuple[Path, dict]:
    index = BASE_DRIVING_DIRECTIONS.index(direction) + 17
    path = SOURCE_DIR / f"scenario_{index:02d}_{direction}_free_flow.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _scenario(direction: str, startup: str, delta: int, a0_left: bool, *, split: str) -> dict:
    source_path, source = _source(direction)
    source_agents = source["agents"]
    center_x = sum(a["x"] for a in source_agents.values()) / 2
    center_y = sum(a["y"] for a in source_agents.values()) / 2
    yaw = float(source_agents["agent0"]["yaw"])
    forward = (math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))
    # CARLA uses a left-handed world: its right vector is (-sin(yaw), cos(yaw)).
    # Do not copy historical scene tags or a right-handed plotting convention.
    left = (forward[1], -forward[0])
    speed = 0.0 if startup == "stationary" else 2.5
    agents, goals, roles = {}, {}, {}
    for aid in range(2):
        is_left = a0_left if aid == 0 else not a0_left
        offset = delta / 2 if aid == 0 else -delta / 2
        longitudinal = SOURCE_START_ADVANCE_M + offset
        lateral = LANE_WIDTH_M / 2 * (1 if is_left else -1)
        x = center_x + forward[0] * longitudinal + left[0] * lateral
        y = center_y + forward[1] * longitudinal + left[1] * lateral
        key = f"agent{aid}"
        agents[key] = {
            "x": round(x, 9), "y": round(y, 9), "z": 0.5,
            "yaw": yaw, "initial_speed_mps": speed,
        }
        goals[key] = {
            "x": round(x + forward[0] * ROUTE_LENGTH_M, 9),
            "y": round(y + forward[1] * ROUTE_LENGTH_M, 9),
            "z": 0.5, "yaw": yaw, "half_length": 3.0, "half_width": 1.6,
        }
        own_delta = delta if aid == 0 else -delta
        relation = "parallel" if not own_delta else ("ahead" if own_delta > 0 else "behind")
        roles[key] = f"{'left' if is_left else 'right'}_{relation}"
    name = _name(direction, startup, delta, a0_left)
    axes = scenario_axes(name)
    shared = {
        "x": round(sum(g["x"] for g in goals.values()) / 2, 9),
        "y": round(sum(g["y"] for g in goals.values()) / 2, 9),
        "z": 0.5, "yaw": yaw, "half_length": 3.0, "half_width": 1.6,
    }
    return {
        "schema_version": 3,
        "town": "Town05",
        "family": "free_flow",
        "task_family": "adjacent_free_flow",
        "difficulty": 1 + int(delta != 0),
        "curriculum_stage": int(delta != 0),
        "direction": direction,
        "startup_mode": startup,
        "description": "Symmetric actor-slot pair, independent adjacent-lane driving, equal individual route lengths",
        "goal_mode": "per_agent_explicit",
        "episode_max_steps": 400,
        "tags": [direction, "two_lane", "free_flow", "non_interactive", "non_adversarial", "same_speed", f"{startup}_start", "base_driving_v1"],
        "agents": agents,
        "agent_goal_zones": goals,
        "goal_zone": shared,
        "agent_roles": roles,
        "delta_s_m": delta,
        "route_length_m": ROUTE_LENGTH_M,
        "goal_entry_distance_m": ROUTE_LENGTH_M - shared["half_length"],
        "pair_id": axes["pair_id"],
        "physical_template_id": axes["pair_id"],
        "symmetry_partner": _name(direction, startup, -delta, not a0_left),
        "split": split,
        "validation_status": "PENDING_CARLA_INITIAL_STATE_AND_ROUTE_PREFLIGHT",
        "geometry_provenance": {
            "source": source_path.relative_to(PROJECT_DIR).as_posix(),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "method": "source_start_midpoint_plus_source_yaw_tangent_and_3.5m_lane_spacing",
            "source_start_advance_m": SOURCE_START_ADVANCE_M,
            "physical_left_vector": "(sin(yaw), -cos(yaw)); negative CARLA right vector",
            "is_live_map_projection": False,
        },
    }


def build_payloads(*, split: str = "train") -> dict[str, dict]:
    if split not in {"train", "validation"}:
        raise ValueError(f"invalid split: {split}")
    deltas = (-8, 0, 8) if split == "train" else (-4, 4)
    payloads = {
        _name(direction, startup, delta, a0_left): _scenario(direction, startup, delta, a0_left, split=split)
        for direction in BASE_DRIVING_DIRECTIONS
        for startup in ("stationary", "rolling")
        for delta in deltas
        for a0_left in (True, False)
    }
    validate_payloads(payloads)
    return payloads


def validate_payloads(payloads: dict[str, dict]) -> None:
    """Check consequential geometry/role invariants without pretending to map-test."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, payload in payloads.items():
        partner = payloads.get(payload["symmetry_partner"])
        if partner is None or partner["symmetry_partner"] != name:
            raise ValueError(f"missing reciprocal actor-swap partner: {name}")
        if payload["goal_mode"] != "per_agent_explicit":
            raise ValueError("equal per-agent routes require explicit goals")
        if payload["pair_id"] != scenario_axes(name)["pair_id"]:
            raise ValueError(f"physical pair identity mismatch: {name}")
        grouped[payload["pair_id"]].append(name)
        for aid, other in (("agent0", "agent1"), ("agent1", "agent0")):
            if payload["agents"][aid] != partner["agents"][other] or payload["agent_goal_zones"][aid] != partner["agent_goal_zones"][other]:
                raise ValueError(f"incomplete initial-state/goal actor swap: {name}/{aid}")
            start, goal = payload["agents"][aid], payload["agent_goal_zones"][aid]
            if not math.isclose(math.hypot(goal["x"] - start["x"], goal["y"] - start["y"]), ROUTE_LENGTH_M, abs_tol=1e-7):
                raise ValueError(f"individual route length mismatch: {name}/{aid}")
        yaw = math.radians(payload["agents"]["agent0"]["yaw"])
        a, b = payload["agents"]["agent0"], payload["agents"]["agent1"]
        actual_delta = (a["x"] - b["x"]) * math.cos(yaw) + (a["y"] - b["y"]) * math.sin(yaw)
        if not math.isclose(actual_delta, payload["delta_s_m"], abs_tol=1e-7):
            raise ValueError(f"relative longitudinal coordinate mismatch: {name}")
    if any(len(members) != 2 for members in grouped.values()):
        raise ValueError("each physical template must have exactly two actor assignments")


def _manifest(payloads: dict[str, dict], *, split: str) -> dict:
    pairs: dict[str, list[str]] = defaultdict(list)
    for name, payload in sorted(payloads.items()):
        pairs[payload["pair_id"]].append(name)
    return {
        "pool_protocol": BASE_DRIVING_POOL_PROTOCOL,
        "sampling_protocol": BASE_DRIVING_SAMPLING_PROTOCOL,
        "split": split,
        "scenario_count": len(payloads),
        "physical_template_count": len(pairs),
        "scenario_sha256": {name: hashlib.sha256(_serialize(payload).encode("utf-8")).hexdigest() for name, payload in sorted(payloads.items())},
        "physical_template_groups": dict(pairs),
        "stage_support_sets": {str(stage): sorted(name for name, payload in payloads.items() if payload["curriculum_stage"] <= stage) for stage in (0, 1)},
        "directions": list(BASE_DRIVING_DIRECTIONS),
        "direction_counts": dict(Counter(payload["direction"] for payload in payloads.values())),
        "route_length_m": ROUTE_LENGTH_M,
        "coordinate_convention": "CARLA left-handed, physical left=(sin(yaw),-cos(yaw))",
        "goal_entry_distance_m": ROUTE_LENGTH_M - 3.0,
        "relative_longitudinal_deltas_m": sorted({payload["delta_s_m"] for payload in payloads.values()}),
        "initial_speed_mps": [0.0, 2.5],
        "initial_speed_semantics": "requested physical speed at first policy observation, committed by existing environment velocity warm-start/readback; not an ongoing target controller",
        "sampling": {"rollout_episodes": 16, "complete_pairs_per_direction_per_update": 2, "support_phase": "safely_completed_update", "actor_weights_are_not_swapped": True},
        "split_rule": "whole actor-swap physical-template groups; validation uses +/-4m offsets absent from training's 0/+/-8m",
        "evaluation_scope": "same-corridor engineering validation of new combinations; not formal heldout/generalization evidence",
        "legacy_r8a_12": "unchanged separate regression evaluation, excluded from this pool and its sampling",
        "validation_status": "OFFLINE_INVARIANTS_PASS_LIVE_CARLA_PREFLIGHT_PENDING",
    }


def write_payloads(*, output_dir: Path = DEFAULT_OUTPUT_DIR, validation_dir: Path = DEFAULT_VALIDATION_DIR, check: bool = False) -> None:
    train, validation = build_payloads(), build_payloads(split="validation")
    if {p["physical_template_id"] for p in train.values()} & {p["physical_template_id"] for p in validation.values()}:
        raise ValueError("physical-template train/validation leakage")
    for root, payloads, split in ((Path(output_dir), train, "train"), (Path(validation_dir), validation, "validation")):
        files = {**payloads, "protocol/manifest.json": _manifest(payloads, split=split)}
        actual_names = {path.name for path in root.glob("*.json")}
        if actual_names - set(payloads):
            raise ValueError(f"unexpected scenario files, refusing overwrite: {root}: {actual_names - set(payloads)}")
        if check and actual_names != set(payloads):
            raise ValueError(f"scenario file set drift: {root}")
        for name, payload in files.items():
            path = root / name
            serialized = _serialize(payload)
            if check:
                if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
                    raise ValueError(f"generated scenario or manifest drift: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(serialized, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    args = parser.parse_args()
    write_payloads(output_dir=args.output_dir, validation_dir=args.validation_dir, check=args.check)
    print("PASS: 48 training / 32 validation scenarios; complete physical-template pairs; CARLA preflight pending")


if __name__ == "__main__":
    main()
