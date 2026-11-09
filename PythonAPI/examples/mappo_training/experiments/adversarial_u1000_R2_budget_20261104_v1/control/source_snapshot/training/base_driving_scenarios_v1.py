"""Versioned, complete-pair sampling for the base-driving scenario matrix.

This module is CARLA-free. A pair exchanges the full initial state and goal
between actor slots; it never exchanges policy or optimizer identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence


BASE_DRIVING_POOL_PROTOCOL = "town05_base_driving_matrix_v1"
BASE_DRIVING_SAMPLING_PROTOCOL = "town05_base_driving_complete_pairs_v1"
BASE_DRIVING_DIRECTIONS = ("northbound", "southbound", "eastbound", "westbound")
BASE_DRIVING_STAGE_NAMES = ("parallel", "staggered")
_NAME = re.compile(
    r"^scenario_bd1_(northbound|southbound|eastbound|westbound)_"
    r"(stationary|rolling)_(m\d{2}|z00|p\d{2})_(a0l|a0r)\.json$"
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def scenario_axes(name: str) -> dict[str, object]:
    """Decode an immutable filename; useful before a CARLA environment exists."""
    match = _NAME.fullmatch(str(name))
    if match is None:
        raise ValueError(f"unregistered base-driving scenario name: {name!r}")
    direction, startup, signed, role = match.groups()
    delta = int(signed[1:]) * (-1 if signed.startswith("m") else 1)
    left_delta = delta if role == "a0l" else -delta
    return {
        "direction": direction,
        "startup_mode": startup,
        "delta_s_m": delta,
        "agent0_lane": "left" if role == "a0l" else "right",
        "physical_left_delta_s_m": left_delta,
        "pair_id": f"bd1_{direction}_{startup}_left_{left_delta:+03d}",
        "curriculum_stage": int(delta != 0),
    }


def _pair_groups(available_scenarios: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
    names = [str(name) for name in available_scenarios]
    if not names or len(set(names)) != len(names):
        raise ValueError("available_scenarios must be nonempty and unique")
    pairs: dict[str, list[str]] = defaultdict(list)
    directions: dict[str, str] = {}
    for name in names:
        axes = scenario_axes(name)
        pair_id = str(axes["pair_id"])
        pairs[pair_id].append(name)
        directions[pair_id] = str(axes["direction"])
    grouped: dict[str, list[tuple[str, str]]] = {d: [] for d in BASE_DRIVING_DIRECTIONS}
    for pair_id in sorted(pairs):
        members = sorted(pairs[pair_id])
        if len(members) != 2:
            raise ValueError(f"complete two-member actor swap required: {pair_id}: {members}")
        axes = [scenario_axes(member) for member in members]
        if {a["agent0_lane"] for a in axes} != {"left", "right"}:
            raise ValueError(f"pair does not exchange actor lane roles: {pair_id}")
        grouped[directions[pair_id]].append(tuple(members))
    counts = {direction: len(pairs_) for direction, pairs_ in grouped.items()}
    if 0 in counts.values() or len(set(counts.values())) != 1:
        raise ValueError(f"four directions require equal nonzero complete pairs: {counts}")
    # Also reject support sets that hide a startup or physical-role imbalance.
    strata = {
        direction: Counter(
            (scenario_axes(pair[0])["startup_mode"], scenario_axes(pair[0])["physical_left_delta_s_m"])
            for pair in pairs_
        )
        for direction, pairs_ in grouped.items()
    }
    if any(value != strata[BASE_DRIVING_DIRECTIONS[0]] for value in strata.values()):
        raise ValueError("each direction must expose the same startup/physical relation strata")
    return grouped


def sample_base_driving_balanced(
    available_scenarios: Sequence[str], n_episodes: int, *, phase: int
) -> list[str]:
    """Select complete pairs and equal direction quotas, normally 16 episodes.

    A full 48-scenario pool has six pairs per direction. Each 16-episode
    update selects two pairs per direction, so every configuration is exposed
    once per three updates. The parallel 16-scenario subset exposes all pairs
    every update. The phase is the number of safely completed updates.
    """
    episode_count = _integer(n_episodes, "n_episodes", minimum=1)
    completed = _integer(phase, "phase")
    if episode_count % 8:
        raise ValueError("n_episodes must be a multiple of 8 for complete pairs/four directions")
    groups = _pair_groups(available_scenarios)
    pairs_per_direction = episode_count // 8
    selected: list[str] = []
    for slot in range(pairs_per_direction):
        for direction_index in range(4):
            direction = BASE_DRIVING_DIRECTIONS[(direction_index + completed) % 4]
            pairs = groups[direction]
            startup_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for candidate in pairs:
                startup_groups[str(scenario_axes(candidate[0])["startup_mode"])].append(candidate)
            startups = sorted(startup_groups)
            startup = startups[(slot + completed) % len(startups)]
            candidates = startup_groups[startup]
            pair = candidates[(completed + slot // len(startups)) % len(candidates)]
            # Alternate pair order without breaking a complete pair at a checkpoint.
            selected.extend(pair if (completed + slot) % 2 == 0 else reversed(pair))
    return selected


def base_driving_support_fingerprint(available_scenarios: Sequence[str]) -> str:
    _pair_groups(available_scenarios)
    return hashlib.sha256("\n".join(sorted(available_scenarios)).encode("utf-8")).hexdigest()


def base_driving_sampler_state(
    completed_update: int, available_scenarios: Sequence[str]
) -> dict[str, object]:
    return {
        "protocol_version": BASE_DRIVING_SAMPLING_PROTOCOL,
        "completed_update": _integer(completed_update, "completed_update"),
        "support_sha256": base_driving_support_fingerprint(available_scenarios),
    }


def validate_base_driving_sampler_state(
    state: object, *, expected_completed_update: int, available_scenarios: Sequence[str]
) -> None:
    expected = base_driving_sampler_state(expected_completed_update, available_scenarios)
    if state != expected:
        raise ValueError(f"base-driving sampler state mismatch: {state!r} != {expected!r}")


def load_base_driving_manifest(scenario_dir: Path, *, verify_hashes: bool = True) -> dict:
    """Load the pool contract and reject unexpected files or changed scenarios."""
    root = Path(scenario_dir)
    manifest = json.loads((root / "protocol" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pool_protocol") != BASE_DRIVING_POOL_PROTOCOL:
        raise ValueError("unrecognized base-driving scenario protocol")
    hashes: Mapping[str, str] = manifest["scenario_sha256"]
    if sorted(path.name for path in root.glob("*.json")) != sorted(hashes):
        raise ValueError("base-driving scenario file set differs from manifest")
    if verify_hashes:
        for name, expected in hashes.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"base-driving scenario hash mismatch: {name}")
    _pair_groups(list(hashes))
    return manifest
