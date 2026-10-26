"""Prepare weak-red MutMAPPO configurations from a separately uploaded model.

File-only operation: verify the expected upload SHA before registering the
checksum sidecar; never import CARLA, connect to it or launch training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training.checkpoint_protocol import (
    atomic_write_json, checksum_path, load_checkpoint_file, sha256_file,
)
from training.mut_mappo_config import ROOT, load_config, portable_config, validate_config


def prepare(checkpoint, output_dir, expected_sha256, *, device="cuda", seed=42,
            rounds=2, warmup_rounds=1, train_tick_budget=12000):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("configuration output exists; preserve it and choose a new directory")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    for name, value in (("rounds", rounds), ("warmup_rounds", warmup_rounds),
                        ("train_tick_budget", train_tick_budget)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected SHA256 must be the source model's 64-character digest")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ValueError(f"uploaded checkpoint SHA256 mismatch: expected={expected}, actual={actual}")
    # Existing sidecars are checked by the normal loader and never overwritten.
    sidecar = checksum_path(checkpoint)
    if not sidecar.exists():
        with sidecar.open("x", encoding="ascii") as stream:
            stream.write(actual + "\n")
    source, digest = load_checkpoint_file(checkpoint, map_location="cpu")
    try:
        portable_path = checkpoint.relative_to(ROOT).as_posix()
    except ValueError:
        portable_path = str(checkpoint)
    base = portable_config(load_config(ROOT / "configs/mut_mappo/local_candidate_v1.json"))
    base.update(device=device, seed=seed, research_status="candidate", algorithm="mappo", team_count=2)
    base["red_checkpoint"] = {"path": portable_path, "sha256": digest,
                              "capability_status": "weak_baseline_not_independently_qualified"}
    base["learning"]["hidden_dim"] = source.get("model_spec", {}).get("hidden_dim")
    base["run"].update(rounds=rounds, lr_horizon_rounds=max(100, rounds),
                       train_tick_budget=train_tick_budget, stop_rule="rounds", schedule_clock="rounds")
    base["mutual"].update(warmup_rounds=warmup_rounds, refresh_rounds=10, schedule_clock="rounds")
    configs, checks = {}, {}
    for group, mode in (("b0", "off"), ("b1", "plain"), ("b2", "selective")):
        config = copy.deepcopy(base)
        config["method"] = mode
        checks[group] = validate_config(config)["_root_validation"]
        configs[group] = config
    # The three configurations are all validated before any output is created.
    output.mkdir(parents=True, exist_ok=False)
    for group, config in configs.items():
        atomic_write_json(config, output / f"{group}.json")
    record = {"status": "PREPARED_NOT_TRAINED", "source_update": source.get("update"),
              "red_sha256": digest, "red_path": portable_path, "rounds": rounds,
              "train_tick_budget": train_tick_budget, "warmup_rounds": warmup_rounds,
              "scope": "engineering_smoke" if rounds == 2 and warmup_rounds == 1 else "candidate_pilot",
              "weak_red_qualified": False, "carla_connected": False, "training_launched": False,
              "configuration_checks": checks}
    atomic_write_json(record, output / "preparation.json")
    return {"status": record["status"], "directory": str(output), "red_sha256": digest,
            "source_update": source.get("update"), "scope": record["scope"],
            "groups": list(configs), "carla_connected": False, "training_launched": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--train-tick-budget", type=int, default=12000)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkpoint, args.output_dir, args.expected_sha256,
          device=args.device, seed=args.seed, rounds=args.rounds,
          warmup_rounds=args.warmup_rounds, train_tick_budget=args.train_tick_budget), indent=2))


if __name__ == "__main__":
    main()
