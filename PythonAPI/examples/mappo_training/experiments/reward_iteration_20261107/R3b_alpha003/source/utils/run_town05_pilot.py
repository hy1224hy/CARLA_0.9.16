"""One-command Town05 pilot validation and one-update MAPPO smoke run.

CARLA must already be listening on port 2000. This launcher is isolated from
the Town02 scenario pool and deliberately never enables the legacy renderer.
Run it from ``PythonAPI/examples/mappo_training``::

    python -m utils.run_town05_pilot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import carla

from training.checkpoint_protocol import (
    LATEST_CHECKPOINT_NAME,
    load_checkpoint_file,
    validate_checkpoint_header,
)
from utils.validate_town05_parallel_scenarios import validate_pool


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SCENARIO_DIR = TRAINING_ROOT / "data" / "training_scenarios_town05_v1"
CONFIG_PATH = (
    TRAINING_ROOT
    / "configs"
    / "base_mappo"
    / "town05_parallel_pilot_smoke_v1.json"
)
XODR_PATH = REPO_ROOT / "CarlaUE4" / "Content" / "Carla" / "Maps" / "OpenDrive" / "Town05.xodr"
EXPECTED_REWARD_GAMMA = 0.997


def _run(command: list[str]) -> None:
    print("+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=TRAINING_ROOT, check=True)


def _validate_checkpoint_gamma(checkpoint_path: Path) -> None:
    checkpoint, digest = load_checkpoint_file(checkpoint_path, map_location="cpu")
    validate_checkpoint_header(checkpoint)
    training_gamma = float(checkpoint["training_config"]["gamma"])
    runtime = checkpoint["run_metadata"].get("runtime", {})
    environment_gamma = float(runtime.get("environment_reward_gamma", float("nan")))
    if training_gamma != EXPECTED_REWARD_GAMMA or environment_gamma != training_gamma:
        raise RuntimeError(
            "Town05 smoke reward discount mismatch: "
            f"expected={EXPECTED_REWARD_GAMMA}, training={training_gamma}, "
            f"environment={environment_gamma}"
        )
    print(
        "PASSED: checkpoint reward gamma "
        f"training=environment={training_gamma}; sha256={digest}",
        flush=True,
    )


def _validate_carla_cleanup(client: carla.Client) -> None:
    world = client.get_world()
    settings = world.get_settings()
    vehicle_count = len(world.get_actors().filter("vehicle.*"))
    sensor_count = len(world.get_actors().filter("sensor.*"))
    if (
        settings.synchronous_mode
        or settings.fixed_delta_seconds is not None
        or vehicle_count
        or sensor_count
    ):
        raise RuntimeError(
            "CARLA cleanup failed: "
            f"synchronous_mode={settings.synchronous_mode}, "
            f"fixed_delta_seconds={settings.fixed_delta_seconds}, "
            f"vehicles={vehicle_count}, sensors={sensor_count}"
        )
    print("PASSED: CARLA cleanup async/None and vehicle/sensor=0/0", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TRAINING_ROOT / "diagnostics" / "town05_parallel_smoke",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only validate OpenDRIVE geometry; do not connect to CARLA.",
    )
    parser.add_argument(
        "--skip-live-smoke",
        action="store_true",
        help="Skip the sixteen-scenario spawn check and run training directly.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors = validate_pool(
        SCENARIO_DIR,
        XODR_PATH,
        expected_count=16,
        expected_difficulty_count=4,
        expected_direction_count=4,
        expected_family_counts={
            "parallel_stationary": 4,
            "parallel_rolling": 4,
            "stagger_rear_closing": 4,
            "required_merge": 4,
        },
        require_validation_holdout=False,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASSED: offline Town05 geometry", flush=True)
    if args.offline_only:
        return 0

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    print(
        f"CARLA {client.get_server_version()} reachable at "
        f"{args.host}:{args.port}",
        flush=True,
    )

    if not args.skip_live_smoke:
        _run([
            sys.executable,
            "-B",
            "-m",
            "utils.smoke_training_scenarios_live",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--town",
            "Town05",
            "--scenario-dir",
            str(SCENARIO_DIR),
            "--position-tolerance",
            "0.25",
            "--speed-tolerance",
            "0.25",
            "--seed",
            str(args.seed),
        ])

    run_id = "T05_parallel_smoke_s{}_{}".format(
        args.seed,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    save_dir = args.output_root.resolve() / run_id
    _run([
        sys.executable,
        "-B",
        "-m",
        "training.train_no_rllib_mappo_robust_v2",
        "--run-id",
        run_id,
        "--config-path",
        str(CONFIG_PATH),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--town",
        "Town05",
        "--scenario-dir",
        str(SCENARIO_DIR),
        "--straight-road-length",
        "50",
        "--updates",
        str(args.updates),
        "--rollout-episodes",
        str(args.rollout_episodes),
        "--min-transitions-per-update",
        "64",
        "--max-steps",
        str(args.max_steps),
        "--ppo-epochs",
        "1",
        "--minibatch-size",
        "256",
        "--hidden-dim",
        "128",
        "--cpu",
        "--scenario-sampling-mode",
        "stratified",
        "--curriculum-mode",
        "metadata",
        "--warmup-updates",
        "0",
        "--full-scenario-updates",
        "1",
        "--curriculum-transition-window",
        "0",
        "--tb-histogram-interval",
        "0",
        "--save-interval",
        "1",
        "--log-interval",
        "1",
        "--reward-version",
        "v3",
        "--gamma",
        str(EXPECTED_REWARD_GAMMA),
        "--seed",
        str(args.seed),
        "--save-dir",
        str(save_dir),
    ])
    _run([
        sys.executable,
        "-B",
        "-m",
        "utils.validate_tensorboard_run",
        "--run-dir",
        str(save_dir),
        "--expected-updates",
        *[str(update) for update in range(1, args.updates + 1)],
        "--expected-episodes-per-update",
        str(args.rollout_episodes),
        "--require-ppo-updates",
        "--max-preupdate-ratio-error",
        "1e-6",
    ])
    _validate_checkpoint_gamma(save_dir / LATEST_CHECKPOINT_NAME)
    _validate_carla_cleanup(client)
    print(f"PASSED: Town05 training smoke -> {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
