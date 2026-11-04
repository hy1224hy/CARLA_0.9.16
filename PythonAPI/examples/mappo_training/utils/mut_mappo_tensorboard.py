"""TensorBoard projection of committed MutMAPPO JSONL, with no simulator calls.

JSONL/checkpoint manifests remain authoritative. Each process purges previous
event projections and replays committed JSONL before appending new boundaries.
This repairs incomplete event writes without duplicating resumed curves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import warnings


PROTOCOL = "mut_mappo_tensorboard_v1"


def add_tensorboard_arguments(parser):
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="disable TensorBoard events; keep authoritative JSONL")
    parser.add_argument("--tensorboard-dir",
                        help="event directory (default: RUN_DIR/tb; experiment runner adds RUN_ID)")


def options_from_args(args):
    return {"enabled": not getattr(args, "no_tensorboard", False),
            "log_dir": getattr(args, "tensorboard_dir", None)}


def _number(value):
    return isinstance(value, Real) and math.isfinite(float(value))


def scalar_projection(value, prefix=""):
    """Finite numbers only. Lists are explicitly aggregated, never time axes.

    PPO loss/KL lists contain epoch means; gradient lists contain minibatch
    measurements. Keep list elements as indexed diagnostics, plus mean/max.
    """
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            yield from scalar_projection(item, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        if value and all(_number(item) for item in value):
            numbers = [float(item) for item in value]
            yield prefix + "/mean", sum(numbers) / len(numbers)
            yield prefix + "/max", max(numbers)
            for index, item in enumerate(numbers, 1):
                yield f"{prefix}/item_{index}", item
    elif _number(value):
        yield prefix, float(value)


def committed_records(run_dir):
    """Read complete metrics bounded by the manifest, without loading a model."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8-sig"))
    boundary = manifest["completed_round"]
    previous = -1
    with (run_dir / "metrics.jsonl").open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                # A writer may be interrupted in the middle of its final line.
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid complete JSONL line {run_dir}:{line_number}") from error
            completed = record.get("completed_round", -1)
            if not record.get("teams") or not 0 < completed <= boundary:
                continue
            if completed <= previous:
                raise ValueError("committed metrics rounds must strictly increase")
            previous = completed
            yield record


class MutMAPPOTensorBoard:
    """Default-on metrics output; runtime I/O failure cannot stop training."""

    def __init__(self, run_dir, *, enabled=True, log_dir=None, writer_factory=None):
        self.run_dir = Path(run_dir)
        self.log_dir = Path(log_dir).resolve() if log_dir else self.run_dir / "tb"
        self.enabled = enabled
        self.writers = {}
        self.last_round = -1
        self.closed = False
        self.bound = False
        if enabled and writer_factory is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError("TensorBoard is unavailable: install tensorboard in the training environment, "
                                   "or pass --no-tensorboard") from error
            writer_factory = SummaryWriter
        self.writer_factory = writer_factory
        if enabled:
            self._check_source()

    def _source(self):
        manifest_path = self.run_dir / "checkpoint_manifest.json"
        identity = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if "identity" in manifest:
                identity = hashlib.sha256(json.dumps(manifest["identity"], sort_keys=True).encode()).hexdigest()
        return {"protocol": PROTOCOL, "run_dir": str(self.run_dir.resolve()), "identity_sha256": identity}

    def _check_source(self):
        path = self.log_dir / "projection_source.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != self._source():
                raise ValueError(f"TensorBoard directory belongs to another run/identity: {self.log_dir}; "
                                 "choose a different --tensorboard-dir/--output")
        elif self.log_dir.exists() and any(self.log_dir.rglob("events.out.tfevents.*")):
            raise ValueError(f"TensorBoard directory has events without a source binding: {self.log_dir}; "
                             "choose a new output directory")

    def _bind_source(self):
        if self.bound:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "projection_source.json"
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(self._source(), stream, sort_keys=True)
        except FileExistsError:
            self._check_source()
        self.bound = True

    def _writer(self, team):
        if team not in self.writers:
            self._bind_source()
            self.writers[team] = self.writer_factory(
                log_dir=str(self.log_dir / team), purge_step=0, flush_secs=5)
        return self.writers[team]

    def record(self, metrics):
        if not self.enabled or self.closed or not metrics.get("teams"):
            return
        completed = metrics["completed_round"]
        if completed <= self.last_round:
            return
        ticks = metrics["actual_counters"]["train"]
        if type(ticks) is not int or ticks < 0:
            raise ValueError("TensorBoard requires nonnegative actual training ticks")
        try:
            for team, optimizer in metrics["teams"].items():
                if team not in {"A", "B"}:
                    raise ValueError("unknown learner team")
                writer = self._writer(team)
                payload = {"ppo": optimizer,
                           "training_window": metrics.get("training_window", {}).get(team, {}),
                           "gate": metrics.get("gate", {}),
                           "cost": metrics.get("actual_counters", {}),
                           "meta": {"completed_round": completed,
                                    "train_tick_budget": metrics.get("train_tick_budget")}}
                for tag, value in scalar_projection(payload):
                    writer.add_scalar(tag, value, ticks)
                writer.flush()
            self.last_round = completed
        except Exception as error:
            warnings.warn(f"TensorBoard output disabled after an I/O/serialization error: {error}; "
                          "training and JSONL continue", RuntimeWarning)
            self.close()
            self.enabled = False

    def replay(self):
        if self.enabled:
            for record in committed_records(self.run_dir):
                self.record(record)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception as error:
                warnings.warn(f"TensorBoard close failed: {error}", RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", type=Path, help="experiment root containing runs/")
    source.add_argument("--run-dir", type=Path, help="single run containing metrics and manifest")
    parser.add_argument("--output", type=Path, required=True,
                        help="output root; each run gets its own subdirectory")
    args = parser.parse_args()
    runs = [args.run_dir] if args.run_dir else sorted(
        path.parent for path in (args.experiment / "runs").glob("*/checkpoint_manifest.json"))
    if not runs:
        parser.error("no committed runs found")
    for run in runs:
        recorder = MutMAPPOTensorBoard(run, log_dir=args.output / run.name)
        try:
            recorder.replay()
            if not recorder.enabled:
                raise RuntimeError("TensorBoard export failed; see the preceding warning")
            print(json.dumps({"run": str(run), "output": str(recorder.log_dir),
                              "last_round": recorder.last_round, "protocol": PROTOCOL}), flush=True)
        finally:
            recorder.close()


if __name__ == "__main__":
    main()
