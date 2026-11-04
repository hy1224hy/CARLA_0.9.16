"""Pure learning-rate schedules for MUT-MAPPO training.

The default linear schedule preserves the historical 0.1 floor and uses the
existing ``schedule_clock`` setting.  Reading a schedule never adds defaults to
the run configuration, so historical configuration hashes remain unchanged.
"""

from collections.abc import Mapping
import math


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _schedule_parameters(run):
    if not isinstance(run, Mapping):
        raise ValueError("run must be a mapping")
    schedule = run.get("lr_schedule", "linear")
    if not isinstance(schedule, str) or schedule not in {"linear", "constant"}:
        raise ValueError("run.lr_schedule must be linear or constant")
    clock = run.get("schedule_clock", "rounds")
    if not isinstance(clock, str) or clock not in {"rounds", "train_ticks"}:
        raise ValueError("run.schedule_clock must be rounds or train_ticks")

    if schedule == "constant":
        if any(key in run for key in ("lr_decay_train_ticks", "lr_min_factor")):
            raise ValueError("constant LR schedules cannot specify lr_decay_train_ticks or lr_min_factor")
        return schedule, clock, None, None

    minimum = run.get("lr_min_factor", 0.1)
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not 0 <= minimum <= 1 or not math.isfinite(minimum)):
        raise ValueError("run.lr_min_factor must be finite and in [0, 1]")
    if clock == "train_ticks":
        budget = _positive_int(run.get("train_tick_budget"), "run.train_tick_budget")
        horizon = _positive_int(run.get("lr_decay_train_ticks", budget), "run.lr_decay_train_ticks")
    else:
        if "lr_decay_train_ticks" in run:
            raise ValueError("run.lr_decay_train_ticks requires schedule_clock=train_ticks")
        horizon = _positive_int(run.get("lr_horizon_rounds"), "run.lr_horizon_rounds")
    return schedule, clock, horizon, float(minimum)


def validate_lr_schedule(run):
    """Raise ``ValueError`` for an invalid schedule without modifying ``run``."""
    _schedule_parameters(run)


def learning_rate_multiplier(run, train_ticks, round_no):
    """Return the current LR factor using training ticks or one-based rounds.

    A linear tick schedule defaults to the training budget, but an explicit
    decay horizon may extend beyond it.  Round schedules preserve the legacy
    ``round_no - 1`` convention.  Both remain at their floor after the horizon.
    """
    schedule, clock, horizon, minimum = _schedule_parameters(run)
    if schedule == "constant":
        return 1.0
    if clock == "train_ticks":
        if isinstance(train_ticks, bool) or not isinstance(train_ticks, int) or train_ticks < 0:
            raise ValueError("train_ticks must be a nonnegative integer")
        elapsed = train_ticks
    else:
        elapsed = _positive_int(round_no, "round_no") - 1
    progress = 1.0 if elapsed >= horizon else elapsed / horizon
    return max(minimum, 1.0 - progress)
