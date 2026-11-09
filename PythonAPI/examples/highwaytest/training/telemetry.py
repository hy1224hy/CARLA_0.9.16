"""TensorBoard is a view of committed JSONL evidence, rebuilt on continuation."""
import json
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def committed_logs(output, completed_round):
    """Discard crash tails; malformed interior records remain an explicit error."""
    for name in ("metrics.jsonl", "episodes.jsonl", "validation.jsonl"):
        path = Path(output) / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise
                break
            if row["round"] <= completed_round:
                rows.append(row)
        temp = path.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        temp.replace(path)


class Telemetry:
    def __init__(self, output, config):
        self.output = Path(output)
        # Replaying authoritative committed records avoids orphan/duplicate curves
        # after termination between JSONL writes and the checkpoint transaction.
        self.writer = SummaryWriter(str(self.output / "tensorboard"), purge_step=0, flush_secs=10)
        self.writer.add_text("configuration", json.dumps(config.to_dict(), indent=2), 0)
        self.writer.add_text("protocol", "x-axis: A+B training policy decisions; validation counted separately; CPU training", 0)
        self.replay()

    def scalars(self, prefix, values, step):
        for key, value in values.items():
            if isinstance(value, (int, float, bool)) and np.isfinite(value):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def round(self, metrics, episodes, validation=None):
        for row in metrics:
            step = row["round_end_steps"]
            self.scalars(f"train/{row['team']}", {k: v for k, v in row.items()
                         if k not in ("round", "train_steps", "round_end_steps", "team")}, step)
            subset = [r for r in episodes if r["team"] == row["team"]]
            for family in [None]+sorted({r['family'] for r in subset}):
                selected = subset if family is None else [r for r in subset if r["family"] == family]
                if not selected:
                    continue
                prefix = f"rollout/{row['team']}" + (f"/{family}" if family else "")
                for key in ("return", "return_agent_0", "return_agent_1", "success", "bad", "collision",
                            "offroad", "episode_steps", "blue_progress_m", "red_delay_m", "qualified_event"):
                    self.writer.add_scalar(f"{prefix}/{key}", np.mean([r[key] for r in selected]), step)
                for key in ('blue_min_progress_m','blue_0_progress_m','blue_1_progress_m',
                            'blue_0_stopped_s','blue_1_stopped_s','blue_min_progress_retention'):
                    values=[r[key] for r in selected if r.get(key) is not None]
                    if values:
                        self.writer.add_scalar(f'{prefix}/{key}',np.mean(values),step)
        if validation:
            step = validation["train_steps"]
            self.scalars('curriculum', {k: validation[k] for k in ('attack_weight','curriculum_stage') if k in validation}, step)
            self.scalars("budget", {"validation_steps": validation["validation_steps"], "train_steps": step}, step)
            for split in ("gate", "selection"):
                for team, report in validation[split].items():
                    self.scalars(f"validation/{split}/{team}", report["overall"], step)
                    for family, values in report["families"].items():
                        self.scalars(f"validation/{split}/{team}/{family}", values, step)
            for team, families in validation["gates"].items():
                self.scalars(f"gate/{team}", families, step)
        self.writer.flush()

    def replay(self):
        def rows(name):
            path = self.output / name
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        metrics, episodes = rows("metrics.jsonl"), rows("episodes.jsonl")
        validations = {r["round"]: r for r in rows("validation.jsonl")}
        for number in sorted({r["round"] for r in metrics}):
            self.round([r for r in metrics if r["round"] == number],
                       [r for r in episodes if r["round"] == number], validations.get(number))

    def close(self):
        self.writer.close()
