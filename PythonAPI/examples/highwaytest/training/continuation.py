"""Extend a trusted full boundary into a NEW run, preserving the parent evidence."""
import hashlib
import json
from pathlib import Path
import shutil
from ..config import ROOT, Config, Task
from .checkpoint import read, save, provenance, source_digest, validate_runtime
from .locking import run_lock
from .telemetry import committed_logs


def extend(checkpoint, output, total_steps, *, allow_source_change=False,
           max_rounds=None, preview=False, preview_speed=0.0):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    parent = checkpoint.parent
    if checkpoint.name != "latest.pt":
        raise ValueError("extension requires latest.pt with optimizers/RNG; selected.pt is for evaluation")
    if not output.is_relative_to((ROOT / "runs").resolve()) or output.is_relative_to(parent):
        raise ValueError("use a new run under highwaytest/runs, outside the parent run")
    if max_rounds is not None and (type(max_rounds) is not int or max_rounds < 1):
        raise ValueError("max_rounds must be a positive integer")
    with run_lock(output):
        with run_lock(parent):
            state = read(checkpoint)
            manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
            validate_runtime(manifest["packages"])
            old = state["config"]
            old_config = Config(**(old | {"task": Task(**old["task"])})).validate()
            if manifest["source_sha256"] != state["source_sha256"]:
                raise ValueError("parent source/checkpoint manifest mismatch")
            if manifest["config_sha256"] != old_config.digest():
                raise ValueError("parent config/checkpoint manifest mismatch")
            if type(total_steps) is not int or total_steps <= old_config.total_train_steps:
                raise ValueError("extension steps must be a larger TOTAL budget, not additional steps")
            active_source = source_digest()
            changed = state["source_sha256"] != active_source
            if changed and not allow_source_change:
                raise ValueError("parent uses older/different source; audit changes, then explicitly --allow-source-change")
            config = Config(**(old | {"total_train_steps": total_steps, "task": Task(**old["task"])})).validate()
            history = manifest.get("training_segments", [
                {"budget": old_config.total_train_steps, "source_sha256": state["source_sha256"]}])
            continuation = {"parent_checkpoint": str(checkpoint),
                            "parent_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            "parent_source_sha256": state["source_sha256"], "source_changed": changed,
                            "from_train_steps": state["train_steps"], "from_round": state["round"],
                            "prior_budget": old_config.total_train_steps, "new_total_budget": total_steps,
                            "optimizer_reset": False, "teacher_and_gate_reset": False, "rng_reset": False}
            # All checks precede creation. provenance rejects an existing target.
            provenance(output, config, manifest.get("initialization"))
            target_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            target_manifest.update(continuation=continuation,
                training_segments=history + [{"budget": total_steps, "source_sha256": active_source}])
            (output / "manifest.json").write_text(json.dumps(target_manifest, indent=2), encoding="utf-8")
            for name in ("metrics.jsonl", "episodes.jsonl", "validation.jsonl"):
                if (parent / name).exists():
                    shutil.copy2(parent / name, output / name)
            committed_logs(output, state["round"])
            state["config"] = config.to_dict()
            state["source_sha256"] = active_source
            save(output / "latest.pt", state)
        # Already hold the target OS lock. Restore through the same strict path
        # used for interrupted training, including TensorBoard reconstruction.
        from .runner import _run
        return _run(config, output, resume=output / "latest.pt", max_rounds=max_rounds,
                    preview=preview, preview_speed=preview_speed)
