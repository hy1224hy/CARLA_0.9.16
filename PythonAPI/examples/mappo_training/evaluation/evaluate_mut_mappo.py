"""Serial, case-paired C0/C1/C2 evaluation for the common Mut-MAPPO runtime.

Run as a module from mappo_training.  Offline validation never creates a CARLA
client.  Live evaluation requires the user-started server and the shared port
owner lock.  Training-window statistics cannot be ingested as evaluation cases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import traceback
from typing import Any, Iterable, Mapping

from evaluation.mut_mappo_metrics import CONDITIONS, paired_report, summarize_cases
from environment.mut_mappo_efficiency_reward import TEAM_PROFILES

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCHEMA = "mut_mappo_evaluation_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else ROOT / value).resolve()


def _load_cases(config: Mapping[str, Any], split: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    source = config["case_manifest"]
    if isinstance(source, Mapping):
        source = source["path"]
    path = _resolve(source)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not manifest.get("schema_version"):
        raise ValueError("case manifest requires schema_version")
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise ValueError("case manifest requires a cases list")
    by_key = {}
    for row in rows:
        key = row.get("case_key")
        if not isinstance(key, str) or not key or key in by_key:
            raise ValueError("case manifest case_key must be nonempty and unique")
        by_key[key] = row
    split_keys = manifest.get("splits", config.get("splits", {})).get(split)
    if not isinstance(split_keys, list) or not split_keys:
        raise ValueError(f"manifest requires a nonempty {split} split")
    if len(set(split_keys)) != len(split_keys) or any(key not in by_key for key in split_keys):
        raise ValueError(f"{split} split contains duplicate or unknown keys")
    # No automatic scenario expansion and no repurposing test cases for selection.
    return path, manifest, [dict(by_key[key]) for key in split_keys]


def evaluate_records(records: Iterable[Mapping[str, Any]], output_dir: str | Path,
                     *, metadata: Mapping[str, Any] | None = None,
                     required_conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """Write external summary and a portable result index for already-run cases."""
    rows = list(records)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": EVALUATION_SCHEMA, "evidence": "independent_evaluation",
              "metadata": dict(metadata or {}), "attempts": summarize_cases(rows)}
    if {"C1", "C2"}.issubset(required_conditions):
        report["paired"] = paired_report(rows, required_conditions=required_conditions)
    else:
        report["paired"] = {"status": "NOT_COMPUTED", "reason": "C1 and C2 are both required"}
    _atomic_json(output / "summary.json", report)
    _atomic_json(output / "result_index.json", {
        "schema_version": "mut_mappo_result_index_v1", "updated_at": _now(),
        "entries": [{"type": "independent_evaluation", "status": report["metadata"].get("status", "UNVERIFIED"),
                     "split": report["metadata"].get("split"),
                     "config_sha256": report["metadata"].get("config_sha256"),
                     "checkpoint_sha256": report["metadata"].get("checkpoint_sha256"),
                     "case_manifest_sha256": report["metadata"].get("case_manifest_sha256"),
                     "manifest": "manifest.json", "records": "cases.jsonl", "summary": "summary.json",
                     "training_repeat_unit": "outer_seed", "teams_are_independent_seeds": False}],
    })
    return report


def preflight(config_path: str | Path, checkpoint_path: str | Path | None, *, split: str,
              team: str, conditions: tuple[str, ...]) -> dict[str, Any]:
    """Resolve identities and cases offline through the shared configuration API."""
    from training.train_mut_mappo import load_config, validate_config
    config_source = _resolve(config_path)
    checkpoint_source = _resolve(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_source is not None and not checkpoint_source.is_file():
        raise FileNotFoundError(checkpoint_source)
    config = validate_config(load_config(config_source), config_path=config_source)
    manifest_path, case_manifest, cases = _load_cases(config, split)
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("conditions must be a unique nonempty subset of C0/C1/C2")
    if "C2" in conditions and checkpoint_source is None:
        raise ValueError("C2 requires a complete Mut-MAPPO checkpoint; initialization is not a trained candidate")
    reference_hashes = {}
    if "C2" in conditions and config.get("task", {}).get("reward_profile") in TEAM_PROFILES:
        directory = _resolve(config["task"]["team_reward"]["reference_dir"])
        reference_hashes = {case["case_key"]: _sha256(directory / (hashlib.sha256(case["case_key"].encode()).hexdigest() + ".json"))
                            for case in cases}
    if team not in {"A", "B", "both"}:
        raise ValueError("team must be A, B, or both")
    # Model/state verification is owned by the common loader, so train and
    # evaluate never grow separate Actor constructors or checkpoint protocols.
    from training import train_mut_mappo
    validator = getattr(train_mut_mappo, "validate_checkpoint_for_config", None)
    checkpoint_validation = "NOT_REQUIRED_FOR_C0_C1"
    checkpoint_payload = None
    if checkpoint_source is not None:
        if validator:
            checkpoint_payload = validator(config, checkpoint_source)
            checkpoint_validation = "MODEL_STATE_VALIDATED"
        else:
            checkpoint_validation = "FILE_IDENTITY_ONLY_MODEL_STATE_NOT_LOADED"
    available_teams = (sorted(checkpoint_payload["teams"]) if checkpoint_payload and "teams" in checkpoint_payload
                       else [checkpoint_payload["team"]] if checkpoint_payload and "team" in checkpoint_payload
                       else ["A"] if config.get("team_count", 2) == 1 else ["A", "B"])
    if team != "both" and team not in available_teams:
        raise ValueError(f"requested team {team} is absent from the checkpoint")
    teams = available_teams if team == "both" else [team]
    checkpoint_spec = checkpoint_payload.get("model_spec", {}) if checkpoint_payload else {}
    if checkpoint_payload and "teams" in checkpoint_payload:
        checkpoint_spec = next(iter(checkpoint_payload["teams"].values())).get("model_spec", {})
    return {"config": config, "cases": cases, "checkpoint_payload": checkpoint_payload,
            "metadata": {"schema_version": EVALUATION_SCHEMA, "split": split,
                         "teams": teams, "conditions": list(conditions),
                         "config_path": str(config_source), "config_sha256": _sha256(config_source),
                         "checkpoint_path": str(checkpoint_source) if checkpoint_source else None,
                         "checkpoint_sha256": _sha256(checkpoint_source) if checkpoint_source else None,
                         "checkpoint_schema": checkpoint_payload.get("schema_version") if checkpoint_payload else None,
                         "checkpoint_completed_round": checkpoint_payload.get("completed_round") if checkpoint_payload else None,
                         "checkpoint_algorithm": checkpoint_spec.get("algorithm", config.get("algorithm", "mappo")),
                         "checkpoint_purpose": checkpoint_payload.get("purpose", "complete_training_boundary") if checkpoint_payload else None,
                         "case_manifest_path": str(manifest_path), "case_manifest_sha256": _sha256(manifest_path),
                         "case_manifest_schema": case_manifest["schema_version"],
                         "case_keys": [case["case_key"] for case in cases],
                         "planned_episodes": len(teams) * len(cases) * len(conditions),
                         "configuration_validation": "OFFLINE_VALIDATED",
                         "configuration_identity": config.get("_root_validation", {}),
                         "reference_curve_sha256": reference_hashes,
                         "checkpoint_validation": checkpoint_validation}}


def evaluate(config_path: str | Path, checkpoint_path: str | Path | None = None, *, team: str = "both",
             split: str = "selection", output_dir: str | Path | None = None,
             conditions: tuple[str, ...] = CONDITIONS, validate_only: bool = False,
             select_best: bool = False) -> dict[str, Any]:
    if select_best and (split != "selection" or not {"C1", "C2"}.issubset(conditions)):
        raise ValueError("--select-best requires independent selection-split C1/C2 paired evaluation")
    resolved = preflight(config_path, checkpoint_path, split=split, team=team, conditions=conditions)
    if select_best:
        from training.mut_mappo_checkpoint import SCHEMA_VERSION
        payload = resolved["checkpoint_payload"]
        if payload is None or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("--select-best requires a complete training boundary; frozen exports cannot be reselected")
        if type(payload.get("completed_round")) is not int or payload["completed_round"] < 1:
            raise ValueError("--select-best refuses round-zero initialization")
    config, cases, metadata = resolved["config"], resolved["cases"], resolved["metadata"]
    if validate_only:
        return {"status": "OFFLINE_VALIDATED", "carla_connected": False, **metadata}
    if output_dir is None:
        raise ValueError("output_dir is required for live evaluation")
    from training.train_mut_mappo import create_runtime, run_case
    from training.mut_mappo_config import portable_config
    from utils.mut_mappo_session import CarlaOwner
    output = _resolve(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    traces = output / "traces"
    traces.mkdir()
    _atomic_json(output / "resolved_config.json", portable_config(config))
    metadata.update(status="RUNNING", started_at=_now(), completed_attempted_episodes=0,
                    completed_case_groups=0, deterministic=True, selection_requested=select_best)
    _atomic_json(output / "manifest.json", metadata)
    records = []
    retries = int(config.get("evaluation", {}).get("max_invalid_retries", 1))
    if retries < 0:
        raise ValueError("max_invalid_retries must be nonnegative")
    try:
        with CarlaOwner(host=config.get("host", "127.0.0.1"), port=int(config.get("port", 2000)),
                        run_dir=output, purpose="evaluation"):
            with (output / "cases.jsonl").open("x", encoding="utf-8") as stream:
                for candidate in metadata["teams"]:
                    for case_index, case in enumerate(cases):
                        seed = int(case.get("seed", config.get("evaluation", {}).get("seed", 62001)))
                        for attempt in range(retries + 1):
                            group = []
                            for condition in conditions:
                                runtime = None
                                fatal_error = None
                                audited_ticks = {"total": 0, "decision": 0, "setup": 0}
                                def audit_tick(event):
                                    count = int(event["ticks"])
                                    phase = event["kind"]
                                    audited_ticks["total"] += count
                                    audited_ticks[phase] = audited_ticks.get(phase, 0) + count
                                context = {"case_key": case["case_key"], "condition": condition,
                                           "team": candidate, "paired_attempt": attempt, "seed": seed,
                                           "split": split, "policy_mode": "deterministic"}
                                # Filenames depend on indices, never untrusted case_key path text.
                                trace = traces / f"{candidate}_{case_index:05d}_{condition}_a{attempt}.jsonl"
                                row = dict(context, valid=False, invalid_reason="execution_interrupted")
                                try:
                                    try:
                                        checkpoint = resolved["checkpoint_payload"] or metadata["checkpoint_path"]
                                        runtime = create_runtime(config, checkpoint, condition=condition, team=candidate,
                                                                 audit_callback=audit_tick)
                                        row = dict(run_case(runtime, case, seed=seed, deterministic=True, trace_path=trace))
                                    except Exception:
                                        row = dict(context, valid=False, invalid_reason="environment_exception",
                                                   error=traceback.format_exc())
                                    except BaseException as error:
                                        row.update(error=traceback.format_exc())
                                        fatal_error = error
                                finally:
                                    if runtime is not None:
                                        try:
                                            runtime.close()
                                        except BaseException as error:
                                            row.update(valid=False, invalid_reason="cleanup_failure", cleanup_error=traceback.format_exc())
                                            fatal_error = error
                                for key in ("case_key", "condition", "team"):
                                    if key in row and row[key] != context[key]:
                                        raise ValueError(f"runtime changed evaluator identity {key}")
                                row.update(context, trace_path=str(trace.relative_to(output)), evidence="independent_evaluation",
                                           family=case.get("family"), tick_counts=audited_ticks)
                                summarize_cases([row])  # result/identity errors are fatal, never scored as task failures
                                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                stream.flush()
                                os.fsync(stream.fileno())
                                records.append(row)
                                group.append(row)
                                metadata["completed_attempted_episodes"] += 1
                                _atomic_json(output / "manifest.json", metadata)
                                print("CASE_DONE", candidate, case["case_key"], condition,
                                      "VALID" if row["valid"] else "INVALID", flush=True)
                                if fatal_error is not None:
                                    raise fatal_error
                            if all(row["valid"] for row in group):
                                break
                            # Entire same-key group is rerun at the same seed; no replacement of only a bad side.
                        metadata["completed_case_groups"] += 1
                        _atomic_json(output / "manifest.json", metadata)
        for field, digest in (("config_path", "config_sha256"), ("checkpoint_path", "checkpoint_sha256"),
                              ("case_manifest_path", "case_manifest_sha256")):
            if metadata[field] is not None and _sha256(Path(metadata[field])) != metadata[digest]:
                raise RuntimeError(f"{field} changed during evaluation")
        identity = config.get("_root_validation", {})
        source_red = config.get("red_checkpoint", {})
        if source_red and _sha256(_resolve(source_red["path"])) != identity["red_sha256"]:
            raise RuntimeError("frozen red checkpoint changed during evaluation")
        for scenario_path, digest in identity.get("scenario_hashes", {}).items():
            if _sha256(_resolve(scenario_path)) != digest:
                raise RuntimeError(f"scenario {scenario_path} changed during evaluation")
        metadata["status"] = "COMPLETED"
    except BaseException:
        metadata["status"] = "FAILED"
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        metadata["ended_at"] = _now()
        _atomic_json(output / "manifest.json", metadata)
        report = evaluate_records(records, output, metadata=metadata, required_conditions=conditions)
    if select_best:
        from evaluation.mut_mappo_selection import select_and_export
        report["selection"] = select_and_export([{
            "report": report, "checkpoint_payload": resolved["checkpoint_payload"],
            "checkpoint_sha256": metadata["checkpoint_sha256"], "evaluation_source": str(output),
        }], config, output)
        _atomic_json(output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="complete A/B checkpoint, required when evaluating C2")
    parser.add_argument("--team", choices=("A", "B", "both"), default="both")
    parser.add_argument("--split", choices=("selection", "test", "gate"), default="selection")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--select-best", action="store_true", help="selection split only: rank evaluated teams and export one frozen 20/2 blue policy")
    args = parser.parse_args()
    if not args.validate_only and not args.output_dir:
        parser.error("--output-dir is required for live evaluation")
    report = evaluate(args.config, args.checkpoint, team=args.team, split=args.split,
                      output_dir=args.output_dir, conditions=tuple(args.conditions), validate_only=args.validate_only,
                      select_best=args.select_best)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
