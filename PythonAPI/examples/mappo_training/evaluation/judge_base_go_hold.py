"""Offline evidence-chain judge for the frozen P0-BASE-04 thresholds."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

from evaluation.base_c0_protocol import (
    load_and_validate_c0_checkpoint,
    load_c0_contract,
    load_strict_json,
    read_jsonl,
)
from evaluation.base_go_hold_protocol import (
    GO_HOLD_REPORT_VERSION,
    GoHoldProtocolError,
    evaluate_milestone_decision,
    evaluate_training_health,
    evaluate_validation_decision,
    load_go_hold_contract,
    summarize_c0_gate,
)
from evaluation.validate_base_c0_run import compare_c0_runs, validate_c0_run
from training.checkpoint_protocol import atomic_write_json, sha256_file, utc_now_iso
from training.experiment_contract import (
    EXPERIMENT_MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    load_experiment_contract,
    verify_critical_sources_clean,
)
from utils.validate_tensorboard_run import validate_metrics_run


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DECISION_ID_RE = re.compile(
    r"^P0_BASE_(GO_HOLD|VALIDATION)_u([0-9]{6})_([0-9]{8}_[0-9]{6})$"
)


def _run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise GoHoldProtocolError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _verify_preregistration(
    *, contract_path: Path, run_origin_git_sha: str, critical_paths: list[str]
) -> Dict[str, Any]:
    verify_critical_sources_clean(
        repository_root=REPOSITORY_ROOT,
        project_dir=PROJECT_DIR,
        relative_paths=critical_paths,
    )
    try:
        repository_relative = contract_path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise GoHoldProtocolError("GO/HOLD contract is outside the repository") from error
    threshold_commit = _run_git("log", "-1", "--format=%H", "--", repository_relative)
    if not re.fullmatch(r"[0-9a-f]{40}", threshold_commit):
        raise GoHoldProtocolError("GO/HOLD contract has no committed Git provenance")
    if not re.fullmatch(r"[0-9a-f]{40}", run_origin_git_sha):
        raise GoHoldProtocolError("formal run origin Git SHA is invalid")
    pinned_blobs: Dict[str, str] = {}
    for relative in critical_paths:
        absolute = (PROJECT_DIR / relative).resolve()
        try:
            repo_relative = absolute.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError as error:
            raise GoHoldProtocolError(
                f"GO/HOLD critical source is outside the repository: {absolute}"
            ) from error
        current_blob = _run_git("rev-parse", f"HEAD:{repo_relative}")
        try:
            origin_blob = _run_git("rev-parse", f"{run_origin_git_sha}:{repo_relative}")
        except GoHoldProtocolError as error:
            raise GoHoldProtocolError(
                f"GO/HOLD critical source did not exist at run origin: {relative}"
            ) from error
        if current_blob != origin_blob:
            raise GoHoldProtocolError(
                f"GO/HOLD critical source differs from the version pinned by run origin: {relative}"
            )
        pinned_blobs[relative] = current_blob
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", threshold_commit, run_origin_git_sha],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if ancestor.returncode != 0:
        raise GoHoldProtocolError(
            "GO/HOLD threshold commit is not an ancestor of the formal run origin; "
            "the thresholds were not demonstrably pre-registered"
        )
    return {
        "status": "PASS",
        "threshold_contract_git_commit": threshold_commit,
        "run_origin_git_commit": run_origin_git_sha,
        "threshold_commit_is_ancestor": True,
        "critical_source_blobs_match_run_origin": True,
        "critical_source_git_blobs": pinned_blobs,
        "critical_sources_clean": True,
    }


def _validate_training_run(
    *, run_dir: Path, expected_update: int, go_hold: Mapping[str, Any]
) -> Dict[str, Any]:
    resolved_path = run_dir / RESOLVED_CONFIG_NAME
    manifest_path = run_dir / EXPERIMENT_MANIFEST_NAME
    resolved = load_strict_json(resolved_path)
    manifest = load_strict_json(manifest_path)
    expected_resolved_fields = {
        "evidence_schema_version",
        "created_at",
        "run_id",
        "run_dir",
        "contract_id",
        "contract_path",
        "contract_sha256",
        "contract_canonical_sha256",
        "requested_stop_update",
        "planned_total_updates",
        "resolved_training_config",
        "resolved_scenario_dir",
        "scenario_pool_fingerprint",
        "git",
        "code_fingerprint",
        "health_stop_rules",
        "evaluation_contract",
    }
    expected_manifest_fields = {
        "evidence_schema_version",
        "run_id",
        "contract_id",
        "contract_sha256",
        "resolved_config_sha256",
        "requested_stop_update",
        "completed_update",
        "status",
        "created_at",
        "updated_at",
        "notes",
    }
    if not isinstance(resolved, Mapping) or set(resolved) != expected_resolved_fields:
        raise GoHoldProtocolError("formal training resolved config schema mismatch")
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_fields:
        raise GoHoldProtocolError("formal training manifest schema mismatch")
    if resolved["evidence_schema_version"] != "p0_base_resolved_config_v1":
        raise GoHoldProtocolError("training resolved config version mismatch")
    if manifest["evidence_schema_version"] != "p0_base_experiment_manifest_v1":
        raise GoHoldProtocolError("training manifest version mismatch")
    if run_dir.name != resolved["run_id"] or manifest["run_id"] != resolved["run_id"]:
        raise GoHoldProtocolError("training run identity mismatch")
    binding = go_hold["bindings"]
    for document, label in ((resolved, "resolved"), (manifest, "manifest")):
        if document["contract_id"] != binding["training_contract_id"]:
            raise GoHoldProtocolError(f"{label} training contract ID mismatch")
        if document["contract_sha256"] != binding["training_contract_sha256"]:
            raise GoHoldProtocolError(f"{label} training contract SHA256 mismatch")
    if manifest["resolved_config_sha256"] != sha256_file(resolved_path):
        raise GoHoldProtocolError("training resolved config SHA256 mismatch")
    if manifest["status"] != "SEGMENT_COMPLETED":
        raise GoHoldProtocolError(
            f"training manifest must be SEGMENT_COMPLETED, got {manifest['status']!r}"
        )
    if manifest["completed_update"] != expected_update:
        raise GoHoldProtocolError("training completed update differs from decision update")
    if manifest["requested_stop_update"] != expected_update:
        raise GoHoldProtocolError("training requested stop differs from decision update")
    training_config = resolved["resolved_training_config"]
    if not isinstance(training_config, Mapping) or training_config.get("seed") != binding["pilot_seed"]:
        raise GoHoldProtocolError("formal training seed differs from the pilot binding")
    contract_record = load_experiment_contract(
        Path(str(resolved["contract_path"])), project_dir=PROJECT_DIR
    )
    if contract_record["sha256"] != binding["training_contract_sha256"]:
        raise GoHoldProtocolError("training contract file no longer matches the binding")
    git = resolved["git"]
    if not isinstance(git, Mapping):
        raise GoHoldProtocolError("training Git provenance is missing")
    return {
        "resolved": resolved,
        "manifest": manifest,
        "run_id": str(resolved["run_id"]),
        "run_origin_git_sha": str(git.get("commit", "")),
        "resolved_config_sha256": sha256_file(resolved_path),
    }


def _decision_id(output_dir: Path, *, mode: str, expected_update: int) -> str:
    match = DECISION_ID_RE.fullmatch(output_dir.name)
    if not match:
        raise GoHoldProtocolError(
            "output basename must match P0_BASE_GO_HOLD_u<6digits>_YYYYMMDD_HHMMSS "
            "or P0_BASE_VALIDATION_u<6digits>_YYYYMMDD_HHMMSS"
        )
    expected_kind = "GO_HOLD" if mode == "milestone" else "VALIDATION"
    if match.group(1) != expected_kind or int(match.group(2)) != expected_update:
        raise GoHoldProtocolError("decision directory mode/update identity mismatch")
    return output_dir.name


def judge(
    *,
    mode: str,
    contract_path: Path,
    run_dir: Path,
    checkpoint_path: Path,
    c0_output_dir: Path,
    c0_repeat_output_dir: Path,
    output_dir: Path,
    expected_update: int,
) -> Dict[str, Any]:
    if mode not in {"milestone", "validation"}:
        raise GoHoldProtocolError(f"unsupported decision mode={mode!r}")
    contract_record = load_go_hold_contract(contract_path, project_dir=PROJECT_DIR)
    contract = contract_record["payload"]
    output_dir = output_dir.expanduser().resolve()
    decision_id = _decision_id(output_dir, mode=mode, expected_update=expected_update)
    decision_root = (PROJECT_DIR / contract["outputs"]["decision_root"]).resolve()
    try:
        output_dir.relative_to(decision_root)
    except ValueError as error:
        raise GoHoldProtocolError(
            f"decision output must stay under {decision_root}"
        ) from error
    report_path = output_dir / contract["outputs"]["decision_report"]
    if output_dir.exists() or report_path.exists():
        raise FileExistsError(f"immutable decision directory already exists: {output_dir}")

    run_dir = run_dir.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    training = _validate_training_run(
        run_dir=run_dir, expected_update=expected_update, go_hold=contract
    )
    preregistration = _verify_preregistration(
        contract_path=Path(contract_record["path"]),
        run_origin_git_sha=training["run_origin_git_sha"],
        critical_paths=list(contract["critical_source_paths"]),
    )

    c0_contract_path = (PROJECT_DIR / contract["bindings"]["c0_contract_path"]).resolve()
    c0_record = load_c0_contract(c0_contract_path, project_dir=PROJECT_DIR)
    checkpoint = load_and_validate_c0_checkpoint(
        checkpoint_path, contract=c0_record["payload"], map_location="cpu"
    )
    if checkpoint["update"] != expected_update or not checkpoint["formal_eligible"]:
        raise GoHoldProtocolError("checkpoint update/formal eligibility mismatch")
    metadata = checkpoint["checkpoint"].get("run_metadata", {})
    if metadata.get("run_id") != training["run_id"]:
        raise GoHoldProtocolError("checkpoint run_id differs from training evidence")
    if metadata.get("origin_git_sha") != training["run_origin_git_sha"]:
        raise GoHoldProtocolError("checkpoint origin Git SHA differs from training evidence")

    metrics_validation = validate_metrics_run(
        run_dir,
        expected_updates=list(range(1, expected_update + 1)),
        expected_episodes_per_update=contract["evidence_rules"]["expected_episodes_per_update"],
        require_ppo_updates=True,
        max_preupdate_ratio_error=contract["evidence_rules"][
            "preupdate_ratio_abs_error_max_exclusive"
        ],
    )
    metric_records = read_jsonl(run_dir / "metrics.jsonl")
    training_health = evaluate_training_health(metric_records, contract=contract)

    profile = (
        contract["bindings"]["milestone_profile"]
        if mode == "milestone"
        else contract["bindings"]["validation_profile"]
    )
    left_validation = validate_c0_run(
        c0_output_dir,
        expected_contract=c0_contract_path,
        expected_checkpoint=checkpoint_path,
        expected_profile=profile,
        write_report=False,
    )
    right_validation = validate_c0_run(
        c0_repeat_output_dir,
        expected_contract=c0_contract_path,
        expected_checkpoint=checkpoint_path,
        expected_profile=profile,
        write_report=False,
    )
    if left_validation["status"] != "PASS" or right_validation["status"] != "PASS":
        raise GoHoldProtocolError(
            f"C0 validators must PASS: left={left_validation['status']}, "
            f"right={right_validation['status']}"
        )
    repeatability = compare_c0_runs(c0_output_dir, c0_repeat_output_dir)
    repeatability_pass = repeatability.get("status") == "PASS"
    episodes_name = c0_record["payload"]["outputs"]["episodes_jsonl"]
    c0_records = read_jsonl(Path(c0_output_dir).resolve() / episodes_name)
    c0_summary = summarize_c0_gate(c0_records)
    if mode == "milestone":
        result = evaluate_milestone_decision(
            update=expected_update,
            training_health=training_health,
            c0_summary=c0_summary,
            repeatability_pass=repeatability_pass,
            contract=contract,
        )
    else:
        if expected_update != 100:
            raise GoHoldProtocolError("registered held-out validation decision is update 100 only")
        result = evaluate_validation_decision(
            training_health=training_health,
            c0_summary=c0_summary,
            repeatability_pass=repeatability_pass,
            contract=contract,
        )

    report = {
        "report_version": GO_HOLD_REPORT_VERSION,
        "created_at": utc_now_iso(),
        "decision_id": decision_id,
        "mode": mode,
        "expected_update": expected_update,
        "contract": {
            "id": contract["contract_id"],
            "path": contract_record["path"],
            "sha256": contract_record["sha256"],
            "canonical_sha256": contract_record["canonical_sha256"],
        },
        "training": {
            "run_dir": str(run_dir),
            "run_id": training["run_id"],
            "resolved_config_sha256": training["resolved_config_sha256"],
            "manifest_status": training["manifest"]["status"],
            "metrics_validation": metrics_validation,
            "health": training_health,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint["sha256"],
            "update": checkpoint["update"],
            "formal_eligible": checkpoint["formal_eligible"],
        },
        "c0": {
            "profile": profile,
            "primary_output_dir": str(Path(c0_output_dir).resolve()),
            "repeat_output_dir": str(Path(c0_repeat_output_dir).resolve()),
            "primary_validation": left_validation,
            "repeat_validation": right_validation,
            "repeatability": repeatability,
            "performance_sample_policy": "primary_only_repeat_is_reproducibility_audit",
            "summary": c0_summary,
        },
        "preregistration": preregistration,
        "result": result,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(report, report_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge a completed P0-BASE milestone using frozen GO/HOLD thresholds"
    )
    parser.add_argument("--mode", choices=("milestone", "validation"), default="milestone")
    parser.add_argument("--go-hold-contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--c0-output-dir", type=Path, required=True)
    parser.add_argument("--c0-repeat-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        report = judge(
            mode=args.mode,
            contract_path=args.go_hold_contract,
            run_dir=args.run_dir,
            checkpoint_path=args.checkpoint,
            c0_output_dir=args.c0_output_dir,
            c0_repeat_output_dir=args.c0_repeat_output_dir,
            output_dir=args.output_dir,
            expected_update=args.expected_update,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "report_version": GO_HOLD_REPORT_VERSION,
                    "status": "HOLD",
                    "error": f"{type(error).__name__}: {error}",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["result"]["decision"] in {
        "HOLD",
        "PILOT_NO_GO",
        "VALIDATION_REJECT",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
