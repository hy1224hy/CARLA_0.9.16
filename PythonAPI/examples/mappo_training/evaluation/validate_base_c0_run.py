"""Independently validate a completed P0-BASE C0 evidence directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from evaluation.base_c0_protocol import (
    C0_MANIFEST_VERSION,
    C0_RESOLVED_CONFIG_VERSION,
    C0_VALIDATOR_REPORT_VERSION,
    C0ProtocolError,
    build_summary,
    compare_repeatability_records,
    ensure_completed_prefix,
    expand_profile_cases,
    load_and_validate_c0_checkpoint,
    load_c0_contract,
    load_strict_json,
    parse_eval_id,
    read_jsonl,
    read_summary_csv,
    sha256_file,
    summary_csv_rows,
    validate_episode_record,
)
from training.checkpoint_protocol import atomic_write_json


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _require_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise C0ProtocolError(f"{context} mismatch: actual={actual!r}, expected={expected!r}")


def _validate_impl(
    output_dir: Path,
    *,
    expected_contract: Path | None,
    expected_checkpoint: Path | None,
    expected_profile: str | None,
    expected_eval_id: str | None,
) -> Dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise C0ProtocolError(f"C0 output directory not found: {output_dir}")

    preliminary_manifest_path = output_dir / "evaluation_manifest.json"
    preliminary_resolved_path = output_dir / "resolved_eval_config.json"
    if not preliminary_manifest_path.is_file() or not preliminary_resolved_path.is_file():
        raise C0ProtocolError("C0 manifest or resolved config is missing")
    manifest = load_strict_json(preliminary_manifest_path)
    resolved = load_strict_json(preliminary_resolved_path)
    expected_manifest_fields = {
        "manifest_version",
        "eval_id",
        "profile",
        "contract_sha256",
        "resolved_config_sha256",
        "checkpoint_sha256",
        "checkpoint_update",
        "formal_eligible",
        "status",
        "completed_cases",
        "total_cases",
        "created_at",
        "updated_at",
        "post_cleanup",
        "cleanup_error",
        "error",
        "episodes_sha256",
        "summary_sha256",
        "summary_csv_sha256",
    }
    expected_resolved_fields = {
        "resolved_config_version",
        "created_at",
        "eval_id",
        "profile",
        "contract_id",
        "contract_path",
        "contract_sha256",
        "contract_canonical_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_update",
        "checkpoint_formal_eligible",
        "checkpoint_formal_ineligibility_reasons",
        "host",
        "port",
        "device",
        "command",
        "runtime",
        "case_count",
        "cases",
        "git",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_fields:
        raise C0ProtocolError("completed manifest schema mismatch")
    if not isinstance(resolved, Mapping) or set(resolved) != expected_resolved_fields:
        raise C0ProtocolError("resolved config schema mismatch")
    if manifest.get("manifest_version") != C0_MANIFEST_VERSION:
        raise C0ProtocolError("manifest version mismatch")
    if resolved.get("resolved_config_version") != C0_RESOLVED_CONFIG_VERSION:
        raise C0ProtocolError("resolved config version mismatch")

    contract_path = expected_contract or Path(str(resolved.get("contract_path", "")))
    contract_record = load_c0_contract(contract_path, project_dir=PROJECT_DIR)
    contract = contract_record["payload"]
    outputs = contract["outputs"]
    manifest_path = output_dir / outputs["manifest"]
    resolved_path = output_dir / outputs["resolved_config"]
    episodes_path = output_dir / outputs["episodes_jsonl"]
    summary_path = output_dir / outputs["summary_json"]
    summary_csv_path = output_dir / outputs["summary_csv"]
    if manifest_path != preliminary_manifest_path or resolved_path != preliminary_resolved_path:
        raise C0ProtocolError("frozen output filenames differ from the validator bootstrap names")
    for path in (episodes_path, summary_path, summary_csv_path):
        if not path.is_file():
            raise C0ProtocolError(f"required C0 artifact is missing: {path.name}")

    eval_id = str(resolved.get("eval_id", ""))
    profile = str(resolved.get("profile", ""))
    checkpoint_sha256 = str(resolved.get("checkpoint_sha256", ""))
    checkpoint_update = resolved.get("checkpoint_update")
    if isinstance(checkpoint_update, bool) or not isinstance(checkpoint_update, int):
        raise C0ProtocolError("resolved checkpoint_update must be an integer")
    parse_eval_id(eval_id, profile=profile, checkpoint_update=checkpoint_update)
    if expected_eval_id is not None:
        _require_equal(eval_id, expected_eval_id, context="expected eval_id")
    if expected_profile is not None:
        _require_equal(profile, expected_profile, context="expected profile")
    _require_equal(output_dir.name, eval_id, context="output directory identity")
    for document, name in ((manifest, "manifest"), (resolved, "resolved config")):
        _require_equal(document.get("eval_id"), eval_id, context=f"{name}.eval_id")
        _require_equal(document.get("profile"), profile, context=f"{name}.profile")
        _require_equal(document.get("contract_sha256"), contract_record["sha256"], context=f"{name}.contract_sha256")
        _require_equal(document.get("checkpoint_sha256"), checkpoint_sha256, context=f"{name}.checkpoint_sha256")
    _require_equal(manifest.get("resolved_config_sha256"), sha256_file(resolved_path), context="resolved config SHA256")
    _require_equal(resolved.get("contract_id"), contract["contract_id"], context="contract id")
    _require_equal(
        resolved.get("contract_canonical_sha256"),
        contract_record["canonical_sha256"],
        context="contract canonical SHA256",
    )
    _require_equal(manifest.get("checkpoint_update"), checkpoint_update, context="checkpoint update")
    if not str(resolved.get("command", "")).strip():
        raise C0ProtocolError("resolved command cannot be empty")
    runtime = resolved.get("runtime")
    if not isinstance(runtime, Mapping):
        raise C0ProtocolError("resolved runtime must be an object")
    expected_carla_version = str(contract["environment"]["carla_version"])
    for name in ("carla_client_version", "carla_server_version"):
        version = str(runtime.get(name, ""))
        if not version.startswith(expected_carla_version):
            raise C0ProtocolError(
                f"resolved runtime {name}={version!r} differs from {expected_carla_version!r}"
            )

    source_checkpoint = expected_checkpoint or Path(str(resolved.get("checkpoint_path", "")))
    if not source_checkpoint.is_file():
        raise C0ProtocolError(f"source checkpoint is unavailable: {source_checkpoint}")
    checkpoint_record = load_and_validate_c0_checkpoint(
        source_checkpoint,
        contract=contract,
        allow_diagnostic_checkpoint=True,
        map_location="cpu",
    )
    _require_equal(checkpoint_record["sha256"], checkpoint_sha256, context="source checkpoint SHA256")
    _require_equal(checkpoint_record["update"], checkpoint_update, context="source checkpoint update")

    cases = expand_profile_cases(contract, profile, project_dir=PROJECT_DIR)
    resolved_cases = resolved.get("cases")
    if not isinstance(resolved_cases, list):
        raise C0ProtocolError("resolved cases must be a list")
    expected_resolved_cases = [
        {key: value for key, value in case.items() if key != "scenario_config"}
        for case in cases
    ]
    _require_equal(resolved_cases, expected_resolved_cases, context="resolved frozen cases")
    _require_equal(resolved.get("case_count"), len(cases), context="resolved case count")

    records = read_jsonl(episodes_path)
    ensure_completed_prefix(records, cases)
    if len(records) != len(cases):
        raise C0ProtocolError(
            f"C0 case set is incomplete: completed={len(records)}, expected={len(cases)}"
        )
    for index, record in enumerate(records):
        validate_episode_record(
            record,
            eval_id=eval_id,
            profile=profile,
            checkpoint_sha256=checkpoint_sha256,
            expected_case=cases[index],
        )
        expected_outcome = None
        reason = record["episode_end_reason"]
        for name in ("success", "collision", "timeout", "other"):
            if reason in contract["outcomes"][f"{name}_reasons"]:
                expected_outcome = name
        _require_equal(record["outcome"], expected_outcome, context=f"case {record['case_id']} outcome")

    _require_equal(manifest.get("completed_cases"), len(cases), context="manifest completed cases")
    _require_equal(manifest.get("total_cases"), len(cases), context="manifest total cases")
    _require_equal(manifest.get("episodes_sha256"), sha256_file(episodes_path), context="episodes SHA256")
    if manifest.get("cleanup_error"):
        raise C0ProtocolError(f"cleanup error is not empty: {manifest['cleanup_error']}")
    cleanup = manifest.get("post_cleanup")
    if not isinstance(cleanup, Mapping):
        raise C0ProtocolError("post_cleanup audit is missing")
    if (
        cleanup.get("vehicle_actor_count") != 0
        or cleanup.get("sensor_actor_count") != 0
        or cleanup.get("world_timing_restored") is not True
    ):
        raise C0ProtocolError("post-cleanup CARLA audit failed")

    formal_eligible = bool(resolved.get("checkpoint_formal_eligible"))
    formal_reasons = resolved.get("checkpoint_formal_ineligibility_reasons")
    if not isinstance(formal_reasons, list):
        raise C0ProtocolError("checkpoint formal ineligibility reasons must be a list")
    if formal_eligible == bool(formal_reasons):
        raise C0ProtocolError("checkpoint formal eligibility and reasons are inconsistent")
    _require_equal(
        checkpoint_record["formal_eligible"],
        formal_eligible,
        context="checkpoint formal eligibility provenance",
    )
    _require_equal(
        checkpoint_record["formal_ineligibility_reasons"],
        formal_reasons,
        context="checkpoint formal ineligibility provenance",
    )
    _require_equal(manifest.get("formal_eligible"), formal_eligible, context="formal eligibility")
    expected_status = "PASS" if formal_eligible else "DIAGNOSTIC_PASS"
    _require_equal(manifest.get("status"), expected_status, context="manifest completion status")
    if not formal_eligible and profile != "smoke":
        raise C0ProtocolError("diagnostic checkpoint evidence is only valid for smoke")

    expected_summary = build_summary(
        records,
        eval_id=eval_id,
        profile=profile,
        checkpoint_sha256=checkpoint_sha256,
    )
    actual_summary = load_strict_json(summary_path)
    _require_equal(actual_summary, expected_summary, context="summary reconstructed from episodes")
    _require_equal(manifest.get("summary_sha256"), sha256_file(summary_path), context="summary SHA256")
    expected_csv = summary_csv_rows(expected_summary)
    actual_csv = read_summary_csv(summary_csv_path)
    _require_equal(actual_csv, expected_csv, context="summary CSV reconstructed from episodes")
    _require_equal(manifest.get("summary_csv_sha256"), sha256_file(summary_csv_path), context="summary CSV SHA256")

    return {
        "status": "PASS",
        "validator_report_version": C0_VALIDATOR_REPORT_VERSION,
        "eval_id": eval_id,
        "profile": profile,
        "formal_eligible": formal_eligible,
        "contract_sha256": contract_record["sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "case_count": len(records),
        "episodes_sha256": sha256_file(episodes_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_csv_sha256": sha256_file(summary_csv_path),
        "checks": {
            "identity_and_hashes": "PASS",
            "frozen_case_set_and_order": "PASS",
            "episode_schema_and_reward_conservation": "PASS",
            "summary_reconstruction": "PASS",
            "cleanup_audit": "PASS",
        },
        "errors": [],
    }


def validate_c0_run(
    output_dir: str | Path,
    *,
    expected_contract: Path | None = None,
    expected_checkpoint: Path | None = None,
    expected_profile: str | None = None,
    expected_eval_id: str | None = None,
    write_report: bool = True,
) -> Dict[str, Any]:
    resolved_output = Path(output_dir).expanduser().resolve()
    try:
        report = _validate_impl(
            resolved_output,
            expected_contract=expected_contract,
            expected_checkpoint=expected_checkpoint,
            expected_profile=expected_profile,
            expected_eval_id=expected_eval_id,
        )
    except Exception as error:
        report = {
            "status": "HOLD",
            "validator_report_version": C0_VALIDATOR_REPORT_VERSION,
            "eval_id": expected_eval_id or resolved_output.name,
            "profile": expected_profile or "",
            "formal_eligible": False,
            "contract_sha256": "",
            "checkpoint_sha256": "",
            "case_count": 0,
            "checks": {},
            "errors": [f"{type(error).__name__}: {error}"],
        }
    if write_report:
        report_name = "validator_report.json"
        try:
            resolved_config = load_strict_json(resolved_output / "resolved_eval_config.json")
            contract = load_c0_contract(
                expected_contract or Path(resolved_config["contract_path"]),
                project_dir=PROJECT_DIR,
            )["payload"]
            report_name = contract["outputs"]["validator_report"]
        except Exception:
            pass
        if resolved_output.is_dir():
            atomic_write_json(report, resolved_output / report_name)
    return report


def compare_c0_runs(left_output_dir: str | Path, right_output_dir: str | Path) -> Dict[str, Any]:
    """Validate two complete runs, then compare their frozen discrete outcomes."""
    left_dir = Path(left_output_dir).expanduser().resolve()
    right_dir = Path(right_output_dir).expanduser().resolve()
    if left_dir == right_dir:
        raise C0ProtocolError("repeatability comparison requires two distinct output directories")
    left_report = validate_c0_run(left_dir, write_report=False)
    right_report = validate_c0_run(right_dir, write_report=False)
    if left_report["status"] != "PASS" or right_report["status"] != "PASS":
        raise C0ProtocolError(
            f"both repeatability inputs must validate: left={left_report['status']}, "
            f"right={right_report['status']}"
        )
    left_resolved = load_strict_json(left_dir / "resolved_eval_config.json")
    right_resolved = load_strict_json(right_dir / "resolved_eval_config.json")
    for key in ("contract_sha256", "checkpoint_sha256", "profile"):
        _require_equal(left_resolved.get(key), right_resolved.get(key), context=f"repeatability {key}")
    contract_record = load_c0_contract(Path(left_resolved["contract_path"]), project_dir=PROJECT_DIR)
    episodes_name = contract_record["payload"]["outputs"]["episodes_jsonl"]
    comparison = compare_repeatability_records(
        read_jsonl(left_dir / episodes_name),
        read_jsonl(right_dir / episodes_name),
        contract=contract_record["payload"],
    )
    comparison.update(
        {
            "left_eval_id": left_resolved["eval_id"],
            "right_eval_id": right_resolved["eval_id"],
            "contract_sha256": left_resolved["contract_sha256"],
            "checkpoint_sha256": left_resolved["checkpoint_sha256"],
        }
    )
    return comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a completed P0-BASE C0 evidence directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-contract", type=Path)
    parser.add_argument("--expected-checkpoint", type=Path)
    parser.add_argument("--expected-profile", choices=("smoke", "gate", "validation"))
    parser.add_argument("--expected-eval-id")
    parser.add_argument(
        "--repeat-output-dir",
        type=Path,
        help="optional second independently completed run for discrete repeatability audit",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = validate_c0_run(
        args.output_dir,
        expected_contract=args.expected_contract,
        expected_checkpoint=args.expected_checkpoint,
        expected_profile=args.expected_profile,
        expected_eval_id=args.expected_eval_id,
        write_report=True,
    )
    if report["status"] == "PASS" and args.repeat_output_dir is not None:
        try:
            report["repeatability"] = compare_c0_runs(
                args.output_dir, args.repeat_output_dir
            )
            if report["repeatability"]["status"] != "PASS":
                report["status"] = "HOLD"
                report["errors"].append("discrete repeatability comparison failed")
        except Exception as error:
            report["status"] = "HOLD"
            report["errors"].append(f"repeatability: {type(error).__name__}: {error}")
        atomic_write_json(report, Path(args.output_dir).resolve() / "validator_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
