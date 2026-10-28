"""Select and export a frozen blue team using independent selection evidence.

The candidate safety threshold is a finite-sample screening rule, not a safety
guarantee. Selection requires every planned paired case to be valid for every
competing team; failures cannot silently reduce the denominator or remove a
competitor. Test/gate evidence and round-zero initialization are never used.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from evaluation.mut_mappo_metrics import paired_report
from training.checkpoint_protocol import atomic_write_json, save_checkpoint_file, sha256_file
from training.mut_mappo_learning import actor_states_hash

SELECTION_SCHEMA = "mut_mappo_selection_v1"
EXPORT_SCHEMA = "mut_mappo_selected_blue_team_v2"
EXPORT_PROTOCOL = "mut_mappo_selected_blue_20_2_v1"
RANKING = ["AttackGain_safe_desc", "q_desc", "task_violation_episodes_asc",
           "completed_round_asc", "team_id_asc", "checkpoint_sha256_asc"]


def _digest(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} requires a lowercase SHA256 digest")
    return value


def _count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _rate(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _candidate_rows(candidate, threshold):
    report, payload = candidate["report"], candidate["checkpoint_payload"]
    metadata = report.get("metadata", {})
    if report.get("evidence") != "independent_evaluation" or metadata.get("split") != "selection":
        raise ValueError("selection only accepts independent selection-split evidence; test/gate are forbidden")
    if metadata.get("status") != "COMPLETED":
        raise ValueError("selection requires a completed independent evaluation")
    completed = _count(payload.get("completed_round"), "completed_round")
    if completed == 0:
        raise ValueError("round-zero initialization is not a trained selection candidate")
    # Export has its own schema; accepting an old two-car checkpoint here would
    # silently turn an incompatible policy into an apparently new blue model.
    from training.mut_mappo_checkpoint import SCHEMA_VERSION, _validate_payload
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("selection requires a complete current-protocol training boundary")
    _validate_payload(payload)
    digest = _digest(candidate["checkpoint_sha256"], "source checkpoint")
    if metadata.get("checkpoint_sha256") != digest:
        raise ValueError("evaluation/checkpoint digest differs")
    identity = payload.get("identity", {})
    _digest(identity.get("red_sha256"), "frozen red")
    _digest(identity.get("case_manifest_sha256"), "case manifest")
    if identity.get("model_spec", {}).get("obs_dim") != 20 or identity.get("model_spec", {}).get("action_dim") != 2:
        raise ValueError("source identity is not the common 20/2 model protocol")
    if metadata.get("case_manifest_sha256") != identity.get("case_manifest_sha256"):
        raise ValueError("selection evidence belongs to a different case manifest")
    observed = metadata.get("configuration_identity", {})
    if observed.get("red_sha256") != identity.get("red_sha256"):
        raise ValueError("selection evidence belongs to a different frozen red policy")
    task_hash = identity.get("task_contract_sha256")
    if task_hash is not None and observed.get("task_contract_sha256") != task_hash:
        raise ValueError("selection evidence belongs to a different task/reward/lifecycle contract")
    keys = metadata.get("case_keys")
    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("selection requires a nonempty unique planned case list")
    teams = metadata.get("teams")
    if not isinstance(teams, list) or not teams or len(teams) != len(set(teams)) or any(t not in payload["teams"] for t in teams):
        raise ValueError("evaluation teams must exist in the source checkpoint")
    paired = report.get("paired", {})
    if set(paired.get("required_conditions", [])) != set(metadata.get("conditions", [])) or not {"C1", "C2"}.issubset(paired.get("required_conditions", [])):
        raise ValueError("selection requires the declared common C1/C2 paired conditions")
    if set(paired.get("teams", {})) != set(teams):
        raise ValueError("paired summary teams differ from planned teams")
    rows = []
    for team in sorted(teams):
        summary = paired["teams"][team]
        included = summary.get("included", [])
        included_keys = [row.get("case_key") for row in included]
        n = _count(summary.get("n_common_valid"), "n_common_valid")
        complete = (n == len(keys) and summary.get("n_cases") == len(keys)
                    and summary.get("n_excluded") == 0 and len(included_keys) == n
                    and len(set(included_keys)) == n and set(included_keys) == set(keys))
        base = {"team": team, "completed_round": completed, "checkpoint_sha256": digest,
                "n_planned": len(keys), "n_common_valid": n,
                "evaluation_source": candidate.get("evaluation_source"),
                "qualified": False, "evidence_complete": complete}
        if not complete:
            rows.append(dict(base, rejection="incomplete_common_paired_cases"))
            continue
        c2 = summary.get("conditions", {}).get("C2", {})
        if c2.get("n_valid") != n or c2.get("n_invalid") != 0:
            raise ValueError("selection C2 statistics must use the final common paired denominator")
        b_count, q_count = _count(c2.get("b_count"), "C2 b_count"), _count(c2.get("q_count"), "C2 q_count")
        if b_count > n or q_count > n - b_count:
            raise ValueError("selection q/b counts violate their common valid denominator")
        b, q = b_count / n, q_count / n
        if not math.isclose(_rate(c2.get("b"), "C2 b"), b) or not math.isclose(_rate(c2.get("q"), "C2 q"), q):
            raise ValueError("selection q/b rates differ from final paired counts")
        gain = _rate(summary.get("AttackGain_safe"), "AttackGain_safe")
        if not math.isclose(gain, sum(_rate(row.get("safe_contribution"), "safe_contribution") for row in included) / n):
            raise ValueError("AttackGain_safe differs from the final paired evidence")
        task = _count(c2.get("blue_violation_episode_counts", {}).get("task_violation", 0), "task_violation episodes")
        if task > n:
            raise ValueError("task-violation count exceeds the final paired denominator")
        rows.append(dict(base, b=b, q=q, AttackGain_safe=gain,
                         safe_efficiency_utility=c2.get("safe_efficiency_utility"),
                         task_violation_episodes=task, qualified=b <= threshold,
                         rejection=None if b <= threshold else "blue_bad_rate_above_candidate_threshold"))
    return rows


def select_and_export(candidates: Iterable[Mapping], config: Mapping, output_dir) -> dict:
    """Select across evaluated teams/rounds from one unchanged experiment.

    Each candidate contains report, checkpoint_payload, checkpoint_sha256 and
    optional evaluation_source. Do not combine different algorithms or seeds
    into a single deployment-selection competition.
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("selection requires at least one evaluated checkpoint")
    threshold = _rate(config.get("evaluation", {}).get("selection_max_bad_rate", 1 / 8), "selection_max_bad_rate")
    if not 0 <= threshold <= 1:
        raise ValueError("selection_max_bad_rate must be in [0,1]")
    reference = candidates[0]["checkpoint_payload"].get("identity")
    planned_reference = (candidates[0]["report"].get("metadata", {}).get("case_keys"),
                         candidates[0]["report"].get("metadata", {}).get("conditions"))
    rows = []
    owners = {}
    for candidate in candidates:
        if candidate["checkpoint_payload"].get("identity") != reference:
            raise ValueError("selection candidates must share one frozen experiment identity")
        metadata = candidate["report"].get("metadata", {})
        if (metadata.get("case_keys"), metadata.get("conditions")) != planned_reference:
            raise ValueError("selection candidates must share the same planned paired cases and conditions")
        for row in _candidate_rows(candidate, threshold):
            key = row["checkpoint_sha256"], row["team"]
            if key in owners:
                raise ValueError("duplicate checkpoint/team selection candidate")
            owners[key] = candidate
            rows.append(row)
    qualified = [row for row in rows if row["qualified"]]
    efficiency_first = config.get("evaluation", {}).get("selection_metric") == "safe_efficiency_utility"
    if efficiency_first and any(row.get("safe_efficiency_utility") is None for row in qualified):
        raise ValueError("efficiency selection requires measured C1-relative utility")
    qualified.sort(key=lambda row: ((-float(row["safe_efficiency_utility"]),) if efficiency_first else ()) +
                                  (-row["AttackGain_safe"], -row["q"], row["task_violation_episodes"],
                                   row["completed_round"], row["team"], row["checkpoint_sha256"]))
    incomplete = any(not row["evidence_complete"] for row in rows)
    selected = None if incomplete or not qualified else qualified[0]
    result = {"schema_version": SELECTION_SCHEMA,
              "status": "INCOMPLETE_EVIDENCE" if incomplete else "SELECTED" if selected else "NONE_QUALIFIED",
              "split": "selection", "selection_max_bad_rate": threshold,
              "threshold_status": "candidate_screening_not_safety_probability_guarantee",
              "invalid_pair_rule": "all_planned_pairs_required_for_every_competing_team",
              "ranking": (["safe_efficiency_utility_desc"] + RANKING if efficiency_first else RANKING),
              "candidates": rows, "selected": copy.deepcopy(selected),
              "export": None, "test_used_for_selection": False}
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists() or (output / "frozen_blue_team.pt").exists() or (output / "frozen_blue_team.pt.sha256").exists():
        raise FileExistsError("selection/export evidence already exists; choose a fresh output")
    if selected is not None:
        source = owners[selected["checkpoint_sha256"], selected["team"]]
        payload = source["checkpoint_payload"]
        state = payload["teams"][selected["team"]]
        spec = state["model_spec"]
        if spec.get("obs_dim") != 20 or spec.get("action_dim") != 2 or set(state["actors"]) != {2, 3}:
            raise ValueError("selected blue policy must contain physical roles 2/3 using 20/2")
        from training.mut_mappo_config import make_actors
        make_actors(state["actors"], state["config"], (2, 3), "cpu")
        actors = {aid: {key: value.detach().cpu().clone() for key, value in weights.items()}
                  for aid, weights in state["actors"].items()}
        frozen = {"schema_version": EXPORT_SCHEMA, "protocol": EXPORT_PROTOCOL,
                  "team": selected["team"], "completed_round": selected["completed_round"],
                  "actors": actors, "model_spec": copy.deepcopy(spec),
                  "training_config": copy.deepcopy(state["config"]), "actor_hash": actor_states_hash(actors),
                  "identity": copy.deepcopy(payload["identity"]),
                  "source_checkpoint_sha256": selected["checkpoint_sha256"],
                  "source_checkpoint_schema": payload["schema_version"],
                  "evaluation_identity": copy.deepcopy(source["report"]["metadata"]),
                  "selection": copy.deepcopy(selected),
                  "research_status": config.get("research_status", "candidate"),
                  "purpose": "frozen_blue_inference_only_not_training_resume"}
        digest = save_checkpoint_file(frozen, output / "frozen_blue_team.pt", immutable=True)
        result["export"] = {"path": "frozen_blue_team.pt", "sha256": digest, "schema_version": EXPORT_SCHEMA}
    atomic_write_json(result, output / "selection.json")
    return result


def select_evaluation_directories(directories, output_dir):
    """Recompute saved paired evidence before selecting multiple round snapshots."""
    from training.mut_mappo_config import validate_config
    from training.train_mut_mappo import validate_checkpoint_for_config
    candidates, config = [], None
    for directory in directories:
        directory = Path(directory).resolve()
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        current = validate_config(json.loads((directory / "resolved_config.json").read_text(encoding="utf-8")))
        records = [json.loads(line) for line in (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if report.get("paired") != paired_report(records, required_conditions=tuple(report["metadata"]["conditions"])):
            raise ValueError("saved selection summary differs from its episode evidence")
        checkpoint = Path(report["metadata"]["checkpoint_path"])
        digest = sha256_file(checkpoint)
        payload = validate_checkpoint_for_config(current, checkpoint)
        candidates.append({"report": report, "checkpoint_payload": payload,
                           "checkpoint_sha256": digest, "evaluation_source": str(directory)})
        if config is None:
            config = current
        elif config.get("evaluation") != current.get("evaluation"):
            raise ValueError("selection candidates use different selection policies")
    return select_and_export(candidates, config, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True, help="completed selection evaluation directories from one run")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = select_evaluation_directories(args.evaluations, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
