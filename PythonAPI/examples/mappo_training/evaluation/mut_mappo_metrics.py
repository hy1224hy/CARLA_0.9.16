"""External Mut-MAPPO metrics, independent of CARLA and training returns.

One record is one attempted episode.  ``valid`` describes whether the episode
measured the task, not whether either team succeeded.  In particular a real red
failure with a censored second red remains a valid team failure.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable, Mapping
from evaluation.driving_policy_metrics import average, summarize_driving

CONDITIONS = ("C0", "C1", "C2")
METRICS_VERSION = "mut_mappo_external_metrics_v2"
PRIMARY_EVENT_METRIC = "qualified_primary_event_credit"
SELECTION_METRICS = frozenset((None, "safe_efficiency_utility", "safe_critical_event", PRIMARY_EVENT_METRIC))


def selection_metric(config):
    value = config.get("evaluation", {}).get("selection_metric")
    if value not in SELECTION_METRICS:
        raise ValueError("unknown selection metric; refusing an implicit ranking fallback")
    return value


def qualified_primary_event_value(row):
    value = row.get(PRIMARY_EVENT_METRIC)
    if type(value) not in (int, float) or value not in (0., 1.):
        raise ValueError("primary-event quality requires a measured binary qualified_primary_event_credit")
    if value and row.get("blue_eligible") is not True:
        raise ValueError("qualified primary event cannot belong to an ineligible blue episode")
    return float(value)
_BOOL_FIELDS = ("red_success", "blue_eligible", "response")
_CENSORED_REASONS = {
    "censored", "external_truncation", "external_censored", "other_red_failure",
    "team_failure_censored", "external_failure",
}


def _validated(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, source in enumerate(records):
        row = dict(source)
        if row.get("record_kind") == "training_episode" or row.get("execution_mode") == "stochastic_training":
            raise ValueError(f"record {index}: training-window episodes are not independent evaluation")
        if not isinstance(row.get("case_key"), str) or not row["case_key"]:
            raise ValueError(f"record {index}: nonempty case_key is required")
        if row.get("condition") not in CONDITIONS:
            raise ValueError(f"record {index}: condition must be C0, C1, or C2")
        if type(row.get("valid")) is not bool:
            raise ValueError(f"record {index}: valid must be a boolean")
        if row["valid"]:
            for field in _BOOL_FIELDS:
                if type(row.get(field)) is not bool:
                    raise ValueError(f"record {index}: {field} must be a boolean")
        attempt = row.get("paired_attempt", 0)
        if type(attempt) is not int or attempt < 0:
            raise ValueError(f"record {index}: paired_attempt must be a nonnegative integer")
        row["paired_attempt"] = attempt
        row["team"] = str(row.get("team", "unspecified"))
        rows.append(row)
    return rows


def _invalid_reason(row: Mapping[str, Any]) -> str:
    return str(row.get("invalid_reason") or row.get("reason") or "unspecified")


def _rate(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _red_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = row.get("red_results", {})
    if isinstance(values, Mapping) and values:
        return [value for value in values.values() if isinstance(value, Mapping)]
    if isinstance(values, list) and values:
        return [value for value in values if isinstance(value, Mapping)]
    outcomes = row.get("agent_outcomes", {})
    if isinstance(outcomes, Mapping):
        return [{"end_reason": value, "success": value == "success"}
                for aid, value in outcomes.items() if str(aid) in {"0", "1"}]
    return []


def _safety_reasons(row: Mapping[str, Any]) -> set[str]:
    reasons: set[str] = set()
    safety = row.get("safety", {})
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"collision", "offroad", "road_rule", "task_violation"} and item is True:
                    reasons.add(key)
                elif key in {"reasons", "violations", "end_reasons"}:
                    visit(item)
                elif isinstance(item, (str, list, Mapping)):
                    visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif value in {"collision", "offroad", "road_rule", "task_violation"}:
            reasons.add(str(value))
    visit(safety)
    # This explicit field is for blue end reasons; general end_reasons can also
    # include red failures and therefore must not be treated as blue violations.
    visit(row.get("blue_end_reasons", {}))
    return reasons


def _red_episode_measurements(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Mean two-red completion and actual team arrival time, when observable."""
    red = _red_rows(row)
    completions, arrivals = [], []
    for result in red:
        completion = result.get("route_completion", result.get("completion"))
        if isinstance(completion, (int, float)) and not isinstance(completion, bool) and math.isfinite(completion):
            completions.append(float(completion))
        seconds = result.get("arrival_seconds")
        if result.get("success") is True and isinstance(seconds, (int, float)) and math.isfinite(seconds):
            arrivals.append(float(seconds))
    completion_mean = sum(completions) / 2 if len(completions) == 2 else None
    team_arrival = max(arrivals) if row.get("red_success") and len(arrivals) == 2 else None
    return completion_mean, team_arrival


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    n = len(valid)
    measured_primary = [row for row in valid if type(row.get(PRIMARY_EVENT_METRIC)) in (int, float)
                        and row[PRIMARY_EVENT_METRIC] in (0., 1.)]
    primary_count = (int(sum(qualified_primary_event_value(row) for row in measured_primary))
                     if n and len(measured_primary) == n else None)
    eligible = sum(row["blue_eligible"] for row in valid)
    responses = sum(row["response"] for row in valid)
    safe_responses = sum(row["response"] and row["blue_eligible"] for row in valid)
    violations = Counter(reason for row in valid for reason in _safety_reasons(row))
    red_reasons: Counter[str] = Counter()
    censored = 0
    completion = []
    arrival = []
    for row in valid:
        for result in _red_rows(row):
            reason = str(result.get("end_reason", result.get("reason", "unspecified")))
            is_censored = result.get("censored") is True or reason in _CENSORED_REASONS
            if is_censored:
                censored += 1
            else:
                red_reasons[reason] += 1
            value = result.get("route_completion", result.get("completion"))
            if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                completion.append(float(value))
            seconds = result.get("arrival_seconds")
            if result.get("success") is True and isinstance(seconds, (int, float)) and math.isfinite(seconds):
                arrival.append(float(seconds))
    ticks: Counter[str] = Counter()
    missing_tick_counts = 0
    for row in rows:
        values = row.get("tick_counts")
        if not isinstance(values, Mapping):
            missing_tick_counts += 1
            continue
        for key, value in values.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"tick_counts.{key} must be a nonnegative integer")
            ticks[str(key)] += value
    return {
        "n_attempted_episodes": len(rows), "n_valid": n, "n_invalid": len(rows) - n,
        "red_success_count": sum(row["red_success"] for row in valid),
        "red_success_rate": _rate(sum(row["red_success"] for row in valid), n),
        "blue_eligible_count": eligible, "blue_eligible_rate": _rate(eligible, n),
        "response_count": responses, "response_rate": _rate(responses, n),
        "q_count": safe_responses, "q": _rate(safe_responses, n),
        "b_count": n - eligible, "b": _rate(n - eligible, n),
        "blue_violation_episode_counts": dict(sorted(violations.items())),
        "blue_violation_episode_rates": {key: _rate(value, n) for key, value in sorted(violations.items())},
        "red_end_reason_counts": dict(sorted(red_reasons.items())),
        "red_censored_count": censored,
        "red_completion_mean": sum(completion) / len(completion) if completion else None,
        "red_successful_arrival_seconds_mean": sum(arrival) / len(arrival) if arrival else None,
        "red_successful_arrival_count": len(arrival),
        "invalid_reason_counts": dict(sorted(Counter(_invalid_reason(row) for row in rows if not row["valid"]).items())),
        "consumed_tick_counts": dict(sorted(ticks.items())),
        "n_attempts_without_tick_counts": missing_tick_counts,
        "attack_integral": average([row.get("attack_integral") for row in valid]),
        "safe_efficiency_utility": average([row.get("safe_efficiency_utility") for row in valid]),
        "safe_critical_event": (average([row['safe_critical_event'] for row in valid])
            if n and all(type(row.get('safe_critical_event')) in (int, float)
                         and row['safe_critical_event'] in (0., 1.) for row in valid) else None),
        "team_utility_total": average([row.get("team_utility_total") for row in valid]),
        PRIMARY_EVENT_METRIC: primary_count / n if primary_count is not None else None,
        "qualified_primary_event_count": primary_count,
        "qualified_primary_event_measured_episodes": len(measured_primary),
        "driving_metrics": summarize_driving(valid),
    }


def summarize_cases(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize attempted episodes, retaining invalids and the all-valid q/b denominator.

    Retry attempts are intentionally retained here as consumed work.  The
    paired report separately uses one final triplet per team/case.
    """
    rows = _validated(records)
    result = {"metrics_version": METRICS_VERSION, **_summarize(rows)}
    result["conditions"] = {condition: _summarize([row for row in rows if row["condition"] == condition])
                            for condition in CONDITIONS}
    result["teams"] = {team: {condition: _summarize([row for row in rows if row["team"] == team and row["condition"] == condition])
                              for condition in CONDITIONS}
                       for team in sorted({row["team"] for row in rows})}
    axes = sorted({(row["condition"], str(row.get("family", "unknown")), str(row.get("direction", "unknown"))) for row in rows})
    result["strata"] = {"|".join(axis): _summarize([row for row in rows if
        (row["condition"], str(row.get("family", "unknown")), str(row.get("direction", "unknown"))) == axis]) for axis in axes}
    return result


def _paired_team(rows: list[dict[str, Any]], required_conditions: tuple[str, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_key"]].append(row)
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for key, case_rows in sorted(grouped.items()):
        attempt = max(row["paired_attempt"] for row in case_rows)
        selected = [row for row in case_rows if row["paired_attempt"] == attempt]
        by_condition = defaultdict(list)
        for row in selected:
            by_condition[row["condition"]].append(row)
        problems = []
        for condition in required_conditions:
            members = by_condition[condition]
            if not members:
                problems.append(f"{condition}:missing")
            elif len(members) != 1:
                problems.append(f"{condition}:duplicate")
            elif not members[0]["valid"]:
                problems.append(f"{condition}:{_invalid_reason(members[0])}")
        if problems:
            excluded.append({"case_key": key, "paired_attempt": attempt, "reasons": problems})
            continue
        members = {condition: by_condition[condition][0] for condition in required_conditions}
        first, second = members["C1"], members["C2"]
        delta = int(first["red_success"]) - int(second["red_success"])
        eligible = first["blue_eligible"] and second["blue_eligible"]
        completion1, arrival1 = _red_episode_measurements(first)
        completion2, arrival2 = _red_episode_measurements(second)
        item = {"case_key": key, "paired_attempt": attempt, "success_difference": delta,
                "blue_eligible_pair": eligible, "safe_contribution": delta if eligible else 0,
                "unsafe_contribution": 0 if eligible else delta,
                "family": first.get("family", second.get("family", "unspecified")),
                "red_completion_difference_C1_minus_C2": completion1 - completion2 if completion1 is not None and completion2 is not None else None,
                "arrival_delay_C2_minus_C1_both_success_seconds": arrival2 - arrival1 if arrival1 is not None and arrival2 is not None else None,
                "red_success": {condition: row["red_success"] for condition, row in members.items()}}
        included.append(item)
    n = len(included)
    raw = sum(row["success_difference"] for row in included)
    safe = sum(row["safe_contribution"] for row in included)
    # q/b on the final common-valid set, rather than just the safe subset.
    final_rows = []
    selected_keys = {row["case_key"]: row["paired_attempt"] for row in included}
    for row in rows:
        if selected_keys.get(row["case_key"], -1) == row["paired_attempt"] and row["condition"] in required_conditions:
            final_rows.append(row)
    counts = {
        "worsened": sum(row["success_difference"] > 0 for row in included),
        "improved": sum(row["success_difference"] < 0 for row in included),
        "unchanged": sum(row["success_difference"] == 0 for row in included),
        "safe_worsened": sum(row["safe_contribution"] > 0 for row in included),
        "safe_improved": sum(row["safe_contribution"] < 0 for row in included),
        "unsafe_worsened": sum(row["unsafe_contribution"] > 0 for row in included),
        "unsafe_improved": sum(row["unsafe_contribution"] < 0 for row in included),
    }
    completion_difference = [row["red_completion_difference_C1_minus_C2"] for row in included
                             if row["red_completion_difference_C1_minus_C2"] is not None]
    arrival_delay = [row["arrival_delay_C2_minus_C1_both_success_seconds"] for row in included
                     if row["arrival_delay_C2_minus_C1_both_success_seconds"] is not None]
    families = {}
    for family in sorted({str(row["family"]) for row in included}):
        members = [row for row in included if str(row["family"]) == family]
        families[family] = {"n_common_valid": len(members),
                            "AttackGain_raw": sum(row["success_difference"] for row in members) / len(members),
                            "AttackGain_safe": sum(row["safe_contribution"] for row in members) / len(members),
                            "AttackGain_unsafe": sum(row["unsafe_contribution"] for row in members) / len(members)}
    return {"n_cases": len(grouped), "n_common_valid": n, "n_excluded": len(excluded),
            "n_eligible_pairs": sum(row["blue_eligible_pair"] for row in included),
            "AttackGain_raw": _rate(raw, n), "AttackGain_safe": _rate(safe, n),
            "AttackGain_unsafe": _rate(raw - safe, n),
            "red_completion_difference_C1_minus_C2_mean": sum(completion_difference) / len(completion_difference) if completion_difference else None,
            "red_completion_difference_n_pairs": len(completion_difference),
            "arrival_delay_C2_minus_C1_both_success_seconds_mean": sum(arrival_delay) / len(arrival_delay) if arrival_delay else None,
            "arrival_delay_both_success_n_pairs": len(arrival_delay), "families": families,
            "counts": counts, "included": included, "excluded": excluded,
            "conditions": {condition: _summarize([row for row in final_rows if row["condition"] == condition])
                           for condition in required_conditions}}


def paired_report(records: Iterable[Mapping[str, Any]], *,
                  required_conditions: tuple[str, ...] = CONDITIONS) -> dict[str, Any]:
    """Compute case-paired C1/C2 effects on the final common-valid triplets.

    An invalid C0 also excludes its C1/C2 pair.  Retries must set a common
    ``paired_attempt`` and rerun all conditions.  The latest attempt wins even
    when invalid; no fallback to a more favorable prior attempt is permitted.
    A/B are reported separately, never counted as independent training seeds.
    """
    if not {"C1", "C2"}.issubset(required_conditions) or set(required_conditions) - set(CONDITIONS):
        raise ValueError("required_conditions must include C1/C2 and contain only C0/C1/C2")
    if len(set(required_conditions)) != len(required_conditions):
        raise ValueError("duplicate required condition")
    rows = _validated(records)
    teams = sorted({row["team"] for row in rows})
    result = {"metrics_version": METRICS_VERSION, "required_conditions": list(required_conditions),
              "teams": {team: _paired_team([row for row in rows if row["team"] == team], required_conditions)
                        for team in teams},
              "n_attempted_episodes": len(rows),
              "n_invalid_attempted_episodes": sum(not row["valid"] for row in rows),
              "training_repeat_unit": "outer_seed; A/B are dependent candidate teams"}
    if len(teams) == 1:
        result.update(result["teams"][teams[0]])
    return result
