"""Additional observational conflict outcomes; no physics or reward changes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from evaluation.critical_interaction_metrics import _box, obb_ttc

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(version="interaction_quality_v2", dt=.1, ttc_s=1.5, duration_s=.3,
                resolution_s=.3, birth_stratum_s=1., geometry_epsilon=1e-10)


def contract():
    path = ROOT / "configs/evaluation/base_driving_vnext_v1.json"
    return dict(**SETTINGS, driving_thresholds=json.loads(path.read_text(encoding="utf-8"))["thresholds"],
        driving_contract_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        lane_reference="existing road_reference_waypoints update after permitted lane change; exclude only recorded legal-transition grace",
        scope="engineering observational quality, not formal base qualification or collision responsibility")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (index - lo)


def driving_quality(rows, episode, settings):
    """Three-valued pass: missing measurements stay unknown, observed failure fails."""
    t = settings["driving_thresholds"]
    roles = {}
    for aid in ("2", "3"):
        metric = episode.get("driving_metrics", {}).get(aid, {})
        active = [r for r in rows if r.get("step", 0) > 0 and r.get("active_mask", {}).get(aid)]
        lateral, missing, exempt, cruise = [], 0, 0, 0
        for index, row in enumerate(active, 1):
            d = row.get("driving_diagnostics", {}).get(aid, {})
            if not _number(d.get("lateral_distance_m")) or d.get("lane_deviation_exempted") not in (0, 1):
                missing += 1
            elif d["lane_deviation_exempted"]:
                exempt += 1
            else:
                lateral.append(abs(d["lateral_distance_m"]))
            if (d.get("pace_front_blocker_exempt") in (0, False) and
                    index * .1 > t["startup_exclusion_seconds"] and
                    d.get("distance_to_goal_m", 0) > t["near_goal_exclusion_m"]):
                cruise += 1
        def check(key, threshold, upper=True):
            value = metric.get(key)
            return (value <= threshold if upper else value >= threshold) if _number(value) else None
        checks = dict(
            state_coverage=check("measured_state_fraction", 1., False),
            actual_control=check("actual_control_fraction", t["minimum_actual_control_fraction"], False),
            peak_speed=check("peak_speed_mps", t["max_peak_speed_mps"]),
            overspeed=check("above_preferred_speed_fraction", t["max_above_preferred_speed_fraction"]),
            cruise=(check("preferred_speed_fraction", t["minimum_preferred_speed_fraction"], False)
                    if cruise >= t["minimum_cruise_samples"] else None),
            unblocked_low_speed=check("unblocked_low_speed_fraction", t["max_unblocked_low_speed_fraction"]),
            unblocked_stall=check("unblocked_low_speed_longest_seconds", t["max_unblocked_low_speed_continuous_seconds"]),
            actual_steering=check("actual_steer_rate_p95_per_second", t["max_actual_steer_rate_p95_per_second"]),
            lane_p95=(_quantile(lateral, .95) <= t["max_reference_lateral_p95_m"] if lateral and not missing else None),
            lane_peak=(max(lateral) <= t["max_reference_lateral_peak_m"] if lateral and not missing else None))
        roles[aid] = dict(checks=checks, active_steps=len(active), lane_missing_steps=missing,
            legal_transition_exempt_steps=exempt, cruise_samples=cruise,
            lane_p95_nonexempt_m=_quantile(lateral, .95), lane_peak_nonexempt_m=max(lateral) if lateral else None)
    values = [episode.get("blue_eligible"), *(v for r in roles.values() for v in r["checks"].values())]
    eligible = False if any(v is False for v in values) else (True if all(v is True for v in values) else None)
    return dict(eligible=eligible, per_blue=roles, status="MEASURED" if eligible is not None else "INCOMPLETE",
                definition="existing research driving preferences plus event eligibility; legal-transition-only lane exemption")


def box_gap(a, b):
    """Euclidean planar rectangle clearance; overlap/touch is zero."""
    if obb_ttc(a, b, 1e-6) == 0:
        return 0.
    def corners(s):
        center, axes, extent, _ = _box(s)
        return [(center[0] + i * extent[0] * axes[0][0] + j * extent[1] * axes[1][0],
                 center[1] + i * extent[0] * axes[0][1] + j * extent[1] * axes[1][1])
                for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    def segment(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        u = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
        return math.hypot(p[0]-a[0]-u*dx, p[1]-a[1]-u*dy)
    ca, cb = corners(a), corners(b)
    return min(segment(p, polygon[i], polygon[(i+1) % 4]) for pts, polygon in ((ca, cb), (cb, ca))
               for p in pts for i in range(4))


def analyze_outcomes(rows, episode, legacy, settings=None):
    settings = settings or contract()
    quality = driving_quality(rows, episode, settings)
    result = dict(version=SETTINGS["version"], status=legacy["status"], predicted_conflict=None,
        observed_contact_free_resolved=None, contact_free_censored=None, joint_normal_resolved=None,
        geometric_contact=False, physical_contact=None, geometry_contact_unconfirmed=None,
        precontact_tet_s=None, precontact_ttc_deficit_s2=None, minimum_positive_ttc_s=None,
        minimum_box_clearance_m=None, events=[], driving_quality=quality)
    if legacy["status"] != "COMPLETE":
        return result
    dt = settings["dt"]
    contacts = episode.get("contact_events")
    if not isinstance(contacts, list):
        result["status"] = "CONTACT_EVIDENCE_MISSING"
        return result
    result["physical_contact"] = bool(contacts)
    result["physical_contact_classes"] = {kind: sum(e.get("pair_class") == kind for e in contacts)
        for kind in ("red_blue", "red_red", "blue_blue", "static_or_unknown", "external_vehicle")}
    result["blue_collision"] = episode.get("safety", {}).get("collision")
    offsets = {r["evidence_frame"] - r["evidence_step"] for r in episode.get("agent_results", {}).values()
               if isinstance(r, dict) and _number(r.get("evidence_frame")) and _number(r.get("evidence_step"))}
    offset = next(iter(offsets)) if len(offsets) == 1 else None
    contact_steps = {}
    for event in contacts:
        ids = (event.get("agent_id"), event.get("other_agent_id"))
        if set(ids) & {0, 1} and set(ids) & {2, 3}:
            if offset is None or not _number(event.get("frame")):
                result["status"] = "CONTACT_TIMING_UNMEASURED"
                return result
            key = tuple(sorted(ids))
            contact_steps[key] = min(contact_steps.get(key, math.inf), event["frame"] - offset)
    pair_samples, clearance = {}, []
    for row in rows:
        step = row.get("step", 0)
        if step <= 0:
            continue
        mask, kin = row["active_mask"], row["kinematics"]
        for red in ("0", "1"):
            for blue in ("2", "3"):
                if not mask.get(red) or not mask.get(blue):
                    continue
                a, b = kin[red], kin[blue]
                ttc = obb_ttc(a, b, 5.)
                gap = box_gap(a, b)
                clearance.append(gap)
                pair_samples.setdefault((int(red), int(blue)), []).append((step, ttc, gap))
    positive, per_step = [], {}
    geom = unconfirmed = False
    for pair, samples in pair_samples.items():
        physical = contact_steps.get(pair, math.inf)
        overlaps = [s for s, t, _ in samples if t is not None and t <= settings["geometry_epsilon"]]
        geom |= bool(overlaps)
        unconfirmed |= bool(overlaps) and pair not in contact_steps
        first_contact = min(physical, min(overlaps, default=math.inf))
        groups = []
        for sample in samples:
            step, ttc, _ = sample
            if step >= first_contact or ttc is None or not 0 < ttc <= settings["ttc_s"]:
                continue
            positive.append(ttc)
            per_step[step] = min(per_step.get(step, math.inf), ttc)
            if not groups or step != groups[-1][-1][0] + 1:
                groups.append([])
            groups[-1].append(sample)
        by_step = {s: (t, gap) for s, t, gap in samples}
        last = samples[-1][0]
        for group in groups:
            if len(group) * dt + 1e-10 < settings["duration_s"]:
                continue
            start, end = group[0][0], group[-1][0]
            clear = 0
            outcome = "contact_free_censored"
            for step in range(end + 1, last + 1):
                if step >= physical:
                    outcome = "physical_contact"
                    break
                if step not in by_step:
                    break
                ttc = by_step[step][0]
                if ttc is not None and ttc <= settings["geometry_epsilon"]:
                    outcome = "geometry_contact_unconfirmed" if pair not in contact_steps else "physical_contact"
                    break
                clear = clear + 1 if ttc is None or ttc > settings["ttc_s"] else 0
                if clear * dt + 1e-10 >= settings["resolution_s"]:
                    outcome = "observed_contact_free_resolved"
                    break
            if end + 1 >= physical:
                outcome = "physical_contact"
            result["events"].append(dict(red_id=pair[0], blue_id=pair[1], start_step=start, end_step=end,
                outcome=outcome, duration_s=len(group)*dt, min_positive_ttc_s=min(t for _, t, _ in group),
                minimum_box_clearance_m=min(g for _, _, g in group),
                birth_stratum=start*dt <= settings["birth_stratum_s"]))
    resolved = any(e["outcome"] == "observed_contact_free_resolved" for e in result["events"])
    result.update(predicted_conflict=bool(result["events"]), geometric_contact=geom,
        geometry_contact_unconfirmed=unconfirmed, physical_red_blue_contact=bool(contact_steps),
        observed_contact_free_resolved=resolved,
        contact_free_censored=any(e["outcome"] == "contact_free_censored" for e in result["events"]),
        joint_normal_resolved=(None if resolved and quality["eligible"] is None else bool(resolved and quality["eligible"])),
        precontact_tet_s=len(per_step)*dt,
        precontact_ttc_deficit_s2=sum(settings["ttc_s"] - t for t in per_step.values())*dt,
        minimum_positive_ttc_s=min(positive) if positive else None,
        minimum_box_clearance_m=min(clearance) if clearance else None)
    return result


def paired_quality_rows(cases, paired, label):
    """Use the identical, latest whole-pair denominator as the legacy report."""
    lookup = {(r["team"], r["case_key"], r["seed"], r["paired_attempt"], r["condition"]): r for r in cases}
    output = []
    fields = ("predicted_conflict", "observed_contact_free_resolved", "contact_free_censored",
              "joint_normal_resolved", "geometric_contact", "physical_contact", "geometry_contact_unconfirmed",
              "precontact_tet_s", "precontact_ttc_deficit_s2", "blue_collision", "physical_red_blue_contact")
    for team, group in paired["teams"].items():
        n = group["n_measured_pairs"]
        record = dict(report=label, team=team, n_pairs=n)
        for condition in ("C1", "C2"):
            subset = [lookup[(team, p["case_key"], p["seed"], p["paired_attempt"], condition)] for p in group["pairs"]]
            measured = [r["interaction_quality"] for r in subset if r["interaction_quality"]["status"] == "COMPLETE"]
            record["n_outcome_measured_"+condition] = len(measured)
            for field in fields:
                values = [r[field] for r in measured]
                record[field+"_"+condition] = (sum(values)/n if n and len(values) == n and all(v is not None for v in values) else None)
            qualities = [r["interaction_quality"]["driving_quality"]["eligible"] for r in subset]
            record["n_driving_unknown_"+condition] = sum(v is None for v in qualities)
            record["driving_quality_rate_"+condition] = sum(v is True for v in qualities)/n if n and all(v is not None for v in qualities) else None
        for field in fields:
            a, b = record[field+"_C1"], record[field+"_C2"]
            record["delta_"+field] = b-a if a is not None and b is not None else None
        output.append(record)
    return output
