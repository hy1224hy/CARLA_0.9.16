"""Bounded critical-interaction objective; pure current-state geometry, no CARLA."""
from __future__ import annotations

import math
from environment.mut_mappo_efficiency_reward import TeamEfficiencyReward
from evaluation.critical_interaction_metrics import obb_ttc, _box

PROFILE = "critical_interaction_team_v1"
DEFAULTS = dict(critical_weight=5.0, exposure_cap_s=2.0, severe_weight=2.0,
                severe_ttc_s=.5, plateau_start_s=1.0, plateau_end_s=1.5,
                band_end_s=3.0, prediction_horizon_s=5.0,
                event_ttc_s=1.5, event_duration_s=.3,
                pace_weight=.01, free_flow_target_mps=2.5)


def validate_settings(settings):
    known = set(DEFAULTS) | {"attack_weight", "driving_weight", "collision_cost", "offroad_cost",
        "road_rule_cost", "task_violation_cost", "reference_dir", "gamma"}
    if set(settings) - known:
        raise ValueError("unknown critical reward settings: " + str(sorted(set(settings)-known)))
    result = {**DEFAULTS, **settings}
    for key, value in result.items():
        if key == "reference_dir":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid critical reward coefficient: " + key)
    if not (0 < result['severe_ttc_s'] < result['plateau_start_s'] <= result['plateau_end_s']
            < result['band_end_s'] <= result['prediction_horizon_s']):
        raise ValueError("critical TTC band must have ordered positive boundaries")
    if result['exposure_cap_s'] <= 0 or result['event_duration_s'] <= 0 or not (
            0 < result['event_ttc_s'] <= result['prediction_horizon_s']):
        raise ValueError("critical exposure/event limits must be positive")
    if 'gamma' in result and not 0 < result['gamma'] <= 1:
        raise ValueError("invalid critical reward gamma")
    if any(not math.isclose(result[key],expected) for key,expected in
           (('event_ttc_s',1.5),('event_duration_s',.3),('prediction_horizon_s',5.))):
        raise ValueError('v1 event measurement protocol is frozen at 1.5s/0.3s/5s')
    return result


def band_score(ttc, cfg):
    if ttc is None or ttc <= cfg['severe_ttc_s'] or ttc >= cfg['band_end_s']:
        return 0.0
    if ttc < cfg['plateau_start_s']:
        return (ttc-cfg['severe_ttc_s'])/(cfg['plateau_start_s']-cfg['severe_ttc_s'])
    if ttc <= cfg['plateau_end_s']:
        return 1.0
    return (cfg['band_end_s']-ttc)/(cfg['band_end_s']-cfg['plateau_end_s'])


class CriticalTeamReward(TeamEfficiencyReward):
    def __init__(self, settings=None, **kwargs):
        self.critical = validate_settings(settings or {})
        super().__init__({"attack_weight": 2.0, **self.critical}, **kwargs)

    def reset(self):
        super().reset()
        self.exposure_s = self.tet_s = self.tit_s = 0.0
        self.critical_event = False
        self.minimum_ttc_s = None
        self.runs = {}
        self.last_geometry_step = 0

    def step(self, *, snapshots, active, driving_context, **kwargs):
        step, eligible = kwargs['step'], kwargs['eligible']
        if step != self.last_geometry_step + 1:
            raise ValueError("critical reward requires consecutive decision geometry")
        self.last_geometry_step = step
        live = [i for i, on in active.items() if on]
        for aid in live:
            if aid not in snapshots:
                raise ValueError(f"missing live actor geometry: {aid}")
            _box(snapshots[aid])  # Missing/nonfinite geometry is never a safe zero.
        pairs, q, severe = [], 0.0, 0.0
        next_runs = {}
        for rid in (0, 1):
            for bid in (2, 3):
                if rid not in live or bid not in live:
                    continue
                red, blue = snapshots[rid], snapshots[bid]
                ttc = obb_ttc(red, blue, self.critical['prediction_horizon_s'])
                key = (rid, bid)
                run = self.runs.get(key, 0) + 1 if ttc is not None and ttc <= self.critical['event_ttc_s'] else 0
                next_runs[key] = run
                if ttc is None:
                    continue
                pairs.append(ttc)
                self.critical_event |= ttc == 0 or run*self.dt+1e-9 >= self.critical['event_duration_s']
                severe = max(severe, max(0., 1-ttc/self.critical['severe_ttc_s']))
                rc, axes, _, _ = _box(red)
                bc, _, _, _ = _box(blue)
                longitudinal = sum((bc[j]-rc[j])*axes[0][j] for j in (0, 1))
                context = driving_context.get(bid, {})
                # Dynamic front hemisphere: rear-spawned blue may overtake, but
                # directly chasing the red from behind earns no positive credit.
                permitted = (eligible and context.get('legal_forward') is True and longitudinal >= 0
                             and not red.get('reached', False))
                if permitted:
                    q = max(q, band_score(ttc, self.critical))
        self.runs = next_runs
        minimum = min(pairs) if pairs else None
        if minimum is not None:
            self.minimum_ttc_s = minimum if self.minimum_ttc_s is None else min(self.minimum_ttc_s, minimum)
            if minimum <= self.critical['event_ttc_s']:
                self.tet_s += self.dt
                self.tit_s += (self.critical['event_ttc_s']-minimum)*self.dt
            if minimum <= self.critical['severe_ttc_s']:
                q = 0.0  # Team suppression: another moderate pair cannot hide contact.
        increment = min(max(0., self.critical['exposure_cap_s']-self.exposure_s), self.dt*q)
        self.exposure_s += increment
        pace = 0.0
        if eligible:
            for bid in (2, 3):
                if bid not in live:
                    continue
                c = driving_context[bid]
                if not any(c.get(k, False) for k in ('front_blocker', 'near_goal', 'terminal')):
                    pace += self.critical['pace_weight']*self.dt*max(0.,
                        self.critical['free_flow_target_mps']-float(c['route_speed_mps']))/2
        utility, components = super().step(**kwargs)
        extra = dict(critical=self.critical['critical_weight']*increment/self.critical['exposure_cap_s'],
                     severe=-self.critical['severe_weight']*self.dt*severe, pace=-pace)
        delta = math.fsum(extra.values())
        if not math.isfinite(delta):
            raise FloatingPointError("nonfinite critical reward")
        self.utility_total += delta
        self.cost_total += pace + self.critical['severe_weight']*self.dt*severe
        for key, value in extra.items():
            self.components_total[key] = self.components_total.get(key, 0.) + value
        return utility+delta, {**components, **extra}

    def metrics(self, eligible):
        return {**super().metrics(eligible), 'critical_event': bool(self.critical_event),
                'safe_critical_event': float(bool(eligible) and self.critical_event),
                'critical_exposure_s': self.exposure_s, 'critical_min_ttc_s': self.minimum_ttc_s,
                'critical_tet_s': self.tet_s, 'critical_tit_s': self.tit_s,
                'critical_geometry_protocol': 'actual_bbox_fixed_yaw_constant_velocity_v1',
                'critical_prediction_horizon_s': self.critical['prediction_horizon_s']}
