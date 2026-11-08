"""Outcome proxies with explicit reference coverage; no causal claim or CARLA calls."""
import math
from evaluation.r4_risk_metrics import _box, dot, response_measures

PROTOCOL = 'r5_outcome_v1'
DEFAULTS = dict(progress_weight=.03, control_weight=.3, collision_scale=1.,
    outcome_weight=1., outcome_budget=8., delay_fraction=.7,
    progress_scale=.10, progress_deadband=.01, burden_scale=.20,
    burden_deadband=.02, noise_sigma=2., filter_tau_s=.3,
    interaction_front_m=30., interaction_side_m=2., lag_s=1.,
    warmup_s=1., min_reference_samples=3)

def clipped(x, low=-1., high=1.):
    return max(low, min(high, x))

def standardized(delta, scale, deadband):
    return clipped(math.copysign(max(0., abs(delta)-deadband), delta)/scale)

class BurdenFilter:
    """EMA the finite-difference response proxy; same filter in references and trials."""
    def __init__(self, dt, tau):
        self.dt, self.tau = dt, tau
        self.previous = {}; self.acceleration = {}; self.filtered = {}

    def step(self, snapshots, active, reached):
        result = {}
        for rid in (0, 1):
            if not active.get(rid) or reached.get(rid):
                self.previous.pop(rid, None); self.acceleration.pop(rid, None)
                self.filtered.pop(rid, None)
                continue
            snap = snapshots[rid]
            if rid in self.previous:
                raw = response_measures(self.previous[rid], snap, self.dt, self.acceleration.get(rid))
                self.acceleration[rid] = raw['acceleration_mps2']
                alpha = 1.-math.exp(-self.dt/self.tau)
                self.filtered[rid] = self.filtered.get(rid, 0.) + alpha*(raw['response_score']-self.filtered.get(rid, 0.))
                result[rid] = self.filtered[rid]
            self.previous[rid] = snap
        return result

def interacts(red, blue, cfg):
    """Broad road-space gate, not a low-TTC or high-risk bonus. Crossing left for later."""
    rc, axes, re, _ = _box(red)
    bc, ba, be, _ = _box(blue)
    if dot(axes[0], ba[0]) < math.cos(math.radians(30.)):
        return False
    delta = (bc[0]-rc[0], bc[1]-rc[1])
    along, side = dot(delta, axes[0]), abs(dot(delta, axes[1]))
    support = lambda axis: sum(e*abs(dot(axis,a)) for e,a in zip(be,ba))
    long_extent = re[0]+support(axes[0])
    side_extent = re[1]+support(axes[1])
    front = 0. <= along <= cfg['interaction_front_m'] and side <= side_extent
    parallel = abs(along) <= long_extent and 0. < side-side_extent <= cfg['interaction_side_m']
    return front or parallel

class OutcomeTracker:
    def __init__(self, dt=.1, settings=None):
        self.dt = dt; self.cfg = {**DEFAULTS, **(settings or {})}
        self.filter = BurdenFilter(dt, self.cfg['filter_tau_s'])
        self.last_step = 0; self.last_interaction = {}; self.blocked = set(); self.red_contacted=set(); self.effect_runs={2:0,3:0}
        self.totals = {i:dict(interaction_s=0., reference_samples=0, candidate_samples=0,
            delay_excess_integral_s=0., burden_excess_integral_s=0.,
            outcome_signed_integral_s=0., outcome_positive_integral_s=0.,
            effective_outcome_duration_s=0.,effective_outcome_longest_s=0.) for i in (2,3)}

    def step(self, *, step, snapshots, active, reached, progress, safety, reference=None, contact_ids=()):
        if step != self.last_step+1:
            raise ValueError('R5 decision steps must be consecutive')
        self.last_step = step
        for aid, on in active.items():
            if on: _box(snapshots[aid])
        burdens = self.filter.step(snapshots, active, reached)
        from evaluation.r4_risk_metrics import pair_risk
        for rid in (0,1):
            if rid in contact_ids or any(active.get(rid) and active.get(bid) and
                pair_risk(snapshots[rid],snapshots[bid])['contact'] for bid in (2,3)):
                self.red_contacted.add(rid)
        reference = (reference or {}).get('outcome_reference', {})
        if reference and reference.get('protocol') != PROTOCOL:
            raise ValueError('R5 reference protocol mismatch')
        if reference and not math.isclose(reference.get('filter_tau_s',-1.),self.cfg['filter_tau_s']):
            raise ValueError('R5 reference response filter changed; regenerate references')
        refs = reference.get('steps', {}).get(str(step), {})
        gates = {}
        for bid in (2,3):
            if bid in contact_ids or any(safety.get(bid, {}).values()):
                self.blocked.add(bid)
            for rid in (0,1):
                key = (bid,rid)
                good = active.get(bid) and active.get(rid) and bid not in self.blocked and rid not in self.red_contacted and not reached.get(rid)
                if good:
                    from evaluation.r4_risk_metrics import pair_risk
                    good = not pair_risk(snapshots[rid], snapshots[bid])['contact']
                if good and interacts(snapshots[rid], snapshots[bid], self.cfg):
                    self.last_interaction[key] = step
                gates[key] = bool(good and (step-self.last_interaction.get(key,-100000))*self.dt <= self.cfg['lag_s'])
        result = {}
        for bid in (2,3):
            if not active.get(bid): continue
            total = self.totals[bid]; delay = burden = outcome = 0.; matched = False
            engaged = any(gates[bid,r] for r in (0,1))
            total['interaction_s'] += self.dt*engaged
            total['candidate_samples'] += int(engaged)
            for rid in (0,1):
                if not gates[bid,rid]: continue
                ref = refs.get(str(rid))
                if not ref or ref.get('valid_samples',0) < self.cfg['min_reference_samples'] or rid not in burdens:
                    continue  # Missing baseline never becomes baseline zero.
                fields = ('progress_mean','progress_std','burden_mean','burden_std')
                if any(not math.isfinite(ref[k]) for k in fields):
                    raise ValueError('nonfinite R5 reference')
                if ref['progress_std']<0 or ref['burden_std']<0:
                    raise ValueError('negative R5 reference spread')
                actual = progress[rid]
                if not math.isfinite(actual): raise ValueError('nonfinite R5 progress')
                matched = True
                dp = standardized(ref['progress_mean']-actual, self.cfg['progress_scale'],
                    self.cfg['progress_deadband']+self.cfg['noise_sigma']*ref['progress_std'])
                db = standardized(burdens[rid]-ref['burden_mean'], self.cfg['burden_scale'],
                    self.cfg['burden_deadband']+self.cfg['noise_sigma']*ref['burden_std'])
                # Sum shares per red <=1; fixed /2 averages the two physical red slots.
                share = 1./sum(gates[b,rid] for b in (2,3))/2.
                delay += share*dp; burden += share*db
            outcome = self.cfg['delay_fraction']*delay+(1.-self.cfg['delay_fraction'])*burden
            total['reference_samples'] += int(matched)
            total['delay_excess_integral_s'] += delay*self.dt
            total['burden_excess_integral_s'] += burden*self.dt
            total['outcome_signed_integral_s'] += outcome*self.dt
            total['outcome_positive_integral_s'] += max(0.,outcome)*self.dt
            self.effect_runs[bid]=self.effect_runs[bid]+1 if outcome>0. else 0
            total['effective_outcome_duration_s']+=self.dt*float(outcome>0.)
            total['effective_outcome_longest_s']=max(total['effective_outcome_longest_s'],self.effect_runs[bid]*self.dt)
            result[bid] = dict(outcome=outcome, delay=delay, burden=burden)
        return result

    def metrics(self):
        return {str(i):dict(v, protocol=PROTOCOL, attribution='gated frozen-reference proxy; not causal contribution',
            rubric_effect=('sustained_proxy_effect' if v['effective_outcome_longest_s']>=1. else
                'transient_proxy_effect' if v['outcome_positive_integral_s']>0. else 'no_measured_effect'),
            rubric_reference=('unavailable' if not v['reference_samples'] else
                'adequate_proxy_coverage' if v['reference_samples']/max(1,v['candidate_samples'])>=.9 else 'limited_coverage'),
            reference_coverage=v['reference_samples']/v['candidate_samples'] if v['candidate_samples'] else None)
            for i,v in self.totals.items()}
