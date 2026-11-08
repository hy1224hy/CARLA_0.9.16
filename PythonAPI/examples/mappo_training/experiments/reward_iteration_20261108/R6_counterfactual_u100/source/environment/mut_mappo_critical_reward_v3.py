"""R3: individual frozen-base rewards plus per-blue bounded adversarial credit."""
import math
from environment.base_reward_v6 import REWARD_COMPONENTS
from environment.mut_mappo_critical_reward_v2 import NormalDrivingCriticalReward, validate_v2_settings

PROFILE = "individual_driving_critical_v3"
DEFAULTS = dict(base_soft_weight=0.3, adversarial_weight=1.0)
HARD_COMPONENTS = frozenset(("collision", "offroad", "road_rule"))

def validate_v3_settings(settings, *, horizon=500, gamma=.999):
    local = {k: settings.get(k, v) for k, v in DEFAULTS.items()}
    for k, v in local.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError("invalid R3 setting: " + k)
    base = {k: v for k, v in settings.items() if k not in DEFAULTS}
    # R2 geometry is reused, not its driving/efficiency objective.
    base.update(attack_weight=0., driving_weight=0., pace_weight=0.,
                normal_lane_scale=0., normal_comfort_scale=1., normal_overspeed_scale=1.)
    resolved = validate_v2_settings(base, horizon=horizon, gamma=gamma)
    positive = local['adversarial_weight'] * (resolved['critical_weight'] + resolved['primary_event_reward'])
    if positive >= min(resolved.get(k,d) for k,d in (('collision_cost',50.),('offroad_cost',50.),('road_rule_cost',50.),('task_violation_cost',20.))) * gamma**(horizon-1):
        raise ValueError('R3 adversarial budget exceeds discounted safety budget')
    return {**resolved, **local}

class IndividualDrivingCriticalReward(NormalDrivingCriticalReward):
    def __init__(self, settings=None, **kwargs):
        resolved = validate_v3_settings(settings or {}, horizon=kwargs.get('horizon',500), gamma=kwargs.get('gamma',.999))
        self.local = {k: resolved.pop(k) for k in DEFAULTS}
        self.geometry_settings = dict(resolved)
        self.geometry_kwargs = dict(kwargs)
        super().__init__(resolved, **kwargs)
        self.role_geometry = {i: NormalDrivingCriticalReward(resolved, **kwargs) for i in (2,3)}

    def reset(self):
        super().reset()
        self.role_totals = {2: {}, 3: {}}
        self.task_paid = set()
        if hasattr(self, 'role_geometry'):
            for geometry in self.role_geometry.values():
                geometry.reset()

    def step(self, **kwargs):
        _, geometry = super().step(**kwargs)
        local_adv = {}
        for aid, tracker in self.role_geometry.items():
            role_kw = dict(kwargs)
            role_kw['active'] = {i: on and (i in (0,1) or i == aid) for i,on in kwargs['active'].items()}
            role_kw['safety'] = {aid: kwargs['safety'].get(aid,{})}
            role_kw['eligible'] = not any(role_kw['safety'][aid].values()) and not tracker.seen_bad
            _, parts = tracker.step(**role_kw)
            local_adv[aid] = self.local['adversarial_weight'] * math.fsum(parts[k] for k in ('critical','primary_event_bonus','severe'))
        rewards, details = {}, {}
        for aid in (2,3):
            if not kwargs['active'].get(aid, False):
                continue
            base = kwargs['base_details'][aid]
            missing = set(REWARD_COMPONENTS) - set(base)
            if missing:
                raise ValueError('R3 requires complete frozen v6 base components: ' + str(sorted(missing)))
            soft = math.fsum(float(base[k]) for k in REWARD_COMPONENTS if k not in HARD_COMPONENTS)
            hard = math.fsum(float(base[k]) for k in HARD_COMPONENTS)
            # Task-stall is an adapter event, absent from frozen base reward.
            task_event = bool(kwargs['safety'].get(aid,{}).get('task_violation',False)) and aid not in self.task_paid
            task = -self.settings['task_violation_cost'] * task_event
            if task_event:
                self.task_paid.add(aid)
            parts = dict(base_soft=self.local['base_soft_weight']*soft, base_hard=hard,
                         task_violation=task, individual_adversarial=local_adv[aid])
            reward = math.fsum(parts.values())
            if not math.isfinite(reward):
                raise FloatingPointError('nonfinite R3 reward')
            rewards[aid], details[aid] = reward, parts
            for key, value in parts.items():
                self.role_totals[aid][key] = self.role_totals[aid].get(key, 0.) + value
        # Inherited geometry metrics remain useful; utility is mean actual role reward.
        actual = math.fsum(rewards.values()) / 2
        self.utility_total += actual - math.fsum(geometry.values())
        return rewards, details, local_adv

    def metrics(self, eligible):
        metrics = super().metrics(eligible)
        keys = set().union(*(d.keys() for d in self.role_totals.values()))
        metrics['team_reward_components'] = {k: sum(d.get(k,0.) for d in self.role_totals.values()) / 2 for k in keys}
        metrics['team_cost_total'] = sum(max(0.,-v) for d in self.role_totals.values() for v in d.values()) / 2
        metrics['individual_adversarial_metrics'] = {i: tracker.metrics(not tracker.seen_bad) for i,tracker in self.role_geometry.items()}
        metrics['individual_reward_components'] = self.role_totals
        metrics['reward_semantics'] = 'individual_base_plus_individual_adversarial_r3'
        return metrics
