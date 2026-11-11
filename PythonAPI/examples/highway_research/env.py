"""Task-preserving reward ablations and matched target perturbations."""
import importlib
from dataclasses import replace


class CanonicalObserver:
    """Replay identical physical frames through full reward weights, no RNG."""
    def __init__(self,actual,canonical):self.actual=actual;self.canonical=canonical;self.delta=0.

    def __getattr__(self,key):
        actual=self.__dict__.get('actual')
        if actual is None:raise AttributeError(key)
        return getattr(actual,key)

    def observe(self,frame,dt):
        result=self.actual.observe(frame,dt);reference=self.canonical.observe(frame,dt)
        if result['done']!=reference['done'] or result['success']!=reference['success']:raise RuntimeError('Reward ablation changed task semantics')
        self.delta+=reference['reward']-result['reward']
        return result


class StudyEnv:
    def __init__(self,cfg,condition='nominal'):
        self.cfg=cfg;self.condition=condition
        self.base=importlib.import_module(cfg.backend+'.environment').HybridEnv(cfg._base)

    def __getattr__(self,key):
        base=self.__dict__.get('base')
        if base is None:raise AttributeError(key)
        return getattr(base,key)

    def reset(self,**kwargs):
        self._canonical_return=0.
        if self.condition=='position_shift':
            kwargs=dict(kwargs);options=dict(kwargs.get('options') or {})
            # Frozen recovery option shifts attacker identically on real/reference roads.
            options['recovery']=True;kwargs['options']=options
        obs,info=self.base.reset(**kwargs)
        if self.cfg.level==2 and (self.cfg.task['risk_weight']!=1. or self.cfg.task['event_weight']!=1.):
            cls=importlib.import_module(self.cfg.backend+'.straight.rewards').RewardState
            self.base.tracker=CanonicalObserver(self.base.tracker,cls(replace(self.cfg._base,risk_weight=1.,event_weight=1.)))
        for road in [self.base.road,self.base.reference]:
            target=road.vehicles[1]
            if self.condition=='idm_short_gap':target.TIME_WANTED=1.
            elif self.condition=='idm_long_gap':target.TIME_WANTED=2.
            elif self.condition=='cautious_lane':
                target.LANE_CHANGE_MIN_ACC_GAIN=.5;target.LANE_CHANGE_DELAY=2.
            elif self.condition not in ('nominal','position_shift'):raise ValueError('unknown condition')
        return obs,info

    def step(self,action):
        before=self.base.tracker.delta if isinstance(self.base.tracker,CanonicalObserver) else 0.
        obs,reward,term,trunc,info=self.base.step(action)
        parts=dict(info['reward_components']);original=reward
        if isinstance(self.base.tracker,CanonicalObserver):original+=self.base.tracker.delta-before
        self._canonical_return+=original
        for name in self.cfg.drop_components:
            reward-=parts.get(name,0.);parts[name]=0.
        info=dict(info,reward_components=parts,original_return_step=original,canonical_return=self._canonical_return)
        return obs,float(reward),term,trunc,info

    def close(self):self.base.close()
