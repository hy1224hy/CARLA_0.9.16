"""Checkpointed, gate-only driving qualification controls training attack weight."""
import copy
import math

def validate_schedule(spec):
    if set(spec) != {'weights','min_steps','qualified','max_bad','patience'}:
        raise ValueError('invalid curriculum fields')
    weights = spec['weights']
    if not isinstance(weights,list) or not weights or any(
            not isinstance(x,(int,float)) or not math.isfinite(x) or x <= 0 for x in weights):
        raise ValueError('curriculum needs finite positive weights')
    if any(a >= b for a,b in zip(weights,weights[1:])):
        raise ValueError('curriculum weights must increase')
    for key in ('min_steps','patience'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError('curriculum steps/patience must be positive integers')
    if any(not isinstance(spec[k],(int,float)) or not math.isfinite(spec[k]) or not 0 <= spec[k] <= 1
           for k in ('qualified','max_bad')):
        raise ValueError('curriculum qualification thresholds must lie in [0,1]')

class Curriculum:
    def __init__(self, spec, state=None):
        self.spec=copy.deepcopy(spec)
        if spec is not None:
            validate_schedule(spec)
        self.state=copy.deepcopy(state) if state is not None else {
            'stage':0,'entered_at':0,'good_checks':0,'bad_checks':0,'transitions':[]}
        if spec is not None and not 0 <= self.state['stage'] < len(spec['weights']):
            raise ValueError('invalid saved curriculum stage')

    @property
    def weight(self):
        return self.spec['weights'][self.state['stage']] if self.spec else None

    @property
    def full_strength(self):
        return self.spec is None or self.state['stage']==len(self.spec['weights'])-1

    def state_dict(self):
        return copy.deepcopy(self.state)

    def observe(self, reports, steps):
        if self.spec is None:
            return
        good=all(reports[t]['overall']['success_rate'] >= self.spec['qualified']
                 and reports[t]['overall']['bad_rate'] <= self.spec['max_bad']
                 and reports[t]['overall'].get('baseline_invalid_count',0)==0 for t in ('A','B'))
        self.state['good_checks']=self.state['good_checks']+1 if good else 0
        self.state['bad_checks']=0 if good else self.state['bad_checks']+1
        stage=self.state['stage']
        reason=None
        if (self.state['good_checks'] >= self.spec['patience'] and not self.full_strength
                and steps-self.state['entered_at'] >= self.spec['min_steps']):
            stage+=1; reason='qualified_promotion'
        elif self.state['bad_checks'] >= self.spec['patience'] and stage>0:
            stage-=1; reason='qualification_regression'
        if reason:
            self.state['transitions'].append({'train_steps':steps,'from':self.weight,
                'to':self.spec['weights'][stage],'reason':reason})
            self.state.update(stage=stage,entered_at=steps,good_checks=0,bad_checks=0)
