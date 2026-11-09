from ..common.outcomes import OutcomeTracker
from .contract import RED_IDS,BLUE_IDS

class BottleneckTracker(OutcomeTracker):
    def __init__(self,t,initial_x):
        super().__init__(t,red_ids=RED_IDS,blue_ids=BLUE_IDS,
            goal_distances={i:t.bottleneck['exit_x']-initial_x[i] for i in BLUE_IDS})
        self.normal_progress=None

    def summary(self):
        r=super().summary();ratios={}
        for role,i in enumerate(BLUE_IDS):
            if i in self.last.vehicles:
                reference=(self.normal_progress or {}).get(i)
                ratios[i]=self.last.vehicles[i].progress/max(reference,1e-6) if reference is not None else 1.
                r[f'blue_{role}_progress_retention']=float(ratios[i])
        flow=len(ratios)==2 and min(ratios.values())>=self.task.bottleneck['min_progress_retention']
        stopped=all(self.stopped_seconds[i]<=self.task.bottleneck['max_stopped_s']+1e-6 for i in BLUE_IDS)
        r.update(flow_retention_met=bool(flow),stall_constraint_met=bool(stopped),
                 qualified=bool(r['qualified'] and flow and stopped),
                 blue_min_progress_retention=min(ratios.values()) if ratios else None)
        return r
