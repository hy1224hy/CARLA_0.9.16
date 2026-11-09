"""Condition backend for the simulator-independent paired experiment runner."""
import numpy as np
from .paired import PairedAttackEnv
from ..common.contract import physical_action
from ..common.outcomes import OutcomeTracker


class HighwayConditionBackend:
    def __init__(self, task, condition, family, seed, *, gamma=0.99, learned_normal=False):
        if task.research_protocol!='route_relative_paired_attack_v3':
            raise ValueError('this is the v3 four-car backend; use the bottleneck backend for v4')
        if condition not in ("C0", "C1", "C2"):
            raise ValueError("unknown traffic condition")
        self.task, self.condition = task, condition
        self.env = PairedAttackEnv(task, gamma=gamma)
        self.env.reset(seed=seed, options={"family": family})
        self.road = (self.env.reference if condition == "C0" else
                     self.env.normal_reference if condition == "C1" and not learned_normal else self.env.road)
        self.learned = condition == "C2" or (condition == "C1" and learned_normal)
        self.tracker = self.env.tracker if self.learned else OutcomeTracker(task)
        self.frame = self.env.snapshot(self.road, time=0)
        self.tracker.update(self.frame)
        self.total_return = 0.0 if self.learned else None
        self.last_reward = np.zeros(2)

    def step(self, actions=None):
        if self.learned:
            _, reward, _, _, _ = self.env.step(actions)
            self.last_reward = reward
            self.total_return += float(reward.mean())
            self.frame = self.env.snapshot()
        else:
            dt = 1 / self.task.simulation_frequency
            for _ in range(self.task.simulation_frequency // self.task.policy_frequency):
                self.road.act()
                self.road.step(dt)
                for v in self.road.vehicles:
                    v.speed = float(np.clip(v.speed, 0, 30))
                self.frame = self.env.snapshot(self.road, time=self.frame.time + dt)
                self.tracker.update(self.frame)
        return self.frame

    def render(self):
        if not self.learned:
            raise ValueError("preview is only available for policy-controlled conditions")
        return self.env.render()

    def close(self):
        self.env.close()
