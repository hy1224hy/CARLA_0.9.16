"""Generic training wrapper, including CARLA: same objective, inputs and PPO API.

Baseline trials are sampled BEFORE the C2 world is loaded. This supports a
single CARLA service. A production scene loader may cache frozen references
by case/controller/task hashes; no simultaneous world owners are required.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ..common.contract import inputs, OBS_DIM
from ..common.objective import PairedObjective
from ..common.outcomes import paired_outcome
from ..environment.scenarios import FAMILIES


class PortablePairedEnv(gym.Env):
    def __init__(self, task, gamma, backend_factory):
        self.task, self.gamma, self.factory = task, gamma, backend_factory
        self.observation_space = spaces.Box(-10, 10, (2, OBS_DIM), np.float32)
        self.action_space = spaces.Box(-1, 1, (2, 2), np.float32)
        self.backend = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.family = (options or {}).get("family") or self.np_random.choice(FAMILIES)
        self.reference_frames, self.reference_outcomes = {}, {}
        for condition in ("C0", "C1"):
            backend = self.factory(condition, self.family, seed)
            frames = [backend.frame]
            try:
                while backend.frame.time < self.task.duration - 1e-6 and not backend.tracker.bad:
                    backend.step(None)
                    frames.append(backend.frame)
                self.reference_frames[condition] = frames
                self.reference_outcomes[condition] = backend.tracker.summary()
            finally:
                backend.close()
        c0, c1 = self.reference_outcomes["C0"], self.reference_outcomes["C1"]
        if not c0["fixed_horizon_complete"] or c0["bad"] or not c1["qualified"]:
            raise ValueError("frozen normal traffic failed qualification; fix/calibrate scene before training")
        self.backend = self.factory("C2", self.family, seed)
        self.objective = PairedObjective(self.task, self.gamma)
        self.steps, self.ended = 0, False
        return inputs(self.backend.frame, self.task)[0], self._info()

    def _info(self):
        safety = self.backend.tracker.summary()
        _, state, links = inputs(self.backend.frame, self.task)
        reference = self.reference_frames["C1"][min(self.steps, len(self.reference_frames["C1"]) - 1)]
        normal_progress = float(np.mean([reference.vehicles[i].progress for i in (0, 1)]))
        loss = normal_progress - safety["red_progress_m"]
        quality = (float(safety["qualified"]) if self.task.attack_weight == 0 else
                   float(safety["qualified"]) * float(np.clip(loss / self.task.goal_distance, 0, 1)))
        return {**safety, "state": state, "links": links, "family": self.family,
                "episode_steps": self.steps, "success": safety["qualified"], "quality": quality,
                "red_delay_m": loss, "normal_red_progress_m": normal_progress, "baseline_valid": True,
                "hard_brakes": safety["hard_brake_events"],
                "safe_event": bool(not safety["bad"] and safety["min_ttc_s"] < self.task.ttc_threshold),
                "qualified_event": bool(safety["qualified"] and safety["min_ttc_s"] < self.task.ttc_threshold)}

    def step(self, actions):
        if self.ended:
            raise RuntimeError("reset required")
        self.backend.step(actions)
        self.steps += 1
        info = self._info()
        reward, components = self.objective.step(self.backend.frame, info, info["normal_red_progress_m"], baseline_valid=True)
        self.ended = info["bad"] or info["fixed_horizon_complete"]
        info["reward_components"] = components
        return inputs(self.backend.frame, self.task)[0], reward, bool(self.ended), False, info

    def render(self):
        return self.backend.render()

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
