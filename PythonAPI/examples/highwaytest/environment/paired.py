"""Finite-horizon C2 task, paired with synchronized frozen C0/C1 references."""
import numpy as np
from gymnasium import spaces
from highway_env.vehicle.behavior import IDMVehicle
from ..common.contract import SceneFrame, VehicleFrame, inputs, physical_action, OBS_DIM
from ..common.outcomes import OutcomeTracker
from ..common.objective import PairedObjective
from .highway import HighwayTeamEnv


class PairedAttackEnv(HighwayTeamEnv):
    def __init__(self, task, gamma=0.99, render_mode=None):
        super().__init__(task, render_mode)
        self.gamma = gamma
        self.observation_space = spaces.Box(-10, 10, (2, OBS_DIM), np.float32)
        self.normal_reference = None
        self.tracker = None
        self.reference_trackers = None

    def _red(self, road, x, y, speed):
        return IDMVehicle(road, [x, y], speed=speed, target_speed=self.task.red_target_speed,
                          target_lane_index=("0", "1", int(round(y / 4))),
                          enable_lane_change=self.task.red_lane_change, timer=0)

    def snapshot(self, road=None, time=None):
        road = self.road if road is None else road
        vehicles = {}
        for identity, v in enumerate(road.vehicles):
            lane = v.lane
            _, error = lane.local_coordinates(v.position)
            vehicles[identity] = VehicleFrame(identity, tuple(v.position), tuple(v.velocity),
                float(v.heading), float(error), float(lane.heading_at(0)),
                float(v.position[0] - self.initial_x[identity]), bool(v.crashed), not v.on_road, float(v.LENGTH))
        return SceneFrame(vehicles, self.steps / self.task.policy_frequency if time is None else time,
                          (self.origin, 0.0))

    def _obs(self):
        return inputs(self.snapshot(), self.task)[0]

    def _info(self):
        result = super()._info()
        _, result["state"], result["links"] = inputs(self.snapshot(), self.task)
        if self.tracker is None or self.tracker.last is None:
            return result
        safety = self.tracker.summary()
        result.update(safety)
        reference_loss = float(np.mean([r.position[0] - v.position[0]
                              for r, v in zip(self.normal_reference.vehicles[:2], self.road.vehicles[:2])]))
        baseline_valid = (self.reference_trackers is not None and
                          not self.reference_trackers[0].bad and self.reference_trackers[1].summary()["qualified"])
        result.update(success=safety["qualified"], bad=safety["bad"],
            baseline_valid=bool(baseline_valid),
            hard_brakes=safety["hard_brake_events"],
            red_delay_m=reference_loss,
            quality=(float(safety["qualified"]) if self.task.attack_weight == 0 else
                     float(safety["qualified"] and baseline_valid) * float(np.clip(reference_loss / self.task.goal_distance, 0, 1))),
            safe_event=bool(not safety["bad"] and safety["min_ttc_s"] < self.task.ttc_threshold),
            qualified_event=bool(safety["qualified"] and safety["min_ttc_s"] < self.task.ttc_threshold))
        return result

    def reset(self, *, seed=None, options=None):
        self.normal_reference, self.tracker, self.reference_trackers = None, None, None
        super().reset(seed=seed, options=options)
        self.normal_reference = self._road()
        self.normal_reference.vehicles = [self._red(self.normal_reference, *v.position, v.speed)
                                          for v in self.road.vehicles]
        for vehicle in self.normal_reference.vehicles[2:]:
            vehicle.enable_lane_change = self.task.normal_blue_lane_change
            vehicle.target_speed = self.task.normal_blue_target_speed
        self.tracker = OutcomeTracker(self.task)
        self.tracker.update(self.snapshot())
        self.reference_trackers = [OutcomeTracker(self.task), OutcomeTracker(self.task)]
        for tracker, road in zip(self.reference_trackers, (self.reference, self.normal_reference)):
            tracker.update(self.snapshot(road, time=0.0))
        self.objective = PairedObjective(self.task, self.gamma)
        return self._obs(), self._info()

    def advance_references(self, dt, time):
        for tracker, road in zip(self.reference_trackers, (self.reference, self.normal_reference)):
            road.act()
            road.step(dt)
            for v in road.vehicles:
                v.speed = float(np.clip(v.speed, 0, 30))
            tracker.update(self.snapshot(road, time=time))

    def step(self, action):
        if self.ended:
            raise RuntimeError("reset required after finite-horizon episode")
        action = np.asarray(action, np.float32)
        if action.shape != (2, 2):
            raise ValueError("expected two role-specific continuous controls")
        commands = [physical_action(a) for a in action]
        dt = 1 / self.task.simulation_frequency
        substeps = self.task.simulation_frequency // self.task.policy_frequency
        for substep in range(substeps):
            for v in self.road.vehicles[:2]:
                v.act()
            for v, (acceleration, steering) in zip(self.road.vehicles[2:], commands):
                v.act({"acceleration": acceleration, "steering": steering})
            self.road.step(dt)
            self.advance_references(dt, self.steps / self.task.policy_frequency + (substep + 1) * dt)
            for v in self.road.vehicles:
                v.speed = float(np.clip(v.speed, 0, 30))
            self.tracker.update(self.snapshot(time=self.steps / self.task.policy_frequency + (substep + 1) * dt))
        self.steps += 1
        info = self._info()
        horizon = self.steps >= round(self.task.duration * self.task.policy_frequency)
        failure = info["bad"]
        terminal = failure or horizon
        normal_progress = float(np.mean([v.position[0] - self.initial_x[i]
                                        for i, v in enumerate(self.normal_reference.vehicles[:2])]))
        rew, components = self.objective.step(self.snapshot(), info, normal_progress,
                                               baseline_valid=info["baseline_valid"])
        self.ended = terminal
        info["reward_components"] = components
        return self._obs(), rew.astype(np.float32), bool(terminal), False, info
