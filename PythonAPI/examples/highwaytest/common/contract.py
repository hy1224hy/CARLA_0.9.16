"""SI units and route-relative features shared by Highway and CARLA adapters."""
from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np

BLUE_IDS = (2, 3)
RED_IDS = (0, 1)
PROTOCOL = "route_relative_paired_attack_v3"
OBS_DIM, STATE_DIM = 25, 29


@dataclass(frozen=True)
class VehicleFrame:
    identity: int
    position: tuple[float, float]
    velocity: tuple[float, float]
    heading: float
    lane_error: float
    route_heading: float
    progress: float
    collision: bool = False
    offroad: bool = False
    length: float = 5.0

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))


@dataclass(frozen=True)
class SceneFrame:
    vehicles: Mapping[int, VehicleFrame]
    time: float
    origin: tuple[float, float]
    route_heading: float = 0.0


def rotation(heading):
    return np.array([[math.cos(heading), math.sin(heading)],
                     [-math.sin(heading), math.cos(heading)]])


def inputs(scene, task):
    if set(scene.vehicles) != {0, 1, 2, 3}:
        raise ValueError("policy inputs require four physical role identities")
    rows = []
    for identity in BLUE_IDS:
        ego = scene.vehicles[identity]
        angle = ego.heading - ego.route_heading
        row = [ego.speed / 30, ego.lane_error / 2, math.sin(angle), math.cos(angle),
               ego.progress / task.goal_distance, identity - 2, scene.time / task.duration]
        for other_id in range(4):
            if other_id == identity:
                continue
            other = scene.vehicles[other_id]
            dp = np.subtract(other.position, ego.position)
            dv = np.subtract(other.velocity, ego.velocity)
            if np.linalg.norm(dp) <= task.sensing_range:
                local_p, local_v = rotation(ego.heading) @ dp, rotation(ego.heading) @ dv
                row.extend([1, local_p[0] / 50, local_p[1] / 8,
                            local_v[0] / 30, local_v[1] / 10, float(other_id in BLUE_IDS)])
            else:
                row.extend([0] * 6)
        rows.append(row)
    state = []
    for identity in range(4):
        v = scene.vehicles[identity]
        xy = rotation(scene.route_heading) @ np.subtract(v.position, scene.origin)
        angle = v.heading - scene.route_heading
        state.extend([xy[0] / task.goal_distance, xy[1] / 4, v.speed / 30,
                      math.sin(angle), math.cos(angle), float(identity in BLUE_IDS),
                      v.progress / task.goal_distance])
    state.append(scene.time / task.duration)
    connected = np.linalg.norm(np.subtract(scene.vehicles[2].position, scene.vehicles[3].position)) <= task.communication_range
    obs, state = np.clip(np.array(rows, np.float32), -10, 10), np.array(state, np.float32)
    if not np.isfinite(obs).all() or not np.isfinite(state).all():
        raise ValueError("nonfinite simulator frame")
    return obs, state, np.full((2, 1), float(connected), np.float32)


def physical_action(action):
    value = np.asarray(action, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 1 + 1e-6:
        raise ValueError("expected bounded [signed acceleration, steering angle]")
    return 5 * float(value[0]), 0.25 * float(value[1])
