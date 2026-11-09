"""CARLA SI/route/action adapter. Importing this file never connects or ticks CARLA.

Scene loader and frozen traffic controllers are injected so Town05 scenario
ownership stays with the existing project. Real-server integration must be
calibrated; model structure portability is not a zero-shot transfer claim.
"""
import math
import copy
import numpy as np
from ..common.contract import SceneFrame, VehicleFrame, physical_action
from ..common.outcomes import OutcomeTracker


class PolylineRoute:
    def __init__(self, points):
        self.points = np.asarray(points, float)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2:
            raise ValueError("route requires at least two world XY points in meters")
        self.delta = np.diff(self.points, axis=0)
        self.length = np.linalg.norm(self.delta, axis=1)
        if not np.isfinite(self.points).all() or np.any(self.length < 1e-6):
            raise ValueError("invalid route geometry")
        self.offset = np.r_[0, np.cumsum(self.length)]

    def project(self, xy):
        relative = np.asarray(xy) - self.points[:-1]
        fraction = np.clip(np.sum(relative * self.delta, axis=1) / self.length ** 2, 0, 1)
        error = np.asarray(xy) - (self.points[:-1] + fraction[:, None] * self.delta)
        index = int(np.argmin(np.linalg.norm(error, axis=1)))
        tangent = self.delta[index] / self.length[index]
        return (float(self.offset[index] + fraction[index] * self.length[index]),
                float(error[index] @ [-tangent[1], tangent[0]]), math.atan2(tangent[1], tangent[0]))


class CarlaFrameAdapter:
    def __init__(self, actors, routes, *, origin, route_heading, collision_flags, road_map=None):
        self.actors, self.routes = actors, routes
        self.origin, self.route_heading = origin, route_heading
        self.collision_flags, self.road_map = collision_flags, road_map
        self.initial_s = {}
        for identity, actor in actors.items():
            location = actor.get_transform().location
            self.initial_s[identity] = routes[identity].project([location.x, location.y])[0]

    def read(self, elapsed_s):
        vehicles = {}
        flags = self.collision_flags()
        if not set(self.actors).issubset(flags):
            raise ValueError("collision sensor flags must cover every active physical actor")
        for identity, actor in self.actors.items():
            pose, velocity = actor.get_transform(), actor.get_velocity()
            xy = (float(pose.location.x), float(pose.location.y))
            s, lateral, tangent = self.routes[identity].project(xy)
            waypoint = self.road_map.get_waypoint(pose.location, project_to_road=False) if self.road_map else None
            if waypoint is not None:
                center = waypoint.transform.location
                tangent = math.radians(waypoint.transform.rotation.yaw)
                lateral = float(np.subtract(xy, [center.x, center.y]) @ [-math.sin(tangent), math.cos(tangent)])
            vehicles[identity] = VehicleFrame(identity, xy, (float(velocity.x), float(velocity.y)),
                math.radians(pose.rotation.yaw), lateral, tangent, s - self.initial_s[identity],
                bool(flags.get(identity, False)), self.road_map is not None and waypoint is None,
                2 * float(actor.bounding_box.extent.x))
        return SceneFrame(vehicles, float(elapsed_s), tuple(self.origin), self.route_heading)


class CarlaAccelerationControl:
    """Closed-loop acceleration tracking; gains require per-vehicle calibration."""
    def __init__(self, max_steer_angle_rad, kp=0.05, ki=0.02):
        if not math.isfinite(max_steer_angle_rad) or max_steer_angle_rad <= 0:
            raise ValueError("provide the vehicle's physical max steering angle in radians")
        self.max_steer, self.kp, self.ki = max_steer_angle_rad, kp, ki
        self.previous = None
        self.integral = 0.0

    def command(self, frame, action, dt):
        if dt <= 0:
            raise ValueError("positive physical controller timestep required")
        acceleration, steering = physical_action(action)
        measured = 0 if self.previous is None else (frame.speed - self.previous) / dt
        self.previous = frame.speed
        error = acceleration - measured
        self.integral = float(np.clip(self.integral + error * dt, -2, 2))
        longitudinal = float(np.clip(acceleration / 5 + self.kp * error + self.ki * self.integral, -1, 1))
        return {"throttle": max(longitudinal, 0), "brake": max(-longitudinal, 0),
                "steer": float(np.clip(steering / self.max_steer, -1, 1))}


class CarlaConditionBackend:
    """Implements the SAME run_trial backend interface as HighwayConditionBackend.

    loader(condition,family,seed) -> (actors, CarlaFrameAdapter, cleanup).
    C0 loads IDs0/1 only; C1/C2 load0..3 from a frozen scenario manifest.
    traffic_control(id, frame) returns fixed red/normal-blue VehicleControl.
    No TrafficManager randomness may be introduced between paired conditions.
    """
    def __init__(self, task, condition, family, seed, *, world, loader,
                 traffic_control, acceleration_controls, control_factory):
        settings = world.get_settings()
        dt = settings.fixed_delta_seconds
        if not settings.synchronous_mode or dt is None or dt <= 0:
            raise ValueError("CARLA must already be configured for synchronous fixed-delta operation")
        if abs(dt - 1 / task.simulation_frequency) > 1e-6:
            raise ValueError("recorded simulation_frequency disagrees with the actual CARLA physics timestep")
        count = 1 / task.policy_frequency / dt
        if abs(count - round(count)) > 1e-6 or count < 1:
            raise ValueError("CARLA physics dt must divide the common policy period")
        self.task, self.condition, self.world, self.dt = task, condition, world, dt
        self.count = round(count)
        self.actors, self.adapter, self.cleanup = loader(condition, family, seed)
        if self.adapter.road_map is None:
            self.cleanup()
            raise ValueError("CARLA experiments require a Driving-lane map for offroad checks")
        if set(self.actors) != ({0, 1} if condition == "C0" else {0, 1, 2, 3}):
            self.cleanup()
            raise ValueError("scene loader did not preserve physical role identities")
        self.traffic_control, self.controls, self.control_factory = traffic_control, copy.deepcopy(acceleration_controls), control_factory
        try:
            self.frame = self.adapter.read(0)
        except Exception:
            self.cleanup()
            raise
        self.tracker = OutcomeTracker(task)
        self.tracker.update(self.frame)
        self.total_return = None  # Not measured here; the training wrapper computes its reward separately.

    def step(self, actions=None):
        for _ in range(self.count):
            for identity, actor in self.actors.items():
                if identity in (0, 1) or actions is None:
                    command = self.traffic_control(identity, self.frame)
                else:
                    command = self.control_factory(**self.controls[identity].command(
                        self.frame.vehicles[identity], actions[identity - 2], self.dt))
                actor.apply_control(command)
            self.world.tick()
            self.frame = self.adapter.read(self.frame.time + self.dt)
            self.tracker.update(self.frame)
        return self.frame

    def close(self):
        self.cleanup()
