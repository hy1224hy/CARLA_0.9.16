"""2-red/2-blue task using Highway-env Road, Vehicle and frozen IDM dynamics.

Gymnasium joint policy interface: local obs [2,24], actions [2,2]. The
central state and link mask are in info, never appended to actor observations.
"""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from .scenarios import starts, FAMILIES
from .rewards import reward

OBS_DIM, STATE_DIM = 24, 28


class HighwayTeamEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, task, render_mode=None):
        self.task, self.render_mode = task, render_mode
        self.observation_space = spaces.Box(-10, 10, (2, OBS_DIM), np.float32)
        self.action_space = spaces.Box(-1, 1, (2, 2), np.float32)

    def _road(self):
        return Road(RoadNetwork.straight_road_network(2, length=5000, speed_limit=20),
                    np_random=self.np_random)

    def _red(self, road, x, y, speed):
        return IDMVehicle(road, [x, y], speed=speed, target_speed=18,
                          target_lane_index=("0", "1", int(y / 4)),
                          enable_lane_change=False, timer=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.family = options["family"] if "family" in options else self.np_random.choice(FAMILIES)
        layout = starts(self.np_random, self.family)
        self.road, self.reference = self._road(), self._road()
        self.road.vehicles = [self._red(self.road, *p) if i < 2 else
                              Vehicle(self.road, p[:2], speed=p[2]) for i, p in enumerate(layout)]
        self.reference.vehicles = [self._red(self.reference, *p) for p in layout[:2]]
        self.initial_x = np.array([v.position[0] for v in self.road.vehicles])
        self.origin = float(self.initial_x.mean())
        self.steps, self.event_seen = 0, False
        self.min_ttc, self.hard_brakes = 100.0, 0
        self.ended = False
        return self._obs(), self._info()

    def _obs(self):
        rows = []
        for i in (2, 3):
            ego = self.road.vehicles[i]
            row = [ego.speed / 30, ego.position[1] / 4, np.sin(ego.heading),
                   np.cos(ego.heading), (ego.position[0] - self.initial_x[i]) / self.task.goal_distance, i - 2]
            for j, other in enumerate(self.road.vehicles):
                if j == i:
                    continue
                dp, dv = other.position - ego.position, other.velocity - ego.velocity
                visible = np.linalg.norm(dp) <= self.task.sensing_range
                row.extend([1, dp[0] / 50, dp[1] / 8, dv[0] / 30, dv[1] / 10, float(j >= 2)]
                           if visible else [0] * 6)
            rows.append(row)
        return np.clip(np.array(rows, np.float32), -10, 10)

    def _info(self):
        cars = self.road.vehicles
        state = np.array([item for i, v in enumerate(cars) for item in
                          [(v.position[0] - self.origin) / self.task.goal_distance,
                           v.position[1] / 4, v.speed / 30, np.sin(v.heading),
                           np.cos(v.heading), float(i >= 2),
                           (v.position[0] - self.initial_x[i]) / self.task.goal_distance]], np.float32)
        collision = any(v.crashed for v in cars)
        offroad = any(not v.on_road for v in cars)
        progress = np.array([v.position[0] for v in cars]) - self.initial_x
        success = bool(np.all(progress[2:] >= self.task.goal_distance) and not collision and not offroad)
        # C0 = no blue vehicles: simultaneous deterministic reference road.
        delay = float(np.mean([r.position[0] - v.position[0]
                              for r, v in zip(self.reference.vehicles, cars[:2])]))
        safe = not collision and not offroad
        quality = float(safe) * (float(success) + (0.25 if self.task.attack_weight > 0 else 0)
                                * np.clip(delay / self.task.goal_distance, 0, 1))
        distance = np.linalg.norm(cars[2].position - cars[3].position)
        return {"state": state, "links": np.full((2, 1), float(distance <= self.task.communication_range), np.float32),
                "family": self.family, "collision": collision, "offroad": offroad,
                "bad": not safe, "success": success, "red_delay_m": delay,
                "blue_progress_m": float(progress[2:].mean()), "min_ttc_s": self.min_ttc,
                "hard_brakes": int(self.hard_brakes), "event_seen": bool(self.event_seen),
                "safe_event": bool(safe and self.event_seen),
                "qualified_event": bool(success and self.event_seen), "quality": float(quality),
                "episode_steps": self.steps}

    def step(self, action):
        if self.ended:
            raise RuntimeError("reset is required after episode termination")
        action = np.asarray(action, np.float32)
        if action.shape != (2, 2) or not np.isfinite(action).all():
            raise ValueError("expected finite [2,2] acceleration/steering actions")
        action = np.clip(action, -1, 1)
        cars = self.road.vehicles
        before = np.array([v.position[0] for v in cars])
        ref_before = np.array([v.position[0] for v in self.reference.vehicles])
        # Same fixed command over simulation substeps; red recomputes IDM each substep.
        for _ in range(self.task.simulation_frequency // self.task.policy_frequency):
            for v in cars[:2]:
                v.act()
            for v, a in zip(cars[2:], action):
                v.act({"acceleration": float(5 * a[0]), "steering": float(0.25 * a[1])})
            self.reference.act()
            self.road.step(1 / self.task.simulation_frequency)
            self.reference.step(1 / self.task.simulation_frequency)
            # This driving task excludes reverse gear. Highway-env's raw Vehicle
            # otherwise permits a negative speed under a sustained braking input.
            for v in self.road.vehicles + self.reference.vehicles:
                v.speed = float(np.clip(v.speed, 0, 30))
            self._measure_ttc()
        self.steps += 1
        info = self._info()
        red_advance = np.array([v.position[0] for v in cars[:2]]) - before[:2]
        ref_advance = np.array([v.position[0] for v in self.reference.vehicles]) - ref_before
        delay_rate = float(np.mean(ref_advance - red_advance) / (18 / self.task.policy_frequency))
        # Do not give positive attack credit on a contact/offroad transition.
        if info["bad"]:
            delay_rate = 0.0
        self.hard_brakes += sum(v.action["acceleration"] < -3 for v in cars[:2])
        lateral = np.array([v.lane.local_coordinates(v.position)[1] for v in cars[2:]])
        rew = reward(self.task, np.array([v.position[0] for v in cars[2:]]) - before[2:],
                     np.array([v.speed for v in cars[2:]]), lateral,
                     np.array([v.heading for v in cars[2:]]), info["collision"], info["offroad"], delay_rate)
        terminated = bool(info["bad"] or info["success"])
        truncated = bool(self.steps >= round(self.task.duration * self.task.policy_frequency) and not terminated)
        self.ended = terminated or truncated
        return self._obs(), rew, terminated, truncated, self._info()

    def _measure_ttc(self):
        for red in self.road.vehicles[:2]:
            for blue in self.road.vehicles[2:]:
                dx = blue.position[0] - red.position[0]
                closing = (red.velocity[0] - blue.velocity[0]) * np.sign(dx)
                if abs(blue.position[1] - red.position[1]) < 2 and closing > 0.1:
                    ttc = max(abs(dx) - (red.LENGTH + blue.LENGTH) / 2, 0) / closing
                    self.min_ttc = min(self.min_ttc, float(ttc))
                    self.event_seen |= ttc < self.task.ttc_threshold

    def render(self):
        from PIL import Image, ImageDraw
        canvas = Image.new("RGB", (900, 220), "#294637")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 35, 900, 175), fill="#363b43")
        draw.line((0, 105, 900, 105), fill="white", width=2)
        center = np.mean([v.position[0] for v in self.road.vehicles])
        for i, v in enumerate(self.road.vehicles):
            x, y = 450 + (v.position[0] - center) * 5, 70 + v.position[1] * 17.5
            corners = np.array([[-13, -10], [13, -10], [13, 10], [-13, 10]])
            rotation = np.array([[np.cos(v.heading), -np.sin(v.heading)],
                                 [np.sin(v.heading), np.cos(v.heading)]])
            points = corners @ rotation.T + [x, y]
            draw.polygon([tuple(p) for p in points], fill="#ef6666" if i < 2 else "#65bfff")
            draw.text((x - 8, y - 8), str(i), fill="black")
            draw.text((x - 15, y + 14), f"{v.speed:.1f}m/s", fill="white")
        draw.text((10, 10), f"{self.family} | step={self.steps} | red=IDM blue=learners", fill="white")
        info = self._info()
        draw.text((10, 195), f"red delay={info['red_delay_m']:.1f}m | blue progress={info['blue_progress_m']:.1f}m | "
                  f"success={info['success']} collision={info['collision']} offroad={info['offroad']}", fill="white")
        return np.array(canvas)
