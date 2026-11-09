"""Bounded, discount-correct terminal objectives, independent of simulator."""
import numpy as np


class PairedObjective:
    def __init__(self, task, gamma, *, red_ids=(0,1), blue_ids=(2,3), goal_distances=None):
        self.task, self.gamma = task, gamma
        self.red_ids,self.blue_ids=tuple(red_ids),tuple(blue_ids)
        self.goal_distances=goal_distances or {}
        self.drive_phi = np.zeros(2)
        self.attack_phi = 0.0

    def step(self, frame, safety, normal_red_progress, *, baseline_valid):
        task = self.task
        blue = [frame.vehicles[i] for i in self.blue_ids]
        next_drive = np.clip([v.progress / self.goal_distances.get(v.identity,task.goal_distance) for v in blue], 0, 1)
        actual_red = float(np.mean([frame.vehicles[i].progress for i in self.red_ids]))
        next_attack = float(np.clip((normal_red_progress - actual_red) / task.goal_distance, 0, 1))
        failure, horizon = safety["bad"], safety["fixed_horizon_complete"]
        if failure:
            next_drive[:] = 0
        if failure or (horizon and (not safety["qualified"] or not baseline_valid)):
            next_attack = 0.0
        n = round(task.duration * task.policy_frequency)
        speed_bonus = 0.1 / n * np.clip([v.speed / 20 for v in blue], 0, 1)
        lane_cost = 0.1 / n * np.minimum(np.abs([v.lane_error for v in blue]) / 2, 1)
        heading_cost = 0.1 / n * np.abs(np.sin([v.heading - v.route_heading for v in blue]))
        drive = self.gamma * next_drive - self.drive_phi
        attack = task.attack_weight * (self.gamma * next_attack - self.attack_phi)
        penalty = (task.collision_penalty if safety["collision"] else task.offroad_penalty) if failure else 0
        reward = drive + attack + speed_bonus - lane_cost - heading_cost - 0.02 / n - penalty
        if horizon and safety["qualified"]:
            reward += 0.5
        self.drive_phi, self.attack_phi = next_drive, next_attack
        return reward.astype(np.float32), {"drive": drive.tolist(), "attack": float(attack), "failure_cost": float(penalty)}
