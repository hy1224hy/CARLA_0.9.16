"""One reward definition for every algorithm group; bounded reference delay."""
import numpy as np


def reward(task, progress, speed, lateral, heading, collision, offroad, red_delay_rate):
    driving = (0.6 * np.clip(progress / (20 / task.policy_frequency), -1, 1)
               + 0.2 * np.clip(speed / 20, 0, 1)
               - 0.1 * np.minimum(np.abs(lateral) / 2, 1)
               - 0.1 * np.abs(np.sin(heading)) - 0.02)
    attack = task.attack_weight * np.clip(red_delay_rate, 0, 1)
    return (driving + attack - task.collision_penalty * collision
            - task.offroad_penalty * offroad).astype(np.float32)
