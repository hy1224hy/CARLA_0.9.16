import math
import unittest
from types import SimpleNamespace

import carla

from environment.carla_multi_agent_env import CarlaMultiAgentEnv


class ScenarioInitialSpeedTests(unittest.TestCase):
    def _make_env_without_server(self):
        env = CarlaMultiAgentEnv.__new__(CarlaMultiAgentEnv)
        env.num_agents = 2
        env.start_waypoints = []
        env.custom_start_transforms = None
        env.custom_initial_speeds_mps = {}
        return env

    def test_custom_positions_parse_explicit_initial_speeds(self):
        env = self._make_env_without_server()
        env.set_custom_start_positions({
            "agent0": {
                "x": 1.0,
                "y": 2.0,
                "z": 0.5,
                "yaw": 0.0,
                "initial_speed_mps": 3.0,
            },
            "agent1": {
                "x": 4.0,
                "y": 5.0,
                "z": 0.5,
                "yaw": 90.0,
                "initial_speed_mps": 5.0,
            },
        })
        self.assertEqual(env.custom_initial_speeds_mps, {0: 3.0, 1: 5.0})
        self.assertEqual(len(env.custom_start_transforms), 2)

    def test_invalid_initial_speed_is_rejected(self):
        for speed in (-0.1, 15.1, math.inf):
            with self.subTest(speed=speed):
                env = self._make_env_without_server()
                with self.assertRaisesRegex(ValueError, "initial_speed_mps"):
                    env.set_custom_start_positions({
                        "agent0": {
                            "x": 1.0,
                            "y": 2.0,
                            "yaw": 0.0,
                            "initial_speed_mps": speed,
                        }
                    })

    def test_initial_velocity_is_aligned_with_spawn_heading(self):
        class FakeTransform:
            @staticmethod
            def get_forward_vector():
                return SimpleNamespace(x=0.6, y=0.8, z=0.0)

        class FakeVehicle:
            is_alive = True

            def __init__(self):
                self.velocity = None

            @staticmethod
            def get_transform():
                return FakeTransform()

            def set_target_velocity(self, velocity):
                self.velocity = velocity

        moving = FakeVehicle()
        stationary = FakeVehicle()
        env = self._make_env_without_server()
        env.vehicles = [moving, stationary]
        env.custom_initial_speeds_mps = {0: 5.0, 1: 0.0}

        env._apply_initial_vehicle_speeds()

        self.assertAlmostEqual(moving.velocity.x, 3.0)
        self.assertAlmostEqual(moving.velocity.y, 4.0)
        self.assertAlmostEqual(moving.velocity.z, 0.0)
        self.assertIsNone(stationary.velocity)

    def test_velocity_warm_start_preoffsets_xy_and_keeps_settled_z(self):
        class FakeVehicle:
            is_alive = True

            def __init__(self, z):
                self.transform = carla.Transform(carla.Location(z=z))

            def get_transform(self):
                return self.transform

            def set_transform(self, transform):
                self.transform = transform

        class FakeWorld:
            @staticmethod
            def get_settings():
                return SimpleNamespace(fixed_delta_seconds=0.1)

        env = self._make_env_without_server()
        env.world = FakeWorld()
        env.vehicles = [FakeVehicle(0.22), FakeVehicle(0.23)]
        env.custom_start_transforms = [
            carla.Transform(
                carla.Location(x=10.0, y=20.0, z=0.5),
                carla.Rotation(yaw=0.0),
            ),
            carla.Transform(
                carla.Location(x=30.0, y=40.0, z=0.5),
                carla.Rotation(yaw=90.0),
            ),
        ]
        env.custom_initial_speeds_mps = {0: 5.0, 1: 3.0}

        env._prepare_velocity_warm_start()

        first = env.vehicles[0].transform.location
        second = env.vehicles[1].transform.location
        self.assertAlmostEqual(first.x, 9.5, places=5)
        self.assertAlmostEqual(first.y, 20.0, places=5)
        self.assertAlmostEqual(first.z, 0.22, places=5)
        self.assertAlmostEqual(second.x, 30.0, places=5)
        self.assertAlmostEqual(second.y, 39.7, places=5)
        self.assertAlmostEqual(second.z, 0.23, places=5)


if __name__ == "__main__":
    unittest.main()
