import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from visualization.overhead_training_camera import OverheadTrainingCamera


class OverheadTrainingCameraTests(unittest.TestCase):
    def setUp(self):
        fake_carla = SimpleNamespace(
            Location=lambda **kw: SimpleNamespace(**kw),
            Rotation=lambda **kw: SimpleNamespace(**kw),
            Transform=lambda location, rotation: SimpleNamespace(
                location=location, rotation=rotation),
        )
        self.patch = patch.dict(sys.modules, {"carla": fake_carla})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.locations = {
            11: SimpleNamespace(x=0.0, y=0.0, z=5.0),
            12: SimpleNamespace(x=100.0, y=20.0, z=7.0),
        }
        self.snapshot = SimpleNamespace(find=lambda aid: SimpleNamespace(
            get_transform=lambda: SimpleNamespace(location=self.locations[aid])))
        self.world = Mock()
        self.world.get_snapshot.return_value = self.snapshot
        self.sensor = Mock(is_alive=True)
        self.world.spawn_actor.return_value = self.sensor
        self.env = SimpleNamespace(world=self.world, vehicles=[
            SimpleNamespace(id=11, is_alive=True),
            SimpleNamespace(id=12, is_alive=True),
        ])
        self.camera = OverheadTrainingCamera(width=640, fps=5, height=35)
        self.addCleanup(self.camera.close)

    def test_follow_fits_all_cars_and_road_elevation_without_tick(self):
        self.camera.update(self.env)
        transform = self.world.spawn_actor.call_args.args[1]
        self.assertEqual(transform.rotation.pitch, -90)
        self.assertEqual((transform.location.x, transform.location.y), (50, 10))
        altitude = transform.location.z - 7
        self.assertGreaterEqual(altitude, 35)
        self.assertGreaterEqual(2 * altitude * 480 / 640, 100 + 16)
        self.assertGreaterEqual(2 * altitude, 20 + 16)
        self.assertEqual(self.world.spawn_actor.call_args.kwargs, {})
        self.world.tick.assert_not_called()
        self.world.wait_for_tick.assert_not_called()
        blueprint = self.world.get_blueprint_library.return_value.find.return_value
        blueprint.set_attribute.assert_any_call("sensor_tick", "0.2")
        self.locations[11].x += 10
        self.locations[12].x += 10
        self.camera.update(self.env)
        self.world.spawn_actor.assert_called_once()
        self.assertEqual(self.sensor.set_transform.call_args.args[0].location.x, 60)

    def test_reset_rejects_late_frames_and_recreates_sensor(self):
        self.camera.update(self.env)
        receive = self.sensor.listen.call_args.args[0]
        pixels = np.zeros((2, 3, 4), dtype=np.uint8)
        pixels[:, :, 2] = 255
        image = SimpleNamespace(raw_data=pixels.tobytes(), height=2, width=3, frame=9)
        receive(image)
        self.assertEqual(self.camera.latest[1], 9)
        np.testing.assert_array_equal(self.camera.latest[0], pixels[:, :, :3])
        self.camera.close()
        self.sensor.stop.assert_called_once()
        self.sensor.destroy.assert_called_once()
        receive(image)
        self.assertEqual(self.camera.latest, (None, None))
        self.camera.update(self.env)
        self.assertEqual(self.world.spawn_actor.call_count, 2)
        receive(image)
        self.assertEqual(self.camera.latest, (None, None))
        self.sensor.listen.call_args.args[0](image)
        self.assertEqual(self.camera.latest[1], 9)

    def test_destroy_attempted_even_when_stop_fails(self):
        self.camera.update(self.env)
        self.sensor.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            self.camera.close()
        self.sensor.destroy.assert_called_once()
        self.camera.close()

    def test_no_live_vehicles_does_not_spawn(self):
        for vehicle in self.env.vehicles:
            vehicle.is_alive = False
        self.camera.update(self.env)
        self.world.spawn_actor.assert_not_called()

    def test_failed_destroy_retains_sensor_for_cleanup_retry(self):
        self.camera.update(self.env)
        self.sensor.destroy.return_value = False
        with self.assertRaisesRegex(RuntimeError, "failed to destroy"):
            self.camera.close()
        self.assertIs(self.camera.sensor, self.sensor)
        self.sensor.destroy.return_value = True
        self.camera.close()
        self.assertIsNone(self.camera.sensor)

    def test_height_validation(self):
        for height in (0, -1, float("nan"), float("inf")):
            with self.subTest(height=height), self.assertRaises(ValueError):
                OverheadTrainingCamera(width=640, fps=5, height=height)


if __name__ == "__main__":
    unittest.main()
