import math
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
        self.snapshot = SimpleNamespace(
            frame=10, timestamp=SimpleNamespace(delta_seconds=0.1),
            find=lambda aid: SimpleNamespace(get_transform=lambda: SimpleNamespace(
                location=self.locations[aid])),
        )
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

    def advance(self, *, delta=0.1):
        self.snapshot.frame += 1
        self.snapshot.timestamp.delta_seconds = delta

    def transform(self):
        if self.sensor.set_transform.called:
            return self.sensor.set_transform.call_args.args[0]
        return self.world.spawn_actor.call_args.args[1]

    @staticmethod
    def image(frame):
        pixels = np.zeros((2, 3, 4), dtype=np.uint8)
        pixels[:, :, 2] = 255
        return SimpleNamespace(
            raw_data=pixels.tobytes(), height=2, width=3, frame=frame
        )

    def assert_cars_covered(self):
        transform = self.transform()
        altitude = transform.location.z - max(p.z for p in self.locations.values())
        self.assertGreaterEqual(altitude, 35)
        for point in self.locations.values():
            self.assertLessEqual(abs(point.y - transform.location.y) + 8, altitude + 1e-9)
            self.assertLessEqual(
                abs(point.x - transform.location.x) + 8,
                altitude * self.camera.image_height / self.camera.width + 1e-9,
            )

    def test_follow_fits_all_cars_and_road_elevation_without_tick(self):
        self.camera.update(self.env)
        transform = self.transform()
        self.assertEqual(transform.rotation.pitch, -90)
        self.assertEqual((transform.location.x, transform.location.y), (50, 10))
        self.assert_cars_covered()
        self.assertEqual(self.world.spawn_actor.call_args.kwargs, {})
        self.world.tick.assert_not_called()
        self.world.wait_for_tick.assert_not_called()
        blueprint = self.world.get_blueprint_library.return_value.find.return_value
        blueprint.set_attribute.assert_any_call("sensor_tick", "0.2")

    def test_center_follows_with_simulation_time_smoothing(self):
        self.camera.update(self.env)
        for point in self.locations.values():
            point.x += 10
            point.y += 4
        self.advance(delta=0.25)
        self.camera.update(self.env)
        alpha = 1 - math.exp(-0.25 / 0.5)
        transform = self.transform()
        self.assertAlmostEqual(transform.location.x, 50 + 10 * alpha)
        self.assertAlmostEqual(transform.location.y, 10 + 4 * alpha)
        self.assert_cars_covered()
        self.world.spawn_actor.assert_called_once()
        self.world.tick.assert_not_called()
        self.world.wait_for_tick.assert_not_called()

    def test_same_simulation_frame_does_not_advance_smoothing(self):
        self.camera.update(self.env)
        for point in self.locations.values():
            point.x += 10
        self.camera.update(self.env)
        self.assertEqual(self.transform().location.x, 50)

    def test_zoom_out_immediately_covers_suddenly_distributed_cars(self):
        self.locations[12].x = 10
        self.locations[12].y = 0
        self.camera.update(self.env)
        self.assertEqual(self.transform().location.z - 7, 35)
        self.locations[12].x = 300
        self.locations[12].y = 200
        self.locations[12].z = 25
        self.advance()
        self.camera.update(self.env)
        self.assert_cars_covered()
        self.assertGreater(self.transform().location.z - 25, 35)

    def test_zoom_in_is_smooth_and_accounts_for_new_ground_height(self):
        self.camera.update(self.env)
        old_altitude = self.transform().location.z - 7
        self.locations[11] = SimpleNamespace(x=49, y=10, z=30)
        self.locations[12] = SimpleNamespace(x=51, y=10, z=30)
        self.advance()
        self.camera.update(self.env)
        altitude = self.transform().location.z - 30
        expected = old_altitude + (1 - math.exp(-0.1 / 0.5)) * (35 - old_altitude)
        self.assertAlmostEqual(altitude, expected)
        self.assertGreater(altitude, 35)
        self.assertLess(altitude, old_altitude)
        self.assert_cars_covered()

    def test_zero_smoothing_follows_immediately(self):
        camera = OverheadTrainingCamera(width=640, fps=5, smoothing_seconds=0)
        self.addCleanup(camera.close)
        camera.update(self.env)
        for point in self.locations.values():
            point.x += 10
        self.advance()
        camera.update(self.env)
        self.assertEqual(self.transform().location.x, 60)

    def test_reset_reuses_sensor_repositions_directly_and_rejects_settling_frames(self):
        self.camera.update(self.env)
        receive = self.sensor.listen.call_args.args[0]
        receive(self.image(11))
        self.assertEqual(self.camera.latest[1], 11)
        self.camera.reset_view()
        self.assertEqual(self.camera.latest, (None, None))
        receive(self.image(12))
        self.assertEqual(self.camera.latest, (None, None))
        for point in self.locations.values():
            point.x += 1000
        self.snapshot.frame = 50
        self.camera.update(self.env)
        self.world.spawn_actor.assert_called_once()
        self.sensor.listen.assert_called_once()
        self.sensor.stop.assert_not_called()
        self.sensor.destroy.assert_not_called()
        self.assertEqual(self.transform().location.x, 1050)
        receive(self.image(49))
        receive(self.image(50))
        self.assertEqual(self.camera.latest, (None, None))
        receive(self.image(51))
        self.assertEqual(self.camera.latest[1], 51)
        np.testing.assert_array_equal(
            self.camera.latest[0],
            np.frombuffer(self.image(51).raw_data, dtype=np.uint8).reshape(2, 3, 4)[:, :, :3],
        )

    def test_reset_during_callback_cannot_publish_into_new_scene(self):
        self.camera.update(self.env)
        receive = self.sensor.listen.call_args.args[0]
        original_frombuffer = np.frombuffer

        def reset_while_decoding(*args, **kwargs):
            self.camera.reset_view()
            self.snapshot.frame = 30
            self.camera.update(self.env)
            return original_frombuffer(*args, **kwargs)

        with patch("visualization.overhead_training_camera.np.frombuffer",
                   side_effect=reset_while_decoding):
            receive(self.image(11))
        self.assertEqual(self.camera.latest, (None, None))
        receive(self.image(31))
        self.assertEqual(self.camera.latest[1], 31)

    def test_out_of_order_image_cannot_replace_newer_scene_frame(self):
        self.camera.update(self.env)
        receive = self.sensor.listen.call_args.args[0]
        receive(self.image(15))
        receive(self.image(14))
        self.assertEqual(self.camera.latest[1], 15)

    def test_latest_property_remains_assignable(self):
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        self.camera.latest = (image, 123)
        self.assertIs(self.camera.latest[0], image)
        self.assertEqual(self.camera.latest[1], 123)
        self.camera.reset_view()
        self.assertEqual(self.camera.latest, (None, None))

    def test_close_rejects_old_sensor_callbacks_and_recreates_sensor(self):
        self.camera.update(self.env)
        receive = self.sensor.listen.call_args.args[0]
        receive(self.image(11))
        self.camera.close()
        self.sensor.stop.assert_called_once()
        self.sensor.destroy.assert_called_once()
        receive(self.image(15))
        self.assertEqual(self.camera.latest, (None, None))
        self.camera.update(self.env)
        self.assertEqual(self.world.spawn_actor.call_count, 2)
        receive(self.image(15))
        self.assertEqual(self.camera.latest, (None, None))
        self.sensor.listen.call_args.args[0](self.image(11))
        self.assertEqual(self.camera.latest[1], 11)

    def test_world_change_releases_old_sensor_before_spawning(self):
        self.camera.update(self.env)
        old_receive = self.sensor.listen.call_args.args[0]
        new_world = Mock()
        new_world.get_snapshot.return_value = self.snapshot
        new_sensor = Mock(is_alive=True)
        new_world.spawn_actor.return_value = new_sensor
        self.env.world = new_world
        self.camera.update(self.env)
        self.sensor.stop.assert_called_once()
        self.sensor.destroy.assert_called_once()
        self.assertIs(self.camera.sensor, new_sensor)
        old_receive(self.image(100))
        self.assertEqual(self.camera.latest, (None, None))
        new_sensor.listen.call_args.args[0](self.image(11))
        self.assertEqual(self.camera.latest[1], 11)

    def test_destroy_attempted_even_when_stop_fails(self):
        self.camera.update(self.env)
        self.sensor.stop.side_effect = RuntimeError("stop failed")
        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            self.camera.close()
        self.sensor.destroy.assert_called_once()
        self.assertIsNone(self.camera.sensor)
        self.camera.close()

    def test_no_live_vehicles_does_not_spawn_or_tick(self):
        for vehicle in self.env.vehicles:
            vehicle.is_alive = False
        self.camera.update(self.env)
        self.world.spawn_actor.assert_not_called()
        self.world.tick.assert_not_called()
        self.world.wait_for_tick.assert_not_called()

    def test_failed_destroy_retains_sensor_for_cleanup_retry(self):
        self.camera.update(self.env)
        self.sensor.destroy.return_value = False
        with self.assertRaisesRegex(RuntimeError, "failed to destroy"):
            self.camera.close()
        self.assertIs(self.camera.sensor, self.sensor)
        self.sensor.destroy.return_value = True
        self.camera.close()
        self.assertIsNone(self.camera.sensor)

    def test_update_retries_failed_cleanup_before_reusing_sensor(self):
        self.camera.update(self.env)
        self.sensor.destroy.return_value = False
        with self.assertRaisesRegex(RuntimeError, "failed to destroy"):
            self.camera.close()
        self.sensor.destroy.return_value = True
        new_sensor = Mock(is_alive=True)
        self.world.spawn_actor.return_value = new_sensor
        self.camera.update(self.env)
        self.assertEqual(self.sensor.destroy.call_count, 2)
        self.assertIs(self.camera.sensor, new_sensor)

    def test_constructor_validation(self):
        valid = {"width": 640, "fps": 5, "height": 35, "smoothing_seconds": 0.5}
        invalid_values = {
            "width": (0, 159, -1, 640.5, True, float("nan"), float("inf")),
            "fps": (0, -1, True, float("nan"), float("inf")),
            "height": (0, -1, True, float("nan"), float("inf")),
            "smoothing_seconds": (-1, True, float("nan"), float("inf")),
        }
        for field, values in invalid_values.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    OverheadTrainingCamera(**{**valid, field: value})


if __name__ == "__main__":
    unittest.main()
