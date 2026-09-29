"""A monitor-only RGB sensor. The caller owns all simulation ticks."""

from __future__ import annotations

import math
import weakref

import numpy as np


class OverheadTrainingCamera:
    def __init__(self, *, width: int, fps: float, height: float = 35.0):
        if not math.isfinite(height) or height <= 0:
            raise ValueError("monitor overhead height must be finite and positive")
        self.width = int(width)
        self.image_height = round(self.width * 0.75)
        self.fps = float(fps)
        self.height = float(height)
        self.sensor = None
        self.latest = (None, None)  # (BGR image, CARLA frame number)
        self._generation = 0

    def update(self, env) -> None:
        """Position before the training tick; never tick or wait for an image."""
        import carla

        snapshot = env.world.get_snapshot()
        locations = []
        for vehicle in env.vehicles:
            if vehicle is not None and vehicle.is_alive:
                actor = snapshot.find(vehicle.id)
                if actor is not None:
                    locations.append(actor.get_transform().location)
        if not locations:
            return

        min_x, max_x = min(p.x for p in locations), max(p.x for p in locations)
        min_y, max_y = min(p.y for p in locations), max(p.y for p in locations)
        # At pitch=-90, yaw=0, world Y spans image width, world X its height.
        # Horizontal FOV is 90 degrees; leave 8 m per side for car size/motion.
        aspect = self.width / self.image_height
        altitude = max(self.height, (max_y - min_y + 16.0) / 2.0,
                       (max_x - min_x + 16.0) * aspect / 2.0)
        transform = carla.Transform(
            carla.Location(x=(min_x + max_x) / 2.0, y=(min_y + max_y) / 2.0,
                           z=max(p.z for p in locations) + altitude),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.set_transform(transform)
            return

        self.close()
        blueprint = env.world.get_blueprint_library().find("sensor.camera.rgb")
        for key, value in {
            "image_size_x": self.width,
            "image_size_y": self.image_height,
            "fov": 90,
            "sensor_tick": 1.0 / self.fps,
            "enable_postprocess_effects": "false",
        }.items():
            blueprint.set_attribute(key, str(value))
        # Independent world camera: no attachment or modification to a vehicle.
        self.sensor = env.world.spawn_actor(blueprint, transform)
        weak_self = weakref.ref(self)
        generation = self._generation

        def receive(image):
            camera = weak_self()
            if camera is None or generation != camera._generation:
                return
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            bgr = array.reshape(image.height, image.width, 4)[:, :, :3].copy()
            if generation == camera._generation:
                camera.latest = (bgr, image.frame)

        self.sensor.listen(receive)

    def close(self) -> None:
        """Invalidate delayed callbacks, then stop and destroy our own sensor."""
        self._generation += 1
        self.latest = (None, None)
        sensor = self.sensor
        if sensor is not None and sensor.is_alive:
            try:
                sensor.stop()
            finally:
                if sensor.destroy() is False:
                    raise RuntimeError("failed to destroy overhead camera")
                self.sensor = None
        else:
            self.sensor = None
