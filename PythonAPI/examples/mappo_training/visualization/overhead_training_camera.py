"""A monitor-only RGB sensor. The caller owns all simulation ticks."""

from __future__ import annotations

import math
import threading
import weakref
from numbers import Integral

import numpy as np


class _SceneMailbox:
    """Callbacks can only change the mailbox for the scene they observed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.minimum_frame = None
        self.latest = (None, None)


def _finite_number(name, value, *, allow_zero=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"monitor {name} must be a finite number") from None
    if isinstance(value, bool) or not math.isfinite(number) or (
        number < 0 if allow_zero else number <= 0
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"monitor {name} must be finite and {qualifier}")
    return number


class OverheadTrainingCamera:
    def __init__(
        self, *, width: int, fps: float, height: float = 35.0,
        smoothing_seconds: float = 0.5,
    ):
        if isinstance(width, bool) or not isinstance(width, Integral) or width < 160:
            raise ValueError("monitor width must be an integer of at least 160 pixels")
        self.width = int(width)
        self.image_height = round(self.width * 0.75)
        self.fps = _finite_number("fps", fps)
        self.height = _finite_number("overhead height", height)
        self.smoothing_seconds = _finite_number(
            "smoothing seconds", smoothing_seconds, allow_zero=True
        )
        self.sensor = None
        self._world = None
        self._cleanup_pending = False
        self._generation = 0
        self._mailbox = _SceneMailbox()
        self._center = None
        self._altitude = None
        self._last_position_frame = None

    @property
    def latest(self):
        mailbox = self._mailbox
        with mailbox.lock:
            return mailbox.latest

    @latest.setter
    def latest(self, value):
        mailbox = self._mailbox
        with mailbox.lock:
            mailbox.latest = value

    def reset_view(self) -> None:
        """Start a new scene without destroying a sensor in the same world."""
        self._mailbox = _SceneMailbox()
        self._center = None
        self._altitude = None
        self._last_position_frame = None

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

        if self.sensor is not None and (
            self._cleanup_pending or self._world is not env.world
            or not self.sensor.is_alive
        ):
            self.close()

        frame = int(getattr(snapshot, "frame", 0))
        target_x = (min(p.x for p in locations) + max(p.x for p in locations)) / 2.0
        target_y = (min(p.y for p in locations) + max(p.y for p in locations)) / 2.0
        first_position = self._center is None
        if first_position or self.smoothing_seconds == 0:
            center_x, center_y = target_x, target_y
            alpha = 1.0
        else:
            timestamp = getattr(snapshot, "timestamp", None)
            delta = float(getattr(timestamp, "delta_seconds", 0.0))
            if not math.isfinite(delta) or delta < 0:
                delta = 0.0
            if frame == self._last_position_frame:
                delta = 0.0
            alpha = -math.expm1(-delta / self.smoothing_seconds)
            center_x = self._center[0] + alpha * (target_x - self._center[0])
            center_y = self._center[1] + alpha * (target_y - self._center[1])

        # At pitch=-90, yaw=0, world Y spans image width, world X its height.
        # Recompute bounds around the smoothed center so lag cannot crop cars.
        aspect = self.width / self.image_height
        required_altitude = max(
            self.height,
            max(abs(p.y - center_y) for p in locations) + 8.0,
            (max(abs(p.x - center_x) for p in locations) + 8.0) * aspect,
        )
        if self._altitude is None or required_altitude >= self._altitude:
            altitude = required_altitude
        else:
            altitude = max(
                required_altitude,
                self._altitude + alpha * (required_altitude - self._altitude),
            )
        transform = carla.Transform(
            carla.Location(
                x=center_x, y=center_y,
                z=max(p.z for p in locations) + altitude,
            ),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
        )
        if self.sensor is not None:
            self.sensor.set_transform(transform)
        else:
            blueprint = env.world.get_blueprint_library().find("sensor.camera.rgb")
            for key, value in {
                "image_size_x": self.width,
                "image_size_y": self.image_height,
                "fov": 90,
                "sensor_tick": 1.0 / self.fps,
                "enable_postprocess_effects": "false",
            }.items():
                blueprint.set_attribute(key, str(value))
            self._generation += 1
            generation = self._generation
            self.sensor = env.world.spawn_actor(blueprint, transform)
            self._world = env.world
            weak_self = weakref.ref(self)

            def receive(image):
                camera = weak_self()
                if camera is None or generation != camera._generation:
                    return
                mailbox = camera._mailbox
                image_frame = int(image.frame)
                with mailbox.lock:
                    if mailbox.minimum_frame is None or image_frame < mailbox.minimum_frame:
                        return
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                bgr = array.reshape(image.height, image.width, 4)[:, :, :3].copy()
                with mailbox.lock:
                    if generation != camera._generation or mailbox is not camera._mailbox:
                        return
                    previous_frame = mailbox.latest[1]
                    if previous_frame is None or image_frame > previous_frame:
                        mailbox.latest = (bgr, image_frame)

            self.sensor.listen(receive)

        self._center = (center_x, center_y)
        self._altitude = altitude
        self._last_position_frame = frame
        if first_position:
            with self._mailbox.lock:
                # Reset settling frames predate this next training tick, even
                # when their sensor callbacks arrive after the scene changes.
                self._mailbox.minimum_frame = frame + 1

    def close(self) -> None:
        """Invalidate delayed callbacks, then stop and destroy our own sensor."""
        self._generation += 1
        self.reset_view()
        sensor = self.sensor
        if sensor is not None and sensor.is_alive:
            self._cleanup_pending = True
            try:
                sensor.stop()
            finally:
                if sensor.destroy() is False:
                    raise RuntimeError("failed to destroy overhead camera")
                self.sensor = None
                self._world = None
                self._cleanup_pending = False
        else:
            self.sensor = None
            self._world = None
            self._cleanup_pending = False
