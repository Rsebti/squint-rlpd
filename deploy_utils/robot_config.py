"""Robot configuration for real deployment.

Note for macOS: lerobot's OpenCVCamera uses cv2.CAP_ANY which flakily times
out on macOS. We bypass it with Cv2Camera (CAP_AVFOUNDATION + background
reader, mirroring infer.py's workaround) and inject it into robot.cameras
after construction.
"""
import sys
import threading
import time
from pathlib import Path

import cv2

from lerobot.robots.robot import Robot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig, SO100FollowerConfig

# Edit these for your setup
ROBOT_PORT = "/dev/tty.usbmodem5B141129871"
CAMERA_INDEX = 0           # macOS: integer index. Try 1 if 0 picks the wrong cam.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30


class Cv2Camera:
    """macOS-safe drop-in replacement for lerobot's OpenCVCamera.

    Uses cv2.CAP_AVFOUNDATION explicitly and runs a tiny background reader
    so async_read() just hands back the latest frame. Satisfies the lerobot
    Camera interface (is_connected, connect, disconnect, async_read).
    """

    def __init__(self, index: int, width: int = 1280, height: int = 720, fps: int = 30):
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self.cap = None
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            return
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self._index, backend)
        if not self.cap.isOpened():
            raise ConnectionError(f"Cv2Camera({self._index}) failed to open.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self.cap.set(cv2.CAP_PROP_FPS, self._fps)
        # AVFoundation often needs warmup + a few discarded frames.
        frame = None
        for _ in range(90):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.033)
        if frame is None:
            self.cap.release()
            self.cap = None
            raise ConnectionError(f"Cv2Camera({self._index}) opened but no frame after 3 s.")
        with self._lock:
            self._latest = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest = rgb

    def async_read(self, timeout_ms: float = 1000.0):
        with self._lock:
            if self._latest is None:
                raise RuntimeError("Cv2Camera: no frame available yet.")
            return self._latest.copy()

    def read(self, color_mode=None):
        return self.async_read()

    def disconnect(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def create_real_robot() -> Robot:
    """Create and configure a real robot. Camera is injected post-hoc
    (Cv2Camera) because lerobot's OpenCVCamera is unreliable on macOS."""
    robot_config = SO101FollowerConfig(
        port=ROBOT_PORT,
        use_degrees=True,
        cameras={},  # injected below; bypasses broken lerobot OpenCVCamera on macOS
        id="so101_follower_arm",
        calibration_dir=Path(__file__).parent.parent,
    )
    robot = make_robot_from_config(robot_config)
    robot.cameras["base_camera"] = Cv2Camera(
        index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
    )
    return robot
