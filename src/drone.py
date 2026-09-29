"""
Drone abstraction layer — hardware UDP driver, virtual simulator, and auto-detection.
"""
import os
import socket
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import (
    DRONE_IP, DRONE_PORT, LOCAL_PORT, VIDEO_STREAM_URL, CONNECT_TIMEOUT,
)


# ── Shared helpers ───────────────────────────────────────────────────

class FrameHolder:
    """Thin wrapper around a single video frame."""
    __slots__ = ("frame",)

    def __init__(self, frame: Optional[np.ndarray] = None):
        self.frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)


# ── Abstract interface ───────────────────────────────────────────────

class BaseDrone(ABC):
    """Contract every drone backend must satisfy."""

    def __init__(self):
        self.left_right_velocity = 0
        self.for_back_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 0
        self.battery = 100

    @abstractmethod
    def connect(self) -> bool: ...
    @abstractmethod
    def get_battery(self) -> int: ...
    @abstractmethod
    def streamon(self) -> None: ...
    @abstractmethod
    def streamoff(self) -> None: ...
    @abstractmethod
    def takeoff(self) -> None: ...
    @abstractmethod
    def land(self) -> None: ...
    @abstractmethod
    def send_rc_control(self, lr: int, fb: int, ud: int, yaw: int) -> None: ...
    @abstractmethod
    def get_frame_read(self) -> FrameHolder: ...

    def rotate_clockwise(self, deg: int) -> None:  # optional
        pass

    def move_left(self, cm: int) -> None:  # optional
        pass


# ── Hardware drone (Tello-style UDP) ─────────────────────────────────

class HardwareDrone(BaseDrone):
    """Real drone communicating over UDP sockets."""

    def __init__(self, host=DRONE_IP, port=DRONE_PORT):
        super().__init__()
        self.host, self.port = host, port
        self.sock: Optional[socket.socket] = None
        self.cap: Optional[cv2.VideoCapture] = None
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("", LOCAL_PORT))
            self.sock.settimeout(1.0)
        except Exception as exc:
            print(f"[HARDWARE] Socket init: {exc}")

    def _cmd(self, cmd: str) -> str:
        if self.sock is None:
            return "error"
        try:
            self.sock.sendto(cmd.encode(), (self.host, self.port))
            data, _ = self.sock.recvfrom(1024)
            return data.decode().strip()
        except Exception:
            return "error"

    def connect(self) -> bool:
        ok = self._cmd("command") == "ok"
        print(f"[HARDWARE] {'Connected' if ok else 'Connection failed'}")
        return ok

    def get_battery(self) -> int:
        try:
            self.battery = int(self._cmd("battery?"))
        except Exception:
            pass
        return self.battery

    def streamon(self):
        self._cmd("streamon")
        if self.cap is None:
            self.cap = cv2.VideoCapture(VIDEO_STREAM_URL, cv2.CAP_FFMPEG)

    def streamoff(self):
        self._cmd("streamoff")
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def takeoff(self):
        self._cmd("takeoff")

    def land(self):
        self._cmd("land")

    def rotate_clockwise(self, deg: int):
        self._cmd(f"cw {deg}")

    def move_left(self, cm: int):
        self._cmd(f"left {cm}")

    def send_rc_control(self, lr, fb, ud, yaw):
        self.left_right_velocity, self.for_back_velocity = lr, fb
        self.up_down_velocity, self.yaw_velocity = ud, yaw
        self._cmd(f"rc {lr} {fb} {ud} {yaw}")

    def get_frame_read(self) -> FrameHolder:
        h = FrameHolder()
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                h.frame = frame
        return h

    def close(self):
        self.streamoff()
        if self.sock:
            self.sock.close()
            self.sock = None


# ── Simulated drone (offline video / webcam) ─────────────────────────

class SimulatedDrone(BaseDrone):
    """Virtual drone that plays a video file or webcam feed."""

    def __init__(self, demo_video: Optional[str] = None):
        super().__init__()
        self.battery = 88
        self.is_flying = False
        self.demo_video = demo_video
        self.use_webcam = False
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_source()

    def _open_source(self):
        if self.cap is not None:
            self.cap.release()
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)
        elif self.demo_video and os.path.exists(self.demo_video):
            self.cap = cv2.VideoCapture(self.demo_video)
        else:
            self.cap = cv2.VideoCapture(0)

    def toggle_source(self):
        """Switch between demo video and webcam."""
        self.use_webcam = not self.use_webcam
        self._open_source()

    def connect(self) -> bool:
        return True

    def get_battery(self) -> int:
        return self.battery

    def streamon(self):
        pass

    def streamoff(self):
        pass

    def takeoff(self):
        self.is_flying = True

    def land(self):
        self.is_flying = False
        self.left_right_velocity = self.for_back_velocity = 0
        self.up_down_velocity = self.yaw_velocity = 0

    def rotate_clockwise(self, deg: int):
        pass

    def move_left(self, cm: int):
        pass

    def send_rc_control(self, lr, fb, ud, yaw):
        self.left_right_velocity, self.for_back_velocity = lr, fb
        self.up_down_velocity, self.yaw_velocity = ud, yaw

    def get_frame_read(self) -> FrameHolder:
        h = FrameHolder()
        if self.cap is None or not self.cap.isOpened():
            return h
        ret, frame = self.cap.read()
        if not ret:  # loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if ret:
            h.frame = frame
        return h

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ── Factory / auto-detection ─────────────────────────────────────────

def is_drone_connected(host=DRONE_IP, port=DRONE_PORT, timeout=CONNECT_TIMEOUT) -> bool:
    """Check if a physical drone is reachable via UDP handshake."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        s.sendto(b"command", (host, port))
        resp, _ = s.recvfrom(100)
        s.close()
        return resp.strip() == b"ok"
    except Exception:
        return False


def create_drone(
    demo_video: Optional[str] = None,
    force_simulated: bool = False,
) -> Tuple[BaseDrone, bool]:
    """Return ``(drone, is_simulated)`` — tries hardware first, falls back to sim."""
    if not force_simulated and is_drone_connected():
        print("[INFO] Physical drone detected on network!")
        return HardwareDrone(), False
    return SimulatedDrone(demo_video=demo_video), True
