"""
OpenCV trackbar manager — crash-proof HSV / edge-threshold calibration UI.
"""
import cv2

from typing import Optional

from .config import (
    HSV_HUE_MIN, HSV_HUE_MAX, HSV_SAT_MIN, HSV_SAT_MAX,
    HSV_VAL_MIN, HSV_VAL_MAX,
    CANNY_THRESH1, CANNY_THRESH2, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA,
    VisionPreset,
)
from .vision import HSVBounds, EdgeThresholds


def _noop(_val: int) -> None:
    pass


class Calibration:
    """Creates trackbar windows and safely reads their values each frame."""

    def __init__(self, preset: Optional[VisionPreset] = None):
        if preset is not None:
            hb = preset.hsv_bounds
            et = preset.edge_thresholds
            self.hsv = HSVBounds(h_min=hb[0], h_max=hb[1], s_min=hb[2], s_max=hb[3], v_min=hb[4], v_max=hb[5])
            self.edge = EdgeThresholds(threshold1=et[0], threshold2=et[1])
            self.min_area = preset.min_area
        else:
            self.hsv = HSVBounds()
            self.edge = EdgeThresholds()
            self.min_area = MIN_CONTOUR_AREA
        self._ready = False

    def init_windows(self) -> None:
        """Create the HSV and Parameters trackbar windows."""
        try:
            cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("HSV", 640, 240)
            cv2.createTrackbar("HUE Min",   "HSV", self.hsv.h_min, 179, _noop)
            cv2.createTrackbar("HUE Max",   "HSV", self.hsv.h_max, 179, _noop)
            cv2.createTrackbar("SAT Min",   "HSV", self.hsv.s_min, 255, _noop)
            cv2.createTrackbar("SAT Max",   "HSV", self.hsv.s_max, 255, _noop)
            cv2.createTrackbar("VALUE Min", "HSV", self.hsv.v_min, 255, _noop)
            cv2.createTrackbar("VALUE Max", "HSV", self.hsv.v_max, 255, _noop)

            cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Parameters", 640, 240)
            cv2.createTrackbar("Threshold1", "Parameters", self.edge.threshold1, 255, _noop)
            cv2.createTrackbar("Threshold2", "Parameters", self.edge.threshold2, 255, _noop)
            cv2.createTrackbar("Area",       "Parameters", self.min_area, MAX_CONTOUR_AREA, _noop)
            self._ready = True
        except Exception as exc:
            print(f"[WARN] Trackbar init failed: {exc}")

    def _get(self, name: str, win: str, fallback: int) -> int:
        if not self._ready:
            return fallback
        try:
            v = cv2.getTrackbarPos(name, win)
            return v if v != -1 else fallback
        except Exception:
            return fallback

    def update(self) -> None:
        """Refresh cached values from trackbar positions."""
        self.hsv.h_min = self._get("HUE Min",   "HSV", self.hsv.h_min)
        self.hsv.h_max = self._get("HUE Max",   "HSV", self.hsv.h_max)
        self.hsv.s_min = self._get("SAT Min",   "HSV", self.hsv.s_min)
        self.hsv.s_max = self._get("SAT Max",   "HSV", self.hsv.s_max)
        self.hsv.v_min = self._get("VALUE Min", "HSV", self.hsv.v_min)
        self.hsv.v_max = self._get("VALUE Max", "HSV", self.hsv.v_max)
        self.edge.threshold1 = self._get("Threshold1", "Parameters", self.edge.threshold1)
        self.edge.threshold2 = self._get("Threshold2", "Parameters", self.edge.threshold2)
        self.min_area        = self._get("Area",       "Parameters", self.min_area)

    @staticmethod
    def window_open(name: str) -> bool:
        """True if the named OpenCV window is still visible."""
        try:
            return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
        except Exception:
            return False
