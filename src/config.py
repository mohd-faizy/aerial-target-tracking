"""
Configuration constants, vision presets, and utility helpers.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ── Frame & Vision Defaults ──────────────────────────────────────────

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEAD_ZONE = 100

# Default HSV colour-space bounds (yellow-green target)
HSV_HUE_MIN, HSV_HUE_MAX = 29, 65
HSV_SAT_MIN, HSV_SAT_MAX = 85, 255
HSV_VAL_MIN, HSV_VAL_MAX = 60, 255

# Canny edge-detection & contour thresholds
CANNY_THRESH1 = 166
CANNY_THRESH2 = 171
MIN_CONTOUR_AREA = 1000
MAX_CONTOUR_AREA = 30000

# Gaussian blur / morphology kernel sizes
BLUR_KERNEL: Tuple[int, int] = (5, 5)
BLUR_SIGMA: float = 1.0
DILATE_KERNEL: Tuple[int, int] = (3, 3)

# ── Flight-control Speeds ────────────────────────────────────────────

YAW_SPEED = 60
VERTICAL_SPEED = 60

# ── Drone Network ────────────────────────────────────────────────────

DRONE_IP = "192.168.10.1"
DRONE_PORT = 8889
LOCAL_PORT = 9000
VIDEO_STREAM_URL = "udp://@0.0.0.0:11111"
CONNECT_TIMEOUT = 0.6


# ── Vision Preset Profiles ───────────────────────────────────────────


@dataclass(frozen=True)
class VisionPreset:
    """Tuned computer vision and contour parameters for specific target types."""

    name: str
    description: str
    hsv_bounds: Tuple[
        int, int, int, int, int, int
    ]  # (h_min, h_max, s_min, s_max, v_min, v_max)
    edge_thresholds: Tuple[int, int]  # (thresh1, thresh2)
    min_area: int
    max_area: int = 40000


PRESETS: Dict[str, VisionPreset] = {
    "default": VisionPreset(
        name="default",
        description="Yellow-Green target (demo ball / bright object)",
        hsv_bounds=(29, 65, 85, 255, 60, 255),
        edge_thresholds=(166, 171),
        min_area=1000,
    ),
    "sky_target": VisionPreset(
        name="sky_target",
        description="Quadcopter drone in open sky",
        hsv_bounds=(20, 80, 50, 255, 40, 255),
        edge_thresholds=(100, 200),
        min_area=800,
    ),
    "sky_uav": VisionPreset(
        name="sky_uav",
        description="Fixed-wing UAV tracking on horizon",
        hsv_bounds=(20, 80, 50, 255, 40, 255),
        edge_thresholds=(100, 200),
        min_area=800,
    ),
    "military_tracking": VisionPreset(
        name="military_tracking",
        description="MQ-9 onboard camera HUD targeting aircraft / reticle",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(40, 120),
        min_area=80,
    ),
    "military_recon": VisionPreset(
        name="military_recon",
        description="MQ-9 Reaper military combat drone against terrain",
        hsv_bounds=(0, 180, 0, 95, 30, 160),
        edge_thresholds=(40, 120),
        min_area=600,
    ),
}


def get_preset_for_video(video_path: Optional[str]) -> VisionPreset:
    """Auto-detect the optimal tracking preset for a given video path or default."""
    if not video_path:
        return PRESETS["default"]
    lower = os.path.basename(video_path).lower()
    if "aircraft" in lower:
        return PRESETS["military_tracking"]
    if "reaper" in lower or "recon" in lower:
        return PRESETS["military_recon"]
    if "sky_target" in lower:
        return PRESETS["sky_target"]
    if "sky_uav" in lower or "uav" in lower:
        return PRESETS["sky_uav"]
    return PRESETS["default"]


# ── Asset Finder ─────────────────────────────────────────────────────


def find_asset(filename: str) -> Optional[str]:
    """Search common project paths for *filename* and return its absolute path."""
    if not filename:
        return None
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename

    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    cwd = os.getcwd()

    for root in (project_root, cwd, here):
        for sub in ("asset", "assets", os.path.join("asset", "demo_gifs"), ""):
            candidate = (
                os.path.join(root, sub, filename)
                if sub
                else os.path.join(root, filename)
            )
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return None
