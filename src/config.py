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

# Canny edge-detection & contour thresholds (tuned for anti-clutter tracking)
CANNY_THRESH1 = 55
CANNY_THRESH2 = 140
MIN_CONTOUR_AREA = 250
MAX_CONTOUR_AREA = 38000

# Gaussian blur / morphology kernel sizes
BLUR_KERNEL: Tuple[int, int] = (7, 7)
BLUR_SIGMA: float = 1.5
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
    default_video: Optional[str] = None
    target_label: str = "AUTO"


# Universal Preset: Auto-detects Moving Target vs Stationary Target
UNIVERSAL_PRESET = VisionPreset(
    name="universal",
    description="Universal Aerial Target Tracker (Moving Target & Static Target Lock)",
    hsv_bounds=(0, 180, 0, 255, 0, 255),
    edge_thresholds=(55, 140),
    min_area=250,
    max_area=38000,
    default_video="aircraft_tracking.mp4",
    target_label="TARGET",
)


PRESETS: Dict[str, VisionPreset] = {
    # ── Primary Unified Trackers ──
    "universal": UNIVERSAL_PRESET,
    "default": VisionPreset(
        name="default",
        description="Yellow-Green target (demo ball / bright object)",
        hsv_bounds=(29, 65, 85, 255, 60, 255),
        edge_thresholds=(166, 171),
        min_area=1000,
        default_video="drone_target_tracking.mp4",
        target_label="DEMO BALL",
    ),
    # ── Target-specific Profiles & Compatibility Aliases ──
    "tank": VisionPreset(
        name="tank",
        description="Armored tank & ground target tracking",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(55, 140),
        min_area=250,
        max_area=38000,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "military_vehicle": VisionPreset(
        name="military_vehicle",
        description="Tactical military vehicle & convoy tracking",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(55, 140),
        min_area=250,
        max_area=38000,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "helicopter": VisionPreset(
        name="helicopter",
        description="Rotary-wing aircraft / helicopter tracking",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(55, 140),
        min_area=250,
        max_area=38000,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "aircraft": VisionPreset(
        name="aircraft",
        description="Aerial fighter jet & aircraft tracking",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(40, 120),
        min_area=80,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "moving_object": VisionPreset(
        name="moving_object",
        description="Generic moving target tracker",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(55, 140),
        min_area=250,
        max_area=38000,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "moving_tank": VisionPreset(
        name="moving_tank",
        description="Armored tank tracking",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(55, 140),
        min_area=250,
        max_area=38000,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "tactical_vehicle": VisionPreset(
        name="tactical_vehicle",
        description="Tactical military vehicle tracking",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(55, 140),
        min_area=250,
        max_area=38000,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "military_tracking": VisionPreset(
        name="military_tracking",
        description="MQ-9 onboard camera HUD targeting reticle",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(40, 120),
        min_area=80,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "military_recon": VisionPreset(
        name="military_recon",
        description="MQ-9 Reaper military combat drone against terrain",
        hsv_bounds=(0, 180, 0, 95, 30, 160),
        edge_thresholds=(40, 120),
        min_area=600,
        default_video="reaper_recon.mp4",
        target_label="TARGET",
    ),
    "sky_target": VisionPreset(
        name="sky_target",
        description="Quadcopter drone / aircraft in open sky",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(40, 120),
        min_area=100,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
    "sky_uav": VisionPreset(
        name="sky_uav",
        description="Fixed-wing UAV tracking on horizon",
        hsv_bounds=(0, 180, 0, 255, 0, 255),
        edge_thresholds=(40, 120),
        min_area=100,
        default_video="aircraft_tracking.mp4",
        target_label="TARGET",
    ),
}


def get_preset_for_video(video_path: Optional[str]) -> VisionPreset:
    """Auto-detect the optimal tracking preset for a given video path or default."""
    if not video_path:
        return PRESETS["default"]
    lower = os.path.basename(video_path).lower()
    if "tank" in lower:
        return PRESETS["tank"]
    if "vehicle" in lower or "convoy" in lower or "apc" in lower or "truck" in lower:
        return PRESETS["military_vehicle"]
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
        for sub in (
            os.path.join("asset", "videos"),
            os.path.join("asset", "gifs"),
            os.path.join("asset", "img"),
            os.path.join("asset", "demo_gifs"),
            "asset",
            "assets",
            "",
        ):
            candidate = (
                os.path.join(root, sub, filename)
                if sub
                else os.path.join(root, filename)
            )
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
    return None
