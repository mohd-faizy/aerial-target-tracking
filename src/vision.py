"""
Computer vision pipeline — HSV filtering, edge detection, contour tracking,
deadzone evaluation, HUD overlays, and multi-view image stacking.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    FRAME_WIDTH, FRAME_HEIGHT, DEAD_ZONE,
    HSV_HUE_MIN, HSV_HUE_MAX, HSV_SAT_MIN, HSV_SAT_MAX,
    HSV_VAL_MIN, HSV_VAL_MAX,
    CANNY_THRESH1, CANNY_THRESH2, MIN_CONTOUR_AREA,
    BLUR_KERNEL, BLUR_SIGMA, DILATE_KERNEL,
    YAW_SPEED, VERTICAL_SPEED,
)


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class HSVBounds:
    """Mutable HSV threshold range (updated live by trackbars)."""
    h_min: int = HSV_HUE_MIN
    h_max: int = HSV_HUE_MAX
    s_min: int = HSV_SAT_MIN
    s_max: int = HSV_SAT_MAX
    v_min: int = HSV_VAL_MIN
    v_max: int = HSV_VAL_MAX


@dataclass
class EdgeThresholds:
    """Mutable Canny threshold pair."""
    threshold1: int = CANNY_THRESH1
    threshold2: int = CANNY_THRESH2


@dataclass
class TargetInfo:
    """Result of contour analysis — centroid, bounding box, area."""
    found: bool = False
    cx: int = 0
    cy: int = 0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    area: float = 0.0
    contour: Optional[np.ndarray] = None


class Direction(IntEnum):
    """Flight direction codes."""
    NONE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


# ── Core image-processing functions ─────────────────────────────────

def apply_hsv_mask(img: np.ndarray, bounds: HSVBounds) -> Tuple[np.ndarray, np.ndarray]:
    """Apply HSV thresholding. Returns *(mask_bgr, masked_result)*."""
    # Full spectrum bypass: allows monochrome / edge contrast tracking
    if (
        bounds.h_min == 0
        and bounds.h_max >= 179
        and bounds.s_min == 0
        and bounds.s_max >= 255
        and bounds.v_min == 0
        and bounds.v_max >= 255
    ):
        return np.full_like(img, 255), img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([bounds.h_min, bounds.s_min, bounds.v_min], np.uint8)
    upper = np.array([bounds.h_max, bounds.s_max, bounds.v_max], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result


def detect_edges(img: np.ndarray, thresh: EdgeThresholds) -> np.ndarray:
    """Blur → grayscale → Canny → dilate.  Returns dilated binary edge image."""
    blur = cv2.GaussianBlur(img, BLUR_KERNEL, BLUR_SIGMA)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, thresh.threshold1, thresh.threshold2)
    kernel = np.ones(DILATE_KERNEL, np.uint8)
    return cv2.dilate(canny, kernel, iterations=1)


def detect_target(
    edge_img: np.ndarray,
    min_area: float,
    max_area: float = 45000.0,
) -> TargetInfo:
    """Find the best candidate contour exceeding *min_area*, rejecting border artifacts."""
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fh, fw = edge_img.shape[:2]

    # Pass 1: Smart filtering (excludes screen borders, letterboxes, and wide horizon/wing bands)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        # Ignore contours touching the outermost frame margin (letterbox / border lines)
        if x <= 10 or y <= 10 or (x + w) >= (fw - 10) or (y + h) >= (fh - 10):
            continue
        # Ignore full-width/height span contours
        if w >= (fw - 60) or h >= (fh - 60):
            continue
        # Ignore horizontal bands (cloud lines, wing edges: w > 350 and thin h < 60)
        if w > 350 and h < 60:
            continue

        return TargetInfo(True, x + w // 2, y + h // 2, x, y, w, h, area, cnt)

    # Pass 2: Fallback for synthetic patterns, indoor demo objects, and clean frames
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w >= (fw - 30) and h >= (fh - 30):
            continue
        return TargetInfo(True, x + w // 2, y + h // 2, x, y, w, h, area, cnt)

    return TargetInfo()


# ── Deadzone / direction logic ───────────────────────────────────────

def compute_direction(target: TargetInfo) -> Direction:
    """Where is the target relative to the central deadzone?"""
    if not target.found:
        return Direction.NONE
    hw, hh = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    if target.cx < hw - DEAD_ZONE:
        return Direction.LEFT
    if target.cx > hw + DEAD_ZONE:
        return Direction.RIGHT
    if target.cy < hh - DEAD_ZONE:
        return Direction.UP
    if target.cy > hh + DEAD_ZONE:
        return Direction.DOWN
    return Direction.NONE


def direction_to_rc(d: Direction) -> Tuple[int, int, int, int]:
    """Map a Direction to *(lr, fb, ud, yaw)* velocities."""
    if d == Direction.LEFT:
        return 0, 0, 0, -YAW_SPEED
    if d == Direction.RIGHT:
        return 0, 0, 0, YAW_SPEED
    if d == Direction.UP:
        return 0, 0, VERTICAL_SPEED, 0
    if d == Direction.DOWN:
        return 0, 0, -VERTICAL_SPEED, 0
    return 0, 0, 0, 0


# ── Drawing / HUD ───────────────────────────────────────────────────

def draw_deadzone_grid(img: np.ndarray) -> None:
    """Cyan deadzone boundary lines + red centre dot."""
    hw, hh = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    for dx in (-DEAD_ZONE, DEAD_ZONE):
        cv2.line(img, (hw + dx, 0), (hw + dx, FRAME_HEIGHT), (255, 255, 0), 2)
    for dy in (-DEAD_ZONE, DEAD_ZONE):
        cv2.line(img, (0, hh + dy), (FRAME_WIDTH, hh + dy), (255, 255, 0), 2)
    cv2.circle(img, (hw, hh), 6, (0, 0, 255), -1)


_DIR_LABEL = {
    Direction.LEFT:  " GO LEFT ",
    Direction.RIGHT: " GO RIGHT ",
    Direction.UP:    " GO UP ",
    Direction.DOWN:  " GO DOWN ",
}

_DIR_RECT = {
    Direction.LEFT:  lambda hw, hh, dz: ((0, hh - dz), (hw - dz, hh + dz)),
    Direction.RIGHT: lambda hw, hh, dz: ((hw + dz, hh - dz), (FRAME_WIDTH, hh + dz)),
    Direction.UP:    lambda hw, hh, dz: ((hw - dz, 0), (hw + dz, hh - dz)),
    Direction.DOWN:  lambda hw, hh, dz: ((hw - dz, hh + dz), (hw + dz, FRAME_HEIGHT)),
}


def draw_target_overlay(img: np.ndarray, target: TargetInfo, direction: Direction) -> None:
    """Draw contour, bounding box, vector line, and direction cue on *img*."""
    hw, hh = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

    if not target.found:
        cv2.putText(img, " SEARCHING TARGET... ", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        return

    if target.contour is not None:
        cv2.drawContours(img, [target.contour], -1, (255, 0, 255), 3)
    cv2.line(img, (hw, hh), (target.cx, target.cy), (0, 0, 255), 3)
    cv2.rectangle(img, (target.x, target.y),
                  (target.x + target.w, target.y + target.h), (0, 255, 0), 3)
    cv2.putText(img, f"Area: {int(target.area)}",
                (target.x + target.w + 10, target.y + 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

    if direction in _DIR_LABEL:
        cv2.putText(img, _DIR_LABEL[direction], (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        p1, p2 = _DIR_RECT[direction](hw, hh, DEAD_ZONE)
        cv2.rectangle(img, p1, p2, (0, 0, 255), cv2.FILLED)
    else:
        cv2.putText(img, " LOCKED ON TARGET ", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


def draw_hud(img: np.ndarray, label: str, battery: int) -> None:
    """Telemetry bar at bottom of frame."""
    cv2.putText(img, f"{label} | BAT: {battery}%",
                (15, img.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def stack_images(scale: float, img_array: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Tile a 2-D grid of images into a single canvas."""
    w = int(img_array[0][0].shape[1] * scale)
    h = int(img_array[0][0].shape[0] * scale)
    rows: List[np.ndarray] = []
    for row in img_array:
        cols: List[np.ndarray] = []
        for im in row:
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            cols.append(cv2.resize(im, (w, h)))
        rows.append(np.hstack(cols))
    return np.vstack(rows)


# ── High-level pipeline ─────────────────────────────────────────────

def process_frame(
    frame: np.ndarray,
    hsv: HSVBounds,
    edge: EdgeThresholds,
    min_area: float,
    prev_direction: Direction = Direction.NONE,
) -> Tuple[np.ndarray, TargetInfo, Direction]:
    """
    Run full vision pipeline on one frame.

    Returns *(stacked_display, target_info, direction)*.
    """
    img = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    contour_img = img.copy()

    _, result = apply_hsv_mask(img, hsv)
    edges = detect_edges(result, edge)
    target = detect_target(edges, min_area)
    direction = compute_direction(target)

    draw_target_overlay(contour_img, target, direction if target.found else prev_direction)
    draw_deadzone_grid(contour_img)

    stacked = stack_images(0.8, ([img, result], [edges, contour_img]))
    return stacked, target, direction
