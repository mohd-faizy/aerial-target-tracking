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
    """Result of contour analysis — centroid, bounding box, area, and motion telemetry."""
    found: bool = False
    cx: int = 0
    cy: int = 0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    area: float = 0.0
    contour: Optional[np.ndarray] = None
    label: str = "TARGET"
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    heading_deg: float = 0.0


class MovingTargetTracker:
    """
    Maintains motion history and estimates target velocity, heading,
    and trajectory breadcrumbs across video frames with jump rejection
    and exponential velocity smoothing.
    """
    def __init__(self, max_history: int = 15):
        self.max_history = max_history
        self.trajectory: List[Tuple[int, int]] = []
        self.prev_cx: Optional[int] = None
        self.prev_cy: Optional[int] = None
        self.smooth_vx: float = 0.0
        self.smooth_vy: float = 0.0

    def update(self, target: TargetInfo, target_label: Optional[str] = None) -> TargetInfo:
        if target_label:
            target.label = target_label

        if not target.found:
            self.trajectory.clear()
            self.prev_cx = None
            self.prev_cy = None
            self.smooth_vx = 0.0
            self.smooth_vy = 0.0
            return target

        if self.prev_cx is not None and self.prev_cy is not None:
            raw_vx = float(target.cx - self.prev_cx)
            raw_vy = float(target.cy - self.prev_cy)
            hop_dist = float(np.hypot(raw_vx, raw_vy))

            # Reject teleportation jumps / contour re-acquisitions (> 65 px in 1 frame)
            if hop_dist > 65.0:
                self.trajectory.clear()
                self.smooth_vx = 0.0
                self.smooth_vy = 0.0
                target.vx = 0.0
                target.vy = 0.0
                target.speed = 0.0
            else:
                # Exponential moving average smoothing for velocity
                self.smooth_vx = 0.35 * raw_vx + 0.65 * self.smooth_vx
                self.smooth_vy = 0.35 * raw_vy + 0.65 * self.smooth_vy
                target.vx = self.smooth_vx
                target.vy = self.smooth_vy
                target.speed = min(45.0, float(np.hypot(self.smooth_vx, self.smooth_vy)))
                if target.speed > 1.2:
                    target.heading_deg = float(np.degrees(np.arctan2(self.smooth_vy, self.smooth_vx))) % 360.0

        self.prev_cx = target.cx
        self.prev_cy = target.cy
        self.trajectory.append((target.cx, target.cy))
        if len(self.trajectory) > self.max_history:
            self.trajectory.pop(0)

        return target

    @property
    def smooth_speed(self) -> float:
        return float(np.hypot(self.smooth_vx, self.smooth_vy))

    def reset(self) -> None:
        self.trajectory.clear()
        self.prev_cx = None
        self.prev_cy = None
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0


def classify_target(
    w: int = 0,
    h: int = 0,
    area: float = 0.0,
    cx: int = 0,
    cy: int = 0,
    speed: float = 0.0,
    requested_mode: str = "AUTO",
) -> str:
    """
    Identifies target tracking state. Returns 'MOVING TARGET' if in motion (speed > 1.2),
    or 'TARGET' when stationary / hovering. Honors explicit user override if provided.
    """
    if isinstance(speed, str):
        requested_mode = speed
        speed = 0.0

    if requested_mode and requested_mode not in ("AUTO", "TARGET", "MOVING TARGET", ""):
        return requested_mode

    return "MOVING TARGET" if speed > 1.2 else "TARGET"


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
    """
    Gaussian blur -> Grayscale -> Canny edge detection -> Morphological closing.
    Suppresses fine high-frequency texture (grass, dirt, noise) while preserving
    structural silhouettes of aircraft, helicopters, tanks, and vehicles.
    """
    blur = cv2.GaussianBlur(img, BLUR_KERNEL, BLUR_SIGMA)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, thresh.threshold1, thresh.threshold2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, close_kernel)
    kernel = np.ones(DILATE_KERNEL, np.uint8)
    return cv2.dilate(closed, kernel, iterations=1)


def detect_target(
    edge_img: np.ndarray,
    min_area: float,
    max_area: float = 45000.0,
) -> TargetInfo:
    """Find the best candidate contour exceeding *min_area*, rejecting border, screen, and grass clutter."""
    contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    fh, fw = edge_img.shape[:2]
    hw, hh = fw // 2, fh // 2

    candidates = []

    # Pass 1: Smart filtering and candidate scoring
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        # Reject high-frequency jagged grass, foliage, and textured dirt loops
        isoperimetric = (peri * peri) / max(area, 1.0)
        if isoperimetric > 62.0:
            continue

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
        # Ignore giant rectangular frames (phone bezels, monitor screens, posters filling > 22% of frame)
        if (w * h) > (fw * fh * 0.22):
            continue
        # Ignore rectangular device displays / sheets of paper (large 4-vertex shapes with near-perfect solidity)
        solidity = area / float(w * h) if (w * h) > 0 else 0.0
        if area > 10000 and len(approx) == 4 and solidity > 0.85:
            continue

        cx, cy = x + w // 2, y + h // 2

        # Candidate quality score:
        # 1. Base score is contour area
        # 2. Solidity weight: solid vehicles (tanks, APCs) score much higher than hollow/ragged noise
        # 3. Distance from deadzone center: actively tracked targets are near center or deadzone
        dist_from_center = float(np.hypot(cx - hw, cy - hh))
        score = float(area) * (1.0 + 3.0 * min(solidity, 0.9)) / (1.0 + 0.0015 * dist_from_center)

        # Heavy penalty for low-solidity noise hugging the extreme bottom of frame (foreground weeds/grass)
        if (y + h) >= (fh - 30):
            score *= 0.25

        candidates.append((score, x, y, w, h, area, cnt, cx, cy))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, x, y, w, h, area, cnt, cx, cy = candidates[0]
        return TargetInfo(True, cx, cy, x, y, w, h, area, cnt)

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


def draw_target_overlay(
    img: np.ndarray,
    target: TargetInfo,
    direction: Direction,
    trajectory: Optional[Sequence[Tuple[int, int]]] = None,
) -> None:
    """Draw tactical HUD: contour, corner brackets, crosshair, motion vector, trajectory, and status cues."""
    hw, hh = FRAME_WIDTH // 2, FRAME_HEIGHT // 2

    if not target.found:
        cv2.putText(img, " SEARCHING TARGET... ", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        return

    # Render motion trajectory history breadcrumbs without teleport lines
    if trajectory and len(trajectory) >= 2:
        for idx in range(len(trajectory) - 1):
            p1, p2 = trajectory[idx], trajectory[idx + 1]
            if float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) < 45.0:
                cv2.line(img, p1, p2, (0, 200, 240), 1, cv2.LINE_AA)
        for idx, pt in enumerate(trajectory):
            radius = 2 if idx < len(trajectory) - 3 else 3
            cv2.circle(img, pt, radius, (0, 230, 255), -1)

    # Draw contour outline
    if target.contour is not None:
        cv2.drawContours(img, [target.contour], -1, (255, 0, 255), 2)

    # Center-to-target tracking vector
    cv2.line(img, (hw, hh), (target.cx, target.cy), (0, 0, 255), 2)

    # Reticle crosshair on target centroid
    cr_size = 8
    cv2.line(img, (target.cx - cr_size, target.cy), (target.cx + cr_size, target.cy), (0, 255, 255), 2)
    cv2.line(img, (target.cx, target.cy - cr_size), (target.cx, target.cy + cr_size), (0, 255, 255), 2)

    # Tactical corner brackets bounding box
    bx, by, bw, bh = target.x, target.y, target.w, target.h
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 180, 0), 1)

    c_len = max(6, min(14, min(bw, bh) // 4))
    bracket_color = (0, 255, 0)
    # top-left
    cv2.line(img, (bx, by), (bx + c_len, by), bracket_color, 2)
    cv2.line(img, (bx, by), (bx, by + c_len), bracket_color, 2)
    # top-right
    cv2.line(img, (bx + bw, by), (bx + bw - c_len, by), bracket_color, 2)
    cv2.line(img, (bx + bw, by), (bx + bw, by + c_len), bracket_color, 2)
    # bottom-left
    cv2.line(img, (bx, by + bh), (bx + c_len, by + bh), bracket_color, 2)
    cv2.line(img, (bx, by + bh), (bx, by + bh - c_len), bracket_color, 2)
    # bottom-right
    cv2.line(img, (bx + bw, by + bh), (bx + bw - c_len, by + bh), bracket_color, 2)
    cv2.line(img, (bx + bw, by + bh), (bx + bw, by + bh - c_len), bracket_color, 2)

    # Target classification badge & telemetry
    badge = f"[{target.label}]"
    cv2.putText(img, badge, (bx, max(20, by - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, f"Area: {int(target.area)}",
                (bx + bw + 8, by + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Speed & Heading vector if moving
    if target.speed > 1.2:
        spd_kmh = min(110, int(target.speed * 2.2))
        cv2.putText(img, f"SPD: {spd_kmh} km/h",
                    (bx + bw + 8, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(img, f"HDG: {int(target.heading_deg):03d}\u00b0",
                    (bx + bw + 8, by + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Bounded velocity heading arrow (max 28px length)
        arrow_len = min(28.0, max(10.0, target.speed * 2.0))
        angle = np.radians(target.heading_deg)
        tip_x = int(target.cx + arrow_len * np.cos(angle))
        tip_y = int(target.cy + arrow_len * np.sin(angle))
        cv2.arrowedLine(img, (target.cx, target.cy), (tip_x, tip_y), (0, 255, 255), 2, tipLength=0.35)

    # Guidance direction cues
    if direction in _DIR_LABEL:
        cv2.putText(img, _DIR_LABEL[direction], (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
        (x1, y1), (x2, y2) = _DIR_RECT[direction](hw, hh, DEAD_ZONE)
        x1, x2 = max(0, min(x1, FRAME_WIDTH)), max(0, min(x2, FRAME_WIDTH))
        y1, y2 = max(0, min(y1, FRAME_HEIGHT)), max(0, min(y2, FRAME_HEIGHT))
        if x2 > x1 and y2 > y1:
            sub = img[y1:y2, x1:x2]
            red_tint = np.zeros_like(sub)
            red_tint[:] = (0, 0, 255)
            # Semi-transparent red tint (28% red, 72% original feed)
            cv2.addWeighted(red_tint, 0.28, sub, 0.72, 0, sub)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        cv2.putText(img, f" LOCKED: {target.label} ", (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)


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
    tracker: Optional[MovingTargetTracker] = None,
    target_label: str = "AUTO",
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

    if tracker is not None:
        target = tracker.update(target)

    if target.found:
        target.label = classify_target(
            target.w,
            target.h,
            target.area,
            target.cx,
            target.cy,
            speed=target.speed,
            requested_mode=target_label,
        )
    else:
        target.label = target_label

    direction = compute_direction(target)

    draw_target_overlay(
        contour_img,
        target,
        direction if target.found else prev_direction,
        trajectory=tracker.trajectory if tracker is not None else None,
    )
    draw_deadzone_grid(contour_img)

    stacked = stack_images(0.8, ([img, result], [edges, contour_img]))
    return stacked, target, direction
