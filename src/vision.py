"""
Computer vision pipeline — HSV filtering, edge detection, contour tracking,
deadzone evaluation, HUD overlays, and multi-view image stacking.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    _YOLO_MODEL = YOLO("yolov8n.pt")  # Tiny model for real-time tracking
except ImportError:
    _YOLO_MODEL = None

# Frame counter — only run expensive YOLO every N frames to maintain real-time playback
_YOLO_FRAME_INTERVAL = 5
_frame_counter = 0

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
    Robust motion tracker with Kalman-filter-based prediction, contour
    association, coast-through-occlusion, and adaptive jump rejection.

    Key improvements over naive per-frame contour picking:
    - **Kalman filter** (4-state: x, y, vx, vy) predicts position each frame
    - **Contour association** prefers detections near the predicted position
    - **Coast mode** maintains the track for up to ``max_coast`` frames when
      the contour is temporarily lost (occlusion, noise, brief off-screen)
    - **Adaptive jump threshold** scales with current speed so fast targets
      aren't falsely rejected
    - **Faster velocity EMA** (α=0.55) for responsive heading estimates
    """

    def __init__(self, max_history: int = 30, max_coast: int = 12):
        self.max_history = max_history
        self.max_coast = max_coast
        self.trajectory: List[Tuple[int, int]] = []
        self.prev_cx: Optional[int] = None
        self.prev_cy: Optional[int] = None
        self.smooth_vx: float = 0.0
        self.smooth_vy: float = 0.0

        # Last known bounding box for IoU association
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

        # Coast-through-occlusion state
        self._coast_frames: int = 0
        self._tracking: bool = False
        
        # OpenCV CSRT Pixel Tracker for polished, rock-solid lock-on
        self._csrt: Optional[cv2.Tracker] = None
        self._csrt_active: bool = False

        # Kalman filter — 4 states (x, y, vx, vy), 2 measurements (x, y)
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)
        self._kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], np.float32)
        # Process noise — moderate, trusts both prediction and measurement
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        # Measurement noise — fairly low since contour centroids are reliable
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self._kf_initialized = False

    # ── Public API ───────────────────────────────────────────────────

    @property
    def predicted_position(self) -> Optional[Tuple[int, int]]:
        """Return the Kalman-predicted (x, y) if the filter is primed."""
        if not self._kf_initialized:
            return None
        pred = self._kf.predict()
        return int(pred[0, 0]), int(pred[1, 0])

    @property
    def is_coasting(self) -> bool:
        return self._coast_frames > 0

    def update(self, target: TargetInfo, frame: np.ndarray, target_label: Optional[str] = None) -> TargetInfo:
        if target_label:
            target.label = target_label

        # ── Kalman predict step (always runs when filter is primed) ──
        if self._kf_initialized:
            self._kf.predict()
            
        # ── CSRT Pixel Tracking Pass ─────────────────────────────────
        if self._csrt_active and self._csrt is not None:
            success, box = self._csrt.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                target.found = True
                target.x, target.y, target.w, target.h = x, y, w, h
                target.cx = x + w // 2
                target.cy = y + h // 2
                target.area = float(w * h)
                
                # Create a simple contour for rendering
                target.contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
                if "[CSRT]" not in target.label:
                    target.label = f"[CSRT] {target.label}"
            else:
                # CSRT lost track, fall back
                self._csrt_active = False
                self._csrt = None

        # ── Initialize CSRT if target found but not tracked ─────────
        if target.found and not self._csrt_active:
            # Prevent locking onto random background objects at the edge of the screen
            # Only lock on if YOLO found it, OR if the object is reasonably close to the center
            dist_from_center = np.hypot(target.cx - FRAME_WIDTH//2, target.cy - FRAME_HEIGHT//2)
            is_yolo = "YOLO" in target.label
            
            if is_yolo or dist_from_center < 120:
                try:
                    self._csrt = cv2.TrackerCSRT_create()
                    bbox = (target.x, target.y, target.w, target.h)
                    self._csrt.init(frame, bbox)
                    self._csrt_active = True
                    target.label = f"[LOCKED] {target.label}"
                except Exception as e:
                    self._csrt_active = False

        # ── Target NOT found this frame ──────────────────────────────
        if not target.found:
            if self._tracking and self._coast_frames < self.max_coast and self._kf_initialized:
                # Coast: synthesise a target from the Kalman prediction
                self._coast_frames += 1
                pred = self._kf.statePost
                px, py = int(pred[0, 0]), int(pred[1, 0])
                # Clamp to frame bounds
                px = max(0, min(px, FRAME_WIDTH - 1))
                py = max(0, min(py, FRAME_HEIGHT - 1))

                target.found = True
                target.cx, target.cy = px, py
                if self.last_bbox:
                    target.x, target.y = px - self.last_bbox[2] // 2, py - self.last_bbox[3] // 2
                    target.w, target.h = self.last_bbox[2], self.last_bbox[3]
                target.vx = self.smooth_vx
                target.vy = self.smooth_vy
                target.speed = float(np.hypot(self.smooth_vx, self.smooth_vy))
                if target.speed > 1.2:
                    target.heading_deg = float(np.degrees(np.arctan2(self.smooth_vy, self.smooth_vx))) % 360.0

                self.trajectory.append((px, py))
                if len(self.trajectory) > self.max_history:
                    self.trajectory.pop(0)
                return target
            else:
                # Truly lost — reset everything
                self._full_reset()
                return target

        # ── Target found — validate and correct ─────────────────────
        self._coast_frames = 0

        if self.prev_cx is not None and self.prev_cy is not None:
            raw_vx = float(target.cx - self.prev_cx)
            raw_vy = float(target.cy - self.prev_cy)
            hop_dist = float(np.hypot(raw_vx, raw_vy))

            # Adaptive jump threshold: base 90px + 3× current smooth speed
            jump_limit = 90.0 + 3.0 * float(np.hypot(self.smooth_vx, self.smooth_vy))

            if hop_dist > jump_limit:
                # Possible contour switch / teleport — only soft-reset velocity
                self.smooth_vx *= 0.3
                self.smooth_vy *= 0.3
                target.vx = self.smooth_vx
                target.vy = self.smooth_vy
                target.speed = float(np.hypot(self.smooth_vx, self.smooth_vy))
            else:
                # Responsive EMA (α=0.55) — tracks direction changes quickly
                alpha = 0.55
                self.smooth_vx = alpha * raw_vx + (1.0 - alpha) * self.smooth_vx
                self.smooth_vy = alpha * raw_vy + (1.0 - alpha) * self.smooth_vy
                target.vx = self.smooth_vx
                target.vy = self.smooth_vy
                target.speed = min(60.0, float(np.hypot(self.smooth_vx, self.smooth_vy)))
                if target.speed > 1.2:
                    target.heading_deg = float(np.degrees(np.arctan2(self.smooth_vy, self.smooth_vx))) % 360.0

        # ── Kalman correct step ──────────────────────────────────────
        measurement = np.array([[np.float32(target.cx)], [np.float32(target.cy)]])
        if not self._kf_initialized:
            self._kf.statePost = np.array([
                [np.float32(target.cx)],
                [np.float32(target.cy)],
                [np.float32(0)],
                [np.float32(0)],
            ])
            self._kf_initialized = True
        else:
            self._kf.correct(measurement)

        # ── Update tracking state ────────────────────────────────────
        self._tracking = True
        self.prev_cx = target.cx
        self.prev_cy = target.cy
        self.last_bbox = (target.x, target.y, target.w, target.h)
        self.trajectory.append((target.cx, target.cy))
        if len(self.trajectory) > self.max_history:
            self.trajectory.pop(0)

        return target

    @property
    def smooth_speed(self) -> float:
        return float(np.hypot(self.smooth_vx, self.smooth_vy))

    def reset(self) -> None:
        self._full_reset()

    def _full_reset(self) -> None:
        self.trajectory.clear()
        self.prev_cx = None
        self.prev_cy = None
        self.smooth_vx = 0.0
        self.smooth_vy = 0.0
        self.last_bbox = None
        self._coast_frames = 0
        self._tracking = False
        self._kf_initialized = False
        self._csrt_active = False
        self._csrt = None
        # Re-initialise Kalman matrices (OpenCV resets state on re-init)
        self._kf.statePost = np.zeros((4, 1), np.float32)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)


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
    img: np.ndarray,
    edge_img: np.ndarray,
    min_area: float,
    max_area: float = 45000.0,
    skip_detection: bool = False,
) -> TargetInfo:
    """
    Find the best candidate target using YOLO (Deep Learning) first.
    If YOLO isn't available or finds nothing, fall back to OpenCV contour detection.
    """
    if skip_detection:
        return TargetInfo()

    fh, fw = edge_img.shape[:2]
    hw, hh = fw // 2, fh // 2

    # --- YOLO Deep Learning Pass (runs every Nth frame for speed) ---
    global _frame_counter
    _frame_counter += 1
    run_yolo_this_frame = (_frame_counter % _YOLO_FRAME_INTERVAL == 0)

    if _YOLO_MODEL is not None and run_yolo_this_frame:
        # Filter classes: 2=car, 3=motorcycle, 4=airplane, 5=bus, 6=train, 7=truck, 8=boat
        # Explicitly excludes 0 (person) so we don't track humans
        # imgsz=320 for ~4x faster inference vs default 640
        results = _YOLO_MODEL(img, verbose=False, classes=[2, 3, 4, 5, 6, 7, 8], imgsz=320)
        
        best_box = None
        best_conf = 0.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > 0.25 and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = r.names[int(box.cls[0])].upper()
                    best_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1), f"YOLO: {label}")
        
        if best_box is not None:
            x, y, w, h, label = best_box
            area = float(w * h)
            if min_area <= area <= max_area:
                cx, cy = x + w // 2, y + h // 2
                
                # We don't have a specific contour for YOLO boxes, so we create a simple rectangular contour
                # for the HUD drawing functions to use
                contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)
                
                info = TargetInfo(True, cx, cy, x, y, w, h, area, contour)
                info.label = label
                return info

    # --- Fallback OpenCV Contour Pass ---
    contours, _ = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
        # Relaxed from 62.0 to 150.0 to allow complex vehicles (like tanks)
        isoperimetric = (peri * peri) / max(area, 1.0)
        if isoperimetric > 150.0:
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
        # 2. Solidity weight: solid vehicles (tanks, APCs) score much higher than hollow noise
        # 3. Center gravity: exponential penalty for objects far from the crosshair
        dist_from_center = float(np.hypot(cx - hw, cy - hh))
        center_weight = float(np.exp(-dist_from_center / 150.0))
        
        score = float(area) * (1.0 + 3.0 * min(solidity, 0.9)) * center_weight

        # Heavy penalty for giant landscape shapes (houses, fields)
        if area > (fw * fh * 0.15):
            score *= 0.1
            
        # Heavy penalty for low-solidity noise hugging the extreme bottom of frame (grass)
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
    """Subtle deadzone boundary lines + professional centre reticle."""
    hw, hh = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    # Thin semi-transparent deadzone boundary
    for dx in (-DEAD_ZONE, DEAD_ZONE):
        cv2.line(img, (hw + dx, 0), (hw + dx, FRAME_HEIGHT), (180, 180, 0), 1, cv2.LINE_AA)
    for dy in (-DEAD_ZONE, DEAD_ZONE):
        cv2.line(img, (0, hh + dy), (FRAME_WIDTH, hh + dy), (180, 180, 0), 1, cv2.LINE_AA)
    # Centre reticle (small crosshair, not a dot)
    r = 10
    cv2.line(img, (hw - r, hh), (hw + r, hh), (0, 0, 200), 1, cv2.LINE_AA)
    cv2.line(img, (hw, hh - r), (hw, hh + r), (0, 0, 200), 1, cv2.LINE_AA)
    cv2.circle(img, (hw, hh), 3, (0, 0, 200), 1, cv2.LINE_AA)


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

    # Draw semi-transparent header
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    if not target.found:
        cv2.putText(img, "SEARCHING TARGET...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return

    # Render motion trajectory history breadcrumbs without teleport lines
    if trajectory and len(trajectory) >= 2:
        for idx in range(len(trajectory) - 1):
            p1, p2 = trajectory[idx], trajectory[idx + 1]
            if float(np.hypot(p2[0] - p1[0], p2[1] - p1[1])) < 45.0:
                cv2.line(img, p1, p2, (0, 200, 240), 2, cv2.LINE_AA)
        for idx, pt in enumerate(trajectory):
            radius = 3 if idx < len(trajectory) - 3 else 5
            cv2.circle(img, pt, radius, (0, 230, 255), -1, cv2.LINE_AA)

    # Center-to-target tracking vector (dashed or thin line)
    cv2.line(img, (hw, hh), (target.cx, target.cy), (0, 0, 255), 1, cv2.LINE_AA)

    # Reticle crosshair on target centroid
    cr_size = 12
    cv2.line(img, (target.cx - cr_size, target.cy), (target.cx + cr_size, target.cy), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, (target.cx, target.cy - cr_size), (target.cx, target.cy + cr_size), (0, 255, 255), 2, cv2.LINE_AA)

    # Tactical corner brackets bounding box
    bx, by, bw, bh = target.x, target.y, target.w, target.h
    
    # Draw faint full bounding box
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 100, 0), 1, cv2.LINE_AA)

    c_len = max(8, min(20, min(bw, bh) // 4))
    bracket_color = (0, 255, 0)
    thick = 3
    # top-left
    cv2.line(img, (bx, by), (bx + c_len, by), bracket_color, thick, cv2.LINE_AA)
    cv2.line(img, (bx, by), (bx, by + c_len), bracket_color, thick, cv2.LINE_AA)
    # top-right
    cv2.line(img, (bx + bw, by), (bx + bw - c_len, by), bracket_color, thick, cv2.LINE_AA)
    cv2.line(img, (bx + bw, by), (bx + bw, by + c_len), bracket_color, thick, cv2.LINE_AA)
    # bottom-left
    cv2.line(img, (bx, by + bh), (bx + c_len, by + bh), bracket_color, thick, cv2.LINE_AA)
    cv2.line(img, (bx, by + bh), (bx, by + bh - c_len), bracket_color, thick, cv2.LINE_AA)
    # bottom-right
    cv2.line(img, (bx + bw, by + bh), (bx + bw - c_len, by + bh), bracket_color, thick, cv2.LINE_AA)
    cv2.line(img, (bx + bw, by + bh), (bx + bw, by + bh - c_len), bracket_color, thick, cv2.LINE_AA)

    # Telemetry Panel next to target
    panel_x, panel_y = bx + bw + 15, max(20, by)
    
    # Draw panel background
    overlay_panel = img.copy()
    cv2.rectangle(overlay_panel, (panel_x - 5, panel_y - 20), (panel_x + 160, panel_y + 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay_panel, 0.6, img, 0.4, 0, img)

    # Target classification badge & telemetry
    badge = f"LOCKED: {target.label}"
    cv2.putText(img, badge, (panel_x, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"AREA: {int(target.area)} px",
                (panel_x, panel_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Speed & Heading vector if moving
    if target.speed > 1.2:
        spd_kmh = min(110, int(target.speed * 2.2))
        cv2.putText(img, f"VELOCITY: {spd_kmh} km/h",
                    (panel_x, panel_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"HEADING: {int(target.heading_deg):03d}\u00b0",
                    (panel_x, panel_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # Bounded velocity heading arrow (max 28px length)
        arrow_len = min(40.0, max(15.0, target.speed * 2.5))
        angle = np.radians(target.heading_deg)
        tip_x = int(target.cx + arrow_len * np.cos(angle))
        tip_y = int(target.cy + arrow_len * np.sin(angle))
        cv2.arrowedLine(img, (target.cx, target.cy), (tip_x, tip_y), (0, 165, 255), 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # Guidance direction cues (Top left)
    if direction in _DIR_LABEL:
        cv2.putText(img, _DIR_LABEL[direction], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        (x1, y1), (x2, y2) = _DIR_RECT[direction](hw, hh, DEAD_ZONE)
        x1, x2 = max(0, min(x1, FRAME_WIDTH)), max(0, min(x2, FRAME_WIDTH))
        y1, y2 = max(0, min(y1, FRAME_HEIGHT)), max(0, min(y2, FRAME_HEIGHT))
        if x2 > x1 and y2 > y1:
            sub = img[y1:y2, x1:x2]
            red_tint = np.zeros_like(sub)
            red_tint[:] = (0, 0, 255)
            # Semi-transparent red tint (28% red, 72% original feed)
            cv2.addWeighted(red_tint, 0.25, sub, 0.75, 0, sub)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, f"TRACKING STABLE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def draw_hud(img: np.ndarray, label: str, battery: int) -> None:
    """Telemetry bar at bottom of frame."""
    h = img.shape[0]
    w = img.shape[1]
    
    # Bottom HUD panel
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
    
    cv2.putText(img, f"SYS: {label} | AI-VISION: ACTIVE",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


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

    # Only run heavy detection if we don't have a solid CSRT lock
    skip_heavy_detection = tracker is not None and tracker._csrt_active

    if not skip_heavy_detection:
        _, result = apply_hsv_mask(img, hsv)
        edges = detect_edges(result, edge)
    else:
        edges = np.zeros_like(img[:, :, 0])

    target = detect_target(img, edges, min_area, skip_detection=skip_heavy_detection)

    if tracker is not None:
        target = tracker.update(target, img)

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

    # ── Build 4 Quadrant Views ──────────────────────────────────────

    # Q1: Raw Camera Feed (clean original)
    q1 = img.copy()

    # Q2: Enhanced Detection View (thermal-style false-color heat map)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q2 = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    # Draw YOLO/detection bbox on thermal view if target found
    if target.found:
        cv2.rectangle(q2, (target.x, target.y),
                      (target.x + target.w, target.y + target.h), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(q2, (target.cx, target.cy), 4, (0, 255, 255), -1, cv2.LINE_AA)

    # Q3: Edge Detection Map (colorized for visibility)
    if len(edges.shape) == 2:
        edge_color = cv2.applyColorMap(edges, cv2.COLORMAP_BONE)
    else:
        edge_color = edges.copy()

    # Q4: Tracking HUD (full overlay)
    q4 = img.copy()
    draw_target_overlay(
        q4,
        target,
        direction if target.found else prev_direction,
        trajectory=tracker.trajectory if tracker is not None else None,
    )
    draw_deadzone_grid(q4)

    # ── Add Quadrant Labels ─────────────────────────────────────────
    label_color = (200, 200, 200)
    label_bg = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for panel, label_text in [(q1, "CAMERA FEED"), (q2, "THERMAL DETECTION"),
                               (edge_color, "EDGE ANALYSIS"), (q4, "TRACKING HUD")]:
        # Semi-transparent label background
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (180, 25), label_bg, -1)
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
        cv2.putText(panel, label_text, (8, 17), font, 0.45, label_color, 1, cv2.LINE_AA)

    # ── Stack into 2×2 Grid ─────────────────────────────────────────
    stacked = stack_images(0.8, ([q1, q2], [edge_color, q4]))
    return stacked, target, direction
