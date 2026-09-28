"""
Autonomous Aerial Target Tracking — unified launcher.

Usage:
    python main.py                     # default: autonomous object tracking
    python main.py --mode color        # standalone vision calibration (no flight)
    python main.py --mode flight-test  # basic flight manoeuvre test
    python main.py --video asset/videos/reaper_recon.mp4 # custom video
    python main.py --webcam            # force webcam input
    python main.py --simulate          # force simulator even if drone on network
    python main.py color               # legacy shorthand
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import cv2

from src.config import find_asset, PRESETS, get_preset_for_video
from src.drone import create_drone
from src.vision import (
    process_frame, draw_hud, direction_to_rc, Direction, MovingTargetTracker,
)
from src.calibration import Calibration


# ── Application modes ────────────────────────────────────────────────

def run_tracking(video_path, webcam=False, force_sim=False, preset=None):
    """Full autonomous object tracking with flight control."""
    drone, simulated = create_drone(demo_video=video_path, force_simulated=force_sim)
    active_preset = preset or get_preset_for_video(video_path if not webcam else None)

    if simulated:
        print("\n" + "=" * 60)
        print(" [INFO] No physical drone connected on network.")
        print(" [INFO] Running in Drone Flight Simulation & Demo Video mode!")
        print(f" [INFO] Demo Video: {video_path or 'Webcam'}")
        print(f" [INFO] Vision Preset: {active_preset.name} ({active_preset.description})")
        print(" [INFO] Controls: [V] Toggle Demo Video / Webcam | [Q] Quit")
        print("=" * 60 + "\n")

    drone.connect()
    print(f"Drone Battery: {drone.get_battery()}%")
    drone.streamoff()
    drone.streamon()

    cal = Calibration(preset=active_preset)
    cal.init_windows()

    win = "Autonomous Drone Object Tracking"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    taken_off = False
    last_dir = Direction.NONE
    tracker = MovingTargetTracker()
    target_label = getattr(active_preset, "target_label", "AUTO")

    try:
        while True:
            frame = drone.get_frame_read().frame
            if frame is None or frame.size == 0:
                continue

            cal.update()
            stacked, target, direction = process_frame(
                frame, cal.hsv, cal.edge, cal.min_area, last_dir,
                tracker=tracker, target_label=target_label,
            )
            last_dir = direction
            lr, fb, ud, yaw = direction_to_rc(direction)

            if not taken_off:
                drone.takeoff()
                taken_off = True

            drone.send_rc_control(lr, fb, ud, yaw)

            mode_label = "SIMULATOR" if simulated else "DRONE LIVE"
            state_text = f"TRACKING: {target.label}" if target.found else "SEARCHING TARGET"
            draw_hud(stacked, f"{mode_label} | {state_text}", drone.get_battery())
            cv2.imshow(win, stacked)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or not Calibration.window_open(win):
                drone.land()
                break
            if key in (ord("v"), ord("V")) and simulated:
                drone.toggle_source()
    finally:
        drone.streamoff()
        if hasattr(drone, "close"):
            drone.close()
        cv2.destroyAllWindows()


def run_color(video_path, webcam=False, preset=None):
    """Standalone vision calibration — no flight commands."""
    using_video = not webcam and video_path and os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path if using_video else 0)
    active_preset = preset or get_preset_for_video(video_path if using_video else None)

    if not using_video:
        cap.set(3, 640)
        cap.set(4, 480)
    print(f"[INFO] {'Playing ' + video_path if using_video else 'Using live webcam'}")
    print(f"[INFO] Vision Preset: {active_preset.name} ({active_preset.description})")

    cal = Calibration(preset=active_preset)
    cal.init_windows()

    win = "Color Object Tracking"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_dir = Direction.NONE
    tracker = MovingTargetTracker()
    target_label = getattr(active_preset, "target_label", "AUTO")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if using_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    continue

            cal.update()
            stacked, target, direction = process_frame(
                frame, cal.hsv, cal.edge, cal.min_area, last_dir,
                tracker=tracker, target_label=target_label,
            )
            last_dir = direction
            state_text = f"TRACKING: {target.label}" if target.found else "SEARCHING TARGET"
            draw_hud(stacked, f"CALIBRATION | {state_text}", 100)
            cv2.imshow(win, stacked)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or not Calibration.window_open(win):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_flight_test(video_path):
    """Basic flight manoeuvre test."""
    drone, simulated = create_drone(demo_video=video_path)
    drone.connect()
    print(f"Drone Battery: {drone.get_battery()}%")
    drone.streamoff()
    drone.streamon()

    win = "Drone Flight Test"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            img = cv2.resize(drone.get_frame_read().frame, (640, 480))
            tag = "SIMULATOR" if simulated else "DRONE LIVE"
            cv2.putText(img, f"BAT: {drone.get_battery()}% | {tag} Flight Test",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                drone.land()
                break
    finally:
        drone.streamoff()
        if hasattr(drone, "close"):
            drone.close()
        cv2.destroyAllWindows()


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aerial Target Tracking",
        epilog="Modes: tracking (default) | color | flight-test",
    )
    parser.add_argument("legacy_mode", nargs="?", default=None)
    parser.add_argument("--mode", "-m", choices=["tracking", "color", "flight-test"])
    parser.add_argument("--preset", "-p", choices=list(PRESETS.keys()), default=None,
                        help="Tracking preset profile (auto-detected if omitted)")
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--webcam", "-w", action="store_true")
    parser.add_argument("--simulate", "-s", action="store_true")
    args = parser.parse_args()

    # resolve mode (support legacy positional args)
    mode = "tracking"
    if args.legacy_mode:
        lm = args.legacy_mode.lower()
        if "color" in lm:
            mode = "color"
        elif any(k in lm for k in ("basic", "flight", "test", "main")):
            mode = "flight-test"
    elif args.mode:
        mode = args.mode

    # Resolve video source and vision preset (Simplified Universal Tracker)
    if args.webcam:
        video = None
        preset = PRESETS.get(args.preset, PRESETS["universal"])
        print("[INFO] Webcam Mode: Universal Aerial Target Tracker active.")
        print("       Real-time dynamic tracking enabled for moving planes, tanks, choppers, and vehicles.\n")
    elif args.video:
        video = find_asset(args.video) or args.video
        preset = PRESETS.get(args.preset) if args.preset else get_preset_for_video(video)
    elif args.preset:
        preset = PRESETS.get(args.preset, PRESETS["universal"])
        default_video_name = preset.default_video or "tank_tracking.mp4"
        video = find_asset(default_video_name)
    else:
        # Default: Universal Tracker
        preset = PRESETS["universal"]
        video = find_asset(preset.default_video or "tank_tracking.mp4")

    print(f"Starting Aerial Target Tracking [{mode.upper()}]")
    print(f"Engine: {preset.name.upper()} — {preset.description}")
    print("Controls: 'v' toggle video/webcam · 'q'/Esc quit\n")

    if mode == "color":
        run_color(video, args.webcam, preset=preset)
    elif mode == "flight-test":
        run_flight_test(video)
    else:
        run_tracking(video, args.webcam, args.simulate, preset=preset)


if __name__ == "__main__":
    main()
