"""All tests for the drone-vision-object-tracking project."""
import os
import unittest

import cv2
import numpy as np

from src.config import (
    find_asset, FRAME_WIDTH, FRAME_HEIGHT, DEAD_ZONE,
    PRESETS, get_preset_for_video,
)
from src.drone import SimulatedDrone, is_drone_connected
from src.vision import (
    HSVBounds, EdgeThresholds, TargetInfo, Direction,
    apply_hsv_mask, detect_edges, detect_target,
    compute_direction, direction_to_rc, process_frame,
)
from src.calibration import Calibration


# ── Config & assets ──────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    def test_find_existing_assets(self):
        for name in (
            "drone_target_tracking.mp4",
            "drone_demo.mp4",  # backwards-compatible alias
            "aircraft_tracking.mp4",
            "reaper_recon.mp4",
            "military_drone_tracking.mp4",  # backwards-compatible alias
            "military_reaper_recon.mp4",    # backwards-compatible alias
            "drone_target_tracking.gif",
            "aircraft_tracking.gif",
            "reaper_recon.gif",
            "military_drone_tracking.gif",  # backwards-compatible alias
            "military_reaper_recon.gif",    # backwards-compatible alias
        ):
            path = find_asset(name)
            self.assertIsNotNone(path, f"Missing: {name}")
            self.assertTrue(os.path.exists(path))

    def test_find_missing_asset(self):
        self.assertIsNone(find_asset("no_such_file_12345.xyz"))
        self.assertIsNone(find_asset(""))

    def test_get_preset_for_video(self):
        self.assertEqual(get_preset_for_video("aircraft_tracking.mp4").name, "military_tracking")
        self.assertEqual(get_preset_for_video("military_drone_tracking.mp4").name, "military_tracking")
        self.assertEqual(get_preset_for_video("reaper_recon.mp4").name, "military_recon")
        self.assertEqual(get_preset_for_video("military_reaper_recon.mp4").name, "military_recon")
        self.assertEqual(get_preset_for_video("drone_target_tracking.mp4").name, "default")
        self.assertEqual(get_preset_for_video("drone_demo.mp4").name, "default")
        self.assertEqual(get_preset_for_video("sky_target_drone.mp4").name, "sky_target")
        self.assertEqual(get_preset_for_video("sky_uav_tracking.mp4").name, "sky_uav")
        self.assertEqual(get_preset_for_video(None).name, "default")


# ── Simulator ────────────────────────────────────────────────────────

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.drone = SimulatedDrone(demo_video=find_asset("drone_target_tracking.mp4"))

    def tearDown(self):
        self.drone.close()

    def test_connect_and_battery(self):
        self.assertTrue(self.drone.connect())
        self.assertGreater(self.drone.get_battery(), 0)

    def test_flight_lifecycle(self):
        self.assertFalse(self.drone.is_flying)
        self.drone.takeoff()
        self.assertTrue(self.drone.is_flying)
        self.drone.land()
        self.assertFalse(self.drone.is_flying)

    def test_rc_control(self):
        self.drone.send_rc_control(10, 20, 30, 40)
        self.assertEqual(self.drone.left_right_velocity, 10)
        self.assertEqual(self.drone.yaw_velocity, 40)

    def test_frame_read(self):
        f = self.drone.get_frame_read()
        self.assertEqual(len(f.frame.shape), 3)


# ── Vision pipeline ─────────────────────────────────────────────────

class TestVision(unittest.TestCase):
    def setUp(self):
        # Synthetic frame: bright green square in the centre
        self.frame = np.zeros((480, 640, 3), np.uint8)
        cv2.rectangle(self.frame, (270, 190), (370, 290), (0, 255, 0), -1)

    def test_hsv_mask(self):
        bounds = HSVBounds(h_min=40, h_max=80, s_min=100, s_max=255, v_min=100, v_max=255)
        mask, result = apply_hsv_mask(self.frame, bounds)
        self.assertEqual(mask.shape, self.frame.shape)
        self.assertTrue(mask[240, 320, 0] > 0)   # centre detected
        self.assertEqual(mask[10, 10, 0], 0)      # corner empty

    def test_detect_edges_and_target(self):
        bounds = HSVBounds(40, 80, 100, 255, 100, 255)
        _, result = apply_hsv_mask(self.frame, bounds)
        edges = detect_edges(result, EdgeThresholds(50, 150))
        target = detect_target(edges, 100)
        self.assertTrue(target.found)
        self.assertAlmostEqual(target.cx, 320, delta=5)
        self.assertAlmostEqual(target.cy, 240, delta=5)

    def test_process_frame(self):
        bounds = HSVBounds(40, 80, 100, 255, 100, 255)
        stacked, target, direction = process_frame(
            self.frame, bounds, EdgeThresholds(50, 150), 100,
        )
        self.assertTrue(target.found)
        self.assertEqual(len(stacked.shape), 3)

    def test_aircraft_tracking_detection(self):
        video_path = find_asset("aircraft_tracking.mp4")
        self.assertIsNotNone(video_path)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret)

        preset = PRESETS["military_tracking"]
        hsv = HSVBounds(*preset.hsv_bounds)
        edge = EdgeThresholds(*preset.edge_thresholds)
        _, target, direction = process_frame(frame, hsv, edge, preset.min_area)
        self.assertTrue(target.found, "Target should be found in aircraft_tracking.mp4")
        self.assertGreater(target.cx, 0)
        self.assertGreater(target.cy, 0)

    def test_reaper_recon_detection(self):
        video_path = find_asset("reaper_recon.mp4")
        self.assertIsNotNone(video_path)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
        ret, frame = cap.read()
        cap.release()
        self.assertTrue(ret)

        preset = PRESETS["military_recon"]
        hsv = HSVBounds(*preset.hsv_bounds)
        edge = EdgeThresholds(*preset.edge_thresholds)
        _, target, direction = process_frame(frame, hsv, edge, preset.min_area)
        self.assertTrue(target.found, "Reaper target should be found in reaper_recon.mp4")
        self.assertGreater(target.cx, 0)
        self.assertGreater(target.cy, 0)


# ── Direction / tracking logic ───────────────────────────────────────

class TestDirection(unittest.TestCase):
    def test_not_found(self):
        self.assertEqual(compute_direction(TargetInfo()), Direction.NONE)

    def test_left(self):
        t = TargetInfo(found=True, cx=100, cy=240)
        self.assertEqual(compute_direction(t), Direction.LEFT)
        lr, fb, ud, yaw = direction_to_rc(Direction.LEFT)
        self.assertEqual(yaw, -60)

    def test_right(self):
        t = TargetInfo(found=True, cx=500, cy=240)
        self.assertEqual(compute_direction(t), Direction.RIGHT)

    def test_up(self):
        t = TargetInfo(found=True, cx=320, cy=80)
        self.assertEqual(compute_direction(t), Direction.UP)
        _, _, ud, _ = direction_to_rc(Direction.UP)
        self.assertEqual(ud, 60)

    def test_down(self):
        t = TargetInfo(found=True, cx=320, cy=400)
        self.assertEqual(compute_direction(t), Direction.DOWN)

    def test_centred(self):
        t = TargetInfo(found=True, cx=320, cy=240)
        self.assertEqual(compute_direction(t), Direction.NONE)


# ── Calibration (headless safe) ──────────────────────────────────────

class TestCalibration(unittest.TestCase):
    def test_safe_defaults(self):
        cal = Calibration()
        self.assertEqual(cal.hsv.h_min, 29)
        cal.update()  # must not crash even without windows
        self.assertEqual(cal.hsv.h_min, 29)

    def test_calibration_with_preset(self):
        preset = PRESETS["military_tracking"]
        cal = Calibration(preset=preset)
        self.assertEqual(cal.hsv.h_min, 0)
        self.assertEqual(cal.hsv.h_max, 180)
        self.assertEqual(cal.edge.threshold1, 40)
        self.assertEqual(cal.min_area, 80)


if __name__ == "__main__":
    unittest.main()
