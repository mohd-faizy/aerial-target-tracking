<div align="center">

# 🛸 Drone Vision & Autonomous Object Tracking

**Real-time closed-loop computer vision tracking, dynamic HSV/edge calibration, and autonomous flight control for aerial drones.**

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Environment-uv%20Package%20Manager-DE5FE9?style=for-the-badge&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![OpenCV](https://img.shields.io/badge/OpenCV-contrib--python-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-20%2F20%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/mohd-faizy/drone-vision-object-tracking)

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-demo-showcase">Demo Showcase</a> •
  <a href="#-quick-start-with-uv">Quick Start (uv)</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-vision-presets">Vision Presets</a> •
  <a href="#-cli-reference">CLI Reference</a> •
  <a href="#-running-tests">Testing</a>
</p>

---

### 🎥 Tactical Aerial Tracking & Intercept (Demo)
<p align="center">
  <img src="asset/demo_gifs/aircraft_tracking.gif" alt="Tactical Aerial Drone Tracking Demo" width="900" style="border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);" />
</p>

</div>

---

## 📌 Overview

**Drone Vision Object Tracking** is a robust, modular computer vision and autonomous flight guidance system engineered for unmanned aerial vehicles (UAVs). It pairs high-speed frame processing with a deadzone feedback control loop, enabling aerial platforms to identify, lock onto, and dynamically pursue moving targets in real time.

The project features a **dual-engine architecture**:
1. **Physical Drone Driver**: Real-time UDP telemetry and RC control socket streaming (DJI Tello compatible).
2. **Virtual Flight Simulator**: Seamless offline fallback mode equipped with looped mission video playback, live webcam hot-switching, and full simulated flight telemetry.

---

## ✨ Key Features

- **🎯 Autonomous Target Tracking**: Real-time target centroid isolation, Euclidean distance vector calculation, and dynamic bounding box fitting.
- **🧭 Central Deadzone Guidance Loop**: Evaluates centroid offsets against a configurable central tolerance window to command corrective pitch, roll, throttle, and yaw velocity.
- **🎛️ Live Calibration GUI**: Interactive OpenCV trackbar control panels for instant fine-tuning of HSV bounds (Hue, Saturation, Value), Canny edge thresholds, and contour area limits without restarts.
- **⚡ Ultra-Fast Environment with `uv`**: Fully configured for [Astral `uv`](https://github.com/astral-sh/uv), delivering lightning-fast virtual environment setup and deterministic dependency management.
- **🖼️ 2x2 Multi-View Quad Canvas**: Simultaneously renders the original camera stream, isolated HSV color mask, dilated Canny edge map, and final target HUD tracking overlay.
- **🛰️ Auto-Detected Mission Presets**: Includes pre-tuned vision profiles for distinct operational environments (sky quadcopters, fixed-wing UAVs, reconnaissance drones, and indoor demo objects).
- **🛡️ Resilient Failover & Safety**: Automatic hardware discovery with graceful fallback to video simulator, frame validation, and one-key safe landing (`[Q]` / `[ESC]`).

---

## 🎬 Demo Showcase

Explore the tracking engine across diverse operational environments:

| Mission Profile | Real-time Vision & Tracking Demo | Details |
| :--- | :---: | :--- |
| **Tactical Reticle & Intercept**<br>`military_tracking` | ![Aircraft Tracking](asset/demo_gifs/aircraft_tracking.gif) | Real-time aircraft & reticle tracking with edge contour isolation and lock-on telemetry. |
| **Autonomous Target Follow**<br>`default` | ![Drone Target Tracking](asset/demo_gifs/drone_target_tracking.gif) | Centroid lock-on with automated yaw and elevation adjustment against high-contrast moving targets. |
| **Aerial Recon & Horizon Surveillance**<br>`military_recon` | ![Reaper Recon](asset/demo_gifs/reaper_recon.gif) | Fixed-wing UAV tracking against complex terrestrial background and variable lighting conditions. |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Input ["Video Capture Layer"]
        A1[Physical Drone UDP Stream\n192.168.10.1:8889] --> B{Drone Connected?}
        A2[Simulated Video File\nasset/*.mp4] --> B
        A3[Live USB Webcam\nDevice 0] --> B
    end

    B -- Yes --> C[Hardware UDP Socket Driver]
    B -- No / --simulate --> D[Virtual Drone Simulator]

    subgraph Vision ["Computer Vision Pipeline"]
        C & D --> E[Frame Acquisition: 640x480]
        E --> F[Color Thresholding: HSV Mask]
        F --> G[Gaussian Smoothing & Canny Edge Detection]
        G --> H[Morphological Dilation]
        H --> I[Contour Extraction & Area Filtering]
        I --> J[Centroid & Target Coordinates Calculation]
    end

    subgraph Control ["Deadzone Feedback & Flight Guidance"]
        J --> K{Inside Central Deadzone?}
        K -- Centroid < X - Deadzone --> L1[Rotate Left / Yaw Counter-Clockwise]
        K -- Centroid > X + Deadzone --> L2[Rotate Right / Yaw Clockwise]
        K -- Centroid < Y - Deadzone --> L3[Ascend / Elevation UP]
        K -- Centroid > Y + Deadzone --> L4[Descend / Elevation DOWN]
        K -- Centroid In Deadzone Window --> L5[LOCKED ON TARGET: Hover Stable]
    end

    subgraph Output ["Display & Flight Telemetry"]
        L1 & L2 & L3 & L4 & L5 --> M[Send RC Velocities: lr, fb, ud, yaw]
        M --> N[Render 2x2 Multi-View Quad Canvas + HUD]
    end
```

### 🔲 2x2 Quad-View Vision Display

When the pipeline executes, the display provides full situational awareness:

```
┌───────────────────────────────┬───────────────────────────────┐
│        [TOP-LEFT]             │         [TOP-RIGHT]           │
│      Raw Camera Stream        │       HSV Masked Result       │
│  Unprocessed live video feed  │ Isolated target color band    │
├───────────────────────────────┼───────────────────────────────┤
│       [BOTTOM-LEFT]           │        [BOTTOM-RIGHT]         │
│     Dilated Canny Edges       │ Target Overlay & HUD Telemetry│
│ Structural edge contours      │ Bounding box, vector & status │
└───────────────────────────────┴───────────────────────────────┘
```

---

## 🚀 Quick Start with `uv`

This repository uses [**`uv`**](https://github.com/astral-sh/uv), an extremely fast Python package manager and resolver.

### 1. Prerequisites

Ensure you have Python 3.10+ (tested up to Python 3.14) and `uv` installed.

If you haven't installed `uv` yet:
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### 2. Clone the Repository

```bash
git clone https://github.com/mohd-faizy/drone-vision-object-tracking.git
cd drone-vision-object-tracking
```

### 3. Create Virtual Environment & Install Dependencies

Create the isolated virtual environment and install dependencies in one step:

```bash
# Create the virtual environment
uv venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
uv pip install -r requirements.txt
```

> **Tip:** You can also run commands directly with `uv run` without manually activating the environment!
> ```bash
> uv run python main.py
> ```

---

## 💻 Usage & Execution Modes

The unified entry point is [main.py](main.py). It supports multiple operational modes and custom configurations.

### 1. Full Autonomous Object Tracking (Default)
Executes target acquisition, deadzone control, and flight commands. If a physical drone is not found on the local Wi-Fi, it seamlessly runs the flight simulator.

```bash
# Run with default tracking
uv run python main.py

# Force simulation mode explicitly (even if drone network is present)
uv run python main.py --simulate

# Run tracking on a specific demo video
uv run python main.py --video asset/aircraft_tracking.mp4

# Run tracking using live webcam
uv run python main.py --webcam
```

### 2. Standalone Vision Calibration Mode (No Flight)
Opens the video feed alongside the interactive HSV & parameter tuning trackbars without sending flight commands.

```bash
uv run python main.py --mode color

# Calibrate against a custom video
uv run python main.py --mode color --video asset/reaper_recon.mp4

# Calibrate using live webcam
uv run python main.py --mode color --webcam
```

### 3. Hardware Flight Test Mode
Runs a basic connection and flight verification test.

```bash
uv run python main.py --mode flight-test
```

---

## ⌨️ Interactive Controls

During active camera/simulator streaming, use the following hotkeys:

| Key | Action |
| :---: | :--- |
| `[V]` / `[v]` | **Toggle Video Source**: Switch on-the-fly between demo video file and live USB webcam. |
| `[Q]` / `[q]` | **Emergency Land & Exit**: Commands safe drone landing and terminates all OpenCV windows. |
| `[ESC]` | **Immediate Exit**: Safely shuts down video capture and releases network sockets. |

---

## ⚙️ Vision Presets

The system includes pre-configured vision tuning profiles located in [src/config.py](src/config.py):

| Preset | Target Description | HSV Bounds (H, S, V) | Canny Thresholds | Min Area |
| :--- | :--- | :--- | :--- | :--- |
| `default` | High-visibility yellow/green objects & demo balls | `[29-65, 85-255, 60-255]` | `(166, 171)` | 1000 px |
| `sky_target` | Quadcopter drone in open airspace | `[20-80, 50-255, 40-255]` | `(100, 200)` | 800 px |
| `sky_uav` | Fixed-wing UAV on horizon | `[20-80, 50-255, 40-255]` | `(100, 200)` | 800 px |
| `military_tracking` | MQ-9 onboard camera HUD targeting aircraft / reticle | `[0-180, 0-255, 0-255]` | `(40, 120)` | 80 px |
| `military_recon` | Combat UAV tracking against ground terrain | `[0-180, 0-95, 30-160]` | `(40, 120)` | 600 px |

To force a specific preset profile via the command line:
```bash
uv run python main.py --preset military_tracking
uv run python main.py --preset military_recon
uv run python main.py --preset sky_target
```

---

## 📖 CLI Reference

```
usage: main.py [-h] [--mode {tracking,color,flight-test}]
               [--preset {default,sky_target,sky_uav,military_tracking,military_recon}]
               [--video VIDEO] [--webcam] [--simulate]
               [legacy_mode]

Drone Vision Object Tracking

positional arguments:
  legacy_mode           Legacy shorthand mode ('color', 'flight-test')

options:
  -h, --help            Show this help message and exit
  --mode, -m            Operational mode: tracking (default) | color | flight-test
  --preset, -p          Vision preset profile (auto-detected if omitted)
  --video VIDEO         Path to custom video file
  --webcam, -w          Force live USB webcam input
  --simulate, -s        Force simulator mode even if physical drone is detected
```

---

## 📡 Hardware & Network Setup

When connecting to a physical drone (e.g., DJI Tello):

1. **Power On**: Power on your drone and wait for its Wi-Fi indicator to flash.
2. **Connect Wi-Fi**: Connect your host computer to the drone's Wi-Fi network (`TELLO-XXXXXX`).
3. **Launch**:
   ```bash
   uv run python main.py
   ```
4. **Auto-Discovery**: The application sends a UDP handshake ping to `192.168.10.1:8889`. Once confirmed, it enables the camera stream (`udp://@0.0.0.0:11111`) and handles real-time flight control commands.

```
Host Computer (Port 9000) ──── UDP Command String ───► Drone (192.168.10.1:8889)
Host Computer (Port 11111) ◄── H.264 Video Stream ──── Drone Video Driver
```

---

## 📂 Project Structure

```text
drone-vision-object-tracking/
├── .venv/                      # Virtual environment (managed by uv)
├── asset/                      # Video assets and demo media
│   ├── demo_gifs/              # Visual showcase recordings
│   │   ├── aircraft_tracking.gif
│   │   ├── drone_target_tracking.gif
│   │   └── reaper_recon.gif
│   ├── aircraft_tracking.mp4
│   ├── drone_target_tracking.mp4
│   └── reaper_recon.mp4
├── src/                        # Core application package
│   ├── __init__.py
│   ├── calibration.py          # OpenCV dynamic trackbar manager
│   ├── config.py               # Constants, thresholds, and preset profiles
│   ├── drone.py                # Hardware UDP driver & virtual simulator
│   └── vision.py               # Computer vision, contour, and HUD engine
├── tests/                      # Automated test suite
│   ├── __init__.py
│   └── test_all.py             # Unit & integration tests (20 test cases)
├── LICENSE                     # MIT License
├── main.py                     # Unified CLI launcher & controller
├── pyproject.toml              # Project metadata & dependencies
├── README.md                   # Project documentation
├── requirements.txt            # Dependency declarations
└── uv.lock                     # Deterministic dependency lockfile
```

---

## 🧪 Running Tests

The test suite covers configuration, asset discovery, simulator flight lifecycle, HSV segmentation, contour detection, and RC command mapping.

Run tests using `uv`:

```bash
uv run pytest -v
```

Expected output:
```text
============================= test session starts =============================
platform win32 -- Python 3.14.0, pytest-9.0.2, pluggy-1.6.0
collected 20 items

tests\test_all.py ....................                                   [100%]

============================= 20 passed in 0.32s ==============================
```

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome!
1. **Fork** the repository.
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`.
3. **Commit** your changes: `git commit -m 'feat: Add amazing feature'`.
4. **Push** to your branch: `git push origin feature/amazing-feature`.
5. **Open** a Pull Request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/mohd-faizy">mohd-faizy</a> for the computer vision and robotics community.</sub>
</div>
