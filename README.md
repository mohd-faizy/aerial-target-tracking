# Aerial Target Tracking


<div align="center">
<img src="asset/img/banner.png" width="800" alt="Aerial Target Tracking Banner" style="border-radius: 12px;" />

<br/>

**Real-time computer vision & autonomous flight guidance for UAVs**

*Detect · Lock · Track · Pursue — all in real time*

<br/>

[![Author](https://img.shields.io/badge/Author-mohd--faizy-red?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/mohd-faizy/aerial-target-tracking?style=for-the-badge&logo=git&logoColor=white)](https://github.com/mohd-faizy/aerial-target-tracking/commits/main)
[![GitHub Issues](https://img.shields.io/github/issues/mohd-faizy/aerial-target-tracking?style=for-the-badge&logo=github&color=yellow)](https://github.com/mohd-faizy/aerial-target-tracking/issues)
[![Stars](https://img.shields.io/github/stars/mohd-faizy/aerial-target-tracking?style=for-the-badge&logo=github&color=gold)](https://github.com/mohd-faizy/aerial-target-tracking/stargazers)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-0059b3?style=for-the-badge&logo=handshake&logoColor=white)](https://github.com/mohd-faizy/aerial-target-tracking)

<br/>

[Key Features](#key-features) · [Live Demos](#live-demos) · [Quick Start](#quick-start) · [Usage](#usage--execution-modes) · [Architecture](#system-architecture) · [CLI Reference](#cli-reference) · [Tests](#running-tests)

</div>

---

## What Is This?

A **modular, real-time computer vision pipeline** + **autonomous flight controller** for UAVs that:

- **Detects & locks** onto moving targets (tanks, vehicles, aircraft, drones)
- **Calculates** centroid offsets & generates corrective flight commands
- **Falls back gracefully** — no drone? Runs a full flight simulator with looped video
- **Sets up in seconds** with `uv` (ultra-fast Python env manager)

> **Dual-Engine Architecture** — works with a **physical DJI Tello drone** (UDP telemetry) *or* a **virtual flight simulator** (offline video playback + simulated telemetry).

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Autonomous Target Tracking** | Real-time centroid isolation, Euclidean distance vectors, dynamic bounding boxes |
| **Military Vehicle Tracking** | MTI, velocity vectors, heading estimation, tactical HUD brackets |
| **Deadzone Guidance Loop** | Configurable tolerance window -> corrective pitch / roll / throttle / yaw |
| **Live Calibration GUI** | OpenCV trackbars for HSV, Canny, and contour tuning — no restarts |
| **2x2 Quad-View Canvas** | Raw feed + HSV mask + Canny edges + HUD overlay — all at once |
| **Auto-Detect Presets** | Pre-tuned profiles: `tank`, `military_vehicle`, `aircraft`, `military_recon`, etc. |
| **Hot-Switch Sources** | Toggle between video file <-> live webcam with `[V]` key mid-stream |
| **Resilient Failover** | Auto hardware discovery -> graceful fallback -> one-key safe landing |

---

## Live Demos

### Armored Tank Tracking

> Top-down aerial surveillance locking onto a moving MBT — velocity, heading & deadzone correction in real time.

<p align="center">
  <img src="asset/gifs/tank.gif" alt="Tank Tracking Demo" width="900" style="border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,0.35);" />
</p>

---

### Aircraft & Reticle Intercept Tracking

> High-precision aircraft contour isolation with dynamic target lock telemetry & edge detection.

<p align="center">
  <img src="asset/gifs/aircraft_tracking.gif" alt="Aircraft Tracking Demo" width="900" style="border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,0.35);" />
</p>

---

### Aerial Reconnaissance & Horizon Surveillance

> Fixed-wing UAV tracking against complex terrain, altitude gradients & shifting horizon lighting.

<p align="center">
  <img src="asset/gifs/reaper_recon.gif" alt="Reaper Recon Demo" width="900" style="border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,0.35);" />
</p>

---

## Quick Start

### Prerequisites

- **Python 3.10+** (tested up to 3.14)
- [**`uv`**](https://github.com/astral-sh/uv) — ultra-fast Python package manager

```bash
# Install uv (pick one)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh                                        # macOS/Linux
pip install uv                                                                          # via pip
```

### Setup (3 commands)

```bash
git clone https://github.com/mohd-faizy/aerial-target-tracking.git
cd aerial-target-tracking
uv venv && uv pip install -r requirements.txt
```

### Activate & Run

```bash
# Activate environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Launch tracker
uv run python main.py
```

> **Tip:** Skip activation entirely — just use `uv run python main.py` directly!

---

## Usage & Execution Modes

### Universal Tracking (Default)

```bash
uv run python main.py                                        # Auto-detect everything
uv run python main.py --webcam                               # Live webcam input
uv run python main.py --video asset/videos/tank.mp4          # Specific video file
uv run python main.py --simulate                             # Force simulator mode
```

### Vision Calibration (No Flight)

```bash
uv run python main.py --mode color                           # Calibrate with default video
uv run python main.py --mode color --webcam                  # Calibrate with live webcam
uv run python main.py --mode color --video <path>            # Calibrate with custom video
```

### Hardware Flight Test

```bash
uv run python main.py --mode flight-test                     # Connection & flight verification
```

---

## Hotkeys

| Key | Action |
|:---:|:-------|
| `T` | **Cycle target profile** — `AUTO` -> `TANK` -> `MIL-VEHICLE` -> `HELICOPTER` -> `AIRCRAFT` -> `MOVING OBJECT` |
| `V` | **Toggle video source** — switch between demo video <-> live webcam |
| `Q` | **Emergency land & exit** — safe drone landing + close all windows |
| `ESC` | **Immediate exit** — release capture & network sockets |

---

## Auto-Classification Engine

The tracker **automatically classifies** targets — no manual preset required:

| Target | ID Tag | How It's Detected |
|:-------|:-------|:------------------|
| **Tank / MBT** | `TANK [MBT]` | Compact armored silhouette · AR 1.05–2.25 · solidity >= 0.28 |
| **Military Vehicle** | `MIL-VEHICLE` | Elongated chassis · AR > 2.25 · solidity >= 0.22 |
| **Helicopter** | `HELICOPTER` | Rotor span + tail boom · AR 1.25–2.8 · solidity 0.15–0.42 |
| **Aircraft / Jet** | `AIRCRAFT` | Fixed-wing silhouette · aerodynamic wingspan · upper horizon |
| **Moving Object** | `MOVING OBJECT` | Velocity vector > 2.2 px/frame · smoothed heading |

**Manual preset override:**

```bash
uv run python main.py --preset tank
uv run python main.py --preset military_vehicle
uv run python main.py --preset aircraft
uv run python main.py --preset military_tracking
uv run python main.py --preset military_recon
```

---

## System Architecture

```mermaid
flowchart TD
    subgraph Input ["Video Capture Layer"]
        A1["Physical Drone<br/>UDP 192.168.10.1:8889"] --> B{"Drone<br/>Connected?"}
        A2["Simulated Video<br/>asset/videos/*.mp4"] --> B
        A3["Live Webcam<br/>Device 0"] --> B
    end

    B -->|Yes| C["Hardware UDP Driver"]
    B -->|No| D["Virtual Simulator"]

    subgraph Vision ["Computer Vision Pipeline"]
        C --> E["Frame Acquisition 640x480"]
        D --> E
        E --> F["HSV Color Thresholding"]
        F --> G["Gaussian Blur + Canny Edges"]
        G --> H["Morphological Dilation"]
        H --> I["Contour Extraction & Filtering"]
        I --> J["Centroid Calculation"]
    end

    subgraph Control ["Deadzone Feedback Loop"]
        J --> K{"Inside<br/>Deadzone?"}
        K -->|Left| L1["Yaw CCW"]
        K -->|Right| L2["Yaw CW"]
        K -->|Above| L3["Throttle UP"]
        K -->|Below| L4["Throttle DOWN"]
        K -->|Centered| L5["LOCKED ON"]
    end

    subgraph Output ["Display & Telemetry"]
        L1 --> M["RC Velocities"]
        L2 --> M
        L3 --> M
        L4 --> M
        L5 --> M
        M --> N["2x2 Quad Canvas + HUD"]
    end
```

### Quad-View Display Layout

```
┌─────────────────────┬─────────────────────┐
│   Raw Camera Feed   │   HSV Masked View   │
│   (unprocessed)     │   (color isolation) │
├─────────────────────┼─────────────────────┤
│   Dilated Canny     │   Target HUD +      │
│   Edge Map          │   Flight Telemetry  │
└─────────────────────┴─────────────────────┘
```

---

## CLI Reference

```
usage: main.py [-h] [--mode {tracking,color,flight-test}]
               [--preset {default,aircraft,tank,...}]
               [--video VIDEO] [--webcam] [--simulate]

Options:
  -h, --help          Show help message
  --mode, -m          tracking (default) | color | flight-test
  --preset, -p        Vision preset profile (auto-detected if omitted)
  --video VIDEO       Path to custom video file
  --webcam, -w        Force live USB webcam input
  --simulate, -s      Force simulator mode
```

---

## Hardware Setup (DJI Tello)

1. **Power on** the drone -> wait for Wi-Fi indicator to flash
2. **Connect** your PC to `TELLO-XXXXXX` Wi-Fi network
3. **Launch** -> `uv run python main.py`
4. **Auto-discovery** handles the rest:

```
Host (Port 9000)  ──── UDP Commands ────►  Drone (192.168.10.1:8889)
Host (Port 11111) ◄── H.264 Video ──────  Drone Video Driver
```

---

## Project Structure

```
aerial-target-tracking/
├── asset/
│   ├── gifs/                    # Demo GIF recordings
│   ├── img/                     # Banner & branding
│   └── videos/                  # Mission video feeds
├── src/
│   ├── calibration.py           # OpenCV trackbar manager
│   ├── config.py                # Constants, thresholds & presets
│   ├── drone.py                 # UDP driver + virtual simulator
│   └── vision.py                # CV pipeline, contour & HUD engine
├── tests/
│   └── test_all.py              # 20+ unit & integration tests
├── main.py                      # Unified CLI entry point
├── pyproject.toml               # Project metadata
├── requirements.txt             # Dependencies
└── uv.lock                      # Deterministic lockfile
```

---

## Running Tests

```bash
uv run pytest -v
```

```
============================= test session starts =============================
collected 20 items
tests/test_all.py ....................                                   [100%]
============================= 20 passed in 0.32s ==============================
```

---

## Contributing

1. **Fork** this repo
2. **Branch** -> `git checkout -b feature/amazing-feature`
3. **Commit** -> `git commit -m 'feat: Add amazing feature'`
4. **Push** -> `git push origin feature/amazing-feature`
5. **PR** -> Open a Pull Request

---

## License

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

### Connect

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://mohdfaizy.vercel.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohd-faizy/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mohd-faizy)
[![Credly](https://img.shields.io/badge/Credly-FF6B00?style=for-the-badge&logo=credly&logoColor=white)](https://www.credly.com/users/mohd-faizy)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/F4izy)
[![Stack Exchange](https://img.shields.io/badge/Stack_Exchange-1E5397?style=for-the-badge&logo=stack-exchange&logoColor=white)](https://ai.stackexchange.com/users/36737/faizy)

</div>