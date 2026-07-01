# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous Guided Vehicle (AGV) challenge project from Makerweek. A camera-based pipeline detects the playing field, finds walls, localizes the AGV via ArUco markers, plans a path with A*, and drives the robot via HTTP.

## Running

All scripts must be run from the `Makerweek/` directory so that relative imports resolve correctly:

```bash
cd Makerweek
python Main.py          # Full autonomous loop
python Test.py          # End-to-end pipeline test (no robot, fixed AGV position)
python TestMarkers.py   # Live ArUco marker detection debug view
python TestWalls.py     # Live wall/blue-mask detection debug view
python TestCamGerade.py # Live top-down perspective rectification test
```

Dependencies: `opencv-python` (with `cv2.aruco`), `numpy`, `requests`.

## Architecture

The pipeline in `Main.py` runs these stages in order each tick:

1. **Camera** (`vision/camera.py`) — streams MJPEG from `http://10.250.150.224:8081/`, resizes to 640×480.
2. **Corner detection** (`vision/markers.py`) — detects ArUco DICT_4X4_100 markers with IDs 1–4 at the four field corners. Caches the last seen position of each corner so all four only need to be visible once. Returns a `{"tl","tr","br","bl"}` dict.
3. **Perspective rectification** (`vision/rectify.py`) — computes a homography from the four corners and warps the frame to an 800×800 top-down view.
4. **Wall detection** (`vision/walls.py`) — filters the top-down image for blue (HSV 100–140) objects with morphological cleanup; returns a binary mask.
5. **Grid** (`planning/grid.py`) — divides the 800×800 mask into a 40×40 grid; a cell is blocked (`1`) if its mean pixel value exceeds 20.
6. **Pose** (`vision/pose.py` + `vision/markers.py`) — detects the AGV's ArUco marker, computes center `(x, y)` and heading `theta` from the first edge vector. `world_to_grid()` converts pixel coords to grid indices.
7. **Path planning** (`planning/astar.py`) — Manhattan-heuristic A* on the 40×40 grid; returns a list of `(gx, gy)` waypoints or `None` if unreachable.
8. **Motion** (`control/kinematics.py` + `control/agv_api.py`) — `turn_to_angle` and `drive_distance` convert geometry to motor steps (`STEPS_PER_CM = 20`, `STEPS_PER_RAD = 100`) and POST to `http://<agv-ip>/drive` with `{"left": steps, "right": steps}`.

## Known API Mismatches in Main.py

`Main.py` was written ahead of some implementations and contains calls that do not match current module APIs:

| Call in `Main.py` | Actual function | Issue |
|---|---|---|
| `compute_warp(corners)` / `apply_warp(frame, M)` | `compute_warp_matrix()` / `warp_topdown()` in `rectify.py` | Different names |
| `build_grid(wall_mask)` | `mask_to_grid(mask)` in `planning/grid.py` | Different name |
| `detect_agv_marker(topdown)` | Not yet implemented in `vision/markers.py` | Missing function |
| `astar(grid, pose)` | `astar(grid, start, goal)` — needs 3 args | Missing `goal` arg |

These need to be reconciled before `Main.py` runs end-to-end.

## Key Constants

| Constant | Value | Location |
|---|---|---|
| Top-down resolution | 800×800 px | `rectify.py` |
| Grid size | 40×40 cells | `grid.py`, `pose.py` |
| Corner ArUco IDs | tl=1, tr=2, br=3, bl=4 | `markers.py` |
| Camera stream URL | `http://10.250.150.224:8081/` | `Main.py`, test scripts |
| Steps per cm | 20 | `kinematics.py` |
| Steps per radian | 100 | `kinematics.py` |
