# Voxel car demo — dataset generator + interactive preview

A small procedural-world driving sim built around voxel terrain, a random
composite trajectory, and up to four ray-marched cameras attached to the car.
Ships with two entry points:

* **`python app.py`** — Gradio UI for live preview and one-off video / sample
  creation.
* **`python generate.py …`** — headless CLI that writes an N-sample dataset.

---

## Install

```bash
pip install gradio numpy imageio imageio-ffmpeg Pillow
```

Python ≥ 3.9 (dataclasses + typing).

---

## Layout

```
voxel_project/
├── app.py            # launches the Gradio UI
├── generate.py       # CLI dataset generator
├── README.md
└── src/
    ├── __init__.py
    ├── config.py     # dataclasses (CameraConfig, WorldConfig, TrajectoryConfig, RenderConfig)
    ├── noise.py      # multi-octave value noise
    ├── trajectory.py # random line / arc / sine composite trajectories
    ├── world.py      # trajectory-aware voxel world with road + shoulder carving
    ├── camera.py     # extrinsics + ray-march rendering
    ├── bev.py        # top-down render with car, trajectory, frustums
    ├── compose.py    # side-by-side frame composer (UI only)
    ├── dataset.py    # generate_sample / generate_dataset
    └── ui.py         # Gradio interface + callbacks
```

Always run the scripts from the **project root** (so that `src.*` imports resolve).

---

## What's new over the single-file version

### 1. Composite random trajectory

Built in `src/trajectory.py`. A sequence of segments is drawn from:

* `line` — straight travel, heading preserved.
* `arc` — signed-radius circular arc; new heading = `θ + L/R`.
* `sine` — perpendicular wiggle with a `sin²(π·τ)` amplitude envelope so that
  both the offset and its derivative are zero at both endpoints — the segment
  joins its neighbours with continuous tangent.

Then Gaussian noise (per-waypoint) is added, followed by a box-filter smoothing
pass, then a hard clip keeps the path inside `[margin, grid − margin]`.

Probabilities, segment lengths, radii, amplitudes and cycle counts all come
from a `RandomState` seeded by `TrajectoryConfig.seed`.

### 2. Adaptive landscape

`src/world.py` now takes the trajectory as input and shapes terrain around it:

* **Road band** (distance ≤ `road_width`): heights flattened to 0.
* **Shoulder** (`road_width < d ≤ road_width + shoulder_extra`): heights are
  capped by a linear fade so the terrain rises smoothly from the road rather
  than presenting a vertical cliff.

Carving is done with per-waypoint vectorised patches — fast even for ~200
waypoints × (2·R_sh+1)² cells.

### 3. Dataset generator

`src/dataset.py` exports `generate_sample(...)` and `generate_dataset(...)`.
Per run (see schema below), the following artefacts are written:

| File | Contents |
|------|----------|
| `voxels.npz`        | compressed bool voxel grid + int32 heightmap |
| `trajectory.npy`    | (N, 2) float32 waypoints in voxel units |
| `bev_static.png`    | top-down PNG at the start pose, frustums shown |
| `bev_video.mp4`     | full drive from the BEV |
| `cam_<i>_<name>.mp4`| first-person video per **enabled** camera |
| `metadata.json`     | all configs, seed, segment types, files map |

The CLI adds a top-level `manifest.json` listing every run.

---

## Running

### Gradio UI

```bash
python app.py
```

Gives you sliders for world, trajectory, drive/render and four camera
accordions, plus a *Save full dataset sample* button that writes everything
described above to a user-specified directory.

### CLI dataset

```bash
python generate.py --out-dir ./data --n 5 --base-seed 100
```

Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--n`             | 5     | Number of runs. |
| `--base-seed`     | 100   | World seed = `base+i`; trajectory seed = `base+i+1000`. |
| `--grid-size`     | 80    | Voxels per side (square grid). |
| `--max-obst-h`    | 14    | Max obstacle height, voxels. |
| `--road-w`        | 3     | Road half-width, voxels. |
| `--shoulder`      | 2     | Extra shoulder radius beyond the road. |
| `--n-segments`    | 6     | Trajectory segments per run. |
| `--traj-noise`    | 0.4   | Per-waypoint Gaussian noise (voxels). |
| `--num-frames`    | 40    | Frames per video. |
| `--img-w/--img-h` | 180/136 | Camera image resolution. |
| `--fps`           | 10    | Output video FPS. |
| `--enable-cams`   | (default setup) | e.g. `1,3,4` to override which cameras produce video. |

All non-seed fields are copied verbatim into every run; only seeds vary.

---

## Output schema (per-run)

```
run_0000/
├── voxels.npz         # dict with 'voxels' (bool, shape (X, Y, Z)) and 'heights' (int32, shape (X, Y))
├── trajectory.npy     # float32 array, shape (N_waypoints, 2), in voxel units (x, y)
├── bev_static.png     # BEV at frame 0
├── bev_video.mp4
├── cam_0_front.mp4    # one per enabled camera
├── cam_2_left.mp4
└── metadata.json
```

### `metadata.json` shape

```jsonc
{
  "run_id":       "run_0000",
  "generated_at": "2026-04-19T10:30:12",
  "world":        { "seed": 100, "grid_size": 80, ... },
  "trajectory_params": { "seed": 1100, "n_segments": 6, ... },
  "trajectory_info": {
    "num_waypoints":        172,
    "segment_types":        ["line", "arc", "sine", "line", "arc", "line"],
    "length_voxels_approx": 112.4,
    "x_range":              [5.0, 73.0],
    "y_range":              [26.8, 53.4],
    "frame_indices":        [0, 4, 8, ...]
  },
  "cameras":     [ { "idx": 0, "name": "front", "enabled": true, "fov": 75, "color_rgb": [60,200,230], ... }, ... ],
  "render":      { "img_w": 180, "img_h": 136, "num_frames": 40, "fps": 10, ... },
  "voxel_shape": [80, 80, 17],
  "files": {
    "voxels":        "voxels.npz",
    "trajectory":    "trajectory.npy",
    "bev_static":    "bev_static.png",
    "bev_video":     "bev_video.mp4",
    "camera_videos": { "0_front": "cam_0_front.mp4", "2_left": "cam_2_left.mp4" }
  }
}
```

---

## Coordinate conventions

* **World** — X east, Y north, Z up, right-handed.
* **Camera** — OpenCV: X right, Y down, Z forward.
* `voxel[i, j, k]` occupies the cube `[i, i+1) × [j, j+1) × [k, k+1)`.
* Car heading = `(hx, hy)`; car right = `(hy, -hx)`; up = `+Z`.
* `+ yaw` rotates a vector toward the car's right side
  (so yaw = 0 → forward, yaw = +90 → right, yaw = ±180 → rear).

These are enforced everywhere — see the docstrings in `src/config.py`.
