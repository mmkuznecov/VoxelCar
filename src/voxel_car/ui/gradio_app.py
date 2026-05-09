"""Gradio UI for the voxel car demo.

Two modes:

1. Random generator
   - mostly the original procedural demo
   - previews random worlds/cameras
   - renders BEV + selected camera video
   - saves full dataset samples

2. Closed-loop model demo
   - loads trained OccNet checkpoint from Hugging Face Hub
   - optionally loads trained PPO policy from Hugging Face Hub
   - default Hub repo: mmkuznecov/SynthOccPredModels
   - runs one random scenario
   - controller:
        A* planner over OccNet predictions
        PPO-RL planner over OccNet predictions
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio
import numpy as np

from ..common.config import (
    NUM_CAMERAS,
    CAMERA_COLORS,
    DEFAULT_CAMERA_SPECS,
    CameraConfig,
    WorldConfig,
    TrajectoryConfig,
    RenderConfig,
)
from ..worldgen.world import get_world_and_trajectory
from ..worldgen.scenarios import SCENARIO_PRESETS, generate_scenario
from ..rendering.camera import compute_camera_world_pose, render_camera_view
from ..rendering.bev import compute_heading, render_bev, build_camera_overlays
from ..rendering.compose import compose_side_by_side
from ..datasets.dataset import generate_sample
from ..geometry import compute_fov_mask
from ..simulation import (
    simulate_episode,
    episode_summary,
    save_episode_video,
    save_summary_figure,
)
from ..rl import RLPolicyAdapter
from ..hub import (
    DEFAULT_HF_REPO,
    DEFAULT_OCC_FILENAME,
    DEFAULT_RL_FILENAME,
    load_occnet_from_hf,
    load_ppo_from_hf,
)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _random_seed():
    return int(np.random.randint(0, 1_000_000))


# ---------------------------------------------------------------------------
# Helpers for mapping between flat Gradio args and typed configs
# ---------------------------------------------------------------------------


def _flat_to_cameras(flat):
    """(enabled, fwd, rgt, height, yaw, fov) × N  →  list[CameraConfig]."""
    if len(flat) != NUM_CAMERAS * 6:
        raise ValueError(f"expected {NUM_CAMERAS * 6} camera values, got {len(flat)}")

    out = []
    for i in range(NUM_CAMERAS):
        b = i * 6
        out.append(
            CameraConfig(
                idx=i,
                name=DEFAULT_CAMERA_SPECS[i][0],
                enabled=bool(flat[b + 0]),
                fwd=float(flat[b + 1]),
                rgt=float(flat[b + 2]),
                height=float(flat[b + 3]),
                yaw=float(flat[b + 4]),
                fov=float(flat[b + 5]),
            )
        )
    return out


def _selected_idx(label):
    """'Camera 2' → 1, clamped to [0, NUM_CAMERAS)."""
    try:
        i = int(str(label).split()[-1]) - 1
    except Exception:
        i = 0
    return max(0, min(NUM_CAMERAS - 1, i))


def _build_configs(
    seed, grid, obsth, roadw, noisec, shoulder, n_segs, noise_amp, render_kw=None
):
    """Build World/Trajectory/Render configs from raw UI values."""
    world_cfg = WorldConfig(
        seed=int(seed),
        grid_size=int(grid),
        max_obstacle_height=int(obsth),
        road_width=int(roadw),
        noise_scale=float(noisec),
        shoulder_extra=int(shoulder),
    )

    traj_cfg = TrajectoryConfig(
        seed=int(seed) + 1000,
        n_segments=int(n_segs),
        noise_amplitude=float(noise_amp),
    )

    render_cfg = RenderConfig(**(render_kw or {}))
    return world_cfg, traj_cfg, render_cfg


def _traj_info_text(seg_types, trajectory):
    diffs = np.diff(trajectory, axis=0)
    length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return (
        f"waypoints: {len(trajectory)}   "
        f"length ≈ {length:.1f} vx   "
        f"segments: {' → '.join(seg_types)}"
    )


# ---------------------------------------------------------------------------
# Original random-generator callbacks
# ---------------------------------------------------------------------------


def update_preview(
    seed, grid, obsth, roadw, noisec, shoulder, n_segs, noise_amp, *cam_and_selected
):
    """Render the BEV at the starting pose with the current camera frustums."""
    sel_label = cam_and_selected[-1]
    flat = cam_and_selected[:-1]
    cams = _flat_to_cameras(flat)
    sel_idx = _selected_idx(sel_label)

    world_cfg, traj_cfg, _ = _build_configs(
        seed,
        grid,
        obsth,
        roadw,
        noisec,
        shoulder,
        n_segs,
        noise_amp,
    )
    voxels, traj, seg_types = get_world_and_trajectory(world_cfg, traj_cfg)

    idx0 = 0
    h = compute_heading(traj, idx0)

    include = {c.idx for c in cams if c.enabled}
    include.add(sel_idx)
    overlays = build_camera_overlays(cams, traj[idx0], h, include_idxs=include)

    bev = render_bev(voxels, traj, idx0, h, camera_overlays=overlays, display_size=420)
    return bev, _traj_info_text(seg_types, traj)


def generate_video(
    seed,
    grid,
    obsth,
    roadw,
    noisec,
    shoulder,
    n_segs,
    noise_amp,
    num_frames,
    img_w,
    img_h,
    fps,
    *cam_and_selected,
):
    """Produce ONE mp4: BEV + selected camera view, side-by-side per frame."""
    sel_label = cam_and_selected[-1]
    flat = cam_and_selected[:-1]
    cams = _flat_to_cameras(flat)
    sel_idx = _selected_idx(sel_label)
    sel_cam = cams[sel_idx]

    world_cfg, traj_cfg, _ = _build_configs(
        seed,
        grid,
        obsth,
        roadw,
        noisec,
        shoulder,
        n_segs,
        noise_amp,
    )
    voxels, traj, _seg_types = get_world_and_trajectory(world_cfg, traj_cfg)
    VX, VY, _VZ = voxels.shape

    nf = int(num_frames)
    W = int(img_w)
    H = int(img_h)
    f = int(fps)
    t_far = max(25.0, 0.85 * max(VX, VY))
    idxs = np.linspace(0, len(traj) - 1, nf).astype(int)

    include = {c.idx for c in cams if c.enabled}
    include.add(sel_idx)

    bev_label = "Bird's-Eye View"
    cam_label = (
        f"Camera {sel_idx + 1} ({sel_cam.name})   "
        f"FOV {sel_cam.fov:.0f}°   yaw {sel_cam.yaw:+.0f}°"
    )

    frames = []
    for idx in idxs:
        wp = traj[idx]
        h = compute_heading(traj, idx)
        pos, _hd, R = compute_camera_world_pose(wp, h, sel_cam)

        cam_img = render_camera_view(
            voxels,
            pos,
            R,
            W=W,
            H=H,
            fov_h_deg=sel_cam.fov,
            t_near=0.2,
            t_far=t_far,
            n_samples=200,
        )

        overlays = build_camera_overlays(cams, wp, h, include_idxs=include)
        bev_img = render_bev(
            voxels, traj, idx, h, camera_overlays=overlays, display_size=420
        )

        frames.append(
            compose_side_by_side(bev_img, cam_img, bev_label, cam_label, target_h=360)
        )

    tmp = tempfile.mkdtemp(prefix="voxel_demo_")
    out = os.path.join(tmp, "drive.mp4")
    imageio.mimsave(out, frames, fps=f, macro_block_size=1)
    return out


def save_dataset_sample(
    out_root_text,
    seed,
    grid,
    obsth,
    roadw,
    noisec,
    shoulder,
    n_segs,
    noise_amp,
    num_frames,
    img_w,
    img_h,
    fps,
    *cam_and_selected,
):
    """Write a single full dataset sample to disk and return a status string."""
    _sel = cam_and_selected[-1]
    flat = cam_and_selected[:-1]
    cams = _flat_to_cameras(flat)

    world_cfg, traj_cfg, _ = _build_configs(
        seed,
        grid,
        obsth,
        roadw,
        noisec,
        shoulder,
        n_segs,
        noise_amp,
    )

    render_cfg = RenderConfig(
        img_w=int(img_w),
        img_h=int(img_h),
        num_frames=int(num_frames),
        fps=int(fps),
    )

    root = Path(str(out_root_text).strip() or "./dataset")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"sample_{stamp}_seed{int(seed)}"

    meta = generate_sample(
        run_dir,
        world_cfg,
        traj_cfg,
        cams,
        render_cfg,
        run_id=run_dir.name,
    )

    summary = {
        "path": str(run_dir.resolve()),
        "num_waypoints": meta["trajectory_info"]["num_waypoints"],
        "segment_types": meta["trajectory_info"]["segment_types"],
        "files": meta["files"],
    }
    return f"✔ saved\n{json.dumps(summary, indent=2)}"


# ---------------------------------------------------------------------------
# Closed-loop model callbacks
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _cached_load_occ_model(
    repo_id: str,
    occ_hf_file: str,
):
    repo_id = str(repo_id).strip() or DEFAULT_HF_REPO
    if not repo_id:
        raise ValueError(
            "HF repo id is empty. Fill the repo textbox or set VOXEL_CAR_HF_REPO."
        )

    return load_occnet_from_hf(
        repo_id=repo_id,
        filename=str(occ_hf_file).strip() or DEFAULT_OCC_FILENAME,
    )


@lru_cache(maxsize=4)
def _cached_load_ppo_model(
    repo_id: str,
    rl_hf_file: str,
):
    repo_id = str(repo_id).strip() or DEFAULT_HF_REPO
    if not repo_id:
        raise ValueError(
            "HF repo id is empty. Fill the repo textbox or set VOXEL_CAR_HF_REPO."
        )

    return load_ppo_from_hf(
        repo_id=repo_id,
        filename=str(rl_hf_file).strip() or DEFAULT_RL_FILENAME,
        device="cpu",
    )


def run_closed_loop_model_demo(
    repo_id,
    occ_hf_file,
    rl_hf_file,
    planner_name,
    preset,
    seed,
    max_steps,
    fps,
    render_video,
):
    """Run one closed-loop scenario with either A* or RL policy."""
    try:
        repo_id = str(repo_id).strip() or DEFAULT_HF_REPO

        model, ego_cfg, cam_cfg, image_hw, device = _cached_load_occ_model(
            repo_id,
            str(occ_hf_file),
        )

        H, W = image_hw
        seed = int(seed)
        preset = str(preset)

        policy = None
        controller = "astar"

        if str(planner_name).startswith("PPO"):
            ppo = _cached_load_ppo_model(
                repo_id,
                str(rl_hf_file),
            )
            policy = RLPolicyAdapter(
                model=ppo,
                step_size=1.0,
                max_turn_deg=15.0,
                min_speed_fraction=0.10,
                deterministic=True,
            )
            controller = "ppo_rl"

        scenario = generate_scenario(
            seed=seed,
            preset=preset,
            name=f"{preset}_seed{seed}_{controller}",
        )

        fov_mask = compute_fov_mask(
            ego_cfg,
            cam_cfg,
            image_w=W,
            image_h=H,
            margin_deg=3.0,
        )

        episode = simulate_episode(
            scenario,
            model,
            ego_cfg,
            cam_cfg,
            (H, W),
            device,
            fov_mask=fov_mask,
            policy=policy,
            margin_deg=3.0,
            max_steps=int(max_steps),
            step_size=1.0,
            max_turn_deg=15.0,
            lookahead_cells=3,
            inflate=2,
            close_range_cells=3,
            treat_unknown_close_as_obstacle=True,
            soft_cost_weight=4.0,
            heading_penalty=0.3,
            forward_bias=0.7,
            use_world_bounds=True,
            world_margin=2.0,
            goal_slow_radius=10.0,
            goal_slow_min_fraction=0.25,
            verbose=False,
        )

        out_dir = Path(tempfile.mkdtemp(prefix="voxel_closed_loop_"))

        final_png = out_dir / f"{scenario.name}_final.png"
        save_summary_figure(episode, scenario, cam_cfg, ego_cfg, final_png)

        video_path = None
        if bool(render_video):
            video_path = out_dir / f"{scenario.name}.mp4"
            save_episode_video(
                episode,
                scenario,
                cam_cfg,
                ego_cfg,
                video_path,
                fps=int(fps),
            )

        summary = episode_summary(episode)
        summary.update(
            {
                "controller": controller,
                "planner": str(planner_name),
                "repo_id": repo_id,
                "occ_hf_file": str(occ_hf_file) or DEFAULT_OCC_FILENAME,
                "rl_hf_file": (
                    str(rl_hf_file) or DEFAULT_RL_FILENAME
                    if controller == "ppo_rl"
                    else None
                ),
                "preset": preset,
                "seed": seed,
                "image_shape": [H, W],
                "ego_cfg": ego_cfg.to_dict(),
                "camera": cam_cfg.to_dict(),
                "entry_xy": [float(scenario.entry_xy[0]), float(scenario.entry_xy[1])],
                "exit_xy": [float(scenario.exit_xy[0]), float(scenario.exit_xy[1])],
                "seg_types": list(scenario.seg_types),
                "outputs_dir": str(out_dir),
            }
        )

        return (
            str(video_path) if video_path is not None else None,
            str(final_png) if final_png.exists() else None,
            summary,
        )

    except Exception as exc:
        raise gr.Error(str(exc)) from exc


# ---------------------------------------------------------------------------
# Blocks UI
# ---------------------------------------------------------------------------


def _color_dot_html(rgb):
    return (
        f"<span style='display:inline-block;width:10px;height:10px;"
        f"border-radius:50%;background:rgb{rgb};margin-right:6px;"
        f"vertical-align:middle;'></span>"
    )


def build_demo():
    with gr.Blocks(title="Voxel Car Demo") as demo:
        gr.Markdown(
            "# Voxel Car Demo\n"
            "Random procedural generation plus closed-loop model evaluation. "
            "Closed-loop mode runs OccNet occupancy prediction with either "
            "standard A* planning or a pretrained PPO-RL controller.\n\n"
            f"**Default model repo:** `{DEFAULT_HF_REPO}`"
        )

        with gr.Tabs():
            # -----------------------------------------------------------------
            # TAB 1: original random generator
            # -----------------------------------------------------------------
            with gr.Tab("Random generator"):
                gr.Markdown(
                    "Generate a procedural voxel road scene, preview camera "
                    "frustums, render a side-by-side video, or save a dataset sample."
                )

                with gr.Row():
                    # ----------------- LEFT: controls -----------------
                    with gr.Column(scale=1, min_width=380):

                        with gr.Group():
                            gr.Markdown("### World")
                            with gr.Row():
                                seed = gr.Slider(
                                    0,
                                    1_000_000,
                                    value=42,
                                    step=1,
                                    label="Seed",
                                )
                                rand_seed_btn = gr.Button("Random seed")

                            grid = gr.Slider(
                                40,
                                140,
                                value=80,
                                step=10,
                                label="Grid size (voxels / side)",
                            )
                            obsth = gr.Slider(
                                5,
                                22,
                                value=14,
                                step=1,
                                label="Max obstacle height (voxels)",
                            )
                            roadw = gr.Slider(
                                2,
                                6,
                                value=3,
                                step=1,
                                label="Road half-width (voxels)",
                            )
                            noisec = gr.Slider(
                                8,
                                40,
                                value=18,
                                step=2,
                                label="Noise scale",
                            )
                            shoulder = gr.Slider(
                                0,
                                8,
                                value=2,
                                step=1,
                                label="Shoulder extra radius (voxels)",
                            )

                        with gr.Group():
                            gr.Markdown("### Trajectory")
                            n_segs = gr.Slider(
                                2,
                                12,
                                value=6,
                                step=1,
                                label="Number of segments",
                            )
                            noise_amp = gr.Slider(
                                0.0,
                                1.5,
                                value=0.4,
                                step=0.05,
                                label="Per-waypoint noise amplitude",
                            )
                            traj_info = gr.Markdown("_(will populate after preview)_")

                        with gr.Group():
                            gr.Markdown("### Drive / render")
                            nfr = gr.Slider(
                                10,
                                120,
                                value=40,
                                step=5,
                                label="Number of frames",
                            )
                            imgw = gr.Slider(
                                96,
                                320,
                                value=180,
                                step=20,
                                label="Camera image width (px)",
                            )
                            imgh = gr.Slider(
                                72,
                                240,
                                value=136,
                                step=4,
                                label="Camera image height (px)",
                            )
                            fps = gr.Slider(
                                5,
                                24,
                                value=10,
                                step=1,
                                label="Output FPS",
                            )

                        gr.Markdown(
                            "### Cameras\n"
                            "Offsets are in metres from car centre. "
                            "**+right** = passenger-side. "
                            "**+yaw** = rotate camera toward car's right."
                        )

                        cam_controls = []
                        for i in range(NUM_CAMERAS):
                            name, en0, fwd0, rgt0, h0, yaw0, fov0 = (
                                DEFAULT_CAMERA_SPECS[i]
                            )
                            dot = _color_dot_html(CAMERA_COLORS[i])
                            with gr.Accordion(
                                f"Camera {i + 1} — {name}", open=(i == 0)
                            ):
                                gr.Markdown(f"{dot} Frustum colour on BEV")
                                en = gr.Checkbox(
                                    value=en0,
                                    label="Draw this camera's frustum / save its video",
                                )
                                fwd = gr.Slider(
                                    -5,
                                    5,
                                    value=fwd0,
                                    step=0.1,
                                    label="Forward offset (m)",
                                )
                                rgt = gr.Slider(
                                    -3,
                                    3,
                                    value=rgt0,
                                    step=0.1,
                                    label="Right offset (m)",
                                )
                                h = gr.Slider(
                                    0.3,
                                    6.0,
                                    value=h0,
                                    step=0.1,
                                    label="Height (m)",
                                )
                                yaw = gr.Slider(
                                    -180,
                                    180,
                                    value=yaw0,
                                    step=1,
                                    label="Yaw vs car forward (deg)",
                                )
                                fov = gr.Slider(
                                    30,
                                    140,
                                    value=fov0,
                                    step=1,
                                    label="Horizontal FOV (deg)",
                                )
                                cam_controls.extend([en, fwd, rgt, h, yaw, fov])

                        selected = gr.Dropdown(
                            choices=[f"Camera {i + 1}" for i in range(NUM_CAMERAS)],
                            value="Camera 1",
                            label="Camera shown in composed video",
                        )

                        with gr.Row():
                            preview_btn = gr.Button("Refresh preview")
                            gen_btn = gr.Button("Generate video", variant="primary")

                        gr.Markdown("### Dataset save")
                        out_root_text = gr.Textbox(
                            value="./dataset",
                            label="Output root directory",
                        )
                        save_btn = gr.Button(
                            "Save full dataset sample",
                            variant="secondary",
                        )
                        save_status = gr.Textbox(
                            label="Save status",
                            lines=8,
                            interactive=False,
                            value=(
                                "Click Save to write voxels + videos + metadata "
                                "for the current configuration."
                            ),
                        )

                    # ----------------- RIGHT: outputs -----------------
                    with gr.Column(scale=2):
                        preview_img = gr.Image(
                            label="Layout preview — BEV at start pose",
                            height=440,
                            type="numpy",
                            show_label=True,
                        )
                        video_out = gr.Video(label="Aligned BEV + selected camera view")

                preview_inputs = [
                    seed,
                    grid,
                    obsth,
                    roadw,
                    noisec,
                    shoulder,
                    n_segs,
                    noise_amp,
                    *cam_controls,
                    selected,
                ]

                for comp in preview_inputs:
                    if isinstance(comp, gr.Slider):
                        comp.release(
                            update_preview,
                            inputs=preview_inputs,
                            outputs=[preview_img, traj_info],
                        )
                    elif isinstance(comp, (gr.Checkbox, gr.Dropdown)):
                        comp.change(
                            update_preview,
                            inputs=preview_inputs,
                            outputs=[preview_img, traj_info],
                        )

                rand_seed_btn.click(_random_seed, outputs=seed)

                preview_btn.click(
                    update_preview,
                    inputs=preview_inputs,
                    outputs=[preview_img, traj_info],
                )

                gen_inputs = [
                    seed,
                    grid,
                    obsth,
                    roadw,
                    noisec,
                    shoulder,
                    n_segs,
                    noise_amp,
                    nfr,
                    imgw,
                    imgh,
                    fps,
                    *cam_controls,
                    selected,
                ]
                gen_btn.click(generate_video, inputs=gen_inputs, outputs=video_out)

                save_inputs = [
                    out_root_text,
                    seed,
                    grid,
                    obsth,
                    roadw,
                    noisec,
                    shoulder,
                    n_segs,
                    noise_amp,
                    nfr,
                    imgw,
                    imgh,
                    fps,
                    *cam_controls,
                    selected,
                ]
                save_btn.click(
                    save_dataset_sample,
                    inputs=save_inputs,
                    outputs=save_status,
                )

                demo.load(
                    update_preview,
                    inputs=preview_inputs,
                    outputs=[preview_img, traj_info],
                )

            # -----------------------------------------------------------------
            # TAB 2: closed-loop model demo
            # -----------------------------------------------------------------
            with gr.Tab("Closed-loop model demo"):
                gr.Markdown(
                    "Run a closed-loop scenario using the trained occupancy model. "
                    "Choose either the standard A* planner or the pretrained PPO-RL planner. "
                    f"Models are loaded from Hugging Face Hub, using `{DEFAULT_HF_REPO}` by default."
                )

                with gr.Row():
                    with gr.Column(scale=1, min_width=420):
                        with gr.Group():
                            gr.Markdown("### Hugging Face model repo")
                            repo_id = gr.Textbox(
                                value=DEFAULT_HF_REPO,
                                label="HF repo id",
                                placeholder="mmkuznecov/SynthOccPredModels",
                            )
                            occ_hf_file = gr.Textbox(
                                value=DEFAULT_OCC_FILENAME,
                                label="HF OccNet checkpoint filename",
                            )
                            rl_hf_file = gr.Textbox(
                                value=DEFAULT_RL_FILENAME,
                                label="HF PPO policy filename",
                            )

                        with gr.Group():
                            gr.Markdown("### Scenario")
                            planner_name = gr.Dropdown(
                                choices=[
                                    "A* planner over OccNet prediction",
                                    "PPO-RL planner over OccNet prediction",
                                ],
                                value="A* planner over OccNet prediction",
                                label="Controller",
                            )
                            preset = gr.Dropdown(
                                choices=list(SCENARIO_PRESETS.keys()),
                                value="winding",
                                label="Scenario preset",
                            )

                            with gr.Row():
                                closed_seed = gr.Number(
                                    value=777,
                                    precision=0,
                                    label="Seed",
                                )
                                closed_rand_seed_btn = gr.Button("Random seed")

                            max_steps = gr.Slider(
                                20,
                                200,
                                value=100,
                                step=1,
                                label="Max simulation steps",
                            )
                            closed_fps = gr.Slider(
                                1,
                                12,
                                value=4,
                                step=1,
                                label="Video FPS",
                            )
                            render_video = gr.Checkbox(
                                value=True,
                                label="Render MP4 video",
                            )

                            run_btn = gr.Button(
                                "Run closed-loop scenario",
                                variant="primary",
                            )

                    with gr.Column(scale=2):
                        closed_video = gr.Video(label="Closed-loop rollout")
                        final_img = gr.Image(
                            label="Final summary snapshot",
                            type="filepath",
                        )
                        run_summary = gr.JSON(label="Run summary")

                closed_rand_seed_btn.click(_random_seed, outputs=closed_seed)

                run_btn.click(
                    run_closed_loop_model_demo,
                    inputs=[
                        repo_id,
                        occ_hf_file,
                        rl_hf_file,
                        planner_name,
                        preset,
                        closed_seed,
                        max_steps,
                        closed_fps,
                        render_video,
                    ],
                    outputs=[closed_video, final_img, run_summary],
                )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
