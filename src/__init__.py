from .config import (
    NUM_CAMERAS,
    CAMERA_COLORS,
    DEFAULT_CAMERA_SPECS,
    CameraConfig,
    WorldConfig,
    TrajectoryConfig,
    RenderConfig,
    default_cameras,
)
from .noise import value_noise_2d
from .trajectory import build_trajectory
from .world import build_world, get_world_and_trajectory
from .camera import make_camera_R, compute_camera_world_pose, render_camera_view
from .bev import compute_heading, build_camera_overlays, render_bev
from .compose import compose_side_by_side
from .dataset import generate_sample, generate_dataset
from .ui import demo
from .train_utils import (
    OccupancyDataset,
    split_by_run,
    OccNet,
    EgoGridConfig,
    compute_fov_mask,
    sample_world_voxels_to_ego,
    save_sample_figure,
    plot_curves,
)

# Closed-loop planning (model-driven navigation) — new modules.
from .planning import (
    PlannerResult,
    plan_next_step,
    bev_astar,
    predict_to_bev_cost,
    world_to_ego_xy,
    ego_to_world_xy,
    check_collision,
)
from .scenarios import Scenario, generate_scenario, SCENARIO_PRESETS
from .closed_loop import (
    StepRecord,
    EpisodeRecord,
    simulate_episode,
    load_model_from_ckpt,
    predict_occupancy,
    episode_summary,
)
from .closed_loop_viz import (
    render_episode_frame,
    save_episode_video,
    save_summary_figure,
)

__all__ = [
    # config
    "NUM_CAMERAS",
    "CAMERA_COLORS",
    "DEFAULT_CAMERA_SPECS",
    "CameraConfig",
    "WorldConfig",
    "TrajectoryConfig",
    "RenderConfig",
    "default_cameras",
    # noise
    "value_noise_2d",
    # trajectory
    "build_trajectory",
    # world
    "build_world",
    "get_world_and_trajectory",
    # camera
    "make_camera_R",
    "compute_camera_world_pose",
    "render_camera_view",
    # bev
    "compute_heading",
    "build_camera_overlays",
    "render_bev",
    # compose
    "compose_side_by_side",
    # dataset
    "generate_sample",
    "generate_dataset",
    # ui
    "demo",
    # train_utils
    "OccupancyDataset",
    "split_by_run",
    "OccNet",
    "EgoGridConfig",
    "compute_fov_mask",
    "sample_world_voxels_to_ego",
    "save_sample_figure",
    "plot_curves",
    # planning
    "PlannerResult",
    "plan_next_step",
    "bev_astar",
    "predict_to_bev_cost",
    "world_to_ego_xy",
    "ego_to_world_xy",
    "check_collision",
    # scenarios
    "Scenario",
    "generate_scenario",
    "SCENARIO_PRESETS",
    # closed-loop simulation
    "StepRecord",
    "EpisodeRecord",
    "simulate_episode",
    "load_model_from_ckpt",
    "predict_occupancy",
    "episode_summary",
    # closed-loop viz
    "render_episode_frame",
    "save_episode_video",
    "save_summary_figure",
]
