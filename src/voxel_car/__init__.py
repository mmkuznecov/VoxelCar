"""Top-level voxel_car package.

Re-exports the most commonly-used names from the subpackages so callers can
write `from voxel_car import OccNet, simulate_episode` instead of hunting
through the subpackage hierarchy. Less-common helpers are still available
via their subpackage, e.g. `voxel_car.rl.VoxelCarRLEnv`.
"""

from .common import (
    NUM_CAMERAS,
    CAMERA_COLORS,
    DEFAULT_CAMERA_SPECS,
    CameraConfig,
    WorldConfig,
    TrajectoryConfig,
    RenderConfig,
    default_cameras,
    value_noise_2d,
    Material,
    MATERIAL_PALETTE,
    MATERIAL_TABLE,
    NUM_MATERIALS,
    FIRST_SOLID_ID,
    IS_SOLID,
    BLOCKS_RAY,
    IS_DRIVABLE,
    material_name,
)
from .worldgen import (
    build_trajectory,
    build_world,
    build_world_from_config,
    get_world_and_trajectory,
    Scenario,
    SCENARIO_PRESETS,
    generate_scenario,
    generate_heightmap,
    assign_materials,
    effective_surface_z,
    surface_material,
    carve_road,
    plant_trees,
)
from .geometry import (
    EgoGridConfig,
    compute_fov_mask,
    sample_world_voxels_to_ego,
)
from .rendering import (
    make_camera_R,
    compute_camera_world_pose,
    render_camera_view,
    compute_heading,
    build_camera_overlays,
    render_bev,
    compose_side_by_side,
)
from .datasets import (
    generate_sample,
    generate_dataset,
    OccupancyDataset,
    split_by_run,
)
from .perception import (
    OccNet,
    save_sample_figure,
    plot_curves,
)
from .planning import (
    PlannerResult,
    plan_next_step,
    bev_astar,
    predict_to_bev_cost,
    world_to_ego_xy,
    ego_to_world_xy,
    check_collision,
)
from .simulation import (
    StepRecord,
    EpisodeRecord,
    simulate_episode,
    load_model_from_ckpt,
    predict_occupancy,
    episode_summary,
    render_episode_frame,
    save_episode_video,
    save_summary_figure,
)
from .estimation import (
    LinearKalmanFilter,
    KalmanState,
    VehicleEKF,
    EKFConfig,
    PoseMeasurement,
    wrap_angle,
    noisy_pose_measurement,
)
from .ui import demo

__version__ = "0.2.0"

__all__ = [
    # common
    "NUM_CAMERAS",
    "CAMERA_COLORS",
    "DEFAULT_CAMERA_SPECS",
    "CameraConfig",
    "WorldConfig",
    "TrajectoryConfig",
    "RenderConfig",
    "default_cameras",
    "value_noise_2d",
    # materials
    "Material",
    "MATERIAL_PALETTE",
    "MATERIAL_TABLE",
    "NUM_MATERIALS",
    "FIRST_SOLID_ID",
    "IS_SOLID",
    "BLOCKS_RAY",
    "IS_DRIVABLE",
    "material_name",
    # worldgen
    "build_trajectory",
    "build_world",
    "build_world_from_config",
    "get_world_and_trajectory",
    "Scenario",
    "SCENARIO_PRESETS",
    "generate_scenario",
    "generate_heightmap",
    "assign_materials",
    "effective_surface_z",
    "surface_material",
    "carve_road",
    "plant_trees",
    # geometry
    "EgoGridConfig",
    "compute_fov_mask",
    "sample_world_voxels_to_ego",
    # rendering
    "make_camera_R",
    "compute_camera_world_pose",
    "render_camera_view",
    "compute_heading",
    "build_camera_overlays",
    "render_bev",
    "compose_side_by_side",
    # datasets
    "generate_sample",
    "generate_dataset",
    "OccupancyDataset",
    "split_by_run",
    # perception
    "OccNet",
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
    # simulation
    "StepRecord",
    "EpisodeRecord",
    "simulate_episode",
    "load_model_from_ckpt",
    "predict_occupancy",
    "episode_summary",
    "render_episode_frame",
    "save_episode_video",
    "save_summary_figure",
    # estimation
    "LinearKalmanFilter",
    "KalmanState",
    "VehicleEKF",
    "EKFConfig",
    "PoseMeasurement",
    "wrap_angle",
    "noisy_pose_measurement",
    # ui
    "demo",
]
