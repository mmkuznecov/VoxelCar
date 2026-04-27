"""World-model (JEPA) subpackage for voxel_car."""

from .sigreg import SIGReg
from .model import VoxelCarJEPA, CNNEncoder, ARPredictor, ActionEncoder
from .dataset import WMDataset
from .planner import WMCEMPlanner, render_goal_image
from .adapter import WMPolicyAdapter

__all__ = [
    "SIGReg",
    "VoxelCarJEPA",
    "CNNEncoder",
    "ARPredictor",
    "ActionEncoder",
    "WMDataset",
    "WMCEMPlanner",
    "render_goal_image",
    "WMPolicyAdapter",
]
