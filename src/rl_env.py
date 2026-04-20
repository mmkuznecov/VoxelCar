"""Gymnasium environment for RL control in the voxel-car world.

Training environment:
    - uses ground-truth ego occupancy from the scenario voxels
    - does not render camera images
    - does not run OccNet
    - is therefore much faster than full closed-loop visual simulation

Deployment / full evaluation:
    - use run_closed_loop_rl.py
    - that path uses OccNet-predicted occupancy and this same policy interface
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "RL support requires gymnasium. Install with: "
        "pip install gymnasium stable-baselines3 tensorboard"
    ) from e

from .scenarios import generate_scenario, SCENARIO_PRESETS
from .train_utils.ego import EgoGridConfig, sample_world_voxels_to_ego
from .planning import check_collision
from .rl_policy import (
    rl_observation_dim,
    build_rl_observation,
    action_to_planner_result,
)


class VoxelCarRLEnv(gym.Env):
    """Continuous-control RL environment for the procedural voxel-car task.

    Observation:
        flat BEV occupancy + ego goal vector + previous action

    Action:
        action[0] = turn command in [-1, 1]
        action[1] = speed command in [-1, 1]

    Reward:
        progress toward goal
        minus step, steering, collision/OOB penalties
        plus success bonus
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        preset: str = "easy",
        base_seed: int = 777,
        max_steps: int = 100,
        ego_cfg: Optional[EgoGridConfig] = None,
        step_size: float = 1.0,
        max_turn_deg: float = 15.0,
        min_speed_fraction: float = 0.10,
        goal_tolerance: float = 3.0,
        max_goal_dist: float = 100.0,
        progress_reward: float = 2.0,
        step_penalty: float = 0.02,
        turn_penalty: float = 0.03,
        speed_penalty: float = 0.00,
        success_bonus: float = 25.0,
        collision_penalty: float = 25.0,
        oob_penalty: float = 25.0,
        timeout_penalty: float = 3.0,
        preset_cycle: bool = False,
    ):
        super().__init__()

        if preset not in SCENARIO_PRESETS:
            raise ValueError(
                f"unknown preset '{preset}'. Valid presets: {list(SCENARIO_PRESETS)}"
            )

        self.preset = str(preset)
        self.base_seed = int(base_seed)
        self.max_steps = int(max_steps)
        self.ego_cfg = ego_cfg or EgoGridConfig(d_x=20, d_y=16, d_z=12, resolution=1.0)

        self.step_size = float(step_size)
        self.max_turn_deg = float(max_turn_deg)
        self.min_speed_fraction = float(min_speed_fraction)
        self.goal_tolerance = float(goal_tolerance)
        self.max_goal_dist = float(max_goal_dist)

        self.progress_reward = float(progress_reward)
        self.step_penalty = float(step_penalty)
        self.turn_penalty = float(turn_penalty)
        self.speed_penalty = float(speed_penalty)
        self.success_bonus = float(success_bonus)
        self.collision_penalty = float(collision_penalty)
        self.oob_penalty = float(oob_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.preset_cycle = bool(preset_cycle)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(rl_observation_dim(self.ego_cfg),),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._episode_i = 0
        self.scenario = None
        self.pos = None
        self.heading = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_dist = 0.0
        self.step_i = 0
        self.last_outcome = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is None:
            seed = self.base_seed + self._episode_i
            self._episode_i += 1

        preset = self._select_preset()

        self.scenario = generate_scenario(
            seed=int(seed),
            preset=preset,
            name=f"rl_{preset}_seed{int(seed)}",
        )

        self.pos = np.array(self.scenario.entry_xy, dtype=np.float32)
        self.heading = np.array(self.scenario.entry_heading, dtype=np.float32)
        self.heading /= np.linalg.norm(self.heading) + 1e-12

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.step_i = 0
        self.prev_dist = self._dist_to_goal()
        self.last_outcome = None

        return self._obs(), self._info(outcome="reset")

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        if action.size < 2:
            action = np.zeros(2, dtype=np.float32)

        # Current oracle ego occupancy is the state the policy acts on.
        occ = self._oracle_ego_occ()

        result = action_to_planner_result(
            action=action,
            pred_occ=occ,
            ego_cfg=self.ego_cfg,
            car_pos=tuple(self.pos),
            car_heading=tuple(self.heading),
            world_goal=self.scenario.exit_xy,
            step_size=self.step_size,
            max_turn_deg=self.max_turn_deg,
            min_speed_fraction=self.min_speed_fraction,
            status="rl_train",
        )

        self.heading = np.array(result.target_heading_xy, dtype=np.float32)
        self.heading /= np.linalg.norm(self.heading) + 1e-12
        self.pos = self.pos + float(result.step_size) * self.heading
        self.step_i += 1

        dist = self._dist_to_goal()
        progress = self.prev_dist - dist
        self.prev_dist = dist

        turn_cmd = float(np.clip(action[0], -1.0, 1.0))
        speed_cmd = float(np.clip(action[1], -1.0, 1.0))
        speed_frac = (speed_cmd + 1.0) * 0.5

        reward = 0.0
        reward += self.progress_reward * float(progress)
        reward -= self.step_penalty
        reward -= self.turn_penalty * abs(turn_cmd)
        reward -= self.speed_penalty * (1.0 - speed_frac)

        terminated = False
        truncated = False
        outcome = "running"

        VX, VY, _VZ = self.scenario.voxels.shape

        if dist <= self.goal_tolerance:
            reward += self.success_bonus
            terminated = True
            outcome = "success"

        elif not (0.0 <= self.pos[0] < VX and 0.0 <= self.pos[1] < VY):
            reward -= self.oob_penalty
            terminated = True
            outcome = "oob"

        elif check_collision(self.scenario.voxels, self.pos, self.heading):
            reward -= self.collision_penalty
            terminated = True
            outcome = "collision"

        elif self.step_i >= self.max_steps:
            reward -= self.timeout_penalty
            truncated = True
            outcome = "timeout"

        self.prev_action = action.copy()
        self.last_outcome = outcome

        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            self._info(
                outcome=outcome,
                reward=reward,
                progress=progress,
                dist_to_goal=dist,
                step_size=result.step_size,
            ),
        )

    def _select_preset(self) -> str:
        if not self.preset_cycle:
            return self.preset

        # Simple curriculum/randomization across all known presets.
        presets = list(SCENARIO_PRESETS.keys())
        idx = self._episode_i % len(presets)
        return presets[idx]

    def _oracle_ego_occ(self):
        return sample_world_voxels_to_ego(
            self.scenario.voxels,
            self.pos,
            self.heading,
            self.ego_cfg,
        )

    def _obs(self):
        occ = self._oracle_ego_occ()
        return build_rl_observation(
            pred_occ=occ,
            ego_cfg=self.ego_cfg,
            car_pos=tuple(self.pos),
            car_heading=tuple(self.heading),
            world_goal=self.scenario.exit_xy,
            prev_action=self.prev_action,
            max_goal_dist=self.max_goal_dist,
        )

    def _dist_to_goal(self):
        gx, gy = self.scenario.exit_xy
        return math.hypot(float(self.pos[0] - gx), float(self.pos[1] - gy))

    def _info(self, **extra):
        info = {
            "preset": (
                self.scenario.preset if self.scenario is not None else self.preset
            ),
            "scenario_name": self.scenario.name if self.scenario is not None else None,
            "step": int(self.step_i),
            "pos": (
                None if self.pos is None else [float(self.pos[0]), float(self.pos[1])]
            ),
            "heading": (
                None
                if self.heading is None
                else [float(self.heading[0]), float(self.heading[1])]
            ),
            "dist_to_goal": (
                float(self._dist_to_goal()) if self.scenario is not None else None
            ),
        }
        info.update(extra)
        return info
