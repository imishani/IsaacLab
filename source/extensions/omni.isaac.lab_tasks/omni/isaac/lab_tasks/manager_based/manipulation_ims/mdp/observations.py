# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import ArticulationData, RigidObjectData
from omni.isaac.lab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def rel_ee_object_distance(env: ManagerBasedEnv, 
                           ee_name: str, 
                           object_name: str) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene[ee_name].data
    object_data: ArticulationData = env.scene[object_name].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedEnv, 
                   ee_name: str) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene[ee_name].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedEnv, 
           ee_name: str) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene[ee_name].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedEnv,
            ee_name: str, 
            make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene[ee_name].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def object_pose(env: ManagerBasedEnv, 
                object_name: str) -> torch.Tensor:
    """The pose of the object in the environment frame."""
    object_data: RigidObjectData = env.scene[object_name].data
    object_pos = object_data.root_pos_w - env.scene.env_origins
    object_quat = object_data.root_quat_w

    return torch.cat([object_pos, object_quat], dim=-1)


def object_size(env: ManagerBasedEnv,
                object_name: str) -> torch.Tensor:
    """The size of the object."""
    object_size = torch.Tensor(env.scene[object_name].cfg.spawn.size)
    object_size = object_size.unsqueeze(0).expand(env.num_envs, -1)
    return object_size
