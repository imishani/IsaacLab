from __future__ import annotations

import torch
import warnings
from typing import TYPE_CHECKING, Literal

import carb

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the physics size on all geometries of the asset.

    This function creates a set of physics size with random static friction, dynamic friction, and restitution
    values. The number of size is specified by ``num_buckets``. The size are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics size in the scene. If the number of size exceeds this
        limit, the simulation will crash.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if not isinstance(asset, (RigidObject, Articulation)):
        raise ValueError(
            f"Randomization term 'randomize_rigid_body_scale' not supported for asset: '{asset_cfg.name}'"
            f" with type: '{type(asset)}'."
        )

    if not isinstance(asset.cfg.spawn, sim_utils.CuboidCfg):
        raise ValueError(
            f"Randomization term 'randomize_rigid_body_scale' not supported for asset: '{asset_cfg.name}'"
        )
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # get the physics size of the asset
    asset.cfg.spawn.size = tuple((torch.rand(3) * torch.tensor([0.1, 0.1, 0.1]) + torch.tensor([0.1, 0.1, 0.1])).tolist())

    # update the asset in the scene
    env.scene.write_data_to_sim()

