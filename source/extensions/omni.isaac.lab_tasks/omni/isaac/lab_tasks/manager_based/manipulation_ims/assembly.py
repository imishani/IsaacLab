# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/03_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


# import argparse

# from omni.isaac.lab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
# parser.add_argument("--robot_type", type=str, default="franka_panda", 
#                     help="type of robot arms to use in the assembly. Currently only franka_panda is supported.")
#                     # help="type of robot arms to use in the assembly. Options: franka_panda, ur10, kinova_jaco2_n7s300, kinova_jaco2_n6s300, kinova_gen3_n7, sawyer.")
# parser.add_argument("--num_envs", type=int, default=12, help="Number of environments to spawn.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

import torch
import random

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer import OffsetCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from omni.isaac.lab.managers import (
    ActionTermCfg, 
    EventTermCfg, 
    ObservationTermCfg, 
    ObservationGroupCfg, 
    SceneEntityCfg, 
    TerminationTermCfg
)
from omni.isaac.lab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from omni.isaac.lab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
# from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass, math

from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

from omni.isaac.lab_tasks.manager_based.manipulation_ims import mdp

##
# Pre-defined configs
##

from omni.isaac.lab_assets import (
    FRANKA_PANDA_HIGH_PD_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG
)


def generate_random_object(index: int):
    """Generates a random object."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MovableObject" + str(index),
        spawn=sim_utils.CuboidCfg(
            size=tuple((torch.rand(3) * torch.tensor([0.1, 0.1, 0.1]) + torch.tensor([0.1, 0.1, 0.1])).tolist()),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=float(torch.rand(1) * 3.0),
                density=float(torch.rand(1) * 3.0),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(torch.rand(3).tolist()),
                metallic=float(torch.rand(1))
            )
        # ),
        # init_state=RigidObjectCfg.InitialStateCfg(
        #     pos=(random.uniform(0.1, 0.5), random.uniform(-0.4, 0.4), 0.05),
        #     rot=tuple(math.quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), 
        #                                        torch.tensor(random.uniform(-3.14 / 2., 3.14 / 2.))).squeeze().tolist())
        # ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, random.choice([0.2, -0.7]), 0.05)
        ),
    )


@configclass
class AssemblySceneCfg(InteractiveSceneCfg):
    """
    Configuration for the assembly scene.
    Comprises of three arms, mounted each on a table with a table in the middle.
    """

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8))
    )    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # tables
    table1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table1",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", 
                                   scale=(0.5, 2.0, 1.0)), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)), 
    )
    # Table two is rotated by 90 degrees and placed at a distance of 2.0 units from table one
    table2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table2",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", 
                                   scale=(0.5, 2.0, 1.0)), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 1.0, 0.0), 
                                                rot=tuple(math.quat_from_euler_xyz(
                                                    torch.zeros(1, dtype=torch.float),
                                                    torch.zeros(1, dtype=torch.float),
                                                    -torch.pi / 2 * torch.ones(1, dtype=torch.float)).squeeze().tolist())), 
    )
    # Table three is rotated by -90 degrees and placed at a distance of 2.0 units from table one to the other side
    table3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table3",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", 
                                   scale=(0.5, 2.0, 1.0)), 
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, -1.0, 0.0),
                                                rot=tuple(math.quat_from_euler_xyz(
                                                    torch.zeros(1, dtype=torch.float),
                                                    torch.zeros(1, dtype=torch.float),
                                                    torch.pi / 2 * torch.ones(1, dtype=torch.float)).squeeze().tolist())), 
    )

    # Table at the center
    table_center = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableCenter",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
                                   scale=(1.0, 1.5, 0.98)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.1, 0.0, -0.02),
                                                rot=tuple(math.quat_from_euler_xyz(
                                                    torch.zeros(1, dtype=torch.float),
                                                    torch.zeros(1, dtype=torch.float),
                                                    -torch.pi * torch.ones(1, dtype=torch.float)).squeeze().tolist())),
    )
    # # articulations
    # if args_cli.robot_type == "franka_panda":
    # put on top of the tables
    robot1 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    robot2 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    robot3 = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot3")
    ee_frame1 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot1/panda_link0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EE1FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot1/panda_hand",
                name="ee_tcp1",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot1/panda_leftfinger",
                name="tool_leftfinger1",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot1/panda_rightfinger",
                name="tool_rightfinger1",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    ee_frame2 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot2/panda_link0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EE2FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot2/panda_hand",
                name="ee_tcp2",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot2/panda_leftfinger",
                name="tool_leftfinger2",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot2/panda_rightfinger",
                name="tool_rightfinger2",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    ee_frame3 = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot3/panda_link0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EE3FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot3/panda_hand",
                name="ee_tcp3",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot3/panda_leftfinger",
                name="tool_leftfinger3",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot3/panda_rightfinger",
                name="tool_rightfinger3",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )


    # elif args_cli.robot_type == "ur10":
    #     robot1 = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    #     robot2 = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    #     robot3 = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot3")
    # elif args_cli.robot_type == "kinova_jaco2_n7s300":
    #     robot1 = KINOVA_JACO2_N7S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    #     robot2 = KINOVA_JACO2_N7S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    #     robot3 = KINOVA_JACO2_N7S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot3")
    # elif args_cli.robot_type == "kinova_jaco2_n6s300":
    #     robot1 = KINOVA_JACO2_N6S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    #     robot2 = KINOVA_JACO2_N6S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    #     robot3 = KINOVA_JACO2_N6S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot3")
    # elif args_cli.robot_type == "kinova_gen3_n7":
    #     robot1 = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    #     robot2 = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    #     robot3 = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot3")
    # elif args_cli.robot_type == "sawyer":
    #     robot1 = SAWYER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    #     robot2 = SAWYER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    #     robot3 = SAWYER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot3")

    # else:
    #     raise ValueError(f"Invalid robot type: {args_cli.robot_type}")
    robot1.init_state.pos = (0.0, 0.0, 0.0)
    robot2.init_state.pos = (0.7, 1.0, 0.0)
    robot2.init_state.rot = tuple(math.quat_from_euler_xyz(
        torch.zeros(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.float),
        -torch.pi / 2 * torch.ones(1, dtype=torch.float)).squeeze().tolist())
    robot3.init_state.pos = (0.7, -1.0, 0.0)
    robot3.init_state.rot = tuple(math.quat_from_euler_xyz(
        torch.zeros(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.float),
        torch.pi / 2 * torch.ones(1, dtype=torch.float)).squeeze().tolist())

    # objects
    object1 = generate_random_object(1)
    # object2 = generate_random_object(2)
    # object3 = generate_random_object(3)
    # object4 = generate_random_object(4)


@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    TODO: this is currently only for Panda robot. Need to extend for other robots.
    """
    # body1_joint_pos : mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
    #     asset_name="robot1",
    #     joint_names=["panda_joint.*"],
    #     scale=1.0,
    #     use_default_offset=True,
    # )
    # body2_joint_pos : mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
    #     asset_name="robot2",
    #     joint_names=["panda_joint.*"],
    #     scale=1.0,
    #     use_default_offset=True,
    # )
    # body3_joint_pos : mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
    #     asset_name="robot3",
    #     joint_names=["panda_joint.*"],
    #     scale=1.0,
    #     use_default_offset=True,
    # )

    # define differential IK actions
    body1_diff_ik : mdp.DifferentialInverseKinematicsActionCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot1",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    body2_diff_ik : mdp.DifferentialInverseKinematicsActionCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot2",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    body3_diff_ik : mdp.DifferentialInverseKinematicsActionCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot3",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )

    # Define actions for the fingers
    finger1_joint_pos : mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot1",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    finger2_joint_pos : mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot2",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    finger3_joint_pos : mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot3",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """
    Observation specifications for the MDP. TODO: add more observations.
    """
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """
        Observations for policy group.
        """
        joint_pos1 = ObservationTermCfg(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg("robot1")})
        joint_vel1 = ObservationTermCfg(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg("robot1")})
        actions1 = ObservationTermCfg(func=mdp.last_action)
        
        joint_pos2 = ObservationTermCfg(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg("robot2")})
        joint_vel2 = ObservationTermCfg(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg("robot2")})
        actions2 = ObservationTermCfg(func=mdp.last_action)
        
        joint_pos3 = ObservationTermCfg(func=mdp.joint_pos, params={'asset_cfg': SceneEntityCfg("robot3")})
        joint_vel3 = ObservationTermCfg(func=mdp.joint_vel, params={'asset_cfg': SceneEntityCfg("robot3")})
        actions3 = ObservationTermCfg(func=mdp.last_action)

        object1_pose = ObservationTermCfg(func=mdp.object_pose, params={'object_name': "object1"})
        # object1_size = ObservationTermCfg(func=mdp.object_size, params={'object_name': "object1"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # Startup events
    robot1_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot1", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot2_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot2", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot3_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot3", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # table1_physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("Table1"),
    #         "static_friction_range": (0.5, 0.75),
    #         "dynamic_friction_range": (0.3, 0.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    # Reset events
    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, 
                             mode="reset")

    # reset_robot1_joints = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot1"),
    #     },
    # )

    # reset_robot2_joints = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot2"),
    #     },
    # )

    # reset_robot3_joints = EventTermCfg(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-0.1, 0.1),
    #         "velocity_range": (0.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot3"),
    #     },
    # )


    # reset_object1_scale = EventTermCfg(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object1"),
    #         "x_range": (0.1, 0.2),
    #         "y_range": (0.1, 0.2),
    #         "z_range": (0.1, 0.2)
    #     },
    # )

    object1_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object1"),
            "static_friction_range": (0.5, 0.75),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_object1_position_yaw = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.1, 0.4), "y": (0.0, 0.4), "z": (0.0, 0.0), "yaw": (-3.14 / 2.0, 3.14 / 2.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object1"),
        },
    )


@configclass
class TerminatiosCfg:
    """Termination terms for the MDP."""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    object_dropping = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object1")}
    )


@configclass
class AssemblyEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the assembly environment.
    """
    # Define the scene configuration
    scene: AssemblySceneCfg = AssemblySceneCfg(num_envs=2, env_spacing=3.0)
    # Define the actions configuration
    actions: ActionsCfg = ActionsCfg()
    # Define the observations configuration
    observations: ObservationsCfg = ObservationsCfg()
    # Define the event configuration
    events: EventCfg = EventCfg()
    # # Define the termination configuration
    terminations: TerminatiosCfg = TerminatiosCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (22.0, 8.0, 15.)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz
        # general settings
        self.episode_length_s = 5.0  # 8 seconds
