# Itamar Mishani
import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--robot_type", type=str, default="franka_panda", 
                    help="type of robot arms to use in the assembly. Currently only franka_panda is supported.")
                    # help="type of robot arms to use in the assembly. Options: franka_panda, ur10, kinova_jaco2_n7s300, kinova_jaco2_n6s300, kinova_gen3_n7, sawyer.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=36, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from omni.isaac.lab_tasks.manager_based.manipulation_ims.assembly import AssemblyEnvCfg
from typing import Sequence
import torch
import warp as wp
import omni.isaac.lab.utils.math as math

from omni.isaac.lab.envs import ManagerBasedRLEnv

# initialize warp
wp.init()


class GripperState:
    """States for the gripper state machine."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PushSmState:
    """States for the push state machine."""

    IDLE = wp.constant(0)
    MOVE_TO_PREPUSH = wp.constant(1)
    MOVE_TO_OBJECT = wp.constant(2)
    PUSH = wp.constant(3)
    DONE = wp.constant(4)


class PushSmTerminationTol:
    """Tolerances for the push state machine."""

    MOVE_TO_PREPUSH = wp.constant(0.01)
    MOVE_TO_OBJECT = wp.constant(0.01)
    PUSH = wp.constant(0.02)


class ContactState:
    """States for the contact state machine."""

    NO_CONTACT = wp.constant(0)
    CONTACT = wp.constant(1)


@wp.func
def quat_error_magnitude(q1: wp.quat,
                         q2_conj: wp.quat):
    """Compute the error between two quaternions."""
    # compute the dot product
    dot = wp.dot(q1, q2_conj)
    # dot_norm_sq = wp.length_sq(dot)
    # compute the magnitude of the error
    error = 1.0 - wp.pow(dot, 2.0)
    # error = 2.0 * wp.acos(wp.abs(dot))
    return error


@wp.func
def pose_error(pose1: wp.transform, 
               pose2: wp.transform):
    """Compute the error between two poses."""
    pos1 = wp.transform_get_translation(pose1)
    pos2 = wp.transform_get_translation(pose2)
    ori1 = wp.transform_get_rotation(pose1)
    ori2 = wp.transform_get_rotation(pose2)
    pos_diff = wp.length(pos1 - pos2)
    # print(pos_diff)
    ori_diff = quat_error_magnitude(ori1, 
                                    wp.quat_inverse(ori2))
    return pos_diff, ori_diff


@wp.func 
def increment_goal_position(ee_pose: wp.transform, 
                            ee_goal_pose: wp.transform):
    """
    Increment the goal position.
    return a new goal pose.
    """
    pos = wp.transform_get_translation(ee_pose)
    goal_pos = wp.transform_get_translation(ee_goal_pose)
    pos_diff = goal_pos - pos
    # If the norm of the difference is less than 0.1, return the goal pose
    if wp.length(pos_diff) < 0.2:
        return ee_goal_pose
    # Otherwise, move towards the goal pose by scale of 0.1
    pos = pos + 0.2 * pos_diff
    # add the orientation of the goal pose and return the new goal pose
    ori = wp.transform_get_rotation(ee_goal_pose)
    return wp.transformation(pos, ori)

@wp.kernel
def infer_state_machine(
    sm_state: wp.array(dtype=int),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    offset_push: wp.array(dtype=wp.transform),
):
    # TODO: Add CONTACT state machine
    # retrieve thread id
    tid = wp.tid()
    state = sm_state[tid]
    if state == PushSmState.IDLE:
        print("Changing State Machine to MOVE_TO_PREPUSH")
        # move to start
        sm_state[tid] = PushSmState.MOVE_TO_PREPUSH
        gripper_state[tid] = GripperState.CLOSE
        des_ee_pose[tid] = ee_pose[tid]
        # print(des_ee_pose[tid])
    elif state == PushSmState.MOVE_TO_PREPUSH:
        des_ee_pose[tid] = wp.transform_multiply(object_pose[tid], offset[tid])
        # print(des_ee_pose[tid])
        # print(object_pose[tid])
        pos_err, ori_err = pose_error(ee_pose[tid], des_ee_pose[tid])
        # move to object
        if pos_err <= PushSmTerminationTol.MOVE_TO_PREPUSH and ori_err <= PushSmTerminationTol.MOVE_TO_PREPUSH:
            # sm_state[tid] = PushSmState.MOVE_TO_OBJECT
            sm_state[tid] = PushSmState.MOVE_TO_OBJECT
            print("Changing State Machine to MOVE_TO_OBJECT")
            des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
    elif state == PushSmState.MOVE_TO_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(object_pose[tid], 
                                                 wp.transform_multiply(offset[tid], offset_push[tid]))
        # move to push
        # print(des_ee_pose[tid])
        pos_err, ori_err = pose_error(ee_pose[tid], des_ee_pose[tid])
        des_ee_pose[tid] = increment_goal_position(ee_pose[tid], des_ee_pose[tid])
        if pos_err < PushSmTerminationTol.MOVE_TO_OBJECT and ori_err < PushSmTerminationTol.MOVE_TO_OBJECT:
            sm_state[tid] = PushSmState.PUSH
            print("Changing State Machine to PUSH")
            des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
    elif state == PushSmState.PUSH:
        des_ee_pose[tid] = wp.transform_multiply(des_object_pose[tid], 
                                                 wp.transform_multiply(offset[tid], offset_push[tid]))
        # print(des_ee_pose[tid])
        pos_err, ori_err = pose_error(ee_pose[tid], des_ee_pose[tid])
        des_ee_pose[tid] = increment_goal_position(ee_pose[tid], des_ee_pose[tid])
        if pos_err <= PushSmTerminationTol.PUSH and ori_err <= PushSmTerminationTol.PUSH:
            sm_state[tid] = PushSmState.DONE
            print("Changing State Machine to DONE")
            des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
    elif state == PushSmState.DONE:
        des_ee_pose[tid] = ee_pose[tid]
        print("State Machine is DONE")
    # else:
    #     raise ValueError(f"Invalid state: {state}")
    

class MoveAndPush:
    """
    A state machine for moving to a pre-push position, moving to the object until a contact point occures, and pushing it.
    """

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        self.num_envs = num_envs
        self.device = device

        # initialize state machine
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        # offset. Default is zero and determined by the objective
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        # self.offset[:, -1] = 0.1
        self.offset_push = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset_push[:, 2] = 0.18
        self.offset_push[:, -1] = 1.0
        self.offset_wp = wp.from_torch(self.offset, dtype=wp.transform)
        self.offset_push_wp = wp.from_torch(self.offset_push, dtype=wp.transform)
        # Convert to warp tensors
        self.sm_state_wp = wp.from_torch(self.sm_state, dtype=wp.int32)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, dtype=wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, dtype=wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        print(f"Resetting State Machine for env_ids: {env_ids}")
        self.sm_state[env_ids] = 0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, 
                des_object_pose: torch.Tensor, object_size: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp tensors
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), dtype=wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), dtype=wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), dtype=wp.transform)
        
        ############## Working, but not aligned with the object ##############
        # self.offset[:, :3] = (torch.max(object_size) / 1.) * math.normalize(object_pose[:, :3] - des_object_pose[:, :3])
        # self.offset[:, :3] = math.quat_rotate(math.quat_inv(object_pose[:, [6, 3, 4, 5]]), 
        #                                       self.offset[:, :3])
        # self.offset[:, 2] += 0.2
        # self.offset = torch.cat((self.offset[:, :3], 
        #                          torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)), 
        #                         dim=1)
        # self.offset[:, :3] = math.quat_rotate(self.offset[:, [4, 5, 6, 3]], self.offset[:, :3])

        ############## Ends here ##############

        # TODO: This is not good. What happens is that everytime we move the object, the offset is updated, and thus the goal is updated. wrongly. 
        # We want to update the orientation of the ee but not changing the END pose of the ee.
        self.offset[:, :3] = (torch.max(object_size) / 1.) * math.normalize(object_pose[:, :3] - des_object_pose[:, :3])
        self.offset[:, :3] = math.quat_rotate(math.quat_inv(object_pose[:, [6, 3, 4, 5]]), self.offset[:, :3])
        self.offset[:, 2] += 0.2
                
        rotation_vec = math.quat_rotate(math.quat_inv(object_pose[:, [6, 3, 4, 5]]), des_object_pose[:, :3])
        rotation_vec = math.normalize(rotation_vec - object_pose[:, :3])
        rotation_vec = math.quat_rotate(math.quat_inv(object_pose[:, [6, 3, 4, 5]]), rotation_vec)

        yaw = torch.atan2(rotation_vec[:, 0], rotation_vec[:, 1])
        rotation_quat = math.quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        self.offset[:, 3:] = math.quat_mul(torch.tensor([0.0, 1.0, 0.0, 0.0], 
                                                        device=self.device).repeat(self.num_envs, 1), rotation_quat)
        self.offset[:, 3:] = self.offset[:, [4, 5, 6, 3]]

        self.offset_wp = wp.from_torch(self.offset.contiguous(), dtype=wp.transform)
        
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_state_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.offset_push_wp,
            ], 
            device=self.device,
        )
        # print(self.des_ee_pose)
        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # des_ee_pose[:, 3:] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    env_cfg = AssemblyEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(env_cfg)
    print(f"Env device: {env.device}")
    object_size = env.scene['object1'].cfg.spawn.size
    object_size = torch.tensor(object_size, device=env.device)
    object_size = object_size.unsqueeze(0).expand(env_cfg.scene.num_envs, -1)
    obs, info = env.reset()
    des_object_pose = obs["policy"]["object1_pose"].clone()
    des_object_pose[:, :2] = torch.tensor([0.4, 0.0], device=env.device)
    move_and_push = MoveAndPush(env_cfg.scene.num_envs, 
                                device=env.device)
    
    actions = torch.zeros_like(env.action_manager.action)
    # robot2_pose = torch.cat([env.scene['ee_frame2'].data.target_pos_w[..., 0, :].clone() - 
    #                         #  torch.tensor(env_cfg.scene.robot2.init_state.pos, device=env.device).repeat(env_cfg.scene.num_envs, 1) -
    #                          env.scene.env_origins,
    #                         env.scene['ee_frame2'].data.target_quat_w[..., 0, :].clone()], 
    #                         dim=-1)
    # robot3_pose = torch.cat([env.scene['ee_frame3'].data.target_pos_w[..., 0, :].clone() - 
    #                         # torch.tensor(env_cfg.scene.robot3.init_state.pos, device=env.device).repeat(env_cfg.scene.num_envs, 1) -
    #                         env.scene.env_origins,
    #                         env.scene['ee_frame3'].data.target_quat_w[..., 0, :].clone()],
    #                         dim=-1)
    # actions[:, 7:14] = robot2_pose
    # actions[:, 14:21] = robot3_pose
    # actions[:, 22] = 0.04
    # actions[:, 23] = 0.0
    # count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            tcp_pos = env.scene['ee_frame1'].data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
            tcp_ori = env.scene['ee_frame1'].data.target_quat_w[..., 0, :].clone()
            tcp_pose = torch.cat([tcp_pos, tcp_ori], dim=-1)
            # if count < 50:
            #     actions[:, :7] = tcp_pose
        # else:
            # if count == 50:
            #     # des_object_pose = obs["policy"]["object1_pose"] + torch.tensor([0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], 
            #     #                                                                device=env.device)
            #     des_object_pose[:, :2] = torch.tensor([0.4, 0.0], device=env.device)
            computed = move_and_push.compute(tcp_pose, obs["policy"]["object1_pose"], des_object_pose, object_size)
            actions[:, :7] = computed[:, :7]
            actions[:, 19] = computed[:, 7]
            obs, rew, terminated, truncated, info = env.step(actions)
            if truncated.any():
                move_and_push.reset_idx(truncated.nonzero(as_tuple=False).squeeze(-1))
                des_object_pose[truncated.nonzero(as_tuple=False).squeeze(-1), :] = obs["policy"]["object1_pose"][truncated.nonzero(as_tuple=False).squeeze(-1), :].clone()
                des_object_pose[truncated.nonzero(as_tuple=False).squeeze(-1), :2] = torch.tensor([0.8, 0.0], device=env.device)
                # count = 0
            # elif torch.any(move_and_push.sm_state == PushSmState.DONE):
            #     env._reset_idx((move_and_push.sm_state == PushSmState.DONE).nonzero(as_tuple=False).squeeze(-1))
            #     move_and_push.reset_idx((move_and_push.sm_state == PushSmState.DONE).nonzero(as_tuple=False).squeeze(-1))
            #     # count = 0
            # # else:
            # #     count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
