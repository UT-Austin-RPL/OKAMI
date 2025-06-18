import numpy as np
import torch
import os

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import mujoco
import mujoco.viewer

def gripper_joint_pos_controller_xml(obs, desired_qpos):
    
    limits = [
        [0, 1.3],
        [0, 0.68],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
        [0, 1.3],
        [0, 0.68],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
    ]

    action = np.zeros(12)
    for i in range(12):
        # map desired_qpos to [-1, 1]
        action[i] = 2 * (desired_qpos[i] - limits[i][0]) / (limits[i][1] - limits[i][0]) - 1
    
    return action

def joint_pos_controller(obs, target_joint_pos):
    '''
    Joint position controller for GR1FloatingBody.
    Args:
        obs (dict): observation from the environment, where obs["robot0_joint_pos"] is the current joint positions of upperbody (without hands, 20-dim).
        target_joint_pos (np.array): target joint positions for the robot (35-dim).
    Returns:
        action (np.array): action to be taken by the robot (35-dim).
    '''

    action = np.zeros(35)

    # for right arm and left arm joints
    action[0:14] = np.clip(5 * (target_joint_pos[0:14] - obs["robot0_joint_pos"][6:20]), -3, 3)

    # for right gripper and left gripper
    action[14:26] = gripper_joint_pos_controller_xml(obs, target_joint_pos[14:26])

    # for base joints
    # TODO

    # for the head joints
    action[29:32] = (target_joint_pos[29:32] - obs["robot0_joint_pos"][3:6])
    
    # for the waist joints
    action[32:35] = np.clip(5 * (target_joint_pos[32:35] - obs["robot0_joint_pos"][0:3]), -3, 3)
    return action

def urdf_to_robosuite_cmds(urdf_q):
    """
    Generate 35-dim robosuite joint position cmds from 56-dim urdf joint position commmands.
    """
    waist_joints = urdf_q[:3]
    head_joints = urdf_q[3:6]
    left_arm_joints = urdf_q[6:13]
    left_gripper_joints = urdf_q[13:25]
    right_arm_joints = urdf_q[25:32]
    right_gripper_joints = urdf_q[32:44]

    actuator_idxs = [0, 1, 8, 10, 4, 6]
    right_gripper_actuator_joints = right_gripper_joints[actuator_idxs]
    left_gripper_actuator_joints = left_gripper_joints[actuator_idxs]

    action = np.zeros(35)
    action[:7] = right_arm_joints
    action[7:14] = left_arm_joints
    action[14:20] = right_gripper_actuator_joints
    action[20:26] = left_gripper_actuator_joints
    action[29:32] = head_joints
    action[32:35] = waist_joints

    return action.copy()

def obs_to_urdf(obs):
    """
    Convert joint positions in obs into URDF joints format.
    """
    right_arm_joints = obs["robot0_joint_pos"][6:13].copy()
    left_arm_joints = obs["robot0_joint_pos"][13:20].copy()

    right_gripper_joints = dex_mapping(obs["robot0_right_gripper_qpos"])
    left_gripper_joints = dex_mapping(obs["robot0_left_gripper_qpos"])

    head_joints = obs["robot0_joint_pos"][3:6].copy()
    waist_joints = obs["robot0_joint_pos"][0:3].copy()

    q = np.zeros(56)
    q[:3] = waist_joints
    q[3:6] = head_joints
    q[6:13] = left_arm_joints
    q[13:25] = left_gripper_joints
    q[25:32] = right_arm_joints
    q[32:44] = right_gripper_joints

    return q.copy()

def dex_mapping(q):
    '''
    Map the joint positions of dexterous hands from robosuite observation to urdf order.
    Args:
        q (np.array): joint positions of dexterous hands from robosuite observation (12-dim).
    Returns:
        q_urdf (np.array): joint positions of dexterous hands in urdf order (12-dim).
    '''
    q_urdf = np.zeros(12)

    thumb_joints = q[:4]
    index_joints = q[4:6]
    middle_joints = q[6:8]
    ring_joints = q[8:10]
    pinky_joints = q[10:]

    q_urdf[:4] = thumb_joints
    q_urdf[8:10] = index_joints
    q_urdf[10:12] = middle_joints
    q_urdf[4:6] = ring_joints
    q_urdf[6:8] = pinky_joints

    return q_urdf