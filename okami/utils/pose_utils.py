import numpy as np
import os
import json
import time

from gr1_interface.gr1_control.gr1_client import gr1_interface
from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface

from utils.real_robot_utils import real_to_urdf
from robot.gr1 import GR1URDFModel

from scipy.spatial.transform import Rotation as R

def up_or_side_grasp(hand_pose, lr):
    # determine -x direction closer to -z or +y (right hand) / -y (left hand)
    local_x = hand_pose[:3, 0]
    local_minus_x = -local_x

    world_z = np.array([0, 0, 1])
    world_minus_z = np.array([0, 0, -1])
    world_y = np.array([0, 1, 0])
    world_minus_y = np.array([0, -1, 0])

    angle_with_general_z = np.arccos(np.dot(local_minus_x, world_minus_z) / (np.linalg.norm(local_minus_x) * np.linalg.norm(world_minus_z)))
    if lr == 'R':
        angle_with_general_y = np.arccos(np.dot(local_minus_x, world_y) / (np.linalg.norm(local_minus_x) * np.linalg.norm(world_y)))
    else:
        angle_with_general_y = np.arccos(np.dot(local_minus_x, world_minus_y) / (np.linalg.norm(local_minus_x) * np.linalg.norm(world_minus_y)))

    print("angle with z, y=", angle_with_general_z / np.pi * 180, angle_with_general_y / np.pi * 180)

    threshold_z = np.pi / 4
    threshold_y = np.pi / 3
    if angle_with_general_z < threshold_z and angle_with_general_y > threshold_y:
        return 'up'
    return 'side'

def clip_wrist_pos_in_base(wrist_pos, lr):
    wrist_pos[0] = np.clip(wrist_pos[0], 0.1, 0.4)
    wrist_pos[2] = np.clip(wrist_pos[2], -0.05, 0.4)
    if lr == 'R':
        wrist_pos[1] = np.clip(wrist_pos[1], -0.5, 0.2)
    else:
        wrist_pos[1] = np.clip(wrist_pos[1], -0.2, 0.5)
    return wrist_pos

def clip_wrist_orientation_in_base(wrist_rot, lr):
    direction = up_or_side_grasp(wrist_rot, lr)
    if direction == 'up':
        # TODO: do some clipping
        return wrist_rot
    else:
        pointing_direction = - wrist_rot[:3, 1]
        pointing_angle = np.arcsin(pointing_direction[2] / np.linalg.norm(pointing_direction))
        print("pointing_angle", pointing_angle / np.pi * 180)
        if pointing_angle < -10 / 180 * np.pi:
            if lr == 'R':
                delta_rot = R.from_euler('xyz', [40, 0, 0], degrees=True).as_matrix()
            else:
                delta_rot = R.from_euler('xyz', [-30, 0, 0], degrees=True).as_matrix()
            print("rotating 20 degree around x", delta_rot)
            wrist_rot = wrist_rot @ delta_rot

        pointing_direction = - wrist_rot[:3, 1]
        pointing_angle = np.arcsin(pointing_direction[2] / np.linalg.norm(pointing_direction))
        print("afterwards pointing_angle", pointing_angle / np.pi * 180)

        palm_direction = - wrist_rot[:3, 0]
        palm_angle = np.arcsin(- palm_direction[0] / np.linalg.norm(palm_direction))
        print("palm_angle", palm_angle / np.pi * 180)
        if palm_angle < 30 / 180 * np.pi:
            if lr == 'R':
                delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
            else:
                delta_rot = R.from_euler('xyz', [0, 0, 20], degrees=True).as_matrix()
            print("rotating 30 degree around z", delta_rot)
            wrist_rot = wrist_rot @ delta_rot
        palm_direction = - wrist_rot[:3, 0]
        palm_angle = np.arcsin(- palm_direction[0] / np.linalg.norm(palm_direction))
        print("afterwards palm_angle", palm_angle / np.pi * 180)

        return wrist_rot

def clip_wrist_pose_in_base(wrist_pose, lr):
    print("original wrist_pose", wrist_pose)
    wrist_pose[:3, 3] = clip_wrist_pos_in_base(wrist_pose[:3, 3], lr)
    wrist_pose[:3, :3] = clip_wrist_orientation_in_base(wrist_pose[:3, :3], lr)
    return wrist_pose