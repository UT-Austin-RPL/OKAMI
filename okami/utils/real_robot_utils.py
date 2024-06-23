import os
import json
import numpy as np

from gr1_interface.gr1_control.gr1_client import gr1_interface
from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

finger_joint_idxs = [name_to_urdf_idx[j] for j in finger_joints]
finger_joint_max = np.array([name_to_limits[j][1] for j in finger_joints])
finger_joint_min = np.array([name_to_limits[j][0] for j in finger_joints])

def process_urdf_joints(joints, shoulder_offset=0):
    joints = joints * 180.0 / np.pi
    sign_array = np.ones(56)
    # body joints
    for name_idx in name_to_sign:
        sign_array[name_to_urdf_idx[name_idx]] = name_to_sign[name_idx]
    joints[7] += shoulder_offset
    joints[26] -= shoulder_offset
    joints *= sign_array
    
    # hand joints
    hand_joints = joints[finger_joint_idxs].copy()
    hand_joints_limited = np.clip(hand_joints, finger_joint_min, finger_joint_max)
    hand_joints_rel = (hand_joints_limited - finger_joint_min) / (
        finger_joint_max - finger_joint_min
    )
    hand_joints_int = (1.0 - hand_joints_rel) * 1000

    hand_joints_int = np.clip(hand_joints_int, 0, 1000).astype(int)
    # switch left right finger control
    hand_joints_int_reorderd = hand_joints_int[[5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]]

    return joints, hand_joints_int_reorderd

def real_to_urdf(joints, shoulder_offset=0):

    sign_array = np.ones(56)
    # body joints
    for name_idx in name_to_sign:
        sign_array[name_to_urdf_idx[name_idx]] = name_to_sign[name_idx]
    joints *= sign_array
    joints[7] -= shoulder_offset
    joints[26] += shoulder_offset
    joints = joints * np.pi / 180.0

    return joints

def mirror_joints(body_joints, hand_joints):
    left_joints = body_joints[6:13]
    right_joints = body_joints[25:32]
    body_joints_mirrored = body_joints.copy()
    # wrist roll no need to reverse
    left_sign = np.array([-1, -1, -1, -1, -1, 1, -1])
    right_sign = np.array([-1, -1, -1, -1, -1, 1, -1])
    body_joints_mirrored[6:13] = left_sign * right_joints
    body_joints_mirrored[25:32] = right_sign * left_joints
    hand_joints_mirrored = hand_joints[[6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]]
    return body_joints_mirrored, hand_joints_mirrored