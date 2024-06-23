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

def calculate_hand_offset_relative_to_object_in_world_frame(lr, direction):
    
    if direction == 'up':
        if lr == 'L':
            return np.array([-0.02, 0.0, -0.02])
        else:
            return np.array([-0.02, -0.05, 0.04]) #np.array([-0.02, -0.05, 0.06])
    
    if lr == 'L':
        # return np.array([-0.13, -0.01, 0]) # object to wrist
        return np.array([-0.03, -0.0, -0.03]) # object to palm
    else:
        # return np.array([-0.13, 0.01, 0]) # object to wrist
        return np.array([-0.03, 0.01, 0.0]) # object to palm

def calculate_wrist_offset_relative_to_object_in_head_frame(lr, gr1_transform, direction, wrist_transformation=None):
    
    offset_to_object_in_world = calculate_hand_offset_relative_to_object_in_world_frame(lr, direction)

    if direction == 'side':
        # object_to_wrist_in_world = offset_to_object_in_world
        # object_to_wrist_in_head = gr1_transform.apply_transform_to_point(object_to_wrist_in_world, 'world', 'head')
    
        object_to_palm_in_world = offset_to_object_in_world.copy()
        object_to_palm_in_head = gr1_transform.apply_transform_to_point(object_to_palm_in_world, 'world', 'head')

        if lr == 'R':
            delta_rot = R.from_euler('xyz', [20, 0, 0], degrees=True).as_matrix()
        else:
            delta_rot = R.from_euler('xyz', [-20, 0, 0], degrees=True).as_matrix()
        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = delta_rot
        wrist_transformation = wrist_transformation @ delta_transformation

        palm_to_wrist_in_chest = wrist_transformation[:3, :3] @ np.array([0.01, 0.12, 0.02 * (1 if lr == 'L' else -1)])

        palm_to_wrist_in_head = gr1_transform.apply_transform_to_point(palm_to_wrist_in_chest, 'chest', 'head')
        print("palm_to_wrist_in_head=", palm_to_wrist_in_head, "palm_to_wrist_in_chest=", palm_to_wrist_in_chest)

        object_to_wrist_in_head = object_to_palm_in_head + palm_to_wrist_in_head

    else:

        object_to_palm_in_world = offset_to_object_in_world.copy()
        object_to_palm_in_head = gr1_transform.apply_transform_to_point(object_to_palm_in_world, 'world', 'head')

        # gr1 = GR1URDFModel()
        # palm_to_wrist_in_chest = gr1.get_joint_pose(f'link_{lr}Arm7', q)[:3, 3] - gr1.get_joint_pose(f'link_{lr}Palm', q)[:3, 3]
        delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = delta_rot
        wrist_transformation = wrist_transformation @ delta_transformation

        palm_to_wrist_in_chest = wrist_transformation[:3, :3] @ np.array([0.01, 0.12, 0.02 * (1 if lr == 'L' else -1)])

        palm_to_wrist_in_head = gr1_transform.apply_transform_to_point(palm_to_wrist_in_chest, 'chest', 'head')
        print("palm_to_wrist_in_head=", palm_to_wrist_in_head, "palm_to_wrist_in_chest=", palm_to_wrist_in_chest)

        object_to_wrist_in_head = object_to_palm_in_head + palm_to_wrist_in_head

    return object_to_wrist_in_head

def obtain_object_target_in_head_frame(rgbd_pc, direction, T_world_head):
    '''
    if grasp from side, return center of object point clouds.
    if grasp from up, return upper 95 percentile of object point clouds.
    '''

    rgbd_pc.transform(T_world_head)
    pcd_points = rgbd_pc.get_points()

    object_center_in_world = np.mean(pcd_points, axis=0)
    print("object_center_in_world=", object_center_in_world)
    if direction == 'up':
        # thres = 0.05
        # center_surface_points = []
        # for pt in pcd_points:
        #     if np.linalg.norm(pt[:2] - object_center_in_world[:2]) < thres:
        #         center_surface_points.append(pt)
        # center_surface_points = np.array(center_surface_points)
        # object_center_in_world[2] = np.percentile(center_surface_points[:, 2], 50)
        object_center_in_world[2] = np.percentile(pcd_points[:, 2], 50)

        # max world height
        high_point_thres = 0.18
        max_world_height = np.percentile(pcd_points[:, 2], 95)
        min_world_height = np.percentile(pcd_points[:, 2], 5)
        print("max_world_height=", max_world_height, "min_world_height=", min_world_height, "diff=", max_world_height - min_world_height)
        if max_world_height - min_world_height > high_point_thres:
            print("This is a high object! take special care!")
            high_pcd = []
            for pt in pcd_points:
                if pt[2] - min_world_height > high_point_thres:
                    high_pcd.append(pt)
            object_center_in_world = np.mean(high_pcd, axis=0)
            object_center_in_world[2] = high_point_thres + min_world_height #+ 0.02

    object_center_in_head = (np.linalg.inv(T_world_head) @ np.append(object_center_in_world, 1))[:3]

    return object_center_in_head