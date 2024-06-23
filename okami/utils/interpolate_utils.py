import os
import re
import time
import json
import argparse
import pickle
import numpy as np
import time
from robot.gr1 import GR1URDFModel
from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config

from gr1_interface.gr1_control.gr1_client import gr1_interface
from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

from utils.save_state_utils import StateSaver
from utils.real_robot_utils import process_urdf_joints, mirror_joints

finger_joint_idxs = [name_to_urdf_idx[j] for j in finger_joints]
finger_joint_max = np.array([name_to_limits[j][1] for j in finger_joints])
finger_joint_min = np.array([name_to_limits[j][0] for j in finger_joints])

def H_pos_joints(lr=None):
    H_pos = np.zeros(56)
    if lr is not None:
        H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10
    # H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    H_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    H_pos[name_to_urdf_idx['l_shoulder_roll']] = -12.5
    H_pos[name_to_urdf_idx['r_shoulder_roll']] = 12.5
    H_pos[name_to_urdf_idx['l_shoulder_yaw']] = -10
    H_pos[name_to_urdf_idx['r_shoulder_yaw']] = 10
    H_pos[name_to_urdf_idx['l_wrist_yaw']] = -45
    H_pos[name_to_urdf_idx['r_wrist_yaw']] = 45   
    H_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    H_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    H_pos = H_pos / 180 * np.pi
    return H_pos

def L_pos_joints(lr=None):
    L_pos = np.zeros(56)
    if lr is not None:
        L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10
    # L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    L_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    L_pos[name_to_urdf_idx['l_shoulder_roll']] = -90
    L_pos[name_to_urdf_idx['r_shoulder_roll']] = 90
    L_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    L_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    L_pos = L_pos / 180 * np.pi
    return L_pos

def higher_H_pos_joints(lr=None):
    H_pos = np.zeros(56)
    if lr is not None:
        H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10
    # H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    H_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    H_pos[name_to_urdf_idx['l_shoulder_roll']] = -20 #-12.5
    H_pos[name_to_urdf_idx['r_shoulder_roll']] = 20 #12.5
    H_pos[name_to_urdf_idx['l_shoulder_yaw']] = -10
    H_pos[name_to_urdf_idx['r_shoulder_yaw']] = 10

    H_pos[name_to_urdf_idx['l_shoulder_pitch']] = 25
    H_pos[name_to_urdf_idx['r_shoulder_pitch']] = -25

    H_pos[name_to_urdf_idx['l_wrist_yaw']] = -45
    H_pos[name_to_urdf_idx['r_wrist_yaw']] = 45   
    H_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    H_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    H_pos = H_pos / 180 * np.pi
    return H_pos

def run_interpolation(start_pos, end_pos, gr1, steps=50, state_saver=None):
    for i in range(steps):
        start_time = time.time_ns()
        
        q = start_pos + (end_pos - start_pos) * (i / steps)
        body_joints, hand_joints = process_urdf_joints(q)
        
        gr1.control(arm_cmd=body_joints, hand_cmd=hand_joints, terminate=False)
        if state_saver is not None:
            state_saver.add_state(body_joints.copy())

        end_time = time.time_ns()
        time.sleep(
            gr1.interval - np.clip(((end_time - start_time) / (10**9)), 0, gr1.interval)
        )

def save_interpolation(start_pos, end_pos, gr1, steps=50):
    q_list = []
    for i in range(steps):
        q = start_pos + (end_pos - start_pos) * (i / steps)
        q_list.append(q)
    return q_list

def run_q_list(q_list, gr1, state_saver=None):
    for q in q_list:
        start_time = time.time_ns()

        body_joints, hand_joints = process_urdf_joints(q)
        
        gr1.control(arm_cmd=body_joints, hand_cmd=hand_joints, terminate=False)
        if state_saver is not None:
            state_saver.add_state(body_joints.copy())

        end_time = time.time_ns()
        time.sleep(
            gr1.interval - np.clip(((end_time - start_time) / (10**9)), 0, gr1.interval)
        )

def interpolate_to_start_pos(gr1, steps=50, state_saver=None, lr=None):
    init_pos = np.zeros(56)

    T_pos = np.zeros(56)
    if lr is not None:
        T_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10 
    T_pos[name_to_urdf_idx['joint_head_pitch']] = 19 
    T_pos[name_to_urdf_idx['l_shoulder_roll']] = -90 
    T_pos[name_to_urdf_idx['r_shoulder_roll']] = 90 
    T_pos = T_pos / 180 * np.pi

    L_pos = np.zeros(56)
    if lr is not None:
        L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10
    # L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    L_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    L_pos[name_to_urdf_idx['l_shoulder_roll']] = -90
    L_pos[name_to_urdf_idx['r_shoulder_roll']] = 90
    L_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    L_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    L_pos = L_pos / 180 * np.pi

    H_pos = np.zeros(56)
    if lr is not None:
        H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if lr == 'R' else 1) * 10
    # H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    H_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    H_pos[name_to_urdf_idx['l_shoulder_roll']] = -12.5
    H_pos[name_to_urdf_idx['r_shoulder_roll']] = 12.5
    H_pos[name_to_urdf_idx['l_shoulder_yaw']] = -10
    H_pos[name_to_urdf_idx['r_shoulder_yaw']] = 10
    H_pos[name_to_urdf_idx['l_wrist_yaw']] = -45
    H_pos[name_to_urdf_idx['r_wrist_yaw']] = 45   
    H_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    H_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    H_pos = H_pos / 180 * np.pi

    # Interpolate to T-pos
    run_interpolation(init_pos, T_pos, gr1, steps, state_saver)
    # Interpolate to L-pos
    run_interpolation(T_pos, L_pos, gr1, steps, state_saver)
    # Interpolate to H-pos
    # run_interpolation(L_pos, H_pos, gr1, steps, state_saver)

    # H_pos = higher_H_pos_joints(lr)
    # run_interpolation(L_pos, H_pos, gr1, steps, state_saver)

    # return H_pos
    return L_pos

def interpolate_to_end_pos(current_pos, gr1, steps=50, state_saver=None):

    L_pos = np.zeros(56)
    # L_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    L_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    L_pos[name_to_urdf_idx['l_shoulder_roll']] = -90
    L_pos[name_to_urdf_idx['r_shoulder_roll']] = 90
    L_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    L_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    L_pos = L_pos / 180 * np.pi

    H_pos = np.zeros(56)
    # H_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10
    H_pos[name_to_urdf_idx['joint_head_pitch']] = 19
    H_pos[name_to_urdf_idx['l_shoulder_roll']] = -12.5
    H_pos[name_to_urdf_idx['r_shoulder_roll']] = 12.5
    H_pos[name_to_urdf_idx['l_shoulder_yaw']] = -10
    H_pos[name_to_urdf_idx['r_shoulder_yaw']] = 10
    H_pos[name_to_urdf_idx['l_wrist_yaw']] = -45
    H_pos[name_to_urdf_idx['r_wrist_yaw']] = 45   
    H_pos[name_to_urdf_idx['l_elbow_pitch']] = 90
    H_pos[name_to_urdf_idx['r_elbow_pitch']] = -90
    H_pos = H_pos / 180 * np.pi

    T_pos = np.zeros(56)
    # T_pos[name_to_urdf_idx['joint_head_yaw']] = (-1 if args.lr == 'R' else 1) * 10 
    T_pos[name_to_urdf_idx['joint_head_pitch']] = 19 
    T_pos[name_to_urdf_idx['l_shoulder_roll']] = -90 
    T_pos[name_to_urdf_idx['r_shoulder_roll']] = 90 
    T_pos = T_pos / 180 * np.pi

    run_interpolation(current_pos, H_pos, gr1, steps, state_saver)
    run_interpolation(H_pos, L_pos, gr1, steps, state_saver)
    run_interpolation(L_pos, T_pos, gr1, steps, state_saver)

grasp_type_dict = {
    'small_diameter': [1.57079633, 0.52359878, 1.30899694, 1.30899694, 1.30899694, 1.30899694], 
    'palmar_pinch': [1.57079633, 0.52359878, 1.04719755, 0.        , 0.        , 0.        ], 
    'palm': [0., 0., 0., 0., 0., 0.], 
    'ready': [1.57079633, 0.        , 0.        , 0.        , 0.        , 0.        ],
    'release_small_diameter': [0.0, 0.0, 1.30899694, 1.30899694, 1.30899694, 1.30899694],
    'release_palmar_pinch': [0.0, 0.0, 1.04719755, 0.        , 0.        , 0.        ]
}

def append_hand_pos(current_pos, l_grasp_type, r_grasp_type):
    res = current_pos.copy()
    res[finger_joint_idxs[:6]] = grasp_type_dict[l_grasp_type]
    res[finger_joint_idxs[6:]] = grasp_type_dict[r_grasp_type]
    return res

def change_hand_pos(current_pos, l_grasp_type, r_grasp_type, gr1, steps=50, state_saver=None):
    end_pos = append_hand_pos(current_pos, l_grasp_type, r_grasp_type)
    run_interpolation(current_pos, end_pos, gr1, steps, state_saver)
    return  end_pos

def save_change_hand_pos(current_pos, l_grasp_type, r_grasp_type, gr1, steps=50):
    end_pos = append_hand_pos(current_pos, l_grasp_type, r_grasp_type)
    return save_interpolation(current_pos, end_pos, gr1, steps)