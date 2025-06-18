import argparse
import pickle
import time
import json
import tabulate
import os

import cv2
import numpy as np
import math

import mujoco
from mujoco import viewer

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import xml.etree.ElementTree as ET

def init_robosuite_env(args, offset, **kwargs):
    config = {}
    config["robots"] = "GR1UpperBody"
    config["env_name"] = args.environment
    config["env_configuration"] = "bimanual"
    config["controller_configs"] = load_controller_config(default_controller="JOINT_POSITION")
    config["controller_configs"]["kp"] = 500
    config["retarget_offsets"] = offset

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=["frontview", "agentview", "handview", "handview2"],
    )
    obs = env.reset()

    mjmodel = env.sim.model._model
    mjdata = env.sim.data._data

    return env, mjmodel, mjdata, obs

def launch_simulator(args, get_action_func, max_steps=100, **kwargs):
    env, mjmodel, mjdata, obs = init_robosuite_env(args, **kwargs)
    
    writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 640))

    fps = 20
    idx = 0
    with mujoco.viewer.launch_passive(
        model=mjmodel,
        data=mjdata,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:

        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
            viewer.cam.azimuth = 180
            viewer.cam.lookat = np.array([0.0, 0.0, 1.5])
            viewer.cam.distance = 1.0
            viewer.cam.elevation = -35

        while viewer.is_running():
            
            action = get_action_func(args, obs=obs, step=idx, **kwargs) 
            idx += 1

            obs, reward, done, _ = env.step(action)
            # print("idx=", idx, "hand pos=", obs["robot0_right_eef_pos"], obs["robot0_left_eef_pos"])

            # save image
            img = obs["handview2_image"]
            img = cv2.flip(img, 0)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            viewer.sync()
            # cv2.imshow("image", img)
            # cv2.waitKey(10)

            img = cv2.resize(img, (640, 640))
            writer.write(img)

            time.sleep(1 / fps)
            if idx > max_steps:
                break
        
        writer.release()

def gripper_joint_pos_controller(obs, desired_qpos, kp=100, damping_ratio=1):
    """
    Calculate the torques for the joints position controller.

    Args:
        obs: dict, the observation from the environment
        desired_qpos: np.array of shape (12, ) that describes the desired qpos (angles) of the joints on hands, right hand first, then left hand

    Returns:
        desired_torque: np.array of shape (12, ) that describes the desired torques for the joints on hands
    """
    # get the current joint position and velocity
    actuator_idxs = [0, 1, 4, 6, 8, 10]
    # order:
    # 'gripper0_right_joint_r_thumb_proximal_1', 'gripper0_right_joint_r_thumb_proximal_2', 'gripper0_right_joint_r_thumb_middle', 'gripper0_right_joint_r_thumb_distal', 'gripper0_right_joint_r_index_proximal', 'gripper0_right_joint_r_index_distal', 'gripper0_right_joint_r_middle_proximal', 'gripper0_right_joint_r_middle_distal', 'gripper0_right_joint_r_ring_proximal', 'gripper0_right_joint_r_ring_distal', 'gripper0_right_joint_r_pinky_proximal', 'gripper0_right_joint_r_pinky_distal''gripper0_right_joint_r_thumb_proximal_1', 'gripper0_right_joint_r_thumb_proximal_2', 'gripper0_right_joint_r_thumb_middle', 'gripper0_right_joint_r_thumb_distal', 'gripper0_right_joint_r_index_proximal', 'gripper0_right_joint_r_index_distal', 'gripper0_right_joint_r_middle_proximal', 'gripper0_right_joint_r_middle_distal', 'gripper0_right_joint_r_ring_proximal', 'gripper0_right_joint_r_ring_distal', 'gripper0_right_joint_r_pinky_proximal', 'gripper0_right_joint_r_pinky_distal'
    joint_qpos = np.concatenate(
        (obs["robot0_right_gripper_qpos"][actuator_idxs], obs["robot0_left_gripper_qpos"][actuator_idxs])
    )
    joint_qvel = np.concatenate(
        (obs["robot0_right_gripper_qvel"][actuator_idxs], obs["robot0_left_gripper_qvel"][actuator_idxs])
    )

    position_error = desired_qpos - joint_qpos
    vel_pos_error = -joint_qvel

    # calculate the torques: kp * position_error + kd * vel_pos_error
    kd = 2 * np.sqrt(kp) * damping_ratio - 10
    desired_torque = np.multiply(np.array(position_error), np.array(kp)) + np.multiply(vel_pos_error, kd)

    # clip and rescale to [-1, 1]
    desired_torque = np.clip(desired_torque, -1, 1)

    return desired_torque

def calculate_target_qpos(ik_joint_qpos):
    """
    Calculate the target joint positions from the results of inverse kinematics.

    Args:
        ik_joint_results: np array of shape (56, ), the results of inverse kinematics for all body joints
                        order: 3 waist + 3 head + 7 left arm + 12 left hand + 7 right arm + 12 right hand + 6 left leg + 6 right leg

    Returns:
        target_qpos: np array of shape (32, ), the target joint qpos for the robot
                    order: 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand

    """
    target_qpos = np.zeros(32)  # 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand
    target_qpos[0:6] = ik_joint_qpos[0:6]  # waist + head
    target_qpos[6:13] = ik_joint_qpos[25:32]  # right arm
    target_qpos[13:20] = ik_joint_qpos[6:13]  # left arm

    actuator_idxs = np.array([0, 1, 8, 10, 4, 6])
    target_qpos[20:26] = ik_joint_qpos[32 + actuator_idxs]  # right hand
    target_qpos[26:32] = ik_joint_qpos[13 + actuator_idxs]  # left hand
    return target_qpos

def calculate_action_from_target_joint_pos(obs, target_joints):
    '''
    Given target joint pos, calculate action for grippers and body joints.

    Args:
        obs: dict, the observation from the environment
        target_joints: np.array of shape (32, ), the target joint qpos for the robot. 
                    order: 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand

    Returns:
        action: np.array of shape (32, ), the action for the robot
    '''
    # order of actions: 3 waist + 3 head + 4 right arm + 6 right hand + 3 rigth arm + 7 left arm + 6 left hand
    action = np.zeros(32)

    # order of target joints: 3 waist + 3 head + 7 right arm + 7 left arm + 6 right hand + 6 left hand
    gripper_action = gripper_joint_pos_controller(obs, target_joints[-12:], kp=100)

    # right hand (action[10:16])
    action[10:16] = gripper_action[:6]
    # left hand (action[26:32])
    action[26:32] = gripper_action[6:]
    # waist + head + arms
    action[0:10] = np.clip(5 * (target_joints[0:10] - obs["robot0_joint_pos"][0:10]), -3, 3)
    action[16:26] = np.clip(5 * (target_joints[10:20] - obs["robot0_joint_pos"][10:20]), -3, 3)
    # special care for the head
    action[3:6] = np.clip(0.1 * (target_joints[3:6] - obs["robot0_joint_pos"][3:6]), -0.0, 0.0)

    return action

def parse_hands_in_ik_res(ik_joint_qpos):
    '''
    Parse the hands in the results of inverse kinematics.

    Args:
        ik_joint_qpos: np array of shape (56, ), the results of inverse kinematics for all body joints
                        order: 3 waist + 3 head + 7 left arm + 12 left hand + 7 right arm + 12 right hand + 6 left leg + 6 right leg

    Returns:
        np array of shape (12, ), the joint qpos for the right hand (6) and left hand (6)
    '''
    hand_qpos = np.zeros(12)
    actuator_idxs = np.array([0, 1, 8, 10, 4, 6])
    hand_qpos[0:6] = ik_joint_qpos[32 + actuator_idxs]  # right hand
    hand_qpos[6:12] = ik_joint_qpos[13 + actuator_idxs]  # left hand
    return hand_qpos