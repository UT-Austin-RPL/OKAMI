"""
A script to test joint position controller for GR1FloatingBody.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import cv2
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper

import mujoco
import mujoco.viewer

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

    # print("position error: ", position_error[2], vel_pos_error[2], position_error[8], vel_pos_error[8])

    # calculate the torques: kp * position_error + kd * vel_pos_error
    kd = 2 * np.sqrt(kp) * damping_ratio - 15
    desired_torque = np.multiply(np.array(position_error), np.array(kp)) + np.multiply(vel_pos_error, kd)

    # print("kp=", kp, "kd=", kd)

    # clip and rescale to [-1, 1]
    desired_torque = np.clip(desired_torque, -1, 1)

    # print("desired_torque", desired_torque[0])

    return desired_torque

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
    action[14:26] = gripper_joint_pos_controller(obs, target_joint_pos[14:26])

    # for base joints
    # TODO

    # for the head joints
    action[29:32] = np.clip(0.1 * (target_joint_pos[29:32] - obs["robot0_joint_pos"][3:6]), -0., 0.)
    
    # for the waist joints
    action[32:35] = np.clip(5 * (target_joint_pos[32:35] - obs["robot0_joint_pos"][0:3]), -1, 1)
    return action

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="HumanoidDrawer")
    parser.add_argument("--robots", nargs="+", type=str, default="GR1FloatingBody", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="JOINT_POSITION", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        help="Use the Nvisii viewer (Nvisii), OpenCV viewer (mujoco), or Mujoco's builtin interactive viewer (mjviewer)",
    )
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_robotview"],
        camera_heights=720,
        camera_widths=1280,
        camera_depths=True,
        reward_shaping=True,
        control_freq=20,
    )

    env.reset()
    obs, reward, done, _ = env.step(np.zeros(35))

    print("keys in obs: ", obs.keys())
    print("shape of joint pos in obs", obs["robot0_joint_pos"].shape)

    m = env.sim.model._model
    d = env.sim.data._data
    # mujoco.viewer.launch(m, d)
    
    with mujoco.viewer.launch_passive(
        model=m,
        data=d,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        fps = 20
        while viewer.is_running():

            target_joint_pos = np.zeros(35)
            right_arm_cmd = np.array([0., 1.57, 0., -1.57, 0., 0., 0.])
            left_arm_cmd = np.array([0., -1.57, 0., 1.57, 0., 0., 0.])
            head_cmd = np.array([0.0, 0.0, 0.34])
            waist_cmd = np.array([0.0, 0.22, 0.0])
            left_hand_cmd = np.array([0.0, 0.0, 1.57, 0.0, 0.0, 0.0])
            right_hand_cmd = np.array([0.0, 0.0, 1.57, 0.0, 0.0, 0.0])

            target_joint_pos[0:7] = right_arm_cmd
            target_joint_pos[7:14] = left_arm_cmd
            target_joint_pos[14:20] = right_hand_cmd
            target_joint_pos[20:26] = left_hand_cmd
            target_joint_pos[29:32] = head_cmd
            target_joint_pos[32:35] = waist_cmd

            action = np.zeros(35)
            action = joint_pos_controller(obs, target_joint_pos)
            # print("for waist roll joint, action=", action[33], "target joint pos=", target_joint_pos[33], "current joint pos=", obs["robot0_joint_pos"][1])

            obs, reward, done, _ = env.step(action)

            assert('robot0_robotview_image' in obs)
            assert('robot0_robotview_depth' in obs)
            img = obs['robot0_robotview_image']
            depth = obs['robot0_robotview_depth']

            # image transformation (cvt BGR to RGB; flip the image)
            img = cv2.flip(img, 0)
            depth = cv2.flip(depth, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imshow("image", img)
            cv2.waitKey(10)

            viewer.sync()
            time.sleep(1 / fps)
    exit()