"""
A script to test camera streaming.
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

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="HumanoidReach")
    parser.add_argument("--robots", nargs="+", type=str, default="GR1FloatingBody", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
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
    env.step(np.zeros(33)) # env.robots[0].action_dim

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
            action = np.zeros(33)

            obs, reward, done, _ = env.step(action)

            print("keys in obs: ", obs.keys())
            assert('robot0_robotview_image' in obs)
            assert('robot0_robotview_depth' in obs)
            
            img = obs['robot0_robotview_image']
            depth = obs['robot0_robotview_depth']
            print("shape of robotview images is", img.shape, depth.shape)

            # image transformation (cvt BGR to RGB; flip the image)
            img = cv2.flip(img, 0)
            depth = cv2.flip(depth, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imshow("image", img)
            cv2.waitKey(10)

            viewer.sync()
            time.sleep(1 / fps)
    exit()