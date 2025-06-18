"""
A script to run okami in simulation.
"""

import argparse
import datetime
import json
import os
import gc
import io
import shutil
import time
from glob import glob

import cv2
import numpy as np
import pickle

import threading

import init_path

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config

import mujoco
import mujoco.viewer

from okami.oar.algos.sim_control import SimControl
from okami.oar.algos.object_localizer import ObjectLocalizer, GroundedSamWrapper
from okami.oar.utils.frame_transformation import GR1Transform
from okami.oar.algos.target_pose_generator import GraspPoseGenerator
from okami.oar.algos.trajectory_generator import TrajGenerator
from okami.oar.utils.urdf_utils import obs_to_urdf, obs_to_robosuite_cmds, append_hand_q, urdf_arm_hand_idxs
from okami.oar.algos.ik_solver import IK
from okami.plan_generation.utils.misc_utils import VideoWriter

all_saved_states = []
thread_exception = []

def run_okami(sim: SimControl, gsam_wrapper, plan, no_vis, baseline, folder):
    
    thread_exception.append(False)

    try:

        time.sleep(1)
        obs = sim.get_obs().copy()
        cur_q = obs_to_urdf(obs)
        transform = GR1Transform()
        transform.update_q(q=cur_q.copy(), base_pos=obs['base_pos'])

        object_det = ObjectLocalizer(transform, gsam_wrapper)
        pose_generator = GraspPoseGenerator(transform)
        traj_generator = TrajGenerator(transform)
        IK_solver = IK(transform)

        print("init ok-------------------------------------------------------------")

        for p_idx, p in enumerate(plan):
            
            if p['segment_type'] == 'release':
                # release hand
                goal_q = append_hand_q(cur_q, np.zeros(12), np.zeros(12))
                interp_traj = []
                num = 5
                for i in range(num):
                    interp_traj.append(cur_q + (goal_q - cur_q) * i / num)
                    
                for i in range(len(interp_traj)):
                    sim.add_urdf_cmd(interp_traj[i], save_state=True)
                    
                cur_q = goal_q
                
                # wait for simControl to process all above commands
                while True:
                    time.sleep(1)
                    if sim.cmd_idx >= len(sim.cmd_lst):
                        time.sleep(1)
                        break 
                continue
            
            # localize object
            if p['segment_type'] == 'reach':
                object_name = p['manipulate_object'][0]
            else:
                object_name = p['reference_object']
            if object_name != 'None':
                object_pos_relative_to_head = object_det.localize(object_name, sim.get_obs())
                print("object_pos_relative_to_head: ", object_pos_relative_to_head, "object_pos_in_world=", transform.apply_transform_to_point(object_pos_relative_to_head, "head", "world"))

                object_pos_in_world = transform.apply_transform_to_point(object_pos_relative_to_head, "head", "world")
                if p['segment_type'] == 'manipulation':
                    object_world_offset = p['target_object_translation']
                    print("object_world_offset: ", object_world_offset)
                    object_pos_in_world += np.array(object_world_offset) + np.array([0, 0, 0.05])
                    object_pos_relative_to_head = transform.apply_transform_to_point(object_pos_in_world, "world", "head")

                print("add translation ok!")
                
                # obtain all original IK targets
                smplh_traj = p['smplh_traj']
                ik_targets_traj, hand_targets_traj = IK_solver.obtain_targets(smplh_traj)

                print("obtain IK targets ok!")

                # generate target pose
                object_point_in_head_frame = object_pos_relative_to_head.copy()
                lr = p['moving_arm']
                ref_waist_pose = ik_targets_traj[-1][f'link_{lr}Arm7']
                
                target_palm_pose = pose_generator.generate_palm_pose(object_point_in_head_frame, ref_waist_pose, lr)

                print("generate target palm pose ok!")

                # warp trajectory and solve IK
                if not baseline:
                    warped_traj, end_pose_in_upperbase = traj_generator.warp_trajectory(ik_targets_traj, 
                                                                cur_q, 
                                                                lr, 
                                                                target_palm_pose, 
                                                                calibrate_data=smplh_traj[0], 
                                                                vis=not no_vis,
                                                                is_reach=((p_idx == 0) or (p_idx == 2)))
                else:
                    warped_traj, _ = traj_generator.warp_trajectory_baseline(ik_targets_traj,
                                                                             cur_q,
                                                                             lr,
                                                                             target_palm_pose[:3, 3],
                                                                             hand_targets_traj,
                                                                             calibrate_data=smplh_traj[0],
                                                                             vis=not no_vis,
                                                                             is_reach=((p_idx == 0) or (p_idx == 2)))
                    
                
            else: # no reference object
                # obtain all original IK targets
                smplh_traj = p['smplh_traj']
                ik_targets_traj, hand_targets_traj = IK_solver.obtain_targets(smplh_traj)

                # translate the trajectory to start at current q
                lr = p['moving_arm']
                warped_traj, end_pose_in_upperbase = traj_generator.translate_trajectroy(
                    ik_targets_traj, cur_q, lr, calibrate_data=smplh_traj[0], vis=not no_vis)
                
            # keep the other arm in the same position
            other_arm_name = 'L' if lr == 'R' else 'R'
            other_arm_joint_idx, other_hand_joint_idx = urdf_arm_hand_idxs(other_arm_name)
            for i in range(len(warped_traj)):
                warped_traj[i][other_arm_joint_idx] = cur_q[other_arm_joint_idx]
                warped_traj[i][other_hand_joint_idx] = cur_q[other_hand_joint_idx]
            
            # interpolate the warped traj
            interpolated_traj = []
            interpolate_steps = 2
            for i in range(len(warped_traj) - 1):
                for j in range(interpolate_steps):
                    interpolated_traj.append(warped_traj[i] + (warped_traj[i+1] - warped_traj[i]) * j / interpolate_steps)
            interpolated_traj.append(warped_traj[-1])

            # send commands to simControl
            for i in range(len(interpolated_traj)):
                sim.add_urdf_cmd(interpolated_traj[i], save_state=True)

            cur_q = interpolated_traj[-1]

            # wait for simControl to process all above commands
            while True:
                time.sleep(1)
                if sim.cmd_idx >= len(sim.cmd_lst):
                    time.sleep(1)
                    break 

        # wait for simControl to process all above commands
        while True:
            time.sleep(1)
            if sim.cmd_idx >= len(sim.cmd_lst):
                time.sleep(1)
                break 
        
        saved_states = sim.saved_info
        # generate video
        video_writer = VideoWriter(video_path=folder, video_name=f"rollout.mp4", fps=30, save_video=True)
        for i in range(len(saved_states)):
            img = saved_states[i]['rgb']
            video_writer.append_image(img)
        video_path = video_writer.save()
        print("video saved to ", video_path)
        
        if sim.get_reward() > 0.5: # meaning this is a successful trail
            print("success!")
            
            # all_saved_states.append(saved_states)
            all_saved_states.append(0)
            num_all_saved_states = len(all_saved_states)

            # save current state
            with open(os.path.join(folder, f"saved/{num_all_saved_states}.pkl"), "wb") as f:
                pickle.dump(saved_states, f)
            print("state saved!")

        sim.reset_episode()
        time.sleep(3)

        print("finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    except Exception as e:
        print("Error: ", e, "occurred!", "Episode failed!")
        time.sleep(1)
        sim.terminate()
        thread_exception[-1] = True

def close_all_open_files():
    for obj in gc.get_objects():
        if isinstance(obj, io.IOBase) and not obj.closed:
            try:
                print(f"Closing file: {obj.name}")
                obj.close()
            except Exception as e:
                print(f"Error closing file: {e}")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="HumanoidPour")
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    parser.add_argument("--robots", nargs="+", type=str, default="GR1FloatingBody", help="Which robot(s) to use in the env")
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

    parser.add_argument("--num-demos", type=int, default=1, help="Number of demonstrations to run")
    parser.add_argument("--no-vis", action="store_true", help="Turn off visualization")
    parser.add_argument("--state-only", action="store_true", default=False, help="Use object states to locate object")
    parser.add_argument("--baseline", action="store_true", default=False, help="Run baseline")

    args = parser.parse_args()

    environment = args.environment
    annotation_folder = os.path.join("annotations", "human_demo", args.human_demo.split("/")[-1].split(".")[0])
    
    # read in data    
    data_path = os.path.join(annotation_folder, "segments_info.json")
    smplh_path = os.path.join(annotation_folder, "smplh_traj.pkl")
    with open(data_path, "rb") as f:
        segments_info = json.load(f)
    with open(smplh_path, "rb") as f:
        data = pickle.load(f)
    print("len of smplh data is", len(data))

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    controller_config["kp"] = 500

    # Create argument configuration
    config = {
        "env_name": environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Create environment
    env = suite.make(
        **config,
        has_renderer=False if args.no_vis else True,
        renderer=args.renderer,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["agentview", "frontview", "robot0_robotview"],
        camera_heights=720,
        camera_widths=1280,
        camera_depths=True,
        reward_shaping=True,
        control_freq=20,
    )

    # generate step by step plan from data
    plan = []
    for i in range(len(segments_info)):
        step_info = segments_info[i]
        start_idx = step_info['start_idx']
        end_idx = step_info['end_idx']
        smplh_traj = data[start_idx:end_idx]

        step_info['smplh_traj'] = smplh_traj
        plan.append(step_info)

    num_demonstrations = args.num_demos
    
    if not args.state_only:
        gsam_wrapper = GroundedSamWrapper()
        print("GSAM wrapper initialized!")
    else:
        gsam_wrapper = None

    os.makedirs(os.path.join(annotation_folder, "rollout/saved"), exist_ok=True)
    result_folder = os.path.join(annotation_folder, "rollout")
    print("results will be saved to: ", result_folder)

    sim = SimControl(env)

    demo_idx = 0
    all_saved_states = []
    for i in range(demo_idx):
        all_saved_states.append(0)
    while demo_idx < num_demonstrations:
        print("demo_idx=", demo_idx, "/", num_demonstrations)
        print("==========================================================================")

        start_time = datetime.datetime.now()

        thread = threading.Thread(target=run_okami, args=(sim, gsam_wrapper, plan, args.no_vis, args.baseline, result_folder))
        thread.daemon = True
        thread.start()

        try:

            sim.run(vis=not args.no_vis)

            if thread_exception[-1]:
                print("thread exception occurred!")
                raise Exception("thread exception occurred!")

            end_time = datetime.datetime.now()
            print("time cost: ", end_time - start_time)
            print("current success rates=", len(all_saved_states), "/", (demo_idx + 1), "=", len(all_saved_states) / (demo_idx + 1))
            print()
            print("==========================================================================")
            print()

            demo_idx += 1

        except Exception as e:

            print("Error: ", e, "occurred!", "Episode failed! Will try to restart SimControl!")
            time.sleep(1)

            while True:
                try:
                    try:
                        sim
                    except NameError:
                        sim = None

                    if sim is not None:
                        sim.terminate()
                        print("sim terminated!")
                        # destroy sim
                        del sim
                        print("sim deleted!")

                    print("recreate sim...")
                    sim = SimControl(env)
                    break
                except Exception as e:
                    print("Error: ", e, "occurred!", "Failed to restart SimControl!")
                    time.sleep(1)
                    close_all_open_files()
                    sim = None

    sim.terminate()

    print("number of successful trials: ", len(all_saved_states))

    exit()