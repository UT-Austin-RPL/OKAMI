import os
import re
import time
import json
import init_path
import argparse
import pickle
import numpy as np
import time
import cv2

from robot.gr1 import GR1URDFModel
from scipy.spatial.transform import Rotation as R

from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config

from gr1_interface.gr1_control.gr1_client import gr1_interface
from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface

from utils.save_state_utils import StateSaver
from utils.interpolate_utils import run_interpolation, interpolate_to_start_pos, interpolate_to_end_pos, change_hand_pos, append_hand_pos
from utils.real_robot_utils import process_urdf_joints
from utils.frame_transformation_utils import GR1Transform
from utils.grasp_offset_utils import calculate_wrist_offset_relative_to_object_in_head_frame, up_or_side_grasp
from algos.hoig import HOIG
from algos.oar import offset_oar, warp_oar, translation_oar, warp_oar_baseline

def communicate_with_notebook_to_obtain_object_target(step_info: HOIG, gr1_transform: GR1Transform):

    if step_info.reference_object_name == 'None' or (step_info.reference_object_name == step_info.moving_object_name):
        print("No reference object. Skip.")
        return np.array([0, 0, 0])

    print("Looking for object: ", step_info.reference_object_name)
    with open('tmp_results/status.json', 'w') as f:
        json.dump({
            "object_name": step_info.reference_object_name,
            "wrist_offset_in_head": step_info.wrist_offset_in_head.tolist(),
            "object_target_in_head": [0, 0, 0],
            "ready": 0,
            "lr": step_info.lr,
            "direction": step_info.direction,
            "T_world_head": gr1_transform.get_transform('head', 'world').tolist(),
        }, f)

    while True:
        with open('tmp_results/status.json', 'r') as f:
            status = json.load(f)
        if status['ready'] == 1:
            object_target_in_head = status['object_target_in_head']
            print("object_target_in_head", object_target_in_head)
            return object_target_in_head
        time.sleep(0.1)

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--smplh-path", type=str, default="R", help="Path to smplh data")
        parser.add_argument("--save-states", action="store_true", default=False, help="Save states")
        parser.add_argument("--task-name", type=str, default="test")
        parser.add_argument("--exp-name", type=str, default="0")
        parser.add_argument("--data-path", type=str, default="R", help="Path to segments info obtained from human video")
        parser.add_argument("--lr", default=None)
        parser.add_argument("--baseline", action="store_true", default=False)
        parser.add_argument("--mirror", action="store_true", default=False)
        args = parser.parse_args()

        gr1 = gr1_interface(
            "10.42.0.21", pub_port_arm=5555, pub_port_hand=6666, sub_port=5556, rate=40
        )
        print("start gr1 interface!")

        if args.save_states:
            state_saver = StateSaver(args.task_name, args.exp_name)
        else: 
            state_saver = None

        start_pos = interpolate_to_start_pos(gr1, steps=50, lr=args.lr) #, state_saver=state_saver

        # calibrate T_world_chest
        gr1_transform = GR1Transform()
        gr1_transform.calibrate_T_world_chest_from_camera_rgbd(start_pos)
        gr1_transform.update_q(start_pos)
        
        with open(args.smplh_path, "rb") as f:
            data = pickle.load(f)
        print("len of smplh data is", len(data))

        with open(args.data_path, "rb") as f:
            segments_info = json.load(f)
        
        last_pos = start_pos.copy()
        for segment_idx, segment in enumerate(segments_info):

            input(f"Ready to start segment {segment_idx}/{len(segments_info)}? If so, press enter.")

            if segment_idx == len(segments_info) - 1:
                print("reached last step. stop.")
                break

            step_info = HOIG(calibrate_data=data[0], lr=segment['moving_arm'])
            step_info.add(data[segment['start_idx']: segment['end_idx']], 
                          hand_pose=segment['grasp_type'], 
                          object_names=[segment['reference_object'], segment['manipulate_object']], 
                          gr1_transform=gr1_transform, 
                          baseline=args.baseline)
            step_info.print()

            object_target_in_head = communicate_with_notebook_to_obtain_object_target(step_info, gr1_transform)
            if segment['manipulate_object'] == 'None' or (segment['manipulate_object'] == segment['reference_object']): # no object to manipulate
                print("no object to manipulate")
                objects_translation_in_world = np.array([0., 0., 0.])
            else:
                if ('target_translation' in segment) and (len(segment['target_translation']) == 3):
                    print("segment['target_translation']", segment['target_translation'])
                    objects_translation_in_world = np.array([0., 0., 0.])
                    objects_translation_in_world[2] = segment['target_translation'][2] + 0.18 # 0.25
                    if objects_translation_in_world[2] < 0.08:
                        objects_translation_in_world[2] = 0.08
                    print("after using the value from video plan, objects_translation_in_world", objects_translation_in_world)
                else:
                    objects_translation_in_world = np.array([0.0, 0., 0.10])

            print("objects_translation_in_world", objects_translation_in_world)
            objects_translation_in_head = gr1_transform.apply_transform_to_point(objects_translation_in_world, 'world', 'head')
            target_pos_in_head = object_target_in_head + objects_translation_in_head + np.array(step_info.wrist_offset_in_head) 

            if segment['reference_object'] == 'None' or (segment['reference_object'] == segment['manipulate_object']):
                ik_traj = translation_oar(step_info, last_pos, target_pos_in_head, gr1_transform)
                # ik_traj = warp_oar(step_info, last_pos, target_pos_in_head, gr1_transform)
            else:
                if args.baseline:
                    ik_traj = warp_oar_baseline(step_info, last_pos, target_pos_in_head, gr1_transform)
                else:
                    ik_traj = warp_oar(step_info, last_pos, target_pos_in_head, gr1_transform)

            input("Ready to replay the trajectory on real robot? If so, press enter.")

            for i in range(len(ik_traj) - 1):
                run_interpolation(ik_traj[i], ik_traj[i + 1], gr1, 5, state_saver=state_saver)
            time.sleep(1)

            if segment_idx == len(segments_info) - 1:
                grasp_pos = change_hand_pos(ik_traj[-1], 'palm', 'palm', gr1, steps=50, state_saver=state_saver)
            else:
                grasp_pos = change_hand_pos(ik_traj[-1], 
                                            segments_info[segment_idx + 1]['grasp_type'][0], 
                                            segments_info[segment_idx + 1]['grasp_type'][1], 
                                            gr1, 
                                            steps=50, 
                                            state_saver=state_saver)
                
            last_pos = grasp_pos.copy()

        input("Ready to get back to end pos? If so, press enter.")
        interpolate_to_end_pos(ik_traj[-1], gr1, steps=50) #, state_saver=state_saver)

        # terminate gr1
        gr1.control(terminate=True)
        gr1.close_threads()

        # save states
        if state_saver is not None:
            state_saver.save()

    except KeyboardInterrupt:
        print("Interrupted")
        gr1.control(terminate=True)
        gr1.close_threads()
        print("Finished")

        if state_saver is not None:
            state_saver.save()
    
    except Exception as e:
        print(e)
        input("Press key to reset")
        gr1.control(terminate=True)
        gr1.close_threads()
        print("Terminated")

        if state_saver is not None:
            state_saver.save()