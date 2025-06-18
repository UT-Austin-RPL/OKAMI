import os
import argparse
import json
import pprint
import time

import cv2
import numpy as np
import h5py
import pickle

import init_path
from okami.oar.utils.urdf_utils import robosuite_cmds_to_urdf, parse_urdf_cmds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    parser.add_argument("--single_arm", action="store_true", default=False)
    parser.add_argument("--delta", action="store_true", default=False)
    return parser.parse_args()

def main():

    args = parse_args()
    annotation_path = os.path.join("annotations/human_demo", args.human_demo.split("/")[-1].split(".")[0])

    input_folder = os.path.join(annotation_path, "rollout/saved")

    data_file_name = []
    # read in all the files in the input directory
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".pkl"):
                data_file_name.append(os.path.join(root, file))

    print("number of data=", len(data_file_name))

    start_time = time.time()

    out_filename = os.path.join(annotation_path, "rollout/data.hdf5")
    with h5py.File(out_filename, 'w') as f:
        # first create a big data group
        data_grp = f.create_group("data")

        num_demo = len(data_file_name)
        num_demo = min(num_demo, 100) # 100 is the largest number of demos contained in one hdf5 dataset
        
        all_samples = 0

        for i in range(num_demo):
            ep_grp = data_grp.create_group(f"demo_{i}")
            obs_grp = ep_grp.create_group("obs")

            img_lst = []
            depth_lst = []
            obs_joint_lst = [] # convert to 56-dim
            obs_body_joint_lst = []
            obs_hand_joint_lst = []
            action_joint_lst = []
            action_body_joint_lst = []
            action_hand_joint_lst = []
            done_lst = []

            with open(data_file_name[i], "rb") as f:
                current_data = pickle.load(f)
            for d in current_data:
                rgb_img = d["rgb"]
                rgb_img = cv2.resize(rgb_img, (224, 224))
                depth_img = d["depth"]
                depth_img = cv2.resize(depth_img, (224, 224))
                depth_img = np.expand_dims(depth_img, axis=-1)
                
                img_lst.append(rgb_img)
                depth_lst.append(depth_img)
                
                done_lst.append(0)

                obs_joints = robosuite_cmds_to_urdf(d["joint_obs"])
                body_joints, hand_joints = parse_urdf_cmds(obs_joints)
                obs_body_joint_lst.append(body_joints)
                obs_hand_joint_lst.append(hand_joints)
                
                if args.delta:
                    action_joints = robosuite_cmds_to_urdf(d["joint_action"]) - obs_joints
                    action_joints *= 5 # scale the delta action
                else:
                    # Use absolute joint positions here
                    action_joints = robosuite_cmds_to_urdf(d["joint_action"])
                
                body_joints, hand_joints = parse_urdf_cmds(action_joints)
                action_body_joint_lst.append(body_joints)
                action_hand_joint_lst.append(hand_joints)

                # only pick the 13-dim corresponding to right arm
                right_idx = [25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 38, 40, 42] 
                
                # also pick the 13-dim corresponding to left arm
                left_idx = [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 21, 23]
                
                if args.single_arm:
                    selected_idx = right_idx
                else:
                    selected_idx = left_idx + right_idx
                
                obs_joints = obs_joints[selected_idx]                
                obs_joint_lst.append(obs_joints)
                
                action_joints = action_joints[selected_idx]
                action_joint_lst.append(action_joints)

            done_lst[-1] = 1

            obs_grp.create_dataset("agentview_rgb", data=np.array(img_lst))
            obs_grp.create_dataset("agentview_depth", data=np.array(depth_lst))
            obs_grp.create_dataset("joint_states", data=np.array(obs_joint_lst))
            obs_grp.create_dataset("body_joint_states", data=np.array(obs_body_joint_lst))
            obs_grp.create_dataset("hand_joint_states", data=np.array(obs_hand_joint_lst))

            ep_grp.create_dataset("actions", data=np.array(action_joint_lst))
            ep_grp.create_dataset("body_joint_actions", data=np.array(action_body_joint_lst))
            ep_grp.create_dataset("hand_joint_actions", data=np.array(action_hand_joint_lst))
            ep_grp.create_dataset("dones", data=np.array(done_lst))

            ep_grp.attrs["num_samples"] = len(current_data)
            all_samples += len(current_data)
            print("finished ", i, "/", num_demo)

        data_grp.attrs["num_demos"] = num_demo
        data_grp.attrs["num_samples"] = all_samples

        print("num_demos=", num_demo)

    end_time = time.time()
    print("Time taken to convert to hdf5 dataset: ", end_time - start_time)

if __name__ == "__main__":
    main()
