import os

import init_path
import argparse
import json
import pprint
import time

import cv2
import numpy as np
import h5py
import shutil

from robot.gr1 import GR1URDFModel
from utils.video_utils import VideoWriter
import deoxys_vision.utils.transformation.transform_utils as T

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str)
    parser.add_argument("--exp-name", type=str)
    return parser.parse_args()

def load_depth_in_rgb(rgb_img):
    depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1])).astype(np.uint16)
    depth_img = rgb_img[..., 1].astype(np.uint16) << 8 | rgb_img[..., 2].astype(np.uint16)
    return depth_img

def main():

    args = parse_args()

    folder = os.path.join("data", args.task_name, args.exp_name)
    os.makedirs(folder, exist_ok=True)
    
    res = []
    # list all files in folder
    # files = os.listdir(folder)
    # files = sorted(files)
    # for f in files:
    #     if not f.endswith(".json"):
    #         continue
    #     if not f.startswith("states_"):
    #         continue
    #     with open(os.path.join(folder, f), 'r') as file:
    #         try:
    #             # print("filename=", f)
    #             data = json.load(file)
    #             res.append(data)
    #         except:
    #             print(f"Error reading {f}")
    #             continue
    with open(os.path.join(folder, "all_states.json"), 'r') as file:
        res = json.load(file)

    gr1 = GR1URDFModel()

    filename = os.path.join(folder, f"states.hdf5")
    with h5py.File(filename, 'w') as f:
        ep_grp = f.create_group(f"demo_{args.exp_name}")
        obs_grp = ep_grp.create_group("obs")

        img_lst = []
        depth_lst = []
        robot_state_lst = []
        eef_lst = []
        for i, data in enumerate(res):
            image_idx = data["image_idx"]
            robot_state = data["robot_state"]

            # print("image idx = ", image_idx)
            # check if image exists
            if not os.path.exists(f"dora_deoxys_vision_example/tmp_image/rgb/{image_idx:07d}.png"):
                print(f"Image {image_idx} does not exist")
                continue

            img = cv2.imread(f"dora_deoxys_vision_example/tmp_image/rgb/{image_idx:07d}.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (640, 360))
            img = np.ascontiguousarray(img)
            
            depth = cv2.imread(f"dora_deoxys_vision_example/tmp_image/depth/{image_idx:07d}.png")
            depth = load_depth_in_rgb(depth)
            # depth = cv2.resize(depth, (640, 360))

            eef_pose = gr1.get_gripper_pose(np.array(robot_state))
            # TODO: transfrom to euler angles
            rot_mat = eef_pose[:3, :3]
            pos = eef_pose[:3, 3]
            euler = T.mat2euler(rot_mat)
            pose = np.concatenate([pos, euler])
            eef_lst.append(pose)

            img_lst.append(img)
            depth_lst.append(depth)
            robot_state_lst.append(robot_state)

        obs_grp.create_dataset("agentview_rgb", data=img_lst)
        obs_grp.create_dataset("agentview_depth", data=depth_lst)

        ep_grp.create_dataset("actions", data=robot_state_lst)
        ep_grp.create_dataset("joint_states", data=robot_state_lst)
        ep_grp.create_dataset("ee_states", data=eef_lst)

        videowriter = VideoWriter(folder, video_name="video.mp4", fps=30, single_video=True, save_video=True)
        for frame in img_lst:
            videowriter.append_image(frame)
        videowriter.save()
        
        cv2.imwrite(os.path.join(folder, "initial_frame.png"), cv2.cvtColor(img_lst[0], cv2.COLOR_RGB2BGR))

    # TODO: delete images and robot states

    res = {}
    str = input("Success or not?[y/n]")
    if str == 'y':
        res["success"] = True
        res["reason"] = ""
    else:
        res["success"] = False
        str = input("Reason?")
        res["reason"] = str
    with open(os.path.join(folder, "result.json"), 'w') as file:
        json.dump(res, file)
        
    print("Finished, hdf5 file saved to", filename, "video saved to ", os.path.join(folder, "video.mp4"))

    # print("deleting images")
    str = input("delete images? [y/n]")
    if str == 'y':
        shutil.rmtree('dora_deoxys_vision_example/tmp_image/')
        print("Images deleted")

if __name__ == "__main__":
    main()
