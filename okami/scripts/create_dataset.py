import os

import init_path
import argparse
import json
import pprint
import time

import cv2
import numpy as np
import h5py

from robot.gr1 import GR1URDFModel
import deoxys_vision.utils.transformation.transform_utils as T

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", type=str)
    return parser.parse_args()

def load_depth_in_rgb(rgb_img):
    depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1])).astype(np.uint16)
    depth_img = rgb_img[..., 1].astype(np.uint16) << 8 | rgb_img[..., 2].astype(np.uint16)
    return depth_img

def main():

    args = parse_args()

    folder = os.path.join("data", args.task_name)
    filename = os.path.join(folder, f"dataset.hdf5")
    
    # list all directories in folder
    dirs = os.listdir(folder)
    dirs = sorted(dirs)

    idx = 0

    with h5py.File(filename, 'w') as f:

        grp = f.create_group("data")

        for d in dirs:
            
            idx += 1
            if idx > 2:
                break

            if not os.path.isdir(os.path.join(folder, d)):
                continue

            demo_filepath = os.path.join(folder, d, "states.hdf5")
            if not os.path.exists(demo_filepath):
                print(f"File {demo_filepath} does not exist")
                continue
            
            print(f"Processing {demo_filepath}")
            with h5py.File(demo_filepath, 'r') as demo_f:
                # ep_grp = grp.create_group(f"demo_{d}")
                for item in demo_f.keys():
                    demo_f.copy(item, grp)
                grp[f'demo_{d}'].attrs["num_samples"] = len(demo_f[f'demo_{d}']['obs']['agentview_rgb'])
                
    
    # TODO: add neccesary attributes
        
    # TODO: delete images and robot states
        
    print("Finished, file saved to ", filename)

if __name__ == "__main__":
    main()
