import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import sys
import json

from PIL import Image
import numpy as np
import argparse
import shutil
import torch

import matplotlib.pyplot as plt
from easydict import EasyDict
import init_path
from okami.plan_generation.utils.misc_utils import (
    load_first_frame_from_hdf5_dataset, 
    export_video_from_hdf5_dataset,
    load_first_frame_from_human_hdf5_dataset, 
    export_video_from_human_hdf5_dataset,
    overlay_xmem_mask_on_image
    )
from okami.plan_generation.algos.grounded_sam_wrapper import GroundedSamWrapper
import argparse


torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_demo", default="datasets/rgbd/salt_demo.hdf5", help="Path to a human demo file")
    return parser.parse_args()

def preprocess_obj_for_gsam(obj):
    obj = obj.lower()
    gsam_replace = {'can': 'bottle', 'canister': 'bottle'} # adjust object name for G-SAM to understand
    if obj in gsam_replace:
        obj = gsam_replace[obj]
    return obj

def main():
    args = parse_args()
    
    wrapper = GroundedSamWrapper()
    
    # args.human_demo = os.path.join("datasets/rgbd", args.human_demo)

    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    tmp_folder = "tmp_images"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    tmp_path = tmp_folder

    os.makedirs(annotation_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "images"), exist_ok=True)

    first_frame = load_first_frame_from_human_hdf5_dataset(args.human_demo, bgr=True)

    cv2.imwrite(os.path.join(os.path.join(tmp_path, "images", "frame.jpg")), first_frame)

    with open(os.path.join(annotation_path, "text_description.json"), "r") as f:
        text_description = json.load(f)["objects"]
    text_description = [preprocess_obj_for_gsam(obj) for obj in text_description]
    with open(os.path.join(annotation_path, "text_description.json"), "w") as f:
        json.dump({"objects": text_description}, f)
    
    print("Annotating the image with text input: ", text_description)

    final_mask_image = wrapper.segment(first_frame, text_description)
    os.makedirs(os.path.join(tmp_path, "masks"), exist_ok=True)

    if isinstance(final_mask_image, np.ndarray):
        final_mask_image = Image.fromarray(final_mask_image)
    if final_mask_image.mode == 'F':
        final_mask_image = final_mask_image.convert('RGB')

    final_mask_image.save(os.path.join(tmp_path, "masks", "frame.png"))
    overlay_image = overlay_xmem_mask_on_image(first_frame, np.array(final_mask_image), use_white_bg=True, rgb_alpha=0.3)

    # copy results to annotation folder
    shutil.copyfile(os.path.join(tmp_path, "images", "frame.jpg"), os.path.join(annotation_path, "frame.jpg"))
    shutil.copyfile(os.path.join(tmp_path, "masks", "frame.png"), os.path.join(annotation_path, "frame_annotation.png"))
    print("Annotation saved to ", os.path.join(annotation_path, "frame_annotation.png"))
    with open(os.path.join(annotation_path, "config.json"), "w") as f:
        config_dict = {"mode": mode}
        config_dict["original_file"] = args.human_demo
        video_path = export_video_from_human_hdf5_dataset(
                        dataset_name=args.human_demo, 
                        video_path=annotation_path,
                        video_name=args.human_demo.split("/")[-1].split(".")[0],
                        bgr=True)
        config_dict["video_file"] = video_path
        config_dict["text"] = text_description
        json.dump(config_dict, f)
        
    # remove tmp folder
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()

