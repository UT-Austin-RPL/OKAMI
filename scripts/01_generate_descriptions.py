import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import sys
import json

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
    load_multiple_frames_from_human_hdf5_dataset,
    export_video_from_human_hdf5_dataset,
    overlay_xmem_mask_on_image
    )
from okami.plan_generation.utils.gpt4v_utils import encode_imgs_from_path, json_parser
from okami.plan_generation.algos.gpt4v import GPT4V
import argparse

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_demo", default="datasets/rgbd/salt_demo.hdf5", help="Path to a human demo file")
    parser.add_argument("--num_frames", type=int, default=8, help="Number of frames to process")
    return parser.parse_args()

def identify_objects_prompt():
    prompt = '''You need to analyze what the human is doing in the images, then tell me:
1. All the objects in front scene (mostly on the table). You should ignore the background objects!
2. The objects of interest. They should be a subset of your answer to the first question. 
They are likely the objects manipulated by human or have interaction with objects manipulated by human, such as serving as their containeres. Note that there are irrelevant objects in the scene, such as objects that does not move and doesn't interact with objects manipulated by human. You should ignore the irelevant objects.

Your output format is:

The human is xxx. 
All objects are xxx.
The objects of interest are:
```json
{
    "objects": ["OBJECT1", "OBJECT2", ...],
}
```

Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc. 
You should output the names of objects of interest in a list ["OBJECT1", "OBJECT2", ...] that can be easily parsed by Python. The name is a string, e.g., "apple", "pen", "keyboard", "bottle", etc.
'''
    return prompt

def identify_objects(vlm, img_paths):
    vlm.begin_new_dialog()
    base64_img_list = [encode_imgs_from_path(vlm, img_paths)]
    text_prompt_list = [identify_objects_prompt()]
    text_response = vlm.run(text_prompt_list, base64_img_list)
    print(text_response)

    json_data = json_parser(text_response)
    if json_data is None:
        return None
    return json_data

def main():
    args = parse_args()

    # args.human_demo = os.path.join("datasets/rgbd", args.human_demo)

    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    tmp_folder = "tmp_images"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    tmp_path = tmp_folder

    os.makedirs(annotation_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(os.path.join(tmp_path, "images"), exist_ok=True)

    image_lst = load_multiple_frames_from_human_hdf5_dataset(args.human_demo, args.num_frames, bgr=True)

    img_paths = []
    for idx, frame in enumerate(image_lst):
        cv2.imwrite(os.path.join(os.path.join(tmp_path, "images", f"frame_{idx}.jpg")), frame)
        img_paths.append(os.path.join(os.path.join(tmp_path, "images", f"frame_{idx}.jpg")))

    with open(os.path.join(annotation_path, "text_description.json"), "w") as f:
        vlm = GPT4V()
        text_description = identify_objects(vlm, img_paths)

        json.dump(text_description, f)

    # remove the folder
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()

