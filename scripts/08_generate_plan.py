import argparse
import numpy as np
import os
import json

import init_path
from okami.plan_generation.utils.misc_utils import *
from okami.plan_generation.algos.human_video_plan import HumanVideoPlan

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument("--human-demo", type=str, default='salt_demo.hdf5')
    args = parser.parse_args()
    
    mode = "human_demo"
    annotation_folder = f"annotations/{mode}"
    annotation_path = os.path.join(annotation_folder, args.human_demo.split("/")[-1].split(".")[0])

    human_video_hoig = HumanVideoPlan()
    human_video_hoig.generate_from_human_video(annotation_path, 
                                               zero_pose_name="ready",
                                               video_smplh_ratio=1.0, 
                                               use_smplh=True)
    
    human_video_hoig.visualize_plan(no_smplh=False)

    segments_info = human_video_hoig.get_all_segments_info()
    with open(os.path.join(args.annotation_folder, "segments_info.json"), "w") as f:
        json.dump(segments_info, f)

if __name__ == "__main__":
    main()