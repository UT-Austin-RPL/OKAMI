import argparse
import random
import torch
import os
import numpy as np

from pathlib import Path

import init_path
from okami.plan_generation.utils.misc_utils import get_first_frame_annotation, overlay_xmem_mask_on_image, resize_image_to_same_shape, plotly_draw_seg_image, sample_points_in_mask,get_annotation_info, get_depth_seq_from_human_demo
from okami.plan_generation.utils.o3d_utils import O3DPointCloud, load_reconstruction_info_from_human_demo, project3Dto2D, filter_pcd


def sample_points_in_mask_wo_outlier(input_depth,
                                     input_annotation,
                                     camera_intrinsics,
                                     num_points,
                                     object_id=None):
    if object_id is None:
        segmented_depth = input_depth * (input_annotation != 0)
    else:
        segmented_depth = np.squeeze(input_depth) * (input_annotation == object_id)
    new_segmentation = filter_pcd(segmented_depth, camera_intrinsics) * object_id
    return sample_points_in_mask(new_segmentation, num_points, object_id)

def sample_points_in_mask_wo_depth(input_annotation, num_points, object_id=None):
    if object_id is None:
        segmentation = (input_annotation != 0) * object_id
    else:
        segmentation = (input_annotation == object_id) * object_id
    return sample_points_in_mask(segmentation, num_points, object_id)

def annotate_video(annotation_folder, num_track_points, save_video=True, use_depth=False):
    try:
        first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)
        info = get_annotation_info(annotation_folder)
        print("get annotation ok")
        dataset_file_path = info["original_file"]
        if use_depth:
            print("need to use depth")
            first_depth = get_depth_seq_from_human_demo(dataset_file_path)[0]
        print("ready to get intrinsics")
        intrinsics = load_reconstruction_info_from_human_demo(dataset_file_path)["intrinsics"]
        print("arrived here")
        points_list = []
        for object_id in range(1, first_frame_annotation.max() + 1):
            # points = sample_points_in_mask(first_frame_annotation, num_track_points, object_id)
            if use_depth:
                points = sample_points_in_mask_wo_outlier(first_depth, first_frame_annotation, intrinsics, num_track_points, object_id)
            else:
                points = sample_points_in_mask_wo_depth(first_frame_annotation, num_track_points, object_id)
            points_list.append(points)

        points_list = np.concatenate(points_list, axis=0)
        torch.save(points_list, os.path.join(annotation_folder, "points.pt"))

        command = f"python scripts/cotracker_annotation.py --annotation-path {annotation_folder}"
        if save_video:
            command += " --save-video"
        print(f"running: {command}")
        os.system(command)
    except Exception as e:
        print(f"Error in {annotation_folder}: {e}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default='annotations/human_demo/salt_demo')
    parser.add_argument("--num-track-points", type=int, default=40)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-depth", action="store_true", default=False)
    args = parser.parse_args()

    if args.no_video:
        save_video = False
    else:
        save_video = True
    
    annotate_video(args.annotation_folder, args.num_track_points, save_video=save_video, use_depth=not args.no_depth)

if __name__ == '__main__':
    main()