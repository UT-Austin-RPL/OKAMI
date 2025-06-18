import os
import argparse
import torchvision
from pathlib import Path
from tqdm import tqdm

from PIL import Image
import numpy as np
import cv2
import init_path

from okami.plan_generation.algos.cutie_tracker import CutieTracker
from okami.plan_generation.utils.misc_utils import get_annotation_path, get_first_frame_annotation, VideoWriter, overlay_xmem_mask_on_image, add_palette_on_mask, get_video_seq_from_annotation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-folder", type=str, default="annotations/human_demo/salt_demo")
    return parser.parse_args()


def cutie_annotation(annotation_folder):
    frames = get_video_seq_from_annotation(annotation_folder)

    first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)

    device = "cuda:0"
    cutie_tracker = CutieTracker(device=device)
    cutie_tracker.clear_memory()

    masks = cutie_tracker.track_video(frames, first_frame_annotation)
    np.savez(os.path.join(annotation_folder, "masks.npz"), np.stack(masks, axis=0))

    with VideoWriter(annotation_folder, "cutie_annotation_video.mp4", save_video=True) as video_writer:
        for rgb_img, mask in tqdm(zip(frames, masks), total=len(frames)):
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            overlay_img = overlay_xmem_mask_on_image(bgr_img, mask, rgb_alpha=0.4)
            video_writer.append_image(overlay_img)

def main():

    args = parse_args()
    
    cutie_annotation(args.annotation_folder)

    

if __name__ == "__main__":
    main()