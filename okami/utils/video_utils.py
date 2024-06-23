import json
import os
from pathlib import Path
import argparse
import cv2
import h5py
import numpy as np
import imageio
from deoxys_vision.utils.img_utils import load_depth_in_rgb

class VideoWriter():
    """A wrapper of imageio video writer
    """
    def __init__(self, video_path, video_name=None, fps=30, single_video=True, save_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.single_video = single_video
        self.last_images = {}
        if video_name is None:
            self.video_name = "video.mp4"
        else:
            self.video_name = video_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(self.video_name)

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_image(self, image, idx=0):
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            self.image_buffer[idx].append(image[::-1])

    def append_vector_obs(self, images):
        if self.save_video:
            for i in range(len(images)):
                self.append_image(images[i], i)

    def save(self, video_name=None, flip=True, bgr=False):
        if video_name is None:
            video_name = self.video_name
        img_convention = 1
        color_convention = 1
        if flip:
            img_convention = -1
        if bgr:
            color_convention = -1
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, video_name)
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                    video_writer.close()
        return video_name


def depth_in_rgb(depth_img):
    assert(depth_img.dtype == np.uint16)
    higher_bytes = depth_img >> 8
    lower_bytes = depth_img & 0xFF
    depth_rgb_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3)).astype(np.uint8)
    depth_rgb_img[..., 1] = higher_bytes.astype(np.uint8)
    depth_rgb_img[..., 2] = lower_bytes.astype(np.uint8)
    return depth_rgb_img

def rgb_to_depth(rgb_img):
    assert(rgb_img.dtype == np.uint8)
    depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1])).astype(np.uint16)
    depth_img = rgb_img[..., 1].astype(np.uint16) << 8
    depth_img += rgb_img[..., 2].astype(np.uint16)
    return depth_img

def load_depth_in_rgb(rgb_img):
    depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1])).astype(np.uint16)
    depth_img = rgb_img[..., 1].astype(np.uint16) << 8 | rgb_img[..., 2].astype(np.uint16)
    return depth_img