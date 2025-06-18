"""Miscellaneous utility functions."""

import io
import os
import cv2
import imageio
import h5py
import json
import yaml
import random
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

# import robosuite.utils.transform_utils as T
import traceback
import time
import torch
import inspect
import signal

from PIL import Image
from scipy.ndimage import binary_erosion
from okami.plan_generation.utils.log_utils import get_orion_logger

ORION_LOGGER = get_orion_logger("okami")

try:
    import robosuite.macros as macros
except:
    ORION_LOGGER.warning("Warning: robosuite is not installed. Macros will not be available.")

davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'

youtube_palette = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\xabyg\xff\xff\xffes~\x0b\x0b\x0b\x0c\x0c\x0c\r\r\r\x0e\x0e\x0e\x0f\x0f\x0f'

def get_annotation_path(dataset_name, parent_folder="datasets/annotations"):
    dataset_folder_name = dataset_name.split("/")[-1].replace(".hdf5", "")
    annotations_folder = os.path.join(parent_folder, dataset_folder_name)
    return annotations_folder

def get_annotation_info(annotations_folder):
    assert(os.path.exists(annotations_folder)), f"Annotation folder {annotations_folder} does not exist."
    with open(os.path.join(annotations_folder, "config.json"), "r") as f:
        config_dict = json.load(f)
    return config_dict

def get_dataset_name_from_annotation(annotations_folder):
    config_dict = get_annotation_info(annotations_folder)
    return config_dict["original_file"]

def get_tracked_points_annotation(annotation_path):
    results = torch.load(os.path.join(annotation_path, "tracked_points.pt"))
    # pred_tracks: [B, T, N, 2]
    # pred_visibility: [B, T, N]
    return results

def get_smplh_traj_annotation(annotation_path):
    with open(os.path.join(annotation_path, "smplh_traj.pkl"), "rb") as f:
        results = pickle.load(f)
    return results

def get_object_names_from_annotation(annotation_path):
    with open(os.path.join(annotation_path, "text_description.json"), "r") as f:
        text_description = json.load(f)["objects"]
    return text_description

def get_hand_object_contacts_annotation(annotation_path):
    with open(os.path.join(annotation_path, "hand_object_contacts.json"), "r") as f:
        hand_object_contacts = json.load(f)
    return hand_object_contacts

def get_hamer_hand_object_contacts_annotation(annotation_path):
    with open(os.path.join(annotation_path, "hamer_hand_object_contacts.json"), "r") as f:
        hand_object_contacts = json.load(f)
    return hand_object_contacts

def get_optical_flow_annotation(annotation_path):
    results = torch.load(os.path.join(annotation_path, "dense_trajs.pt"))
    # pred_tracks: [B, T, N, 2]
    # pred_visibility: [B, T, N]
    return results

def get_hamer_result(annotation_path):
    hamer_result_path = os.path.join(annotation_path, "hamer_output")
    hamer_result = torch.load(os.path.join(hamer_result_path, "hand_keypoints.pt"))
    return hamer_result, hamer_result_path

def get_video_seq_from_annotation(annotation_path, bgr=False):
    config_dict = get_annotation_info(annotation_path)
    assert(config_dict["mode"] == "human_demo"), f"Annotation folder {annotation_path} is not a human demo."
    demo_file_path = config_dict["original_file"]
    with h5py.File(demo_file_path, "r") as f:
        video_seq = f["data/human_demo/obs/agentview_rgb"][()]
    if bgr:
        video_seq = video_seq[:, :, :, ::-1]

    return video_seq

def get_camera_intrinsics_from_annotation(annotation_path):
    info = get_annotation_info(annotation_path)
    camera_intrinsics = info["intrinsics"]
    return camera_intrinsics
    
def get_depth_seq_from_human_demo(dataset_name, start_idx=0, end_idx=None):
    with h5py.File(dataset_name, "r") as f:
        depth_seq = f["data/human_demo/obs/agentview_depth"].astype(np.float32)
        if end_idx is None:
            depth_seq = depth_seq[start_idx:]
        else:
            depth_seq = depth_seq[start_idx:end_idx]
    return depth_seq

def get_image_seq_from_human_demo(dataset_name, start_idx=0, end_idx=None):
    with h5py.File(dataset_name, "r") as f:
        image_seq = f["data/human_demo/obs/agentview_rgb"][()]
        if end_idx is None:
            image_seq = image_seq[start_idx:]
        else:
            image_seq = image_seq[start_idx:end_idx]
    return image_seq

def get_depth_seq_from_annotation(annotation_path, start_idx=0, end_idx=None):
    info = get_annotation_info(annotation_path)
    dataset_name = info["original_file"]
    return get_depth_seq_from_human_demo(dataset_name, start_idx=start_idx, end_idx=end_idx)


def get_image_seq_from_annotation(annotation_path, start_idx=0, end_idx=None):
    info = get_annotation_info(annotation_path)
    dataset_name = info["original_file"]
    return get_image_seq_from_human_demo(dataset_name, start_idx=start_idx, end_idx=end_idx)

def get_first_frame_annotation(annotations_folder):
    """A helper function to get the first frame and its annotation from the specified annotations folder.

    Args:
        annotations_folder (str): path to where the annotation is stored

    Returns:
        image, annotated_mask
    """
    first_frame = cv2.imread(os.path.join(annotations_folder, "frame.jpg"))[:, :, ::-1]
    first_frame_annotation = np.array(Image.open((os.path.join(annotations_folder, "frame_annotation.png"))))
    # Resize first_frame to first_frame_annotation if shape does not match
    if first_frame.shape[0] != first_frame_annotation.shape[0] or first_frame.shape[1] != first_frame_annotation.shape[1]:
        first_frame = cv2.resize(first_frame, (first_frame_annotation.shape[1], first_frame_annotation.shape[0]), interpolation=cv2.INTER_AREA)
    return first_frame, first_frame_annotation

def get_overlay_video_from_dataset(dataset_name, demo_idx=None, palette=davis_palette, video_name="overlay_video.mp4", flip=True):
    annotations_folder = get_annotation_path(dataset_name)
    video_path = annotations_folder

    with h5py.File(dataset_name, "r") as original_dataset, h5py.File(os.path.join(annotations_folder, "masks.hdf5"), "r") as mask_dataset:
        demo_keys = [f"demo_{demo_idx}" if demo_idx is not None else demo for demo in original_dataset["data"].keys()]
        overlay_images = []
        for demo in demo_keys:
            images = original_dataset[f"data/{demo}/obs/agentview_rgb"][()]
            masks = mask_dataset[f"data/{demo}/obs/agentview_masks"][()]
            for (image, mask) in zip(images, masks):
                colored_mask = Image.fromarray(mask)
                colored_mask.putpalette(palette)
                colored_mask = np.array(colored_mask.convert("RGB"))
                # resize image to colored_mask
                image = cv2.resize(image, (colored_mask.shape[1], colored_mask.shape[0]), interpolation=cv2.INTER_AREA)
                overlay_img = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
                overlay_images.append(overlay_img)

        video_writer = VideoWriter(video_path=video_path, fps=30, save_video=True)
        for overlay_img in overlay_images:
            video_writer.append_image(overlay_img)
        video_writer.save(video_name, flip=flip)
    return video_path

def get_first_frame_annotation_from_dataset(dataset_name):
    with h5py.File(dataset_name, "r") as dataset:
        first_frame = dataset["annotation"]["first_frame"][()]
        first_frame_annotation = dataset["annotation"]["first_frame_annotation"][()]
    return first_frame, first_frame_annotation

def get_palette(palette="davis"):
    if palette == "davis":
        return davis_palette
    elif palette == "youtube":
        return youtube_palette
    
def convert_convention(image, real_robot=True):
    if not real_robot:
        if macros.IMAGE_CONVENTION == "opencv":
            return np.ascontiguousarray(image[::1])
        elif macros.IMAGE_CONVENTION == "opengl":
            return np.ascontiguousarray(image[::-1])
    else:
        # return np.ascontiguousarray(image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return np.ascontiguousarray(image)

def overlay_xmem_mask_on_image(rgb_img, mask, use_white_bg=False, rgb_alpha=0.7):
    """

    Args:
        rgb_img (np.ndarray):rgb images
        mask (np.ndarray)): binary mask
        use_white_bg (bool, optional): Use white backgrounds to visualize overlap. Note that we assume mask ids 0 as the backgrounds. Otherwise the visualization might be screws up. . Defaults to False.

    Returns:
        np.ndarray: overlay image of rgb_img and mask
    """
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    if use_white_bg:
        colored_mask[mask == 0] = [255, 255, 255]
    overlay_img = cv2.addWeighted(rgb_img, rgb_alpha, colored_mask, 1-rgb_alpha, 0)

    return overlay_img

def mask_to_rgb(mask):
    """Make sure this mask is directly taken from `xmem_tracker.track`"""
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    return colored_mask

def depth_to_rgb(depth_image, colormap="jet"):
    # Normalize depth values between 0 and 1
    normalized_depth = cv2.normalize(depth_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # Apply a colormap to the normalized depth image
    if colormap == "jet":
        colormap = cv2.COLORMAP_JET
    elif colormap == "magma":
        colormap = cv2.COLORMAP_MAGMA
    elif colormap == "viridis":
        colormap = cv2.COLORMAP_VIRIDIS
    else:
        raise ValueError(f"Unknown colormap: {colormap}. Please choose from 'jet', 'magma', or 'viridis'")

    depth_colormap = cv2.applyColorMap(np.uint8(normalized_depth * 255), colormap)
    return depth_colormap

def load_depth(depth_img_name):
    return cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED)

def save_depth(depth_img_name, depth_img):
    assert(depth_img_name.endswith(".tiff")), "You are not using tiff file for saving uint16 data. Things will be screwed."
    cv2.imwrite(depth_img_name, cv2.cvtColor(depth_img, cv2.CV_16U))

def load_depth_in_rgb(depth_img_name):
    rgb_img = cv2.imread(depth_img_name).astype(np.uint8)
    
    depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1])).astype(np.uint16)
    depth_img = rgb_img[..., 1].astype(np.uint16) << 8 | rgb_img[..., 2].astype(np.uint16)

    return depth_img

def save_depth_in_rgb(depth_img_name, depth_img):
    """
    Saving depth image in the format of rgb images. The nice thing is that we can leverage the efficient PNG encoding to save almost 50% spaces compared to using tiff.
    """
    assert(depth_img.dtype == np.uint16)
    assert(depth_img_name.endswith(".png")), "You are not using lossless saving. Depth image will be messed up if you want to use rgb format."
    higher_bytes = depth_img >> 8
    lower_bytes = depth_img & 0xFF
    depth_rgb_img = np.zeros((depth_img.shape[0], depth_img.shape[1], 3)).astype(np.uint8)
    depth_rgb_img[..., 1] = higher_bytes.astype(np.uint8)
    depth_rgb_img[..., 2] = lower_bytes.astype(np.uint8)
    cv2.imwrite(depth_img_name, depth_rgb_img)

def set_pillow_image_alpha(img, alpha):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    assert(img.mode == "RGBA"), f"Image mode must be 'RGBA', but got {img.mode} instead"
    alpha = int(alpha * 255)
    img.putalpha(alpha)
    return img

def add_palette_on_mask(mask_img, palette="davis", alpha=1.0):
    if isinstance(mask_img, np.ndarray):
        assert(len(mask_img.shape) == 2 or mask_img.shape[2] == 1), f"The mask image needs to be a single channel image, but got shape {mask_img.shape} instead"
        mask_img = Image.fromarray(mask_img)
    
    assert(len(np.array(mask_img).shape) == 2 or np.array(mask_img).shape[2] == 1), f"The mask image needs to be a single channel image, but got shape {mask_img.shape} instead"
    # copy mask_img
    new_mask_img = mask_img.copy()
    if palette == "davis":
        new_mask_img.putpalette(get_palette(palette="davis"))
    elif palette == "youtube":
        new_mask_img.putpalette(get_palette(palette="youtube"))
    else:
        raise ValueError(f"Unknown palette: {palette}. Please choose from 'davis' or 'youtube'")
    new_mask_img = new_mask_img.convert("RGBA")
    if alpha < 1.0:
        new_mask_img = set_pillow_image_alpha(new_mask_img, alpha)
    return new_mask_img

def resize_image_to_same_shape(source_img, reference_img=None, reference_size=None):
    # if source_img is larger than reference_img
    if reference_img is None and reference_size is None:
        raise ValueError("Either reference_img or reference_size must be specified.")
    if reference_img is not None:
        reference_size = (reference_img.shape[1], reference_img.shape[0])
    if source_img.shape[0] >  reference_size[0] or source_img.shape[1] > reference_size[1]:
        result_img = cv2.resize(source_img, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_NEAREST)
    else:
        result_img = cv2.resize(source_img, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_NEAREST)
    return result_img

def plotly_draw_image(image, 
                      width=300,
                      height=300,
                      offline=False):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=width,   # you can adjust this as needed
        height=height,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )
    if not offline:
        fig.show()
    else:
        return fig

def plotly_draw_seg_image(image, 
                          mask,
                          width=300,
                          height=300,
                          offline=False):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        mask (_type_): _description_
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)

    fig.data[0].customdata = mask
    # fig.data[0].hovertemplate = '<b>Mask ID:</b> %{customdata}'
    fig.data[0].hovertemplate = 'x: %{x}<br>y: %{y}<br>Mask ID: %{customdata}'

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=width,   # you can adjust this as needed
        height=height,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )

    if not offline:
        fig.show()
    else:
        return fig

def plotly_draw_image_with_points(image, 
                                  points,
                                  width=300,
                                  height=300,
                                  offline=False):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        point (_type_): _description_
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)

    # fig.data[0].hovertemplate = '<b>Mask ID:</b> %{customdata}'
    fig.data[0].hovertemplate = 'x: %{x}<br>y: %{y}'
    if isinstance(points, list):
        for point in points:
            fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers',
            marker=dict(size=10)))  # You can change the color and size
    else:
        fig.add_trace(go.Scatter(x=[points[0]], y=[points[1]], mode='markers',
                      marker=dict(color='red', size=10)))  # You can change the color and size
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=width,   # you can adjust this as needed
        height=height,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )
    if not offline:
        fig.show()
    else:
        return fig

def plotly_draw_image_with_object_keypoints(image, 
                                            keypoints,
                                            width=300,
                                            height=300,
                                            offline=False,
                                            default_point_color=None):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        keypoints (_type_): (NUM_OBJECTS, NUM_KEYPOINTS, 2) array of keypoints
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)

    # fig.data[0].hovertemplate = '<b>Mask ID:</b> %{customdata}'
    fig.data[0].hovertemplate = 'x: %{x}<br>y: %{y}'

    marker_dict = dict(size=10)
    if default_point_color is not None:
        marker_dict["color"] = default_point_color
    for point in keypoints:
        fig.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode='markers',
        marker=marker_dict))  # You can change the color and size
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            width=width,   # you can adjust this as needed
            height=height,   # you can adjust this as needed
            margin=dict(l=0, r=0, b=0, t=0)
        )
    if not offline:
        fig.show()      
    else:
        return fig

def plotly_draw_image_with_hand_keypoints(image, 
                                  hand_keypoints,
                                  width=300,
                                  height=300,
                                  save_to_buffer=False,
                                  select_indices=None,
                                  offline=False):
    """Draw segmentation image using plotly

    Args:
        image (_type_): _description_
        point (_type_): _description_
        width (int, optional): _description_. Defaults to 300.
        height (int, optional): _description_. Defaults to 300.
    """
    fig = px.imshow(image)

    # fig.data[0].hovertemplate = '<b>Mask ID:</b> %{customdata}'
    fig.data[0].hovertemplate = 'x: %{x}<br>y: %{y}'

    if select_indices is None:
        select_indices = list(range(21))

    wrist_idx = 0
    thumb_indices = [1, 2, 3, 4]
    index_indices = [5, 6, 7, 8]
    middle_indices = [9, 10, 11, 12]
    ring_indices = [13, 14, 15, 16]
    pinky_indices = [17, 18, 19, 20]
    indices_dict = {
        "thumb": thumb_indices,
        "index": index_indices,
        "middle": middle_indices,
        "ring": ring_indices,
        "pinky": pinky_indices
    }
    colors_dict = {
        "wrist": "red",
        "thumb": "blue",
        "index": "green",
        "middle": "yellow",
        "ring": "orange",
        "pinky": "purple"
    }
    fig.add_trace(go.Scatter(x=[hand_keypoints[wrist_idx, 0]], y=[hand_keypoints[wrist_idx, 1]], mode='markers',
                      marker=dict(color=colors_dict["wrist"], size=10)))
    for finger_name in indices_dict.keys():
        indices = indices_dict[finger_name]

        x = []
        y = []
        for idx in indices:
            if idx not in select_indices:
                continue
            x.append(hand_keypoints[idx, 0])
            y.append(hand_keypoints[idx, 1])
        if len(x) > 0:
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                        marker=dict(color=colors_dict[finger_name], size=10),
                        line=dict(color=colors_dict[finger_name], width=5)
                        ))
  
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=width,   # you can adjust this as needed
        height=height,   # you can adjust this as needed
        margin=dict(l=0, r=0, b=0, t=0)
    )

    if save_to_buffer:
        img_buf = io.BytesIO()
        fig.write_image(img_buf, format="png")
        img_buf.seek(0)
        return img_buf
    
    if not offline:
        fig.show()
    else:
        return fig

def plotly_draw_image_correspondences(image1, points1, image2, points2, max_width=1000, max_height=1000, offline=False):
    """Draw two images side by side with point correspondences using plotly.

    Args:
        image1: First image.
        points1: Points on the first image.
        image2: Second image.
        points2: Points on the second image.
    """
    # Dimensions of the images
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    ORION_LOGGER.debug(f"first image size: {height1} x {width1}, second image size: {height2} x {width2}")

    # Create a blank canvas
    max_height = max(height1, height2)
    total_width = width1 + width2
    blank_canvas = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255

    # Place images side by side on the canvas
    blank_canvas[:height1, :width1, :] = image1
    blank_canvas[:height2, width1:width1+width2, :] = image2

    # Create plotly figure
    fig = px.imshow(blank_canvas)

    # Add points and lines
    for p1, p2 in zip(points1, points2):
        fig.add_trace(go.Scatter(x=[p1[0], p2[0] + width1], y=[p1[1], p2[1]], mode='lines+markers',
                                 line=dict(
                                            width=2, 
                                            # color='green'
                                            ),
                                 marker=dict(
                                        # color='red', 
                                        size=10)))

    # Update layout
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=min(max_width, total_width),
        height=min(max_height, max(height1, height2)),
        margin=dict(l=0, r=0, b=0, t=0)
)    
    if not offline:
        fig.show()
    else:
        return fig

def plotly_draw_3d_pcd(pcd_points, pcd_colors=None, addition_points=None, marker_size=3, equal_axis=True, title="", offline=False, no_background=False, default_rgb_str="(255,0,0)",additional_point_draw_lines=False, uniform_color=False):

    if pcd_colors is None:
        color_str = [f'rgb{default_rgb_str}' for _ in range(pcd_points.shape[0])]
    else:
        color_str = ['rgb('+str(r)+','+str(g)+','+str(b)+')' for r,g,b in pcd_colors]

    # Extract x, y, and z columns from the point cloud
    x_vals = pcd_points[:, 0]
    y_vals = pcd_points[:, 1]
    z_vals = pcd_points[:, 2]

    # Create the scatter3d plot
    rgbd_scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(size=3, color=color_str, opacity=0.8)
    )
    data = [rgbd_scatter]
    if addition_points is not None:
        assert(addition_points.shape[-1] == 3)
        # check if addition_points are three dimensional
        if len(addition_points.shape) == 2:
            addition_points = [addition_points]
        for points in addition_points:
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            if additional_point_draw_lines:
                mode = "lines+markers"
            else:
                mode = "markers"
            marker_dict = dict(size=marker_size,
                                opacity=0.8)
            
            if uniform_color:
                marker_dict["color"] = f'rgb{default_rgb_str}'
            rgbd_scatter2 = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode=mode,
                marker=marker_dict,
                )
            data.append(rgbd_scatter2)

    if equal_axis:
        scene_dict = dict(   
            aspectmode='data',  
        )
    else:
        scene_dict = dict()
    # Set the layout for the plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        # axes range
        scene=scene_dict,
        title=dict(text=title, automargin=True)
    )

    fig = go.Figure(data=data, layout=layout)

    if no_background:
        fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
            yaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
            zaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        margin=dict(l=0, r=0, b=0, t=0),  # No margins
        showlegend=False,
    )


    if not offline:
        fig.show()
    else:
        return fig


def rotate_camera_pose(camera_pose, angle=-10, point=[0, 0, 0], axis=[0, 0, 1]):   
    """Rotating a camera pose

    Args:
        camera_pose (_type_): _description_
        angle (int, optional): _description_. Defaults to -10.
        point (list, optional): _description_. Defaults to [0, 0, 0].
        axis (list, optional): _description_. Defaults to [0, 0, 1].

    Returns:
        _type_: _description_
    """
    rad = np.pi * angle / 180.0

    homo_rot_z = T.make_pose(np.array([0., 0., 0.]), T.quat2mat(T.axisangle2quat(np.array(axis) * rad)))
    
    new_camera_pose = camera_pose.copy()
    new_camera_pose[:3, 3] = new_camera_pose[:3, 3] - np.array(point)
    new_camera_pose = homo_rot_z @ new_camera_pose
    new_camera_pose[:3, 3] = new_camera_pose[:3, 3] + np.array(point)

    return new_camera_pose

def get_intrinsics_matrix_from_dict(camera_intrinsics_dict):
    camera_intrinsics = np.zeros((3, 3))
    camera_intrinsics[0, 0] = camera_intrinsics_dict["fx"]
    camera_intrinsics[1, 1] = camera_intrinsics_dict["fy"]
    camera_intrinsics[0, 2] = camera_intrinsics_dict["cx"]
    camera_intrinsics[1, 2] = camera_intrinsics_dict["cy"]
    camera_intrinsics[2, 2] = 1
    return camera_intrinsics

def get_extrinsics_matrix_from_dict(camera_extrinsics_dict):
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, 3] = np.array(camera_extrinsics_dict["translation"]).reshape(3,)
    camera_extrinsics[:3, :3] = np.array(camera_extrinsics_dict["rotation"]).reshape(3, 3)
    return camera_extrinsics


def get_transformed_depth_img(point_cloud, 
                              camera_intrinsics,
                              new_camera_extrinsics,
                              camera_width,
                              camera_height):
    """Transform point clouds to get a new depth image under a given camera extrinsics.

    Args:
        point_cloud (_type_): _description_
        camera_intrinsics (_type_): _description_
        new_camera_extrinsics (_type_): _description_
        camera_width (_type_): _description_
        camera_height (_type_): _description_

    Returns:
        _type_: _description_
    """

    new_point_cloud = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=-1)
    new_point_cloud = np.linalg.inv(new_camera_extrinsics) @ new_point_cloud.T

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    z = new_point_cloud[2, :] * 1000
    u = fx * new_point_cloud[0, :] * 1000 / z + cx
    v = fy * new_point_cloud[1, :] * 1000 / z + cy

    px = u.astype(np.int32)
    py = v.astype(np.int32)

    new_depth_img = np.ones((camera_height, camera_width)) * z.max()
    sorted_indices = np.argsort(z)[::-1]

    px = px[sorted_indices]
    py = py[sorted_indices]
    z = z[sorted_indices]

    valid_indices = (px >= 0) & (px < camera_width) & (py >= 0) & (py < camera_height)

    new_depth_img[py[valid_indices], px[valid_indices]] = np.minimum(
                                new_depth_img[py[valid_indices], px[valid_indices]], z[valid_indices]
                            )
    new_depth_img = new_depth_img.astype(np.uint16)
    return new_depth_img, z.max()

def edit_h5py_datasets(base_dataset_name, additional_dataset_name, mode="merge  "):
    # load base dataset in an edit mode
    base_dataset = h5py.File(base_dataset_name, "r+")
    additional_dataset = h5py.File(additional_dataset_name, "r")

    # check if the additional dataset is compatible with the base dataset
    for demo in base_dataset["data"].keys():
        assert(demo in additional_dataset["data"].keys()), f"Demo {demo} does not exist in the additional dataset."
    try:
        # merge mode
        if mode == "merge":
            for demo in base_dataset["data"].keys():
                for key in additional_dataset[f"data/{demo}/obs"].keys():
                    assert(key not in base_dataset[f"data/{demo}/obs"].keys()), f"Key {key} already exists in the base dataset."
                    additional_dataset.copy(additional_dataset[f"data/{demo}/obs/{key}"], base_dataset[f"data/{demo}/obs"], key)

        # separate mode
        if mode == "separate":
            for demo in base_dataset["data"].keys():
                for key in additional_dataset[f"data/{demo}/obs"].keys():
                    assert(key in base_dataset[f"data/{demo}/obs"].keys()), f"Key {key} does not exist in the base dataset."
                    del base_dataset[f"data/{demo}/obs/{key}"]
    except Exception as e:
        ORION_LOGGER.error(e)
        base_dataset.close()
        additional_dataset.close()
        raise e

    base_dataset.close()
    additional_dataset.close()
 

def normalize_pcd(obs, max_array, min_array):
    """Normalize the point cloud data.

    Args:
        obs (_type_): _description_
        max_array (_type_): _description_
        min_array (_type_): _description_

    Returns:
        _type_: _description_
    """
    max_array = np.array(max_array, dtype=np.float32)
    min_array = np.array(min_array, dtype=np.float32)
    return (obs - min_array) / (max_array - min_array)


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

    def save(self, video_name=None, flip=True, bgr=True):
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
                video_writer = imageio.get_writer(video_name, fps=self.fps, format='FFMPEG')
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
            ORION_LOGGER.info(f"Saved videos to {video_name}.")
        return video_name



class Timer:
    """A context manager for timing a block of code."""
    def __init__(self, unit="second", verbose=False):
        if unit == "second":
            self.factor = 10 ** 9
        elif unit == "millisecond":
            self.factor = 10 ** 6
        elif unit == "microsecond":
            self.factor = 10 ** 3
        
        self.verbose = verbose
        self.value = None

    def __enter__(self):
        frame = inspect.currentframe()
        self.line_number = frame.f_back.f_lineno
        self.filename = inspect.getframeinfo(frame.f_back).filename
        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time_ns()
        elapsed_time = (end_time - self.start_time) / self.factor
        if self.verbose:
            ORION_LOGGER.info(f"{elapsed_time} seconds to execute in the block of {self.filename}: {self.line_number}.")

        self.value = elapsed_time

        return None        

    def get_elapsed_time(self):
        return self.value
    

# Functionalities for loading images / videos from robot demo data

def load_image_from_hdf5_dataset(
        dataset_name,
        demo_idx,
        frame_idx,
        image_name,
        flip=False,
        bgr=False,
):
    with h5py.File(dataset_name, "r") as dataset:
        image = dataset[f"data/demo_{demo_idx}/obs/{image_name}"][frame_idx]
    
    if flip:
        image = image[::-1]
    if bgr:
        image = image[:, :, ::-1]
    return image

def load_video_from_hdf5_dataset(
        dataset_name,
        demo_idx,
        image_name="agentview_rgb",
        flip=False,
        bgr=False,
):
    with h5py.File(dataset_name, "r") as dataset:
        images = dataset[f"data/demo_{demo_idx}/obs/{image_name}"][()]
    
    if flip:
        images = images[:, ::-1]
    if bgr:
        images = images[:, :, :, ::-1]
    return images

def export_video_from_hdf5_dataset(
        dataset_name,
        demo_idx,
        video_path,
        video_name,
        image_name="agentview_rgb",
        flip=False,
        bgr=False,
):
    video_seq = load_video_from_hdf5_dataset(dataset_name, demo_idx, image_name, flip=flip, bgr=bgr)
    video_writer = VideoWriter(video_path=video_path, video_name=f"{video_name}.mp4", fps=30, save_video=True)
    for image in video_seq:
        video_writer.append_image(image)
    video_path = video_writer.save()
    return video_path

def load_first_frame_from_hdf5_dataset(
        dataset_name,
        demo_idx,
        flip=False,
        bgr=False,
        image_name="agentview_rgb"
):
    return load_image_from_hdf5_dataset(dataset_name, demo_idx, 0, image_name, flip=flip, bgr=bgr)


# Functionalities for loading images / videos from human video demonstration that are saved in a hdf5 format.

def load_image_from_human_hdf5_dataset(
        dataset_name,
        frame_idx,
        image_name,
        flip=False,
        bgr=False,
):
    with h5py.File(dataset_name, "r") as dataset:
        image = dataset[f"data/human_demo/obs/{image_name}"][frame_idx]
    
    if flip:
        image = image[::-1]
    if bgr:
        image = image[:, :, ::-1]
    return image

def load_video_from_human_hdf5_dataset(
        dataset_name,
        demo_idx,
        image_name="agentview_rgb",
        flip=False,
        bgr=False,
):
    with h5py.File(dataset_name, "r") as dataset:
        images = dataset[f"data/human_demo/obs/{image_name}"][()]
    
    if flip:
        images = images[:, ::-1]
    if bgr:
        images = images[:, :, :, ::-1]
    return images

def export_video_from_human_hdf5_dataset(
        dataset_name,
        video_path,
        video_name,
        image_name="agentview_rgb",
        flip=False,
        bgr=False,
):
    video_seq = load_video_from_human_hdf5_dataset(dataset_name, image_name, flip=flip, bgr=bgr)
    video_writer = VideoWriter(video_path=video_path, video_name=f"{video_name}.mp4", fps=30, save_video=True)
    for image in video_seq:
        video_writer.append_image(image)
    video_path = video_writer.save()
    return video_path

def load_multiple_frames_from_human_hdf5_dataset(
        dataset_name,
        num_frames,
        flip=False,
        bgr=False,
        image_name="agentview_rgb",
):
    with h5py.File(dataset_name, "r") as dataset:
        dataset_length = len(dataset[f"data/human_demo/obs/{image_name}"])
        # print("dataset length = ", dataset_length, dataset[f"data/human_demo/obs/{image_name}"].shape)
        frame_ids = np.linspace(0, dataset_length - 1, num_frames).astype(int)
        images = []
        for frame_id in frame_ids:
            image = dataset[f"data/human_demo/obs/{image_name}"][frame_id]
            if flip:
                image = image[::-1]
            if bgr:
                image = image[:, :, ::-1]
            images.append(image)
    
    return images

def load_first_frame_from_human_hdf5_dataset(
        dataset_name,
        flip=False,
        bgr=False,
        image_name="agentview_rgb"
):
    return load_image_from_human_hdf5_dataset(dataset_name, 0, image_name, flip=flip, bgr=bgr)

def sample_points_in_mask(original_segmentation_mask, num_points, object_id=None):
    """
    Sample a specified number of points from within a given segmentation mask.

    :param original_segmentation_mask: A numpy array representing the segmentation mask,
                              where the mask's object is represented by non-zero values.
    :param num_points: The number of points to sample within the mask.
    :return: A list of sampled points (tuples) within the mask.
    """
    # Get the indices of all non-zero elements in the mask (i.e., the object's location)
    if object_id is None:
        segmentation_mask = original_segmentation_mask
    else:
        segmentation_mask = original_segmentation_mask == object_id
    y_indices, x_indices = np.nonzero(segmentation_mask)
    # Check if there are enough unique points in the mask to sample
    if len(y_indices) < num_points:
        raise ValueError(f"Not enough unique points in the mask to sample the requested number of points. Specified to sample {num_points} points, but only found {len(y_indices)} unique points in the mask.")

    # Randomly sample indices
    sampled_indices = random.sample(range(len(y_indices)), num_points)

    # Get the corresponding coordinates
    sampled_points = [(x_indices[i], y_indices[i]) for i in sampled_indices]

    return sampled_points

def sample_points_in_annotation(annotation, num_points):
    sampled_points = {}
    for object_id in range(1, annotation.max() + 1):
        points = sample_points_in_mask(annotation, num_points, object_id=object_id)
        sampled_points[object_id] = points
    return sampled_points

# 3D processing of point clouds

def transform_points(extrinsics_matrix, points):
    """Transform an array of 3d points using a given extrinsics matrix.

    Args:
        extrinsics_matrix (np.ndarray): 4x4 homoegeous matrix
        points (np.ndarray): Nx3 array of 3d points

    Returns:
        np.ndarray: Nx3 array of transformed 3d points
    """
    assert(points.shape[-1] == 3), f"Points must be an Nx3 array, but got {points.shape} instead."
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = extrinsics_matrix @ points.T
    points = points.T[:, :3]
    return points


def simple_filter_outliers(points):
    """
    Replace outlier points (0, 0, 0) in a 3D trajectory with an average of the previous five valid points.
    
    :param points: A numpy array of shape (N, 3) representing the 3D trajectory.
    :return: A numpy array with outliers replaced.
    """
    n, _ = points.shape
    for i in range(n):
        if np.array_equal(points[i], np.zeros(3)):
            valid_points = points[max(0, i-5):i]
            valid_points = valid_points[~np.all(valid_points == 0, axis=1)]
            if valid_points.size > 0:
                points[i] = np.mean(valid_points, axis=0)
            else:
                # If there are not enough previous valid points, keep the outlier as is.
                pass
    return points

def moving_average_filter(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def create_point_clouds_from_keypoints(keypoints, depth_img, camera_intrinsics_matrix):
    """Manual computation of backprojection. This is used to compute every single point clouds based on a list of given keypoints.

    Args:
        keypoints (np.ndarray): (N, 2) keypoints
        depth_img (np.ndarray): HxW depth image
        camera_intrinsics_matrix (np.ndarray): 3x3 camera intrinsics matrix

    Returns:
        _type_: _description_
    """
    z = depth_img.squeeze() / 1000
    if len(depth_img.shape) == 3:
        height, width = depth_img[0].shape[:2]
    else:
        height, width = depth_img.shape[:2]
    x = keypoints[:, 0]
    y = keypoints[:, 1]

    x = np.clip(x, 0, z.shape[1] - 1)
    # print("before clip, y=", y, y.max())
    y = np.clip(y, 0, z.shape[0] - 1)
    # # new_y = y.copy()
    # # new_y[y < 0] = 0
    # # new_y[y >= height] = height - 1
    # for i in range(len(y)):
    #     print("y[i]=", y[i], "height=", height, y[i]>=height)
    #     if y[i] >= height:
    #         print("y[", i, "]=", y[i], "height=", height)
    #         y[i] = height - 1
    #         print("after: y[", i, "]=", y[i], "height=", height)
    # t = new_y.copy()
    # print("after clip, y=", y, y.max())
    # print("height, width = ", height, width, x.max(), y.max())

    new_z = z[y.astype(np.int32), x.astype(np.int32)]
    fx = camera_intrinsics_matrix[0, 0]
    fy = camera_intrinsics_matrix[1, 1]
    cx = camera_intrinsics_matrix[0, 2]
    cy = camera_intrinsics_matrix[1, 2]

    X = (x - cx) * new_z / fx
    Y = (y - cy) * new_z / fy
    Z = new_z

    points = np.ascontiguousarray(np.stack([X, Y, Z], axis=-1))
    points = points.reshape(-1, 3)
    return points


def plotly_offline_visualization(plotly_figs, filename):
    import plotly.offline as pyo
    assert(filename.endswith(".html"))
    if len(plotly_figs) == 1:
        pyo.plot(plotly_figs[0], filename=filename)

    else:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=len(plotly_figs), 
                            cols=1,
                            specs=[[{'type': 'scene'}]] * len(plotly_figs))
        for j, fig_subplot in enumerate(plotly_figs):
            for trace in fig_subplot.data:
                fig.add_trace(trace, row=j+1, col=1)
        # fig.write_html(filename)
        pyo.plot(fig, filename=filename)


def plotly_oog_seq(sequence, offline=False):
        # Color mapping: 0 -> Red, 1 -> Green, 2 -> Blue
    color_map = {0: "black", 1:  '#007FA1', 2: '#FF5A5F', 3: '#00D084'}
    legend_text = {0: 'Null', 1: 'Free', 2: 'Close Move', 3: 'Open Move'}

    if (type(sequence[0]) is not int):
        # we assume the only alternative is the OOGMode sequence.
        sequence = [int(s.value) for s in sequence]
    # Create traces for each segment
    traces = []
    traces = []
    for value in sorted(set(sequence)):
        trace = go.Scatter(
            x=[None],  # Invisible point
            y=[None],
            mode='lines',
            name=legend_text[value],  # Custom text for legend
            line=dict(color=color_map[value], width=6)
        )
        traces.append(trace)
    for i in range(len(sequence) - 1):
        x_values = [i, i + 1]  # X values for the current and next point
        y_values = [0, 0]  # Y values to keep the line horizontal
        color = color_map[sequence[i]]  # Color based on the current point
        
        # Create a trace for the current segment
        trace = go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(color=color, width=6), showlegend=False)
        traces.append(trace)
    
    # Define layout
    layout = go.Layout(
        title='Sequence Visualization',
        xaxis=dict(title='Sequence Index', tickvals=list(range(len(sequence)))),
        yaxis=dict(showticklabels=False, zeroline=False),  # Hide y-axis details
        legend=dict(title='Legend'),  # Customize legend title
        plot_bgcolor='white',  # Set background color to white for clarity
        height=400,  # Smaller figure height to make the layout more concise
        width=600   # Adjust width as neededmargin=dict(l=20, r=20, t=50, b=20),  # Reduced margins

    )
    
    # Create figure and plot
    fig = go.Figure(data=traces, layout=layout)
    if not offline:
        fig.show()
    else:
        return fig




def read_from_runtime_file():
    def load_from_runtime_config():
        with open("experiments/run_time.json") as f:
            run_time_config = json.load(f)
        return run_time_config

    with open("experiments/experiment_cfg.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    task_mapping = cfg["task_mapping"]
    task_id_mapping = cfg["task_id_mapping"]
    model_mapping = cfg["model_mapping"]
    demo_mapping = cfg["demo_mapping"]
    result_mapping = cfg["result_mapping"]

    runtime_config = load_from_runtime_config()
    runtime_folder = runtime_config["runtime_folder"]

    task_id = runtime_folder.split("/")[-3]
    model_id = runtime_folder.split("/")[-2]
    # captialize all letters in model_id
    model_id = model_id.upper()
    run_id = runtime_folder.split("/")[-1]

    task_name_str = task_id_mapping[task_id]

    rollout_name = f"{task_name_str}_{model_id}_{run_id}"

    rollout_prefix = "annotations/rollout"
    rollout_folder = f"{rollout_prefix}/{rollout_name}"
    human_video_annotation_path = f"annotations/human_demo/{demo_mapping[task_id]}"
    return runtime_folder, rollout_folder, human_video_annotation_path



def finish_experiments(runtime_folder, rollout_folder, human_video_annotation_folder):
    with open(f"{runtime_folder}/finish.txt", "w") as f:
        f.write("finish")

    with open(f"{runtime_folder}/info.json", "w") as f:
        json.dump({"human_video_annotation_folder": human_video_annotation_folder,
                     "rollout_folder": rollout_folder,
                   }, f)
    ORION_LOGGER.info(f"Writing {runtime_folder}/info.json")

def convert_to_json_serializable(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy().tolist()
    elif isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    else:
        return data
