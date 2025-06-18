import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

import init_path
from okami.plan_generation.utils.misc_utils import *
from okami.plan_generation.algos.tap_segmentation import TAPSegmentation

from okami.plan_generation.utils.o3d_utils import (
    scene_pcd_fn, 
    O3DPointCloud, 
    estimate_rotation, 
    remove_outlier,
    load_reconstruction_info_from_human_demo, 
    project3Dto2D, 
    create_o3d_from_points_and_color,
    transform_point_clouds,
    filter_pcd)

def get_3d_points(input_image, depth, mask, filter=False, remove_outlier_kwargs={"nb_neighbors": 80, "std_ratio": 1}, downsample=True):
    # if object_id is None:
    #     masked_depth = self.input_depth * (self.input_annotation > 0).astype(np.float32)
    # else:
    #     masked_depth = self.input_depth * (self.input_annotation == object_id).astype(np.float32)
    masked_depth = depth * mask.astype(np.float32)
    camera_extrinsics = np.eye(4)
    camera_intrinsics = np.array([[909.83630371,   0.        , 651.97015381], [  0.        , 909.12280273, 376.37097168], [  0.        ,   0.        ,   1.        ]])
    pcd_points, pcd_colors = scene_pcd_fn(
        rgb_img_input=input_image,
        depth_img_input=masked_depth,
        extrinsic_matrix=camera_extrinsics,
        intrinsic_matrix=camera_intrinsics,
        downsample=downsample,
        depth_trunc=10.0,
    )
    # print("number of pcd points =", len(pcd_points))
    if filter:
        pcd_points, pcd_colors = remove_outlier(pcd_points, pcd_colors,
                                                **remove_outlier_kwargs)
    if len(pcd_points) == 0:
        input('warning: len of pcd points after filtering is 0')
    return pcd_points, pcd_colors

def determine_hand_object_contact_3d(img, depth, hand_masks, obj_masks, num_objs):
    """
    img: numpy array (H, W, 3), RGB image
    depth: numpy array (H, W), depth image
    hand_masks: list of numpy array (H, W), hand masks
    obj_masks: list of numpy array (H, W), object masks
    num_objs: int, number of objects
    """
    obj_pcds = []
    for i in range(num_objs):
        try:
            pcd_array, _ = get_3d_points(img, depth, obj_masks == i+1)
            pcd = create_o3d_from_points_and_color(pcd_array)
            obj_pcds.append(pcd)
        except:
            obj_pcds.append(None)
    hand_pcds = []
    for i in range(2):
        try:
            pcd_array, _ = get_3d_points(img, depth, hand_masks[i, :, :, 0])
            pcd = create_o3d_from_points_and_color(pcd_array)
            hand_pcds.append(pcd)
        except:
            hand_pcds.append(None)
    
    contacts = [-1, -1]
    for i in range(2):
        for j in range(num_objs):
            if hand_pcds[i] is None or obj_pcds[j] is None:
                continue

            dists = hand_pcds[i].compute_point_cloud_distance(obj_pcds[j])
            intersections = [d for d in dists if d < 0.05]
            if len(intersections) > 800:
                contacts[i] = j
                break
    # print("contacts =", contacts)

    # if contacts[0] > -1:
    #     print("left hand center translation with relative to object center:")
    #     left_hand_center = np.mean(np.array(hand_pcds[0].points), axis=0)
    #     obj_center = np.mean(np.array(obj_pcds[contacts[0]].points), axis=0)
    #     print(left_hand_center - obj_center)
    # if contacts[1] > -1:
    #     print("right hand center translation with relative to object center:")
    #     right_hand_center = np.mean(np.array(hand_pcds[1].points), axis=0)
    #     obj_center = np.mean(np.array(obj_pcds[contacts[1]].points), axis=0)
    #     print(right_hand_center - obj_center)

    hand_pcd_centers = [np.mean(np.array(hand_pcds[i].points), axis=0) for i in range(2)]

    return contacts, hand_pcd_centers

def determine_hand_object_contact_2d(img, hand_masks, obj_masks, num_objs):
    """
    img: numpy array (H, W, 3), RGB image
    hand_masks: list of numpy array (H, W), hand masks
    obj_masks: list of numpy array (H, W), object masks
    num_objs: int, number of objects
    """
    obj_mask_lst = []
    for i in range(num_objs):
        obj_mask_lst.append(obj_masks == i+1)
    hand_mask_lst = [hand_masks[0, :, :, 0], hand_masks[1, :, :, 0]]

    contacts = [-1, -1]
    for i in range(2):
        for j in range(num_objs):

            obj_pts = np.where(obj_mask_lst[j])
            hand_pts = np.where(hand_mask_lst[i])
            
            # sample no more than 10000 points
            obj_pts_idx = np.random.choice(len(obj_pts[0]), min(7000, len(obj_pts[0])), replace=False)
            hand_pts_idx = np.random.choice(len(hand_pts[0]), min(800, len(hand_pts[0])), replace=False)

            filtered_obj_pts = np.array([obj_pts[0][obj_pts_idx], obj_pts[1][obj_pts_idx]]).T
            filtered_hand_pts = np.array([hand_pts[0][hand_pts_idx], hand_pts[1][hand_pts_idx]]).T
            
            dists = np.concatenate([np.linalg.norm(filtered_obj_pts - hand_pt, axis=1) for hand_pt in filtered_hand_pts])
            intersections = np.sum(dists < 10)

            if intersections > 300:
                contacts[i] = j
                break
    # print("contacts =", contacts)
            
    hand_mask_centers = [np.mean(np.array(np.where(hand_mask_lst[i])), axis=1) for i in range(2)] # TODO: check axis

    return contacts, hand_mask_centers

def main():
    parser = argparse.ArgumentParser(description='Process annotation and detect changepoints.')
    parser.add_argument('--annotation-folder', type=str, help='Path to the annotation file.')
    parser.add_argument("--no-depth", action="store_true", default=False)
    args = parser.parse_args()

    tap_segmentation = TAPSegmentation()
    tap_segmentation.load(args.annotation_folder)

    # load video sequence
    video_seq = get_video_seq_from_annotation(args.annotation_folder)
    
    # load hand masks
    hand_mask_file = os.path.join(args.annotation_folder, 'hamer', "hand_masks.npz")
    hand_masks = np.load(hand_mask_file)['arr_0']
    print("shape of hand_masks =", hand_masks.shape)

    # load object masks
    mask_file = f"{args.annotation_folder}/masks.npz"
    if not os.path.exists(mask_file):
        raise ValueError(f"Mask file {mask_file} does not exist. You need to run XMem annotation first in order to proceed.")
    obj_masks = np.load(mask_file)['arr_0']

    num_objs = obj_masks[0].max()
    print("number of objects =", num_objs)
    
    if not args.no_depth:
        depth_seq = get_depth_seq_from_annotation(args.annotation_folder)

    all_segments_info = []
    for idx, seg in enumerate(tap_segmentation.temporal_segments.segments):
        
        all_res = []
        all_centers = []
        all_vel = []

        # sample several keyframes from each segment
        keyframe_indices = np.linspace(seg.start_idx, seg.end_idx, 35).astype(int)
        keyframe_indices = keyframe_indices[3:32]
        for i in range(len(keyframe_indices)):
            # check hand-object relationship in this frame
            img = video_seq[keyframe_indices[i]]
            hand_mask = hand_masks[keyframe_indices[i]]
            obj_mask = obj_masks[keyframe_indices[i]]
            if not args.no_depth:
                depth = depth_seq[keyframe_indices[i]]
                contacts, hand_centers = determine_hand_object_contact_3d(img, depth, hand_mask, obj_mask, num_objs)
            else:
                contacts, hand_centers = determine_hand_object_contact_2d(img, hand_mask, obj_mask, num_objs)
        
            all_res.append(contacts)
            all_centers.append(hand_centers)

        # calculate velocity from centers
        all_vel = [[0., 0.]]
        for i in range(1, len(all_centers)):
            vel = [0., 0.]
            for j in range(2):
                vel[j] = np.linalg.norm(all_centers[i][j] - all_centers[i-1][j])
            all_vel.append(vel)
        assert(len(all_vel) == len(all_res))

        segment_res = [-1, -1]
        for hand_idx in range(2):
            v = [vel[hand_idx] for vel in all_vel]
            contact = [res[hand_idx] for res in all_res]
            # print("len of v =", len(v), "len of contact =", len(contact))

            # smooth contact list using a window of 5
            smooth_contact = []
            for i in range(len(contact)):
                if i < 2 or i > len(contact) - 3:
                    smooth_contact.append(contact[i])
                else:
                    # pick the majority number among the 5
                    contact_window = contact[i-2:i+3]
                    # find the most frequent number
                    contact_dict = {}
                    for c in contact_window:
                        if c not in contact_dict:
                            contact_dict[c] = 0
                        contact_dict[c] += 1
                    max_contact = -1
                    max_count = 0
                    for key, value in contact_dict.items():
                        if value > max_count:
                            max_contact = key
                            max_count = value
                    smooth_contact.append(max_contact)
            
            # find the contact with largest velocity, and largest apperaence
            contact_vel = {}
            contact_num = {}
            for kpidx, c in enumerate(contact):
                if c not in contact_vel:
                    contact_vel[c] = 0
                    contact_num[c] = 0
                contact_vel[c] += v[kpidx]
                contact_num[c] += 1

            contact_mean_v = {}
            for key, value in contact_vel.items():
                contact_mean_v[key] = value / contact_num[key]

            if hand_idx == 0:
                print("left hand")
            else: print("right hand")
            print("length of smoooth contact =", len(smooth_contact), "len of all vel =", len(all_vel))
            print("contact_vel =", contact_vel, "contact_num =", contact_num)
            print("contact_mean_v =", contact_mean_v)
            print()

            max_contact_v = -1
            max_vel = 0
            for key, value in contact_mean_v.items():
                if value > max_vel:
                    max_contact_v = key
                    max_vel = value
            
            max_contact_n = -1
            max_num = 0
            for key, value in contact_num.items():
                if value > max_num:
                    max_contact_n = key
                    max_num = value

            if max_contact_v == max_contact_n:
                max_contact = max_contact_v
            else:
                if contact_mean_v[max_contact_v] - contact_mean_v[max_contact_n] > 0.4:
                    max_contact = max_contact_v
                elif contact_num[max_contact_v] < 8:
                    max_contact = max_contact_n
                elif max_num > 19:
                    max_contact = max_contact_n
                else:
                    max_contact = max_contact_v

            segment_res[hand_idx] = max_contact


        # num_left = {}
        # num_right = {}
        # for contact in all_res:
        #     if contact[0] not in num_left:
        #         num_left[contact[0]] = 0
        #     if contact[1] not in num_right:
        #         num_right[contact[1]] = 0
        #     num_left[contact[0]] += 1
        #     num_right[contact[1]] += 1

        # # found bigest number of left and right hand
        # max_left = 0
        # max_right = 0
        # segment_res = [-1, -1]
        # for key, value in num_left.items():
        #     if value >= max_left:
        #         max_left = value
        #         segment_res[0] = key
        # for key, value in num_right.items():
        #     if value >= max_right:
        #         max_right = value
        #         segment_res[1] = key

        print(f"Segment {idx} hand info: {segment_res}")
        print()
        print("---------------------------------------------")

        all_segments_info.append(segment_res)
    
    with open(os.path.join(args.annotation_folder, 'hamer_hand_object_contacts.json'), 'w') as f:
        json.dump(all_segments_info, f)

if __name__ == "__main__":
    main()