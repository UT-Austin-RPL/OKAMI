import cv2
import os
import torch
import math
import json
import numpy as np

from enum import Enum
from okami.plan_generation.utils.misc_utils import (
    get_video_seq_from_annotation,
    get_tracked_points_annotation,
    get_hamer_hand_object_contacts_annotation,
    get_smplh_traj_annotation, 
    get_first_frame_annotation, 
    get_annotation_info,
    get_depth_seq_from_human_demo, 
    get_image_seq_from_human_demo,
    create_point_clouds_from_keypoints,
    transform_points,
    simple_filter_outliers,)

from okami.plan_generation.utils.hand_utils import (
    get_thumb_tip_points,
    get_index_tip_points,
    get_estimate_palm_point
)

from okami.plan_generation.utils.o3d_utils import (
    scene_pcd_fn, 
    O3DPointCloud, 
    estimate_rotation, 
    remove_outlier,
    load_reconstruction_info_from_human_demo
)
from okami.plan_generation.utils.log_utils import get_orion_logger

from okami.plan_generation.algos.retargeter_wrapper import Retargeter

ORION_LOGGER = get_orion_logger("okami")

class HandType(Enum):
    LEFT = 0
    RIGHT = 1

class Hand:
    def __init__(self, lr):
        self.lr = HandType.LEFT if lr == 'left' else HandType.RIGHT
        self.hand_object_contacts = []
        self.grasp_type = None

class SMPLHTraj:
    def __init__(self, ratio=1.0):
        self.smplh_traj = []
        self.video_smplh_ratio = ratio # num_frames_in_video / num_frames_in_smplh_traj
    
    def cal_idx_from_smplh_to_video(self, idx):
        return math.floor(idx * self.video_smplh_ratio)

    def cal_idx_from_video_to_smplh(self, idx):
        return math.floor(idx / self.video_smplh_ratio)
    
    def add_smplh_traj(self, smplh_traj):
        self.smplh_traj = smplh_traj

class HumanNode:
    def __init__(self, ratio=1.0):
        self.left_hand = Hand('left')
        self.right_hand = Hand('right')
        self.important_body_parts = []
        self.smplh_traj = SMPLHTraj(ratio)

    def add_smplh_traj(self, smplh_traj):
        self.smplh_traj.add_smplh_traj(smplh_traj)

class ObjectNode:
    def __init__(self, object_id=-1):
        self.object_id = object_id
        self.point_ids = []
        self.pcd_points = np.array([])
        self.pcd_colors = np.array([])

class PointNode:
    def __init__(self, point_id=-1, object_id=-1):
        self.point_id = point_id
        self.object_id = object_id

        self.pixel_point = np.array([-1, -1])
        self.world_point = np.array([-1, -1, -1])
        self.tracked_pixel_traj = np.array([])
        self.tracked_world_traj = np.array([])
        self.tracked_visibility_traj = np.array([])

class HandObjectInteractionGraph:
    def __init__(self):
        self.human_node = None
        self.object_nodes = []
        self.point_nodes = []
        self.contact_states = []
        
        self.segment_start_idx = -1
        self.segment_end_idx = -1
        self.human_video_annotation_path = ""

        self.smplh_traj = np.array([])

        self.retargeted_ik_traj = np.array([])
        self.grasp_type = [None, None]
        self.moving_arm = None

        self.arms_contact = [-1, -1]
        self.arms_functional = [0, 0]
        self.arms_moving = [0, 0]

        self.camera_extrinsics = np.eye(4)
        self.camera_intrinsics = np.array([
            [909.83630371,   0.        , 651.97015381],
            [  0.        , 909.12280273, 376.37097168],
            [  0.        ,   0.        ,   1.        ],
        ])

        self.representative_images = []
        self.target_translation_btw_objects = np.array([])
        self.target_hand_object_translation = [np.array([]), np.array([])]

        self.segment_type = None

    def get_representative_images(self, num_images=5):
        img_idx = np.linspace(self.segment_start_idx, self.segment_end_idx, num_images, dtype=int)
        video_seq = get_video_seq_from_annotation(self.human_video_annotation_path)
        img_lst = [video_seq[idx] for idx in img_idx]
        return img_lst

    def get_pixel_trajs(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        pixel_trajs = []
        for point_node in selected_point_nodes:
            pixel_trajs.append(point_node.tracked_pixel_traj)
        return np.stack(pixel_trajs, axis=0)
    
    def get_visibility_trajs(self, object_ids=[]):
        if len(object_ids) == 0:
            object_ids = self.object_ids
        selected_point_nodes = self.select_point_nodes(object_ids=object_ids)
        visibility = []
        for point_node in selected_point_nodes:
            visibility.append(point_node.tracked_visibility_traj)
        return np.stack(visibility, axis=0)

    def get_point_list(self, point_nodes=None, mode="pixel"):
        if point_nodes is None:
            # use all point nodes if not specified
            point_nodes = self.point_nodes
        point_list = []
        for point_node in point_nodes:
            if mode == "pixel":
                point_list.append(point_node.pixel_point)
            elif mode == "world":
                point_list.append(point_node.world_point)
        return point_list

    def select_point_nodes(self, object_ids=[]):
        selected_point_nodes = []
        for point_node in self.point_nodes:
            if point_node.object_id in object_ids:
                selected_point_nodes.append(point_node)
        return selected_point_nodes
    
    def get_points_by_objects(self, object_ids=[], mode="pixel"):
        point_nodes = self.select_point_nodes(object_ids=object_ids)
        return self.get_point_list(point_nodes=point_nodes, mode=mode)

    def set_pixel_trajs(self, pixel_trajs):
        assert(pixel_trajs.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_pixel_traj = pixel_trajs[i]

    def set_visibility_trajs(self, visibility_traj):
        assert(visibility_traj.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_visibility_traj = visibility_traj[i]
            point_node.tracked_visibility = visibility_traj[i][0]

    def set_world_trajs(self, world_trajs):
        assert(world_trajs.shape[0] == len(self.point_nodes)), "Number of points must be the same"
        for i, point_node in enumerate(self.point_nodes):
            point_node.tracked_world_traj = world_trajs[i]
            point_node.world_point = world_trajs[i][0]

    def set_manipulate_object_id(self, object_id):
        self.manipulate_object_id = object_id

    def get_manipulate_object_id(self):
        return self.manipulate_object_id
    
    def set_reference_object_id(self, object_id):
        self.reference_object_id = object_id

    def get_reference_object_id(self):
        return self.reference_object_id
    
    def set_segment_type(self, segment_type):
        self.segment_type = segment_type

    def get_segment_type(self):
        return self.segment_type

    def load_depth(self, input_depth):
        self.input_depth = np.squeeze(np.ascontiguousarray(input_depth))
        # update the points
        self.create_world_points()

    def set_camera_extrinsics(self, extrinsics):
        self.camera_extrinsics = extrinsics

    def set_camera_intrinsics(self, intrinsics):
        self.camera_intrinsics = intrinsics

    def get_objects_3d_points(self, object_id=None, filter=True, remove_outlier_kwargs={"nb_neighbors": 40, "std_ratio": 0.7}, downsample=True):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"

        if object_id is None:
            masked_depth = self.input_depth * (self.input_annotation > 0).astype(np.float32)
        else:
            masked_depth = self.input_depth * (self.input_annotation == object_id).astype(np.float32)
        pcd_points, pcd_colors = scene_pcd_fn(
            rgb_img_input=self.input_image,
            depth_img_input=masked_depth,
            extrinsic_matrix=self.camera_extrinsics,
            intrinsic_matrix=self.camera_intrinsics,
            downsample=downsample,
        )
        if filter:
            pcd_points, pcd_colors = remove_outlier(pcd_points, pcd_colors,
                                                    **remove_outlier_kwargs)
        return pcd_points, pcd_colors
    
    def get_objects_2d_image(self, object_id=None):
        if object_id is None:
            mask = np.expand_dims((self.input_annotation > 0), axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
        else:
            mask = np.expand_dims((self.input_annotation == object_id), axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
        masked_rgb = (self.input_image * mask).astype(np.uint8)
        return masked_rgb
    
    def create_world_points(self):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"
        points = create_point_clouds_from_keypoints(
            np.array(self.get_point_list(mode="pixel")),
            self.input_depth[..., np.newaxis],
            self.camera_intrinsics,
        )
        for i in range(len(self.point_nodes)):
            self.point_nodes[i].world_point = transform_points(self.camera_extrinsics, points[i:i+1])

    def estimate_plane_rotation(self, 
                       depth_trunc=5.0,
                       z_up=True,
                       plane_estimation_kwargs={
                            "ransac_n": 3,
                            "num_iterations": 1000,
                            "distance_threshold": 0.01
                       }):
        assert(self.camera_extrinsics is not None), "Camera extrinsics not set"
        assert(self.camera_intrinsics is not None), "Camera intrinsics not set"
        assert(self.input_depth is not None), "Depth not loaded"
        
        o3d_pcd = O3DPointCloud()
        o3d_pcd.create_from_rgbd(
            self.input_image,
            self.input_depth,
            self.camera_intrinsics,
            depth_trunc=depth_trunc,
        )

        plane_estimation_result = o3d_pcd.plane_estimation(**plane_estimation_kwargs)
        
        T_xy_plane_align = estimate_rotation(plane_estimation_result["plane_model"], z_up=z_up)
        return T_xy_plane_align, plane_estimation_result
    
    def compute_world_trajs(self, 
                            tracked_pixel_trajs, 
                            depth_seq, 
                            camera_intrinsics_matrix, 
                            camera_extrinsics_matrix):
        points_list = []
        print("tracked_pixel_trajs.shape", tracked_pixel_trajs.shape, depth_seq.shape)
        for t in range(tracked_pixel_trajs.shape[1]):
            points = create_point_clouds_from_keypoints(tracked_pixel_trajs[:, t], depth_seq[t], camera_intrinsics_matrix)
            points_list.append(points)
        points_list = np.stack(points_list, axis=0)
        world_trajs = []
        for i in range(tracked_pixel_trajs.shape[0]):
            segment = points_list[:, i]
            segment = simple_filter_outliers(segment)
            segment = transform_points(camera_extrinsics_matrix, segment)
            world_trajs.append(segment)
        return np.stack(world_trajs, axis=0)

    @property
    def segment_length(self):
        return self.segment_end_idx - self.segment_start_idx + 1

    def create_from_human_video(self, 
                                human_video_annotation_path, 
                                segment_start_idx, 
                                segment_end_idx, 
                                segment_idx, 
                                extrinsics,
                                calibrate_grasp,
                                zero_pose_name,
                                video_smplh_ratio=1.0,
                                use_smplh=True):
        self.segment_start_idx = segment_start_idx
        self.segment_end_idx = segment_end_idx
        self.human_video_annotation_path = human_video_annotation_path

        is_first_frame = (segment_start_idx == 0)
        if is_first_frame:
            _, human_annotation = get_first_frame_annotation(human_video_annotation_path)
        else:
            mask_file = f"{human_video_annotation_path}/masks.npz"
            if not os.path.exists(mask_file):
                raise ValueError(f"Mask file {mask_file} does not exist. You need to run XMem annotation first in order to proceed.")
            masks = np.load(mask_file)['arr_0']
            human_annotation = masks[segment_start_idx]

        self.representative_images = self.get_representative_images()

        tap_results = get_tracked_points_annotation(human_video_annotation_path)
        pred_tracks, pred_visibility = tap_results["pred_tracks"], tap_results["pred_visibility"]

        total_points = pred_tracks.shape[2]
        num_points_per_object = total_points // human_annotation.max()
        print("number of objects=", human_annotation.max())

        sampled_points = {}
        tracked_trajs = {}
        visibility_trajs = {}
        for object_id in range(1, human_annotation.max()+1):
            points_per_object = pred_tracks[0, segment_start_idx, (object_id - 1) * num_points_per_object: object_id * num_points_per_object, :2]
            sampled_points[object_id] = points_per_object.detach().cpu().numpy()
            tracked_trajs[object_id] = pred_tracks[0, :, (object_id - 1) * num_points_per_object: object_id * num_points_per_object, :2].detach().cpu().permute(1, 0, 2).numpy()
            visibility_trajs[object_id] = pred_visibility[0, :, (object_id - 1) * num_points_per_object: object_id * num_points_per_object].detach().cpu().permute(1, 0).numpy()
        self.object_ids = list(range(1, human_annotation.max()+1))

        config_info = get_annotation_info(human_video_annotation_path)
        human_demo_dataset_name = config_info["original_file"]
        image_seg_seq = get_image_seq_from_human_demo(human_demo_dataset_name, start_idx=segment_start_idx, end_idx=segment_end_idx)
        depth_seg_seq = get_depth_seq_from_human_demo(human_demo_dataset_name, start_idx=segment_start_idx, end_idx=segment_end_idx)
        self.input_image = np.ascontiguousarray(image_seg_seq[0])
        self.input_annotation = human_annotation

        # create object nodes and point nodes
        for object_id in range(1, human_annotation.max()+1):
            object_node = ObjectNode(object_id)
            object_node.point_ids = list(range((object_id - 1) * num_points_per_object, object_id * num_points_per_object))
            
            point_idx = (object_id - 1) * num_points_per_object
            for point in sampled_points[object_id]:
                point_node = PointNode()
                point_node.point_id = point_idx
                point_node.pixel_point = point
                point_node.object_id = object_id
                self.point_nodes.append(point_node)
                point_idx += 1
        
        # set camera intrinsics and esitimate camera extrinsics
        recon_info = load_reconstruction_info_from_human_demo(human_demo_dataset_name)
        self.set_camera_intrinsics(recon_info["intrinsics"])
        self.load_depth(depth_seg_seq[0])
        if is_first_frame:
            T_xy_plane_align, _ = self.estimate_plane_rotation(z_up=False,
                                                               depth_trunc=5.0,
                                                               plane_estimation_kwargs={
                                                                    "ransac_n": 3,
                                                                    "num_iterations": 1000,
                                                                    "distance_threshold": 0.01
                                                                })
            self.set_camera_extrinsics(T_xy_plane_align)
        else:
            self.set_camera_extrinsics(extrinsics)

        # Add trajectory for point nodes
        pixel_trajs = tap_results["pred_tracks"].squeeze().permute(1, 0, 2).detach().cpu().numpy()
        # get the visibility of point trajectories
        visibility = tap_results["pred_visibility"].squeeze().permute(1, 0).detach().cpu().numpy()

        print("previously, pixel_trajs.shape", pixel_trajs.shape, segment_start_idx, segment_end_idx)
        pixel_trajs = pixel_trajs[:, segment_start_idx:segment_end_idx]
        visibility = visibility[:, segment_start_idx:segment_end_idx]
        self.set_pixel_trajs(pixel_trajs)
        self.set_visibility_trajs(visibility)
        world_trajs = self.compute_world_trajs(pixel_trajs, depth_seg_seq, self.camera_intrinsics, self.camera_extrinsics)
        self.set_world_trajs(world_trajs)

        print("set points trajectory ok!")

        # create human node
        self.human_node = HumanNode(video_smplh_ratio)

        # hand keypoints
        hand_kpts_file = os.path.join(human_video_annotation_path, 'hamer', "vitpose_res.npz")
        vitpose_detections = np.load(hand_kpts_file)['arr_0']
        vitpose_detections = vitpose_detections[segment_start_idx:segment_end_idx]
        vitpose_detections = vitpose_detections.squeeze(1)
        
        # get the detection of first frame
        vitpose_detections = vitpose_detections[0, :, :, :2]
        # print("shape of vitpose_detections", vitpose_detections.shape)

        self.thumb_tip_points = [
            get_thumb_tip_points(vitpose_detections[0], depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics),
            get_thumb_tip_points(vitpose_detections[1], depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics)
        ]
        self.index_tip_points = [
            get_index_tip_points(vitpose_detections[0], depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics),
            get_index_tip_points(vitpose_detections[1], depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics)
        ]
        self.estimate_thumb_point = [np.mean(self.thumb_tip_points[0], axis=0), np.mean(self.thumb_tip_points[1], axis=0)]
        self.estimate_index_point = [np.mean(self.index_tip_points[0], axis=0), np.mean(self.index_tip_points[1], axis=0)]
        self.estimate_hand_interaction_center = [
            (self.estimate_thumb_point[0] + self.estimate_index_point[0]) / 2,
            (self.estimate_thumb_point[1] + self.estimate_index_point[1]) / 2
        ]
        self.estimate_palm_point = [
            get_estimate_palm_point(vitpose_detections[0], depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics),
            get_estimate_palm_point(vitpose_detections[1], depth_seg_seq[0], self.camera_intrinsics, self.camera_extrinsics)
        ]
        print("estimate_hand_interaction_center", self.estimate_hand_interaction_center)
        print("estimate_palm_point", self.estimate_palm_point)

        if use_smplh:
            # load whole smplh traj, and parse out the needed part
            smplh_traj = get_smplh_traj_annotation(human_video_annotation_path)
            smplh_start_idx = self.human_node.smplh_traj.cal_idx_from_video_to_smplh(segment_start_idx)
            smplh_end_idx = self.human_node.smplh_traj.cal_idx_from_video_to_smplh(segment_end_idx + 1)
            self.human_node.add_smplh_traj(smplh_traj[smplh_start_idx:smplh_end_idx])

            self.smplh_traj = self.human_node.smplh_traj.smplh_traj

            print("video_idx=", segment_start_idx, segment_end_idx)
            print("smplh_idx=", smplh_start_idx, smplh_end_idx)

            type_l = 'none'
            type_r = 'none'
            
            all_arms_contact = get_hamer_hand_object_contacts_annotation(human_video_annotation_path)
            self.arms_contact = all_arms_contact[segment_idx] if segment_idx < len(all_arms_contact) else all_arms_contact[-1]
            type_l = 'close' if (self.arms_contact[0] >= 0) else 'open'
            type_r = 'close' if (self.arms_contact[1] >= 0) else 'open'
            self.hand_type = [type_l, type_r]
            
            self.identify_moving_arm()
    
    def identify_moving_arm(self):
        retargeter = Retargeter()
        retargeter.calibrate(self.smplh_traj[0])
        L_targets = []
        R_targets = []
        for i in range(len(self.smplh_traj)):
            targets = retargeter.get_targets_from_smplh(self.smplh_traj[i])
            L_targets.append(targets['link_LArm7'][:3, 3])
            R_targets.append(targets['link_RArm7'][:3, 3])

        L_targets = np.array(L_targets)
        R_targets = np.array(R_targets)

        L_diff = np.linalg.norm(L_targets[1:] - L_targets[:-1], axis=1)
        R_diff = np.linalg.norm(R_targets[1:] - R_targets[:-1], axis=1)

        L_diff = np.mean(L_diff)
        R_diff = np.mean(R_diff)

        print("L_diff=", L_diff, "R_diff=", R_diff)
        L_big = 0
        R_big = 0
        if L_diff > R_diff:
            L_big += 1
        else:
            R_big += 1

        cur_len = len(self.smplh_traj)
        while abs(L_diff - R_diff) < 0.003:
            # most likely both arms are moving. Then we need to identify the arm is moves in the latter part of the trajectory
            cur_len = int(cur_len * 0.5)

            if cur_len <= 1:
                if L_big > R_big:
                    L_diff = R_diff + 1
                else:
                    R_diff = L_diff + 1
                break

            L_diff = np.linalg.norm(L_targets[-(cur_len - 1):] - L_targets[-cur_len: -1], axis=1)
            R_diff = np.linalg.norm(R_targets[-(cur_len - 1):] - R_targets[-cur_len: -1], axis=1)

            L_diff = np.mean(L_diff)
            R_diff = np.mean(R_diff)

            print("cur_len=", cur_len, "L_diff=", L_diff, "R_diff=", R_diff)

            if L_diff > R_diff:
                L_big += 1
            else:
                R_big += 1
        
        if L_diff > R_diff:
            self.moving_arm = 'L'
        else:
            self.moving_arm = 'R'

        print("moving arm is", self.moving_arm)

    def get_representative_images(self, num_images=10):
        img_idx = np.linspace(self.segment_start_idx, self.segment_end_idx, num_images, dtype=int)
        video_seq = get_video_seq_from_annotation(self.human_video_annotation_path)
        img_lst = [video_seq[idx] for idx in img_idx]
        return img_lst