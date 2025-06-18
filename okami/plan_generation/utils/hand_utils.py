from typing import Any
import numpy as np

from enum import Enum

from okami.plan_generation.utils.o3d_utils import remove_outlier
from okami.plan_generation.utils.misc_utils import get_hamer_result, create_point_clouds_from_keypoints


class SimpleEEFAction(Enum):
    """Modes of different eef actions that will be considered in the pipeline. Define a more complex one if needed."""
    NULL = 0
    OPEN = 1
    CLOSE = 2

class InteractionAffordance():
    def __init__(self):
        # self.affordance_type = "Pose"
        # self.affordance_location = np.empty((1, 7), dtype=np.float32)

        self.affordance_type = "Pose"
        self.affordance_centroid = np.empty((1, 3), dtype=np.float32)
        self.affordance_thumb_tip = np.empty((1, 3), dtype=np.float32)
        self.affordance_index_tip = np.empty((1, 3), dtype=np.float32)

    def set_affordance_centroid(self, location):
        self.affordance_centroid = location.reshape(1, 3)

    def to_dict(self):
        return {
            "affordance_type": self.affordance_type,
            "affordance_centroid": self.affordance_centroid.tolist(),
            "affordance_thumb_tip": self.affordance_thumb_tip.tolist(),
            "affordance_index_tip": self.affordance_index_tip.tolist(),
        }
    
    @classmethod
    def from_dict(self, affordance_dict):
        self.affordance_type = affordance_dict["affordance_type"]
        self.affordance_centroid = np.array(affordance_dict["affordance_centroid"])
        self.affordance_thumb_tip = np.array(affordance_dict["affordance_thumb_tip"])
        self.affordance_index_tip = np.array(affordance_dict["affordance_index_tip"])

    def set_affordance_thumb_tip(self, thumb_tip):
        self.affordance_thumb_tip = thumb_tip.reshape(1, 3)

    def set_affordance_index_tip(self, index_tip):
        self.affordance_index_tip = index_tip.reshape(1, 3)

    def get_affordance_centroid(self):
        return self.affordance_centroid
    
    def get_interaction_points(self, include_centroid=False):
        if include_centroid:
            return np.concatenate((self.affordance_thumb_tip, 
                                self.affordance_index_tip,
                                self.affordance_centroid), axis=0)
        else:
            return np.concatenate((self.affordance_thumb_tip, 
                                self.affordance_index_tip), axis=0)



def compute_thumb_index_joints(annotation_path):
    hamer_result, result_path = get_hamer_result(annotation_path)
    thumb_index_joints = hamer_result["hand_joints_seq"][0].squeeze()
    distances = np.linalg.norm(thumb_index_joints[:, 4, :] - thumb_index_joints[:, 8, :], 2, axis=-1)
    return distances

def get_finger_joint_points(vit_pose_detections, 
                            idx, 
                            depth, 
                            intrinsics_matrix, 
                            extrinsics_matrix):
    """_summary_

    Args:
        vit_pose_detections (numpy.ndarray): 21x2 hand pose keypoints
        idx (integer):integer index of the finger
        depth (depth map): HxW depth map
        intrinsics_matrix (numpy.ndarray): 3x3 intrinsics matrix
        extrinsics_matrix (numpy.ndarray): 4x4 extrinsics matrix

    Returns:
        np.ndarray: Nx3 points
    """
    pose_of_interest = vit_pose_detections[idx:idx+1]
    keypoints = pose_of_interest
    radius = 3
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            keypoints = np.concatenate((keypoints, pose_of_interest + np.array([i, j])), axis=0)
    points = create_point_clouds_from_keypoints(keypoints, depth, intrinsics_matrix)
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1)
    points = extrinsics_matrix @ points.T
    points = points.T[:, 0:3]
    points = remove_outlier(points)
    return points

def get_thumb_tip_points(vit_pose_detections, 
                         depth, 
                         intrinsics_matrix, 
                         extrinsics_matrix):
    """_summary_

    Args:
        vit_pose_detections (_type_): _description_
        depth (depth map): HxW depth map
        intrinsics_matrix (numpy.ndarray): 3x3 intrinsics matrix
        extrinsics_matrix (numpy.ndarray): 4x4 extrinsics matrix

    Returns:
        _type_: _description_
    """
    thumb_tip_points = get_finger_joint_points(vit_pose_detections, 4, depth, intrinsics_matrix, extrinsics_matrix)
    return thumb_tip_points

def get_thumb_dip_points(vit_pose_detections, 
                         depth, 
                         intrinsics_matrix, 
                         extrinsics_matrix):
    """_summary_

    Args:
        vit_pose_detections (_type_): _description_
        depth (depth map): HxW depth map
        intrinsics_matrix (numpy.ndarray): 3x3 intrinsics matrix
        extrinsics_matrix (numpy.ndarray): 4x4 extrinsics matrix

    Returns:
        _type_: _description_
    """
    thumb_tip_points = get_finger_joint_points(vit_pose_detections, 3, depth, intrinsics_matrix, extrinsics_matrix)
    return thumb_tip_points

def get_index_tip_points(vit_pose_detections, 
                         depth, 
                         intrinsics_matrix, 
                         extrinsics_matrix):
    """_summary_

    Args:
        vit_pose_detections (_type_): _description_
        depth (depth map): HxW depth map
        intrinsics_matrix (numpy.ndarray): 3x3 intrinsics matrix
        extrinsics_matrix (numpy.ndarray): 4x4 extrinsics matrix

    Returns:
        _type_: _description_
    """
    index_tip_points = get_finger_joint_points(vit_pose_detections, 8, depth, intrinsics_matrix, extrinsics_matrix)
    return index_tip_points

def get_index_dip_points(vit_pose_detections,
                            depth, 
                            intrinsics_matrix, 
                            extrinsics_matrix):
        """_summary_
    
        Args:
            vit_pose_detections (_type_): _description_
            depth (depth map): HxW depth map
            intrinsics_matrix (numpy.ndarray): 3x3 intrinsics matrix
            extrinsics_matrix (numpy.ndarray): 4x4 extrinsics matrix
    
        Returns:
            _type_: _description_
        """
        index_dip_points = get_finger_joint_points(vit_pose_detections, 7, depth, intrinsics_matrix, extrinsics_matrix)
        return index_dip_points

def get_estimate_palm_point(vit_pose_detections, 
                            depth, 
                            intrinsics_matrix, 
                            extrinsics_matrix):
    """_summary_

    Args:
        vit_pose_detections (_type_): _description_
        depth (depth map): HxW depth map
        intrinsics_matrix (numpy.ndarray): 3x3 intrinsics matrix
        extrinsics_matrix (numpy.ndarray): 4x4 extrinsics matrix

    Returns:
        _type_: _description_
    """
    index_MCP = get_finger_joint_points(vit_pose_detections, 5, depth, intrinsics_matrix, extrinsics_matrix)
    middle_MCP = get_finger_joint_points(vit_pose_detections, 9, depth, intrinsics_matrix, extrinsics_matrix)
    ring_MCP = get_finger_joint_points(vit_pose_detections, 13, depth, intrinsics_matrix, extrinsics_matrix)
    pinky_MCP = get_finger_joint_points(vit_pose_detections, 17, depth, intrinsics_matrix, extrinsics_matrix)

    palm_point = np.mean(np.concatenate((index_MCP, middle_MCP, ring_MCP, pinky_MCP), axis=0), axis=0)
    print("palm_point: ", palm_point)
    return palm_point