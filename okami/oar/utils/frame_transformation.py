import numpy as np
import os
import json
import time
import cv2

from okami.oar.algos.gr1 import GR1URDFModel

import pytransform3d.transform_manager as pt3_tm
import pytransform3d.transformations as pt3_tf

class TransformManager:
    """
    Transform Manager for RPL lab.
    """

    def __init__(self):
        self.tm = pt3_tm.TransformManager()

    def add_transform(self, from_frame: str, to_frame: str, R: np.ndarray, p: np.ndarray):
        """
        from_frame (str): frame name of targets
        to_frame (str): The specified coordinate frame to represent from_frame.
        transformation: 4x4 transformation matrices.
        """
        assert R.shape == (3, 3)
        if p.shape == (3, 1):
            p = p.reshape(
                3,
            )
        if p.shape == (1, 3):
            p = p.transpose().squeeze()
        assert p.shape == (3,)
        transformation = pt3_tf.transform_from(R=R, p=p)
        self.tm.add_transform(from_frame, to_frame, transformation)

    def get_transform(self, from_frame: str, to_frame: str):
        try:
            return self.tm.get_transform(from_frame, to_frame)
        except ValueError:
            return None

    def remove_transform(self, from_frame: str, to_frame: str):
        self.tm.remove_transform(from_frame, to_frame)

    def apply_transform_to_point(self):
        """Apply to 3d points"""
        raise NotImplementedError

    def apply_transform_to_pose(self):
        """Apply to 6DoF poses"""
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError


class GR1Transform:
    def __init__(self):
        self.gr1 = GR1URDFModel()
        self.waist_head_joints = np.zeros(6)

        self.camera_intrinsics = np.array([
            [909.83630371,   0.        , 651.97015381],
            [  0.        , 909.12280273, 376.37097168],
            [  0.        ,   0.        ,   1.        ],
        ])

        self.camera_extrinsics = np.eye(4)
        self.camera_extrinsics[:3, 3] = np.array([0.10742, 0.0125, 0.09])
        self.camera_extrinsics[:3, :3] = np.array([
            [ 0.        ,  0.        ,  1.        ],
            [-1.        ,  0.        ,  0.        ],
            [ 0.        , -1.        ,  0.        ],
        ])

    def update_q(self, q, base_pos):
        assert len(q) == 56

        self.waist_head_joints = q[:6].copy()
        self._transform_manager = TransformManager()

        T_head_camera = self.camera_extrinsics
        self._transform_manager.add_transform("camera", "head", T_head_camera[:3, :3], T_head_camera[:3, 3])

        T_head_base = self.gr1.get_joint_pose_relative_to_head('base', q)
        self._transform_manager.add_transform("base", "head", T_head_base[:3, :3], T_head_base[:3, 3])

        q[:3] *= 0
        T_head_upperbase = self.gr1.get_joint_pose_relative_to_head('base', q)
        self._transform_manager.add_transform("upperbase", "head", T_head_upperbase[:3, :3], T_head_upperbase[:3, 3])

        T_world_base = np.eye(4)
        T_world_base[:3, 3] = base_pos.copy()
        self._transform_manager.add_transform("base", "world", T_world_base[:3, :3], T_world_base[:3, 3])

    def get_transform(self, from_frame: str, to_frame: str):
        '''
        get T_{to_frame}_{from_frame}
        '''
        return self._transform_manager.get_transform(from_frame, to_frame)
    
    def apply_transform_to_point(self, point, from_frame, to_frame):
        '''
        Given the representation of point in from_frame, return the representation of point in to_frame.
        '''
        T = self.get_transform(from_frame, to_frame)
        pos_in_to_frame = (T @ np.array([point[0], point[1], point[2], 1]))[:3]
        return pos_in_to_frame
    
    def apply_transform_to_vector(self, vector, from_frame, to_frame):
        '''
        Given the representation of vector in from_frame, return the representation of vector in to_frame.
        '''
        T = self.get_transform(from_frame, to_frame).copy()
        T[:3, 3] = np.zeros(3)
        pos_in_to_frame = (T @ np.array([vector[0], vector[1], vector[2], 1]))[:3]
        return pos_in_to_frame

        # T = self.get_transform(from_frame, to_frame)
        # T[:3, 3] = np.zeros(3)
        # pos_in_to_frame = (T @ np.array([point[0], point[1], point[2], 0]))[:3]
        # return pos_in_to_frame
    
    def get_joint_pose_relative_to_head(self, link_name, joints):
        return self.gr1.get_joint_pose_relative_to_head(link_name, joints)
    
    def get_joint_pose(self, link_name, joints):
        return self.gr1.get_joint_pose(link_name, joints)