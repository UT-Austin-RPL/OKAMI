import numpy as np
import os
import json
import time
import cv2

from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface

from utils.real_robot_utils import real_to_urdf
from robot.gr1 import GR1URDFModel

from deoxys_vision.utils.camera_utils import assert_camera_ref_convention, get_camera_info
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys_vision.utils.o3d_utils import scene_pcd_fn, O3DPointCloud, estimate_rotation
from deoxys_vision.utils.plotly_utils import plotly_draw_3d_pcd
from deoxys_vision.utils.transformation.transform_manager import TransformManager

class GR1Transform:
    def __init__(self):
        self.gr1 = GR1URDFModel()
        
        self.intrinsics = np.array([
            [909.83630371,   0.        , 651.97015381],
            [  0.        , 909.12280273, 376.37097168],
            [  0.        ,   0.        ,   1.        ],
        ])

        self.extrinsics = np.array([
            [ 2.22044605e-16,  2.07353665e-16,  1.00000000e+00 , 7.74200000e-02],
            [-1.00000000e+00 , 6.93889390e-18 , 2.22044605e-16 , 3.25000000e-02],
            [ 6.93889390e-18 ,-1.00000000e+00 , 2.31531374e-16 ,-2.13700000e-02],
            [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]
        ])

        self.calibrated_T_world_chest = None
        self.q = np.zeros(56)
        self.waist_head_joints = np.zeros(6)

        self._transform_manager = TransformManager()

    def calibrate_T_world_chest_from_camera_rgbd(self, q):
        assert_camera_ref_convention('rs_0')
        camera_info = get_camera_info('rs_0')
        cr_interface = CameraRedisSubInterface(redis_host="localhost", camera_info=camera_info, use_depth=True)
        cr_interface.start()
        imgs = cr_interface.get_img()
        img_info = cr_interface.get_img_info()
        rgb_image = cv2.cvtColor(imgs['color'], cv2.COLOR_BGR2RGB)
        depth_image = imgs['depth']

        self.update_q(q)

        rgbd_ori = O3DPointCloud(max_points=50000)
        rgbd_ori.create_from_rgbd(rgb_image, depth_image, self.intrinsics, depth_trunc=1.0)
        rgbd_ori.transform(self.get_transform("camera", "chest"))
        plane_model = rgbd_ori.plane_estimation()["plane_model"]
        R_world_chest = estimate_rotation(plane_model, z_up=True)

        self.calibrated_T_world_chest = R_world_chest.copy()
        print("Calibrated T_world_chest: ", self.calibrated_T_world_chest)

    def update_q(self, q):
        assert len(q) == 56

        self.q = q
        self.waist_head_joints = q[:6].copy()
        self._transform_manager = TransformManager()

        T_head_camera = self.extrinsics.copy()
        self._transform_manager.add_transform("camera", "head", T_head_camera[:3, :3], T_head_camera[:3, 3])

        T_head_base = self.gr1.get_joint_pose_relative_to_head('base', q)
        R_head_chest = T_head_base.copy()
        R_head_chest[:3, 3] = 0
        self._transform_manager.add_transform("chest", "head", R_head_chest[:3, :3], R_head_chest[:3, 3])
        self._transform_manager.add_transform("base", "head", T_head_base[:3, :3], T_head_base[:3, 3])

        if self.calibrated_T_world_chest is not None:
            T_world_chest = self.calibrated_T_world_chest.copy()
            self._transform_manager.add_transform("chest", "world", T_world_chest[:3, :3], T_world_chest[:3, 3])

    def get_transform(self, from_frame: str, to_frame: str):
        '''
        get T_{to_frame}_{from_frame}
        '''
        return self._transform_manager.get_transform(from_frame, to_frame)
    
    def apply_transform_to_point(self, point, from_frame, to_frame):
        T = self.get_transform(from_frame, to_frame)
        pos_in_to_frame = (T @ np.array([point[0], point[1], point[2], 1]))[:3]
        return pos_in_to_frame