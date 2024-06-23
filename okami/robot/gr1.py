import os
import json
import cv2

import numpy as np
from typing import Dict, Tuple
import meshcat_shapes

from copy import deepcopy
import pink
from pink.configuration import get_root_joint_dim
import pinocchio as pin
from easydict import EasyDict

import retarget

def get_package_dirs(module):
    """Get package directories for a given module.

    Args:
        module: Robot description module.

    Returns:
        Package directories.
    """
    return [
        module.PACKAGE_PATH,
        module.REPOSITORY_PATH,
        os.path.dirname(module.PACKAGE_PATH),
        os.path.dirname(module.REPOSITORY_PATH),
        os.path.dirname(module.URDF_PATH),  # e.g. laikago_description
    ]
def get_root_dir():
    """Get the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RobotVisualizer:
    def __init__(self, robot, config: Dict = None):
        self.viz = pin.visualize.MeshcatVisualizer(
            robot.model, robot.collision_model, robot.visual_model
        )
        robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.viz.display(robot.q0)

    def visualize(self, robot_state: np.ndarray = None, pose_dict: Dict = None):
        # visualize robot state
        if robot_state is not None:
            self.viz.display(robot_state)

class GR1URDFModel:
    def __init__(self):

        root_dir = get_root_dir()

        config = EasyDict()
        retarget_repo_dir = os.path.dirname(retarget.__file__)
        config.REPOSITORY_PATH = os.path.join(retarget_repo_dir, "assets")
        config.PACKAGE_PATH = os.path.join(retarget_repo_dir, "assets/gr1")
        config.URDF_PATH = os.path.join(retarget_repo_dir, "assets/gr1/urdf/gr1_dex.urdf")

        # print("package_dirs=", get_package_dirs(config))

        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=config.URDF_PATH,
            package_dirs=get_package_dirs(config),
            root_joint=None
        )
        self.configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.vis = None
        # robot_description:
        #     REPOSITORY_PATH: assets
        #     PACKAGE_PATH: assets/gr1
        #     URDF_PATH: assets/gr1/urdf/gr1_dex.urdf
    
    def update(self, q, tol=1e-6):
        q = deepcopy(q)
        
        root_nq, _ = get_root_joint_dim(self.configuration.model)
        q_max = self.configuration.model.upperPositionLimit[root_nq:]
        q_min = self.configuration.model.lowerPositionLimit[root_nq:]
        q[root_nq:] = np.clip(q[root_nq:], q_min + tol, q_max - tol)

        self.configuration.q = q
        self.configuration.update()

    def get_link_transformations(self, link_names):
        return [
            self.configuration.get_transform_frame_to_world(link_name).np
            for link_name in link_names
        ]

    def get_marker_pose(self, joints):
        return None

    def get_gripper_pose(self, joints):
        # 56-dim joints
        # order: 3 waist + 3 head + 7 left arm + 12 left hand + 7 right arm + 12 right hand + 6 left leg + 6 right leg
        
        self.update(joints)
        gripper_link_names = ["link_RArm7", "link_LArm7"]
        gripper_poses = self.get_link_transformations(gripper_link_names)
        return gripper_poses[1]
    
    def get_gripper_pose_relative_to_head(self, joints):
        # 56-dim joints
        # order: 3 waist + 3 head + 7 left arm + 12 left hand + 7 right arm + 12 right hand + 6 left leg + 6 right leg
        
        return self.get_joint_pose_relative_to_head('link_LArm7', joints)
    
    def get_joint_pose(self, link_name, joints):
        self.update(joints)
        gripper_link_names = [link_name]
        gripper_poses = self.get_link_transformations(gripper_link_names)
        # print("gripper_poses=", gripper_poses)
        return gripper_poses[0]
    
    def get_joint_pose_relative_to_head(self, link_name, joints):
        self.update(joints)
        joint_link_names = [link_name]
        joint_pose = self.get_link_transformations(joint_link_names)[0]

        waist_link_name = ['link_head_pitch'] #['link_head_yaw']
        waist_pose = self.get_link_transformations(waist_link_name)[0]

        # print("joint_pose=", joint_pose, "waist_pose=", waist_pose)

        joint_relative_to_waist = np.linalg.inv(waist_pose) @ joint_pose
        return joint_relative_to_waist
    
    def visualize(self, joints):
        if self.vis is None:
            self.vis = RobotVisualizer(self.robot)
        self.vis.visualize(joints)

if __name__ == '__main__':
    gr1 = GR1URDFModel()