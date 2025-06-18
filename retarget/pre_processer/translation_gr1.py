from copy import deepcopy

import numpy as np

from retarget.robot import Robot
from scipy.spatial.transform import Rotation as R
from .pre_processor import PreProcessor

SMPL_WRIST = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

class TranslationGR1Preprocessor(PreProcessor):
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)

        self.left_shoulder = "l_shoulder_roll"
        self.right_shoulder = "r_shoulder_roll"

        self.joint2idx = {
            "link_LArm1": 13,
            "link_RArm1": 14,
            "link_LArm2": 16,
            "link_RArm2": 17,
            "link_LArm4": 18,
            "link_RArm4": 19,
            "link_LArm7": 20,
            "link_RArm7": 21,
        }

        self.offset = config['offset']
        self.correct_orientation = config['correct_orientation']

    def calibrate(self, data):
        self.init_pelvis_pos = data["body"][0][:3, 3]

        # set to T pose
        q0 = deepcopy(self.robot.q0)
        q0[self.robot.joint2idx[self.left_shoulder]] = -np.pi / 2
        q0[self.robot.joint2idx[self.right_shoulder]] = np.pi / 2
        self.robot.update(q0)

        # assume the first frame is in T pose
        # calculate wrist rotation matrix
        left_wrist, right_wrist = self.robot.get_link_transformations(["link_LArm7", "link_RArm7"])
        # smpl_wrist_left = data[20][:3, :3]
        # smpl_wrist_right = data[21][:3, :3]
        self.left_rotation = np.linalg.inv(SMPL_WRIST) @ left_wrist[:3, :3]
        self.right_rotation = np.linalg.inv(SMPL_WRIST) @ right_wrist[:3, :3]

    def fix_transformation(self, data):
        data[:, :3, 3] -= self.init_pelvis_pos
        for link_name, idx in self.joint2idx.items():
            if idx >= len(data):
                continue
            if link_name[5] == "L":
                data[idx][:3, :3] = data[idx][:3, :3] @ self.left_rotation
            elif link_name[5] == "R":
                data[idx][:3, :3] = data[idx][:3, :3] @ self.right_rotation

        return data

    def __call__(self, data):
        # position = data[:, :3, 3]
        # orientation = data[:, :3, :3]
        data_new = deepcopy(data["body"])
        data_new = self.fix_transformation(data_new)
        target = {}
        for joint, idx in self.joint2idx.items():
            if idx >= len(data_new):
                continue
            target[joint] = data_new[idx]
            if joint in self.offset:
                target[joint][:3, 3] += np.array(self.offset[joint])
        
        if self.correct_orientation:
            delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
            # delta_pos = target['link_LArm7'][:3, :3] @ np.array([0.1, 0.0, 0.0])

            delta_transformation = np.eye(4)
            delta_transformation[:3, :3] = delta_rot
            # delta_transformation[:3, 3] = delta_pos
            target['link_LArm7'] = target['link_LArm7'] @ delta_transformation
            
            # delta_pos = target['link_RArm7'][:3, :3] @ np.array([0.1, 0.0, 0.0])
            # delta_transformation[:3, 3] = delta_pos
            target['link_RArm7'] = target['link_RArm7'] @ delta_transformation

        return target