import os
from copy import deepcopy
from typing import Dict

import numpy as np
import pink
from easydict import EasyDict
from pink.configuration import get_root_joint_dim
from scipy.spatial.transform import Rotation as RRR

from retarget.utils.load_robot import load_robot_description
from retarget.utils.misc import fix_urdf_joints, get_root_dir, import_class

# from retarget.solver.solver import Solver


class Robot:
    def __init__(self, config):
        self.config = config
        self.robot = self.process_and_load_robot_description()
        self.joint2idx = {
            k: self.robot.model.getJointId(k) - 1 for k in self.robot.model.names[1:]
        }
        self.initial_upper = deepcopy(self.robot.model.upperPositionLimit)
        self.initial_lower = deepcopy(self.robot.model.lowerPositionLimit)
        # for joint in self.joint2idx.keys():
        #     if joint in fixed_joints:
        #         self.robot.model.upperPositionLimit[self.joint2idx[joint]] = 0
        #         self.robot.model.lowerPositionLimit[self.joint2idx[joint]] = 0
        #     else:
        #         # self.robot.model.upperPositionLimit[self.joint2idx[joint]] = 2 * np.pi
        #         # self.robot.model.lowerPositionLimit[self.joint2idx[joint]] = -2 * np.pi
        #         pass
        self.configuration = pink.Configuration(self.robot.model, self.robot.data, self.robot.q0)
        self.auto_clip = config.get("auto_clip", False)
        self.q0 = self.robot.q0
        self.nq = self.q0.shape[0]

        self.init_retarget()

    def process_and_load_robot_description(self):
        root_dir = get_root_dir()
        for k, v in self.config["robot_description"].items():
            if "path" in k.lower():
                self.config["robot_description"][k] = os.path.join(root_dir, v)
            else:
                self.config["robot_description"][k] = v
        fixed_joints = self.config.get("fixed_joints", [])
        if len(fixed_joints) > 0:
            print(f"Fixing joints in URDF: {', '.join(fixed_joints)}")
            self.config["robot_description"]["URDF_PATH"] = fix_urdf_joints(
                self.config["robot_description"]["URDF_PATH"], fixed_joints
            )
        return load_robot_description(self.config["robot_description"], root_joint=None)

    def update(self, q, tol=1e-6):
        q = deepcopy(q)
        if self.auto_clip:
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

    def reset(self):
        self.update(self.q0)
        for pre_filter in self.pre_filters:
            pre_filter.reset_history()
        for post_filter in self.post_filters:
            post_filter.reset_history()

    def init_retarget(self):
        self.pre_processor = import_class(self.config.pre_processor["_target_"])(
            self.config.pre_processor, self
        )
        self.pre_filters = []
        for pre_filter_config in self.config.get("pre_filters", []):
            pre_filter = import_class(pre_filter_config["_target_"])(pre_filter_config)
            self.pre_filters.append(pre_filter)
        self.solver = import_class(self.config.solver["_target_"])(self.config.solver, self)
        self.post_filters = []
        for post_filter_config in self.config.get("post_filters", []):
            # joint position filter need specific processing
            if post_filter_config["_target_"] == "retarget.filter.JointPositionFilter":
                post_filter_config["joint_size"] = self.nq
                joint_mask = []
                for name in post_filter_config.get("joint_mask_names", []):
                    if name in self.joint2idx:
                        joint_mask.append(self.joint2idx[name])
                post_filter_config["joint_mask"] = joint_mask

            post_filter = import_class(post_filter_config["_target_"])(post_filter_config)
            self.post_filters.append(post_filter)

    def calibrate(self, data):
        self.pre_processor.calibrate(data)

    def retarget(self, data):
        target = self.pre_processor(data)
        for pre_filter in self.pre_filters:
            target = pre_filter(target)
        q = self.solver(target)
        for post_filter in self.post_filters:
            q = post_filter(q)
        return q, target

    def control(
        self, weights: Dict, delta_trans
    ):  # pos=None, rot_mat=None, rot_vec=None, quat=None):
        # if not pos:
        #     pos = np.zeros(3)
        # # Can only use one of rot_mat, rot_vec, quat
        # assert sum([rot_mat is not None, rot_vec is not None, quat is not None]) <= 1
        # if rot_vec is not None:
        #     rot_mat = RRR.from_rotvec(rot_vec).as_matrix()
        # if quat is not None:
        #     rot_mat = RRR.from_quat(quat).as_matrix()
        # if rot_mat is None:
        #     rot_mat = np.eye(3)
        # relative_trans = np.eye(4)
        # relative_trans[:3, :3] = rot_mat
        # relative_trans[:3, 3] = pos

        self.fix_fingers()
        adjuster = import_class(self.config.adjuster["_target_"])(self.config.adjuster, self)
        adjuster.update_weights(weights)
        cur_target = self.get_link_transformations(list(weights.keys()))
        new_target = np.einsum("ij,njk->nik", delta_trans, cur_target)
        target = {k: v for k, v in zip(weights.keys(), new_target)}
        self.release_fingers()
        return adjuster(target)

    def fix_fingers(self):
        fixed_joints = self.config.get("fingers", [])
        for joint in fixed_joints:
            idx = self.joint2idx[joint]
            self.configuration.model.upperPositionLimit[idx] = self.configuration.q[idx]
            self.configuration.model.lowerPositionLimit[idx] = self.configuration.q[idx]

    def release_fingers(self):
        fixed_joints = self.config.get("fingers", [])
        for joint in fixed_joints:
            idx = self.joint2idx[joint]
            self.configuration.model.upperPositionLimit[idx] = self.initial_upper[idx]
            self.configuration.model.lowerPositionLimit[idx] = self.initial_lower[idx]
