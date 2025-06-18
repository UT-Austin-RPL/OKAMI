import pickle

import numpy as np
from easydict import EasyDict

from retarget.robot import Robot
from retarget.utils.constants import name_to_urdf_idx
from retarget.utils.misc import compute_transformation_error
from retarget.visualizer import RobotVisualizer

from .retargeter import Retargeter


class TeleopAVPGR1Retargeter(Retargeter):
    def __init__(self, config: EasyDict, vis=False, save_path=None):
        super().__init__(config)
        self.body = Robot(config["GR1_body"])
        self.left_hand = Robot(config["GR1_left_hand"])
        self.right_hand = Robot(config["GR1_right_hand"])

        # including manually fixed joints
        self.q_fill_idx = np.zeros(len(self.body.configuration.q), dtype=int)
        for k, v in self.body.joint2idx.items():
            self.q_fill_idx[v] = name_to_urdf_idx[k]

        if save_path:
            self.save_path = save_path
            self.history = []
        else:
            self.save_path = None
            self.history = None

        if vis:
            self.vis = RobotVisualizer(self.body)
        else:
            self.vis = None

    def calibrate(self, data):
        self.body.calibrate(data)
        self.left_hand.calibrate(data)
        self.right_hand.calibrate(data)

    def reset(self):
        self.body.reset()
        self.left_hand.reset()
        self.right_hand.reset()

    def __call__(self, data):
        q, target = self.body.retarget(data)
        q_left_hand, _ = self.left_hand.retarget(data)
        q_right_hand, _ = self.right_hand.retarget(data)

        for k, v in self.left_hand.joint2idx.items():
            q[self.body.joint2idx[k]] = q_left_hand[v]
        for k, v in self.right_hand.joint2idx.items():
            q[self.body.joint2idx[k]] = q_right_hand[v]
        self.body.update(q)

        error_dict, cur_target_dict = self.evaluate(target)
        self.visualize(target)
        if self.history is not None:
            self.history.append(
                {
                    "robot_state": q,
                    "cur_target_dict": cur_target_dict,
                    "error_dict": error_dict,
                    "input": data,
                }
            )

        q_complete = np.zeros(len(name_to_urdf_idx))
        q_complete[self.q_fill_idx] = q
        return q_complete, error_dict

    def control(self, weights, relative_trans):
        raise NotImplementedError("Control not implemented for Apple Vision Pro")

    def evaluate(self, target):
        (
            robot_head_pose,
            robot_l_wrist_pose,
            robot_r_wrist_pose,
        ) = self.body.get_link_transformations(["link_head_pitch", "link_LArm7", "link_RArm7"])
        target_head_pose = target["link_head_pitch"]
        target_l_wrist_pose = target["link_LArm7"]
        target_r_wrist_pose = target["link_RArm7"]

        cur_target_dict = {
            "head": (robot_head_pose, target_head_pose),
            "left_wrist": (robot_l_wrist_pose, target_l_wrist_pose),
            "right_wrist": (robot_r_wrist_pose, target_r_wrist_pose),
        }
        error_dict = {}
        for k, (cur, target) in cur_target_dict.items():
            error_dict[k] = compute_transformation_error(cur, target)
        return error_dict, cur_target_dict

    def visualize(self, target):
        target_pose_visualize_params = {
            "origin_color": 0x00FF00,
            "axis_length": 0.1,
            "axis_thickness": 0.01,
            "origin_radius": 0.03,
        }

        cur_pose_visualize_params = {
            "origin_color": 0xFF0000,
            "axis_length": 0.1,
            "axis_thickness": 0.01,
            "origin_radius": 0.03,
        }

        finger_visualize_params = {
            "origin_color": 0x0000FF,
            "axis_length": 0.03,
            "axis_thickness": 0.003,
            "origin_radius": 0.01,
        }
        if self.vis is not None:
            (
                robot_head_pose,
                robot_l_wrist_pose,
                robot_r_wrist_pose,
            ) = self.body.get_link_transformations(["link_head_pitch", "link_LArm7", "link_RArm7"])
            pose_dict = {
                "left_wrist": (target["link_LArm7"], target_pose_visualize_params),
                "right_wrist": (target["link_RArm7"], target_pose_visualize_params),
                "head": (target["link_head_pitch"], cur_pose_visualize_params),
                "robot_left_wrist": (robot_l_wrist_pose, cur_pose_visualize_params),
                "robot_right_wrist": (robot_r_wrist_pose, cur_pose_visualize_params),
                "robot_head": (robot_head_pose, target_pose_visualize_params),
            }
            self.vis.visualize(self.body.configuration.q, pose_dict)

    def visualize_robot_only(self):
        if self.vis is not None:
            self.vis.visualize(self.body.configuration.q, {})

    def save(self):
        if self.save_path is not None:
            print(f"Saving data to {self.save_path}")
            with open(self.save_path, "wb") as f:
                pickle.dump(self.history, f)
        else:
            print("Not saving data, save_path is None")
