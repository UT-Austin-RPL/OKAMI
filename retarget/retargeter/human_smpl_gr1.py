import numpy as np
from easydict import EasyDict

from retarget.robot import Robot
from retarget.utils.constants import name_to_urdf_idx
from retarget.visualizer import RobotVisualizer

from .retargeter import Retargeter


class SMPLGR1Retargeter(Retargeter):
    def __init__(self, config: EasyDict, vis=False):
        super().__init__(config)
        self.body = Robot(config["GR1_body"])
        self.left_hand = Robot(config["GR1_left_hand"])
        self.right_hand = Robot(config["GR1_right_hand"])

        # including manually fixed joints
        self.q_fill_idx = np.zeros(len(self.body.configuration.q), dtype=int)
        for k, v in self.body.joint2idx.items():
            self.q_fill_idx[v] = name_to_urdf_idx[k]

        if vis:
            self.vis = RobotVisualizer(self.body)
        else:
            self.vis = None

    def calibrate(self, data):
        self.body.calibrate(data)
        self.left_hand.calibrate(data)
        self.right_hand.calibrate(data)

    def __call__(self, data):
        q, target = self.body.retarget(data)
        q_left_hand, _ = self.left_hand.retarget(data)
        q_right_hand, _ = self.right_hand.retarget(data)
        # print(q_left_hand[-10:])

        for k, v in self.left_hand.joint2idx.items():
            q[self.body.joint2idx[k]] = q_left_hand[v]
            # print(k, self.body.joint2idx[k], q_left_hand[v])
        for k, v in self.right_hand.joint2idx.items():
            q[self.body.joint2idx[k]] = q_right_hand[v]
        self.body.update(q)
        self.visualize(target)

        q_complete = np.zeros(len(name_to_urdf_idx))
        q_complete[self.q_fill_idx] = q
        return q_complete

    def control(self, weights, relative_trans):
        q = self.body.control(weights, relative_trans)
        self.visualize({})
        return q

    def reset(self):
        self.body.reset()

    def visualize(self, target):
        link_pose_visualize_params = {
            "origin_color": 0x000000,
            "axis_length": 0.1,
            "axis_thickness": 0.01,
            "origin_radius": 0.03,
        }

        target_link_pose_visualize_params = {
            "origin_color": 0xFFFFFF,
            "axis_length": 0.1,
            "axis_thickness": 0.01,
            "origin_radius": 0.03,
        }

        fingle_pose_visualize_params = {
            "origin_color": 0x000000,
            "axis_length": 0.03,
            "axis_thickness": 0.005,
            "origin_radius": 0.01,
        }

        target_fingle_pose_visualize_params = {
            "origin_color": 0xFFFFFF,
            "axis_length": 0.03,
            "axis_thickness": 0.005,
            "origin_radius": 0.01,
        }

        if self.vis is not None:
            pose_dict = {}
            for k in self.body.solver.tasks.keys():
                if k in target.keys():
                    target_pose = target[k]
                    robot_pose = self.body.get_link_transformations([k])[0]
                    if "finger" in k.lower():
                        pose_dict["target_" + k] = (
                            target_pose,
                            target_fingle_pose_visualize_params,
                        )
                        pose_dict["robot_" + k] = (robot_pose, fingle_pose_visualize_params)
                    else:
                        pose_dict["target_" + k] = (target_pose, target_link_pose_visualize_params)
                        pose_dict["robot_" + k] = (robot_pose, link_pose_visualize_params)
            self.vis.visualize(self.body.configuration.q, pose_dict)
