import numpy as np
from easydict import EasyDict

from retarget.robot import Robot
from retarget.visualizer import RobotVisualizer

from .retargeter import Retargeter


class TeleopAVPGR1HandRetargeter(Retargeter):
    def __init__(self, config: EasyDict, side="left", vis=False):
        super().__init__(config)
        self.hand = Robot(config[f"GR1_{side}_hand"])
        self.side = side

        if vis:
            self.vis = RobotVisualizer(self.hand)
        else:
            self.vis = None

    def calibrate(self, data):
        self.hand.calibrate(data)

    def reset(self):
        self.hand.reset()

    def __call__(self, data):
        q, target = self.hand.retarget(data)
        self.hand.update(q)
        self.visualize(target)
        return q

    def control(self, weights, relative_trans):
        raise NotImplementedError("Control is not implemented for Apple Visiion Pro")

    def visualize(self, target):
        # if self.vis is not None:
        #     self.vis.visualize(self.hand.configuration.q)
        if self.vis is not None:
            left_pose_visualize_params = {
                "origin_color": 0xFF0000,
                "axis_length": 0.01,
                "axis_thickness": 0.001,
                "origin_radius": 0.01,
            }
            if isinstance(target, np.ndarray):
                target_dict = {f"{idx}": pose for idx, pose in enumerate(target)}
            else:
                target_dict = target

            pose_dict = {k: (v, left_pose_visualize_params) for k, v in target_dict.items()}
            self.vis.visualize(self.hand.configuration.q, pose_dict)
