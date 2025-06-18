from typing import Dict, Tuple

import meshcat_shapes
import numpy as np
import pinocchio as pin

from retarget.robot import Robot
import cv2
import os

img_idx = 0

class RobotVisualizer:
    def __init__(self, robot: Robot, config: Dict = None):
        self.viz = pin.visualize.MeshcatVisualizer(
            robot.robot.model, robot.robot.collision_model, robot.robot.visual_model
        )
        robot.robot.setVisualizer(self.viz, init=False)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.viz.display(robot.robot.q0)
        self.viz_targets = {}

        self.img_idx = 0

    def visualize(self, robot_state: np.ndarray = None, pose_dict: Dict = None):
        # visualize robot state
        if robot_state is not None:
            # print("displaying robot state")
            self.viz.display(robot_state)
            pass

        # visualize target poses
        if pose_dict is not None:
            removed_keys = []
            for k in self.viz_targets.keys():
                if k not in pose_dict:
                    removed_keys.append(k)
            for k in removed_keys:
                del self.viz_targets[k]

            for k, (pose, kwargs) in pose_dict.items():
                if k in self.viz_targets:
                    self.viz_targets[k].set_transform(pose)
                else:
                    viz_target = self.viz.viewer[k]
                    self.viz_targets[k] = viz_target
                    meshcat_shapes.frame(viz_target, **kwargs)
                    self.viz_targets[k].set_transform(pose)

        img = self.viz.captureImage()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # save img
        # print("saving img", self.img_idx)
        # os.makedirs("/home/bb8/workspace/gr1/GR1_vision/imgs", exist_ok=True)
        # cv2.imwrite(f"/home/bb8/workspace/gr1/GR1_vision/imgs/img_{self.img_idx}.png", img)
        self.img_idx += 1

        # stream image
        # cv2.imshow("img", img)
        # cv2.waitKey(10)

        # return self.viz.captureImage()
