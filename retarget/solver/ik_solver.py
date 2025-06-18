from typing import Dict
import numpy as np

import pink
import pinocchio as pin
import qpsolvers
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

from retarget.robot import Robot

from .solver import Solver
from retarget.utils.misc import interpolate_se3


class IKSolver(Solver):
    def __init__(self, config: Dict, robot: Robot):
        # load robot
        super().__init__(config, robot)
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"
        else:
            self.solver = qpsolvers.available_solvers[0]

        self.dt = config["dt"]
        self.num_step_per_frame = config["num_step_per_frame"]
        self.amplify_factor = config.get("amplify_factor", 1.0)

        # initialize tasks
        self.tasks = {}
        for link_name, weight in config["link_costs"].items():
            assert link_name != "posture", "posture is a reserved task name"
            task = FrameTask(
                link_name,
                **weight,
            )
            self.tasks[link_name] = task

        # add posture task
        self.tasks["posture"] = PostureTask(
            cost=config["posture_cost"],
        )
        for task in self.tasks.values():
            task.set_target_from_configuration(self.robot.configuration)

    def interpolate_targets(self, target_pose: Dict):

        # query initial link poses
        init_pose = {}
        for link_name in target_pose.keys():
            if link_name not in self.tasks:
                continue
            init_pose[link_name] = self.robot.get_link_transformations([link_name])[0]

        # determine number of interpolation steps
        thres = 0.05
        max_translation_dis = 0
        for link_name, pose in target_pose.items():
            if link_name not in self.tasks:
                continue
            max_translation_dis = max(
                max_translation_dis,
                np.linalg.norm(pose[:3, 3] - init_pose[link_name][:3, 3]),
            )
        num_steps = int(max_translation_dis / thres)
        # num_steps = np.clip(num_steps, 40, 50)
        # print("num_steps", num_steps)
        num_steps = 20

        # interpolate
        target_poses = []
        for i in range(num_steps):
            target_pose_i = {}
            for link_name, pose in target_pose.items():
                if link_name not in self.tasks:
                    continue
                target_pose_i[link_name] = interpolate_se3(
                    init_pose[link_name], pose, i / num_steps
                )
            target_poses.append(target_pose_i)
        target_poses.append(target_pose)
        
        return target_poses
        
    def __call__(self, target_pose: Dict):

        # interpolate target poses
        target_poses_seq = self.interpolate_targets(target_pose.copy())
        # target_poses_seq = [target_pose]

        for sub_target_pose in target_poses_seq:
            for link_name, pose in sub_target_pose.items():
                if link_name not in self.tasks:
                    continue
                pose = pin.SE3(pose[:3, :3], pose[:3, 3])
                self.tasks[link_name].set_target(pose)
            # for link_name in self.tasks.keys():
            #     if link_name == "posture":
            #         continue
            #     if link_name not in target_pose:
            #         continue
            #     target_transform = target_pose[link_name]
            #     target_transform = pin.SE3(target_transform[:3, :3], target_transform[:3, 3])
            #     self.tasks[link_name].set_target(target_transform)

            for _ in range(self.num_step_per_frame):
                velocity = solve_ik(
                    self.robot.configuration,
                    self.tasks.values(),
                    dt=self.dt,
                    solver=self.solver,
                )
                self.robot.update(
                    self.robot.configuration.q + velocity * self.dt * self.amplify_factor,
                    tol=1e-6,
                )

        return self.robot.configuration.q.copy()

    def update_weights(self, weights):
        for link_name, weight in weights.items():
            if "position_cost" in weight:
                self.tasks[link_name].set_position_cost(weight["position_cost"])
            if "orientation_cost" in weight:
                self.tasks[link_name].set_orientation_cost(weight["orientation_cost"])
