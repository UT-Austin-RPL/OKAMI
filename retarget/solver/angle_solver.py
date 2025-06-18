from typing import Any, Dict

import numpy as np

from retarget.robot import Robot

from .solver import Solver


class AngleSolver(Solver):
    def __init__(self, config: Dict, robot: Robot):
        super().__init__(config, robot)
        self.x_plane_idx = config["x_plane_idx"]
        self.z_plane_idx = config["z_plane_idx"]
        self.thumb_idx = config["thumb_idx"]

    def angle_between(self, v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(cos_theta)
        if theta > np.pi / 2:
            theta = np.pi - theta
        return theta

    def calculate_x_plane(self, target_points):
        # Find the plane of the hand
        x_plane_points = target_points[self.x_plane_idx]
        centroid = np.mean(x_plane_points, axis=0)
        # Translate points to the origin
        translated_points = x_plane_points - centroid
        # Apply SVD
        _, _, Vt = np.linalg.svd(translated_points)
        # The normal vector of the plane is the last column of V
        return Vt[-1, :]

    def calculate_z_planes(self, target_points, x_plane_normal):
        z_plane_normals = []
        for finger in self.z_plane_idx:
            # Assuming we have the normal vector from the previous step
            point1, point2 = target_points[finger]
            # Find a vector connecting the two points
            vector_points = point2 - point1
            # The new plane is perpendicular to the first plane, so its normal vector
            # is perpendicular to the normal vector of the first plane and to the vector between the two new points
            normal_new_plane = np.cross(x_plane_normal, vector_points)
            z_plane_normals.append(normal_new_plane)
        return z_plane_normals

    def calculate_thumb_rot(self, target_points, x_plane_normal, z_plane_normals):
        # use the index finger plane as the new z_plane
        mid_f_normal = z_plane_normals[1]
        y_plane_normal = np.cross(x_plane_normal, mid_f_normal)

        # calculate 1,2 shadow on y_plane
        p1, p2 = target_points[self.thumb_idx]
        v = p2 - p1
        v_proj = (
            v - (np.dot(v, y_plane_normal) / np.linalg.norm(y_plane_normal) ** 2) * y_plane_normal
        )

        # calculate angle between z_plane_normal and v_proj
        theta = self.angle_between(mid_f_normal, v_proj)
        return theta

    def __call__(self, target_transforms: Any):
        raise NotImplementedError(
            "Do not call AngleSolver directly. Use the derived classes instead."
        )

    def update_weights(self, weights):
        pass


class AVPAngleSolver(AngleSolver):
    def __init__(self, config, robot) -> None:
        super().__init__(config, robot)
        self.finger_end_idx = [[6, 7], [11, 12], [16, 17], [21, 22]]
        self.finger_up_idx = [
            # thumb
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            # ring
            [16, 17],
            [16, 17, 18],
            # little
            [21, 22],
            [21, 22, 23],
            # index
            [6, 7],
            [6, 7, 8],
            # middle
            [11, 12],
            [11, 12, 13],
        ]

    def __call__(self, target_transforms: Any):
        target_points = target_transforms[:, :3, 3]

        x_plane_normal = self.calculate_x_plane(target_points)
        z_plane_normals = self.calculate_z_planes(target_points, x_plane_normal)

        thumb_rot_theta = self.calculate_thumb_rot(target_points, x_plane_normal, z_plane_normals)

        thetas = [thumb_rot_theta]
        for points in self.finger_up_idx:
            if len(points) == 3:
                # calculate angle between p1p2 and p2p3
                p1, p2, p3 = points
                v1 = target_points[p2] - target_points[p1]
                v2 = target_points[p3] - target_points[p2]
                theta = self.angle_between(v1, v2)
                thetas.append(theta)
            else:
                p1, p2 = points
                v1 = target_points[p2] - target_points[p1]
                theta = self.angle_between(v1, x_plane_normal)
                theta = np.pi / 2 - theta
                thetas.append(theta)

        # discretize thetas
        thetas = np.array(thetas)

        return thetas


class SMPLAngleSolver(AngleSolver):
    def __init__(self, config, robot):
        super().__init__(config, robot)
        self.finger_order = [
            # thumb
            12,
            13,
            14,
            # ring
            9,
            10,
            # little
            6,
            7,
            # index
            0,
            1,
            # middle
            3,
            4,
        ]

    def __call__(self, targets):
        angles = targets["angles"]
        positions = targets["transformations"][:, :3, 3]
        x_plane_normal = self.calculate_x_plane(positions)
        z_plane_normals = self.calculate_z_planes(positions, x_plane_normal)

        thumb_rot = self.calculate_thumb_rot(positions, x_plane_normal, z_plane_normals)

        thetas = [thumb_rot]
        for idx in self.finger_order:
            thetas.append(angles[idx][2])

        thetas = np.abs(np.array(thetas))
        return thetas
