import numpy as np
from scipy.spatial.transform import Rotation as R

from .filter import Filter


class AngleLimiter(Filter):
    """Filter the rotation of the target to be within a certain range."""

    def __init__(self, config) -> None:
        super().__init__(config)
        config = config["target_links"]
        self.keys = config.keys()
        self.x_limit = {k: config[k]["x"] for k in self.keys}
        self.y_limit = {k: config[k]["y"] for k in self.keys}
        self.z_limit = {k: config[k]["z"] for k in self.keys}
        self.last_good_matrix = {k: np.eye(3) for k in self.keys}

    def __call__(self, target):
        for k in self.keys:
            new_cartesian_matrix = target[k][:3, :3]
            euler_angles = R.from_matrix(new_cartesian_matrix).as_euler("xyz", degrees=True)
            is_good = (
                self.x_limit[k][0] <= np.abs(euler_angles[0]) <= self.x_limit[k][1]
                and self.y_limit[k][0] <= np.abs(euler_angles[1]) <= self.y_limit[k][1]
                and self.z_limit[k][0] <= np.abs(euler_angles[2]) <= self.z_limit[k][1]
            )
            if is_good:
                self.last_good_matrix[k] = new_cartesian_matrix
            # else:
            #     print("Bad head angle", euler_angles)

            target[k][:3, :3] = self.last_good_matrix[k]
        return target


class GR1AngleDiscretizer(Filter):
    """For GR1 hand only."""

    def __init__(self, config):
        super().__init__(config)
        self.num_bins = config["num_bins"]
        self.real_robot = config.get("real_robot", False)

    def __call__(self, thetas):
        if self.real_robot:
            return self.discritize_real(thetas)
        else:
            return self.discritize(thetas)

    def discritize_real(self, thetas):
        thetas[1:12] = np.clip(thetas[1:12], 0, np.pi / 4)
        # 2,4,6,8,10 should be the first part of the finger
        thetas[[1, 4, 6, 8, 10]] += thetas[[2, 5, 7, 9, 11]]
        thetas[[2, 5, 7, 9, 11]] = thetas[[1, 4, 6, 8, 10]]
        bins = np.linspace(0, np.pi / 2, self.num_bins) - np.pi / (4 * self.num_bins)
        # indices = np.abs(bins - thetas[:, None]).argmin(axis=1)
        indices = np.digitize(thetas, bins) - 1
        thetas = bins[indices] + np.pi / (4 * self.num_bins)
        # raise NotImplementedError("Not implemented for real robot yet")
        return thetas

    def discritize(self, thetas):
        thetas[[2, 5, 7, 9, 11]] = thetas[[1, 4, 6, 8, 10]]
        return thetas
    
class GR1DexRetargetFilter(Filter):
    """For GR1 hands only. Adapt to dex-retarget method."""

    def __init__(self, config):
        super().__init__(config)
    
    def __call__(self, thetas):
        # thetas[[2, 5, 7, 9, 11]] = thetas[[1, 4, 6, 8, 10]]
        thetas += 10 / 180 * np.pi

        # map back to gr1 urdf
        q_gr1 = np.zeros(12)

        thumb_joints = thetas[8:12]
        index_joints = thetas[:2]
        middle_joints = thetas[2:4]
        ring_joints = thetas[6:8]
        pinky_joints = thetas[4:6]

        def clip_and_map(arr, a, b, c, d):
            a = a / 180 * np.pi
            b = b / 180 * np.pi
            c = c / 180 * np.pi
            d = d / 180 * np.pi
            return c + (np.clip(arr, a, b) - a) * (d - c) / (b - a)

        # rescale the joint angles
        thumb_joints[0] = clip_and_map(thumb_joints[0], 20, 35, 0, 90)
        thumb_joints[1] = clip_and_map(thumb_joints[1], 10, 15, 0, 30)
        index_joints[0] = clip_and_map(index_joints[0], 17, 35, 0, 75)
        middle_joints[0] = clip_and_map(middle_joints[0], 17, 35, 0, 75)
        ring_joints[0] = clip_and_map(ring_joints[0], 17, 35, 0, 75)
        pinky_joints[0] = clip_and_map(pinky_joints[0], 25, 45, 0, 75)

        # change to urdf order
        q_gr1[:4] = thumb_joints
        q_gr1[8:10] = index_joints
        q_gr1[10:12] = middle_joints
        q_gr1[4:6] = ring_joints
        q_gr1[6:8] = pinky_joints

        # clip within range
        q_gr1 = np.clip(q_gr1, 0, np.pi / 2)

        return q_gr1
