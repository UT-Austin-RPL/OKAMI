from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .filter import Filter


class CartesianFilter(Filter):
    def __init__(self, config) -> None:
        super().__init__(config)
        config = config["target_links"]
        self.keys = config.keys()
        self.history_size = {k: config[k]["history_size"] for k in self.keys}
        self.smoothing_strength = {k: config[k]["smoothing_strength"] for k in self.keys}
        self.speed_limit = {k: config[k]["speed_limit"] for k in self.keys}
        self.rotation_limit = {k: config[k]["rotation_limit"] for k in self.keys}

        self.x_limit = {k: config[k]["x"] for k in self.keys}
        self.y_limit = {k: config[k]["y"] for k in self.keys}
        self.z_limit = {k: config[k]["z"] for k in self.keys}

        self.last_good_transform = {k: np.eye(4) for k in self.keys}

        self.reset_history()

    def reset_history(self):
        self.position_history = {k: np.zeros((self.history_size[k], 3)) for k in self.keys}
        self.rotation_history = {k: None for k in self.keys}
        self.init_history = True

    def __call__(self, target):
        if self.init_history:
            self.position_history = {
                key: np.tile(target[key][:3, 3], (self.history_size[key], 1)) for key in self.keys
            }
            self.rotation_history = {key: R.from_matrix(target[key][:3, :3]) for key in self.keys}

            self.init_history = False
            return target

        # Extrapolate to predict the next cartesian positions
        for k in self.keys:
            new_cartesian_position = target[k][:3, 3]

            smoothing_strength = self.smoothing_strength[k]
            times = np.arange(self.history_size[k]).reshape(-1, 1)
            ones = np.ones((self.history_size[k], 1))
            A = np.hstack([times, ones])
            slopes = np.linalg.lstsq(A, self.position_history[k], rcond=None)[0][0]
            predicted_cartesian_positions = slopes + self.position_history[k][-1]

            # Smooth the movement
            smoothed_cartesian_positions = (
                smoothing_strength * predicted_cartesian_positions
                + (1 - smoothing_strength) * new_cartesian_position
            )

            actual_change = np.linalg.norm(
                smoothed_cartesian_positions - self.position_history[k][-1]
            )

            if actual_change < 1e-6:
                output_cartesian_positions = smoothed_cartesian_positions
            else:
                constrained_change_ratio = (
                    np.clip(actual_change, -self.speed_limit[k], self.speed_limit[k])
                    / actual_change
                )
                output_cartesian_positions = self.position_history[k][
                    -1
                ] + constrained_change_ratio * (
                    new_cartesian_position - self.position_history[k][-1]
                )

            # Use slerp to interpolate the rotation
            rot1 = self.rotation_history[k]
            new_rot_matrix = R.from_matrix(target[k][:3, :3])
            relative_rotation = new_rot_matrix * rot1.inv()

            angle_change = relative_rotation.magnitude()

            if angle_change < 1e-6:
                output_rot_matrix = new_rot_matrix
            else:
                constrained_change_ratio = (
                    np.clip(angle_change, -self.rotation_limit[k], self.rotation_limit[k])
                    / angle_change
                )

                # slerp rot1 to new_rot_matrix
                slerp = Slerp([0, 1], R.concatenate([rot1, new_rot_matrix]))
                output_rot_matrix = slerp(constrained_change_ratio)

            # limit the positions inside a specific range
            good = (
                self.x_limit[k][0] <= np.abs(output_cartesian_positions[0]) <= self.x_limit[k][1]
                and self.y_limit[k][0]
                <= np.abs(output_cartesian_positions[1])
                <= self.y_limit[k][1]
                and self.z_limit[k][0]
                <= np.abs(output_cartesian_positions[2])
                <= self.z_limit[k][1]
            )
            if not good:
                target[k] = self.last_good_transform[k]
                continue

            self.position_history[k] = np.roll(self.position_history[k], -1, axis=0)
            self.position_history[k][-1] = output_cartesian_positions
            self.rotation_history[k] = output_rot_matrix

            target[k][:3, 3] = output_cartesian_positions
            target[k][:3, :3] = output_rot_matrix.as_matrix()
            self.last_good_transform[k] = target[k]

        return target
