from typing import Any
import numpy as np

class TrajProcessorBase:
    def __init__(self):
        pass

    def normalize_traj(self, traj):
        raise NotImplementedError

    def unnormalize_traj(self, normalized_traj, new_init_point, new_end_point):
        raise NotImplementedError

class SimpleTrajProcessor(TrajProcessorBase):
    """Currently only simple implemetnation of reshaping trajectories. In the future, DMP or its equivalent can be implemetned to obtain some additional benefits, such as time-invariant, goal-convergence, etc.
    """
    def __init__(self, init_point, end_point):
        super()
        self.init_point = init_point
        self.end_point = end_point

    def normalize_traj(self, traj):
        return traj - self.init_point
    
    def unnormalize_traj(self, normalized_traj, new_init_point, new_end_point):
        new_traj = np.copy(normalized_traj)

        v1 = self.end_point - self.init_point
        v2 = new_end_point - new_init_point
        scale1 = np.linalg.norm(v1)
        scale2 = np.linalg.norm(v2)
        scaling = scale2 / scale1
        v1 = v1 / scale1
        v2 = v2 / scale2
        cos_theta = np.dot(v1, v2)
        theta = np.arccos(cos_theta)
        axis = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))

        I = np.eye(3)
        K = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
        R = I + K * np.sin(theta) + np.matmul(K, K) * (1 - np.cos(theta))
        # print("R: ", R)
        # print("scaling: ", scaling)
        for i in range(new_traj.shape[0]):
            new_traj[i] = np.matmul(R, new_traj[i]) * scaling + new_init_point
        return new_traj
    
class Interpolator:
    def __init__(self, type='linear'):
        self.type = type
        if self.type not in ['linear', 'cosine', 'min_jerk']:
            raise ValueError("Interpolation type should be either 'linear' or 'cosine' or 'min_jerk'.")
        
        if self.type == 'min_jerk':
            self.interpolator = self.linear_interpolation
        elif self.type == 'cosine':
            self.interpolator = self.cosine_interpolation
        else:
            self.interpolator = self.linear_interpolation

    def __call__(self, last_target, next_target, cur_step, cycle_len):
        return self.interpolator(last_target, next_target, cur_step, cycle_len)

    def linear_interpolation(self, last_target, next_target, cur_step, cycle_len):
        return last_target + (next_target - last_target) * (cur_step / cycle_len)

    def cosine_interpolation(self, last_target, next_target, cur_step, cycle_len):
        return last_target + (next_target - last_target) * (1 - np.cos(np.pi * cur_step / cycle_len)) / 2

    def min_jerk_interpolation(self, last_target, next_target, cur_step, cycle_len):
        return last_target + (next_target - last_target) * (
            10 * (cur_step / cycle_len) ** 3 - 15 * (cur_step / cycle_len) ** 4 + 6 * (cur_step / cycle_len) ** 5
        )