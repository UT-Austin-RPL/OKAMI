import numpy as np

from .filter import Filter


class JointPositionFilter(Filter):
    def __init__(self, config):
        super().__init__(config)
        self.history_size = config["history_size"]
        self.smoothing_strength = config["smoothing_strength"]
        self.outlier_threshold = config["outlier_threshold"]
        self.speed_limit = config["speed_limit"]
        self.joint_size = config["joint_size"]
        self.verbose = config.get("verbose", False)
        self.outlier_counter = np.zeros(self.history_size)
        joint_mask = config.get("joint_mask", None)
        self.joint_mask = np.ones(self.joint_size).astype(bool)
        if joint_mask is not None:
            self.joint_mask[joint_mask] = False

        self.reset_history()

    def reset_history(self):
        self.init_history = True
        self.joint_history = np.zeros((self.history_size, self.joint_size))
        self.outlier_counter = np.zeros(self.history_size)

    def __call__(self, new_joint_positions):
        # takes about 0.5ms
        # Add the new joint positions to the history
        if self.init_history:
            self.joint_history = np.tile(new_joint_positions, (self.history_size, 1))
            self.init_history = False

        # Extrapolate to predict the next joint positions
        # Linear regression on the history to predict the next point
        time = np.arange(self.history_size).reshape(-1, 1)  # Column vector
        ones = np.ones((self.history_size, 1))  # Column vector of ones for the intercept
        A = np.hstack([time, ones])  # Design matrix for linear regression
        # Perform linear regression for all joints simultaneously
        slopes = np.linalg.lstsq(A, self.joint_history, rcond=None)[0][0]
        predicted_joint_positions = slopes + self.joint_history[-1]

        # Detect and handle outliers
        delta = new_joint_positions[self.joint_mask] - predicted_joint_positions[self.joint_mask]
        is_outlier = np.max(np.abs(delta)) > self.outlier_threshold

        self.outlier_counter = np.roll(self.outlier_counter, -1)
        if is_outlier:
            if self.verbose:
                print(f"Outlier detected")
            output_joint_positions = predicted_joint_positions
            self.outlier_counter[-1] = 1
            self.outlier_counter[-1] = 1

        else:
            # Smooth the movement
            smoothed_joint_positions = (
                self.smoothing_strength * predicted_joint_positions
                + (1 - self.smoothing_strength) * new_joint_positions
            )
            self.outlier_counter[-1] = 0

            # Constrain changes based on the speed limit
            max_change = self.speed_limit
            actual_change = smoothed_joint_positions - self.joint_history[-1]
            constrained_change = np.clip(actual_change, -max_change, max_change)
            output_joint_positions = self.joint_history[-1] + constrained_change

        # Update the history
        self.joint_history = np.roll(self.joint_history, -1, axis=0)
        self.joint_history[-1] = output_joint_positions

        if np.all(self.outlier_counter > 0.5):
            print("All outliers, resetting history")
            self.reset_history()

        return output_joint_positions
