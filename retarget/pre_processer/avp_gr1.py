from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R

from retarget.robot import Robot
from retarget.utils.constants import HUMAN_FINGER_CHAIN, ROBOT_FINGER_CHAIN

from .pre_processor import PreProcessor

RIGHT_FINGER_TRANSFORM = np.array(
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
).reshape(4, 4)

LEFT_FINGER_TRANSFORM = np.array(
    [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
).reshape(4, 4)


class AVPGR1BodyPreProcessor(PreProcessor):
    # Apple vision pro pre-processor
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)

        self.head_axis_transform = np.array(
            [0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)

        self.left_wrist_axis_transform = np.array(
            [0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)

        self.right_wrist_axis_transform = np.array(
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)
        self.human_motion_amplifier = config.get("human_motion_amplifier", 1.0)

    def calibrate(self, data):
        l_elbow_pitch_idx = self.robot.joint2idx["l_elbow_pitch"]
        r_elbow_pitch_idx = self.robot.joint2idx["r_elbow_pitch"]

        q = deepcopy(self.robot.q0)
        q[l_elbow_pitch_idx] = np.pi / 2
        q[r_elbow_pitch_idx] = -np.pi / 2
        self.robot.update(q)
        (
            robot_head_pose,
            robot_l_wrist_pose,
            robot_r_wrist_pose,
        ) = self.robot.get_link_transformations(["link_head_pitch", "link_LArm7", "link_RArm7"])

        head_pose = data["head"][0] @ self.head_axis_transform
        head_pose_rot = R.from_matrix(head_pose[:3, :3]).as_euler("xyz", degrees=True)
        # set x, y to 0
        head_pose[:3, :3] = R.from_euler("xyz", [0, 0, head_pose_rot[2]], degrees=True).as_matrix()
        self.relative_transfrom = robot_head_pose @ np.linalg.inv(head_pose)
        # print(self.relative_transfrom)
        # breakpoint()

        left_wrist_pose = self.relative_transfrom @ (
            data["left_wrist"][0] @ self.left_wrist_axis_transform
        )
        right_wrist_pose = self.relative_transfrom @ (
            data["right_wrist"][0] @ self.right_wrist_axis_transform
        )

        self.left_wrist_delta = deepcopy(left_wrist_pose[:3, 3] - robot_l_wrist_pose[:3, 3])
        self.right_wrist_delta = deepcopy(right_wrist_pose[:3, 3] - robot_r_wrist_pose[:3, 3])

        # self.init_left_wrist_pos = left_wrist_pose[:3, 3].copy()
        # self.init_right_wrist_pos = right_wrist_pose[:3, 3].copy()
        # self.init_robot_left_wrist_pos = robot_l_wrist_pose[:3, 3].copy()
        # self.init_robot_right_wrist_pos = robot_r_wrist_pose[:3, 3].copy()

        # human_center = (left_wrist_pose[:3, 3] + right_wrist_pose[:3, 3]) / 2
        # robot_center = (robot_l_wrist_pose[:3, 3] + robot_r_wrist_pose[:3, 3]) / 2
        # human_dist = np.linalg.norm(left_wrist_pose[:3, 3] - right_wrist_pose[:3, 3])
        # robot_dist = np.linalg.norm(robot_l_wrist_pose[:3, 3] - robot_r_wrist_pose[:3, 3])
        # self.rel_dist = robot_dist / human_dist
        self.rel_dist = 1.0
        # self.center_delta = robot_center - human_center

    def __call__(self, data):
        left_wrist_pose = self.relative_transfrom @ (
            data["left_wrist"][0] @ self.left_wrist_axis_transform
        )
        right_wrist_pose = self.relative_transfrom @ (
            data["right_wrist"][0] @ self.right_wrist_axis_transform
        )
        head_pose = self.relative_transfrom @ (data["head"][0] @ self.head_axis_transform)

        left_wrist_pose[:3, 3] -= self.left_wrist_delta
        right_wrist_pose[:3, 3] -= self.right_wrist_delta

        # left_wrist_pos = left_wrist_pose[:3, 3].copy()
        # left_wrist_pose[:3, 3] = (
        #     left_wrist_pos - self.init_left_wrist_pos
        # ) * self.rel_dist * self.human_motion_amplifier + self.init_robot_left_wrist_pos

        # right_wrist_pos = right_wrist_pose[:3, 3].copy()
        # right_wrist_pose[:3, 3] = (
        #     right_wrist_pos - self.init_right_wrist_pos
        # ) * self.rel_dist * self.human_motion_amplifier + self.init_robot_right_wrist_pos

        return {
            "link_LArm7": left_wrist_pose,
            "link_RArm7": right_wrist_pose,
            "link_head_pitch": head_pose,
        }


class AVPVuerGR1BodyPreProcessor(PreProcessor):
    # Apple vision pro pre-processor
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)

        self.global_transform = np.array(
            [0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)

        self.head_axis_transform = np.array(
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)

        self.left_wrist_axis_transform = np.array(
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)

        self.right_wrist_axis_transform = np.array(
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ).reshape(4, 4)
        self.human_motion_amplifier = config.get("human_motion_amplifier", 1.0)

    def calibrate(self, data):
        l_elbow_pitch_idx = self.robot.joint2idx["l_elbow_pitch"]
        r_elbow_pitch_idx = self.robot.joint2idx["r_elbow_pitch"]

        q = deepcopy(self.robot.q0)
        q[l_elbow_pitch_idx] = np.pi / 2
        q[r_elbow_pitch_idx] = -np.pi / 2
        self.robot.update(q)
        (
            robot_head_pose,
            robot_l_wrist_pose,
            robot_r_wrist_pose,
        ) = self.robot.get_link_transformations(["link_head_pitch", "link_LArm7", "link_RArm7"])

        head_pose = self.global_transform @ data["head"] @ self.head_axis_transform
        head_pose_rot = R.from_matrix(head_pose[:3, :3]).as_euler("xyz", degrees=True)
        # set x, y to 0
        head_pose[:3, :3] = R.from_euler("xyz", [0, 0, head_pose_rot[2]], degrees=True).as_matrix()
        self.relative_transfrom = robot_head_pose @ np.linalg.inv(head_pose)
        # print(self.relative_transfrom)
        # breakpoint()

        left_wrist_pose = (
            self.relative_transfrom
            @ self.global_transform
            @ (data["left_hand"] @ self.left_wrist_axis_transform)
        )
        right_wrist_pose = (
            self.relative_transfrom
            @ self.global_transform
            @ (data["right_hand"] @ self.right_wrist_axis_transform)
        )

        self.left_wrist_delta = deepcopy(left_wrist_pose[:3, 3] - robot_l_wrist_pose[:3, 3])
        self.right_wrist_delta = deepcopy(right_wrist_pose[:3, 3] - robot_r_wrist_pose[:3, 3])
        left_delta_len = np.linalg.norm(self.left_wrist_delta)
        right_delta_len = np.linalg.norm(self.right_wrist_delta)
        if left_delta_len < 0.1 and right_delta_len < 0.1:
            self.left_wrist_delta = np.zeros(3)
            self.right_wrist_delta = np.zeros(3)
        else:
            print(f"Left wrist delta: {self.left_wrist_delta}")
            print(f"Right wrist delta: {self.right_wrist_delta}")

        # self.init_left_wrist_pos = left_wrist_pose[:3, 3].copy()
        # self.init_right_wrist_pos = right_wrist_pose[:3, 3].copy()
        # self.init_robot_left_wrist_pos = robot_l_wrist_pose[:3, 3].copy()
        # self.init_robot_right_wrist_pos = robot_r_wrist_pose[:3, 3].copy()

        # human_center = (left_wrist_pose[:3, 3] + right_wrist_pose[:3, 3]) / 2
        # robot_center = (robot_l_wrist_pose[:3, 3] + robot_r_wrist_pose[:3, 3]) / 2
        # human_dist = np.linalg.norm(left_wrist_pose[:3, 3] - right_wrist_pose[:3, 3])
        # robot_dist = np.linalg.norm(robot_l_wrist_pose[:3, 3] - robot_r_wrist_pose[:3, 3])
        # self.rel_dist = robot_dist / human_dist
        self.rel_dist = 1.0
        # self.center_delta = robot_center - human_center

    def __call__(self, data):
        left_wrist_pose = (
            self.relative_transfrom
            @ self.global_transform
            @ (data["left_hand"] @ self.left_wrist_axis_transform)
        )
        right_wrist_pose = (
            self.relative_transfrom
            @ self.global_transform
            @ (data["right_hand"] @ self.right_wrist_axis_transform)
        )

        # _, lh_pose, rh_pose = self.robot.get_link_transformations(["link_head_pitch", "link_LArm7", "link_RArm7"])
        # print(np.linalg.inv(left_wrist_pose) @ lh_pose)
        # print(np.linalg.inv(right_wrist_pose) @ rh_pose)
        # breakpoint()

        # left_wrist_pose = self.relative_transfrom @ self.global_transform @ data["left_hand"]
        # right_wrist_pose = self.relative_transfrom @ self.global_transform @ data["right_hand"]

        head_pose = (
            self.relative_transfrom
            @ self.global_transform
            @ (data["head"] @ self.head_axis_transform)
        )
        # left_wrist_pose = data["left_hand"]
        # right_wrist_pose = data["right_hand"]
        # head_pose = data["head"] # @ self.head_axis_transform
        # left_wrist_pose = self.global_transform @ data["left_hand"]
        # right_wrist_pose = self.global_transform @ data["right_hand"]
        # head_pose = self.global_transform @ data["head"] @ self.head_axis_transform

        left_wrist_pose[:3, 3] -= self.left_wrist_delta
        right_wrist_pose[:3, 3] -= self.right_wrist_delta

        # left_wrist_pos = left_wrist_pose[:3, 3].copy()
        # left_wrist_pose[:3, 3] = (
        #     left_wrist_pos - self.init_left_wrist_pos
        # ) * self.rel_dist * self.human_motion_amplifier + self.init_robot_left_wrist_pos

        # right_wrist_pos = right_wrist_pose[:3, 3].copy()
        # right_wrist_pose[:3, 3] = (
        #     right_wrist_pos - self.init_right_wrist_pos
        # ) * self.rel_dist * self.human_motion_amplifier + self.init_robot_right_wrist_pos

        return {
            "link_LArm7": left_wrist_pose,
            "link_RArm7": right_wrist_pose,
            "link_head_pitch": head_pose,
        }


class AVPGR1HandPreProcessor(PreProcessor):
    # Apple vision pro pre-processor
    def __init__(self, config, hand: Robot):
        super().__init__(config, hand)
        self.side = "L" if config["side"].lower() in ["left", "l"] else "R"
        self.relative_transform = (
            np.array(
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ).reshape(4, 4)
            if self.side == "L"
            else np.array(
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ).reshape(4, 4)
        )
        self.finger_link = [f"link_{self.side}Arm7"] + [
            f"link_{self.side}Arm{i}" for i in range(10, 25)
        ]
        self.ratio = None

    def __call__(self, data):
        side = "left" if self.side == "L" else "right"
        data_new = deepcopy(data[f"{side}_fingers"][HUMAN_FINGER_CHAIN[:, -1], :, :])
        data_new = self.relative_transform @ data_new
        data_new[:, :3, 3] = data_new[:, :3, 3] * self.ratio
        targets = {f"link_{self.side}Arm{i*3+12}": finger for i, finger in enumerate(data_new)}
        return targets

    def calibrate(self, data):
        ratio_list = []
        finger_pose = self.hand.get_link_transformations(self.finger_link)
        for i in range(5):
            human_tip = data["left_fingers"][HUMAN_FINGER_CHAIN[i][-1]][:3, 3]
            robot_tip = finger_pose[ROBOT_FINGER_CHAIN[i][-1]][:3, 3]
            human_length = np.linalg.norm(human_tip)
            robot_length = np.linalg.norm(robot_tip)
            ratio = robot_length / human_length
            ratio_list.append([ratio])
            print(f"Ratio of finger {i}: {ratio}")
        self.ratio = np.array(ratio_list)


class AVPGR1HandAnglePreProcessor(PreProcessor):
    """Dummy class just takes out the fingers from the data."""

    def __init__(self, config, hand: Robot):
        super().__init__(config, hand)
        self.side = "left" if config["side"].lower() in ["left", "l"] else "right"

    def __call__(self, data):
        return data[f"{self.side}_fingers"]

    # no need to calibrate
    def calibrate(self, data):
        pass


class AVPVuerGR1HandAnglePreProcessor(PreProcessor):
    """Dummy class just takes out the fingers from the data."""

    def __init__(self, config, hand: Robot):
        super().__init__(config, hand)
        self.side = "left" if config["side"].lower() in ["left", "l"] else "right"

    def __call__(self, data):
        trans = np.linalg.inv(data[f"{self.side}_hand"])
        # left_landmarks are 25*3, make it 25*4*4
        landmarks = np.zeros((25, 4, 4))
        landmarks[:, :3, 3] = data[f"{self.side}_landmarks"]
        landmarks[:, 3, 3] = 1
        landmarks[:, :3, :3] = np.eye(3)
        left_fingers = trans @ landmarks
        return left_fingers

    # no need to calibrate
    def calibrate(self, data):
        pass
