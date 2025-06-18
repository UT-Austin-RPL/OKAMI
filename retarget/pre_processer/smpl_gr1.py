from copy import deepcopy

import numpy as np

from retarget.robot import Robot

from .pre_processor import PreProcessor

SMPL_WRIST = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


class SMPLGR1Preprocessor(PreProcessor):
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)

        self.left_shoulder = "l_shoulder_roll"
        self.right_shoulder = "r_shoulder_roll"

        self.joint2idx = {
            "link_LArm1": 13,
            "link_RArm1": 14,
            "link_LArm2": 16,
            "link_RArm2": 17,
            "link_LArm4": 18,
            "link_RArm4": 19,
            "link_LArm7": 20,
            "link_RArm7": 21,
        }

    def calibrate(self, data):
        self.init_pelvis_pos = data["body"][0][:3, 3]

        # set to T pose
        q0 = deepcopy(self.robot.q0)
        q0[self.robot.joint2idx[self.left_shoulder]] = -np.pi / 2
        q0[self.robot.joint2idx[self.right_shoulder]] = np.pi / 2
        self.robot.update(q0)

        # assume the first frame is in T pose
        # calculate wrist rotation matrix
        left_wrist, right_wrist = self.robot.get_link_transformations(["link_LArm7", "link_RArm7"])
        # smpl_wrist_left = data[20][:3, :3]
        # smpl_wrist_right = data[21][:3, :3]
        self.left_rotation = np.linalg.inv(SMPL_WRIST) @ left_wrist[:3, :3]
        self.right_rotation = np.linalg.inv(SMPL_WRIST) @ right_wrist[:3, :3]

    def fix_transformation(self, data):
        data[:, :3, 3] -= self.init_pelvis_pos
        for link_name, idx in self.joint2idx.items():
            if idx >= len(data):
                continue
            if link_name[5] == "L":
                data[idx][:3, :3] = data[idx][:3, :3] @ self.left_rotation
            elif link_name[5] == "R":
                data[idx][:3, :3] = data[idx][:3, :3] @ self.right_rotation

        return data

    # def fix_position(self, position):
    #     position = position.copy()
    #     # use two shoulder to center the position
    #     left_shoulder = self.robot.get_transform(self.left_shoulder).translation
    #     right_shoulder = self.robot.get_transform(self.right_shoulder).translation
    #     middle_point = (left_shoulder + right_shoulder) / 2
    #     delta = (
    #         position[self.smpl_l_shoulder_idx] + position[self.smpl_r_shoulder_idx]
    #     ) / 2 - middle_point
    #     # delta = position[0] - self.get_transform("base").translation
    #     # print(delta)
    #     position -= delta
    #     return position

    def __call__(self, data):
        # position = data[:, :3, 3]
        # orientation = data[:, :3, :3]
        data_new = deepcopy(data["body"])
        data_new = self.fix_transformation(data_new)
        target = {}
        for joint, idx in self.joint2idx.items():
            if idx >= len(data_new):
                continue
            target[joint] = data_new[idx]

        return target


class SMPLGR1HandPreProcessor(PreProcessor):
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)
        self.idx = [12, 14, 0, 1, 3, 4, 9, 10, 6, 7]
        self.side = "L" if config["side"].lower() in ["left", "l"] else "R"
        self.finger_matchings = {
            "thumb": [(f"link_{self.side}Arm8", 12), (f"link_{self.side}Arm10", 14)],
            "index": [(f"link_{self.side}Arm13", 0), (f"link_{self.side}Arm14", 1)],
            "middle": [(f"link_{self.side}Arm16", 3), (f"link_{self.side}Arm17", 4)],
            "ring": [(f"link_{self.side}Arm19", 9), (f"link_{self.side}Arm20", 10)],
            "pinky": [(f"link_{self.side}Arm22", 6), (f"link_{self.side}Arm23", 7)],
        }
        self.relative_trans = (
            np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                ]
            )
            if self.side == "L"
            else np.array(
                [
                    [0, 1, 0],
                    [0, 0, -1],
                    [-1, 0, 0],
                ]
            )
        )

    def calibrate(self, data):
        pass

    def __call__(self, data) -> dict:
        side = "left" if self.side == "L" else "right"
        data_new = deepcopy(data[f"{side}_fingers"])
        data_new[:, :3, 3] = np.einsum("ij,nj->ni", self.relative_trans, data_new[:, :3, 3])

        targets = {}
        for _, f2idx_list in self.finger_matchings.items():
            root_link, root_link_idx = f2idx_list[0]
            root_link_pose = self.robot.get_link_transformations([root_link])[0]
            offset = root_link_pose[:3, 3] - data_new[root_link_idx, :3, 3]
            for link_name, link_idx in f2idx_list[1:]:
                target_pose = data_new[link_idx]
                target_pose[:3, 3] += offset
                targets[link_name] = target_pose

        return targets


class SMPLGR1HandAnglePreProcessor(PreProcessor):
    def __init__(self, config, hand: Robot):
        super().__init__(config, hand)
        self.side = "left" if config["side"].lower() in ["left", "l"] else "right"
        self.hand_idx = (
            [20] + list(range(22, 37)) if self.side == "left" else [21] + list(range(37, 52))
        )

    def __call__(self, data):
        return {
            "angles": data[f"{self.side}_angles"].reshape(-1, 3),
            "transformations": data["body"][self.hand_idx],
        }

    # no need to calibrate
    def calibrate(self, data):
        pass


class SMPLGR1HandHybridPreProcessor(PreProcessor):
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)
        self.ik_preprocessor = SMPLGR1HandPreProcessor(config, robot)
        self.angle_preprocessor = SMPLGR1HandAnglePreProcessor(config, robot)

    def calibrate(self, data):
        self.ik_preprocessor.calibrate(data)
        self.angle_preprocessor.calibrate(data)

    def __call__(self, data) -> dict:
        ik_target = self.ik_preprocessor(data)
        angle_target = self.angle_preprocessor(data)
        return {**ik_target, **angle_target}

class SMPLGR1HandDexRetargetPreProcessor(PreProcessor):
    def __init__(self, config, robot: Robot):
        super().__init__(config, robot)
        self.side = "left" if config["side"].lower() in ["left", "l"] else "right"
        self.relative_trans = (
            np.array(
                [
                    [0, -1, 0],
                    [0, 0, -1],
                    [1, 0, 0],
                ]
            )
            if self.side == "left"
            else np.array(
                [
                    [0, -1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                ]
            )
        )

    def calibrate(self, data):
        pass

    def __call__(self, data) -> dict:
        finger_positions = data[f"{self.side}_fingers"][:, :3, 3].copy()
        finger_positions = np.einsum("ij,nj->ni", self.relative_trans, finger_positions)
        return {
            "finger_positions": np.vstack((finger_positions,[[0, 0, 0]])), 
        }