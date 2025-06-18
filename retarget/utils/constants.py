import numpy as np

HUMAN_FINGER_CHAIN = np.array(
    [
        [0, 0, 1, 2, 3, 4],  # add a dummy 0 to make size consistent
        [0, 5, 6, 7, 8, 9],
        [0, 10, 11, 12, 13, 14],
        [0, 15, 16, 17, 18, 19],
        [0, 20, 21, 22, 23, 24],
    ]
)

ROBOT_FINGER_CHAIN = np.array(
    [
        [0, 1, 2, 3],
        [0, 4, 5, 6],
        [0, 7, 8, 9],
        [0, 10, 11, 12],
        [0, 13, 14, 15],
    ]
)

name_to_urdf_idx = {
    "joint_waist_yaw": 0,
    "joint_waist_pitch": 1,
    "joint_waist_roll": 2,
    "joint_head_yaw": 3,
    "joint_head_roll": 4,
    "joint_head_pitch": 5,
    "l_shoulder_pitch": 6,
    "l_shoulder_roll": 7,
    "l_shoulder_yaw": 8,
    "l_elbow_pitch": 9,
    "l_wrist_yaw": 10,
    "l_wrist_roll": 11,
    "l_wrist_pitch": 12,
    "joint_LFinger0": 13,
    "joint_LFinger1": 14,
    "joint_LFinger2": 15,
    "joint_LFinger3": 16,
    "joint_LFinger11": 17,
    "joint_LFinger12": 18,
    "joint_LFinger14": 19,
    "joint_LFinger15": 20,
    "joint_LFinger5": 21,
    "joint_LFinger6": 22,
    "joint_LFinger8": 23,
    "joint_LFinger9": 24,
    "r_shoulder_pitch": 25,
    "r_shoulder_roll": 26,
    "r_shoulder_yaw": 27,
    "r_elbow_pitch": 28,
    "r_wrist_yaw": 29,
    "r_wrist_roll": 30,
    "r_wrist_pitch": 31,
    "joint_RFinger0": 32,
    "joint_RFinger1": 33,
    "joint_RFinger2": 34,
    "joint_RFinger3": 35,
    "joint_RFinger11": 36,
    "joint_RFinger12": 37,
    "joint_RFinger14": 38,
    "joint_RFinger15": 39,
    "joint_RFinger5": 40,
    "joint_RFinger6": 41,
    "joint_RFinger8": 42,
    "joint_RFinger9": 43,
    "l_hip_roll": 44,
    "l_hip_yaw": 45,
    "l_hip_pitch": 46,
    "l_knee_pitch": 47,
    "l_ankle_pitch": 48,
    "l_ankle_roll": 49,
    "r_hip_roll": 50,
    "r_hip_yaw": 51,
    "r_hip_pitch": 52,
    "r_knee_pitch": 53,
    "r_ankle_pitch": 54,
    "r_ankle_roll": 55,
}

arm_joints = [
    "l_shoulder_pitch",
    "l_shoulder_roll",
    "l_shoulder_yaw",
    "l_elbow_pitch",
    "l_wrist_yaw",
    "l_wrist_roll",
    "l_wrist_pitch",
    "r_shoulder_pitch",
    "r_shoulder_roll",
    "r_shoulder_yaw",
    "r_elbow_pitch",
    "r_wrist_yaw",
    "r_wrist_roll",
    "r_wrist_pitch",
]

waist_joints = ["joint_waist_yaw", "joint_waist_pitch", "joint_waist_roll"]

head_joints = ["joint_head_yaw", "joint_head_roll", "joint_head_pitch"]

leg_joints = [
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee_pitch",
    "l_ankle_pitch",
    "l_ankle_roll",
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee_pitch",
    "r_ankle_pitch",
    "r_ankle_roll",
]

finger_joints = [
    "joint_LFinger0",  # thumb rot
    "joint_LFinger1",  # thumb bend
    "joint_LFinger5",  # index bend
    "joint_LFinger8",  # middle bend
    "joint_LFinger11",  # ring bend
    "joint_LFinger14",  # pinky bend
    "joint_RFinger0",  # thumb rot
    "joint_RFinger1",  # thumb bend
    "joint_RFinger5",  # index bend
    "joint_RFinger8",  # middle bend
    "joint_RFinger11",  # ring bend
    "joint_RFinger14",  # pinky bend
]
