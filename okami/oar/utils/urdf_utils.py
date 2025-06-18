import numpy as np

def dex_mapping(q):
    '''
    Map the joint positions of dexterous hands from robosuite observation to urdf order.
    Args:
        q (np.array): joint positions of dexterous hands from robosuite observation (12-dim).
    Returns:
        q_urdf (np.array): joint positions of dexterous hands in urdf order (12-dim).
    '''
    q_urdf = np.zeros(12)

    thumb_joints = q[:4]
    index_joints = q[4:6]
    middle_joints = q[6:8]
    ring_joints = q[8:10]
    pinky_joints = q[10:]

    q_urdf[:4] = thumb_joints
    q_urdf[8:10] = index_joints
    q_urdf[10:12] = middle_joints
    q_urdf[4:6] = ring_joints
    q_urdf[6:8] = pinky_joints

    return q_urdf

def obs_to_urdf(obs):
    """
    Convert joint positions in obs into URDF joints format.
    """
    right_arm_joints = obs["robot0_joint_pos"][6:13].copy()
    left_arm_joints = obs["robot0_joint_pos"][13:20].copy()

    right_gripper_joints = dex_mapping(obs["robot0_right_gripper_qpos"])
    left_gripper_joints = dex_mapping(obs["robot0_left_gripper_qpos"])

    head_joints = obs["robot0_joint_pos"][3:6].copy()
    waist_joints = obs["robot0_joint_pos"][0:3].copy()

    q = np.zeros(56)
    q[:3] = waist_joints
    q[3:6] = head_joints
    q[6:13] = left_arm_joints
    q[13:25] = left_gripper_joints
    q[25:32] = right_arm_joints
    q[32:44] = right_gripper_joints

    return q.copy()

def append_hand_q(current_pos, l_q, r_q):
    res = current_pos.copy()
    res[13:25] = l_q
    res[32:44] = r_q
    return res

def parse_hand_q(q):
    '''
    Given 56-dim q, return left and right hand q (12-dim each)
    '''
    return q[13:25].copy(), q[32:44].copy()

def urdf_to_robosuite_cmds(urdf_q):
    """
    Generate 35-dim robosuite joint position cmds from 56-dim urdf joint position commmands.
    """
    waist_joints = urdf_q[:3]
    head_joints = urdf_q[3:6]
    left_arm_joints = urdf_q[6:13]
    left_gripper_joints = urdf_q[13:25]
    right_arm_joints = urdf_q[25:32]
    right_gripper_joints = urdf_q[32:44]

    actuator_idxs = [0, 1, 8, 10, 4, 6]
    right_gripper_actuator_joints = right_gripper_joints[actuator_idxs]
    left_gripper_actuator_joints = left_gripper_joints[actuator_idxs]

    action = np.zeros(35)
    action[:7] = right_arm_joints
    action[7:14] = left_arm_joints
    action[14:20] = right_gripper_actuator_joints
    action[20:26] = left_gripper_actuator_joints
    action[29:32] = head_joints
    action[32:35] = waist_joints

    return action.copy()

def robosuite_cmds_to_body_cmds(cmds):
    """
    Convert 35-dim robosuite joint position cmds to 20-dim body joint position cmds.
    """
    right_arm_joints = cmds[:7]
    left_arm_joints = cmds[7:14]
    head_joints = cmds[29:32]
    waist_joints = cmds[32:35]
    
    body_cmds = np.zeros(20)
    body_cmds[:3] = waist_joints
    body_cmds[3:6] = head_joints
    body_cmds[6:13] = right_arm_joints
    body_cmds[13:] = left_arm_joints

    return body_cmds


def obs_to_robosuite_cmds(obs):
    """
    Generate 35-dim robosuite joint position cmds from observation.
    """
    q = obs_to_urdf(obs)
    return urdf_to_robosuite_cmds(q)

def extend_robosuite_finger_cmds(finger_cmds):
    """
    Convert 6-dim finger cmds to 12-dim finger cmds in robosuite observation.
    """
    mapping_base = [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    mapping_scale = [1, 1, 1, 1.13, 1, 1.13, 1, 1.13, 1, 1.08, 1, 1.15]

    actions = finger_cmds[mapping_base] * mapping_scale

    return np.array(actions)

def extend_urdf_finger_cmds(finger_cmds):
    """
    Convert 6-dim finger cmds to 12-dim finger cmds in urdf.
    """
    mapping_base = [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    mapping_scale = [1, 1, 1, 1.13, 1, 1.08, 1, 1.15, 1, 1.13, 1, 1.13]

    actions = finger_cmds[mapping_base] * mapping_scale

    return np.array(actions)

def robosuite_cmds_to_urdf(cmds):
    """
    Generate 56-dim urdf joint position cmds from 35-dim robosuite joint position commmands.
    """
    right_arm_joints = cmds[:7]
    left_arm_joints = cmds[7:14]
    right_gripper_actuator_joints = cmds[14:20]
    left_gripper_actuator_joints = cmds[20:26]
    head_joints = cmds[29:32]
    waist_joints = cmds[32:35]

    right_gripper_joints = dex_mapping(extend_robosuite_finger_cmds(right_gripper_actuator_joints))
    left_gripper_joints = dex_mapping(extend_robosuite_finger_cmds(left_gripper_actuator_joints))

    q = np.zeros(56)
    q[:3] = waist_joints
    q[3:6] = head_joints
    q[6:13] = left_arm_joints
    q[13:25] = left_gripper_joints
    q[25:32] = right_arm_joints
    q[32:44] = right_gripper_joints

    return q.copy()

def parse_urdf_cmds(q):
    """
    Parse out hand and body joints from 56-dim urdf joint position cmds.
    """

    body_joints = np.concatenate([q[:3], q[3:6], q[6:13], q[25:32]])

    left_gripper_joints = q[13:25]
    right_gripper_joints = q[32:44]

    actuator_idxs = [0, 1, 8, 10, 4, 6]
    right_gripper_actuator_joints = right_gripper_joints[actuator_idxs]
    left_gripper_actuator_joints = left_gripper_joints[actuator_idxs]

    hand_joints = np.concatenate([left_gripper_actuator_joints, right_gripper_actuator_joints])

    return body_joints, hand_joints

def urdf_arm_hand_idxs(lr):

    name_mapping = {
        'L': 'l',
        'R': 'r',
        'l': 'l',
        'r': 'r',
        'left': 'l',
        'right': 'r',
        0: 'l',
        1: 'r'
    }

    assert lr in name_mapping, f'Invalid lr: {lr}'
    lr = name_mapping[lr]
    
    arm_idx = []
    hand_idx = []
    for key in name_to_urdf_idx.keys():
        if key.startswith(lr):
            arm_idx.append(name_to_urdf_idx[key])
        if key.startswith('joint_' + lr[0].upper()):
            hand_idx.append(name_to_urdf_idx[key])
    
    return arm_idx, hand_idx

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

name_to_limits = {
    'l_hip_roll': (-0.08726646259971647, 0.7853981633974483), 
    'l_hip_yaw': (-0.6981317007977318, 0.6981317007977318), 
    'l_hip_pitch': (-1.7453292519943295, 0.6981317007977318), 
    'l_knee_pitch': (-0.08726646259971647, 1.9198621771937625), 
    'l_ankle_pitch': (-1.0471975511965976, 0.5235987755982988), 
    'l_ankle_roll': (-0.4363323129985824, 0.4363323129985824), 
    'r_hip_roll': (-0.7853981633974483, 0.08726646259971647), 
    'r_hip_yaw': (-0.6981317007977318, 0.6981317007977318), 
    'r_hip_pitch': (-1.7453292519943295, 0.6981317007977318), 
    'r_knee_pitch': (-0.08726646259971647, 1.9198621771937625), 
    'r_ankle_pitch': (-1.0471975511965976, 0.5235987755982988), 
    'r_ankle_roll': (-0.4363323129985824, 0.4363323129985824), 
    'joint_waist_yaw': (-1.0471975511965976, 1.0471975511965976), 
    'joint_waist_pitch': (-0.5235987755982988, 1.2217304763960306), 
    'joint_waist_roll': (-0.6981317007977318, 0.6981317007977318), 
    'joint_head_yaw': (-2.705260340591211, 2.705260340591211), 
    'joint_head_roll': (-0.3490658503988659, 0.3490658503988659), 
    'joint_head_pitch': (-0.5235987755982988, 0.3490658503988659), 
    'l_shoulder_pitch': (-1.0471975511965976, 2.6179938779914944), 
    'l_shoulder_roll': (-2.4085543677521746, 0.20943951023931956), 
    'l_shoulder_yaw': (-1.5707963267948966, 1.5707963267948966), 
    'l_elbow_pitch': (0.0, 1.5707963267948966), 
    'l_wrist_yaw': (-1.5707963267948966, 1.5707963267948966), 
    'l_wrist_roll': (-0.3665191429188092, 0.3665191429188092), 
    'l_wrist_pitch': (-0.3665191429188092, 0.3665191429188092), 
    'joint_LFinger0': (0.0, 1.2915436464758039), 
    'joint_LFinger1': (0.0, 0.6806784082777885), 
    'joint_LFinger2': (0.0, 0.767944870877505), 
    'joint_LFinger3': (0.0, 0.5934119456780721), 
    'joint_LFinger5': (0.0, 1.6231562043547265), 
    'joint_LFinger6': (0.0, 1.8151424220741028), 
    'joint_LFinger8': (0.0, 1.6231562043547265), 
    'joint_LFinger9': (0.0, 1.7453292519943295), 
    'joint_LFinger11': (0.0, 1.6231562043547265), 
    'joint_LFinger12': (0.0, 1.7453292519943295), 
    'joint_LFinger14': (0.0, 1.6231562043547265), 
    'joint_LFinger15': (0.0, 1.8675022996339325), 
    'r_shoulder_pitch': (-2.6179938779914944, 1.0471975511965976), 
    'r_shoulder_roll': (-0.20943951023931956, 2.4085543677521746), 
    'r_shoulder_yaw': (-1.5707963267948966, 1.5707963267948966), 
    'r_elbow_pitch': (-1.5707963267948966, 0.0), 
    'r_wrist_yaw': (-1.5707963267948966, 1.5707963267948966), 
    'r_wrist_roll': (-0.3665191429188092, 0.3665191429188092), 
    'r_wrist_pitch': (-0.3665191429188092, 0.3665191429188092), 
    'joint_RFinger0': (0.0, 1.2915436464758039), 
    'joint_RFinger1': (0.0, 0.6806784082777885), 
    'joint_RFinger2': (0.0, 0.767944870877505), 
    'joint_RFinger3': (0.0, 0.5934119456780721), 
    'joint_RFinger5': (0.0, 1.6231562043547265), 
    'joint_RFinger6': (0.0, 1.8151424220741028), 
    'joint_RFinger8': (0.0, 1.6231562043547265), 
    'joint_RFinger9': (0.0, 1.7453292519943295), 
    'joint_RFinger11': (0.0, 1.6231562043547265), 
    'joint_RFinger12': (0.0, 1.7453292519943295), 
    'joint_RFinger14': (0.0, 1.6231562043547265), 
    'joint_RFinger15': (0.0, 1.8675022996339325)
}