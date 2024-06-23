import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

from robot.gr1 import GR1URDFModel
from utils.grasp_offset_utils import up_or_side_grasp, calculate_wrist_offset_relative_to_object_in_head_frame
from utils.interpolate_utils import append_hand_pos, H_pos_joints, L_pos_joints
from utils.frame_transformation_utils import GR1Transform
from utils.pose_utils import clip_wrist_pose_in_base
from algos.hoig import HOIG

from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config
import retarget

from gr1_interface.gr1_control.utils.variables import (
    finger_joints,
    name_to_limits,
    name_to_sign,
    name_to_urdf_idx,
)

def offset_oar(step_info: HOIG, start_q, end_pos_in_head, gr1_transform: GR1Transform):
    '''
    Given the HOIG information of one step, and the new target position obtained from vision pipeline, do object-aware retargeting by adding an offset to the original trajectory.
    
    Return: the new trajectory after retargeting. np.array of shape (n, 56)
    '''
    
    offset_in_head = end_pos_in_head - step_info.original_end_pos_in_head
    offset_in_base = gr1_transform.apply_transform_to_point(offset_in_head, 'head', 'chest')

    offset = {
        f'''link_{step_info.lr}Arm7''': offset_in_base.tolist(),
    }
    print("offset=", offset)

    retarget_repo_dir = os.path.dirname(retarget.__file__)
    config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1_translation.yaml")
    config = load_config(config_path)
    for joint_name in offset:
        config['GR1_body']['pre_processor']['offset'][joint_name] = offset[joint_name]
    if step_info.direction == 'up':
        config['GR1_body']['pre_processor']['correct_orientation'] = True
    else:
        config['GR1_body']['pre_processor']['correct_orientation'] = False

    retargeter = SMPLGR1Retargeter(config, vis=True)
    retargeter.calibrate(step_info.smplh_traj[0])

    traj = []
    for i in range(len(step_info.smplh_traj)):
        traj.append(append_hand_pos(retargeter(step_info.smplh_traj[i]), step_info.hand_pose[0], step_info.hand_pose[1]))
        traj[-1][:6] = gr1_transform.waist_head_joints.copy()
    
    # add trajectory which interpolate from start_q to traj[0]
    interpolate_traj = []
    for i in range(50):
        interpolate_traj.append(start_q + (traj[0] - start_q) * i / 50)
    
    traj = np.array(interpolate_traj + traj)

    return traj

def translation_oar(step_info: HOIG, start_q, end_pos_in_head, gr1_transform: GR1Transform):
    '''
    Given the HOIG information of one step, and the new target position obtained from vision pipeline, do object-aware retargeting by translating the original trajectory to start at current start_q.
    
    Return: the new trajectory after retargeting. np.array of shape (n, 56)
    '''

    gr1_urdf = GR1URDFModel()
    start_pos_in_head = gr1_urdf.get_joint_pose_relative_to_head(f'link_{step_info.lr}Arm7', start_q)[:3, 3]
    
    offset_in_head = start_pos_in_head - step_info.original_start_pos_in_head
    offset_in_base = gr1_transform.apply_transform_to_point(offset_in_head, 'head', 'chest')

    offset = {
        f'''link_{step_info.lr}Arm7''': offset_in_base.tolist(),
    }
    print("offset=", offset)

    retarget_repo_dir = os.path.dirname(retarget.__file__)
    config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1_translation.yaml")
    config = load_config(config_path)
    for joint_name in offset:
        config['GR1_body']['pre_processor']['offset'][joint_name] = offset[joint_name]
    # if step_info.direction == 'up':
    #     config['GR1_body']['pre_processor']['correct_orientation'] = True
    # else:
    #     config['GR1_body']['pre_processor']['correct_orientation'] = False
    config['GR1_body']['pre_processor']['correct_orientation'] = True

    retargeter = SMPLGR1Retargeter(config, vis=True)
    retargeter.calibrate(step_info.smplh_traj[0])

    this_arm_joint_idx = []
    other_arm_joint_idx = []
    for key in name_to_urdf_idx.keys():
        if key.startswith('l') or key.startswith('r'):
            if key.startswith('l' if step_info.lr == 'L' else 'r'):
                this_arm_joint_idx.append(name_to_urdf_idx[key])
            else:
                other_arm_joint_idx.append(name_to_urdf_idx[key])

    # First let other arm interpolate to a safe L pos
    traj = []
    L_pos = L_pos_joints()
    
    diff = L_pos[other_arm_joint_idx] - start_q[other_arm_joint_idx]
    if np.linalg.norm(diff) > 0.1:
        for i in range(50):
            cur = start_q + (L_pos - start_q) * i / 50
            new_q = start_q.copy()
            new_q[other_arm_joint_idx] = cur[other_arm_joint_idx]
            traj.append(new_q)
        start_q = traj[-1].copy()

    # traj = []
    for i in range(len(step_info.smplh_traj)):
        traj.append(append_hand_pos(retargeter(step_info.smplh_traj[i]), step_info.hand_pose[0], step_info.hand_pose[1]))
        traj[-1][:6] = gr1_transform.waist_head_joints.copy()
        traj[-1][other_arm_joint_idx] = start_q[other_arm_joint_idx]
    
    traj = np.array(traj)

    return traj

def rotation_matrix_from_vectors(vec1, vec2):
    '''
    Given two vectors vec1 and vec2, return the rotation matrix that rotates vec1 to vec2.
    '''
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_mat = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rotation_mat

def warp_oar(step_info: HOIG, start_q, end_pos_in_head, gr1_transform: GR1Transform):

    gr1_urdf = GR1URDFModel()
    start_pos_in_head = gr1_urdf.get_joint_pose_relative_to_head(f'link_{step_info.lr}Arm7', start_q)[:3, 3]

    start_pos_in_base = gr1_transform.apply_transform_to_point(start_pos_in_head, 'head', 'base')
    end_pos_in_base = gr1_transform.apply_transform_to_point(end_pos_in_head, 'head', 'base')

    original_start_pos_in_base = step_info.original_start_pos_in_base
    original_end_pos_in_base = step_info.original_end_pos_in_base

    # TODO: need to warp the trajectory of both hands, or assume that one hand doesn't move, only step_info.lr hand moves

    rotation_mat = rotation_matrix_from_vectors(original_end_pos_in_base - original_start_pos_in_base, end_pos_in_base - start_pos_in_base)
    scale_factor = np.linalg.norm(end_pos_in_base - start_pos_in_base) / np.linalg.norm(original_end_pos_in_base - original_start_pos_in_base)

    # assume only one hand moves. For now we do not change the trajectory of the other hand (let its q still be the same in start_q)
    warp_traj = []
    for i in range(len(step_info.ik_targets_traj)):
        ori_pose = step_info.ik_targets_traj[i][f'link_{step_info.lr}Arm7'].copy()

        if i == 0:
            assert(np.linalg.norm(ori_pose[:3, 3] - original_start_pos_in_base) < 0.01)

        # algo 1: rotate + scale
        vec = ori_pose[:3, 3] - original_start_pos_in_base
        vec = vec * scale_factor
        vec = np.dot(rotation_mat, vec)
        vec += start_pos_in_base

        # # algo 2: scale each dimension
        # vec = ori_pose[:3, 3] - original_start_pos_in_base
        # for j in range(3):
        #     vec[j] = vec[j] * (end_pos_in_base[j] - start_pos_in_base[j]) / (original_end_pos_in_base[j] - original_start_pos_in_base[j])
        # vec += start_pos_in_base

        # # algo 3: mixed. rotate z axis, scale x, y axis
        # vec = ori_pose[:3, 3] - original_start_pos_in_base
        # vec[2] = np.dot(rotation_mat, vec * scale_factor)[2]
        # vec[0] = vec[0] * (end_pos_in_base[0] - start_pos_in_base[0]) / (original_end_pos_in_base[0] - original_start_pos_in_base[0])
        # vec[1] = vec[1] * (end_pos_in_base[1] - start_pos_in_base[1]) / (original_end_pos_in_base[1] - original_start_pos_in_base[1])
        # vec += start_pos_in_base

        new_pose = ori_pose.copy()
        new_pose[:3, 3] = vec

        # correct orientation
        if step_info.direction == 'up':
            delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
        else:
            # delta_rot = R.from_euler('xyz', [-25, 0, -50], degrees=True).as_matrix()
            if step_info.lr == 'R':
                delta_rot = R.from_euler('xyz', [20, 20, 0], degrees=True).as_matrix()
            else:
                delta_rot = R.from_euler('xyz', [-20, 0, 0], degrees=True).as_matrix()
        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = delta_rot
        new_pose = new_pose @ delta_transformation

        warp_traj.append(new_pose)

    assert(np.linalg.norm(warp_traj[0][:3, 3] - start_pos_in_base) < 0.01)
    assert(np.linalg.norm(warp_traj[-1][:3, 3] - end_pos_in_base) < 0.01)

    # for i in range(len(warp_traj)):
        # warp_traj[i] = clip_wrist_pose_in_base(warp_traj[i], step_info.lr)
    warp_traj[-1] = clip_wrist_pose_in_base(warp_traj[-1], step_info.lr)

    # print("warp_traj[-1] pos=", warp_traj[-1][:3, 3], 'end_pos_in_base=', end_pos_in_base)
    # print("equal?", np.linalg.norm(warp_traj[-1][:3, 3] - end_pos_in_base) < 0.01, np.linalg.norm(warp_traj[-1][:3, 3] - end_pos_in_base))
    # input()

    this_arm_joint_idx = []
    other_arm_joint_idx = []
    for key in name_to_urdf_idx.keys():
        if key.startswith('l') or key.startswith('r'):
            if key.startswith('l' if step_info.lr == 'L' else 'r'):
                this_arm_joint_idx.append(name_to_urdf_idx[key])
            else:
                other_arm_joint_idx.append(name_to_urdf_idx[key])

    # now we have all ik targets, we can do retarget to obtain all the joints
    retarget_repo_dir = os.path.dirname(retarget.__file__)
    config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1_warp.yaml")
    config = load_config(config_path)
    retargeter = SMPLGR1Retargeter(config, vis=True)
    retargeter.calibrate(step_info.smplh_traj[0])

    traj = []
    # First let other arm interpolate to a safe H pos
    L_pos = L_pos_joints()
    
    other_hand_necessary = step_info.other_hand_necessary
    diff = L_pos[other_arm_joint_idx] - start_q[other_arm_joint_idx]
    if (np.linalg.norm(diff) > 0.1) and (other_hand_necessary == False):
        for i in range(50):
            cur = start_q + (L_pos - start_q) * i / 50
            new_q = start_q.copy()
            new_q[other_arm_joint_idx] = cur[other_arm_joint_idx]
            traj.append(new_q)
        start_q = traj[-1].copy()

    H_pos = H_pos_joints()
    # retargeter.body.update(H_pos[3:])
    retargeter.body.update(start_q[3:])
    for i in range(len(warp_traj)):
        target = {f'link_{step_info.lr}Arm7': warp_traj[i],
                  f'link_{step_info.lr}Arm4': step_info.ik_targets_traj[i][f'link_{step_info.lr}Arm4'],
                  f'link_{step_info.lr}Arm2': step_info.ik_targets_traj[i][f'link_{step_info.lr}Arm2'],
                }
        target_q, _ = retargeter.body.retarget(target)

        target_q = np.append(np.array([0, 0, 0]), target_q)
        target_q[:6] = gr1_transform.waist_head_joints.copy()
        target_q = append_hand_pos(target_q, step_info.hand_pose[0], step_info.hand_pose[1])

        # fix q of other arm to be the same as start_q
        target_q[other_arm_joint_idx] = start_q[other_arm_joint_idx]

        traj.append(target_q)
        retargeter.body.update(target_q[3:])
        # retargeter.body.update(H_pos[3:])
        retargeter.visualize(target)

    return np.array(traj)

def warp_oar_baseline(step_info: HOIG, start_q, end_pos_in_head, gr1_transform: GR1Transform):

    gr1_urdf = GR1URDFModel()
    start_pos_in_head = gr1_urdf.get_joint_pose_relative_to_head(f'link_{step_info.lr}Palm', start_q)[:3, 3]

    start_pos_in_base = gr1_transform.apply_transform_to_point(start_pos_in_head, 'head', 'base')
    end_pos_in_base = gr1_transform.apply_transform_to_point(end_pos_in_head, 'head', 'base')

    original_start_pos_in_base = step_info.original_finger_start_pos_in_base
    original_end_pos_in_base = step_info.original_finger_end_pos_in_base

    rotation_mat = rotation_matrix_from_vectors(original_end_pos_in_base - original_start_pos_in_base, end_pos_in_base - start_pos_in_base)
    scale_factor = np.linalg.norm(end_pos_in_base - start_pos_in_base) / np.linalg.norm(original_end_pos_in_base - original_start_pos_in_base)

    # assume only one hand moves. For now we do not change the trajectory of the other hand (let its q still be the same in start_q)
    warp_traj = []
    for i in range(len(step_info.ik_targets_traj)):
        finger_idx = [10, 14, 17, 20, 23]
        mean_pose = np.mean([step_info.hand_targets_traj[i][f'link_{step_info.lr}Arm{idx}'] for idx in finger_idx], axis=0)
        ori_pose = mean_pose.copy()
        ori_pose[:3, :3] *= 0

        # algo 1: rotate + scale
        vec = ori_pose[:3, 3] - original_start_pos_in_base
        vec = vec * scale_factor
        vec = np.dot(rotation_mat, vec)
        vec += start_pos_in_base

        new_pose = ori_pose.copy()
        new_pose[:3, 3] = vec

        warp_traj.append(new_pose)

    assert(np.linalg.norm(warp_traj[0][:3, 3] - start_pos_in_base) < 0.01)
    assert(np.linalg.norm(warp_traj[-1][:3, 3] - end_pos_in_base) < 0.01)

    this_arm_joint_idx = []
    other_arm_joint_idx = []
    for key in name_to_urdf_idx.keys():
        if key.startswith('l') or key.startswith('r'):
            if key.startswith('l' if step_info.lr == 'L' else 'r'):
                this_arm_joint_idx.append(name_to_urdf_idx[key])
            else:
                other_arm_joint_idx.append(name_to_urdf_idx[key])

    # now we have all ik targets, we can do retarget to obtain all the joints
    retarget_repo_dir = os.path.dirname(retarget.__file__)
    config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1_baseline.yaml")
    config = load_config(config_path)
    retargeter = SMPLGR1Retargeter(config, vis=True)
    retargeter.calibrate(step_info.smplh_traj[0])

    traj = []
    # First let other arm interpolate to a safe H pos
    L_pos = L_pos_joints()
    
    other_hand_necessary = step_info.other_hand_necessary
    diff = L_pos[other_arm_joint_idx] - start_q[other_arm_joint_idx]
    if (np.linalg.norm(diff) > 0.1) and (other_hand_necessary == False):
        for i in range(50):
            cur = start_q + (L_pos - start_q) * i / 50
            new_q = start_q.copy()
            new_q[other_arm_joint_idx] = cur[other_arm_joint_idx]
            traj.append(new_q)
        start_q = traj[-1].copy()

    H_pos = H_pos_joints()
    # retargeter.body.update(H_pos[3:])
    retargeter.body.update(start_q[3:])
    for i in range(len(warp_traj)):
        target = {f'link_{step_info.lr}Palm': warp_traj[i]}
        target_q, _ = retargeter.body.retarget(target)

        target_q = np.append(np.array([0, 0, 0]), target_q)
        target_q[:6] = gr1_transform.waist_head_joints.copy()
        target_q = append_hand_pos(target_q, step_info.hand_pose[0], step_info.hand_pose[1])

        # fix q of other arm to be the same as start_q
        target_q[other_arm_joint_idx] = start_q[other_arm_joint_idx]

        traj.append(target_q)
        retargeter.body.update(target_q[3:])
        # retargeter.body.update(H_pos[3:])
        retargeter.visualize(target)

    return np.array(traj)