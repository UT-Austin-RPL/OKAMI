import numpy as np

from okami.oar.algos.ik_solver import IK
from okami.oar.utils.frame_transformation import GR1Transform
from scipy.spatial.transform import Rotation as R

class TrajGenerator:
    def __init__(self, transform: GR1Transform):
        self.transform = transform

    def detect_outliers(self, palm_targets):
        def delta_pos(pos1, pos2):
            return np.linalg.norm(pos1 - pos2)
        def delta_rot(rot1, rot2):
            relative_rot = rot1.T @ rot2
            relative_rot = R.from_matrix(relative_rot)
            return relative_rot.magnitude()
        
        position_threshold = 0.1
        rotation_threshold = 0.5
        window_size = 5
        is_outlier = [False] * len(palm_targets)
        for i in range(len(palm_targets) - window_size):
            window_poses = palm_targets[i:i+window_size]
            position_changes = [delta_pos(window_poses[j][:3, 3], window_poses[j+1][:3, 3]) for j in range(window_size - 1)]
            rotation_changes = [delta_rot(window_poses[j][:3, :3], window_poses[j+1][:3, :3]) for j in range(window_size - 1)] 
            
            # print("max position change, rotation change=", np.max(position_changes), np.max(rotation_changes))
            
            if np.max(position_changes) > position_threshold or np.max(rotation_changes) > rotation_threshold:
                is_outlier[i + window_size // 2] = True
        
        print("number of outliers:", sum(is_outlier))
        return is_outlier
    
    def smooth_trajectory(self, palm_targets):
        is_outlier = self.detect_outliers(palm_targets)
        # for all the continous outliers, replace the pose with the interpolation of the two ends
        i = 0
        while i < len(is_outlier):
            if is_outlier[i]:
                j = i
                while j < len(is_outlier) and is_outlier[j]:
                    j += 1
                print("len of continous outliers=", j - i)
                if j < len(is_outlier):
                    start_pose = palm_targets[i]
                    end_pose = palm_targets[j]
                    for k in range(i, j):
                        t = (k - i) / (j - i)
                        interp_pose = interpolate_se3(start_pose, end_pose, t)
                        palm_targets[k] = interp_pose
                i = j
            else:
                i += 1
        return palm_targets

    def up_or_side_grasp(self, hand_pose, lr):
        # determine -x direction closer to -z or +y (right hand) / -y (left hand)
        local_x = hand_pose[:3, 0]
        local_minus_x = -local_x

        world_z = np.array([0, 0, 1])
        world_minus_z = np.array([0, 0, -1])
        world_y = np.array([0, 1, 0])
        world_minus_y = np.array([0, -1, 0])

        angle_with_general_z = np.arccos(np.dot(local_minus_x, world_minus_z) / (np.linalg.norm(local_minus_x) * np.linalg.norm(world_minus_z)))
        if lr == 'R':
            angle_with_general_y = np.arccos(np.dot(local_minus_x, world_y) / (np.linalg.norm(local_minus_x) * np.linalg.norm(world_y)))
        else:
            angle_with_general_y = np.arccos(np.dot(local_minus_x, world_minus_y) / (np.linalg.norm(local_minus_x) * np.linalg.norm(world_minus_y)))

        # print("angle with z, y=", angle_with_general_z / np.pi * 180, angle_with_general_y / np.pi * 180)

        threshold_z = np.pi / 4
        threshold_y = np.pi / 3
        if angle_with_general_z < threshold_z and angle_with_general_y > threshold_y:
            return 'up'
        return 'side'

    def warp_trajectory_baseline(self, ref_traj, start_q, lr, target_palm_pos, hand_targets, calibrate_data, vis=True, for_palm=False, is_reach=False):
        '''
        Baseline: Given reference IK targets and hand targets, calculate the reference palm trajectory, then warp the trajectory based on current palm position and target palm position.
        '''

        # calculate reference palm trajectory using the center of hand joints
        ref_palm_traj = []
        for i in range(len(hand_targets)):
            finger_idx = [10, 14, 17, 20, 23]
            mean_pos = np.mean([hand_targets[i][f'link_{lr}Arm{idx}'][:3, 3] for idx in finger_idx], axis=0)
            ref_palm_traj.append(mean_pos)

        ref_link_name = f'link_{lr}Palm'
        
        # obtain the start and end position of the new trajectory
        start_pos_in_head = self.transform.get_joint_pose_relative_to_head(ref_link_name, start_q)[:3, 3]
        start_pos_in_upperbase = self.transform.apply_transform_to_point(start_pos_in_head, 'head', 'upperbase')
        end_pos_in_upperbase = target_palm_pos.copy()

        # obtain the start and end position of the reference trajectory
        ref_start_pos_in_upperbase = ref_palm_traj[0]
        ref_end_pos_in_upperbase = ref_palm_traj[-1]

        # calculate parameters for warping position
        rotation_mat = rotation_matrix_from_vectors(ref_end_pos_in_upperbase - ref_start_pos_in_upperbase, end_pos_in_upperbase - start_pos_in_upperbase)
        scale_factor = np.linalg.norm(end_pos_in_upperbase - start_pos_in_upperbase) / np.linalg.norm(ref_end_pos_in_upperbase - ref_start_pos_in_upperbase)

        # warp the palm targets of the target arm
        warp_palm_targets = []
        for i in range(len(ref_palm_traj)):

            ref_pose_in_upperbase = np.eye(4)
            ref_pose_in_upperbase[:3, 3] = ref_palm_traj[i]

            if i == 0:
                assert(np.linalg.norm(ref_pose_in_upperbase[:3, 3] - ref_start_pos_in_upperbase) < 0.01)

            # algo: rotate + scale
            vec = ref_pose_in_upperbase[:3, 3] - ref_start_pos_in_upperbase
            vec = vec * scale_factor
            vec = np.dot(rotation_mat, vec)
            vec += start_pos_in_upperbase

            ref_pose_in_upperbase[:3, 3] = vec
            warp_palm_targets.append(ref_pose_in_upperbase)

        # obtain a list of IK targets
        target_seq = []
        for i in range(len(warp_palm_targets)):
            target = {ref_link_name: warp_palm_targets[i],
                    f'link_{lr}Arm4': ref_traj[i][f'link_{lr}Arm4'],
                    f'link_{lr}Arm2': ref_traj[i][f'link_{lr}Arm2'],
                    'left_fingers': ref_traj[i]['left_fingers'],
                    'right_fingers': ref_traj[i]['right_fingers'],
                    }
            target_seq.append(target)

        # use IK to solve for joints
        ik_solver = IK(self.transform)
        traj = ik_solver.retarget_from_warped_poses(target_seq, start_q, calibrate_data=calibrate_data, vis=vis, is_reach=is_reach, baseline=True)
        
        print()

        return traj, ref_pose_in_upperbase
        

    def warp_trajectory(self, ref_traj, start_q, lr, target_pose, calibrate_data, vis=True, is_reach=False):
        '''
        Given the reference trajectory (a sequence of target pose of different body parts), the current joint states, the side of the hand, and the target pose, warp the trajectory to the target pose.

        Return the desired trajectory.
        '''

        ref_link_name = f'link_{lr}Palm'

        # obtain the start and end position of the new trajectory
        start_pos_in_head = self.transform.get_joint_pose_relative_to_head(ref_link_name, start_q)[:3, 3]
        start_pos_in_upperbase = self.transform.apply_transform_to_point(start_pos_in_head, 'head', 'upperbase')
        end_pos_in_upperbase = target_pose[:3, 3].copy()

        # obtain the start and end position of the reference trajectory
        ref_start_pos_in_upperbase = ref_traj[0][ref_link_name][:3, 3]
        ref_end_pos_in_upperbase = ref_traj[-1][ref_link_name][:3, 3]

        # obtain the start and end rotation of the new trajectory
        start_rot_in_upperbase = self.transform.get_joint_pose(ref_link_name, start_q)[:3, :3]
        end_rot_in_upperbase = target_pose[:3, :3].copy()

        # obtain the start and end rotation of the reference trajectory
        ref_start_rot_in_upperbase = ref_traj[0][ref_link_name][:3, :3]
        ref_end_rot_in_upperbase = ref_traj[-1][ref_link_name][:3, :3]

        # calculate parameters for warping position
        rotation_mat = rotation_matrix_from_vectors(ref_end_pos_in_upperbase - ref_start_pos_in_upperbase, end_pos_in_upperbase - start_pos_in_upperbase)
        scale_factor = np.linalg.norm(end_pos_in_upperbase - start_pos_in_upperbase) / np.linalg.norm(ref_end_pos_in_upperbase - ref_start_pos_in_upperbase)

        # calculate parameters for warping rotation
        start_transform = start_rot_in_upperbase @ np.linalg.inv(ref_start_rot_in_upperbase)
        end_transform = end_rot_in_upperbase @ np.linalg.inv(ref_end_rot_in_upperbase)

        # warp the palm targets of the target arm
        warp_palm_targets = []
        for i in range(len(ref_traj)):
            ref_pose_in_upperbase = ref_traj[i][ref_link_name].copy()

            if i == 0:
                assert(np.linalg.norm(ref_pose_in_upperbase[:3, 3] - ref_start_pos_in_upperbase) < 0.01)

            # algo: rotate + scale
            vec = ref_pose_in_upperbase[:3, 3] - ref_start_pos_in_upperbase
            vec = vec * scale_factor
            vec = np.dot(rotation_mat, vec)
            vec += start_pos_in_upperbase

            ref_pose_in_upperbase[:3, 3] = vec

            rot = ref_pose_in_upperbase[:3, :3]
            interp_transform = slerp(R.from_matrix(start_transform), 
                                     R.from_matrix(end_transform), 
                                     i / (len(ref_traj) - 1)).as_matrix()
            rot = interp_transform @ rot
            if i == len(ref_traj) - 1:
                print("rot=", rot)
                print("end_rot_in_upperbase=", end_rot_in_upperbase)
            ref_pose_in_upperbase[:3, :3] = rot

            warp_palm_targets.append(ref_pose_in_upperbase)

        warp_palm_targets = self.smooth_trajectory(warp_palm_targets)

        # obtain a list of IK targets
        target_seq = []
        for i in range(len(warp_palm_targets)):
            target = {ref_link_name: warp_palm_targets[i],
                    f'link_{lr}Arm4': ref_traj[i][f'link_{lr}Arm4'],
                    f'link_{lr}Arm2': ref_traj[i][f'link_{lr}Arm2'],
                    'left_fingers': ref_traj[i]['left_fingers'],
                    'right_fingers': ref_traj[i]['right_fingers'],
                    }
            target_seq.append(target)

        # use IK to solve for joints
        ik_solver = IK(self.transform)
        traj = ik_solver.retarget_from_warped_poses(target_seq, start_q, calibrate_data=calibrate_data, vis=vis, is_reach=is_reach)
        
        print()

        return traj, ref_pose_in_upperbase

    def translate_trajectroy(self, 
                             ref_traj, 
                             start_q, 
                             lr, 
                             calibrate_data, 
                             vis=True):
        '''
        Given the reference trajectory (a sequence of target pose of different body parts), the current joint states, and the side of the hand, translate the trajectory to the target pose.

        Return the desired trajectory.
        '''

        ref_link_name = f'link_{lr}Palm'

        start_pos_in_head = self.transform.get_joint_pose_relative_to_head(ref_link_name, start_q)[:3, 3]
        start_pos_in_upperbase = self.transform.apply_transform_to_point(start_pos_in_head, 'head', 'upperbase')
        
        ref_start_pos_in_upperbase = ref_traj[0][ref_link_name][:3, 3]
        offset_in_upperbase = start_pos_in_upperbase - ref_start_pos_in_upperbase

        start_rot_in_upperbase = self.transform.get_joint_pose(ref_link_name, start_q)[:3, :3]
        ref_start_rot_in_upperbase = ref_traj[0][ref_link_name][:3, :3]
        transform_rot = start_rot_in_upperbase @ np.linalg.inv(ref_start_rot_in_upperbase)

        # translate the palm targets of the target arm
        translate_palm_targets = []
        for i in range(len(ref_traj)):
            ref_pose_in_upperbase = ref_traj[i][ref_link_name].copy()

            if i == 0:
                assert(np.linalg.norm(ref_pose_in_upperbase[:3, 3] - ref_start_pos_in_upperbase) < 0.01)

            vec = ref_pose_in_upperbase[:3, 3] + offset_in_upperbase
            
            rot = ref_pose_in_upperbase[:3, :3]
            rot = transform_rot @ rot

            ref_pose_in_upperbase[:3, 3] = vec
            ref_pose_in_upperbase[:3, :3] = rot

            translate_palm_targets.append(ref_pose_in_upperbase)

        # obtain a list of IK targets
        target_seq = []
        for i in range(len(translate_palm_targets)):
            target = {ref_link_name: translate_palm_targets[i],
                    f'link_{lr}Arm4': ref_traj[i][f'link_{lr}Arm4'],
                    f'link_{lr}Arm2': ref_traj[i][f'link_{lr}Arm2'],
                    'left_fingers': ref_traj[i]['left_fingers'],
                    'right_fingers': ref_traj[i]['right_fingers'],
                    }
            target_seq.append(target)

        # use IK to solve for joints
        ik_solver = IK(self.transform)
        traj = ik_solver.retarget_from_warped_poses(target_seq, start_q, calibrate_data=calibrate_data, vis=vis, is_reach=False)

        return traj, None

    # def translate_trajectroy(self, ref_traj, smplh_traj, start_q, lr, calibrate_data, vis=True, for_palm=False):
    #     '''
    #     Given the reference trajectory (a sequence of target pose of different body parts), the current joint states, and the side of the hand, translate the trajectory to the target pose.

    #     Return the desired trajectory.
    #     '''

    #     ref_link_name = f'link_{lr}Arm7' #if not for_palm else f'link_{lr}Palm'

    #     start_pos_in_head = self.transform.get_joint_pose_relative_to_head(ref_link_name, start_q)[:3, 3]
    #     start_pos_in_upperbase = self.transform.apply_transform_to_point(start_pos_in_head, 'head', 'upperbase')
        
    #     ref_start_pos_in_upperbase = ref_traj[0][ref_link_name][:3, 3]

    #     offset_in_upperbase = start_pos_in_upperbase - ref_start_pos_in_upperbase

    #     offset = {
    #         ref_link_name: offset_in_upperbase.tolist(),
    #     }
    #     print("offset=", offset)

    #     # use IK to solve for joints
    #     ik_solver = IK(self.transform)
    #     traj = ik_solver.retarget_for_translated_traj(smplh_traj, start_q, calibrate_data=calibrate_data, offset=offset, vis=vis)

    #     return traj, None

        

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

def slerp(rot1, rot2, t):
    """ Spherical linear interpolation between two rotations. """
    dot_product = np.clip(np.dot(rot1.as_quat(), rot2.as_quat()), -1.0, 1.0)
    if dot_product < 0.0:
        rot2 = R.from_quat(-rot2.as_quat())
        dot_product = -dot_product

    theta_0 = np.arccos(dot_product)
    theta = theta_0 * t

    if np.abs(theta_0) < 1e-10:
        return rot1

    rot1_quat = rot1.as_quat()
    rot2_quat = rot2.as_quat()
    
    sin_theta_0 = np.sin(theta_0)
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot_product * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    interp_quat = (s0 * rot1_quat) + (s1 * rot2_quat)
    return R.from_quat(interp_quat)

def interpolate_se3(pose1, pose2, t):
    """
    Interpolate between two SE(3) poses.
    
    Args:
        pose1 (np.ndarray): The first pose as a 4x4 transformation matrix.
        pose2 (np.ndarray): The second pose as a 4x4 transformation matrix.
        t (float): The interpolation factor (0 <= t <= 1).

    Returns:
        np.ndarray: The interpolated pose as a 4x4 transformation matrix.
    """
    # Ensure the interpolation factor is within bounds
    t = np.clip(t, 0, 1)
    
    # Extract rotation matrices and translation vectors
    rot1 = R.from_matrix(pose1[:3, :3])
    rot2 = R.from_matrix(pose2[:3, :3])
    
    trans1 = pose1[:3, 3]
    trans2 = pose2[:3, 3]
    
    # Interpolate translation linearly
    trans_interp = (1 - t) * trans1 + t * trans2
    
    # Interpolate rotation using SLERP
    rot_interp = slerp(rot1, rot2, t).as_matrix()
    
    # Combine interpolated rotation and translation into a single SE(3) pose
    pose_interp = np.eye(4)
    pose_interp[:3, :3] = rot_interp
    pose_interp[:3, 3] = trans_interp
    
    return pose_interp