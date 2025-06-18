import numpy as np

from okami.oar.algos.ik_solver import IK
from okami.oar.utils.frame_transformation import GR1Transform
from scipy.spatial.transform import Rotation as R

class TargetPoseGenerator:
    def __init__(self, transform: GR1Transform):
        self.transform = transform

    def generate(self, object_point_in_head_frame, ref_pose, lr):
        raise NotImplementedError

class GraspPoseGenerator(TargetPoseGenerator):
    def __init__(self, transform: GR1Transform):
        super().__init__(transform)

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

    def clip_wrist_orientation_in_upperbase(self, wrist_rot, lr):
        direction = self.up_or_side_grasp(wrist_rot, lr)
        if direction == 'up':
            return wrist_rot
        else: 
            palm_direction = - wrist_rot[:3, 0]
            if lr == 'L' and palm_direction[1] > 0:
                return wrist_rot
            if lr == 'R' and palm_direction[1] < 0:
                return wrist_rot

            pointing_direction = - wrist_rot[:3, 1]
            pointing_angle = np.arcsin(pointing_direction[2] / np.linalg.norm(pointing_direction))
            if pointing_angle < -10 / 180 * np.pi:
                if lr == 'R':
                    delta_rot = R.from_euler('xyz', [40, 0, 0], degrees=True).as_matrix()
                else:
                    delta_rot = R.from_euler('xyz', [-30, 0, 0], degrees=True).as_matrix()
                print("rotating 20 degree around x", delta_rot)
                wrist_rot = wrist_rot @ delta_rot

            palm_direction = - wrist_rot[:3, 0]
            palm_angle = np.arcsin(- palm_direction[0] / np.linalg.norm(palm_direction))
            if palm_angle < 30 / 180 * np.pi:
                if lr == 'R':
                    delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
                else:
                    delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
                wrist_rot = wrist_rot @ delta_rot

            palm_direction = - wrist_rot[:3, 0]
            palm_angle = np.arcsin(palm_direction[2] / np.linalg.norm(palm_direction))
            if palm_angle > 5 / 180 * np.pi:
                if lr == 'R':
                    delta_rot = R.from_euler('xyz', [0, 10, 0], degrees=True).as_matrix()
                else:
                    delta_rot = R.from_euler('xyz', [0, -10, 0], degrees=True).as_matrix() # TODO: check direction
                print("rotating 10 degree around y", delta_rot)
                wrist_rot = wrist_rot @ delta_rot
            elif palm_angle < -5 / 180 * np.pi:
                if lr == 'R':
                    delta_rot = R.from_euler('xyz', [0, -10, 0], degrees=True).as_matrix()
                    print("rotating -10 degree around y", delta_rot)
                else:
                    print("rotate", -(palm_angle / np.pi * 180 - 10), "degree around y")
                    delta_rot = R.from_euler('xyz', [0, -(palm_angle / np.pi * 180), 0], degrees=True).as_matrix()
                wrist_rot = wrist_rot @ delta_rot
                
                print("afterwards: palm_angle=", np.arcsin(- wrist_rot[:3, 0][2] / np.linalg.norm(wrist_rot[:3, 0])) / np.pi * 180)
        
            return wrist_rot

    def world_grasp_offset(self, wrist_rot, lr):
        direction = self.up_or_side_grasp(wrist_rot, lr)
        if direction == 'up':
            return np.array([0., 0., 0.])
        else:
            if lr == 'L':
                # determine if palm is facing body or not
                palm_direction = - wrist_rot[:3, 0]
                if palm_direction[1] < 0: # facing right (body)
                    return np.array([0.029, 0.025, 0.0])
                else: 
                    return np.array([0.0, -0.02, 0.0])
            else:
                # determine if palm is facing body or not
                palm_direction = - wrist_rot[:3, 0]
                if palm_direction[1] > 0: # facing left (body)
                    return np.array([0.04, -0.04, 0.0])
                else:
                    return np.array([0.0, 0.02, 0.0])

    def generate(self, object_point_in_head_frame, ref_pose_in_upperbase, lr):
        '''
        Given the object point that palm needs to reach in head frame, and the reference rotation of the waist in upperbase frame, calculate the desired waist pose (both orientation and position) in upperbase frame.
        Alter the orientation according to grasping type, and calculate the position.

        Return the desired waist pose in upperbase frame.
        '''
        # Decide whether it's grasping from up or side
        direction = self.up_or_side_grasp(ref_pose_in_upperbase, lr)

        # Alter the orientation a little for better grasping
        if True:
            if direction == 'up':
                delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
                delta_transformation = np.eye(4)
                delta_transformation[:3, :3] = delta_rot
                ref_pose_in_upperbase = ref_pose_in_upperbase @ delta_transformation
            else:
                if lr == 'R':
                    delta_rot = R.from_euler('xyz', [20, 0, 0], degrees=True).as_matrix()
                else:
                    delta_rot = R.from_euler('xyz', [-20, 0, 0], degrees=True).as_matrix()
                delta_transformation = np.eye(4)
                delta_transformation[:3, :3] = delta_rot
                ref_pose_in_upperbase = ref_pose_in_upperbase @ delta_transformation

            # Further alter the orientation for better grasping
            wrist_rot = ref_pose_in_upperbase[:3, :3]
            wrist_rot = self.clip_wrist_orientation_in_upperbase(wrist_rot, lr)
            ref_pose_in_upperbase[:3, :3] = wrist_rot

        # calculate offset between palm and object point in world frame
        grasp_offset_in_world = self.world_grasp_offset(wrist_rot, lr)
        grasp_offset_in_head = self.transform.apply_transform_to_vector(grasp_offset_in_world, 'world', 'head')

        palm_to_wrist_in_wrist = np.array([0.01, 0.12, 0.02 * (1 if lr == 'R' else -1)])
        palm_to_wrist_in_upperbase = ref_pose_in_upperbase[:3, :3] @ palm_to_wrist_in_wrist
        palm_to_wrist_in_head = self.transform.apply_transform_to_vector(palm_to_wrist_in_upperbase, 'upperbase', 'head')

        print("palm_to_wrist_in_head=", palm_to_wrist_in_head, "in upperbase=", palm_to_wrist_in_upperbase)

        waist_target_pos_in_head = object_point_in_head_frame + palm_to_wrist_in_head + grasp_offset_in_head
        waist_target_pos_in_upperbase = self.transform.apply_transform_to_point(waist_target_pos_in_head, 'head', 'upperbase')

        ref_pose_in_upperbase[:3, 3] = waist_target_pos_in_upperbase
        return ref_pose_in_upperbase
    
    def generate_palm_pose(self, object_point_in_head_frame, ref_pose_in_upperbase, lr):
        '''
        Given the object point that palm needs to reach in head frame, and the reference rotation of the waist in upperbase frame, calculate the desired waist pose (both orientation and position) in upperbase frame.
        Alter the orientation according to grasping type, and calculate the position.

        Return the desired palm pose in upperbase frame.
        '''
        # Decide whether it's grasping from up or side
        direction = self.up_or_side_grasp(ref_pose_in_upperbase, lr)
        print("direction=", direction)

        # Alter the orientation a little for better grasping
        if True:
            if direction == 'up':
                delta_rot = R.from_euler('xyz', [0, 0, -20], degrees=True).as_matrix()
                delta_transformation = np.eye(4)
                delta_transformation[:3, :3] = delta_rot
                ref_pose_in_upperbase = ref_pose_in_upperbase @ delta_transformation
            else:
                if lr == 'R':
                    delta_rot = R.from_euler('xyz', [20, 0, 0], degrees=True).as_matrix()
                else:
                    delta_rot = R.from_euler('xyz', [-20, 0, 0], degrees=True).as_matrix()
                delta_transformation = np.eye(4)
                delta_transformation[:3, :3] = delta_rot
                ref_pose_in_upperbase = ref_pose_in_upperbase @ delta_transformation

            # Further alter the orientation for better grasping
            wrist_rot = ref_pose_in_upperbase[:3, :3]
            wrist_rot = self.clip_wrist_orientation_in_upperbase(wrist_rot, lr)
            ref_pose_in_upperbase[:3, :3] = wrist_rot
        
        # calculate offset between palm and object point in world frame
        grasp_offset_in_world = self.world_grasp_offset(wrist_rot, lr)
        grasp_offset_in_upperbase = self.transform.apply_transform_to_vector(grasp_offset_in_world, 'world', 'upperbase')

        object_point_in_upperbase = self.transform.apply_transform_to_point(object_point_in_head_frame, 'head', 'upperbase')
        palm_pose_in_upperbase = ref_pose_in_upperbase.copy()
        palm_pose_in_upperbase[:3, 3] = object_point_in_upperbase + grasp_offset_in_upperbase

        return palm_pose_in_upperbase

class ObjectInteractionPoseGenerator(TargetPoseGenerator):
    def __init__(self):
        super().__init__()

    def generate(self, object_point_in_head_frame, ref_pose, lr):
        raise NotImplementedError