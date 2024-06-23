import os
import json
import numpy as np
import cv2

from robot.gr1 import GR1URDFModel
from utils.grasp_offset_utils import up_or_side_grasp, calculate_wrist_offset_relative_to_object_in_head_frame

from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config
import retarget

def mirror_targets(targets):
    new_targets = {}
    pass

class HOIG:
    def __init__(self, calibrate_data, lr='R'):
        self.smplh_traj = None
        self.lr = lr
        self.direction = 'side'
        self.obj_in_hand = False
        self.hand_pose = ['ready', 'ready']
        self.hand_type = ['open', 'open']
        self.other_hand_necessary = False

        self.original_end_pos_in_head = np.zeros(3)
        self.original_end_pos_in_base = np.zeros(3)
        self.original_start_pos_in_head = np.zeros(3)
        self.original_start_pos_in_base = np.zeros(3)
        self.target_waist_pose = np.eye(4)

        self.original_finger_end_pos_in_base = np.zeros(3)
        self.original_finger_end_pos_in_head = np.zeros(3)
        self.original_finger_start_pos_in_base = np.zeros(3)
        self.original_finger_start_pos_in_head = np.zeros(3)

        self.wrist_offset_in_head = np.zeros(3)

        self.reference_object_name = ""
        self.moving_object_name = ""

        self.ik_targets_traj = []
        self.calibrate_data = calibrate_data
    
    def add(self, smplh_traj, hand_pose=['ready', 'ready'], hand_type=['open', 'open'], object_names=["", ""], gr1_transform=None, baseline=False, mirror=False):
        self.reference_object_name = object_names[0]
        self.moving_object_name = object_names[1]
        self.obj_in_hand = self.moving_object_name != "None"
        
        self.hand_type = hand_type
        self.other_hand_necessary = hand_type[1 if self.lr == 'L' else 0] != 'open'
        
        self.smplh_traj = smplh_traj

        retarget_repo_dir = os.path.dirname(retarget.__file__)
        self.config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1_translation.yaml")
        config = load_config(self.config_path)
        retargeter = SMPLGR1Retargeter(config, vis=False)
        retargeter.calibrate(self.calibrate_data)

        # obtain ik targets trajectory
        self.ik_targets_traj = []
        self.hand_targets_traj = []
        for i in range(len(smplh_traj)):
            self.ik_targets_traj.append(retargeter.body.retarget(smplh_traj[i])[1])

            left_hand_targets = retargeter.left_hand.retarget(smplh_traj[i])[1]
            right_hand_targets = retargeter.right_hand.retarget(smplh_traj[i])[1]
            # combine two dicts
            hand_targets = {**left_hand_targets, **right_hand_targets}
            self.hand_targets_traj.append(hand_targets)

        smplh_target = self.ik_targets_traj[-1]
        smplh_hand_target = self.hand_targets_traj[-1]

        # TODO: somehow determine L or R

        # determine up or side grasp
        self.target_waist_pose = smplh_target[f'link_{self.lr}Arm7']
        self.direction = up_or_side_grasp(smplh_target[f'link_{self.lr}Arm7'], self.lr)
        self.hand_pose = hand_pose

        self.original_end_pos_in_head = gr1_transform.apply_transform_to_point(self.target_waist_pose[:3, 3], 'base', 'head')
        self.original_end_pos_in_base = self.target_waist_pose[:3, 3]

        # calculate original finger end pos
        finger_idx = [10, 14, 17, 20, 23]
        mean_position = np.mean([smplh_hand_target[f'link_{self.lr}Arm{idx}'][:3, 3] for idx in finger_idx], axis=0)
        self.original_finger_end_pos_in_head = gr1_transform.apply_transform_to_point(mean_position, 'base', 'head')
        self.original_finger_end_pos_in_base = mean_position

        if baseline == False:
            self.wrist_offset_in_head = calculate_wrist_offset_relative_to_object_in_head_frame(self.lr, 
                                                                                           gr1_transform=gr1_transform, 
                                                                                           direction=self.direction, 
                                                                                           wrist_transformation=smplh_target[f'link_{self.lr}Arm7'])
        else:
            self.wrist_offset_in_head = np.zeros(3)

        smplh_start = self.ik_targets_traj[0]
        smplh_hand_start = self.hand_targets_traj[0]
        self.original_start_pos_in_base = smplh_start[f'link_{self.lr}Arm7'][:3, 3]
        self.original_start_pos_in_head = gr1_transform.apply_transform_to_point(self.original_start_pos_in_base, 'base', 'head')

        # calculate original finger start pos
        mean_position = np.mean([smplh_hand_start[f'link_{self.lr}Arm{idx}'][:3, 3] for idx in finger_idx], axis=0)
        self.original_finger_start_pos_in_head = gr1_transform.apply_transform_to_point(mean_position, 'base', 'head')
        self.original_finger_start_pos_in_base = mean_position

    
    def print(self):
        print("------- HOIG Info ------")
        print(f"lr: {self.lr}")
        print(f"direction: {self.direction}")
        print(f"obj_in_hand: {self.obj_in_hand}")
        print(f"hand_pose: {self.hand_pose}")
        print(f"hand_type: {self.hand_type}")
        print(f"original_end_pos_in_head: {self.original_end_pos_in_head}")
        print(f"original_end_pos_in_base: {self.original_end_pos_in_base}")
        print(f"original_start_pos_in_base: {self.original_start_pos_in_base}")
        print(f"original_start_pos_in_head: {self.original_start_pos_in_head}")
        print(f"target_waist_pose: {self.target_waist_pose}")
        print(f"wrist_offset_in_head: {self.wrist_offset_in_head}")
        print(f"reference_object_name: {self.reference_object_name}")
        print(f"moving_object_name: {self.moving_object_name}")
        print("------------------------")