import os
import numpy as np

from retarget.retargeter import SMPLGR1Retargeter
from retarget.utils.configs import load_config
import retarget

from okami.oar.utils.frame_transformation import GR1Transform
from okami.oar.utils.urdf_utils import append_hand_q, parse_hand_q

class IK:
    def __init__(self, transform: GR1Transform):
        self.retarget_repo_dir = os.path.dirname(retarget.__file__)
        self.transform = transform

    def obtain_targets(self, smplh_traj):
        self.config_path = os.path.join(self.retarget_repo_dir, "../configs/smpl_gr1_translation.yaml")
        self.config = load_config(self.config_path)
        self.retargeter = SMPLGR1Retargeter(self.config)
        self.retargeter.calibrate(smplh_traj[0])

        ik_targets_traj = []
        hand_targets_traj = []
        for i in range(len(smplh_traj)):           

            body_targets = self.retargeter.body.retarget(smplh_traj[i])[1]
            
            # calculate and append palm targets
            l_palm_target = body_targets['link_LArm7'].copy()
            l_palm_target[:3, 3] += body_targets['link_LArm7'][:3, :3] @ np.array([-0.01, -0.12, 0.02])
            r_palm_target = body_targets['link_RArm7'].copy()
            r_palm_target[:3, 3] += body_targets['link_RArm7'][:3, :3] @ np.array([-0.01, -0.12, -0.02])
            body_targets['link_LPalm'] = l_palm_target
            body_targets['link_RPalm'] = r_palm_target

            # add smplh hand parameters in body targets
            body_targets['left_fingers'] = smplh_traj[i]['left_fingers']
            body_targets['right_fingers'] = smplh_traj[i]['right_fingers']

            ik_targets_traj.append(body_targets)
            
            left_hand_targets = self.retargeter.left_hand.retarget(smplh_traj[i])[1]
            right_hand_targets = self.retargeter.right_hand.retarget(smplh_traj[i])[1]
            # combine two dicts
            hand_targets = {**left_hand_targets, **right_hand_targets}
            hand_targets_traj.append(hand_targets)

        return ik_targets_traj, hand_targets_traj
    
    def retarget_from_warped_poses(self, target_seq, start_q, calibrate_data, vis=False, is_reach=False, baseline=False):
        
        if baseline:
            self.config_path = os.path.join(self.retarget_repo_dir, "../configs/smpl_gr1_warp_dex_baseline.yaml")
        else:
            self.config_path = os.path.join(self.retarget_repo_dir, "../configs/smpl_gr1_warp_dex.yaml")
        self.config = load_config(self.config_path)
            
        self.retargeter = SMPLGR1Retargeter(self.config, vis=vis)
        self.retargeter.calibrate(calibrate_data)

        # start_q[:6] *= 0
        self.retargeter.body.update(start_q[3:])

        traj = []
        traj.append(start_q) # Important! The first frame is the initial frame, otherwise interpolation is not correct
        for i in range(len(target_seq)):
            
            target_q, _ = self.retargeter.body.retarget(target_seq[i].copy())
            left_hand_q, _ = self.retargeter.left_hand.retarget(target_seq[i].copy())
            right_hand_q, _ = self.retargeter.right_hand.retarget(target_seq[i].copy())

            target_q = np.append(np.array([0, 0, 0]), target_q)
            target_q[:6] = self.transform.waist_head_joints.copy()
            target_q = append_hand_q(target_q, left_hand_q, right_hand_q)

            traj.append(target_q)

            self.retargeter.body.update(target_q[3:])
            if vis:
                self.retargeter.visualize(target_seq[i])

        if is_reach == True:
            # put all hand movements to the end
            init_left_hand_q, init_right_hand_q = parse_hand_q(traj[0])
            
            left_hand_q_lst = []
            right_hand_q_lst = []

            for i in range(len(traj)):
                left_hand_q, right_hand_q = parse_hand_q(traj[i])
                left_hand_q_lst.append(left_hand_q)
                right_hand_q_lst.append(right_hand_q)

                traj[i] = append_hand_q(traj[i], init_left_hand_q, init_right_hand_q)

            last_traj_q = traj[-1].copy()
            # for i in range(0, len(traj), 5):
            #     new_q = append_hand_q(last_traj_q, left_hand_q_lst[i], right_hand_q_lst[i])
            #     traj.append(new_q)
            
            # interpolate from first hand q to last hand q
            for i in range(0, len(traj), 5):
                interp_left_hand_q = left_hand_q_lst[0] + (left_hand_q_lst[-1] - left_hand_q_lst[0]) * i / len(traj)
                interp_right_hand_q = right_hand_q_lst[0] + (right_hand_q_lst[-1] - right_hand_q_lst[0]) * i / len(traj)
                
                new_q = append_hand_q(last_traj_q, interp_left_hand_q, interp_right_hand_q)
                traj.append(new_q)

        return np.array(traj)

    def retarget_for_translated_traj(self, smplh_traj, start_q, calibrate_data, offset=None, vis=False):

        self.config_path = os.path.join(self.retarget_repo_dir, "../configs/smpl_gr1_translation_dex.yaml")
        self.config = load_config(self.config_path)

        for joint_name in offset:
            self.config['GR1_body']['pre_processor']['offset'][joint_name] = offset[joint_name]
        self.config['GR1_body']['pre_processor']['correct_orientation'] = True

        self.retargeter = SMPLGR1Retargeter(self.config, vis=vis)
        self.retargeter.calibrate(calibrate_data)

        self.retargeter.body.update(start_q[3:])
        
        traj = []
        traj.append(start_q) # Important! The first frame is the initial frame, otherwise interpolation is not correct
        for i in range(len(smplh_traj)):
            traj.append(self.retargeter(smplh_traj[i]))
            traj[-1][:6] = self.transform.waist_head_joints.copy()

        return np.array(traj)