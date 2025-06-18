import numpy as np
import cv2
import time

import threading

import mujoco
import mujoco.viewer

from okami.oar.utils.urdf_utils import urdf_to_robosuite_cmds, obs_to_robosuite_cmds, robosuite_cmds_to_body_cmds

class SimControl:
    def __init__(self, env):
        self.env = env
        self.reset()

        print("obs keys: ", self.obs.keys())

    def reset(self):
        self.obs = self.env.reset()

        init_pose = np.zeros(35)
        right_arm_cmd = np.array([0., 1.57, 0., -1.57, 0., 0., 0.])
        left_arm_cmd = np.array([0., -1.57, 0., 1.57, 0., 0., 0.])
        head_cmd = np.array([0.0, 0.0, 0.34])
        waist_cmd = np.array([0.0, 0.22, 0.0])
        init_pose[0:7] = right_arm_cmd
        init_pose[7:14] = left_arm_cmd
        init_pose[29:32] = head_cmd
        init_pose[32:35] = waist_cmd

        self.cmd_idx = 1
        self.cmd_lst = [[init_pose, False]]
        self.saved_info = []

        self.terminate_cmd = False
        self.reset_episode_cmd = False

        self.reward = 0

        print("reset!")

    def add_urdf_cmd(self, cmd, save_state=False):
        robosuite_cmd = urdf_to_robosuite_cmds(cmd)
        self.add_cmd(robosuite_cmd, save_state)

    def add_cmd(self, cmd, save_state=False):
        self.cmd_lst.append([cmd, save_state])

    def reset_episode(self):
        print("begin to terminate the episode")
        self.reset_episode_cmd = True
    
    def terminate(self):
        self.terminate_cmd = True

    def get_obs(self):
        return self.obs

    def get_camera_image(self):
        assert('robot0_robotview_image' in self.obs)
        assert('robot0_robotview_depth' in self.obs)
        img = self.obs['robot0_robotview_image']
        depth = self.obs['robot0_robotview_depth']

        # image transformation (cvt BGR to RGB; flip the image)
        img = cv2.flip(img, 0)
        depth = cv2.flip(depth, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cv2.imshow("image", img)
        # cv2.waitKey(10)

        # assert('agentview_image' in self.obs)
        # img_agent = self.obs['agentview_image']
        # img_agent = cv2.flip(img_agent, 0)
        # img_agent = cv2.cvtColor(img_agent, cv2.COLOR_BGR2RGB)
        
        # # save agentview image
        # num = len(self.saved_info)
        # cv2.imwrite(f"tmp/agentview_{num}.png", img_agent)
        
        assert('frontview_image' in self.obs)
        img_agent = self.obs['frontview_image']
        img_agent = cv2.flip(img_agent, 0)
        img_agent = cv2.cvtColor(img_agent, cv2.COLOR_BGR2RGB)
        
        # save agentview image
        num = len(self.saved_info)
        cv2.imwrite(f"tmp/frontview_{num}.png", img_agent)
        
        return img, depth
    
    def sim_step(self):

        init_joint_pos = obs_to_robosuite_cmds(self.obs)
        target_joint_pos = np.zeros(35)
        
        if self.cmd_idx < len(self.cmd_lst):
            target_joint_pos, save_state = self.cmd_lst[self.cmd_idx]
            self.cmd_idx += 1

            action = np.zeros(35)
            action = joint_pos_controller(self.obs, target_joint_pos)
        else:
            target_joint_pos, save_state = self.cmd_lst[-1]
            save_state = False

            action = np.zeros(35)
            action = joint_pos_controller(self.obs, target_joint_pos)
            # print("action=", action[29:32], target_joint_pos[29:32], self.obs['robot0_joint_pos'][3:6])

        # # Directly set position
        # body_cmds = robosuite_cmds_to_body_cmds(target_joint_pos)
        # self.env.robots[0].set_robot_joint_positions(body_cmds)
        # action[:14] *= 0
        # action[26:] *= 0

        self.obs, reward, done, _ = self.env.step(action)
        self.reward = max(self.reward, reward)

        # # Directly set position
        # self.env.robots[0].set_robot_joint_positions(body_cmds)

        img, depth = self.get_camera_image()

        if save_state:
            self.saved_info.append({'joint_action': target_joint_pos, 
                                    'rgb': img, 
                                    'depth': depth,
                                    'joint_obs': init_joint_pos})

    def get_reward(self):
        return self.reward

    def run(self, vis=True):
        m = self.env.sim.model._model
        d = self.env.sim.data._data

        fps = 20

        if vis:
            with mujoco.viewer.launch_passive(
                model=m,
                data=d,
                show_left_ui=True,
                show_right_ui=True,
            ) as viewer:
                while viewer.is_running():
                    
                    self.sim_step()            

                    # extent = self.env.sim.model.stat.extent
                    # near = self.env.sim.model.vis.map.znear
                    # far = self.env.sim.model.vis.map.zfar
                    # print("near: ", near, "far: ", far, "extent: ", extent)

                    viewer.sync()
                    time.sleep(1 / fps)

                    if self.reset_episode_cmd or self.terminate_cmd:
                        break
        else:
            while True:

                self.sim_step()
                time.sleep(1 / fps)
                if self.reset_episode_cmd or self.terminate_cmd:
                    break
        
        if self.reset_episode_cmd:
            self.reset()
        else:
            print("sim terminted!")

def gripper_joint_pos_controller_xml(obs, desired_qpos):
    
    limits = [
        [0, 1.1],
        [0, 0.68],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
        [0, 1.3],
        [0, 0.68],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
        [0, 1.62],
    ]
    
    if desired_qpos[0] < limits[0][1] * 0.9:
        desired_qpos[1:4] = 0

    action = np.zeros(12)
    for i in range(12):
        # map desired_qpos to [-1, 1]
        action[i] = 2 * (desired_qpos[i] - limits[i][0]) / (limits[i][1] - limits[i][0]) - 1
    
    return action

def gripper_joint_pos_controller(obs, desired_qpos, kp=60, damping_ratio=1):
    """
    Calculate the torques for the joints position controller.

    Args:
        obs: dict, the observation from the environment
        desired_qpos: np.array of shape (12, ) that describes the desired qpos (angles) of the joints on hands, right hand first, then left hand

    Returns:
        desired_torque: np.array of shape (12, ) that describes the desired torques for the joints on hands
    """
    # get the current joint position and velocity
    actuator_idxs = [0, 1, 4, 6, 8, 10]
    # order:
    # 'gripper0_right_joint_r_thumb_proximal_1', 'gripper0_right_joint_r_thumb_proximal_2', 'gripper0_right_joint_r_thumb_middle', 'gripper0_right_joint_r_thumb_distal', 'gripper0_right_joint_r_index_proximal', 'gripper0_right_joint_r_index_distal', 'gripper0_right_joint_r_middle_proximal', 'gripper0_right_joint_r_middle_distal', 'gripper0_right_joint_r_ring_proximal', 'gripper0_right_joint_r_ring_distal', 'gripper0_right_joint_r_pinky_proximal', 'gripper0_right_joint_r_pinky_distal''gripper0_right_joint_r_thumb_proximal_1', 'gripper0_right_joint_r_thumb_proximal_2', 'gripper0_right_joint_r_thumb_middle', 'gripper0_right_joint_r_thumb_distal', 'gripper0_right_joint_r_index_proximal', 'gripper0_right_joint_r_index_distal', 'gripper0_right_joint_r_middle_proximal', 'gripper0_right_joint_r_middle_distal', 'gripper0_right_joint_r_ring_proximal', 'gripper0_right_joint_r_ring_distal', 'gripper0_right_joint_r_pinky_proximal', 'gripper0_right_joint_r_pinky_distal'
    joint_qpos = np.concatenate(
        (obs["robot0_right_gripper_qpos"][actuator_idxs], obs["robot0_left_gripper_qpos"][actuator_idxs])
    )
    joint_qvel = np.concatenate(
        (obs["robot0_right_gripper_qvel"][actuator_idxs], obs["robot0_left_gripper_qvel"][actuator_idxs])
    )

    position_error = desired_qpos - joint_qpos
    vel_pos_error = -joint_qvel

    # print("position error: ", position_error[2], vel_pos_error[2], position_error[8], vel_pos_error[8])

    # calculate the torques: kp * position_error + kd * vel_pos_error
    kd = 2 * np.sqrt(kp) * damping_ratio - 15
    desired_torque = np.multiply(np.array(position_error), np.array(kp)) + np.multiply(vel_pos_error, kd)

    # print("kp=", kp, "kd=", kd)

    # clip and rescale to [-1, 1]
    desired_torque = np.clip(desired_torque, -1, 1)

    # print("desired_torque", desired_torque[0])

    return desired_torque

def joint_pos_controller(obs, target_joint_pos):
    '''
    Joint position controller for GR1FloatingBody.
    Args:
        obs (dict): observation from the environment, where obs["robot0_joint_pos"] is the current joint positions of upperbody (without hands, 20-dim).
        target_joint_pos (np.array): target joint positions for the robot (35-dim).
    Returns:
        action (np.array): action to be taken by the robot (35-dim).
    '''

    action = np.zeros(35)

    # for right arm and left arm joints
    action[0:14] = np.clip(5 * (target_joint_pos[0:14] - obs["robot0_joint_pos"][6:20]), -3, 3)

    # for right gripper and left gripper
    action[14:26] = gripper_joint_pos_controller_xml(obs, target_joint_pos[14:26])

    # for base joints
    # TODO

    # for the head joints
    action[29:32] = (target_joint_pos[29:32] - obs["robot0_joint_pos"][3:6])
    
    # for the waist joints
    action[32:35] = np.clip(5 * (target_joint_pos[32:35] - obs["robot0_joint_pos"][0:3]), -3, 3)
    return action