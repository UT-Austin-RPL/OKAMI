import math
import numpy as np
import matplotlib.pyplot as plt
import os
from pytransform3d import rotations

from pathlib import Path
import h5py
from tqdm import tqdm
import time
import cv2

import torch
from einops import rearrange

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.okami_utils import joint_pos_controller, urdf_to_robosuite_cmds, obs_to_urdf

import init_path
from okami.act.act.utils import joint_state_13_to_56, joint_state_26_to_56, name_to_urdf_idx, name_to_limits, KaedeVideoWriter, make_grid

from pathlib import Path
current_dir = Path(__file__).parent.resolve()

class Player:
    def __init__(self, single_arm=True, horizon=300, env_name="HumanoidPour"):
        
        self.single_arm = single_arm
        
        # Get controller config
        controller_config = load_controller_config(default_controller="JOINT_POSITION")
        controller_config["kp"] = 500
        
        # Create argument configuration
        config = {
            "env_name": env_name,
            "robots": "GR1FloatingBody",
            "controller_configs": controller_config,
        }

        # Create environment
        self.env = suite.make(
            **config,
            renderer="mujoco",
            has_offscreen_renderer=True,
            ignore_done=False,
            horizon=horizon,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_robotview", "frontview"],
            camera_heights=720,
            camera_widths=1280,
            camera_depths=True,
            control_freq=20,
        )
        print("Simulation environment initialized")
        
        self.reset()
    
    def get_state_and_images(self):
        joint_states_urdf = obs_to_urdf(self.obs) # shape (56,)
        right_idx = [25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 38, 40, 42]
        left_idx = [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 21, 23]
        if self.single_arm:
            joint_states =  joint_states_urdf[right_idx] # shape (13,)
        else:
            all_arms_idx = left_idx + right_idx
            joint_states = joint_states_urdf[all_arms_idx] # shape (26,)
        
        rgb_img = self.obs["robot0_robotview_image"]
        rgb_img = cv2.flip(rgb_img, 0)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        rgb_img = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_AREA)
        
        return joint_states, rgb_img
    
    def step(self, action, agentview_rgb):
        
        if self.single_arm:
            assert len(action) == 13
            urdf_q = joint_state_13_to_56(action)
        else:
            assert len(action) == 26
            urdf_q = joint_state_26_to_56(action)
        
        for j, (name, idx) in enumerate(name_to_urdf_idx.items()):
            limits = name_to_limits[name]
            # clip between limits
            urdf_q[idx] = max(limits[0], min(urdf_q[idx], limits[1]))
            
        target_joint_pos = urdf_to_robosuite_cmds(urdf_q)
        mujoco_action = joint_pos_controller(self.obs, target_joint_pos)
        
        self.obs, reward, done, info = self.env.step(mujoco_action)
        
        frame = self.obs['frontview_image']
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.episode_recording.append(frame)
        
        if done:
            print("Episode done!")
            if reward > 0.5:
                green_filter = np.zeros_like(frame)
                green_filter[:, :, 1] = 255
                alpha = 0.2
                success_img = cv2.addWeighted(frame, 1 - alpha, green_filter, alpha, 0)
                self.episode_recording.append(success_img)
            else:
                red_filter = np.zeros_like(frame)
                red_filter[:, :, 2] = 255
                alpha = 0.2
                fail_img = cv2.addWeighted(frame, 1 - alpha, red_filter, alpha, 0)
                self.episode_recording.append(fail_img)
        
        time.sleep(1/20)
        
        return reward, done

    def end(self):
        del self.env
    
    def reset(self):
        self.obs = self.env.reset()
        self.episode_recording = []
        time.sleep(3)
        print("start new episode!")
        
    def get_episode_recording(self):
        return self.episode_recording
    
    def render_single_episode_video(self, output_path, video_name):
        frames = self.get_episode_recording()
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        filename = Path(output_path) / f"{video_name}"
        os.makedirs(output_path, exist_ok=True)
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print("episode video saved at", filename)

    def render_multiple_episode_video(self, video_out, output_path, video_name):
        video_length = max([len(episode) for episode in video_out])
        video_out = [episode + [episode[-1]] * (video_length - len(episode)) for episode in video_out]
        
        print("Video_length=", video_length, "Making video...")
        video_writer = KaedeVideoWriter(output_path, save_video=True, video_name=video_name, fps=30, single_video=True)
        for i in range(video_length):
            images = []
            for episode in video_out:
                img = episode[i]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.flip(img, 0)
                images.append(torch.tensor(img))
            # images = [torch.tensor(episode[i]) for episode in video_out]
            for j in range(len(images)):
                images[j] = rearrange(images[j], 'h w c -> c h w')
            grid_image = make_grid(images, nrow=4).numpy()
            video_writer.append_image(grid_image)
        video_writer.save()

if __name__ == '__main__':
    
    dataset_path = 'annotations/human_demo/salt_demo/rollout/data.hdf5'
    
    with h5py.File(dataset_path, 'r') as f:
        root = f['data']['demo_0']
        actions = np.array(root['actions'][()])
        agentview_rgb = np.array(root['obs']['agentview_rgb'][()])
        states = np.array(root['obs']['joint_states'][()])
    
    timestamps = states.shape[0]
    single_arm = (len(actions[0]) == 13)
    
    player = Player(single_arm)
    
    try:
        for t in tqdm(range(timestamps)):
            player.step(actions[t], agentview_rgb[t])
        player.render_single_episode_video(current_dir, 'test_dataset.mp4')
    except KeyboardInterrupt:
        player.end()
        exit()
