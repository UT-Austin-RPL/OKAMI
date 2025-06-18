import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import time 
from einops import rearrange
import IPython
e = IPython.embed
from pathlib import Path

import imageio
import torchvision

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_path, camera_names, norm_stats, episode_len, history_stack=0):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path = dataset_path
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_pad_len = 200

        self.history_stack = history_stack

        self.is_sims = []
        self.original_action_shapes = []

        self.states = []
        self.image_dict = dict()
        for cam_name in self.camera_names:
            self.image_dict[cam_name] = []
        self.actions = []

        # Read in all the data
        with h5py.File(self.dataset_path, 'r') as f:
            root = f['data']
            for i, episode_id in enumerate(self.episode_ids):
                self.is_sims.append(None)
                self.original_action_shapes.append(root[f'demo_{episode_id}']['actions'][()].shape)

                self.states.append(np.array(root[f'demo_{episode_id}']['obs']['joint_states'][()]))
                for cam_name in self.camera_names:
                    imgs = root[f'demo_{episode_id}']['obs'][cam_name][()]
                    # rearrange the image data to be in the form of (time, C, H, W)
                    imgs = rearrange(imgs, 't h w c -> t c h w')
                    self.image_dict[cam_name].append(imgs)
                self.actions.append(np.array(root[f'demo_{episode_id}']['actions'][()]))

        self.is_sim = None 

        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts
    
    def __getitem__(self, ts_index):
        sample_full_episode = False # hardcode

        index, start_ts = self._locate_transition(ts_index)

        original_action_shape = self.original_action_shapes[index]
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.states[index][start_ts]

        if self.history_stack > 0:
            last_indices = np.maximum(0, np.arange(start_ts-self.history_stack, start_ts)).astype(int)
            last_action = self.actions[index][last_indices, :]

        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = self.image_dict[cam_name][index][start_ts]
        # get all actions after and including start_ts
        all_time_action = self.actions[index][:]

        # pad the last action for max_pad_len times
        all_time_action_padded = np.zeros((self.max_pad_len+original_action_shape[0], original_action_shape[1]), dtype=np.float32) 
        all_time_action_padded[:episode_len] = all_time_action
        all_time_action_padded[episode_len:] = all_time_action[-1]
        
        padded_action = all_time_action_padded[start_ts:start_ts+self.max_pad_len] 
        real_len = episode_len - start_ts

        is_pad = np.zeros(self.max_pad_len)
        is_pad[real_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.history_stack > 0:
            last_action_data = torch.from_numpy(last_action).float()

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.history_stack > 0:
            last_action_data = (last_action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            qpos_data = torch.cat((qpos_data, last_action_data.flatten()))
        # print(f"qpos_data: {qpos_data.shape}, action_data: {action_data.shape}, image_data: {image_data.shape}, is_pad: {is_pad.shape}")
        
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path, num_episodes):
    
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    
    with h5py.File(dataset_path, 'r') as f:
        root = f['data']
        for episode_idx in range(num_episodes):
            qpos = root[f'demo_{episode_idx}']['obs']['joint_states'][()]
            action = root[f'demo_{episode_idx}']['actions'][()]
            
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
            all_episode_len.append(len(qpos))
    
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True).float()  # (episode, timstep, action_dim)
    action_std = all_action_data.std(dim=0, keepdim=True).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True).float()
    qpos_std = all_qpos_data.std(dim=0, keepdim=True).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}
    print("stats: ", stats)

    return stats, all_episode_len

def find_episodes_num(dataset_path):
    """
    Output number of episodes in the dataset.
    """
    with h5py.File(dataset_path, 'r') as f:
        root = f['data']
        episodes_num = root.attrs['num_demos'] 
    return episodes_num

def BatchSampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_path, camera_names, batch_size_train, batch_size_val=None):
    print(f'\nData from: {dataset_path}\n')

    num_episodes = find_episodes_num(dataset_path)
    
    # No validation set
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:]
    print(f"Train episodes: {len(train_indices)}")
    
    # obtain normalization stats for qpos and action
    norm_stats, all_episode_len = get_norm_stats(dataset_path, num_episodes)

    train_episode_len_l = [all_episode_len[i] for i in train_indices]
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_path, camera_names, norm_stats, train_episode_len_l)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=16, prefetch_factor=2)

    return train_dataloader, None, norm_stats, train_dataset.is_sim

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_id(base_dir, prefix):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"The provided base directory does not exist or is not a directory: \n{base_path}")

    # Loop through all subdirectories of the base path
    for subfolder in base_path.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith(prefix):
            return str(subfolder), subfolder.name
    
    # If no matching subfolder is found
    return None, None

def find_all_ckpt(base_dir, prefix="policy_epoch_"):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError("The provided base directory does not exist or is not a directory.")

    ckpt_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.startswith(prefix):
            ckpt_files.append(file.name)
    # find latest ckpt
    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split(prefix)[-1].split('_')[0]), reverse=True)
    epoch = int(ckpt_files[0].split(prefix)[-1].split('_')[0])
    return ckpt_files[0], epoch

def extend_urdf_finger_cmds(finger_cmds):
    """
    Convert 6-dim finger cmds to 12-dim finger cmds in urdf.
    """
    mapping_base = [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    mapping_scale = [1, 1, 1, 1.13, 1, 1.08, 1, 1.15, 1, 1.13, 1, 1.13]

    actions = finger_cmds[mapping_base] * mapping_scale

    return np.array(actions)

def extend_delta_from_13_to_56(right_arm_q):
    # right_arm_q: shape (13,)
    # return: shape (56,)
    
    q = np.zeros(56)
    
    q[25:32] = right_arm_q[:7]
    q[32:44] = extend_urdf_finger_cmds(right_arm_q[7:])
    
    return q

def joint_state_13_to_56(joint_states):
    """
    Convert 13-dim joint states to 56-dim joint states in urdf.
    """
    q = np.zeros(56)
    
    head_init = np.array([0.0, 0.0, 0.34])
    torso_init = np.array([0.0, 0.22, 0.0])
    q[:3] = torso_init
    q[3:6] = head_init
    
    q[25:32] = joint_states[:7]
    q[32:44] = extend_urdf_finger_cmds(joint_states[7:])
    return q

def joint_state_26_to_56(joint_states):
    """
    Convert 26-dim joint states to 56-dim joint states in urdf.
    """
    q = np.zeros(56)
    
    head_init = np.array([0.0, 0.0, 0.34])
    torso_init = np.array([0.0, 0.22, 0.0])
    q[:3] = torso_init
    q[3:6] = head_init
    
    # left arm and hand
    q[6:13] = joint_states[:7]
    q[13:25] = extend_urdf_finger_cmds(joint_states[7:13])
    
    # right arm and hand
    q[25:32] = joint_states[13:20]
    q[32:44] = extend_urdf_finger_cmds(joint_states[20:26])
    
    return q

def joint_state_26_to_56_real(joint_states):
    """
    Convert 26-dim joint states to 56-dim joint states in urdf for real robot deployment.
    """
    q = np.zeros(56)
    
    head_init = np.array([0.0, 0.0, 19 / 180 * np.pi])
    torso_init = np.array([0.0, 0.0, 0.0])
    q[:3] = torso_init
    q[3:6] = head_init
    
    # left arm and hand
    q[6:13] = joint_states[:7]
    q[13:25] = extend_urdf_finger_cmds(joint_states[7:13])
    
    # right arm and hand
    q[25:32] = joint_states[13:20]
    q[32:44] = extend_urdf_finger_cmds(joint_states[20:26])
    
    return q

def joint_state_56_to_26(joint_states_urdf):
    """
    Convert 56-dim joint states in urdf to 26-dim joint states for neural network processing.
    """

    right_idx = [25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 38, 40, 42]
    left_idx = [6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19, 21, 23]
    all_arms_idx = left_idx + right_idx
    joint_states = joint_states_urdf[all_arms_idx] # shape (26,)
    
    return joint_states

def make_grid(images, nrow=8, padding=2, normalize=False, pad_value=0):
    """Make a grid of images. Make sure images is a 4D tensor in the shape of (B x C x H x W)) or a list of torch tensors."""
    grid_image = torchvision.utils.make_grid(images, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).permute(1, 2, 0)
    return grid_image

class KaedeVideoWriter():
    def __init__(self, video_path, save_video=False, video_name=None, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.single_video = single_video
        self.last_images = {}
        if video_name is None:
            self.video_name = "video.mp4"
        else:
            self.video_name = video_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save(self.video_name)

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_image(self, image, idx=0):
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            self.image_buffer[idx].append(image[::-1])

    def append_vector_obs(self, images):
        if self.save_video:
            for i in range(len(images)):
                self.append_image(images[i], i)

    def save(self, video_name=None, flip=False, bgr=False):
        if video_name is None:
            video_name = self.video_name
        img_convention = 1
        color_convention = 1
        if flip:
            img_convention = -1
        if bgr:
            color_convention = -1
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, video_name)
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im[::img_convention, :, ::color_convention])
                    video_writer.close()
            print(f"Saved videos to {video_name}.")


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


if __name__ == '__main__':
    load_data("annotations/human_demo/salt_demo/rollout/data.hdf5", ["agentview_rgb"], 64)