import numpy as np
import os
import time
import pickle

from retarget.utils.configs import load_config
from retarget.retargeter import SMPLGR1Retargeter
import retarget

file_path = 'robosuite/scripts_okami/data/fold_clothes.pkl'
with open(file_path, 'rb') as file:
    # Load the content of the file
    smplh_traj = pickle.load(file)

retarget_repo_dir = os.path.dirname(retarget.__file__)
config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1_dex_retarget.yaml")
# config_path = os.path.join(retarget_repo_dir, "../configs/smpl_gr1.yaml")
config = load_config(config_path)
retargeter = SMPLGR1Retargeter(config, vis=True)
retargeter.calibrate(smplh_traj[0])

for d in smplh_traj:
    q = retargeter(d)

    right_joint_idx = [32 + i for i in [0, 1, 8, 10, 4, 6]]
    left_joint_idx = [13 + i for i in [0, 1, 8, 10, 4, 6]]

    q_right = q[right_joint_idx]
    q_left = q[left_joint_idx]

    print(q_right / np.pi * 180)

    time.sleep(0.05)