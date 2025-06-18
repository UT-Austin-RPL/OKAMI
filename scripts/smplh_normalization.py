import argparse
import os
import pickle

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as RRR

import init_path
from okami.plan_generation.algos.smpl_render import SMPLHRender

os.environ["DISPLAY"] = ":0.0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=True)

    smpl_model_dir = "configs/smpl_models"

    betas = torch.from_numpy(np.load("configs/smpl_models/betas.npy")).float()

    rot = RRR.from_euler("xyz", [-np.pi / 2, 0, np.pi / 2])
    global_orient = rot * RRR.from_rotvec(data["root_orient"][0])
    global_orient = global_orient.as_rotvec()

    transl = data["trans"]
    transl = transl.dot(rot.as_matrix().T)

    global_orient = torch.from_numpy(global_orient).float().squeeze(0)
    transl = torch.from_numpy(transl).float().squeeze(0)
    body_pose = torch.from_numpy(data["pose_body"]).float().squeeze(0)
    hand_pose = torch.from_numpy(data["hand_pose"]).float().squeeze(0)
    left_hand_pose = hand_pose[:, :45].to("cuda")
    right_hand_pose = hand_pose[:, 45:].to("cuda")

    init_params = {
        "gender": "male",
        "num_pca_comps": 45,
        "flat_hand_mean": False,
        "use_pca": True,
        "batch_size": 1,
    }

    renderer = SMPLHRender(smpl_model_dir, init_params)
    left_hand_pose_raw = (
        (
            torch.einsum("bi,ij->bj", [left_hand_pose, renderer.smplh.left_hand_components])
            + renderer.smplh.left_hand_mean.unsqueeze(0)
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    right_hand_pose_raw = (
        (
            torch.einsum("bi,ij->bj", [right_hand_pose, renderer.smplh.right_hand_components])
            + renderer.smplh.right_hand_mean.unsqueeze(0)
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    res = [720, 720]
    joint_transforms = renderer.init_renderer(
        res=res,
        smplh_param={
            "betas": betas[:, :16].expand(global_orient.shape[0], -1),
            "global_orient": global_orient,
            "body_pose": body_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "transl": transl,
        },
    ).astype(np.float64)

    positions = joint_transforms[:, :, :3, 3]
    orientations = joint_transforms[:, :, :3, :3]
    positions -= positions[:, [0], :]
    left_rotation = (
        RRR.from_rotvec(left_hand_pose_raw.reshape(-1, 3)).as_euler("xyz").reshape(-1, 15, 3)
    )
    right_rotation = (
        RRR.from_rotvec(right_hand_pose_raw.reshape(-1, 3)).as_euler("xyz").reshape(-1, 15, 3)
    )

    left_wrist_inv = np.linalg.inv(joint_transforms[:, 20])
    right_wrist_inv = np.linalg.inv(joint_transforms[:, 21])
    
    left_hand = np.einsum("bij,bnjk->bnik", left_wrist_inv, joint_transforms[:, 22:37])
    right_hand = np.einsum("bij,bnjk->bnik", right_wrist_inv, joint_transforms[:, 37:52])
    assert left_hand.shape == (joint_transforms.shape[0], 15, 4, 4)
    assert right_hand.shape == (joint_transforms.shape[0], 15, 4, 4)
    data = [
        {
            "left_fingers": left_hand[i],
            "right_fingers": right_hand[i],
            "body": joint_transforms[i],
            "left_angles": left_hand_pose_raw[i],
            "right_angles": right_hand_pose_raw[i],
        }
        for i in range(joint_transforms.shape[0])
    ]

    output = args.input[:-4] + ".pkl"
    with open(output, "wb") as f:
        print(f"Save hand pose to {output}")
        pickle.dump(data, f)