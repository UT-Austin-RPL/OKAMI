import glob
import os
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import trimesh
from scipy.spatial.transform.rotation import Rotation as RRR

import init_path
from okami.plan_generation.algos.smpl_mod import smplx


class Renderer:
    """Renderer used for visualizing the SMPL model Code adapted from
    https://github.com/vchoutas/smplify-x."""

    def __init__(self, vertices, focal_length=5000, img_res=(224, 224), faces=None):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res[0], viewport_height=img_res[1], point_size=2.0
        )

        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.faces = faces

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # self.rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

        minx, miny, minz = vertices.min(axis=(0, 1))
        maxx, maxy, maxz = vertices.max(axis=(0, 1))
        minx = minx - 0.5
        maxx = maxx + 0.5
        miny = miny - 0.5
        maxy = maxy + 0.5

        self.floor = trimesh.creation.box(extents=(maxx - minx, maxy - miny, 1e-5))
        self.floor.visual.face_colors = [0, 0, 0, 0.2]
        self.floor.apply_translation([(minx + maxx) / 2, (miny + maxy) / 2, minz])

        self.floor_pose = np.eye(4)

        # c = -np.pi / 6
        self.camera_pose = np.eye(4)
        self.camera_pose[:3, 3] = np.array([(minx + maxx) / 2 + 4, (miny + maxy) / 2, minz + 3])
        self.camera_pose[:3, :3] = RRR.from_euler("xyz", [np.pi / 3, 0, np.pi / 2]).as_matrix()

    def __call__(self, vertices, camera_pose=None):
        scene = self.create_scene_human_floor(vertices, camera_pose)
        color = self.render_scene(scene)
        return color

    def render_scene(self, scene):
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL
        color, rend_depth = self.renderer.render(scene, flags=flags)
        return color

    def create_scene_human_floor(self, vertices, camera_pose=None):

        floor_render = pyrender.Mesh.from_trimesh(self.floor, smooth=False)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1, alphaMode="OPAQUE", baseColorFactor=(0.658, 0.214, 0.0114, 0.2)
        )
        mesh = trimesh.Trimesh(vertices, self.faces)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=800)

        scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 0.8), ambient_light=(0.4, 0.4, 0.4))
        scene.add(floor_render, pose=self.floor_pose)
        scene.add(mesh, "mesh")

        light_pose = self.camera_pose.copy()
        scene.add(light, pose=light_pose)
        # light_pose[:3, 3] = np.array([0, 2, 2])
        # scene.add(light, pose=light_pose)
        # light_pose[:3, 3] = np.array([2, 2, 4])
        # scene.add(light, pose=light_pose)

        scene.add(camera, pose=self.camera_pose if camera_pose is None else camera_pose)
        return scene


class SMPLRender:
    def __init__(self, SMPL_MODEL_DIR):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(self.device)
        self.smpl = smplx.create(
            Path(SMPL_MODEL_DIR),
            model_type="smpl",
            gender="neutral",
            ext="npz",
            batch_size=1,
        ).to(self.device)

        self.focal_length = 110

    def smpl_forward(self, smpl_param):
        poses = smpl_param["pred_pose"]
        pred_rotmats = []
        for pose in poses:
            if pose.size == 72:
                pose = pose.reshape(-1, 3)
                pose = RRR.from_rotvec(pose).as_matrix()
                pose = pose.reshape(1, 24, 3, 3)
            pred_rotmats.append(torch.from_numpy(pose.astype(np.float32)[None]).to(self.device))
        pred_rotmat = torch.cat(pred_rotmats, dim=0)

        pred_betas = torch.from_numpy(
            smpl_param["pred_shape"].reshape(1, 10).astype(np.float32)
        ).to(self.device)
        pred_root = torch.tensor(
            smpl_param["pred_root"].reshape(-1, 3).astype(np.float32), device=self.device
        )
        smpl_output = self.smpl(
            betas=pred_betas,
            body_pose=pred_rotmat[:, 1:],
            transl=pred_root,
            global_orient=pred_rotmat[:, :1],
            pose2rot=False,
        )
        return smpl_output

    def init_renderer(self, res, smpl_param, is_headroot=False):
        poses = smpl_param["pred_pose"]
        pred_rotmats = []
        for pose in poses:
            if pose.size == 72:
                pose = pose.reshape(-1, 3)
                pose = RRR.from_rotvec(pose).as_matrix()
                pose = pose.reshape(24, 3, 3)
            pred_rotmats.append(torch.from_numpy(pose.astype(np.float32)[None]).to(self.device))
        pred_rotmat = torch.cat(pred_rotmats, dim=0)
        print(pred_rotmat.shape)

        pred_betas = torch.from_numpy(
            smpl_param["pred_shape"].reshape(1, 10).astype(np.float32)
        ).to(self.device)
        pred_root = torch.tensor(
            smpl_param["pred_root"].reshape(-1, 3).astype(np.float32), device=self.device
        )
        smpl_output = self.smpl(
            betas=pred_betas,
            body_pose=pred_rotmat[:, 1:],
            transl=pred_root,
            global_orient=pred_rotmat[:, :1],
            pose2rot=False,
        )

        self.vertices = smpl_output.vertices.detach().cpu().numpy()
        pred_root = pred_root[0]

        if is_headroot:
            pred_root = pred_root - smpl_output.joints[0, 12].detach().cpu().numpy()

        self.renderer = Renderer(
            vertices=self.vertices,
            focal_length=self.focal_length,
            img_res=(res[1], res[0]),
            faces=self.smpl.faces,
        )
        return smpl_output.joint_transforms.detach().cpu().numpy()

    def render(self, index):
        renderImg = self.renderer(self.vertices[index, ...])
        return renderImg


class SMPLHRender:
    def __init__(self, SMPLX_MODEL_DIR, init_params):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.smplh = smplx.create(Path(SMPLX_MODEL_DIR), model_type="smplh", **init_params).to(
            self.device
        )

        self.focal_length = 110

    def init_renderer(self, res, smplh_param):
        for k, v in smplh_param.items():
            if isinstance(v, torch.Tensor):
                smplh_param[k] = v.to(self.device)
            else:
                smplh_param[k] = torch.tensor(v, device=self.device)

        smpl_output = self.smplh(**smplh_param)

        self.vertices = smpl_output.vertices.detach().cpu().numpy()
        joint_transforms = smpl_output.joint_transforms.detach().cpu().numpy()

        self.renderer = Renderer(
            vertices=self.vertices,
            focal_length=self.focal_length,
            img_res=(res[1], res[0]),
            faces=self.smplh.faces,
        )
        return joint_transforms

    def render(self, index):
        renderImg = self.renderer(self.vertices[index, ...])
        return renderImg