import importlib
import os
import tempfile
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

import numpy as np


def get_root_dir():
    """Get the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def import_class(full_class_string):
    """Dynamically import a class from a given string.

    Args:
    full_class_string (str): Full path to the class 'module.submodule.ClassName'

    Returns:
    type: The class referred to by 'full_class_string'
    """
    class_data = full_class_string.split(".")
    module_path = ".".join(class_data[:-1])
    class_name = class_data[-1]

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def compute_transformation_error(transform_src, transform_dst):
    rot_src = transform_src[:3, :3]
    rot_dst = transform_dst[:3, :3]
    translation_src = transform_src[:3, 3]
    translation_dst = transform_dst[:3, 3]
    transl_error = np.linalg.norm(translation_src - translation_dst)
    rot_error = np.arccos((np.trace(np.dot(rot_src.T, rot_dst)) - 1) / 2)
    return {"translation_error": transl_error, "rotation_error": rot_error}


def fix_urdf_joints(file_path, joint_names):
    # Load the URDF file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Modify the joint types
    for joint in root.findall("joint"):
        if joint.get("name") in joint_names:
            # Change the type attribute to 'fixed'
            joint.set("type", "fixed")

    # Create a temporary file to save the modified URDF
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".urdf") as temp_file:
        tree.write(temp_file.name)
        return temp_file.name

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