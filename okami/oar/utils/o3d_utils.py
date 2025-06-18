"""Simple wrappers for using Open3D functionalities."""
import cv2
import json
import h5py
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def convert_convention(image, real_robot=True):
    if not real_robot:
        if macros.IMAGE_CONVENTION == "opencv":
            return np.ascontiguousarray(image[::1])
        elif macros.IMAGE_CONVENTION == "opengl":
            return np.ascontiguousarray(image[::-1])
    else:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            return np.ascontiguousarray(image)

def transform_point_clouds(transformation, points):
    new_points = transformation @ np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T
    new_points = new_points[:3, :].T
    return new_points

class O3DPointCloud():
    def __init__(self, 
                 max_points=512):
        self.pcd = o3d.geometry.PointCloud()

        self.max_points = max_points
        

    def create_from_rgbd(self, color, depth, intrinsic_matrix, convert_rgb_to_intensity=False, depth_scale=1000.0, depth_trunc=3.0):
        """Create a point cloud from RGB-D images.

        Args:
            color (np.ndarray): RGB image.
            depth (np.ndarray): Depth image.
            intrinsic_matrix (np.ndarray): Intrinsic matrix.
            convert_rgb_to_intensity (bool, optional): Whether to convert RGB to intensity. Defaults to False.
        """
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            convert_rgb_to_intensity=convert_rgb_to_intensity,
            depth_scale=depth_scale, depth_trunc=depth_trunc)
        
        width, height = color.shape[:2]
        pinholecameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix= intrinsic_matrix)
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinholecameraIntrinsic)

    def create_from_depth(self, depth, intrinsic_matrix, depth_trunc=5):
        width, height = depth.shape[:2]
        pinholecameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix= intrinsic_matrix)

        self.pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), pinholecameraIntrinsic, depth_trunc=depth_trunc)
    
    def create_from_selected_pixel(self, selected_pixel, color, depth, intrinsic_matrix, convert_rgb_to_intensity=False, region_size=2):
        y, x = selected_pixel
        x = int(x)
        y = int(y)
        new_depth = depth.copy()
        binary_mask = np.zeros_like(depth)
        print(f"Selected pixel: {x}, {y}")
        binary_mask[x-region_size:x+region_size, y-region_size:y+region_size] = 1
        new_depth = new_depth * binary_mask
        self.create_from_rgbd(color, new_depth, intrinsic_matrix, convert_rgb_to_intensity=convert_rgb_to_intensity)

    def create_from_keypoints(self, keypoints, color, depth, intrinsic_matrix, convert_rgb_to_intensity=False, region_size=2):
        new_depth = depth.copy()
        binary_mask = np.zeros_like(depth)
        for keypoint in keypoints:
            y, x = keypoint
            x = int(x)
            y = int(y)
            binary_mask[x, y] = 1
        new_depth = new_depth * binary_mask
        self.create_from_rgbd(color, new_depth, intrinsic_matrix, convert_rgb_to_intensity=convert_rgb_to_intensity)
    
    def create_from_points(self, points):
        # points: (num_points, 3)
        self.pcd.points = o3d.utility.Vector3dVector(points)

    def add_more_points(self, points):
        # points: (num_points, 3)
        self.pcd.points.extend(o3d.utility.Vector3dVector(points))

    def preprocess(self, use_rgb=True):
        num_points = self.get_num_points()

        if num_points < self.max_points:
            num_pad_points = self.max_points - num_points

            if num_pad_points > 0:
                # Randomly select points from the original point cloud for padding
                if num_points == 0:
                    print("Warning: No points in the point cloud. Cannot pad.")
                else:
                    pad_indices = np.random.randint(0, num_points, size=(num_pad_points,))
                    pad_points = self.get_points()[pad_indices]
                    if use_rgb:
                        pad_colors = self.get_colors()[pad_indices]
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(pad_points)
                    if use_rgb:
                        new_pcd.colors = o3d.utility.Vector3dVector(pad_colors)
                    self.pcd += new_pcd
        else:
            self.pcd = self.pcd.random_down_sample(self.max_points / num_points)
            # In case downsampling results in fewer points
            if self.get_num_points() < self.max_points:
                self.preprocess(use_rgb=use_rgb)

    def transform(self, extrinsic_matrix):
        """Transform the point cloud.

        Args:
            extrinsic_matrix (np.ndarray): Extrinsic matrix.
        """
        return self.pcd.transform(extrinsic_matrix)
    
    def get_points(self):
        """Get the points.

        Returns:
            np.ndarray: (num_points, 3), where each point is (x, y, z).
        """
        return np.asarray(self.pcd.points)
    
    def get_num_points(self):
        """Get the number of points.

        Returns:
            int: Number of points.
        """
        return len(self.get_points())
    
    def get_colors(self):
        """Get the colors.

        Returns:
            np.ndarray: (num_points, 3), where each color is (r, g, b).
        """
        return np.asarray(self.pcd.colors)
    
    def save(self, filename):
        assert(filename.endswith(".ply")), "Only .ply format is supported."
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.get_points())
        pcd.colors = o3d.utility.Vector3dVector(self.get_colors())
        o3d.io.write_point_cloud(filename, pcd)

    def plane_estimation(self, distance_threshold=0.001, ransac_n=100, num_iterations=1000, verbose=True):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.get_points())
        pcd.colors = o3d.utility.Vector3dVector(self.get_colors())
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        [a, b, c, d] = plane_model
        if verbose:
            print("Plane equation: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(a, b, c, d))
            print("Number of inliers: {}".format(len(inliers)))
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return {
            "plane_model": plane_model,
            "inliers": inliers,
            "inlier_cloud": inlier_cloud,
            "outlier_cloud": outlier_cloud
        }

def visualize_o3d_point_cloud(o3d_pcd):
    point_cloud = o3d_pcd.get_points()
    colors_rgb = o3d_pcd.get_colors()

    color_str = ['rgb('+str(r)+','+str(g)+','+str(b)+')' for r,g,b in colors_rgb]

    # Extract x, y, and z columns from the point cloud
    x_vals = point_cloud[:, 0]
    y_vals = point_cloud[:, 1]
    z_vals = point_cloud[:, 2]

    # Create the scatter3d plot
    rgbd_scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(size=3, color=color_str, opacity=0.8)
    )

    # Set the layout for the plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[rgbd_scatter], layout=layout)
    # Show the figure
    fig.show()

def scene_pcd_fn(
        rgb_img_input,
        depth_img_input,
        intrinsic_matrix,
        extrinsic_matrix,
        max_points=10000,
        is_real_robot=True,
        downsample=True,
        depth_trunc=3.0
    ):
        rgbd_pc = O3DPointCloud(max_points=max_points)
        rgbd_pc.create_from_rgbd(rgb_img_input, depth_img_input, intrinsic_matrix, depth_trunc=depth_trunc)
        rgbd_pc.transform(extrinsic_matrix)
        if downsample:
            rgbd_pc.preprocess()

        return rgbd_pc.get_points(), rgbd_pc.get_colors()


def create_o3d_from_points_and_color(pcd_points, pcd_colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    if pcd_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    return pcd


def estimate_rotation(plane_model, z_up=True):
    # Normal vector of the plane
    a, b, c, d = plane_model
    n = np.array([a, b, c])

    # Z-axis unit vector
    if z_up:
        k = np.array([0, 0, 1])
    else:
        # z down case
        k = np.array([0, 0, -1])

    # Calculate the rotation axis (cross product of n and k)
    axis = np.cross(n, k)

    # Normalize the rotation axis
    axis_normalized = axis / np.linalg.norm(axis)

    # Calculate the angle of rotation (dot product and arccosine)
    cos_theta = np.dot(n, k) / np.linalg.norm(n)
    theta = np.arccos(cos_theta)
    # theta = 2.1
    print(theta)

    # Rodrigues' rotation formula
    # Skew-symmetric matrix of axis
    axis_skew = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                        [axis_normalized[2], 0, -axis_normalized[0]],
                        [-axis_normalized[1], axis_normalized[0], 0]])

    # Rotation matrix
    R = np.eye(3) + np.sin(theta) * axis_skew + (1 - np.cos(theta)) * np.dot(axis_skew, axis_skew)
    T = np.eye(4)
    T[:3, :3] = R
    return T
