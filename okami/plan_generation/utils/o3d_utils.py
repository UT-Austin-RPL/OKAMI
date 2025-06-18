"""Simple wrappers for using Open3D functionalities."""
import cv2
import json
import h5py
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from okami.plan_generation.utils.misc_utils import get_intrinsics_matrix_from_dict, get_extrinsics_matrix_from_dict, load_first_frame_from_human_hdf5_dataset, Timer

from okami.plan_generation.utils.log_utils import get_orion_logger

ORION_LOGGER = get_orion_logger("okami")

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
        ORION_LOGGER.debug(f"Selected pixel: {x}, {y}")
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

    def preprocess(self, use_rgb=True):
        num_points = self.get_num_points()

        if num_points < self.max_points:
            num_pad_points = self.max_points - num_points

            if num_pad_points > 0:
                # Randomly select points from the original point cloud for padding
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
            ORION_LOGGER.info("Plane equation: {:.2f}x + {:.2f}y + {:.2f}z + {:.2f} = 0".format(a, b, c, d))
            ORION_LOGGER.info("Number of inliers: {}".format(len(inliers)))
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
    ORION_LOGGER.debug(theta)

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


def global_registration(pcd_1_points, 
                        pcd_2_points,
                        voxel_size=0.005,
                        ransac_n=3,
                        ransac_max_iter=40000,
                        ransac_max_validation=500,
                        multiscale_icp=True):
    source_pcd = create_o3d_from_points_and_color(pcd_1_points)
    target_pcd = create_o3d_from_points_and_color(pcd_2_points)

    voxel_size = voxel_size  # A parameter you may need to adjust
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    normal_radius = 0.1
    fpfh_radius = 0.25
    fpfh_nn = 100
    # normal_radius = 0.01 # 0.1
    # fpfh_radius = 0.01 # 0.25
    # fpfh_nn = 50 # 100
    verbose = False
    with Timer(verbose=verbose) as timer:
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

        # Compute FPFH features
    with Timer(verbose=verbose) as timer:
        fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_nn))
        fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_nn))

    with Timer(verbose=verbose) as timer:
        # Prepare RANSAC based global registration
        distance_threshold = voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, fpfh_source, fpfh_target, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n, 
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_iter, ransac_max_validation))
        
    # (TODO): reject transformation whose rotation axis is deviated from the z-axis too much

    # Refine the alignment using ICP
    # result_icp = o3d.pipelines.registration.registration_icp(
    #     source_down, target_down, voxel_size, result.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
    result_transform = result.transformation
    if multiscale_icp:
        voxel_sizes = [0.05, 0.01, 0.005, 0.001]
    else:
        voxel_sizes = [voxel_size]
    source_down = source_pcd
    target_down = target_pcd
    for i in range(len(voxel_sizes)):
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_sizes[i], result_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
        source_down = source_pcd.voxel_down_sample(voxel_sizes[i])
        target_down = target_pcd.voxel_down_sample(voxel_sizes[i])
        result_transform = result_icp.transformation


    # result_icp = o3d.pipelines.registration.registration_icp(
    #     source_pcd, target_pcd, voxel_size * 0.01, result.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    #     )
    # Print the transformation matrix
    return result_icp.transformation


def load_reconstruction_info_from_human_demo(dataset_name, camera_name="front_camera"):
    # human_first_frame_depth = load_first_frame_from_human_hdf5_dataset(dataset_name, image_name="agentview_depth")
    with h5py.File(dataset_name, "r") as f:
        data_config = json.loads(f["data"].attrs["data_config"])
        print("data_config: ", data_config)
        if camera_name not in data_config["intrinsics"].keys():
            camera_name = "iphone"
        camera_intrinsics = data_config["intrinsics"][camera_name]
        camera_intrinsics_matrix = get_intrinsics_matrix_from_dict(camera_intrinsics)
        camera_extrinsics = data_config["extrinsics"][camera_name]
        camera_extrinsics_matrix = get_extrinsics_matrix_from_dict(camera_extrinsics)

    if camera_intrinsics_matrix[0][0] == 0:
        # it means that the data does not have intrinsics. In this project, such data is assumed to be from metric depth estimation.
        camera_intrinsics_matrix = np.array(
            # [[640, 0, 240],
            # [0, 640, 321],
            # [0, 0, 1]]
        [[1.97547873e+03, 0, 1.06077279e+03],
        [0, 2.05341424e+03, 5.13500761e+02],
        [0, 0, 1]]
        )
    info = {
        # "depth": human_first_frame_depth,
        "intrinsics": camera_intrinsics_matrix,
        "extrinsics": camera_extrinsics_matrix
    }
    return info



def project3Dto2D(points, camera_intrinsics):
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    z = points[:, 2] * 1000
    u = points[:, 0] * 1000 / z * fx + cx
    v = points[:, 1] * 1000 / z * fy + cy
    return u.astype(np.int32), v.astype(np.int32)

def remove_outlier(point_array, color_array=None, nb_neighbors=60, std_ratio=0.7):
    """This function directly operates on the 3d point clouds"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    if color_array is not None:
        pcd.colors = o3d.utility.Vector3dVector(color_array)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    if color_array is not None:
        return np.asarray(inlier_cloud.points), np.asarray(inlier_cloud.colors)
    else:
        return np.asarray(inlier_cloud.points)

def filter_pcd(input_depth, camera_intrinsics, nb_neighbors=40, std_ratio=0.7):
    o3d_pcd = O3DPointCloud()
    o3d_pcd.create_from_depth(input_depth, camera_intrinsics)
    
    cl, ind = o3d_pcd.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = o3d_pcd.pcd.select_by_index(ind)
    inlier_points = np.asarray(inlier_cloud.points)
    # project to image
    new_segmentation = np.zeros_like(input_depth).astype(np.uint8)
    u, v = project3Dto2D(inlier_points, camera_intrinsics)
    new_segmentation[v, u] = 1
    return new_segmentation