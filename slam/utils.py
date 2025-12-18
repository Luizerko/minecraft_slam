import ast
import cv2
import numpy as np
import os
import open3d as o3d
import pandas as pd
import sys
import torch

from plyfile import PlyData, PlyElement

# Building camera intrinsics K
def camera_intrinsics(image_width=640, image_height=360, horizontal_fov_deg=70.0):
    # Building intrinsics matrix (camera coords to pixel coords)
    w, h = image_width, image_height

    # Computing horizontal focal length from hFoV
    hFoV = np.deg2rad(horizontal_fov_deg)
    fx = w / (2.0 * np.tan(hFoV / 2.0))

    # Since we assume square pixels, we can simply do fx = fy
    fy = fx

    # Computing principal point (where the camera's optical axis hits the image)
    cx, cy = w / 2.0, h / 2.0

    # Finally building the matrix
    K = torch.Tensor([[fx, 0.0, cx],\
                      [0.0, fy, cy],\
                      [0.0, 0.0, 1.0]])

    return K


# Inverse sigmoid function for logits
def _inverse_sigmoid(x):
    return np.log(x / (1 - x + 1e-7))


# Initializing 3D Gaussians based on RGB-D images. We 
# initialize one gaussian per pixel and we'll reduce
# later during optimization
def init_gaussians(K, image, depth, max_depth=20, device='cuda'):
    # Preparing input
    if image.max() > 1.0:
        image = image / 255.0
    image = torch.from_numpy(image).to(device)
    depth = torch.from_numpy(depth).to(device)
    K = K.to(device)

    # Creating a meshgrid of pixel coordinates (u, v)
    H, W, _ = image.shape
    i, j = torch.meshgrid(torch.arange(H, device=device), 
                          torch.arange(W, device=device), 
                          indexing='ij')
    u = j.flatten()
    v = i.flatten()
    z = depth.flatten()
    
    # Filtering out invalid depth (0.1 to avoid very close
    # points collpasing into the camera and max_depth to
    # restrict our range of view and reconstruction)
    valid_mask = (z >= 0.1) & (z <= max_depth)
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]
    colors = image.reshape(-1, 3)[valid_mask].to(device)

    # Backprojection assuming camera frame is the same as
    # world frame (so there's no need for extrinsics)
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = torch.stack([x, y, z], dim=1).to(device)

    # Initializing scales. We need bigger gaussians on far 
    # distances because they comprehend more region, 
    # (even though less pixels), so at depth Z, a pixel 
    # should cover roughly Z/f world units. We also 
    # initialize isotropic Gaussians, so 
    # scale_x = scale_y = scale_z (and we account for 
    # covariance with rotations because it's much easier
    # than optimizing a 3x3 covariance matrix that always 
    # needs to be positive semi-definite). Finally and
    # we use log-space for stability during training later
    scale_scalar = z / fx 
    scales = scale_scalar.unsqueeze(1).repeat(1, 3).to(device)
    log_scales = torch.log(scales + 1e-6).to(device)

    # Initializing rotation quaternions: [w, x, y, z]. 
    # We rebuild covariance later with 
    # \mathbf{\sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T. 
    # Initially identity rotation, so for all points: 
    # [1, 0, 0, 0]
    rotations = torch.zeros((xyz.shape[0], 4), device=device)
    rotations[:, 0] = 1.0 

    # Initializing opacities and using logit space for 
    # optimizations
    opacities = torch.ones((xyz.shape[0], 1), \
                           device=device) * 0.9
    logit_opacities = torch.ones((xyz.shape[0], 1), \
                                 device=device) * _inverse_sigmoid(0.9)
    
    # Spherical harmonics base color coefficient for 
    # (diffuse) colors. The DC component (f_dc) therefore 
    # is just the RGB color adjusted by a constant 
    # SH_{C0} = 0.28209479177387814 (first SH function, 
    # representing ambient light), so f_dc = (RGB - 0.5) / SH_C0 
    # (the conversion used in 3DGS code)
    SH_C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / SH_C0 

    return xyz, colors, scales, log_scales, rotations, opacities, \
           logit_opacities, f_dc


# Computing Open3D intrinsics
def o3d_intrinsics(K):
    fx = K[0, 0].item()
    fy = K[1, 1].item()
    cx = K[0, 2].item()
    cy = K[1, 2].item()
    
    K_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=image_width, height=image_height, \
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    return K_o3d


# Tracking: estimating camera motion by aligning 
# point-cloud from frame t with point-cloud of 
# frame t-1 using ICP
def tracking(K, curr_rgb, curr_depth, prev_rgb, \
             prev_depth, max_depth=20.0, \
             prev_camera_pose=np.eye(4)):
    # Open3D intrinsics computation
    K_o3d = o3d_intrinsics(K)
    
    # Creating Open3D RGBD images (and keeping colors 
    # for colored ICP later)
    curr_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(curr_rgb),
        o3d.geometry.Image(curr_depth.astype(np.float32)),
        depth_scale=1.0, depth_trunc=max_depth,
        convert_rgb_to_intensity=False
    )
    
    prev_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(prev_rgb),
        o3d.geometry.Image(prev_depth.astype(np.float32)),
        depth_scale=1.0, depth_trunc=max_depth,
        convert_rgb_to_intensity=False
    )

    # Converting to point-clouds for alignment
    pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, K_o3d)
    pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, K_o3d)

    # Downsampling to make alignment smoother 
    # (like a low-pass filter) and for faster ICP
    voxel_size = 0.1
    pcd_curr = pcd_curr.voxel_down_sample(voxel_size=voxel_size)
    pcd_prev = pcd_prev.voxel_down_sample(voxel_size=voxel_size)

    # Estimating (local) normals (needed for point-to-plane ICP)
    pcd_curr.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_prev.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Using previous pose as initial guess to make 
    # optimization faster (visual odometry is 
    # incremental, so identity is usually a fine guess 
    # for the relative motion between two adjacent 
    # frames)
    trans_init = np.eye(4)
    
    # Using colored ICP because it uses colors alongside 
    # (point-to-plane) geometry, useful for our Minecraft
    # scenario where texture is very much repeated
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        pcd_curr, pcd_prev,
        max_correspondence_distance=voxel_size*2,
        init=trans_init,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        )
    )

    # The pose is the inverse of the extrinsics matrix.
    # Normally we have X_c = [R|t] @ X_w, with 
    # [R|t] = T_cw the extrinsics, but the pose (T_wc) 
    # gives us X_w = T_wc @ X_c.
    # Now let's define X_c0 a point in the previous 
    # camera frame (target) and X_c1 the same point in 
    # the current camera frame (source). We start with 
    # the pose from the previous camera frame, so we
    # know the mapping X_w = T_wc0 @ X_c0. With the ICP
    # result, we get a matrix M that maps source points to
    # target points, so we have X_c0 = M @ X_c1. Now
    # substituting the latter into the former, we get to
    # X_w = T_wc0 @ (M @ X_c1) = (T_wc0 @ M) @ X_c1 and
    # we have just computed the pose for the current frame
    # T_wc1 = T_wc0 @ M
    relative_motion = result_icp.transformation
    current_camera_pose = prev_camera_pose @ relative_motion
    
    return current_camera_pose, relative_motion


# Visualizing tracking to make sure we computed
# it correctly
def visualize_tracking(K, curr_rgb, curr_depth, prev_rgb, prev_depth, \
                       relative_motion, max_depth=20.0):
    # Open3D intrinsics computation
    K_o3d = o3d_intrinsics(K)
    
    # Creating both point-clouds
    prev_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(prev_rgb), \
        o3d.geometry.Image(prev_depth.astype(np.float32)),
        depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False
    )
    curr_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(curr_rgb), \
        o3d.geometry.Image(curr_depth.astype(np.float32)),
        depth_scale=1.0, depth_trunc=max_depth, convert_rgb_to_intensity=False
    )
    
    pcd_prev = o3d.geometry.PointCloud.create_from_rgbd_image(prev_rgbd, K_o3d)
    pcd_curr = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd, K_o3d)

    # Coloring point-clouds distinctively to see 
    # the difference
    pcd_prev.paint_uniform_color([0.5, 0.5, 0.5]) 
    pcd_curr.paint_uniform_color([1.0, 0.0, 0.0])

    # Applying the calculated transform to the 
    # current cloud, so that it (ideally) fits the
    # previous cloud
    # pcd_curr.transform(relative_motion)

    # Flipping Y and Z to match Open3D's coordinate system
    # (OpenCV is Y-Down and Z-Forward while Open3D is 
    # Y-Up, Z-Back)
    R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pcd_prev.rotate(R_flip, center=(0,0,0))
    pcd_curr.rotate(R_flip, center=(0,0,0))

    o3d.visualization.draw_geometries([pcd_prev, pcd_curr], 
                                      window_name="Tracking Verification",
                                      width=800, height=600)


# Saving ply file for plotting the splat
def save_ply(filepath, xyz, scales, rotations, opacities, f_dc):
    # Moving to CPU and converting to numpy
    xyz = xyz.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()
    opacities = opacities.detach().cpu().numpy()
    f_dc = f_dc.detach().cpu().numpy()

    # Creating structured array with 3DGS PLY format:
    # x,y,z, nx,ny,nz, f_dc_0,1,2, opacity, scale_0,1,2, rot_0,1,2,3
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), \
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), \
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), \
             ('f_dc_2', 'f4'), ('opacity', 'f4'), \
             ('scale_0', 'f4'), ('scale_1', 'f4'), \
             ('scale_2', 'f4'), ('rot_0', 'f4'), \
             ('rot_1', 'f4'), ('rot_2', 'f4'), \
             ('rot_3', 'f4')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    
    # Normals are unused by renderer but required by spec
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    # Applying inverse sigmoid to opacity and log to scales
    elements['opacity'] = _inverse_sigmoid(opacities).flatten()
    elements['scale_0'] = np.log(scales[:, 0])
    elements['scale_1'] = np.log(scales[:, 1])
    elements['scale_2'] = np.log(scales[:, 2])
    
    elements['rot_0'] = rotations[:, 0]
    elements['rot_1'] = rotations[:, 1]
    elements['rot_2'] = rotations[:, 2]
    elements['rot_3'] = rotations[:, 3]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filepath)


# 3D Gaussian Splatting world point-cloud visualizer
def visualize_gaussians_point_cloud(xyz_tensor, color_tensor):
    # We have to review that and maybe change it to GPU at
    # some point
    points = xyz_tensor.detach().cpu().numpy()
    colors = color_tensor.detach().cpu().numpy()

    # Creating point-cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Flipping Y and Z to match Open3D's coordinate system
    # (OpenCV is Y-Down and Z-Forward while Open3D is 
    # Y-Up, Z-Back)
    R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pcd.rotate(R_flip, center=(0,0,0))
    
    o3d.visualization.draw_geometries([pcd], 
                                      window_name="Minecraft SLAM Visualizer",
                                      width=960,
                                      height=720)


# Testing Gaussian Splatting Visual SLAM pipeline
if __name__ == '__main__':
    try:
        debug = ast.literal_eval(sys.argv[1])
        start_frame = int(sys.argv[2])
        max_depth = float(sys.argv[3])
    except:
        debug = True
        start_frame = 0
        max_depth = 20.0

    # Creating intrinsics K
    poses_df = pd.read_csv('../data/small_corridor/poses.txt')
    poses_df.columns = ['X', 'Y', 'Z', 'yaw', 'pitch', 'image_path']
    image_width, image_height = 640, 360
    horizontal_fov_deg = 70.0
    K = camera_intrinsics(image_width, image_height, \
                          horizontal_fov_deg)
    
    # Reading and processing data
    img_path_0 = poses_df["image_path"].iloc[start_frame]
    image_0 = cv2.imread(img_path_0)
    image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
    
    # When the character starts, he's in Z=0 and, for the
    # small_corridor experiment, the wall in front of him 
    # is 10 meters aways, so convert from normalized range 
    # to proper meters range
    depth_0 = cv2.imread(img_path_0.replace("rgb", "d").replace("ppm", "pgm"), -1)
    depth_0 = depth_0/(255/10)
    
    # Initializing gaussians for frame and visualizing them
    device = torch.device("cuda" if \
                          torch.cuda.is_available() \
                          else "cpu")
    xyz, colors, scales, log_scales, rotations, opacities, \
        logit_opacities, f_dc = init_gaussians(K, image_0, depth_0, \
                                               max_depth, device)
    
    if debug:
        os.makedirs("../data/slam_output/", exist_ok=True)
        save_ply("../data/slam_output/small_corridor.ply", xyz, scales, \
                 rotations, opacities, f_dc)
        visualize_gaussians_point_cloud(xyz, colors)

    # Estimating motion
    img_path_1 = poses_df["image_path"].iloc[start_frame+2]
    image_1 = cv2.imread(img_path_1)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    depth_1 = cv2.imread(img_path_1.replace("rgb", "d").replace("ppm", "pgm"), -1)
    depth_1 = depth_1/(255/10)

    # Pose of frame 0 is identity because we understand 
    # it as the world origin, so R|t + homogenous 
    # coordinate at the end, with identity rotation and
    # null translation
    pose_0 = np.eye(4) 
    
    pose_1, relative_motion = tracking(K, image_1, \
                                         depth_1, image_0, \
                                         depth_0, max_depth, \
                                         pose_0)

    # Visualizing tracking
    if debug:
        visualize_tracking(K, image_1, depth_1, image_0, \
                           depth_0, relative_motion, max_depth)
        
    # Visualizing mapping