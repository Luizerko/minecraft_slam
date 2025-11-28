import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys

from utils import (
    camera_matrices_from_pose,
    image_feature_extractor,
    match_features,
    filter_matches,
    triangulate_matches,
    build_tracks_from_pairs,
    triangulate_track_multi_view
)


# Samlping colors for 3D points from the images
def sample_colors(img_rgb, keypoints):
    h, w, _ = img_rgb.shape
    colors = []
    for kp in keypoints:
        u, v = kp.pt
        x = np.clip(int(round(u)), 0, w-1)
        y = np.clip(int(round(v)), 0, h-1)
        colors.append(img_rgb[y, x])
    return np.array(colors, dtype=np.uint8)

# Running 3D estimation for a pair of poses/images
def process_pair(row1, row2, descriptor="SIFT", horizontal_fov_deg=70.0, image_width=640, image_height=360):
    # Camera matrices from ground‑truth poses
    K, _, _, P1 = camera_matrices_from_pose([row1.X, row1.Y, row1.Z], row1.yaw, \
                                              row1.pitch, 0.0, image_width, \
                                              image_height, horizontal_fov_deg)
    _, _, _, P2 = camera_matrices_from_pose([row2.X, row2.Y, row2.Z], row2.yaw, \
                                              row2.pitch, 0.0, image_width, \
                                              image_height, horizontal_fov_deg)

    # Skipping rotation-only frames
    C1 = np.array([row1.X, row1.Y, row1.Z])
    C2 = np.array([row2.X, row2.Y, row2.Z])
    baseline = np.linalg.norm(C2 - C1)
    if baseline <= 1e-3:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    # Features
    kps1, desc1 = image_feature_extractor(row1.image_path, descriptor, show=False)
    kps2, desc2 = image_feature_extractor(row2.image_path, descriptor, show=False)
    if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    # Matching + RANSAC inlier filtering
    matches = match_features(kps1, desc1, kps2, desc2, descriptor)
    inlier_matches, _, _ = filter_matches(kps1, kps2, matches, K)
    if len(inlier_matches) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    # Triangulation
    points_3d = triangulate_matches(P1, P2, kps1, kps2, inlier_matches)

    # Color sampling from first image
    img_bgr1 = cv2.imread(row1.image_path)
    img_rgb1 = cv2.cvtColor(img_bgr1, cv2.COLOR_BGR2RGB)
    colors = sample_colors(img_rgb1, [kps1[m.queryIdx] for m in inlier_matches])

    return points_3d, colors


# Running 3D estimation for a window of poses/images
def process_window(poses_df, start_idx, descriptor, horizontal_fov_deg, \
                   image_width, image_height, min_track_length=3):
    frame_indices = [start_idx + i for i in range(min_track_length)]
    
    # Not enough frames remaining
    if frame_indices[-1] >= len(poses_df):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    # Getting camera matrices
    Ps = []
    cameras_centers = []
    Ks = []
    for idx in frame_indices:
        row = poses_df.iloc[idx]
        x, y, z = row['X'], row['Y'], row['Z']
        yaw, pitch = row['yaw'], row['pitch']
        roll = 0.0

        K, _, _, P = camera_matrices_from_pose([x, y, z], yaw, pitch, roll, \
                                               image_width, image_height, \
                                               horizontal_fov_deg)
        Ps.append(P)
        Ks.append(K)
        cameras_centers.append(np.array([x, y, z]))
    K = Ks[0]

    # Extracting features and loading images
    keypoints_list = []
    descriptors_list = []
    images_rgb = []
    for idx in frame_indices:
        img_path = poses_df['image_path'].iloc[idx]
        kps, desc = image_feature_extractor(img_path, descriptor, False)
        keypoints_list.append(kps)
        descriptors_list.append(desc)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images_rgb.append(img_rgb)

    # Creating pairwise matches and running RANSAC filtering for pairs of 
    # consecutive frames. Also filtering out consecutive frames with 
    # rotation-only realtionship
    pair_inlier_matches = []
    for i in range(min_track_length-1):
        kps1, desc1 = keypoints_list[i], descriptors_list[i]
        kps2, desc2 = keypoints_list[i + 1], descriptors_list[i + 1]

        C1, C2 = cameras_centers[i], cameras_centers[i + 1]
        baseline = np.linalg.norm(C2 - C1)
        if baseline <= 1e-3 or desc1 is None or desc2 is None:
            pair_inlier_matches.append([])
            continue

        matches = match_features(kps1, desc1, kps2, desc2, descriptor)
        inlier_matches, _, _ = filter_matches(kps1, kps2, matches, K)
        pair_inlier_matches.append(inlier_matches)

    # Building multi-view tracks with at least min_track_length views
    tracks = build_tracks_from_pairs(pair_inlier_matches, min_track_length-1)
    if len(tracks) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)

    # Multi-view triangulation and color addition
    points_3d = []
    colors = []
    for tr in tracks:
        X = triangulate_track_multi_view(Ps, keypoints_list, tr)
        points_3d.append(X)

        # Choosing color from earliest frame in which this track appears
        f_idx = sorted(tr.keys())[0]
        kp_idx = tr[f_idx]
        u, v = keypoints_list[f_idx][kp_idx].pt
        u = np.clip(int(round(u)), 0, image_width-1)
        v = np.clip(int(round(v)), 0, image_height-1)
        img = images_rgb[f_idx]
        col = img[v, u, :]
        colors.append(col)

    points_3d = np.array(points_3d)
    colors = np.array(colors, dtype=np.uint8)

    return points_3d, colors


def main(experiment, descriptor, window_size):
    # poses_df = pd.read_csv(f'../data/{experiment}/poses.txt')
    # poses_df.columns = ['X', 'Y', 'Z', 'yaw', 'pitch', 'image_path']
    # horizontal_fov_deg = 70.0
    # image_width, image_height = 640, 360

    # all_points = []
    # all_colors = []

    # for i in range(len(poses_df) - 1):
    #     row1 = poses_df.iloc[i]
    #     row2 = poses_df.iloc[i + 1]
    #     pts, cols = process_pair(row1, row2, descriptor, horizontal_fov_deg, \
    #                              image_width, image_height)
    #     if pts.size == 0:
    #         print(f"No 3D estimations from pair {i}-{i+1}")
    #         continue
        
    #     all_points.append(pts)
    #     all_colors.append(cols)

    # if len(all_points) == 0:
    #     print("No 3D points reconstructed.")
    #     return

    # points = np.vstack(all_points)
    # colors = np.vstack(all_colors)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors/255.0, s=1)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.set_title("Accumulated 3D Point Cloud")
    # plt.show()

    poses_df = pd.read_csv(f'../data/{experiment}/poses.txt')
    poses_df.columns = ['X', 'Y', 'Z', 'yaw', 'pitch', 'image_path']
    horizontal_fov_deg = 70.0
    image_width, image_height = 640, 360

    all_points = []
    all_colors = []

    for start_idx in range(len(poses_df) - window_size + 1):
        pts, cols = process_window(poses_df, start_idx, descriptor, \
                                   horizontal_fov_deg, image_width, \
                                   image_height, window_size)

        if pts.size == 0:
            print(f"No 3D estimations from window {start_idx}-{start_idx+window_size-1}")
            continue

        all_points.append(pts)
        all_colors.append(cols)

    if len(all_points) == 0:
        print("No 3D points reconstructed")
        return

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    # Filtering out extreme points
    center = np.median(points, axis=0)
    dists = np.linalg.norm(points-center, axis=1)
    thresh = np.percentile(dists, 95)
    mask = dists < thresh
    points = points[mask]
    colors = colors[mask]

    color_strings = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colors]

    fig = go.Figure(data=[go.Scatter3d(x=-points[:, 0], y=points[:, 2], \
                                       z=-points[:, 1], mode='markers', \
                                       marker=dict(size=2, \
                                                   color=color_strings, \
                                                    opacity=0.8))])

    fig.update_layout(title="Accumulated 3D Point Cloud", \
                      scene=dict(xaxis_title="X", yaxis_title="Y",\
                                 zaxis_title="Z", aspectmode='data'),\
                                 margin=dict(l=0, r=0, b=0, t=30))

    fig.show()

# Running the entire Structure From Motion pipeline for a chosen minecraft
# experiment
if __name__ == "__main__":
    try:
        experiment = sys.argv[1]
        descriptor = sys.argv[2]
        window_size = int(sys.argv[3])
    except:
        experiment = "small_corridor"
        descriptor = "SIFT"
        window_size = 3

    main(experiment, descriptor, window_size)