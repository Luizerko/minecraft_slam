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
    build_tracks_from_pairs,
    triangulate_track_multi_view
)


# Running 3D estimation for global windows of poses/images
def process_windows(poses_df, descriptor, horizontal_fov_deg, \
                   image_width, image_height, min_track_length=3):
    # Global pre-calculation of features to avoid repetition
    indices = []
    Ps = []
    Ks = []
    keypoints_list = []
    descriptors_list = []
    images_rgb = []
    for i, row in poses_df.iterrows():
        K, _, _, P = camera_matrices_from_pose([row.X, row.Y, row.Z], \
                                               row.yaw, row.pitch, \
                                               image_width, image_height, \
                                               horizontal_fov_deg)
        
        kps, desc = image_feature_extractor(row.image_path, descriptor)
        if desc is None:
            continue

        img = cv2.cvtColor(cv2.imread(row.image_path), cv2.COLOR_BGR2RGB)

        indices.append(i)
        Ps.append(P)
        Ks.append(K)
        keypoints_list.append(kps)
        descriptors_list.append(desc)
        images_rgb.append(img)

    # Creating pair-sequential matching and filtering
    pair_matches = []
    for i, _ in enumerate(indices[:-1]):
        desc1, desc2 = descriptors_list[i], descriptors_list[i+1]
        kps1, kps2 = keypoints_list[i], keypoints_list[i+1]
        
        # Filtering out consecutive frames with rotation-only realtionship
        C1 = np.array([poses_df.iloc[indices[i]].X, \
                      poses_df.iloc[indices[i]].Y, \
                      poses_df.iloc[indices[i]].Z])
        C2 = np.array([poses_df.iloc[indices[i+1]].X, \
                      poses_df.iloc[indices[i+1]].Y, \
                      poses_df.iloc[indices[i+1]].Z])
        if np.linalg.norm(C2 - C1) < 1e-3:
            pair_matches.append([])
            continue

        matches = match_features(desc1, desc2, descriptor)
        inliers, _, _ = filter_matches(kps1, kps2, matches, Ks[0])
        pair_matches.append(inliers)

    # Building global tracking (windows)
    tracks = build_tracks_from_pairs(pair_matches, min_track_length)

    # Global multi-view triangulation
    points_3d = []
    colors = []
    for tr in tracks:
        X = triangulate_track_multi_view(Ps, keypoints_list, tr)
        
        # Filtering negative Z points (behind the camera)
        # mask = points_3d[:, 2] < 0
        # points_3d = points_3d[mask]
        points_3d.append(X)
        
        # Color Sampling
        f_idx = list(tr.keys())[0]
        kp_idx = tr[f_idx]
        u, v = keypoints_list[f_idx][kp_idx].pt
        u = np.clip(int(u), 0, image_width-1)
        v = np.clip(int(v), 0, image_height-1)
        colors.append(images_rgb[f_idx][v, u])

    points_3d = np.array(points_3d)
    colors = np.array(colors)

    print(points_3d.shape)

    # Filtering out extreme points
    center = np.median(points_3d, axis=0)
    dists = np.linalg.norm(points_3d-center, axis=1)
    thresh = np.percentile(dists, 95)
    mask = dists < thresh
    points_3d = points_3d[mask]
    colors = colors[mask]

    return points_3d, colors

def main(experiment, descriptor, window_size):
    poses_df = pd.read_csv(f'../data/{experiment}/poses.txt')
    poses_df.columns = ['X', 'Y', 'Z', 'yaw', 'pitch', 'image_path']
    horizontal_fov_deg = 70.0
    image_width, image_height = 640, 360

    points, colors = process_windows(poses_df, descriptor, \
                                     horizontal_fov_deg, image_width, \
                                     image_height, window_size)

    color_strings = [f'rgb({c[0]}, {c[1]}, {c[2]})' for c in colors]

    fig = go.Figure(data=[go.Scatter3d(x=points[:, 0], y=-points[:, 2], \
                                       z=points[:, 1], mode='markers', \
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
        window_size = 2

    main(experiment, descriptor, window_size)