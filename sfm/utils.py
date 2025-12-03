import ast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


# Building camera intrinsics K, extrinsics [R|t] and full projection P = K [R|t]
def camera_matrices_from_pose(position_xyz, yaw_deg, pitch_deg, image_width=640, image_height=360, horizontal_fov_deg=70.0):
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
    K = np.array([[fx, 0.0, cx],\
                  [0.0, fy, cy],\
                  [0.0, 0.0, 1.0]])

    # Building rotation matrix by converting Minecraft Angles to an OpenCV 
    # Camera Basis. Minecraft Inputs:
    #   Yaw:   0 -> +Z, -90 -> +X, 90 -> -X, 180 -> -Z
    #   Pitch: -90 -> +Y, 90 -> -Y
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    
    # Calculating the forward vector (OpenCV +Z), the direction the camera 
    # is looking in World Coordinates, derived from spherical coordinates
    f_x = -np.sin(yaw) * np.cos(pitch)
    f_y = -np.sin(pitch)
    f_z = np.cos(yaw) * np.cos(pitch)
    forward = np.array([f_x, f_y, f_z])
    forward = forward / np.linalg.norm(forward)

    # Calculating the right vector (OpenCV +X). Minecraft cameras don't roll,
    # so right is always perpendicular to Y-axis
    r_x = -np.cos(yaw) 
    r_y = 0.0
    r_z = -np.sin(yaw)
    right = np.array([r_x, r_y, r_z])
    right = right / np.linalg.norm(right)

    # Calculating the down vector (OpenCV +Y) as orthogonal to forward and 
    # right. In OpenCV: right (X) cross forward (Z) = up (-Y), but we want
    # down (+Y), so we do forward cross right.
    down = np.cross(forward, right)

    # Building "rotation matrix" (World -> Camera) using the vectors as our 
    # new basis
    R = np.array([right, down, forward])

    # Building translation matrix
    C = position_xyz
    t = -R @ C

    # Building projection matrix (mixing it all to directly project from world coords to image pixel coords)
    Rt = np.hstack([R, t.reshape(3, 1)])
    P = K @ Rt

    return K, R, t, P


# Extracting image features using SIFT or ORB
def image_feature_extractor(img_path, descriptor='SIFT', show=False):
    img_bgr = cv2.imread(img_path)
    
    # Convert BGR to grayscale for features
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Extracting keypoitns and descriptors
    if descriptor == 'SIFT':
        sift = cv2.SIFT_create()
        kps, desc = sift.detectAndCompute(img_gray, None)
    
    elif descriptor == 'ORB':
        orb = cv2.ORB_create()
        kps, desc = orb.detectAndCompute(img_gray, None)
    
    # Plotting keypoints on copies of the original image and showing scale 
    # and orientation
    if show:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.drawKeypoints(img_rgb, kps, None, \
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Grayscale Image")
        plt.imshow(img_gray, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"{descriptor} Descriptors")
        plt.imshow(img)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return kps, desc


# Matching features between consecutive frames
def match_features(desc1, desc2, descriptor="SIFT", \
                   ratio_thresh=0.45, k=2):
    # KNN matching (k neighbors for each descriptor in desc1)
    if descriptor == "SIFT":
        norm_type = cv2.NORM_L2
    elif descriptor == "ORB":
        norm_type = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=k)
    good_matches = []

    # Classic Lowe ratio test
    for neighbors in knn_matches:
        if len(neighbors) < 2:
            continue

        # Keeping m only if it's much better than n
        m, n = neighbors[0], neighbors[1]
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return good_matches


# Filtering match outliers based on RANSAC and the essential matrix (since
# we have camera intrinsics)
def filter_matches(kps1, kps2, matches, K, ransac_thresh=1.0, \
                  ransac_prob=0.999):
    # Not enough points to estimate E
    if len(matches) < 5:
        return [], None, None

    # Estimating Essential matrix with RANSAC
    pts1 = np.array([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kps2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, \
                                   method=cv2.RANSAC, prob=ransac_prob, \
                                   threshold=ransac_thresh)

    # Case in which RANSAC failed to find a valid model
    if E is None or mask is None:
        return [], None, None

    # Keeping only inlier matches
    mask = mask.ravel().astype(bool)
    inlier_matches = [m for m, inlier in zip(matches, mask) if inlier]

    return inlier_matches, E, mask


# Triangulating (pair of) matches to find 3D points
def triangulate_matches(P1, P2, kps1, kps2, matches):
    # Collect matched 2D points in pixel coordinates
    pts1 = np.array([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kps2[m.trainIdx].pt for m in matches])
    pts1, pts2 = pts1.T, pts2.T

    # Shooting rays to triangulate points and removing homogenous coordinates 
    # to get to 3D coordinates
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d


# Building tracking dictionaries for each keypoint so we know in which
# frames to find any object
def build_tracks_from_pairs(pair_matches, min_length=3):
    if len(pair_matches) == 0:
        return []

    tracks = []
    kp_to_track = {}
    for i, matches in enumerate(pair_matches):
        frame_idx_curr = i
        frame_idx_next = i+1

        for m in matches:
            kp_curr = m.queryIdx
            kp_next = m.trainIdx

            # Checking if the feature in the current frame is already part 
            # of a track
            key = (frame_idx_curr, kp_curr)
            if key in kp_to_track:
                # We found an existing track ending here, so we add the 
                # next frame
                track = kp_to_track[key]
                track[frame_idx_next] = kp_next
                
                # Updating dictionary
                kp_to_track[(frame_idx_next, kp_next)] = track
                del kp_to_track[key]
            
            else:
                # Starting a new track
                new_track = {frame_idx_curr: kp_curr, frame_idx_next: kp_next}
                tracks.append(new_track)
                
                # Updating dictionary
                kp_to_track[(frame_idx_next, kp_next)] = new_track

    # Keeping only tracks with enough observations
    valid_tracks = [tr for tr in tracks if len(tr) >= min_length]
    return valid_tracks


# Triangulating multi-view tracks to find 3D points
def triangulate_track_multi_view(Ps, keypoints_list, track):
    # Manually constructing matrix A
    A_rows = []
    for f_idx, kp_idx in sorted(track.items()):
        P = Ps[f_idx]
        kps = keypoints_list[f_idx]
        u, v = kps[kp_idx].pt

        # Rows for each pixel: 
        # u * P[2,:] - P[0,:]
        # v * P[2,:] - P[1,:]
        A_rows.append(u * P[2, :] - P[0, :])
        A_rows.append(v * P[2, :] - P[1, :])

    A = np.stack(A_rows, axis=0)

    # Solving AX = 0 with SVD and going from homogenous coordinates to 
    # 3D points
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X_h /= X_h[3]
    X = X_h[:3]

    # Checking if the point is in front of the camera (positive Z)
    first_frame = list(track.keys())[0]
    P_test = Ps[first_frame]
    X_homog = np.append(X, 1)
    projected = P_test @ X_homog
    if projected[2] < 0:
        return None

    return X


# Testing 3D point cloud generation pipeline
if __name__ == '__main__':
    try:
        debug = ast.literal_eval(sys.argv[1])
        descriptor = sys.argv[2]
        start_frame = int(sys.argv[3])
        num_frames = int(sys.argv[4])
    except:
        debug = True
        descriptor = "SIFT"
        start_frame = 20
        num_frames = 2

    poses_df = pd.read_csv('../data/small_corridor/poses.txt')
    poses_df.columns = ['X', 'Y', 'Z', 'yaw', 'pitch', 'image_path']
    if debug:
        print("\nFirst few poses:")
        print(poses_df.head())

    # Getting matrices from poses
    image_width, image_height = 640, 360
    horizontal_fov_deg = 70.0

    Ks = []
    Rs = []
    ts = []
    Ps = []
    cameras_centers = []
    frame_indices = [start_frame + i for i in range(num_frames)]
    for idx in frame_indices:
        x = poses_df['X'].iloc[idx]
        y = poses_df['Y'].iloc[idx]
        z = poses_df['Z'].iloc[idx]
        yaw = poses_df['yaw'].iloc[idx]
        pitch = poses_df['pitch'].iloc[idx]

        K, R, t, P = camera_matrices_from_pose([x, y, z], yaw, pitch, \
                                               image_width, image_height, \
                                               horizontal_fov_deg)

        Ks.append(K)
        Rs.append(R)
        ts.append(t)
        Ps.append(P)
        cameras_centers.append(np.array([x, y, z], dtype=float))

    #Because are the Ks are the same
    K = Ks[0]

    if debug:
        # Sanity checking camera matrices
        print(f'\nIntrinsics K:\n{K}\n')
        print(f'Rotation R:\n{Rs[-1]}\n')
        print(f'Translation t:\n{ts[-1]}\n')
        print(f'Projection P:\n{Ps[-1]}\n')

        # Sanity checking math
        print(f'R^T @ R == I:\n{Rs[-1].T @ Rs[-1]}\n')
        print(f'det(R) == 1: {np.linalg.det(Rs[-1])}\n')
        print(f'Original camera position: {[x.item(), y.item(), z.item()]}\nCamera center from extrinsics: {-Rs[-1].T @ ts[-1]}\n')

        # Sanity checking camera movement and orientation
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='3d')
        for i, (C, R) in enumerate(zip(cameras_centers, Rs)):
            ax.scatter(C[0], C[1], C[2], c='r', marker='o')
            
            view_dir = R.T[:, 2] 
            ax.quiver(C[0], C[1], C[2], view_dir[0], view_dir[1], \
                      view_dir[2], length=0.05, color='b')
            
            ax.text(C[0], C[1], C[2], f"Camera {i}")
        
        ax.set_title("Blue arrows should point at where we look during mission")
        plt.show()

    # Extracting (and eventually ploting for debugging) image features
    keypoints_list = []
    descriptors_list = []
    images_rgb = []
    for idx in frame_indices:
        img_path = poses_df['image_path'].iloc[idx]
        show_kps = debug and (idx == frame_indices[-1])
        kps, desc = image_feature_extractor(img_path, descriptor, show_kps)
        keypoints_list.append(kps)
        descriptors_list.append(desc)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images_rgb.append(img_rgb)

        if debug:
            print(f"Frame {idx}: {len(kps)} {descriptor} keypoints")

    # Matching image features
    pair_inlier_matches = []
    for i in range(num_frames - 1):
        idx1 = frame_indices[i]
        idx2 = frame_indices[i + 1]

        print(f"\nPair of frames {idx1} -> {idx2}:")
        kps1, desc1 = keypoints_list[i], descriptors_list[i]
        kps2, desc2 = keypoints_list[i + 1], descriptors_list[i + 1]

        C1, C2 = cameras_centers[i], cameras_centers[i + 1]
        baseline = np.linalg.norm(C2 - C1)
        print(f"Camera centers distance: {baseline:.3f}")
        if baseline <= 1e-3:
            print("Baseline too small, skipping this pair")
            pair_inlier_matches.append([])
            continue

        matches = match_features(desc1, desc2, descriptor)
        print(f"Good matches: {len(matches)}")

        inlier_matches, E, mask = filter_matches(kps1, kps2, matches, K)
        print(f"Inliers after RANSAC: {len(inlier_matches)}\n")

        if debug and len(inlier_matches) > 0 and i == num_frames-2:
            match_img = cv2.drawMatches(images_rgb[i], kps1, \
                                        images_rgb[i+1], kps2, \
                                        inlier_matches, None, \
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(15, 6))
            plt.title(f"Inlier matches frames {idx1} <-> {idx2}")
            plt.imshow(match_img)
            plt.axis('off')
            plt.show()

        pair_inlier_matches.append(inlier_matches)

        # Two-view triangulation
        if num_frames == 2:
            points_3d = triangulate_matches(Ps[0], Ps[1], kps1, kps2, \
                                            inlier_matches)

    # Building multi-view tracks
    min_length = num_frames
    tracks = build_tracks_from_pairs(pair_inlier_matches, \
                                     min_length=min_length-1)
    print(f"\nTotal multi-view tracks: {len(tracks)}")

    # Multi-view triangulation
    if num_frames > 2:
        points_3d = []
        for tr in tracks:
            X = triangulate_track_multi_view(Ps, keypoints_list, tr)
            points_3d.append(X)

    points_3d = np.array(points_3d)
    print("\nTriangulated 3D points shape:", points_3d.shape[0])
    
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Triangulated 3D Points")
        plt.show()