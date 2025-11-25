import ast
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


# Building camera intrinsics K, extrinsics [R|t] and full projection P = K [R|t]
def camera_matrices_from_pose(position_xyz, yaw_deg, pitch_deg, roll_deg=0.0, image_width=640, image_height=360, horizontal_fov_deg=70.0):
    # Building intrinsics matrix (camera coords to pixel coords)
    w, h = image_width, image_height

    # Computing horizontal focal length from hFoV
    hFoV = np.deg2rad(horizontal_fov_deg)
    fx = w / (2.0 * np.tan(hFoV / 2.0))

    # Computing vertical FoV from aspect ratio and then vertical focal length
    vFoV = 2.0 * np.arctan((h / w) * np.tan(hFoV / 2.0))
    fy = h / (2.0 * np.tan(vFoV / 2.0))

    # Computing principal point (where the camera's optical axis hits the image)
    cx, cy = w / 2.0, h / 2.0

    # Finally building the matrix
    K = np.array([[fx, 0.0, cx],\
                  [0.0, fy, cy],\
                  [0.0, 0.0, 1.0]])

    # Building rotation matrix (world coords to camera coords) in ZYX order
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0.0],\
                   [ np.sin(yaw), np.cos(yaw), 0.0],\
                   [ 0.0, 0.0, 1.0]])

    Ry = np.array([[np.cos(pitch), 0.0, np.sin(pitch)],\
                   [ 0.0, 1.0, 0.0],\
                   [-np.sin(pitch), 0.0, np.cos(pitch)]])

    Rx = np.array([[1.0, 0.0, 0.0],\
                   [0.0, np.cos(roll), -np.sin(roll)],\
                   [0.0, np.sin(roll), np.cos(roll)]])
    
    R = Rz @ Ry @ Rx

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
        plt.title("SIFT Descriptors")
        plt.imshow(img)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return kps, desc

# Matching features between consecutive frames
def match_features(kps1, desc1, kps2, desc2, descriptor="SIFT", \
                   ratio_thresh=0.8, k=10):
    # KNN matching (k neighbors for each descriptor in desc1)
    if descriptor == "SIFT":
        norm_type = cv2.NORM_L2
    elif descriptor == "ORB":
        norm_type = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    knn_matches = bf.knnMatch(desc1, desc2, k=k)
    good_matches = []

    # Classic Lowe ratio test with
    for neighbors in knn_matches:
        if len(neighbors) < 2:
            continue

        # Best candidate neighbor
        best = neighbors[0]
        best_kp2 = kps2[best.trainIdx]
        best_class = best_kp2.class_id

        # Finding the nearest neighbor from a different class_id (if possible)
        second = None
        for cand in neighbors[1:]:
            cand_kp2 = kps2[cand.trainIdx]
            cand_class = cand_kp2.class_id

            # If class_id is -1 (default) for best, we can't really separate
            # classes, so any other neighbor can serve as "second"
            if best_class == -1 or cand_class != best_class:
                second = cand
                break

        # If we couldn't find a second from a different class (and 
        # best_class != -1), fall back to the second-best overall
        if second is None:
            if len(neighbors) >= 2:
                second = neighbors[1]
            else:
                continue

        # Accepting best if it's sufficiently better than second
        if best.distance < ratio_thresh * second.distance:
            good_matches.append(best)

    return good_matches

# Testing 3D point cloud generation pipeline
if __name__ == '__main__':
    try:
        debug = ast.literal_eval(sys.argv[1])
    except:
        debug = False

    poses_df = pd.read_csv('../data/small_corridor/poses.txt')
    poses_df.columns = ['X', 'Y', 'Z', 'yaw', 'pitch', 'image_path']
    if debug:
        print("First few poses:")
        print(poses_df.head())

    # Getting matrices from poses
    x, y, z = poses_df['X'].iloc[50], poses_df['Y'].iloc[50], \
              poses_df['Z'].iloc[50]
    yaw, pitch, roll = poses_df['yaw'].iloc[50], poses_df['pitch'].iloc[50], \
                       0.0
    image_width, image_height = 640, 360
    horizontal_fov_deg = 70.0
    
    K, R, t, P = camera_matrices_from_pose([x, y, z], yaw, pitch, roll, \
                                           image_width, image_height, horizontal_fov_deg)
    if debug:
        # Sanity checking camera matrices
        print("\n=== Camera Matrices ===")
        print(f'Intrinsics K:\n{K}\n')
        print(f'Rotation R:\n{R}\n')
        print(f'Translation t:\n{t}\n')
        print(f'Projection P:\n{P}\n')

        # Sanity checking math
        print("\n=== Sanity Checks ===")
        print(f'R^T @ R == I:\n{R.T @ R}\n')
        print(f'det(R) == 1: {np.linalg.det(R)}\n')
        print(f'Original camera position: {[x.item(), y.item(), z.item()]}\nCamera center from extrinsics: {-R.T @ t}')

    # Extracting (and eventually ploting for debugging) image features
    descriptor = "SIFT"
    img_path1 = poses_df['image_path'].iloc[50]
    img_path2 = poses_df['image_path'].iloc[51]
    kps1, desc1 = image_feature_extractor(img_path1, descriptor, debug)
    kps2, desc2 = image_feature_extractor(img_path2, descriptor, False)
    if debug:
        print(f"\nNumber of {descriptor} keypoints in image 1: {len(kps1)}")
        print(f"\nNumber of {descriptor} keypoints in image 2: {len(kps2)}")
    
    # Matching image features
    matches = match_features(kps1, desc1, kps2, desc2, "SIFT")
    if debug:
        img_bgr1 = cv2.imread(img_path1)
        img_rgb1 = cv2.cvtColor(img_bgr1, cv2.COLOR_BGR2RGB)
        img_bgr2 = cv2.imread(img_path2)
        img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)

        matched_img = cv2.drawMatches(
            img_rgb1, kps1,
            img_rgb2, kps2,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        plt.figure(figsize=(15,8))
        plt.imshow(matched_img)
        plt.axis('off')
        plt.show()