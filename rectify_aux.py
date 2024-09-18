import os
import json
import cv2
import numpy as np
from pathlib import Path
import csv

# Paths
image_save_folder = Path('multisense/somerset/')
aux_folder = image_save_folder / 'aux'
calibration_file = image_save_folder / 'calibration_data/aux.json'
aux_rectified_folder = image_save_folder / 'aux_rectified'
aux_rectified_folder.mkdir(exist_ok=True)

# Load the aux camera calibration data from JSON file
with open(calibration_file, 'r') as f:
    calibration_data = json.load(f)

# Extract the calibration matrices
K = np.array(calibration_data['K']).reshape(3, 3)  # Reshape intrinsic matrix to 3x3
D = np.array(calibration_data['D'])  # Distortion coefficients (1D array)
R = np.array(calibration_data['R']).reshape(3, 3)  # Rectification matrix (3x3)
P = np.array(calibration_data['P']).reshape(3, 4)  # Projection matrix (3x4)

# Ensure the image size is correctly formatted (width, height)
image_size = (int(calibration_data['width']), int(calibration_data['height']))

# Debug prints
print("K (Intrinsic Camera Matrix):", K)
print("D (Distortion Coefficients):", D)
print("R (Rectification Matrix):", R)
print("P (Projection Matrix):", P)
print("Image Size:", image_size)

# Iterate over each image in the aux folder
for img_filename in os.listdir(aux_folder):
    if img_filename.endswith('.png'):
        img_path = aux_folder / img_filename

        # Load the image
        img = cv2.imread(str(img_path))

        # Ensure the image size matches the expected size from calibration
        if img.shape[1] != image_size[0] or img.shape[0] != image_size[1]:
            print(f"Skipping {img_filename}: Image size {img.shape} doesn't match calibration size {image_size}.")
            continue

        # Compute the rectification (undistortion and stereo alignment) transformation map
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P, image_size, cv2.CV_32FC1)

        # Apply rectification transformation to the image
        rectified_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Save the rectified image in the aux_rectified folder
        rectified_img_path = aux_rectified_folder / img_filename
        cv2.imwrite(str(rectified_img_path), rectified_img)

        print(f"Rectified and saved: {rectified_img_path}")

print(f"Stereo rectification completed. Rectified images saved in: {aux_rectified_folder}")
