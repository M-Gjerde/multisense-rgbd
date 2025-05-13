import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
# Root folder where extraction saved images and calibration
root_dir = Path('logqs_dataset/desert')
# Calibration folder (contains back.json, front.json, left.json, right.json)
calib_folder = root_dir / 'calibration_data'
# Cameras to process
cameras = ['back', 'front', 'left', 'right']

# === Helper to load calibration ===
def load_calibration(calib_file: Path):
    with open(calib_file, 'r') as f:
        data = json.load(f)
    K = np.array(data['K']).reshape(3, 3)
    D = np.array(data['D'])
    R = np.array(data['R']).reshape(3, 3)
    P = np.array(data['P']).reshape(3, 4)
    size = (int(data['width']), int(data['height']))
    return K, D, R, P, size

# === Main rectification loop ===
for cam in cameras:
    # Paths
    img_folder = root_dir / cam
    rect_folder = root_dir / f"{cam}_rectified"
    rect_folder.mkdir(exist_ok=True)
    calib_file = calib_folder / f"{cam}.json"

    if not calib_file.exists():
        print(f"Calibration file missing for '{cam}': {calib_file}")
        continue
    if not img_folder.exists():
        print(f"Image folder missing for '{cam}': {img_folder}")
        continue

    # Load calibration
    K, D, R, P, image_size = load_calibration(calib_file)
    print(f"Loaded calibration for '{cam}': image size {image_size}")

    # Precompute undistort/rectify map
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P, image_size, cv2.CV_32FC1)

    # Process all PNG images
    for img_file in tqdm(sorted(img_folder.glob('*.png')), desc=f"Rectifying {cam}"):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Failed loading {img_file}, skipping.")
            continue
        # Check and resize if needed
        if (img.shape[1], img.shape[0]) != image_size:
            print(f"Skipping {img_file.name}: size {img.shape[1], img.shape[0]} != {image_size}")
            continue
        # Apply rectification
        rectified = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        # Save
        out_path = rect_folder / img_file.name
        cv2.imwrite(str(out_path), rectified)
    print(f"Finished rectification for '{cam}', saved to {rect_folder}")

print("All cameras rectified.")
