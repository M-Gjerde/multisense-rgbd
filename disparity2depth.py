import argparse
import os
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return image


def disparity_to_depth(disparity_image, focal_length, baseline):
    """Convert disparity image to depth map."""
    mask = disparity_image < 5  # Mask where disparity is less than 5 (invalid disparity)
    disparity_image[mask] = 0.1  # Avoid division by zero by setting small value
    depth_map = (focal_length * baseline) / disparity_image
    depth_map[mask] = 0  # Reset masked depth values to 0
    return depth_map


def read_calibration_file(calibration_file):
    """Read calibration parameters from a YAML file."""
    fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

    # Extract P1 and P2 matrices
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()

    fs.release()

    # Calculate focal length and baseline
    focal_length = P1[0, 0]  # Assuming the same focal length for both cameras
    tx = P2[0, 3]
    baseline = abs(-tx / focal_length)

    return focal_length, baseline


def process_disparity_folder(disparity_folder, focal_length, baseline, save_format, dataset_source, depth_unit_scale=10000):
    """Process a folder of disparity images and save depth images."""
    # Create output directory for depth images
    depth_folder = os.path.join(Path(disparity_folder).parent, "depth")
    os.makedirs(depth_folder, exist_ok=True)
    if dataset_source == "logqs":
        disparity_folder = disparity_folder
    else:
        disparity_folder = os.path.join(disparity_folder, "tiff")

    # Process each TIFF image in the disparity folder
    for disparity_file in tqdm(sorted(os.listdir(disparity_folder)), desc="Processing disparity images"):
        if dataset_source == "logqs":
            disparity_path = (os.path.join(disparity_folder, disparity_file))
            disparity_image = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 16
        else:
            disparity_image = load_image(os.path.join(disparity_folder, disparity_file))

        # Convert disparity to depth
        depth_image = disparity_to_depth(disparity_image, focal_length, baseline)
        # Scale depth to the specified units (millimeters by default)
        depth_image *= depth_unit_scale
        depth_folder_ext = save_format.replace(".", "")
        # Save the depth image in the specified format
        depth_output_path = os.path.join(depth_folder, depth_folder_ext, os.path.splitext(disparity_file)[0] + f".{save_format}")
        os.makedirs(os.path.join(depth_folder, depth_folder_ext), exist_ok=True)


        if save_format == "npy":
            np.save(depth_output_path, depth_image.astype(np.float32))
        else:
            cv2.imwrite(depth_output_path, depth_image.astype(np.float32))
        print(f"Saved depth image to: {depth_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert disparity images to depth maps.")
    parser.add_argument("--disparity_folder", type=str, required=True,
                        help="Folder containing disparity images (.tiff).")
    parser.add_argument("--calibration_file", type=str, required=True, help="Path to the YAML calibration file.")
    parser.add_argument("--save_format", type=str, choices=["png", "npy", "tiff"], default="png",help="Output format for depth images (default: png).")
    parser.add_argument("--source", type=str, choices=["logqs", "viewer"], default="viewer",help="Source from viewer or logqs?")
    args = parser.parse_args()

    # Read focal length and baseline from the calibration file
    focal_length, baseline = read_calibration_file(args.calibration_file)

    # Process disparity images
    process_disparity_folder(args.disparity_folder, focal_length, baseline, args.save_format, source.source)


if __name__ == "__main__":
    main()
