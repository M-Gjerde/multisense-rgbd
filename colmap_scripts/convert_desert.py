#!/usr/bin/env python3
import os
import logging
import shutil
import json
from argparse import ArgumentParser
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------------
# Multi-camera COLMAP sparse reconstruction script with SIMPLE_PINHOLE intrinsics
# Uses only front, left, and right cameras; places all COLMAP I/O under colmap/
# Copies all selected images into colmap/images and per-camera subfolders
# Exports a colored point cloud via model_converter
# ----------------------------------------------------------------------------

def load_calibration(calib_file: Path):
    """
    Reads JSON calibration, returns K, D, image size.
    """
    data = json.load(open(calib_file, 'r'))
    K = np.array(data['K']).reshape(3, 3)
    D = np.array(data['D'])
    width = int(data['width'])
    height = int(data['height'])
    return K, D, (width, height)


def format_camera_params(K, D):
    """
    Format intrinsics for COLMAP SIMPLE_PINHOLE model.
    PARAMETERS: [f, cx, cy]
    fx==fy for rectified images.
    """
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return 'SIMPLE_PINHOLE', ','.join(map(str, (fx, cx, cy)))


def main():
    parser = ArgumentParser("COLMAP multicam sparse with SIMPLE_PINHOLE intrinsics")
    parser.add_argument("--no_gpu", action='store_true', help="Disable GPU for SIFT")
    parser.add_argument("--skip_matching", action='store_true', help="Skip matching/mapping")
    parser.add_argument("--max_images", type=int, default=30,
                        help="Max images per camera to use")
    parser.add_argument("--source_path", "-s", required=True,
                        help="Folder with rectified images and calibration_data/")
    parser.add_argument("--colmap_executable", default="colmap",
                        help="Path to COLMAP binary")
    args = parser.parse_args()

    source = Path(args.source_path)
    colmap_root = source / 'colmap'
    colmap_root.mkdir(exist_ok=True)

    # Use only these cameras
    cam_names = ['front', 'left', 'right']
    calib_folder = source / 'calibration_data'

    # Prepare per-camera subfolders and flat images folder under colmap/
    input_root = colmap_root / 'input'
    images_root = colmap_root / 'images'
    if input_root.exists():
        shutil.rmtree(input_root)
    if images_root.exists():
        shutil.rmtree(images_root)
    images_root.mkdir(parents=True, exist_ok=True)

    for cam in cam_names:
        cam_dir = input_root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        src_folder = source / f"{cam}_rectified"
        if not src_folder.exists():
            logging.warning(f"Missing rectified folder for '{cam}'")
            continue
        imgs = sorted(src_folder.glob('*.png'))[:args.max_images]
        for img in imgs:
            # per-camera folder
            link_cam = cam_dir / img.name
            os.symlink(img.resolve(), link_cam)
            # flat images folder, prefix to avoid name collisions
            link_flat = images_root / img.name
            os.symlink(img.resolve(), link_flat)
        logging.info(f"Linked {len(imgs)} images for camera '{cam}'")

    # Paths for COLMAP outputs under colmap/
    db_path = colmap_root / 'database.db'
    sparse_path = colmap_root / 'sparse'
    sparse_path.mkdir(exist_ok=True)

    # Remove old database
    if db_path.exists():
        db_path.unlink()

    colmap = args.colmap_executable
    use_gpu = 0 if args.no_gpu else 1

    if not args.skip_matching:
        # 1) Feature extraction per camera
        for cam in cam_names:
            cam_dir = input_root / cam
            calib_file = calib_folder / f"{cam}.json"
            if not cam_dir.exists() or not calib_file.exists():
                logging.warning(f"Skipping '{cam}': missing data.")
                continue
            K, D, _ = load_calibration(calib_file)
            cam_model, params = format_camera_params(K, D)
            logging.info(f"Extracting features for '{cam}' using {cam_model}")
            cmd = (
                f"{colmap} feature_extractor "
                f"--database_path {db_path} "
                f"--image_path {cam_dir} "
                f"--ImageReader.single_camera 1 "
                f"--SiftExtraction.use_gpu {use_gpu}"
            )
            if os.system(cmd) != 0:
                logging.error(f"Feature extraction failed for {cam}")
                return

        # 2) Feature matching
        match_cmd = (
            f"{colmap} exhaustive_matcher "
            f"--database_path {db_path} "
            f"--SiftMatching.use_gpu {use_gpu}"
        )
        logging.info("Running exhaustive matching...")
        if os.system(match_cmd) != 0:
            logging.error("Feature matching failed")
            return

        # 3) Sparse mapping (no recursive flag)
        mapper_cmd = (
            f"{colmap} mapper "
            f"--database_path {db_path} "
            f"--image_path {input_root} "
            f"--output_path {sparse_path} "
            f"--Mapper.ba_global_function_tolerance 0.0000001"
        )
        logging.info("Running sparse mapper...")
        if os.system(mapper_cmd) != 0:
            logging.error("Sparse mapping failed")
            return

        # 4) Export colored point cloud
        colored_ply = sparse_path / '0' / 'points_colored.ply'
        convert_cmd = (
            f"{colmap} model_converter "
            f"--input_path {sparse_path}/0 "
            f"--output_path {colored_ply} "
            f"--output_type PLY"
        )
        logging.info("Converting model to colored PLY...")
        if os.system(convert_cmd) != 0:
            logging.error("Model conversion failed")
            return

    logging.info("Completed sparse reconstruction.")
    print(f"Sparse model in: {sparse_path}/0")
    print(f"Colored point cloud saved to: {sparse_path}/0/points_colored.ply")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()