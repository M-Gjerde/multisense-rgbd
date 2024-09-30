#!/usr/bin/env python3

import argparse
import os
import sys
import math
import cv2
import shutil
import numpy as np
import json
from glob import glob
from pathlib import Path, PurePosixPath

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a text colmap export to NeRF format transforms.json")
    parser.add_argument("--colmap_text", required=True, help="Path to the COLMAP text files (cameras, images, points3D)")
    parser.add_argument("--images", required=True, help="Path to the images used in COLMAP")
    parser.add_argument("--out", default="transforms.json", help="Output JSON file path")
    parser.add_argument("--aabb_scale", default=32, choices=["1", "2", "4", "8", "16", "32", "64", "128"],
                        help="Scale factor for the bounding box")
    parser.add_argument("--skip_early", default=0, type=int, help="Skip this many images from the start")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep original COLMAP coordinate system")
    return parser.parse_args()

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

def load_camera_params(colmap_text_path):
    cameras = {}
    with open(os.path.join(colmap_text_path, "sparse/0/cameras.txt"), "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["k1"], camera["k2"], camera["p1"], camera["p2"] = 0, 0, 0, 0
            if els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            # Add handling for more camera models here if necessary
            cameras[camera_id] = camera
    return cameras

def main():
    args = parse_args()
    cameras = load_camera_params(args.colmap_text)
    out = {"frames": [], "aabb_scale": int(args.aabb_scale)}

    with open(os.path.join(args.colmap_text, "sparse/0/images.txt"), "r") as f:
        line_num = 0  # Counter to track lines

        for line in f:
            if line[0] == "#":
                continue

            # Skip every other line since it contains matches
            if line_num % 2 == 1:
                line_num += 1
                continue
            elems = line.split(" ")
            print(elems)
            image_id = int(elems[0])
            qvec = np.array([float(v) for v in elems[1:5]])
            tvec = np.array([float(v) for v in elems[5:8]])
            camera_id = int(elems[8])  # Get camera_id from the correct column
            image_path = os.path.join(args.images, "_".join(elems[9:]).strip())

            # Get the correct camera parameters for this image using camera_id
            camera = cameras.get(camera_id)
            if camera is None:
                print(f"Warning: No camera found for camera_id {camera_id}")
                continue

            R = qvec2rotmat(qvec)
            t = tvec.reshape([3, 1])
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(m)
            frame = {"file_path": image_path, "transform_matrix": c2w.tolist()}
            frame.update(camera)  # Use camera parameters based on camera_id
            out["frames"].append(frame)
            line_num += 1

    with open(args.out, "w") as outfile:
        json.dump(out, outfile, indent=2)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
