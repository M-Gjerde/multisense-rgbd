import argparse
import os
import json
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a text colmap export to NeRF format transforms.json")
    parser.add_argument("--input_folder", required=True, help="Path to the COLMAP text files (cameras, images, points3D)")
    parser.add_argument("--csv_file", required=True, help="Path to the downsampled synced_frames.csv file")
    parser.add_argument("--out", default="transforms.json", help="Output JSON file path")
    parser.add_argument("--aabb_scale", default=1, choices=["1", "2", "4", "8", "16", "32", "64", "128"],
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
            cameras[camera_id] = camera
    return cameras

def main():
    args = parse_args()

    # Read the downsampled CSV file
    csv_path = Path(args.csv_file)
    df = pd.read_csv(csv_path)

    # Run the COLMAP model converter
    command = [
        "colmap",
        "model_converter",
        "--input_path", f'{os.path.join(args.input_folder, "sparse" , "0")}',
        "--output_path", f'{os.path.join(args.input_folder, "sparse" , "0")}',
        "--output_type", "TXT"
    ]
    # Run the COLMAP model converter
    command = [
        "colmap",
        "model_converter",
        "--input_path", f'{os.path.join(args.input_folder, "sparse" , "0")}',
        "--output_path", f'{os.path.join(args.input_folder, "sparse_pc.ply")}',
        "--output_type", "PLY"
    ]

    try:
        subprocess.run(command, check=True)
        print("COLMAP model conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

    cameras = load_camera_params(args.input_folder)
    out = {"frames": [], "aabb_scale": int(args.aabb_scale)}

    # Open COLMAP images file to extract poses
    with open(os.path.join(args.input_folder, "sparse/0/images.txt"), "r") as f:
        line_num = 0  # Counter to track lines

        for line in f:
            if line[0] == "#":
                continue

            # Skip every other line since it contains matches
            if line_num % 2 == 1:
                line_num += 1
                continue
            line_num += 1

            elems = line.split(" ")
            image_id = int(elems[0])
            qvec = np.array([float(v) for v in elems[1:5]])
            tvec = np.array([float(v) for v in elems[5:8]])
            camera_id = int(elems[8])
            image_path = "_".join(elems[9:]).strip()

            # Get the correct camera parameters for this image using camera_id
            camera = cameras.get(camera_id)
            if camera is None:
                print(f"Warning: No camera found for camera_id {camera_id}")
                continue

            # Extract the image filename (without the path) from the image_path
            image_filename = os.path.basename(image_path)

            # Match the extracted image filename to the 'left' column in the CSV
            matched_row = df[df['aux'].str.contains(image_filename, regex=False)]

            if matched_row.empty:
                print(f"Warning: No CSV entry found for image {image_filename}")
                continue

            # Use aux_rectified for color and disparity for depth
            color_path = f"images/{matched_row['aux_rectified'].values[0].split('/')[-1]}"
            depth_path = f"depth/{matched_row['disparity'].values[0].split('/')[-1]}"

            # Convert quaternion to rotation matrix
            R = qvec2rotmat(qvec)
            t = tvec.reshape([3, 1])
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            c2w = np.linalg.inv(m)

            frame = {
                "file_path": color_path,
                "depth_file_path": depth_path,
                "transform_matrix": c2w.tolist()
            }
            frame.update(camera)

            out["frames"].append(frame)

    # Save the transforms.json
    with open(os.path.join(args.input_folder, args.out), "w") as outfile:
        json.dump(out, outfile, indent=2)
    print(f"Saved transforms to {args.out}")

if __name__ == "__main__":
    main()
