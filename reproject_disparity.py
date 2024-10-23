import argparse
import csv
import os
import json
import sys

import glfw
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm

first_frame = True


def disparity_to_depth(disparity_image, focal_length, baseline):
    """Convert disparity image to depth map."""
    mask = disparity_image < 5  # Mask where disparity is less than 5 (invalid disparity)
    disparity_image[mask] = 0.1  # Avoid division by zero by setting small value
    depth_map = (focal_length * baseline) / disparity_image
    depth_map[mask] = 0  # Reset masked depth values to 0
    return depth_map


def load_image(file_path, color=False):
    """Load image from file."""
    if color:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    else:
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return image


def remove_noise_from_point_cloud_simple(point_cloud, nb_neighbors=20, std_ratio=2.0):
    """Remove comet tails or floaters from point cloud using statistical outlier removal."""
    # Statistical outlier removal
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Inlier point cloud
    point_cloud_clean = point_cloud.select_by_index(ind)

    return point_cloud_clean


def remove_noise_from_point_cloud(point_cloud, nb_neighbors=20, std_ratio_start=0.5, distance_bin_size=0.1):
    """
    Remove noise from point cloud with an exponentially decreasing filter strength based on distance.

    Parameters:
    - point_cloud: The input point cloud (Open3D PointCloud object).
    - nb_neighbors: Number of neighbors to use for statistical outlier removal.
    - std_ratio_start: Initial standard deviation ratio (strong filtering).
    - distance_bin_size: The size of each distance bin (in meters, default is 0.5 meters).

    Returns:
    - point_cloud_clean: Cleaned point cloud.
    """

    # Get point coordinates (Nx3 array)
    points = np.asarray(point_cloud.points)

    # Calculate distances from origin (or from another reference point if needed)
    distances = np.linalg.norm(points, axis=1)

    # Find the maximum distance to determine the number of bins
    max_distance = np.max(distances)

    # Determine the number of bins based on the distance bin size
    num_bins = int(np.ceil(max_distance / distance_bin_size))

    # Initialize an empty point cloud to store the cleaned points
    point_cloud_clean = point_cloud.select_by_index([])  # Empty point cloud

    # Loop through each bin and apply noise removal with decreasing strength
    for bin_idx in range(num_bins):
        # Define the range of the current bin
        min_distance = bin_idx * distance_bin_size
        max_distance = (bin_idx + 1) * distance_bin_size

        # Get the points that fall within this bin
        bin_indices = np.where((distances >= min_distance) & (distances < max_distance))[0]

        if len(bin_indices) == 0:
            continue  # Skip empty bins

        # Select points within the current bin
        point_cloud_bin = point_cloud.select_by_index(bin_indices)

        # Apply exponentially decreasing filter strength based on the bin index
        std_ratio = std_ratio_start * np.exp(bin_idx * 0.2)  # Exponential growth of std_ratio

        # Apply statistical outlier removal
        point_cloud_bin_clean, bin_inlier_ind = point_cloud_bin.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

        if len(point_cloud_bin_clean.points) == 0:
            print("Warning: point cloud bin clean is empty.")
        else:
            point_cloud_bin_clean = point_cloud_bin.select_by_index(bin_inlier_ind)

        # Combine the cleaned points from this bin with the overall clean point cloud
        point_cloud_clean += point_cloud_bin_clean

    return point_cloud_clean


def create_colored_point_cloud(depth_map, color_image, intrinsic_matrix):
    """Generate and view colored point cloud from depth map and color image."""

    # Create Open3D depth image from depth map
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    # Create Open3D color image from the color image
    color_image_o3d = o3d.geometry.Image(color_image.astype(np.uint8))

    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0],
                             fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
                             cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2])

    # Generate RGBD image from depth and color images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image,
                                                                    depth_scale=10.0, depth_trunc=1000.0,
                                                                    convert_rgb_to_intensity=False)

    # Generate point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Transform point cloud to align with camera perspective
    point_cloud.transform([[1, 0, 0, 0],
                           [0, 1, 0, 0],  # Flip the point cloud for a correct view
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    return point_cloud


def point_cloud_to_rgbd(point_cloud, width, height, intrinsic_matrix):
    """Convert point cloud back to RGBD images (color and depth)."""
    # Create an empty depth image and color image
    depth_image = np.zeros((height, width), dtype=np.float32)
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Get points and colors from point cloud
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Project 3D points back into 2D space using the intrinsic matrix
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    for i, point in enumerate(points):
        # Perspective projection: convert 3D points to 2D
        x, y, z = point
        if z > 0:
            if z < 20:
                u = int((fx * x) / z + cx)
                v = int((fy * y) / z + cy)
                if 0 <= u < width and 0 <= v < height:
                    depth_image[v, u] = z * 1000  # Convert to millimeters
                    color_image[v, u] = (colors[i] * 255).astype(np.uint8)  # Convert normalized color to uint8
    color_image = color_image[:, :, ::-1]  # Convert BGR to RGB by reversing the color channels

    return color_image, depth_image


def save_rgbd_tum_format(color_image, depth_image, output_dir, disparity_filename, color_filename, frame_id):
    """Save color and depth images in TUM RGB-D format."""
    # Save RGB image
    rgb_file = os.path.join(output_dir, f"rgb/{disparity_filename}")
    os.makedirs(os.path.dirname(rgb_file), exist_ok=True)
    cv2.imwrite(rgb_file, color_image)

    # Save Depth image (in millimeters, 16-bit PNG)
    depth_file = os.path.join(output_dir, f"depth/{color_filename}")
    os.makedirs(os.path.dirname(depth_file), exist_ok=True)
    depth_image_mm = depth_image.astype(np.uint16)  # Convert depth to 16-bit
    cv2.imwrite(depth_file, depth_image_mm)

    print(f"Saved frame {frame_id} as RGB-D (TUM format) to: {os.path.join(output_dir, f'rgb/{color_filename}')}")


# Apply mask to depth and color images
def apply_mask_to_rgbd(color_image, depth_image, mask):
    # Mask should be a binary image where 1 means invalid
    color_image[mask != 0] = 0  # Set invalid color pixels to 0 (black)
    depth_image[mask != 0] = 0  # Set invalid depth pixels to 0
    return color_image, depth_image


def load_and_update_point_cloud(vis, point_cloud, csv_rows, index, use_aux, K_left, K_aux, args,
                                baseline=0.27006103515625):
    global first_frame
    """Load and update point cloud based on the current index."""
    row = csv_rows[index]

    left_img_path = os.path.join(args.left_folder, os.path.basename(row[0]))
    aux_img_path = os.path.join(args.aux_folder, os.path.basename(row[3]))  # Load rectified aux image

    if "igev" in args.disparity_folder:
        disparity_file = os.path.join(args.disparity_folder, os.path.basename(row[0]).replace('.png', '.npy'))
    else:
        disparity_file = os.path.join(args.disparity_folder, os.path.basename(row[2]))

    if not os.path.exists(disparity_file):
        print("Disparity file not found:", disparity_file)
        return

    # Load the color image (left or aux based on the flag)
    if use_aux:
        color_image = load_image(aux_img_path, color=True)
        intrinsic_matrix = K_aux  # Use aux intrinsic matrix
    else:
        color_image = load_image(left_img_path, color=True)
        intrinsic_matrix = K_left  # Use left intrinsic matrix

    # Load disparity
    if ".npy" in disparity_file:
        disparity_image = np.load(disparity_file)
    else:
        disparity_image = load_image(disparity_file) / 16

    # Convert disparity to depth map
    focal_length = K_left[0, 0]  # Use left camera focal length for depth calculation
    depth_image = disparity_to_depth(disparity_image, focal_length, baseline)
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    # Resize the color image to match the depth map dimensions if they are different
    if color_image.shape[:2] != depth_image.shape:
        color_image = cv2.resize(color_image, (width, height))

    #filtered_color_image, filtered_depth_image = apply_mask_to_rgbd(color_image, depth_image, mask)

    # Create point cloud and update geometry
    updated_point_cloud = create_colored_point_cloud(depth_image, color_image, intrinsic_matrix,)

    #updated_point_cloud = remove_noise_from_point_cloud_simple(updated_point_cloud)

    point_cloud.points = updated_point_cloud.points
    point_cloud.colors = updated_point_cloud.colors

    if args.save_rgbd:
        print("Removing points further than 20 meters and applying o3d statistical outlier filter")
        # Filter points by distance (Euclidean distance from origin)
        points = np.asarray(point_cloud.points)
        distances = np.linalg.norm(points, axis=1)
        mask = distances <= 20.0  # Filter points that are within 20 meters

        # Apply mask to filter the points and colors
        point_cloud.points = o3d.utility.Vector3dVector(points[mask])
        point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[mask])

        color_image_rgbd, depth_image_rgbd = point_cloud_to_rgbd(point_cloud, width, height, intrinsic_matrix)

        # Assuming depth_image is in millimeters and has a range of 0 to 20 mm
        min_depth = 0  # Minimum depth value in mm
        max_depth = 10000  # Maximum depth value in mm

        # Clip the depth values to the 0-20 range
        #depth_clipped = np.clip(depth_image_rgbd, min_depth, max_depth)
        #depth_scaled = ((depth_clipped - min_depth) / (max_depth - min_depth)) * 255
        #depth_scaled = depth_scaled.astype(np.uint8)

        # Apply the Jet colormap
        #depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        #color_image_rgbd = color_image_rgbd[:, :, ::-1]  # Convert BGR to RGB by reversing the color channels

        #cv2.imshow("Color", color_image_rgbd)
        #cv2.imshow("filtered_depth_image", depth_colored)
        #cv2.waitKey(1)
        # Save the RGB and depth images in TUM RGB-D format
        save_rgbd_tum_format(color_image_rgbd, depth_image_rgbd, args.save_rgbd_path, os.path.basename(row[3]), os.path.basename(row[2]), index)

    vis.add_geometry(point_cloud, reset_bounding_box=first_frame)
    vis.poll_events()
    vis.update_renderer()
    print(f"Frame: {index}, File: {os.path.basename(row[0])} ")
    first_frame = False

def read_args_from_file(file_path):  # Read arguments from the text file and return them as a list
    with open(file_path, 'r') as f:
        args = f.read().split()
    return args

if __name__ == "__main__":
    dataset_folder = "indoor_store"
    parser = argparse.ArgumentParser(description='Reproject disparity files into 3D point clouds')
    parser.add_argument('--csv_file', type=str, required=False, help="Path to synced_frames.csv",
                        default=f"logqs_dataset/{dataset_folder}/synced_frames.csv")
    parser.add_argument('--disparity_folder', type=str, required=False, help="Folder containing disparity .npy files",
                        default=f"logqs_dataset/{dataset_folder}/disparity")
    parser.add_argument('--left_folder', type=str, required=False, help="Folder containing left images",
                        default=f"logqs_dataset/{dataset_folder}/left")
    parser.add_argument('--aux_folder', type=str, required=False, help="Folder containing aux images",
                        default=f"logqs_dataset/{dataset_folder}/aux_rectified")
    parser.add_argument('--calibration_file', type=str, required=False, help="Path to left camera calibration file",
                        default=f"logqs_dataset/{dataset_folder}/calibration_data/left.json")
    parser.add_argument('--aux_calibration_file', type=str, required=False, help="Path to aux camera calibration file",
                        default=f"logqs_dataset/{dataset_folder}/calibration_data/aux.json")
    parser.add_argument('--right_calibration_file', type=str, required=False,
                        help="Path to aux camera calibration file",
                        default=f"logqs_dataset/{dataset_folder}/calibration_data/right.json")
    parser.add_argument('--use_aux', action='store_true', help="Flag to use aux rectified images for coloring")
    parser.add_argument('--save_rgbd', action='store_true',
                        help="Flag to save rgbd_images, Make sure to update output folder")
    parser.add_argument('--save_rgbd_path', type=str,
                        help="Flag to save rgbd_images, Make sure to update output folder")
    # Check if args should come from a file
    if len(sys.argv) == 2 and sys.argv[1].endswith('.txt'):
        # Read arguments from file
        args_from_file = read_args_from_file(sys.argv[1])
        # Pass the arguments to the parser
        args = parser.parse_args(args_from_file)
    else:
        # Parse the command-line arguments
        args = parser.parse_args()

    # Load calibration data for the left camera
    with open(args.calibration_file, 'r') as f:
        calibration_data_left = json.load(f)

    # Load calibration data for the left camera
    with open(args.right_calibration_file, 'r') as f:
        calibration_data_right = json.load(f)

    baseline = abs(float(calibration_data_right['P'][3]) / float(calibration_data_right['P'][0]))

    # Extract intrinsic matrix and baseline from the calibration file (left camera)
    K_left = np.array(calibration_data_left['K']).reshape(3, 3)
    K_aux = None
    # Load calibration data for the aux camera if using aux
    if args.use_aux:
        with open(args.aux_calibration_file, 'r') as f:
            calibration_data_aux = json.load(f)
        K_aux = np.array(calibration_data_aux['K']).reshape(3, 3)

    # Set up Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer")

    # Parse CSV file and store rows
    with open(args.csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        csv_rows = list(reader)

    # State to track the current index in the CSV file
    state = {"current_index": 105}


    # Function to go to the next frame
    def next_frame(vis):
        if state["current_index"] < len(csv_rows) - 1:
            state["current_index"] += 1
        load_and_update_point_cloud(vis, point_cloud, csv_rows, state["current_index"], args.use_aux, K_left, K_aux,
                                    args, baseline)


    # Function to go to the previous frame
    def prev_frame(vis):
        if state["current_index"] > 0:
            state["current_index"] -= 1
        load_and_update_point_cloud(vis, point_cloud, csv_rows, state["current_index"], args.use_aux, K_left, K_aux,
                                    args, baseline)


    def toggle(vis):
        print("Toggle")
        while True:
            if state["current_index"] < len(csv_rows) - 1:
                state["current_index"] += 1
                load_and_update_point_cloud(vis, point_cloud, csv_rows, state["current_index"], args.use_aux, K_left,
                                            K_aux,
                                            args, baseline)


    # Register key callbacks for left and right arrow keys
    vis.register_key_callback(262, next_frame)  # Right arrow key
    vis.register_key_callback(263, prev_frame)  # Left arrow key
    vis.register_key_callback(glfw.KEY_T, toggle)  # Left arrow key

    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud, reset_bounding_box=True)

    # Load the first frame initially
    load_and_update_point_cloud(vis, point_cloud, csv_rows, state["current_index"], args.use_aux, K_left, K_aux, args, baseline)

    # if args.save_rgbd:
    # toggle(vis)

    # Main visualization loop
    vis.run()
    vis.destroy_window()
