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
                                                                    depth_scale=1.0, depth_trunc=1000.0,
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
                    depth_image[v, u] = z  # Convert to millimeters
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
    depth_image_mm = (depth_image * 1000).astype(np.uint16)
    cv2.imwrite(depth_file, depth_image_mm)

    # Save Depth image (in millimeters, 16-bit NPY)
    depth_file = os.path.join(output_dir, f"npy/{color_filename.replace('.tiff', '.npy')}")
    os.makedirs(os.path.dirname(depth_file), exist_ok=True)
    depth_image_mm = (depth_image * 1000).astype(np.uint16)
    np.save(depth_file, depth_image_mm)

    print(f"Saved frame {frame_id} as RGB-D (TUM format) to: {os.path.join(output_dir, f'rgb/{color_filename}')}")


# Apply mask to depth and color images
def apply_mask_to_rgbd(color_image, depth_image, mask):
    # Mask should be a binary image where 1 means invalid
    color_image[mask != 0] = 0  # Set invalid color pixels to 0 (black)
    depth_image[mask != 0] = 0  # Set invalid depth pixels to 0
    return color_image, depth_image


def load_and_update_point_cloud(vis, index, state, point_cloud, use_aux, K_left, K_aux, args,
                                baseline):
    global first_frame
    """Load and update point cloud based on the current index."""

    aux_img_path = os.path.join(args.aux_folder, state["aux"][index])
    disparity_file = os.path.join(args.disparity_folder, state["aux"][index].replace(".png", ".tiff"))


    if not os.path.exists(disparity_file):
        print("Disparity file not found:", disparity_file)
        return


    color_image = load_image(aux_img_path, color=True)
    intrinsic_matrix = K_aux  # Use aux intrinsic matrix

    disparity_image = load_image(disparity_file)

    # Convert disparity to depth map
    focal_length = K_left[0, 0]  # Use left camera focal length for depth calculation
    depth_image = disparity_to_depth(disparity_image, focal_length, baseline) * 10
    width = depth_image.shape[1]
    height = depth_image.shape[0]

    # Resize the color image to match the depth map dimensions if they are different
    if color_image.shape[:2] != depth_image.shape:
        color_image = cv2.resize(color_image, (width, height))

    # Create point cloud and update geometry
    updated_point_cloud = create_colored_point_cloud(depth_image, color_image, intrinsic_matrix,)

    # Convert Open3D point cloud to numpy array for easier distance calculation
    points = np.asarray(updated_point_cloud.points)
    colors = np.asarray(updated_point_cloud.colors)

    # Calculate the distance of each point from the origin
    distances = np.linalg.norm(points, axis=1)

    # Filter out points that are within 10 meters
    mask = distances <= 10.0
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Update the point cloud with the filtered points
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    if args.save_rgbd:
        color_image_rgbd, depth_image_rgbd = point_cloud_to_rgbd(point_cloud, width, height, intrinsic_matrix)
        # Save the RGB and depth images in TUM RGB-D format
        save_rgbd_tum_format(color_image_rgbd, depth_image_rgbd, args.save_rgbd_path, os.path.basename(state['aux'][index]), os.path.basename(state["aux"][index].replace(".png", ".tiff")), index)

    vis.add_geometry(point_cloud, reset_bounding_box=first_frame)
    vis.poll_events()
    vis.update_renderer()
    print(f"Frame: {index}, File: {os.path.basename(state['aux'][index])} ")
    first_frame = False

def read_args_from_file(file_path):  # Read arguments from the text file and return them as a list
    with open(file_path, 'r') as f:
        args = f.read().split()
    return args

def read_calibration_file(calibration_file):
    """Read calibration parameters from a YAML file."""
    fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

    # Extract P1 and P2 matrices
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    P3 = fs.getNode("P3").mat()

    fs.release()

    # Calculate focal length and baseline
    focal_length = P1[0, 0]  # Assuming the same focal length for both cameras
    tx = P2[0, 3]
    baseline = abs(-tx / focal_length)

    return focal_length, baseline, P1, P3

if __name__ == "__main__":
    dataset_folder = "/home/magnus/datasets/multisense/desk3/multisense_capture"
    parser = argparse.ArgumentParser(description='Reproject disparity files into 3D point clouds')
    parser.add_argument('--disparity_folder', type=str, required=False, help="Folder containing disparity .npy files",
                        default=f"{dataset_folder}/Disparity_Left/tiff/")
    parser.add_argument('--left_folder', type=str, required=False, help="Folder containing left images",
                        default=f"{dataset_folder}/Luma_Rectified_Left")
    parser.add_argument('--aux_folder', type=str, required=False, help="Folder containing aux images",
                        default=f"{dataset_folder}/Color_Rectified_Aux/png")
    parser.add_argument('--calibration_file', type=str, required=False, help="Path to left camera calibration file",
                        default=f"{dataset_folder}/")
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


    files = os.listdir(dataset_folder)
    for file in files:
        if "extrinsics" in file:
            focal_length, baseline, K_left, K_aux= read_calibration_file(os.path.join(dataset_folder, file))


    # Set up Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud Viewer")
    index = 0
    auxFiles = os.listdir(args.aux_folder)
    disparityFiles = os.listdir(args.disparity_folder)
    state = {"aux": auxFiles, "disparity": disparityFiles}
    # State to track the current index in the CSV file

    # Function to go to the next frame
    def next_frame(vis):
        load_and_update_point_cloud(vis, index, state, point_cloud, args.use_aux, K_left, K_aux,
                                    args, baseline)


    # Function to go to the previous frame
    def prev_frame(vis):
        load_and_update_point_cloud(vis, index, state, point_cloud, args.use_aux, K_left, K_aux,
                                    args, baseline)


    def toggle(vis):
        print("Toggle")
        global index
        while True:
            index += 1
            load_and_update_point_cloud(vis, index, state, point_cloud, args.use_aux, K_left,
                                        K_aux,
                                        args, baseline)




    # Register key callbacks for left and right arrow keys
    vis.register_key_callback(262, next_frame)  # Right arrow key
    vis.register_key_callback(263, prev_frame)  # Left arrow key
    vis.register_key_callback(glfw.KEY_T, toggle)  # Left arrow key

    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud, reset_bounding_box=True)

    # Load the first frame initially
    load_and_update_point_cloud(vis, index, state, point_cloud, args.use_aux, K_left, K_aux, args, baseline)

    # if args.save_rgbd:
    # toggle(vis)

    # Main visualization loop
    vis.run()
    vis.destroy_window()
