import argparse
import os.path

import numpy as np
import open3d as o3d
import cv2
import scipy
import yaml

import glob


def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    return {
        'rows': mapping['rows'],
        'cols': mapping['cols'],
        'dt': mapping['dt'],
        'data': mapping['data']
    }


def preprocess_yaml_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove the first line if it contains a YAML directive
    if lines[0].startswith('%YAML'):
        lines = lines[1:]

    return ''.join(lines)


def load_yaml_with_opencv_matrix(filepath):
    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)

    yaml_content = preprocess_yaml_file(filepath)
    return yaml.safe_load(yaml_content)

def load_yaml_file(filename):
    """
    Load a YAML file and return its content as a dictionary.

    Parameters:
    - filename: str, the path to the YAML file.

    Returns:
    - dict: The content of the YAML file as a dictionary.
    """
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return data



def backproject_disparity(calibration_left, calibration_right, disparity_image, skip_nth_points=1):
    if disparity_image is None:
        print("No disparity image is loaded")
        return
    points = [()]
    scale = 1
    fx = calibration_left[0] / scale
    fy = calibration_left[4] / scale
    cx = calibration_left[2] / scale
    cy = calibration_left[5] / scale

    tx = calibration_right[3] / calibration_right[0]
    # Construct the 4x4 projection matrix K_proj
    K_proj = np.array([
        [fy * tx, 0, 0, -fy * cx * tx],
        [0, fx * tx, 0, -fx * cy * tx],
        [0, 0, 0, fx * fy * tx],
        [0, 0, -fy, fy * (cx - cx)]
    ])

    # Initialize an empty array to hold the world coordinates
    # The shape is based on the disparity_array, and we'll have 3 values for each point (X, Y, Z)
    world_coordinates = np.zeros((disparity_image.shape[0], disparity_image.shape[1], 3))

    # Create a grid of x, y coordinates
    x, y = np.meshgrid(np.arange(disparity_image.shape[1]), np.arange(disparity_image.shape[0]))

    # Filter out the points where disparity is <= 8 or > 252
    mask = (disparity_image > 10) & (disparity_image <= 252)

    # Only keep the points that pass the filter
    x, y, d = x[mask], y[mask], disparity_image[mask]

    # Calculate beta for all points at once
    beta = fx / d

    # Create 4D homogeneous coordinates for all points
    uv = np.column_stack((x, y, d, np.ones_like(x)))

    # Calculate world coordinates for all points
    sensor_coord = np.dot(K_proj, uv.T) / beta

    # Normalize by the last element
    sensor_coord_scaled = sensor_coord / sensor_coord[-1, :]
    world_coordinates[y, x, :] = sensor_coord_scaled[:3, :].T

    # Store in the world_coordinates array

    world_coordinates_reshaped = world_coordinates.reshape(-1, 3)

    # Remove rows that contain just zeros
    world_coordinates_filtered = world_coordinates_reshaped[
        np.any(world_coordinates_reshaped != 0, axis=1)
    ]

    # Take every nth point if specified
    if skip_nth_points > 1:
        world_coordinates_filtered = world_coordinates_filtered[::skip_nth_points, :]

    return np.column_stack((world_coordinates_filtered, np.ones((len(world_coordinates_filtered), 1))))


def main():
    parser = argparse.ArgumentParser(description='Load a YAML file with custom opencv-matrix tags into a Python dict.')
    parser.add_argument('intrinsics', type=str, help='Path to the YAML file')
    parser.add_argument('extrinsics', type=str, help='Path to the YAML file')
    parser.add_argument('disparity', type=str, help='Path to the YAML file')
    parser.add_argument('rgb', type=str, help='Path to the YAML file')

    # args = parser.parse_args()
    record_folder = "/home/magnus/PycharmProjects/logqs/data"
    disparity_path = os.path.join(record_folder, "images/disparity/front/")
    rgb_path = os.path.join(record_folder, "images/aux/front/")

    disparity_path_names = sorted(glob.glob(os.path.join(disparity_path, '*.png')))
    rgb_path_names = sorted(glob.glob(os.path.join(rgb_path, '*.png')))

    #intrinsics = load_yaml_with_opencv_matrix(os.path.join(record_folder, "830-01107-0012269_intrinsics.yml"))
    #extrinsics = load_yaml_with_opencv_matrix(os.path.join(record_folder, "830-01107-0012269_extrinsics.yml"))

    aux_calibration = load_yaml_with_opencv_matrix(os.path.join(record_folder, "calibration/front/aux_calibration.yaml"))
    left_calibration = load_yaml_with_opencv_matrix(os.path.join(record_folder, "calibration/front/left_calibration.yaml"))
    right_calibration = load_yaml_with_opencv_matrix(os.path.join(record_folder, "calibration/front/right_calibration.yaml"))

    save_location = "/home/magnus/phd/SplaTAM/data/custom_rgbd/work_desk1"
    save_location_disparity = os.path.join(save_location, "depth")
    save_location_rgb = os.path.join(save_location, "rgb")
    # Create the new directory if it does not exist
    if not os.path.exists(save_location_disparity):
        os.makedirs(save_location_disparity)
    if not os.path.exists(save_location_rgb):
        os.makedirs(save_location_rgb)

    # List for storing new file paths
    new_disparity_path_names = []
    new_rgb_path_names = []

    # Loop over each file path and create a new path
    for file_path in disparity_path_names:
        # Extract the base name of the file
        file_name = os.path.basename(file_path).replace(".tiff", ".png")
        # Create a new path with the same file name in the new directory
        new_file_path = os.path.join(save_location_disparity, file_name)
        new_disparity_path_names.append(new_file_path)
    # Loop over each file path and create a new path
    for file_path in rgb_path_names:
        # Extract the base name of the file
        file_name = os.path.basename(file_path).replace(".ppm", ".png")
        # Create a new path with the same file name in the new directory
        new_file_path = os.path.join(save_location_rgb, file_name)
        new_rgb_path_names.append(new_file_path)


    color_intrinsic = np.array(aux_calibration["K"]).reshape(3, 3)
    color_intrinsic = color_intrinsic[0:3, 0:3]
    color_intrinsic[2, 2] = 1
    color_intrinsic = np.hstack([color_intrinsic, np.array([0, 0, 0]).reshape(-1, 1)])
    color_intrinsic = np.vstack([color_intrinsic, np.array([0, 0, 0, 1])])

    color_extrinsic = np.array(aux_calibration["R"]).reshape(3, 3)
    x = aux_calibration["P"][3] / aux_calibration["P"][0]
    y = 0
    z =  0
    color_extrinsic = np.hstack([color_extrinsic, np.array([x, y, z]).reshape(-1, 1)])
    color_extrinsic = np.vstack([color_extrinsic, np.array([0, 0, 0, 1])])

    csv_rgb = os.path.join(save_location, 'rgb.txt')
    csv_disparity = os.path.join(save_location, 'depth.txt')
    csv_ground_truth = os.path.join(save_location, 'groundtruth.txt')
    # Open the .txt file in append mode
    csv_rgb_file = open(csv_rgb, 'a')
    csv_depth_file = open(csv_disparity, 'a')
    csv_ground_truth_file = open(csv_ground_truth, 'a')

    csv_rgb_file.write("# color images\n")
    csv_rgb_file.write("# file: 'work_desk1.bag'\n")
    csv_rgb_file.write("# filename filenamePath\n")
    csv_ground_truth_file.write("# ground truth trajectory\n")
    csv_ground_truth_file.write("# file: 'work_desk1.bag'\n")
    csv_ground_truth_file.write("# filename filenamePath\n")

    csv_depth_file.write("# depth images\n")
    csv_depth_file.write("# file: 'work_desk1.bag'\n")
    csv_depth_file.write("# filename filenamePath\n")

    for fileIdx in range(0, len(new_rgb_path_names)):
        print(f"Loading disparity: {disparity_path_names[fileIdx]} and color image: {rgb_path_names[fileIdx]}")
        disparity = cv2.imread(disparity_path_names[fileIdx], cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_path_names[fileIdx], cv2.IMREAD_UNCHANGED)
        world_coordinates_filtered = backproject_disparity(left_calibration["K"], right_calibration["P"],
                                                           disparity)
        P = color_intrinsic @ color_extrinsic
        # P = 1 / world_coordinates_filtered[:, 2] * P
        # Perform the matrix multiplication in one step for all coordinates
        colorCamCoords_matrix = world_coordinates_filtered @ P.T

        # Divide by the z-coordinate (the third column of world_coordinates_filtered)
        colorCamCoords_matrix /= colorCamCoords_matrix[:, 2].reshape(-1, 1)

        # u = colorCamCoords_matrix[:, 0] / colorCamCoords_matrix[:, 2]
        # v = colorCamCoords_matrix[:, 1] / colorCamCoords_matrix[:, 2]
        # z
        # uv = np.vstack([u, v]).T
        # Extract the first three columns to get the 3D coordinates in color camera space
        colorCamCoords = colorCamCoords_matrix[:, :3]

        output_rgb = np.ones((600, 960, 3), dtype=np.uint8)
        output_depth = np.zeros((600, 960, 1), dtype=np.uint16)
        for i, point in enumerate(colorCamCoords_matrix):
            if 0 < point[0] < 959 and 0 < point[1] < 594:
                distance = 1 / point[3]
                if 1.5 < distance < 30:
                    x = round(point[0])
                    y = round(point[1])
                    col = rgb[y, x, :]
                    output_rgb[y, x, 0:3] = col
                    output_depth[y, x, 0] = distance * 5000

        view_rgb = output_rgb.copy()
        view_rgb = cv2.circle(view_rgb, (480, 500), 10, (255, 0, 0), 5)
        view_rgb = cv2.circle(view_rgb, (480, 250), 10, (0, 255, 0), 5)
        view_rgb = cv2.circle(view_rgb, (430, 300), 10, (0, 0, 255), 5)
        print(
            f"Blue depth :{output_depth[500, 480]}, Green depth :{output_depth[250, 480]}, Red depth :{output_depth[430, 300]}")

        cv2.imwrite(new_rgb_path_names[fileIdx], output_rgb)
        cv2.imwrite(new_disparity_path_names[fileIdx], output_depth)
        cv2.imshow("output_rgb", view_rgb)
        cv2.imshow("output_depth", output_depth)

        # Extract the file name
        file_name = os.path.basename(new_rgb_path_names[fileIdx].replace(".png", ""))
        # Construct the relative path
        relative_path = os.path.join("rgb", file_name + ".png")
        # Construct the timestamp from the filename (assuming the filename is the timestamp)
        timestamp = os.path.splitext(file_name)[0]

        # Write the line to the csv file
        csv_rgb_file.write(f"{timestamp} {relative_path}\n")
        # Extract the file name
        file_name = os.path.basename(new_disparity_path_names[fileIdx].replace(".png", ""))
        # Construct the relative path
        relative_path = os.path.join("depth", file_name + ".png")
        # Construct the timestamp from the filename (assuming the filename is the timestamp)
        timestamp = os.path.splitext(file_name)[0]
        csv_depth_file.write(f"{timestamp} {relative_path}\n")


        # Extract the file name
        file_name = os.path.basename(new_disparity_path_names[fileIdx].replace(".png", ""))
        timestamp = os.path.splitext(file_name)[0]
        # Write the line to the csv file
        csv_ground_truth_file.write(f"{timestamp} {'1.2334 -0.0113 1.6941 0.7907 0.4393 -0.1770 -0.3879'}\n")


        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        plot = False
        if plot == True:
            # print(f"Colored {k} points")
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            # Create a coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            # Visualize
            vis.add_geometry(frame)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
