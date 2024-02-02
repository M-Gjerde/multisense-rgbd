import argparse

import numpy as np
import open3d as o3d
import cv2
import scipy
import yaml


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


def backproject_disparity(calibration_left, calibration_right, disparity_image, skip_nth_points=1):
    if disparity_image is None:
        print("No disparity image is loaded")
        return
    points = [()]
    scale = 2
    fx = calibration_left[0] / scale
    fy = calibration_left[5] / scale
    cx = calibration_left[2] / scale
    cy = calibration_left[6] / scale

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
    sensor_coord_scaled = sensor_coord / sensor_coord[-1, :] * scale

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

    args = parser.parse_args()

    intrinsics = load_yaml_with_opencv_matrix(args.intrinsics)
    extrinsics = load_yaml_with_opencv_matrix(args.extrinsics)
    disparity = cv2.imread(args.disparity, cv2.IMREAD_UNCHANGED) / 16
    rgb = cv2.imread(args.rgb, cv2.IMREAD_UNCHANGED)
    world_coordinates_filtered = backproject_disparity(extrinsics["P1"]["data"], extrinsics["P2"]["data"], disparity)

    color_intrinsic = np.array(extrinsics["P3"]["data"]).reshape(3, 4) / 2
    color_intrinsic = color_intrinsic[0:3, 0:3]
    color_intrinsic = np.hstack([color_intrinsic, np.array([0, 0, 0]).reshape(-1, 1)])
    color_intrinsic = np.vstack([color_intrinsic, np.array([0, 0, 0, 1])])
    print(color_intrinsic)

    color_extrinsic = np.array(extrinsics["R3"]["data"]).reshape(3, 3)
    x = extrinsics["P3"]["data"][3] / extrinsics["P3"]["data"][0]
    y = extrinsics["P3"]["data"][7] / extrinsics["P3"]["data"][0]
    z = extrinsics["P3"]["data"][11] / extrinsics["P3"]["data"][0]
    color_extrinsic = np.hstack([color_extrinsic, np.array([x, y, z]).reshape(-1, 1)])
    color_extrinsic = np.vstack([color_extrinsic, np.array([0, 0, 0, 1])])
    print()
    print(color_extrinsic)

    # Convert inCoords to a homogeneous coordinate (4-element vector)
    colorCamCoords = []
    for i, row in enumerate(world_coordinates_filtered):
        inCoords_h = np.array(
            [world_coordinates_filtered[i, 0], world_coordinates_filtered[i, 1], world_coordinates_filtered[i, 2], 1.0])

        # Reshape inCoords_h to a row vector (1x4 matrix)
        inCoords_h_row = inCoords_h.reshape(1, -1)

        # Reverse the multiplication order for row-major order
        colorCamCoords_row = inCoords_h_row @ color_extrinsic.T @ color_intrinsic.T

        # Divide by the z-coordinate (the third element)
        colorCamCoords_row /= world_coordinates_filtered[i, 2]

        # Now colorCamCoords_row is a 1x4 matrix where the first three elements are x, y, z in color camera space
        # You may take just the first three elements to get the 3D coordinates
        colorCamCoords.append(colorCamCoords_row[0, :3])
    colorCamCoords = np.array(colorCamCoords)

    print(colorCamCoords)
    # Project world coordinates into the color image.
    # (nx3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_coordinates_filtered[:, :3])
    arr = np.array([[0, 0, 0] for i, point in enumerate(colorCamCoords)]).reshape(-1, 3)
    pcd.colors = o3d.utility.Vector3dVector(arr)
    k = 0
    maxVal = 0
    minVal = 10000
    for i, point in enumerate(colorCamCoords):
        if point[2] > maxVal:
            maxVal = point[2]
        if point[2] < minVal:
            minVal = point[2]

    output_image = np.ones((600, 960, 4), dtype=np.float32)
    for i, point in enumerate(colorCamCoords):
        if 0 < point[0] < 960 and 0 < point[1] < 600:
            x = int(point[0])
            y = int(point[1])
            col = rgb[y, x, :]
            pcd.colors[i] = col / 255
            k += 1
            depth = (point[2] - minVal) / (maxVal - minVal)
            output_image[y, x, 0:3] = col
            output_image[y, x, 3] = depth
            # print(f"d channel: {depth}")
            # print(x, y)
    cv2.imshow("output", output_image)
    cv2.imwrite("output.png", output_image)


    readImg = cv2.imread("output.png")
    cv2.imshow("readImg", readImg)

    cv2.waitKey(0)
    print(f"Colored {k} points")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # Create a coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # Visualize
    vis.add_geometry(frame)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
