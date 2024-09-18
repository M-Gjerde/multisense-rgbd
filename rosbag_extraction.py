from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from PIL import Image
import numpy as np
import json

# Define the bag file path and the root folder to save images and calibration data
bagpath = Path('rosbags/somerset.bag')
image_save_folder = Path('multisense/somerset/')
image_save_folder.mkdir(exist_ok=True, parents=True)

# Create subfolders for the different image types
right_folder = image_save_folder / 'right'
right_folder.mkdir(exist_ok=True)
left_folder = image_save_folder / 'left'
left_folder.mkdir(exist_ok=True)
disparity_folder = image_save_folder / 'disparity'
disparity_folder.mkdir(exist_ok=True)
aux_folder = image_save_folder / 'aux'
aux_folder.mkdir(exist_ok=True)

# Create a folder for calibration data
calibration_save_folder = image_save_folder / 'calibration_data'
calibration_save_folder.mkdir(exist_ok=True)

# Create a type store to use if the bag has no message definitions
typestore = get_typestore(Stores.ROS2_FOXY)

# Function to save calibration data
def save_calibration_data(reader, camera_name, camera_info_topic, calibration_save_folder):
    # Filter connections for the camera info topic
    camera_info_connection = next((x for x in reader.connections if x.topic == camera_info_topic), None)

    # Extract and save calibration data (camera info)
    if camera_info_connection:
        for connection, timestamp, rawdata in reader.messages(connections=[camera_info_connection]):
            # Deserialize the camera info message
            camera_info_msg = reader.deserialize(rawdata, connection.msgtype)

            # Convert NumPy arrays to lists for JSON serialization
            calibration_data = {
                'K': camera_info_msg.K.tolist(),
                'D': camera_info_msg.D.tolist(),
                'R': camera_info_msg.R.tolist(),
                'P': camera_info_msg.P.tolist(),
                'width': camera_info_msg.width,
                'height': camera_info_msg.height
            }

            # Save the calibration data as a JSON file
            calibration_file_path = calibration_save_folder / f'{camera_name}.json'
            with open(calibration_file_path, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print(f"Saved calibration data for {camera_name}: {calibration_file_path}")
    else:
        print(f"No connection found for {camera_name} camera info topic: {camera_info_topic}")


# Open the bag file for reading
with AnyReader([bagpath], default_typestore=typestore) as reader:
    # Define the list of topics you want to extract data from
    image_topics = [
        '/crl_rzr/multisense_front/right/image_rect',
        '/crl_rzr/multisense_front/left/image_rect',
        '/crl_rzr/multisense_front/aux/image_color',
        '/crl_rzr/multisense_front/left/disparity'
    ]
    # Define the camera info topics
    camera_info_topics = {
        'left': '/crl_rzr/multisense_front/left/image_rect/camera_info',
        'right': '/crl_rzr/multisense_front/right/image_rect/camera_info',
        'right': '/crl_rzr/multisense_front/right/camera_info',
        'aux': '/crl_rzr/multisense_front/aux/image_color/camera_info'
    }


    # Filter connections to include only those with image data
    image_connections = [x for x in reader.connections if x.topic in image_topics]
    # Assuming you have initialized `reader` with bag file data
    for camera_name, camera_info_topic in camera_info_topics.items():
        save_calibration_data(reader, camera_name, camera_info_topic, calibration_save_folder)

    # Extract image data and save it
    for connection, timestamp, rawdata in reader.messages(connections=image_connections):
        # Deserialize the message
        msg = reader.deserialize(rawdata, connection.msgtype)
        timestamp_str = str(timestamp)  # Convert timestamp to string for the filename

        if connection.topic == '/crl_rzr/multisense_front/left/disparity':
            # Handle disparity images separately (16-bit grayscale)
            width = msg.width
            height = msg.height
            img_data = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)  # Assuming 16-bit grayscale
            disparity_image = Image.fromarray(img_data, mode='I;16')  # I;16 mode for 16-bit grayscale
            image_file_path = disparity_folder / f'{timestamp_str}.png'
            disparity_image.save(image_file_path)
            print(f"Saved disparity image: {image_file_path}")

        elif connection.topic == '/crl_rzr/multisense_front/right/image_rect':
            # Handle right grayscale images
            width = msg.width
            height = msg.height
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)  # Assuming 8-bit grayscale
            grayscale_image = Image.fromarray(img_data, mode='L')  # L mode for grayscale
            image_file_path = right_folder / f'{timestamp_str}.png'
            grayscale_image.save(image_file_path)
            print(f"Saved right image: {image_file_path}")

        elif connection.topic == '/crl_rzr/multisense_front/left/image_rect':
            # Handle left grayscale images
            width = msg.width
            height = msg.height
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)  # Assuming 8-bit grayscale
            grayscale_image = Image.fromarray(img_data, mode='L')  # L mode for grayscale
            image_file_path = left_folder / f'{timestamp_str}.png'
            grayscale_image.save(image_file_path)
            print(f"Saved left image: {image_file_path}")

        elif connection.topic == '/crl_rzr/multisense_front/aux/image_color':
            # Handle aux RGB image
            width = msg.width
            height = msg.height
            img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)  # Assuming 3 channels (RGB)
            img_data_rgb = img_data[:, :, ::-1]  # Convert BGR to RGB by reversing the color channels
            rgb_image = Image.fromarray(img_data_rgb)
            image_file_path = aux_folder / f'{timestamp_str}.png'
            rgb_image.save(image_file_path)
            print(f"Saved aux RGB image: {image_file_path}")

