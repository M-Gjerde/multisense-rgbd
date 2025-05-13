from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from PIL import Image
import json
from tqdm import tqdm

# Define the bag file path and the root folder to save images and calibration data
bagpath = Path('download/thoro.bag')
image_save_folder = Path('logqs_dataset/thoro/')
image_save_folder.mkdir(exist_ok=True, parents=True)

# Create subfolders for the four cameras
folders = {
    'front': image_save_folder / 'front',
    #'back':  image_save_folder / 'back',
    #'left':  image_save_folder / 'left',
    #'right': image_save_folder / 'right'
}
for f in folders.values():
    f.mkdir(exist_ok=True, parents=True)

# Folder for calibration data
calibration_save_folder = image_save_folder / 'calibration_data'
calibration_save_folder.mkdir(exist_ok=True)

# Type store in case the bag has no message definitions
typestore = get_typestore(Stores.ROS2_FOXY)

# Mapping of camera names to their image and camera_info topics
camera_topics = {
    'front': '/KS21i/full_resolution/aux/image_rect_color',
    #'front': '/crl_rzr/multisense_front/aux/image_color',
    #'back':  '/crl_rzr/multisense_back/aux/image_color',
    #'left':  '/crl_rzr/multisense_left/aux/image_color',
    #'right': '/crl_rzr/multisense_right/aux/image_color',

}
camera_info_topics = {
    name: topic + '/camera_info'
    for name, topic in camera_topics.items()
}

# Function to save calibration data
def save_calibration_data(reader, camera_name, camera_info_topic, out_folder):
    conn = next((c for c in reader.connections if c.topic == camera_info_topic), None)
    if not conn:
        print(f"No connection for {camera_name}: {camera_info_topic}")
        return
    # Use only the first CameraInfo message
    for connection, timestamp, raw in reader.messages(connections=[conn]):
        msg = reader.deserialize(raw, connection.msgtype)
        calib = {
            'K': msg.K.tolist(),
            'D': msg.D.tolist(),
            'R': msg.R.tolist(),
            'P': msg.P.tolist(),
            'width': msg.width,
            'height': msg.height
        }
        out_file = out_folder / f'{camera_name}.json'
        with open(out_file, 'w') as f:
            json.dump(calib, f, indent=4)
        print(f"Saved calibration for {camera_name}: {out_file}")
        break

# Read the bag and process messages
with AnyReader([bagpath], default_typestore=typestore) as reader:
    # Gather image connections
    print(reader.connections)
    image_conns = {name: next((c for c in reader.connections if c.topic == topic), None)
                   for name, topic in camera_topics.items()}

    # Save calibration data
    for name, info_topic in camera_info_topics.items():
        save_calibration_data(reader, name, info_topic, calibration_save_folder)

    # Extract and save images
    for name, conn in image_conns.items():
        if conn is None:
            print(f"No image connection for {name}: {camera_topics[name]}")

    # Iterate through all image messages
    for connection, timestamp, raw in tqdm(
        reader.messages(connections=[c for c in image_conns.values() if c]),
        desc="Extracting images"
    ):
        msg = reader.deserialize(raw, connection.msgtype)
        ts = str(timestamp)
        # Determine which camera this message belongs to
        cam = next(name for name, c in image_conns.items() if c == connection)
        # Assume RGB BGR format for color images
        width, height = msg.width, msg.height
        img = Image.frombuffer('RGB', (width, height), msg.data, 'raw', 'BGR', 0, 1)
        out_path = folders[cam] / f'{ts}.png'
        img.save(out_path)

    print("Done extracting images and calibration.")
