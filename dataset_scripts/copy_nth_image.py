import os
import shutil
import ctypes

# Source directory containing the images
source_directory = '../datasets/viewer/desk3/multisense_capture/Color_Rectified_Aux/png'
# Destination directory to copy the selected images
destination_directory = '../datasets/viewer/desk3/images'
n = 4  # Copy every n-th image

# Ensure the destination directory exists
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Get a sorted list of all image files in the source directory
image_files = sorted([f for f in os.listdir(source_directory) if f.endswith(('.png', '.ppm'))])
# Iterate over the list and copy every n-th image
for index, filename in enumerate(image_files):
    if (index + 1) % n == 0:
        source_file_path = os.path.join(source_directory, filename)
        destination_file_path = os.path.join(destination_directory, filename)
        try:
            shutil.copy2(source_file_path, destination_file_path)  # Use copy2 to preserve metadata
            print(f"Copied {filename} to {destination_directory}")
        except Exception as e:
            print(f"Error copying {filename}: {e}")

print("Copying process completed.")
