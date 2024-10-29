import os
import shutil
import numpy as np
import cv2


def load_and_save_depth_as_npy(depth_image_path, output_depth_image_path):
    """Load a depth image from a .tiff file and save it as a .npy file."""
    # Load the .tiff file as a 32-bit float image
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Save the image as a .npy file
    np.save(output_depth_image_path, depth_image)


def rename_and_copy_depth_images(images_folder, depth_left_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the sorted list of image filenames in the images folder
    image_filenames = sorted(os.listdir(depth_left_folder))

    # Iterate through each image filename and rename/copy to the output folder
    for index, image_filename in enumerate(image_filenames):
        # Construct the corresponding depth image path in depth_left_folder
        depth_image_path = os.path.join(depth_left_folder, image_filename.replace("png", "tiff"))

        # Check if the corresponding depth image exists
        if os.path.exists(depth_image_path):
            # Construct the new filename in the output folder with the format "frame_XXXXX.npy"
            new_filename = f"frame_{index:05d}_aligned.npy"
            output_depth_image_path = os.path.join(output_folder, new_filename)

            # Load and save the depth image as .npy
            shutil.copy(depth_image_path, output_depth_image_path)
            print(f"Loaded {image_filename} and saved as {new_filename}")
        else:
            print(f"Warning: Depth image not found for {image_filename}")


if __name__ == "__main__":
    # Define paths
    images_folder = "/home/magnus/datasets/multisense/jeep_gravel/images"
    depth_left_folder = "/home/magnus/datasets/multisense/jeep_gravel/rgbd/npy"
    output_folder = "/home/magnus/datasets/nerfstudio/jeep_gravel/depth_images_nerfstudio"

    # Run the function to copy and rename depth images
    rename_and_copy_depth_images(images_folder, depth_left_folder, output_folder)
