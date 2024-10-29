import os
import shutil


def copy_depth_images(images_folder, depth_left_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image filenames in the images folder
    image_filenames = sorted(os.listdir(images_folder))

    # Iterate through each image filename
    for image_filename in image_filenames:
        # Construct the corresponding depth image path in depth_left_folder
        depth_image_path = os.path.join(depth_left_folder, image_filename)

        # Check if the corresponding depth image exists
        if os.path.exists(depth_image_path):
            # Construct the destination path in the output folder
            output_depth_image_path = os.path.join(output_folder, image_filename)

            # Copy the depth image to the output folder
            shutil.copy2(depth_image_path, output_depth_image_path)
            print(f"Copied {image_filename} to {output_folder}")
        else:
            print(f"Warning: Depth image not found for {image_filename}")


if __name__ == "__main__":
    # Define paths
    images_folder = "/home/magnus/datasets/multisense/desk3/images"
    depth_left_folder = "/home/magnus/datasets/multisense/desk3/multisense_capture/depth_left"
    output_folder = "/home/magnus/datasets/multisense/desk3/depth_images"

    # Run the function to copy depth images
    copy_depth_images(images_folder, depth_left_folder, output_folder)
