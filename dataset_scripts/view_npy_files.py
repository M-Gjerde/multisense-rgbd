import cv2
import numpy as np
import os
import argparse


def load_images(folder_path):
    """Load .npy and .png files from a given folder."""
    # Get sorted list of .npy and .png files
    npy_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
    png_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')])
    return npy_files + png_files  # Combine both lists


def display_images(images):
    """Display .npy and .png images with OpenCV and allow navigation with arrow keys."""
    if not images:
        print("No .npy or .png files found in the given folder.")
        return

    index = 0
    total_images = len(images)
    target_width, target_height = 1280, 720  # Target window size

    while True:
        # Load the image based on the current index and file type
        current_image = images[index]
        if current_image.endswith('.npy'):
            image = np.load(current_image)
            image_normalized = image

            # Normalize and scale for better visualization (assuming the image is depth data)
            #image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:  # Assume it's a .png file
            image_normalized = cv2.imread(current_image, cv2.IMREAD_UNCHANGED)
            image = image_normalized

        minVal = min(image[0])
        maxVal = max(image[0])
        print(f"min {minVal}, max: {maxVal}")

        # Get original dimensions
        original_height, original_width = image_normalized.shape[:2]

        # Calculate scaling factor while maintaining aspect ratio
        scale_factor = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Resize image while maintaining the aspect ratio
        resized_image = cv2.resize(image_normalized, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow(f"Image Viewer - {current_image}", resized_image)
        key = cv2.waitKey(0)

        if key == 27:  # ESC key to exit
            break
        elif key == 81:  # Left arrow key
            index = (index - 1) % total_images
        elif key == 83:  # Right arrow key
            index = (index + 1) % total_images

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="View .npy and .png images and navigate using arrow keys.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing .npy and .png images.")
    args = parser.parse_args()

    images = load_images(args.folder_path)
    display_images(images)


if __name__ == "__main__":
    main()
