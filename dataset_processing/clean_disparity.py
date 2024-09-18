import cv2
import numpy as np
import glob
import os

# Path to disparity images
disparity_images = sorted(glob.glob('../multisense/somerset/igev_disparity/*.npy'))

# Output folder for filtered images
output_folder = '../multisense/somerset/igev_disparity_filtered/'
os.makedirs(output_folder, exist_ok=True)

# Parameters for thresholding
gradient_threshold = 3  # Adjust this value to tune noise vs valid edge distinction

# Loop through each image
for image_path in disparity_images:
    # Load disparity image (as a float32 numpy array from .npy)
    disparity = np.load(image_path).astype(np.float32)

    # Normalize disparity to 8-bit range (0-255) for edge detection
    norm_disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute the gradients using Sobel operator
    grad_x = cv2.Sobel(disparity, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(disparity, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude of gradient
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Threshold gradient magnitude to detect abrupt changes (noisy background)
    mask = gradient_magnitude < gradient_threshold

    # Optional: Detect edges using Canny to preserve actual edges
    edges = cv2.Canny(norm_disparity, 50, 150)

    # Combine mask with edge detection to avoid masking valid edges
    final_mask = np.logical_or(mask, edges > 0)

    # Apply mask to the original disparity: set masked areas to 0 in the original disparity
    filtered_disparity = np.where(final_mask, disparity, 0)

    # Save the filtered disparity as .npy
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    np.save(output_path, filtered_disparity)

    print(f'Saved filtered disparity image to: {output_path}')
