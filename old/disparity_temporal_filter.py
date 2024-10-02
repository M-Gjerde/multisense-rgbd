import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt



plot = False
# Paths to disparity images and segmentation masks
if plot:
    disparity_images = sorted(glob.glob('../saved_images/igev_disparity/*.npy'))[500:]
    segmentation_masks = sorted(glob.glob('../saved_images/masks/segmentation/*.png'))[500:]
else:
    disparity_images = sorted(glob.glob('../saved_images/igev_disparity/*.npy'))
    segmentation_masks = sorted(glob.glob('../saved_images/masks/segmentation/*.png'))
# Output folder for filtered disparity images (currently not saving)
output_folder = '../saved_images/igev_disparity_temporal_filtered/'
os.makedirs(output_folder, exist_ok=True)

# Threshold for standard deviation
std_threshold = 2  # Adjust this threshold for strictness

# Temporal filtering buffer
temporal_window = 5  # More frames for better standard deviation calculation
disparity_buffer = []


# Function to apply mask to disparity images
def apply_mask_to_depth(depth_image, mask):
    # Mask should be a binary image where 1 means invalid
    depth_image[mask != 0] = 0  # Set invalid depth pixels to 0
    return depth_image


# Function to mask pixels based on standard deviation over time
def temporal_filtering_with_std_masking(current_disparity, disparity_buffer, std_threshold):
    if len(disparity_buffer) >= temporal_window:
        disparity_buffer.pop(0)
    disparity_buffer.append(current_disparity)

    if len(disparity_buffer) == temporal_window:
        # Stack disparity images in the buffer and calculate the standard deviation for each pixel
        stacked_disparities = np.stack(disparity_buffer, axis=0)
        pixel_std = np.std(stacked_disparities, axis=0)

        # Mask out pixels where the standard deviation exceeds the threshold
        mask = pixel_std > std_threshold
        filtered_disparity = np.copy(current_disparity)
        filtered_disparity[mask] = 0  # Set large-change pixels to 0 (masked)

        return filtered_disparity
    else:
        return None  # Skip until we have enough frames for filtering


# Loop through each disparity image and mask
for i, image_path in enumerate(disparity_images):
    # Load disparity image (as a float32 numpy array from .npy)
    depth_image = np.load(image_path).astype(np.float32)

    # Load the corresponding segmentation mask
    mask_path = segmentation_masks[i]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the mask size matches the disparity image
    mask = cv2.resize(mask, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask to the disparity image
    masked_disparity = depth_image.copy()
    masked_disparity = apply_mask_to_depth(masked_disparity, mask)

    # Apply temporal filtering with standard deviation masking on the masked disparity image
    filtered_disparity = temporal_filtering_with_std_masking(masked_disparity.copy(), disparity_buffer, std_threshold)

    # Only proceed if we have enough frames for filtering
    if filtered_disparity is not None:
        # Commented out saving to file
        # Visualize the result (starting from the first filtered image)
        if plot:
            plt.figure(figsize=(22, 6))
            plt.subplot(1, 3, 1)
            plt.title('Original Disparity')
            plt.imshow(depth_image, cmap='jet')
            plt.subplot(1, 3, 2)
            plt.title('Masked Disparity')
            plt.imshow(masked_disparity, cmap='jet')
            plt.subplot(1, 3, 3)
            plt.title('Filtered (Temporal Std) Disparity')
            plt.imshow(filtered_disparity, cmap='jet')
            plt.show()
        else:
            np.save(os.path.join(output_folder, os.path.basename(image_path)), filtered_disparity)
            print("Saved disparity mask to {}".format(os.path.join(output_folder,  os.path.basename(image_path))))

