import os
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load Mask2Former model and image processor
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

# The class ID for "sky" in ADE20k is typically 17
SKY_CLASS_ID = 2

# ADE20K color map (150 classes)
ADE20K_COLORMAP = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140],
    [204, 5, 255], [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70],
    [0, 0, 0]  # Background class (black) - add remaining colors as needed
])

# Define paths
base_dir = '/home/magnus/PycharmProjects/multisense-rgbd/'
aux_rectified_dir = os.path.join(base_dir, 'saved_images/aux_rectified')
mask_output_dir = os.path.join(base_dir, 'saved_images/masks/segmentation')

# Create output directory if it doesn't exist
os.makedirs(mask_output_dir, exist_ok=True)

# Helper function to apply the ADE20K colormap
def apply_ade20k_colormap(segmentation_map):
    # Map each class ID to its respective color
    colored_mask = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for class_id in np.unique(segmentation_map):
        colored_mask[segmentation_map == class_id] = ADE20K_COLORMAP[class_id % len(ADE20K_COLORMAP)]
    return colored_mask

filenames = sorted(os.listdir(aux_rectified_dir))
# Process each image in aux_rectified
for filename in filenames:
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions as needed
        image_path = os.path.join(aux_rectified_dir, filename)

        # Open the image
        image = Image.open(image_path)

        # Prepare inputs for the model
        inputs = image_processor(image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the segmentation map and convert it to NumPy
        pred_semantic_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        pred_semantic_map = pred_semantic_map.numpy()  # Convert PyTorch tensor to NumPy array

        # Create a binary mask where sky is 1 and everything else is 0
        binary_sky_mask = (pred_semantic_map == SKY_CLASS_ID).astype(np.uint8)  # Convert to binary (0 or 1)

        # Apply ADE20K colormap to the full semantic segmentation map
        #full_colored_segmentation = apply_ade20k_colormap(pred_semantic_map)

        # Save the binary mask as an image
        mask_filename = os.path.join(mask_output_dir, filename)
        binary_sky_mask_img = Image.fromarray(binary_sky_mask * 255)  # Multiply by 255 to make it a proper binary mask (0 or 255)
        binary_sky_mask_img.save(mask_filename)

print(f"Binary sky masks saved in {mask_output_dir}")
