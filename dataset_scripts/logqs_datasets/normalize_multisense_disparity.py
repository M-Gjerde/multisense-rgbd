import cv2
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Normalize 16-bit disparity images to 8-bit.")
parser.add_argument('--input_folder', required=True, help='Path to the folder containing 16-bit disparity images.')
parser.add_argument('--output_folder', required=True, help='Path to save normalized 8-bit images.')

args = parser.parse_args()

# Assign input and output folders from arguments
input_folder = args.input_folder
output_folder = args.output_folder

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # Read the 16-bit disparity image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # Normalize to 8-bit by dividing by 16
        normalized_image = (image / 16).astype('uint8')
        color_image = cv2.applyColorMap(normalized_image, cv2.COLORMAP_JET)
        # Save the normalized image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, color_image)

print('Normalization complete. Images saved to', output_folder)

