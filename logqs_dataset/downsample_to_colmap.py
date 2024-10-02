#!/usr/bin/env python3

import os
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Downsample images for COLMAP.')
    parser.add_argument('--input_folder', required=True, help='Path to the input folder containing RGB images.')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder to store downsampled images.')
    parser.add_argument('--nth', type=int, default=1, help='Copy every nth image.')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    nth = args.nth

    # Create output_folder/images if it doesn't exist
    images_output_folder = output_folder / 'images'
    images_output_folder.mkdir(parents=True, exist_ok=True)

    # List image files in input_folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    # Sort the image files
    image_files.sort()

    # Select every nth image
    selected_images = image_files[::nth]

    # Copy selected images to the output folder
    for img_path in selected_images:
        destination = images_output_folder / img_path.name
        shutil.copy(img_path, destination)

    print(f'Copied {len(selected_images)} images to {images_output_folder}')

if __name__ == '__main__':
    main()
