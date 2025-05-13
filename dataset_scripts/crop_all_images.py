import os
import argparse
from PIL import Image


def crop_top_bottom(folder_path, top_crop=100, bottom_crop=100):
    # Get all files in the directory, ignoring subdirectories
    files = sorted(
        [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    )

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                # Crop the top and bottom
                cropped_img = img.crop((0, top_crop, width, height - bottom_crop))
                # Save the image in place, preserving the original format
                cropped_img.save(file_path)
                print(f"Cropped: {file_path}")

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    print(f"Processed {len(files)} images in '{folder_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop the top and bottom of all images in a directory.")
    parser.add_argument("folder_path", help="Path to the folder containing images")
    parser.add_argument("--top_crop", type=int, default=100, help="Pixels to crop from the top (default: 100)")
    parser.add_argument("--bottom_crop", type=int, default=100, help="Pixels to crop from the bottom (default: 100)")

    args = parser.parse_args()

    crop_top_bottom(args.folder_path, args.top_crop, args.bottom_crop)
