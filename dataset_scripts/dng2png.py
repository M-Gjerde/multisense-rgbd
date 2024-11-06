import os
import rawpy
from PIL import Image
import sys


def convert_dng_to_png(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(".dng"):
            # Load the DNG file
            input_path = os.path.join(input_folder, filename)
            with rawpy.imread(input_path) as raw:
                # Process the raw data and convert to RGB
                rgb_image = raw.postprocess()

                # Convert to a PIL Image
                image = Image.fromarray(rgb_image)

                # Save as .png in the output folder
                output_path = os.path.join(output_folder, filename.replace(".dng", ".png").replace(".DNG", ".png"))
                image.save(output_path, "PNG")

                print(f"Converted and saved {filename} to {output_path}")


if __name__ == "__main__":
    # Check if the user provided the required arguments
    if len(sys.argv) != 3:
        print("Usage: python dng2png.py <input_folder> <output_folder>")
        sys.exit(1)

    # Get the input and output folder paths from the command line arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Run the conversion
    convert_dng_to_png(input_folder, output_folder)
