import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Directory containing the .ppm images
input_directory = '../datasets/viewer/desk3/multisense_capture/Color_Rectified_Aux/ppm'
output_directory = '../datasets/viewer/desk3/multisense_capture/Color_Rectified_Aux/png'
max_workers = 32# Adjust this value based on your CPU's core count

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to convert a single .ppm image to .png
def convert_image(filename):
    if filename.endswith(".ppm"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename.replace(".ppm", ".png"))

        try:
            with Image.open(input_path) as img:
                img.save(output_path, "PNG")
            print(f"Converted {filename} to PNG format.")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

# Get a list of all .ppm files in the directory
ppm_files = [f for f in os.listdir(input_directory) if f.endswith(".ppm")]

# Use ThreadPoolExecutor to parallelize the conversion
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(convert_image, ppm_files)

print("Conversion completed.")
