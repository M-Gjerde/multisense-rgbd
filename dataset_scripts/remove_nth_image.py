import os

# Directory containing the images
image_directory = 'desk_high_res_png'
n = 2  # Delete every n-th image

# Get a sorted list of all image files in the directory
image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(('.png', '.ppm'))])

# Iterate over the list and delete every n-th image
for index, filename in enumerate(image_files):
    if (index + 1) % n == 0:
        file_path = os.path.join(image_directory, filename)
        try:
            os.remove(file_path)
            print(f"Deleted {filename}")
        except Exception as e:
            print(f"Error deleting {filename}: {e}")

print("Deletion process completed.")
