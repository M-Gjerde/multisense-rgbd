import os
import cv2

# Source directory containing the images
source_directory = '../datasets/viewer/desk2_phone'
# Create destination directories for half-size and quarter-size images
half_size_directory = os.path.join(source_directory, "images_2")
quarter_size_directory = os.path.join(source_directory, "images_4")

os.makedirs(half_size_directory, exist_ok=True)
os.makedirs(quarter_size_directory, exist_ok=True)

# Get a sorted list of all image files in the source directory
image_files = sorted([f for f in os.listdir(os.path.join(source_directory, "images")) if f.endswith(('.png', '.ppm', '.jpg', '.jpeg'))])

# Iterate over the list, load, resize, and save each image
for filename in image_files:
    source_file_path = os.path.join(source_directory, "images", filename)

    try:
        # Load the image using OpenCV
        img = cv2.imread(source_file_path)

        # Get the original image dimensions
        height, width = img.shape[:2]

        # Resize the image to half the original dimensions
        half_resized_img = cv2.resize(img, (width // 2, height // 2))
        half_size_file_path = os.path.join(half_size_directory, filename)
        cv2.imwrite(half_size_file_path, half_resized_img)
        print(f"Resized to half and saved {filename} to {half_size_directory}")

        # Resize the image to quarter the original dimensions
        quarter_resized_img = cv2.resize(img, (width // 4, height // 4))
        quarter_size_file_path = os.path.join(quarter_size_directory, filename)
        cv2.imwrite(quarter_size_file_path, quarter_resized_img)
        print(f"Resized to quarter and saved {filename} to {quarter_size_directory}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Resizing process completed.")
