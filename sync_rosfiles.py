import os
import csv
from pathlib import Path

# Define the image folders
image_save_folder = Path('multisense/somerset/')
folders = {
    'right': image_save_folder / 'right',
    'left': image_save_folder / 'left',  # Use left as the reference
    'disparity': image_save_folder / 'disparity',
    'aux': image_save_folder / 'aux'
}

# Define the output CSV file path
output_csv = image_save_folder / 'synced_frames.csv'

# Time threshold for matching in nanoseconds based on 15fps (66.67 milliseconds = 66,670,000 nanoseconds)
time_threshold_ns = 66_670_000  # Approximately one frame time difference


# Function to extract the timestamp from the filename (assumes filenames are timestamps)
def extract_timestamp(filename):
    return int(Path(filename).stem)


# Function to find the closest match within the threshold
def find_closest_match(target_timestamp, timestamps, key):
    closest_timestamp = None
    min_diff = float('inf')

    for timestamp in timestamps:
        diff = abs(target_timestamp - timestamp)
        if diff < min_diff and diff <= time_threshold_ns:
            closest_timestamp = timestamp
            min_diff = diff

    # Debug: Print the timestamp matching process
    if closest_timestamp:
        print(f"Match found for {key}: Target: {target_timestamp}, Closest: {closest_timestamp}, Diff: {min_diff}")
    else:
        print(f"No match found for {key}: Target: {target_timestamp}")

    return closest_timestamp


# Load the filenames and their timestamps from each folder
timestamps = {}
for key, folder in folders.items():
    timestamps[key] = [extract_timestamp(f) for f in os.listdir(folder) if f.endswith('.png')]

# Sort timestamps for easier matching
for key in timestamps:
    timestamps[key].sort()

# Debug: Print the first 10 timestamps from each folder to inspect
for key, ts_list in timestamps.items():
    print(f"{key} timestamps: {ts_list[:10]}")  # Print first 10 timestamps

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['left', 'right', 'disparity', 'aux', 'aux_rectified'])  # Write the header row

    # Iterate through the left folder timestamps as the reference
    for left_timestamp in timestamps['left']:
        # Find the closest matching timestamps in the other folders (right, disparity, aux)
        right_timestamp = find_closest_match(left_timestamp, timestamps['right'], 'right')
        disparity_timestamp = find_closest_match(left_timestamp, timestamps['disparity'], 'disparity')
        aux_timestamp = find_closest_match(left_timestamp, timestamps['aux'], 'aux')

        # If a match is found for all, write the corresponding filenames to the CSV
        if right_timestamp and disparity_timestamp and aux_timestamp:
            csv_writer.writerow([
                f'left/{left_timestamp}.png',
                f'right/{right_timestamp}.png',
                f'disparity/{disparity_timestamp}.png',
                f'aux/{aux_timestamp}.png',
                f'aux_rectified/{aux_timestamp}.png'
            ])

print(f"Synced frames CSV saved at {output_csv}")
