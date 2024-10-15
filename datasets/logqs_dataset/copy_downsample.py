import os
import argparse
import shutil
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Downsample and copy images based on a CSV file.')
    parser.add_argument('--input_csv', required=True, help='Path to the input CSV file.')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder to store downsampled images.')
    parser.add_argument('--nth', type=int, default=1, help='Copy every nth image.')
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_folder = Path(args.output_folder)
    nth = args.nth

    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Create output sub-folders for each image type
    subfolders = ['left', 'right', 'disparity', 'aux', 'aux_rectified', 'rgbd/depth', 'rgbd/rgb']
    for subfolder in subfolders:
        (output_folder / subfolder).mkdir(parents=True, exist_ok=True)

    # Store rows that correspond to the copied images
    filtered_rows = []

    # Process and copy every nth image from the CSV
    for index, row in df.iterrows():
        if index > 150:
            continue

        # Define the mappings between CSV columns and their respective folders
        file_mappings = {
            'left': 'left',
            'right': 'right',
            'disparity': 'disparity',
            'aux': 'aux',
            'aux_rectified': 'aux_rectified',
            'disparity_rgbd': 'rgbd/depth',  # Disparity used in rgbd/depth
            'aux_rgbd': 'rgbd/rgb'  # Aux used in rgbd/rgb
        }


        # Iterate over each mapping and copy files
        for column, subfolder in file_mappings.items():
            # Special handling for disparity and aux files for rgbd paths
            if column == 'disparity_rgbd':
                source_file = input_csv.parent / row['disparity']
            elif column == 'aux_rgbd':
                source_file = input_csv.parent / row['aux']
            else:
                source_file = input_csv.parent / row[column]

            destination = output_folder / subfolder / source_file.name

            if source_file.is_file():
                shutil.copy(source_file, destination)
                print(f"Copied {source_file} to {destination}")
            else:
                print(f"Warning: File {source_file} does not exist. Skipping.")

        # If all files were copied for this row, add it to filtered_rows
        filtered_rows.append(row)

    # Create a new DataFrame from the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)

    # Save the filtered DataFrame as a new synced_frames.csv
    output_csv_path = output_folder / 'synced_frames.csv'
    filtered_df.to_csv(output_csv_path, index=False)
    print(f'Saved filtered CSV to {output_csv_path}')

    print(f'Processed and copied images to {output_folder}')

if __name__ == '__main__':
    main()
