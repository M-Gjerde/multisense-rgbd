import os
import subprocess
import argparse

def extract_frames(video_file, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract frames using ffmpeg
    command = [
        "ffmpeg",
        "-i", video_file,
        "-q:v", "1",  # Highest quality for JPEG (lowest compression)
        "-vf", "fps=30",  # Adjust frame rate if needed
        f"{output_folder}/frame_%06d.png"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Frames extracted successfully to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from .mov video.")
    parser.add_argument("video_file", help="Path to the .mov video file")
    parser.add_argument("output_folder", help="Directory to save the extracted frames")
    args = parser.parse_args()

    extract_frames(args.video_file, args.output_folder)
