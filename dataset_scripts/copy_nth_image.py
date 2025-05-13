import os
import shutil
import argparse


def copy_nth_image(src_dir, dest_dir, n):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get a sorted list of files in the source directory, ignoring subdirectories
    files = sorted(
        [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    )

    # Copy every nth file
    for i in range(0, len(files), n):
        src_file = os.path.join(src_dir, files[i])
        dest_file = os.path.join(dest_dir, files[i])
        shutil.copy2(src_file, dest_file)
        print(f"Copied: {src_file} -> {dest_file}")

    print(f"Copied every {n}th image from '{src_dir}' to '{dest_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy every nth image from a folder to a new folder.")
    parser.add_argument("src_dir", help="Source directory containing images")
    parser.add_argument("dest_dir", help="Destination directory to copy images to")
    parser.add_argument("n", type=int, help="Copy every nth image")

    args = parser.parse_args()

    copy_nth_image(args.src_dir, args.dest_dir, args.n)
