import os
import subprocess
import argparse
import sys

def extract_frames(video_file, output_folder, fmt, fps, jpeg_quality):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Determine output extension and codec options
    fmt = fmt.lower()
    if fmt == "png":
        ext = "png"
        codec_opts = ["-vcodec", "png", "-compression_level", "0"]
    elif fmt == "tiff":
        ext = "tiff"
        codec_opts = ["-vcodec", "tiff"]
    elif fmt in ("jpg", "jpeg"):
        ext = "jpg"
        # -q:v 1 is the highest quality on a 1–31 scale
        codec_opts = ["-q:v", str(jpeg_quality)]
    else:
        print(f"Unsupported format: {fmt}")
        sys.exit(1)

    output_pattern = os.path.join(output_folder, f"frame_%06d.{ext}")

    command = [
        "ffmpeg",
        "-i", video_file,
        "-vsync", "0",               # one image per frame
        "-vf", f"fps={fps}",         # frame rate
        *codec_opts,                 # codec & quality
        output_pattern
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Frames extracted successfully to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting frames: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract high-quality frames from a video with ffmpeg."
    )
    parser.add_argument("video_file", help="Path to the input video file (e.g. .mov)")
    parser.add_argument("output_folder", help="Directory to save the extracted frames")
    parser.add_argument(
        "--format", "-f",
        choices=["png", "tiff", "jpg", "jpeg"],
        default="png",
        help="Output image format (default: png, lossless)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30,
        help="Frames per second to extract (default: 30; use 'fps=1' for one per second)"
    )
    parser.add_argument(
        "--jpeg-quality", "-q",
        type=int,
        default=1,
        metavar="QUALITY",
        help="JPEG quality 1–31 (1=best). Only used if format is jpg/jpeg (default: 1)"
    )
    args = parser.parse_args()

    extract_frames(
        args.video_file,
        args.output_folder,
        args.format,
        args.fps,
        args.jpeg_quality
    )
