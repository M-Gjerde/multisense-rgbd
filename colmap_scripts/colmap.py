import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import argparse

@dataclass
class ConvertColmap:
    """Convert images to COLMAP format"""

    input_image_path: Path
    output_image_path: Path
    use_gpu: bool = True
    skip_matching: bool = False
    skip_undistortion: bool = True
    camera: Literal["OPENCV", "PINHOLE"] = "PINHOLE"
    resize: bool = False

    def main(self):
        if not os.path.exists(self.output_image_path):
            os.makedirs(self.output_image_path, exist_ok=True)

        input_image_path = str(self.input_image_path.resolve())
        output_image_path = str(self.output_image_path.resolve())
        use_gpu = 1 if self.use_gpu else 0
        colmap_command = "colmap"

        if not os.path.exists(colmap_command):
            os.makedirs(output_image_path, exist_ok=True)

        if not self.skip_matching:
            os.makedirs(f"{output_image_path}/sparse", exist_ok=True)
            # Feature extraction
            feat_extracton_cmd = (
                f"{colmap_command} feature_extractor "
                f"--database_path {output_image_path}/database.db "
                f"--image_path {input_image_path} "
                f"--ImageReader.single_camera 1 "
                f"--ImageReader.camera_model {self.camera} "
                f"--SiftExtraction.use_gpu {use_gpu} "
            )
            exit_code = os.system(feat_extracton_cmd)
            if exit_code != 0:
                logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
                exit(exit_code)

            # Feature matching
            feat_matching_cmd = (
                f"{colmap_command} exhaustive_matcher "
                f"--database_path {output_image_path}/database.db "
                f"--SiftMatching.use_gpu {use_gpu}"
            )
            exit_code = os.system(feat_matching_cmd)
            if exit_code != 0:
                logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
                exit(exit_code)

            # Bundle adjustment
            mapper_cmd = (
                f"{colmap_command} mapper "
                f"--database_path {output_image_path}/database.db "
                f"--image_path {input_image_path} "
                f"--output_path {output_image_path}/sparse "
                f"--Mapper.ba_global_function_tolerance=0.000001"
            )
            exit_code = os.system(mapper_cmd)
            if exit_code != 0:
                logging.error(f"Mapper failed with code {exit_code}. Exiting.")
                exit(exit_code)

        # Image undistortion
        if not self.skip_undistortion:
            img_undist_cmd = (
                f"{colmap_command} image_undistorter "
                f"--image_path {input_image_path} "
                f"--input_path {output_image_path}/sparse/0 "
                f"--output_path {output_image_path} "
                f"--output_type COLMAP"
            )
            exit_code = os.system(img_undist_cmd)
            if exit_code != 0:
                logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
                exit(exit_code)

        # Convert .bin to .txt files
        sparse_path = Path(output_image_path) / "sparse" / "0"
        if sparse_path.exists():
            model_conversion_cmd = (
                f"{colmap_command} model_converter "
                f"--input_path {sparse_path} "
                f"--output_path {sparse_path} "
                f"--output_type TXT"
            )
            exit_code = os.system(model_conversion_cmd)
            if exit_code != 0:
                logging.error(f"Model conversion from .bin to .txt failed with code {exit_code}. Exiting.")
                exit(exit_code)
        else:
            logging.error(f"Sparse model path {sparse_path} does not exist. Cannot convert .bin files to .txt.")

        if self.resize:
            raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format")
    parser.add_argument("input_image_path", type=Path, help="Path to folder that contains input images")
    parser.add_argument("output_image_path", type=Path, help="Path to output folder")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU with COLMAP", default=True)
    parser.add_argument("--skip_matching", action="store_true", help="Skip matching")
    parser.add_argument("--skip_undistortion", action="store_true", help="Skip undistorting images")
    parser.add_argument("--camera", type=str, default="PINHOLE", help="Camera type")
    parser.add_argument("--resize", action="store_true", help="Resize images")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    converter = ConvertColmap(
        input_image_path=args.input_image_path,
        output_image_path=args.output_image_path,
        use_gpu=args.use_gpu,
        skip_matching=args.skip_matching,
        skip_undistortion=args.skip_undistortion,
        camera=args.camera,
        resize=args.resize
    )
    converter.main()
