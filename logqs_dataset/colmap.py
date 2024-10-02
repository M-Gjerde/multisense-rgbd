import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import argparse

@dataclass
class ConvertColmap:
    """Convert images to COLMAP format"""

    image_path: Path
    use_gpu: bool = True
    skip_matching: bool = False
    skip_undistortion: bool = True
    camera: Literal["OPENCV"] = "OPENCV"
    resize: bool = False

    def main(self):
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path, exist_ok=True)

        image_path = str(self.image_path.resolve())
        use_gpu = 1 if self.use_gpu else 0
        colmap_command = "colmap"

        base_dir = str(Path(image_path))

        if not self.skip_matching:
            os.makedirs(base_dir + "/sparse", exist_ok=True)
            # Feature extraction
            feat_extracton_cmd = (
                f"{colmap_command} feature_extractor "
                f"--database_path {base_dir}/database.db "
                f"--image_path {image_path} "
                f"--ImageReader.single_camera 1 "
                f"--ImageReader.camera_model {self.camera} "
                f"--SiftExtraction.use_gpu {use_gpu}"
            )
            exit_code = os.system(feat_extracton_cmd)
            if exit_code != 0:
                logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
                exit(exit_code)

            # Feature matching
            feat_matching_cmd = (
                f"{colmap_command} exhaustive_matcher "
                f"--database_path {base_dir}/database.db "
                f"--SiftMatching.use_gpu {use_gpu}"
            )
            exit_code = os.system(feat_matching_cmd)
            if exit_code != 0:
                logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
                exit(exit_code)

            # Bundle adjustment
            mapper_cmd = (
                f"{colmap_command} mapper "
                f"--database_path {base_dir}/database.db "
                f"--image_path {image_path} "
                f"--output_path {base_dir}/sparse "
                f"--Mapper.ba_global_function_tolerance=0.000001 "
                f"--Mapper.min_num_matches 15 "
                f"--Mapper.multiple_models 0"
            )
            exit_code = os.system(mapper_cmd)
            if exit_code != 0:
                logging.error(f"Mapper failed with code {exit_code}. Exiting.")
                exit(exit_code)

        # Image undistortion
        if not self.skip_undistortion:
            img_undist_cmd = (
                f"{colmap_command} image_undistorter "
                f"--image_path {image_path} "
                f"--input_path {base_dir}/sparse/0 "
                f"--output_path {base_dir} "
                f"--output_type COLMAP"
            )
            exit_code = os.system(img_undist_cmd)
            if exit_code != 0:
                logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
                exit(exit_code)

        if self.resize:
            raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format")
    parser.add_argument("image_path", type=Path, help="Input to images folder")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU with COLMAP", default=True)
    parser.add_argument("--skip_matching", action="store_true", help="Skip matching")
    parser.add_argument("--skip_undistortion", action="store_true", help="Skip undistorting images")
    parser.add_argument("--camera", type=str, default="PINHOLE", help="Camera type")
    parser.add_argument("--resize", action="store_true", help="Resize images")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    converter = ConvertColmap(
        image_path=args.image_path,
        use_gpu=args.use_gpu,
        skip_matching=args.skip_matching,
        skip_undistortion=args.skip_undistortion,
        camera=args.camera,
        resize=args.resize
    )
    converter.main()
