#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, cv2

def load_calib(fn):
    j = json.load(open(fn))
    K = np.array(j["K"]).reshape(3,3)
    P = np.array(j["P"]).reshape(3,4)
    return K, P

def disparity_to_depth(disp, f, B, min_disp=1e-3):
    """disp in pixels → depth in metres"""
    d = disp.copy()
    d[d < min_disp] = 0
    depth = np.zeros_like(d, dtype=np.float32)
    valid = d > 0
    depth[valid] = (f * B) / d[valid]
    return depth

def main(args):
    color_folder = Path(args.color_folder)
    disp_folder  = Path(args.disp_folder)
    out          = Path(args.out_root)
    Kc, _ = load_calib(args.color_calib)
    _,  Pr = load_calib(args.disp_calib)
    R_Kc,  R_Pr = load_calib(args.right_calib)

    f = Kc[0,0]
    B = abs(R_Pr[0,3]) / Pr[0,0]   # metres

    # Prepare TUM layout
    rgb_dir   = out / "rgb";   rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = out / "depth"; depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_lines   = []
    depth_lines = []

    # iterate by common timestamp (stem)
    # assumes .png in color, .npy in disp
    for color_png in sorted(color_folder.glob("*.png")):
        stem = color_png.stem            # e.g. "1627463923.345678"
        ts   = float(stem)

        # — copy color →
        dest_rgb = rgb_dir / f"{stem}.png"
        img      = cv2.imread(str(color_png), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(dest_rgb), img)

        # — load & convert disparity →
        disp_npy = disp_folder / f"{stem}.npy"
        if not disp_npy.exists():
            print(f"[warn] no disparity for {stem}")
            continue

        disp = np.load(disp_npy).astype(np.float32)
        # if your extractor DID NOT yet divide by 16, uncomment:
        # disp /= 16.0

        depth_m = disparity_to_depth(disp, f, B)
        depth_mm = (depth_m * 1000.0).astype(np.uint16)

        # — save TUM-style depth PNG →
        dest_depth = depth_dir / f"{stem}.png"
        cv2.imwrite(str(dest_depth), depth_mm)

        # — record lines →
        rgb_lines.append(  (ts, f"rgb/{stem}.png")   )
        depth_lines.append((ts, f"depth/{stem}.png") )

    # — write txt lists —
    def write_list(fn, lines):
        with open(fn, "w") as f:
            for t,p in lines:
                f.write(f"{t:.6f} {p}\n")

    write_list(out/"rgb.txt",        rgb_lines)
    write_list(out/"depth.txt",      depth_lines)

    # — associations: one-to-one by index —
    with open(out/"associations.txt", "w") as f:
        for (t1,p1), (t2,p2) in zip(rgb_lines, depth_lines):
            f.write(f"{t1:.6f} {p1} {t2:.6f} {p2}\n")

    print("TUM-RGBD dataset written to:", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--color-folder",  required=True,
                   help="path to aux (color) PNGs")
    p.add_argument("--disp-folder",   required=True,
                   help="path to disparity .npy files")
    p.add_argument("--color-calib",   required=True,
                   help="aux/calib.json")
    p.add_argument("--disp-calib",    required=True,
                   help="disparity/calib.json")
    p.add_argument("--right-calib",    required=True,
                   help="right/calib.json")
    p.add_argument("--out-root",      required=True,
                   help="where to write the TUM-RGBD structure")
    args = p.parse_args()
    main(args)
