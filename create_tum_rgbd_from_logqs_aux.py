#!/usr/bin/env python3
"""
Convert a stereo-disparity + aux-RGB recording into a TUM-RGBD-style
dataset, re-projecting the depth map from the left-camera frame into
the aux-camera frame.

Expected directory tree
├── aux/              RGB PNGs (24-bit colour)
│   └── calib.json
├── disparity/        .npy disparity maps (same resolution as left)
│   └── calib.json
├── right/            right-camera calib (for baseline only)
│   └── calib.json
└── (left/)           left-camera calib (usually identical to disparity)

Usage example
-------------
python3 make_tum_aux.py \
    --aux-folder    seq/aux        \
    --disp-folder   seq/disparity  \
    --aux-calib     seq/aux/calib.json \
    --left-calib    seq/left/calib.json \
    --disp-calib    seq/disparity/calib.json \
    --right-calib   seq/right/calib.json \
    --out-root      seq/tum_aux
"""
import argparse, json
from pathlib import Path
import numpy as np, cv2

# ---------------------------------------------------------------------- helpers
def load_calib(fn):
    """Return K (3×3) and P (3×4) from the json saved by the sensor."""
    j = json.load(open(fn))
    K = np.asarray(j["K"], dtype=np.float64).reshape(3, 3)
    P = np.asarray(j["P"], dtype=np.float64).reshape(3, 4)
    return K, P

def disparity_to_depth(disp_px, f, B, min_disp=1e-3):
    """Convert disparity (px) → depth (m)."""
    d = disp_px.astype(np.float32)
    d[d < min_disp] = 0
    depth = np.zeros_like(d, dtype=np.float32)
    mask = d > 0
    depth[mask] = (f * B) / d[mask]
    return depth

def reproject_depth_to_aux(depth_l, K_l, K_a, R_la, t_la, out_shape):
    """
    Re-project a left-frame depth map into aux camera pixels.
    • depth_l … (H_l×W_l) depth in metres, left camera frame
    • K_l     … 3×3 left intrinsics
    • K_a     … 3×3 aux  intrinsics
    • R_la,t_la … transform left→aux  (from calib)
    • out_shape … (H_a, W_a) of aux image
    Returns a float32 depth map (H_a×W_a) in metres.
    """
    H_l, W_l = depth_l.shape
    fy_l, fx_l = K_l[1, 1], K_l[0, 0]
    cy_l, cx_l = K_l[1, 2], K_l[0, 2]

    # pixel grid in the left image
    ys, xs = np.indices((H_l, W_l), dtype=np.float32)
    zs = depth_l
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    # un-project to metric 3-D in the left frame
    X_l = np.stack(((xs - cx_l) * zs / fx_l,
                    (ys - cy_l) * zs / fy_l,
                    zs), axis=1).T          # 3×N

    # transform to aux camera coordinates
    X_a = R_la @ X_l + t_la[:, None]        # 3×N

    # project on aux image plane
    fx_a, fy_a = K_a[0, 0], K_a[1, 1]
    cx_a, cy_a = K_a[0, 2], K_a[1, 2]
    us = (X_a[0] * fx_a) / X_a[2] + cx_a
    vs = (X_a[1] * fy_a) / X_a[2] + cy_a

    # round to nearest integer pixel
    us_i = np.round(us).astype(np.int32)
    vs_i = np.round(vs).astype(np.int32)

    H_a, W_a = out_shape
    good = (X_a[2] > 0) & (us_i >= 0) & (us_i < W_a) & (vs_i >= 0) & (vs_i < H_a)
    us_i, vs_i, zs_a = us_i[good], vs_i[good], X_a[2][good]

    # build aux-depth map, keep the closest point if several hit the same pixel
    depth_a = np.zeros(out_shape, dtype=np.float32)
    for u, v, z in zip(us_i, vs_i, zs_a):
        d_prev = depth_a[v, u]
        if d_prev == 0 or z < d_prev:       # take nearest surface
            depth_a[v, u] = z
    return depth_a

# -------------------------------------------------------------------------- main
def main(args):
    aux_folder  = Path(args.aux_folder)
    disp_folder = Path(args.disp_folder)
    out_root    = Path(args.out_root)

    # --- calibration ---
    K_l,  P_l  = load_calib(args.left_calib)
    K_a,  P_a  = load_calib(args.aux_calib)
    _,    P_r  = load_calib(args.right_calib)          # right for baseline
    _,    P_dl = load_calib(args.disp_calib)           # == P_l in most rigs

    # left → aux extrinsics
    M_a  = np.linalg.inv(K_a) @ P_a                    # 3×4  [R|t]
    R_la = M_a[:, :3]
    t_la = M_a[:, 3]

    # stereo baseline (metres) and focal length (pixels) for depth
    f_l = K_l[0, 0]
    B   = abs(P_r[0, 3]) / P_dl[0, 0]                  # |Tx| / fx_left

    # --- prepare TUM-style folders ---
    rgb_dir   = out_root / "rgb"
    depth_dir = out_root / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_lines, depth_lines = [], []

    # iterate over aux PNGs (assumes filenames like 1234567.890123.png)
    for rgb_png in sorted(aux_folder.glob("*.png")):
        stem = rgb_png.stem
        ts   = float(stem)

        # load aux image for its size and to copy into the dataset
        img_aux = cv2.imread(str(rgb_png), cv2.IMREAD_UNCHANGED)
        if img_aux is None:
            print(f"[warn] could not read {rgb_png}")
            continue
        H_a, W_a = img_aux.shape[:2]
        cv2.imwrite(str(rgb_dir / f"{stem}.png"), img_aux)

        # disparity → left-depth
        disp_npy = disp_folder / f"{stem}.npy"
        if not disp_npy.exists():
            print(f"[warn] no disparity for {stem}")
            continue
        disp = np.load(disp_npy).astype(np.float32)
        depth_l = disparity_to_depth(disp, f_l, B)

        # left-depth → aux-depth
        depth_a = reproject_depth_to_aux(depth_l, K_l, K_a, R_la, t_la, (H_a, W_a))
        depth_mm = (depth_a * 1000.0).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)

        # bookkeeping
        rgb_lines.append(  (ts, f"rgb/{stem}.png")   )
        depth_lines.append((ts, f"depth/{stem}.png") )

    # --- helper for txt file creation ---
    def write_list(path, lines):
        with open(path, "w") as f:
            for t, p in lines:
                f.write(f"{t:.6f} {p}\n")

    write_list(out_root / "rgb.txt",   rgb_lines)
    write_list(out_root / "depth.txt", depth_lines)

    with open(out_root / "associations.txt", "w") as f:
        for (t1, p1), (t2, p2) in zip(rgb_lines, depth_lines):
            f.write(f"{t1:.6f} {p1} {t2:.6f} {p2}\n")

    print("TUM-RGBD dataset written to", out_root.resolve())

# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aux-folder",    required=True, help="path to aux (RGB) PNGs")
    p.add_argument("--disp-folder",   required=True, help="path to disparity .npy files")
    p.add_argument("--aux-calib",     required=True, help="aux/calib.json")
    p.add_argument("--left-calib",    required=True, help="left/calib.json (intrinsics/extrinsics)")
    p.add_argument("--disp-calib",    required=True, help="disparity/calib.json (for baseline fx)")
    p.add_argument("--right-calib",   required=True, help="right/calib.json (for baseline Tx)")
    p.add_argument("--out-root",      required=True, help="output directory for TUM-RGBD structure")
    args = p.parse_args()
    main(args)
