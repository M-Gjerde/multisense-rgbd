#!/usr/bin/env python3
"""
Extract all image & disparity streams + their calibrations from a ROS bag
into a dataset folder structure:

dataset/
├── left/          (RGB or mono)
│   ├── <stamp>.png
│   └── calib.json
├── right/
├── aux/
├── disparity/     (float disparity .npy or uint16 png)
└── unknown_<topic_hash>/
"""

from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from PIL import Image
import numpy as np, json, re, hashlib, argparse
from tqdm import tqdm

# ---------------------------------------------------------------------------

# --- classification heuristics ­---------------------------------------------------
def classify(topic: str) -> str:
    """Return folder name for an image topic."""
    t = topic.lower()
    if "disparity" in t:
        return "disparity"
    if "left"  in t:
        return "left"
    if "right" in t:
        return "right"
    if "aux"   in t or "rgb" in t or "color" in t:
        return "aux"
    # Fall-back based on hash so multiple unknown topics don't clash
    return f"unknown_{hashlib.md5(topic.encode()).hexdigest()[:6]}"

# --- save helpers ­----------------------------------------------------------------
def ensure(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    return folder

def _to_builtin(x):
    """convert numpy → builtin so json.dumps works"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):        # numpy scalar
        return x.item()
    return x                                # already builtin

def save_calib(msg, out_path):
    if not hasattr(msg, "K"):               # guard: only CameraInfo allowed
        print(f"[warn] not a CameraInfo message → {out_path} skipped")
        return

    data = {k: _to_builtin(getattr(msg, k)) for k in ("K", "D", "R", "P")}
    data["width"]  = int(msg.width)
    data["height"] = int(msg.height)

    out_path.write_text(json.dumps(data, indent=2))
    print(f"saved calibration → {out_path}")

# --- main ­------------------------------------------------------------------------
def main(bagfile: Path, out_root: Path):
    typestore = get_typestore(Stores.ROS2_FOXY)
    ensure(out_root)

    with AnyReader([bagfile], default_typestore=typestore) as reader:
        # all connections
        conns = {c.topic: c for c in reader.connections}

        # image producers
        is_image = lambda c: c.msgtype == "sensor_msgs/msg/Image"
        img_conns = [c for c in conns.values() if is_image(c)]

        # all CameraInfo streams
        caminfo_conns = [c for c in conns.values()
                         if c.msgtype == "sensor_msgs/msg/CameraInfo"]

        folders = {}

        # 1) first, make folders & save every CameraInfo you find
        for info_c in caminfo_conns:
            # classify by topic name
            name = classify(info_c.topic)
            folder = ensure(out_root / bagfile.name.split(".")[0] / name)
            folders[info_c.topic] = folder

            # grab the first CameraInfo message
            conn, ts, raw = next(reader.messages(connections=[info_c]))
            msg = reader.deserialize(raw, conn.msgtype)
            save_calib(msg, folder / "calib.json")

        # 2) now make sure every image topic still ends up in a folder
        for img_c in img_conns:
            name = classify(img_c.topic)
            folders.setdefault(img_c.topic, ensure(out_root / bagfile.name.split(".")[0]/ name))

        # 3) extract images & disparity as before, using folders[img_c.topic]
        for conn, ts, raw in tqdm(reader.messages(connections=img_conns),
                                  desc="Extracting images…"):
            folder = folders[conn.topic]
            stamp  = f"{ts}.png"   # default name

            if conn.msgtype == "sensor_msgs/msg/Image":
                msg = reader.deserialize(raw, conn.msgtype)
                mode = "RGB" if msg.encoding in ("rgb8","bgr8","bgr8") else "L"
                if "bgr" in msg.encoding.lower():
                    img = Image.frombuffer(mode, (msg.width,msg.height),
                                           msg.data, "raw", "BGR", 0, 1)
                    img.save(folder / stamp)
                elif "disparity" in conn.topic:
                    # Deserialize and interpret raw data as uint16
                    disp_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

                    # Convert to float32 disparity
                    disp_float = disp_raw.astype(np.float32) / 16.0
                    np.save(folder / f"{ts}.npy", disp_float)

                    # For preview: map [0, 256) → [0, 255] and save as 8-bit PNG
                    disp_vis = np.clip(disp_float * (255.0 / 256.0), 0, 255).astype(np.uint8)
                    Image.fromarray(disp_vis).save(folder / f"{ts}.png")
                else:
                    img = Image.frombuffer(mode, (msg.width,msg.height),
                                           msg.data, "raw", mode, 0, 1)
                    img.save(folder / stamp)



    print(f"Finished.  Output in → {out_root}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=Path, help="ROS1/ROS2 bag file")
    ap.add_argument("out", type=Path, help="output dataset root folder")
    args = ap.parse_args()
    main(args.bag, args.out)
