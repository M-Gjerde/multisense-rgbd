#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Copy and rename numerically-sorted PNGs to 0..N-1.png"
    )
    p.add_argument("input_dir", type=Path, help="Folder containing source images (.png)")
    p.add_argument("output_dir", type=Path, help="Destination folder for renamed images")
    p.add_argument("--no-pad", action="store_true",
                   help="Do not zero-pad output filenames (default pads to digit width of N-1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be copied/renamed without writing files")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing files in output_dir if present")
    return p.parse_args()

def numeric_key(p: Path) -> int:
    """
    Extract a numeric key from filename stem. Requires fully numeric stems.
    Example: '1749749355844371557.png' -> 1749749355844371557
    """
    stem = p.stem
    if not stem.isdigit():
        raise ValueError(f"Filename is not purely numeric: {p.name}")
    return int(stem)

def main():
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    # Collect .png files (case-insensitive)
    pngs = sorted(
        [p for p in args.input_dir.iterdir()
         if p.is_file() and p.suffix.lower() == ".png"],
        key=numeric_key
    )
    if not pngs:
        raise SystemExit(f"No .png files found in: {args.input_dir}")

    # Prepare output dir
    if not args.output_dir.exists():
        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ok] Created: {args.output_dir}")
    elif not args.output_dir.is_dir():
        raise SystemExit(f"Output path exists and is not a directory: {args.output_dir}")

    count = len(pngs)
    pad_width = 0 if args.no_pad else len(str(count - 1 if count > 1 else 1))

    print(f"[info] Found {count} image(s).")
    print(f"[info] Zero-padding: {'off' if args.no_pad else f'{pad_width} digits'}")
    print(f"[info] Output dir: {args.output_dir}")

    # Copy & rename
    for i, src in enumerate(pngs):
        name = f"{i}.png" if args.no_pad else f"{i:0{pad_width}d}.png"
        dst = args.output_dir / name

        if dst.exists() and not args.overwrite:
            raise SystemExit(f"Destination file exists (use --overwrite): {dst}")

        print(f"{src.name}  ->  {name}")
        if not args.dry_run:
            # copy2 preserves timestamps/metadata
            shutil.copy2(src, dst)

    print("[done] All images copied and renamed.")

if __name__ == "__main__":
    main()
