#!/usr/bin/env python3
"""Generate unity_manifest.json for an existing GSCodec compression result."""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob


def infer_num_gop(compression_dir: str) -> int:
    metas = glob(os.path.join(compression_dir, "gop*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No gop*_meta.json files in {compression_dir}")
    ids = []
    for path in metas:
        match = re.search(r"gop(\d+)_meta\.json$", os.path.basename(path))
        if match:
            ids.append(int(match.group(1)))
    return max(ids) + 1


def infer_frame_num(compression_dir: str, gop_size: int, num_gop: int) -> int:
    meta_path = os.path.join(compression_dir, "gop0_meta.json")
    with open(meta_path) as fp:
        meta = json.load(fp)
    frames_in_gop0 = meta["means"]["shape"][0]
    if num_gop == 1:
        return frames_in_gop0
    return (num_gop - 1) * gop_size + frames_in_gop0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir",
        help="GSCodec result directory containing compression/",
    )
    parser.add_argument("--gop-size", type=int, default=16)
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument(
        "--n-orig-per-frame",
        type=int,
        nargs="*",
        default=None,
        help="Original splat counts per frame (before grid padding)",
    )
    args = parser.parse_args()

    compression_dir = os.path.join(args.result_dir, "compression")
    if not os.path.isdir(compression_dir):
        raise FileNotFoundError(f"Missing compression directory: {compression_dir}")

    num_gop = infer_num_gop(compression_dir)
    frame_num = args.frame_num or infer_frame_num(
        compression_dir, args.gop_size, num_gop
    )

    with open(os.path.join(compression_dir, "gop0_meta.json")) as fp:
        gop0_meta = json.load(fp)
    grid_h, grid_w = gop0_meta["means"]["shape"][1:3]
    n_padded = grid_h * grid_w

    if args.n_orig_per_frame:
        n_orig = list(args.n_orig_per_frame)
        if len(n_orig) != frame_num:
            raise ValueError(
                f"Expected {frame_num} n_orig values, got {len(n_orig)}"
            )
    else:
        n_orig = [n_padded] * frame_num
        print(
            f"Warning: n_orig_per_frame not provided; using padded count {n_padded} "
            "for every frame. Pass --n-orig-per-frame for accurate trimming."
        )

    manifest = {
        "version": 1,
        "compression": "seq_hevc",
        "gop_size": args.gop_size,
        "frame_num": frame_num,
        "num_gop": num_gop,
        "n_orig_per_frame": n_orig,
        "transform_attributes": {
            "means": False,
            "scales": False,
            "quats": False,
            "opacities": False,
            "sh0": False,
            "shN": False,
        },
        "use_sort": True,
        "use_all_intra": False,
        "quat_convention": "wxyz",
        "sh_degree": 3,
    }

    out_path = os.path.join(compression_dir, "unity_manifest.json")
    with open(out_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
