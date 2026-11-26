"""Render uncompressed LOD spiral videos (scene LOD + environment) and trim from 12s."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_encoded_results import Config, render_spiral_video, splats_to_device
from utils import concat_splats, load_ply

ITER_DIR = "../AbbeyStreet_Stairs/ply-result/all"
SPIRAL_CAMERAS = "eva/AbbeyStreet_Stairs/_eval_100views/spiral_video/cameras_spiral.json"
OUT_DIR = "eva/AbbeyStreet_Stairs/_eval_100views/spiral_video"
TRIM_START_S = 12.0


def load_spiral_c2w(path: str) -> tuple[np.ndarray, dict]:
    with open(path) as f:
        meta = json.load(f)
    c2w = np.asarray(meta["c2w"], dtype=np.float32)
    return c2w, meta


def merge_scene_env(scene_path: str, env_path: str) -> torch.nn.ParameterDict:
    scene = load_ply(scene_path)
    env = load_ply(env_path)
    merged = concat_splats([env, scene])
    print(
        f"  merged {os.path.basename(scene_path)} + environment: "
        f"{int(merged['means'].shape[0])} Gaussians "
        f"(scene={int(scene['means'].shape[0])}, env={int(env['means'].shape[0])})"
    )
    return merged


def trim_video(src: str, dst: str, start_s: float, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
            "-i",
            src,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            dst,
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    iter_dir = os.path.abspath(ITER_DIR)
    env_path = os.path.join(iter_dir, "environment.ply")
    spiral_path = os.path.abspath(SPIRAL_CAMERAS)
    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    spiral_c2w, cam_meta = load_spiral_c2w(spiral_path)
    cfg = Config(
        width=int(cam_meta.get("width", 1920)),
        height=int(cam_meta.get("height", 1080)),
        spiral_fps=int(cam_meta.get("fps", 25)),
        spiral_frames=spiral_c2w.shape[0],
        skip_existing=False,
        render_spiral_video=True,
    )
    if cam_meta.get("fx") is not None:
        cfg.fx = float(cam_meta["fx"])
    if cam_meta.get("fy") is not None:
        cfg.fy = float(cam_meta["fy"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Spiral: {spiral_c2w.shape[0]} frames @ {cfg.spiral_fps} fps, {cfg.width}x{cfg.height}")

    for lod in (1, 2):
        scene_path = os.path.join(iter_dir, f"point_cloud_{lod}.ply")
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(scene_path)

        print(f"\n=== LOD{lod} uncompressed ===")
        splats = merge_scene_env(scene_path, env_path)
        packed = splats_to_device(splats, device)
        del splats

        full_mp4 = os.path.join(out_dir, f"uncompressed_LOD{lod}_spiral.mp4")
        trimmed_mp4 = os.path.join(out_dir, f"uncompressed_LOD{lod}_spiral_trim12s.mp4")

        render_spiral_video(
            packed,
            spiral_c2w,
            cfg,
            full_mp4,
            desc=f"spiral uncompressed LOD{lod}",
        )
        del packed
        torch.cuda.empty_cache()

        print(f"Trimming from {TRIM_START_S}s -> {trimmed_mp4}")
        trim_video(full_mp4, trimmed_mp4, TRIM_START_S, cfg.spiral_fps)
        print(f"Done: {trimmed_mp4}")


if __name__ == "__main__":
    main()
