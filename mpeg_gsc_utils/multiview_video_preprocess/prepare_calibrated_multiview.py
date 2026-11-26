"""Prepare a calibrated multiview mp4 capture for run_per_frame_colmap.py.

Replaces steps 1-2 of the README (yuv -> png, COLMAP SfM for poses) when the
capture already comes with a calibration file (caliscope-style config.toml:
per-camera `matrix`, `distortions`, and OpenCV world->camera `rotation`
(Rodrigues) / `translation`).

Input:
    <video_dir>/port_<i>.mp4, <video_dir>/config.toml
Output (layout expected by run_per_frame_colmap.py):
    <base_dir>/colmap/poses_bounds.npy
    <base_dir>/colmap/colmap_<t>/input/v<i:02d>.png     t = 0 .. frame_num-1

All views are undistorted onto one shared pinhole camera (fx = fy = focal,
principal point at the image centre), because the dynamic-GS loader uses a
single intrinsic matrix taken from poses_bounds.npy.

After run_per_frame_colmap.py, run this script again with --thin_points_every N
to keep the triangulated points only in every N-th frame.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import toml
import tyro
from tqdm import trange


@dataclass
class Config:
    base_dir: str
    """Output scene folder; frames go to <base_dir>/colmap/colmap_<t>."""
    video_dir: Optional[str] = None
    """Folder with port_<i>.mp4 and config.toml."""
    start_frame: int = 0
    """First video frame to extract."""
    frame_num: int = 60
    """Number of frames to extract."""
    focal: Optional[float] = None
    """Shared focal length in pixels (default: mean fx of all cameras)."""
    near: float = 0.3
    far: float = 10.0
    """Scene depth bounds (metres) written to poses_bounds.npy."""
    thin_points_every: Optional[int] = None
    """Post-COLMAP mode: empty points3D.bin in all frames except every N-th."""


def load_cameras(config_path: Path):
    cfg = toml.load(config_path)
    cams = []
    for i in range(cfg["camera_count"]):
        c = cfg[f"cam_{i}"]
        R, _ = cv2.Rodrigues(np.array(c["rotation"], dtype=np.float64))
        cams.append({
            "port": c["port"],
            "size": tuple(c["size"]),
            "K": np.array(c["matrix"], dtype=np.float64),
            "dist": np.array(c["distortions"], dtype=np.float64),
            "R": R,
            "t": np.array(c["translation"], dtype=np.float64),
        })
    return cams


def write_poses_bounds(path: Path, cams, focal: float, near: float, far: float):
    rows = []
    for c in cams:
        w, h = c["size"]
        c2w = np.eye(4)
        c2w[:3, :3] = c["R"].T
        c2w[:3, 3] = -c["R"].T @ c["t"]
        pose = np.concatenate([c2w[:3, :4], np.array([[h], [w], [focal]])], axis=1)
        # LLFF column order [down, right, backwards]: inverse of run_per_frame_colmap.inversestep1
        pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:, 2:3], pose[:, 3:4], pose[:, 4:5]], axis=1)
        rows.append(np.concatenate([pose.ravel(), [near, far]]))
    np.save(path, np.array(rows))


def extract_frames(cfg: Config, cams, colmap_dir: Path, focal: float):
    w, h = cams[0]["size"]
    K_new = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    for i, c in enumerate(cams):
        assert c["size"] == (w, h), "all cameras must share one resolution"
        map1, map2 = cv2.initUndistortRectifyMap(c["K"], c["dist"], None, K_new, (w, h), cv2.CV_32FC1)
        cap = cv2.VideoCapture(os.path.join(cfg.video_dir, f"port_{c['port']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)
        for t in trange(cfg.frame_num, desc=f"cam {i}"):
            ok, img = cap.read()
            assert ok, f"port_{c['port']}.mp4 ended before frame {cfg.start_frame + t}"
            out_dir = colmap_dir / f"colmap_{t}" / "input"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"v{i:02d}.png"), cv2.remap(img, map1, map2, cv2.INTER_LANCZOS4))
        cap.release()


def thin_points(colmap_dir: Path, every: int):
    frames = sorted((p for p in colmap_dir.glob("colmap_*") if p.is_dir()), key=lambda p: int(p.name.split("_")[1]))
    kept = 0
    for p in frames:
        t = int(p.name.split("_")[1])
        bin_path = p / "sparse" / "0" / "points3D.bin"
        for cached in (p / "sparse" / "0").glob("points3D_total*.ply"):
            cached.unlink()
        if t % every == 0:
            with open(bin_path, "rb") as f:
                kept += int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        else:
            with open(bin_path, "wb") as f:
                f.write(np.uint64(0).tobytes())
    print(f"Kept points in every {every}-th frame ({kept} points in total)")


def main(cfg: Config):
    colmap_dir = Path(cfg.base_dir) / "colmap"
    if cfg.thin_points_every is not None:
        thin_points(colmap_dir, cfg.thin_points_every)
        return

    assert cfg.video_dir is not None, "--video_dir is required"
    cams = load_cameras(Path(cfg.video_dir) / "config.toml")
    focal = cfg.focal if cfg.focal is not None else float(np.mean([c["K"][0, 0] for c in cams]))
    colmap_dir.mkdir(parents=True, exist_ok=True)
    write_poses_bounds(colmap_dir / "poses_bounds.npy", cams, focal, cfg.near, cfg.far)
    extract_frames(cfg, cams, colmap_dir, focal)
    print(f"{len(cams)} cameras, {cfg.frame_num} frames from video frame {cfg.start_frame}, "
          f"shared focal {focal:.1f}px -> {colmap_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
