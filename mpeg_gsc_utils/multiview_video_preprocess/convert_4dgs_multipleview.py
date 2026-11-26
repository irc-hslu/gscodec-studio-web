"""Convert a 4DGaussians "MultipleView" export into the per-frame COLMAP layout
used by examples/simple_trainer_dyngs.py (datasets/INVR_N3D.py).

Input (e.g. 4DAnyone/data/4dgs/010_multicam):
    cam01/frame_00001.jpg ... camNN/frame_XXXXX.jpg
    sparse/cameras.bin, sparse/images.bin      (shared poses, names imageK.jpg)
    poses_bounds_multipleview.npy              (LLFF, one row per camera)
    points3D_multipleview.ply                  (SfM seed points)

Output:
    <out>/colmap/poses_bounds.npy
    <out>/colmap/colmap_<t>/images/imageK.jpg
    <out>/colmap/colmap_<t>/sparse/0/{cameras,images,points3D}.bin

imageK.jpg is camera folder camKK (matched by number). Cameras whose folder has been
removed from the export are dropped from images.bin and poses_bounds.npy.
"""

import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from plyfile import PlyData
from tqdm import tqdm


@dataclass
class Config:
    src: str
    """4DGaussians MultipleView scene folder (contains cam01/, sparse/, ...)."""
    out: str
    """Output scene folder; frames are written to <out>/colmap/colmap_<t>."""
    start_frame: int = 0
    """First frame index (0-based) to convert."""
    frames: Optional[int] = None
    """Number of frames to convert (default: all available)."""
    points_every: int = 30
    """Write the seed point cloud into every N-th frame (others get an empty
    points3D.bin). The trainer merges all frames' points, so this controls the
    initial number of Gaussians: (#frames / N) x #seed points. Stacking many
    copies of the same static cloud (e.g. N=5 with 30+ frame groups) stalls
    training; keep at most ~2 copies per group."""


def read_qt_names(images_bin: Path):
    """Return {name: (qvec, tvec, raw_record_bytes)} from a COLMAP images.bin."""
    out = {}
    with open(images_bin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            head = f.read(64)
            vals = struct.unpack("<idddddddi", head)
            q, t = np.array(vals[1:5]), np.array(vals[5:8])
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n2d_raw = f.read(8)
            n2d = struct.unpack("<Q", n2d_raw)[0]
            pts = f.read(24 * n2d)
            out[name.decode()] = (q, t, head + name + b"\x00" + n2d_raw + pts)
    return out


def write_images_bin(path: Path, records):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(records)))
        for rec in records:
            f.write(rec)


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def write_points3d_bin(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            f.write(struct.pack("<QdddBBBdQ", i + 1, *p, *c, 0.0, 0))


def main(cfg: Config):
    src, out = Path(cfg.src), Path(cfg.out) / "colmap"
    cam_dirs = {int(p.name[3:]): p for p in src.iterdir() if p.is_dir() and re.fullmatch(r"cam\d+", p.name)}
    poses_bounds = np.load(src / "poses_bounds_multipleview.npy")

    # imageK.jpg in images.bin <-> folder camKK <-> row K-1 of poses_bounds.
    qt = read_qt_names(src / "sparse" / "images.bin")
    name_to_cam, kept_rows, records = {}, [], []
    center_err = 0.0
    for name, (q, t, rec) in sorted(qt.items(), key=lambda kv: int(re.search(r"(\d+)", kv[0]).group(1))):
        k = int(re.search(r"(\d+)", name).group(1))
        center = -qvec2rotmat(q).T @ t
        center_err = max(center_err, np.abs(center - poses_bounds[k - 1, :15].reshape(3, 5)[:, 3]).max())
        if k not in cam_dirs:
            continue
        name_to_cam[name] = cam_dirs[k]
        kept_rows.append(k - 1)
        records.append(rec)
    print(f"{len(qt)} cameras in images.bin, max camera-centre mismatch vs poses_bounds: {center_err:.2e}")
    if center_err > 1e-3:
        print("[warn] images.bin order does not match poses_bounds; check the imageK <-> camKK mapping")
    dropped = sorted(set(qt) - set(name_to_cam), key=lambda n: int(re.search(r"(\d+)", n).group(1)))
    print(f"Using {len(name_to_cam)} cameras" + (f"; no folder for {dropped}, dropped" if dropped else ""))

    n_avail = min(len(list(d.glob("frame_*.jpg"))) for d in name_to_cam.values())
    n_frames = cfg.frames if cfg.frames is not None else n_avail - cfg.start_frame
    assert cfg.start_frame + n_frames <= n_avail, f"only {n_avail} frames available"

    ply = PlyData.read(str(src / "points3D_multipleview.ply"))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)
    rgb = np.stack([ply["red"], ply["green"], ply["blue"]], axis=1).astype(np.uint8)
    empty = np.zeros((0, 3))

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "poses_bounds.npy", poses_bounds[kept_rows])
    images_bin = out / "images.bin"
    write_images_bin(images_bin, records)

    for t in tqdm(range(cfg.start_frame, cfg.start_frame + n_frames), desc="frames"):
        frame_dir = out / f"colmap_{t}"
        img_dir, sparse_dir = frame_dir / "images", frame_dir / "sparse" / "0"
        img_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        for name, cam_dir in name_to_cam.items():
            shutil.copy2(cam_dir / f"frame_{t + 1:05d}.jpg", img_dir / name)
        shutil.copy2(src / "sparse" / "cameras.bin", sparse_dir / "cameras.bin")
        shutil.copy2(images_bin, sparse_dir / "images.bin")
        if t % cfg.points_every == 0:
            write_points3d_bin(sparse_dir / "points3D.bin", xyz, rgb)
        else:
            write_points3d_bin(sparse_dir / "points3D.bin", empty, empty)

    print(f"Wrote {n_frames} frames to {out} ({len(xyz)} seed points every {cfg.points_every} frames)")


if __name__ == "__main__":
    main(tyro.cli(Config))
