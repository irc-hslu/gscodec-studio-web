"""Render videos of a compressed dynamic GS PLY sequence (compress_ply_sequence.py output).

Decodes ``<result_dir>/compression`` and renders, from the COLMAP cameras of the capture:
  * ``compressed_<cam>.mp4``              compressed sequence from a fixed camera
  * ``original_vs_compressed_<cam>.mp4``  original | compressed side by side
  * ``compressed_sweep.mp4``              camera sweeps through all capture cameras (or orbits them,
                                          ``--sweep-mode orbit``) while time plays
and writes per-frame PSNR (compressed vs original render, and vs ground-truth image) to ``metrics.json``.

Run from ``examples/``:

    python render_compressed_sequence_video.py \\
      --result-dir results/dyngs/Cagliari_seated_3300_v2_x265_rp3 \\
      --orig-ply-dir results/dyngs/Cagliari_seated_3300_v2/plys_all \\
      --scene-dir data/Cagliari/seated_3300/colmap
"""

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torch import Tensor
from tqdm import tqdm

from decode_and_view import _activate_for_render, decode_sequence
from gsplat.rendering import rasterization
from utils import load_ply

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mpeg_gsc_utils"))
from colmap_read_model import qvec2rotmat, read_cameras_binary, read_images_binary  # noqa: E402


@dataclass
class Config:
    # compress_ply_sequence.py result folder (contains compression/)
    result_dir: str = "results/dyngs/Cagliari_seated_3300_v2_x265_rp3"
    # Uncompressed PLY sequence that was fed to the encoder
    orig_ply_dir: str = "results/dyngs/Cagliari_seated_3300_v2/plys_all"
    # Folder with colmap_<t>/sparse/0 + images/ per frame
    scene_dir: str = "data/Cagliari/seated_3300/colmap"
    # Fixed camera (image name without extension) for the fixed-view videos
    camera: str = "v00"
    # Output scale relative to the capture resolution
    scale: float = 0.5
    fps: int = 30
    # Times the sequence is looped in the sweep video (camera does one round trip overall)
    sweep_loops: int = 2
    # "cameras": round trip through the capture cameras (front-facing arcs);
    # "orbit": smooth 360 deg look-at orbit around the cameras' convergence point (surrounding rigs)
    sweep_mode: str = "cameras"
    # Composite on white (for models trained with --white-background)
    white_background: bool = False
    output_dir: str = ""


def load_cameras(scene_dir: str) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], int, int]:
    sparse = os.path.join(scene_dir, "colmap_0", "sparse", "0")
    images = read_images_binary(os.path.join(sparse, "images.bin"))
    cams = read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    out = {}
    for im in images.values():
        c = cams[im.camera_id]
        fx, fy, cx, cy = c.params[:4]
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(im.qvec)
        w2c[:3, 3] = im.tvec
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        out[os.path.splitext(im.name)[0]] = (w2c, K)
        W, H = c.width, c.height
    return out, W, H


def slerp_pose(c2w_a: np.ndarray, c2w_b: np.ndarray, t: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    rots = Rotation.from_matrix(np.stack([c2w_a[:3, :3], c2w_b[:3, :3]]))
    out = np.eye(4)
    out[:3, :3] = Slerp([0, 1], rots)([t]).as_matrix()[0]
    out[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    return out


def sweep_path(cameras: Dict[str, Tuple[np.ndarray, np.ndarray]], n: int) -> List[np.ndarray]:
    """Round trip through the capture cameras ordered by azimuth around their mean centre."""
    c2ws = {k: np.linalg.inv(w2c) for k, (w2c, _) in cameras.items()}
    centres = np.stack([c[:3, 3] for c in c2ws.values()])
    mean = centres.mean(0)
    up = -np.mean([c[:3, 1] for c in c2ws.values()], axis=0)
    up /= np.linalg.norm(up)
    ref = centres[0] - mean
    ref -= ref.dot(up) * up
    ref /= np.linalg.norm(ref)
    side = np.cross(up, ref)
    az = {k: np.arctan2((c[:3, 3] - mean).dot(side), (c[:3, 3] - mean).dot(ref)) for k, c in c2ws.items()}
    order = sorted(az, key=az.get)
    keys = order + order[-2:0:-1]
    seg = len(keys)
    poses = []
    for i in range(n):
        s = i / n * seg
        a, t = int(s) % seg, s - int(s)
        b = (a + 1) % seg
        poses.append(np.linalg.inv(slerp_pose(c2ws[keys[a]], c2ws[keys[b]], t)))
    print("Sweep camera order:", " -> ".join(order))
    return poses


def orbit_path(cameras: Dict[str, Tuple[np.ndarray, np.ndarray]], n: int) -> List[np.ndarray]:
    """One 360 deg orbit around the point closest to all optical axes; elevation sweeps the rig's range."""
    c2ws = [np.linalg.inv(w2c) for w2c, _ in cameras.values()]
    A, b = np.zeros((3, 3)), np.zeros(3)
    for c in c2ws:
        P = np.eye(3) - np.outer(c[:3, 2], c[:3, 2])
        A += P
        b += P @ c[:3, 3]
    target = np.linalg.solve(A, b)
    up = -np.mean([c[:3, 1] for c in c2ws], axis=0)
    up /= np.linalg.norm(up)
    v = np.stack([c[:3, 3] for c in c2ws]) - target
    radius = np.linalg.norm(v, axis=1)
    elev = np.arcsin(v @ up / radius)
    ref = v[0] - v[0].dot(up) * up
    ref /= np.linalg.norm(ref)
    side = np.cross(up, ref)
    poses = []
    for i in range(n):
        az = 2 * np.pi * i / n
        el = (elev.max() + elev.min()) / 2 + (elev.max() - elev.min()) / 2 * np.sin(az)
        pos = target + radius.mean() * (np.cos(el) * (np.cos(az) * ref + np.sin(az) * side) + np.sin(el) * up)
        z = target - pos
        z /= np.linalg.norm(z)
        x = np.cross(-up, z)
        x /= np.linalg.norm(x)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
        c2w[:3, 3] = pos
        poses.append(np.linalg.inv(c2w))
    print(f"Orbit around {np.round(target, 3)}, radius {radius.mean():.2f}, "
          f"elevation {np.degrees(elev.min()):.0f}..{np.degrees(elev.max()):.0f} deg")
    return poses


@torch.no_grad()
def render(splat: Dict[str, Tensor], w2c: np.ndarray, K: np.ndarray, W: int, H: int,
           white_background: bool = False) -> np.ndarray:
    dev = splat["means"].device
    img, _, _ = rasterization(
        splat["means"], splat["quats"], splat["scales"], splat["opacities"], splat["colors"],
        torch.tensor(w2c, dtype=torch.float32, device=dev)[None],
        torch.tensor(K, dtype=torch.float32, device=dev)[None],
        W, H, sh_degree=splat["sh_degree"],
        backgrounds=torch.ones(1, 3, device=dev) if white_background else None,
    )
    return (img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255**2 / max(mse, 1e-10)))


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def writer(path: str, fps: int):
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=None,
                              ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "medium"])


def main(cfg: Config) -> None:
    out_dir = cfg.output_dir or os.path.join(cfg.result_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")

    decoded = decode_sequence(os.path.join(cfg.result_dir, "compression"), keep_padding=False)
    orig_paths = sorted(glob.glob(os.path.join(cfg.orig_ply_dir, "*.ply")))[: len(decoded)]
    n = len(decoded)

    cameras, W0, H0 = load_cameras(cfg.scene_dir)
    W, H = int(round(W0 * cfg.scale)), int(round(H0 * cfg.scale))
    S = np.diag([cfg.scale, cfg.scale, 1.0])
    w2c_fixed, K_fixed = cameras[cfg.camera]
    K = S @ K_fixed

    stats = {"frames": {}, "camera": cfg.camera}
    rate = sum(os.path.getsize(p) for p in glob.glob(os.path.join(cfg.result_dir, "compression", "*")) if os.path.isfile(p))
    stats["compressed_bytes"] = rate
    stats["original_bytes"] = sum(os.path.getsize(p) for p in orig_paths)
    stats["bitrate_mbps_at_30fps"] = rate * 8 / n * 30 / 1e6

    w_comp = writer(os.path.join(out_dir, f"compressed_{cfg.camera}.mp4"), cfg.fps)
    w_side = writer(os.path.join(out_dir, f"original_vs_compressed_{cfg.camera}.mp4"), cfg.fps)
    for i in tqdm(range(n), desc="fixed-camera videos"):
        comp = _activate_for_render(decoded[i], device)
        orig = _activate_for_render(dict(load_ply(orig_paths[i]).items()), device)
        r_c = render(comp, w2c_fixed, K, W, H, cfg.white_background)
        r_o = render(orig, w2c_fixed, K, W, H, cfg.white_background)
        gt_paths = glob.glob(os.path.join(cfg.scene_dir, f"colmap_{i}", "images", f"{cfg.camera}.*"))
        entry = {"psnr_comp_vs_orig": psnr(r_c, r_o), "n_gs": int(comp["means"].shape[0])}
        if gt_paths:
            gt = cv2.resize(imageio.imread(gt_paths[0])[..., :3], (W, H), interpolation=cv2.INTER_AREA)
            entry["psnr_orig_vs_gt"] = psnr(r_o, gt)
            entry["psnr_comp_vs_gt"] = psnr(r_c, gt)
        stats["frames"][f"{i:03d}"] = entry
        w_comp.append_data(r_c)
        w_side.append_data(np.concatenate([label(r_o, f"original  frame {i:03d}"),
                                           label(r_c, "x265 rp3 compressed")], axis=1))
    w_comp.close()
    w_side.close()

    path = orbit_path if cfg.sweep_mode == "orbit" else sweep_path
    poses = path(cameras, n * cfg.sweep_loops)
    w_sweep = writer(os.path.join(out_dir, "compressed_sweep.mp4"), cfg.fps)
    for j, w2c in enumerate(tqdm(poses, desc="sweep video")):
        comp = _activate_for_render(decoded[j % n], device)
        w_sweep.append_data(render(comp, w2c, K, W, H, cfg.white_background))
    w_sweep.close()

    frames = stats["frames"].values()
    stats["average"] = {k: float(np.mean([f[k] for f in frames if k in f]))
                        for k in ["psnr_comp_vs_orig", "psnr_orig_vs_gt", "psnr_comp_vs_gt", "n_gs"]}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fp:
        json.dump(stats, fp, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "frames"}, indent=2))
    print(f"Videos written to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Config))
