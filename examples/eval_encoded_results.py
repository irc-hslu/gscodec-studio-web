"""Render encoded GSCodec MP4 results vs a baseline PLY on a shared view set.

Cameras: concentric horizontal circles around a look-at center (default: baseline
Gaussian centroid; override with ``--center-from-ply`` so a merged env+scene
baseline still orbits the abbey / scene shell only).

Optional spiral video: continuous helix from camera 0 (innermost) to the last
camera (outermost), looking at center — written as MP4 for baseline + each result.

No COLMAP required. Discovers every ``*/compression`` under ``--results-root``,
decodes frame 0, renders the same camera list used for the baseline, and writes
PSNR / SSIM / LPIPS.

Run from ``examples/``:

    python eval_encoded_results.py
    python eval_encoded_results.py --num-views 100 --views-per-ring 10 --radius-step-m 1
    python eval_encoded_results.py --render-spiral-video --spiral-frames 180 --spiral-rots 2
    python eval_encoded_results.py --spiral-video-only --render-spiral-video
    python eval_encoded_results.py --merge-environment-baseline \\
      --center-from-ply ../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply \\
      --num-views 104 --views-per-ring 8 --render-spiral-video
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from datasets.traj import viewmatrix
from decode_and_view import decode_sequence
from gsplat.rendering import rasterization
from utils import load_ply


@dataclass
class Config:
    results_root: str = "results"
    baseline_ply: str = (
        "../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply"
    )
    # Orbit/look-at center from this PLY's Gaussian centroid (not the baseline).
    # Use scene point_cloud.ply when baseline merges environment.ply.
    center_from_ply: Optional[str] = None
    # If True and environment.ply sits next to baseline_ply, concat both for renders.
    merge_environment_baseline: bool = False
    # Shared output for cameras + baseline renders + summary.
    output_dir: str = "eva/_eval_100views"
    num_views: int = 100
    width: int = 960
    height: int = 540
    fx: Optional[float] = None
    fy: Optional[float] = None
    # Concentric circles around scene center: views_per_ring on each ring,
    # radius = (ring_index + 1) * radius_step_m meters (ring 0 starts 1m out).
    views_per_ring: int = 10
    radius_step_m: float = 2.0
    # Camera height offset above the Gaussian centroid (meters).
    height_offset_m: float = 0.0
    # Extra height added per ring (every views_per_ring cameras).
    ring_height_step_m: float = 0.2
    # Pitch down around camera local X (degrees); positive = look down.
    pitch_down_deg: float = 15.0
    seed: int = 0
    frame_index: int = 0
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Skip far-away Gaussians (pixels); 0 disables.
    radius_clip: float = 0.0
    # Re-use existing baseline renders / skip results that already have metrics.
    skip_existing: bool = True
    # Optional subset, e.g. "MellifontAbbey_SH_1_rp0 MellifontAbbey_SH_1_rp1"
    only: Optional[List[str]] = None
    # Decode keeps encoder pads if True (usually False).
    keep_padding: bool = False
    # Recompute metrics from existing PNGs only (no decode/render).
    metrics_only: bool = False
    # Render spiral fly-through video (camera 0 → last camera).
    render_spiral_video: bool = True
    # Skip PSNR/SSIM/LPIPS; only decode + spiral videos.
    spiral_video_only: bool = False
    spiral_frames: int = 180
    # Full revolutions from first to last camera along the helix.
    spiral_rots: float = 2.0
    spiral_fps: int = 25
    # Where spiral videos land under output_dir / each result dir.
    spiral_subdir: str = "spiral_video"


def _intrinsics(cfg: Config) -> Tuple[float, float]:
    fx = cfg.fx if cfg.fx is not None else float(cfg.height)
    fy = cfg.fy if cfg.fy is not None else float(cfg.height)
    return fx, fy


def scene_center_from_means(means: torch.Tensor) -> np.ndarray:
    """Centroid of Gaussian means (baseline scene center)."""
    return means.detach().float().mean(dim=0).cpu().numpy().astype(np.float64)


def generate_concentric_c2w(
    means: torch.Tensor,
    num_views: int,
    views_per_ring: int,
    radius_step_m: float,
    height_offset_m: float,
    ring_height_step_m: float = 0.2,
    pitch_down_deg: float = 30.0,
    center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Concentric circles around the GS center, looking inward and pitched down.

    Ring ``r`` (0-based) has radius ``(r + 1) * radius_step_m`` meters and height
    ``height_offset_m + r * ring_height_step_m``. Each pose is pitched down by
    ``pitch_down_deg`` around the camera local X axis.

    If ``center`` is provided it overrides the centroid of ``means`` (useful when
    the rendered cloud includes a distant environment shell).
    """
    if views_per_ring <= 0:
        raise ValueError("views_per_ring must be > 0")
    if num_views % views_per_ring != 0:
        raise ValueError(
            f"num_views ({num_views}) must be divisible by views_per_ring ({views_per_ring})"
        )
    num_rings = num_views // views_per_ring
    if center is None:
        center = scene_center_from_means(means)
    else:
        center = np.asarray(center, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Local-X pitch: with Y = up / Z = forward, negative angle looks down.
    pitch = -math.radians(pitch_down_deg)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    rx_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, c_p, -s_p], [0.0, s_p, c_p]],
        dtype=np.float64,
    )

    poses = []
    ring_radii = []
    ring_heights = []
    for ring in range(num_rings):
        radius = float((ring + 1) * radius_step_m)
        height = float(height_offset_m + ring * ring_height_step_m)
        ring_radii.append(radius)
        ring_heights.append(height)
        for i in range(views_per_ring):
            theta = 2.0 * math.pi * i / views_per_ring
            pos = center + np.array(
                [
                    radius * math.cos(theta),
                    radius * math.sin(theta),
                    height,
                ],
                dtype=np.float64,
            )
            # Look toward center, then flip vertical, then pitch down about local X.
            lookdir = center - pos
            c2w3x4 = viewmatrix(lookdir, up, pos)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :4] = c2w3x4.astype(np.float32)
            c2w[:3, 0] *= -1.0
            c2w[:3, 1] *= -1.0
            c2w[:3, :3] = (c2w[:3, :3].astype(np.float64) @ rx_pitch).astype(np.float32)
            poses.append(c2w)

    meta = {
        "center": center.tolist(),
        "views_per_ring": views_per_ring,
        "num_rings": num_rings,
        "radius_step_m": radius_step_m,
        "ring_radii_m": ring_radii,
        "height_offset_m": height_offset_m,
        "ring_height_step_m": ring_height_step_m,
        "ring_heights_m": ring_heights,
        "pitch_down_deg": pitch_down_deg,
    }
    return np.stack(poses, axis=0), meta


def resolve_orbit_center(
    baseline_means: torch.Tensor,
    center_from_ply: Optional[str],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return orbit center; prefer ``center_from_ply`` centroid when set."""
    baseline_center = scene_center_from_means(baseline_means)
    info: Dict[str, Any] = {
        "baseline_center": baseline_center.tolist(),
        "center_from_ply": None,
    }
    if not center_from_ply:
        return baseline_center, info

    path = os.path.abspath(center_from_ply)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--center-from-ply not found: {path}")
    center_splats = load_ply(path)
    center = scene_center_from_means(center_splats["means"])
    info["center_from_ply"] = path
    info["center_ply_gaussians"] = int(center_splats["means"].shape[0])
    info["center"] = center.tolist()
    delta = center - baseline_center
    print(
        f"Orbit center from {path}: {center.tolist()} "
        f"(n={info['center_ply_gaussians']}); "
        f"baseline centroid={baseline_center.tolist()}; "
        f"delta={delta.tolist()}"
    )
    del center_splats
    return center, info


def maybe_merge_environment_baseline(
    baseline: "torch.nn.ParameterDict",
    baseline_ply: str,
    merge: bool,
) -> "torch.nn.ParameterDict":
    """Optionally concat sibling environment.ply onto the baseline cloud."""
    if not merge:
        return baseline
    env_path = os.path.join(os.path.dirname(baseline_ply), "environment.ply")
    if not os.path.isfile(env_path):
        print(f"Warning: --merge-environment-baseline set but no {env_path}; skipping.")
        return baseline
    from utils import concat_splats

    print(f"Merging environment into baseline: {env_path}")
    env = load_ply(env_path)
    merged = concat_splats([env, baseline])
    print(
        f"Baseline after merge: {int(merged['means'].shape[0])} Gaussians "
        f"(scene={int(baseline['means'].shape[0])}, env={int(env['means'].shape[0])})"
    )
    return merged


def generate_spiral_c2w_from_cameras(
    c2w: np.ndarray,
    n_frames: int,
    n_rots: float,
    pitch_down_deg: float,
    center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Helix from camera 0 to the last camera, looking at ``center``.

    Radius and height interpolate linearly between the first and last pose while
    the azimuth sweeps ``n_rots`` revolutions. Orientation matches the concentric
    path (flip vertical + pitch down about local X).
    """
    if c2w.shape[0] < 2:
        raise ValueError("Need at least 2 cameras to build a spiral from first to last.")
    if n_frames < 2:
        raise ValueError("spiral_frames must be >= 2")

    pos0 = c2w[0, :3, 3].astype(np.float64)
    pos1 = c2w[-1, :3, 3].astype(np.float64)
    if center is None:
        # Concentric rings: mean XY of all cameras is the look-at; keep Z from cam0 look.
        xy = c2w[:, :3, 3].astype(np.float64).mean(axis=0)
        # Look target ≈ position + forward * ||pos - xy_mean||
        fwd0 = -c2w[0, :3, 2].astype(np.float64)
        center = xy.copy()
        # Project cam0 onto the horizontal plane through xy, then step along look.
        dist = float(np.linalg.norm(pos0[:2] - xy[:2]))
        center = pos0 + fwd0 * dist
        center[:2] = xy[:2]
    else:
        center = np.asarray(center, dtype=np.float64).reshape(3)

    r0 = float(np.linalg.norm(pos0[:2] - center[:2]))
    r1 = float(np.linalg.norm(pos1[:2] - center[:2]))
    h0 = float(pos0[2])
    h1 = float(pos1[2])
    theta0 = math.atan2(pos0[1] - center[1], pos0[0] - center[0])
    theta1 = math.atan2(pos1[1] - center[1], pos1[0] - center[0])
    # Sweep n_rots full turns, then land on the last camera's azimuth.
    dtheta = (theta1 - theta0) + 2.0 * math.pi * n_rots

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pitch = -math.radians(pitch_down_deg)
    c_p, s_p = math.cos(pitch), math.sin(pitch)
    rx_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, c_p, -s_p], [0.0, s_p, c_p]],
        dtype=np.float64,
    )

    poses = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        radius = r0 + (r1 - r0) * t
        height = h0 + (h1 - h0) * t
        theta = theta0 + dtheta * t
        pos = np.array(
            [
                center[0] + radius * math.cos(theta),
                center[1] + radius * math.sin(theta),
                height,
            ],
            dtype=np.float64,
        )
        lookdir = center - pos
        c2w3x4 = viewmatrix(lookdir, up, pos)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :4] = c2w3x4.astype(np.float32)
        pose[:3, 0] *= -1.0
        pose[:3, 1] *= -1.0
        pose[:3, :3] = (pose[:3, :3].astype(np.float64) @ rx_pitch).astype(np.float32)
        poses.append(pose)

    meta = {
        "spiral_frames": n_frames,
        "spiral_rots": n_rots,
        "radius_start_m": r0,
        "radius_end_m": r1,
        "height_start_m": h0,
        "height_end_m": h1,
        "center": center.tolist(),
        "pitch_down_deg": pitch_down_deg,
        "from_camera": 0,
        "to_camera": int(c2w.shape[0] - 1),
    }
    return np.stack(poses, axis=0), meta


def save_cameras(
    path: str, c2w: np.ndarray, cfg: Config, path_meta: Optional[Dict[str, Any]] = None
) -> None:
    fx, fy = _intrinsics(cfg)
    payload = {
        "num_views": int(c2w.shape[0]),
        "width": cfg.width,
        "height": cfg.height,
        "fx": fx,
        "fy": fy,
        "cx": cfg.width * 0.5,
        "cy": cfg.height * 0.5,
        "views_per_ring": cfg.views_per_ring,
        "radius_step_m": cfg.radius_step_m,
        "height_offset_m": cfg.height_offset_m,
        "ring_height_step_m": cfg.ring_height_step_m,
        "pitch_down_deg": cfg.pitch_down_deg,
        "seed": cfg.seed,
        "c2w": c2w.tolist(),
    }
    if path_meta:
        payload.update(path_meta)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump(payload, fp)
    print(f"Wrote cameras: {path}")


def load_cameras(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(path, "r") as fp:
        payload = json.load(fp)
    return np.asarray(payload["c2w"], dtype=np.float32), payload


def discover_results(results_root: str, only: Optional[List[str]]) -> List[str]:
    names = []
    for name in sorted(os.listdir(results_root)):
        if name.startswith("_"):
            continue
        compress = os.path.join(results_root, name, "compression")
        if os.path.isdir(compress) and os.path.isfile(
            os.path.join(compress, "gop0_meta.json")
        ):
            if only is not None and name not in only:
                continue
            names.append(name)
    return names


def splats_to_device(
    splat: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, Any]:
    means = splat["means"].to(device)
    quats = F.normalize(splat["quats"].to(device), p=2, dim=-1)
    scales = torch.exp(splat["scales"].to(device))
    opacities = torch.sigmoid(splat["opacities"].to(device))
    sh0 = splat["sh0"].to(device)
    shN = splat["shN"].to(device)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(max(colors.shape[-2], 1)) - 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": max(sh_degree, 0),
    }


@torch.no_grad()
def render_views(
    packed: Dict[str, Any],
    c2w: np.ndarray,
    cfg: Config,
    out_dir: str,
    desc: str,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    device = packed["means"].device
    fx, fy = _intrinsics(cfg)
    K = torch.tensor(
        [[fx, 0.0, cfg.width * 0.5], [0.0, fy, cfg.height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    paths = []
    kwargs = {}
    if cfg.radius_clip > 0:
        kwargs["radius_clip"] = cfg.radius_clip

    for i in tqdm(range(c2w.shape[0]), desc=desc):
        path = os.path.join(out_dir, f"view_{i:03d}.png")
        paths.append(path)
        if cfg.skip_existing and os.path.isfile(path):
            continue
        cam = torch.from_numpy(c2w[i]).float().to(device)
        viewmat = torch.linalg.inv(cam)[None]
        colors, _, _ = rasterization(
            packed["means"],
            packed["quats"],
            packed["scales"],
            packed["opacities"],
            packed["colors"],
            viewmat,
            K[None],
            cfg.width,
            cfg.height,
            sh_degree=packed["sh_degree"],
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            render_mode="RGB",
            packed=False,
            **kwargs,
        )
        rgb = torch.clamp(colors[0, ..., :3], 0.0, 1.0).cpu().numpy()
        imageio.imwrite(path, (rgb * 255.0 + 0.5).astype(np.uint8))
    return paths


@torch.no_grad()
def render_spiral_video(
    packed: Dict[str, Any],
    spiral_c2w: np.ndarray,
    cfg: Config,
    video_path: str,
    desc: str,
) -> str:
    """Render spiral poses and write an MP4 (libx264 via imageio-ffmpeg)."""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    if cfg.skip_existing and os.path.isfile(video_path):
        print(f"[skip] spiral video exists: {video_path}")
        return video_path

    device = packed["means"].device
    fx, fy = _intrinsics(cfg)
    K = torch.tensor(
        [[fx, 0.0, cfg.width * 0.5], [0.0, fy, cfg.height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    kwargs = {}
    if cfg.radius_clip > 0:
        kwargs["radius_clip"] = cfg.radius_clip

    writer = imageio.get_writer(
        video_path,
        fps=cfg.spiral_fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    try:
        for i in tqdm(range(spiral_c2w.shape[0]), desc=desc):
            cam = torch.from_numpy(spiral_c2w[i]).float().to(device)
            viewmat = torch.linalg.inv(cam)[None]
            colors, _, _ = rasterization(
                packed["means"],
                packed["quats"],
                packed["scales"],
                packed["opacities"],
                packed["colors"],
                viewmat,
                K[None],
                cfg.width,
                cfg.height,
                sh_degree=packed["sh_degree"],
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                packed=False,
                **kwargs,
            )
            rgb = torch.clamp(colors[0, ..., :3], 0.0, 1.0).cpu().numpy()
            frame = (rgb * 255.0 + 0.5).astype(np.uint8)
            writer.append_data(frame)
    finally:
        writer.close()

    print(f"Wrote spiral video: {video_path}")
    return video_path


def load_rgb_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3].astype(np.float32) / 255.0
    return torch.from_numpy(img).to(device)


def _psnr_db(pred: torch.Tensor, ref: torch.Tensor, max_val: float = 1.0) -> float:
    """Finite PSNR; identical images cap at 100 dB so averages stay JSON-safe."""
    mse = float(torch.mean((pred - ref) ** 2).item())
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10((max_val**2) / mse))


@torch.no_grad()
def evaluate_pair(
    ref_paths: List[str],
    pred_paths: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(
        device
    )

    per_view = []
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for i, (rp, pp) in enumerate(zip(ref_paths, pred_paths)):
        ref = load_rgb_tensor(rp, device)
        pred = load_rgb_tensor(pp, device)
        # metrics expect NCHW
        ref_b = ref.permute(2, 0, 1).unsqueeze(0)
        pred_b = pred.permute(2, 0, 1).unsqueeze(0)
        p = _psnr_db(pred, ref)
        s = float(ssim_m(pred_b, ref_b).item())
        l = float(lpips_m(pred_b, ref_b).item())
        psnr_vals.append(p)
        ssim_vals.append(s)
        lpips_vals.append(l)
        per_view.append({"view": i, "psnr": p, "ssim": s, "lpips": l})

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "lpips_std": float(np.std(lpips_vals)),
        "num_views": len(per_view),
        "per_view": per_view,
    }


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for gsplat rasterization.")

    if cfg.spiral_video_only:
        cfg.render_spiral_video = True
    if cfg.metrics_only and cfg.spiral_video_only:
        raise ValueError("Use either --metrics-only or --spiral-video-only, not both.")

    results_root = os.path.abspath(cfg.results_root)
    baseline_ply = os.path.abspath(cfg.baseline_ply)
    output_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cameras_path = os.path.join(output_dir, "cameras_100.json")
    baseline_dir = os.path.join(output_dir, "baseline")
    summary_path = os.path.join(output_dir, "summary.json")
    spiral_dir = os.path.join(output_dir, cfg.spiral_subdir)
    spiral_cameras_path = os.path.join(spiral_dir, "cameras_spiral.json")
    baseline_spiral_mp4 = os.path.join(spiral_dir, "baseline_spiral.mp4")

    print(f"Loading baseline PLY: {baseline_ply}")
    cam_meta: Dict[str, Any] = {}
    packed_base: Optional[Dict[str, Any]] = None
    ref_paths: List[str] = []
    n_base: Optional[int] = None

    if cfg.metrics_only:
        if not os.path.isfile(cameras_path):
            raise FileNotFoundError(
                f"--metrics-only needs existing cameras at {cameras_path}"
            )
        c2w, cam_meta = load_cameras(cameras_path)
        cfg.width = int(cam_meta.get("width", cfg.width))
        cfg.height = int(cam_meta.get("height", cfg.height))
        if cam_meta.get("fx") is not None:
            cfg.fx = float(cam_meta["fx"])
        if cam_meta.get("fy") is not None:
            cfg.fy = float(cam_meta["fy"])
        ref_paths = [
            os.path.join(baseline_dir, f"view_{i:03d}.png") for i in range(c2w.shape[0])
        ]
        if not all(os.path.isfile(p) for p in ref_paths):
            raise FileNotFoundError(f"Missing baseline renders under {baseline_dir}")
        print(f"Metrics-only: reusing {len(ref_paths)} baseline views")
    else:
        baseline = load_ply(baseline_ply)
        baseline = maybe_merge_environment_baseline(
            baseline, baseline_ply, cfg.merge_environment_baseline
        )
        n_base = int(baseline["means"].shape[0])
        print(f"Baseline Gaussians: {n_base}")

        orbit_center, center_info = resolve_orbit_center(
            baseline["means"], cfg.center_from_ply
        )

        force_new_cameras = False
        if os.path.isfile(cameras_path):
            c2w, cam_meta = load_cameras(cameras_path)
            saved_center = cam_meta.get("center")
            params_match = (
                int(cam_meta.get("num_views", -1)) == cfg.num_views
                and int(cam_meta.get("views_per_ring", -1)) == cfg.views_per_ring
                and float(cam_meta.get("radius_step_m", -1)) == cfg.radius_step_m
                and float(cam_meta.get("ring_height_step_m", -1)) == cfg.ring_height_step_m
                and float(cam_meta.get("pitch_down_deg", -1)) == cfg.pitch_down_deg
                and int(cam_meta.get("width", -1)) == cfg.width
                and int(cam_meta.get("height", -1)) == cfg.height
            )
            center_match = True
            if saved_center is not None:
                center_match = (
                    np.linalg.norm(
                        np.asarray(saved_center, dtype=np.float64) - orbit_center
                    )
                    < 1e-3
                )
            if params_match and center_match:
                print(f"Reusing cameras ({c2w.shape[0]} views) from {cameras_path}")
                cfg.width = int(cam_meta.get("width", cfg.width))
                cfg.height = int(cam_meta.get("height", cfg.height))
                if cam_meta.get("fx") is not None:
                    cfg.fx = float(cam_meta["fx"])
                if cam_meta.get("fy") is not None:
                    cfg.fy = float(cam_meta["fy"])
            else:
                print(
                    "Existing cameras do not match requested params/center; regenerating."
                )
                force_new_cameras = True
        else:
            print(
                    "no existing cameras, regenerating."
                )
            force_new_cameras = True 
            
        if force_new_cameras:
            c2w, path_meta = generate_concentric_c2w(
                baseline["means"],
                cfg.num_views,
                cfg.views_per_ring,
                cfg.radius_step_m,
                cfg.height_offset_m,
                ring_height_step_m=cfg.ring_height_step_m,
                pitch_down_deg=cfg.pitch_down_deg,
                center=orbit_center,
            )
            path_meta.update(center_info)
            cam_meta = path_meta
            print(
                f"Cameras: {path_meta['num_rings']} rings × {cfg.views_per_ring} views, "
                f"radii={path_meta['ring_radii_m']} m, "
                f"heights={path_meta['ring_heights_m']} m, "
                f"pitch_down={path_meta['pitch_down_deg']}°, "
                f"center={path_meta['center']}"
            )
            save_cameras(cameras_path, c2w, cfg, path_meta=path_meta)

        if c2w.shape[0] != cfg.num_views:
            print(
                f"Warning: cameras file has {c2w.shape[0]} views; "
                f"--num-views was {cfg.num_views}."
            )

        packed_base = splats_to_device(baseline, device)
        del baseline

        if not cfg.spiral_video_only:
            print("Rendering baseline views...")
            ref_paths = render_views(
                packed_base, c2w, cfg, baseline_dir, desc="baseline"
            )

    spiral_c2w: Optional[np.ndarray] = None
    spiral_meta: Optional[Dict[str, Any]] = None
    if cfg.render_spiral_video:
        pitch = float(cam_meta.get("pitch_down_deg", cfg.pitch_down_deg))
        center = cam_meta.get("center")
        center_arr = np.asarray(center, dtype=np.float64) if center is not None else None
        spiral_c2w, spiral_meta = generate_spiral_c2w_from_cameras(
            c2w,
            n_frames=cfg.spiral_frames,
            n_rots=cfg.spiral_rots,
            pitch_down_deg=pitch,
            center=center_arr,
        )
        os.makedirs(spiral_dir, exist_ok=True)
        with open(spiral_cameras_path, "w") as fp:
            json.dump(
                {
                    **spiral_meta,
                    "width": cfg.width,
                    "height": cfg.height,
                    "fx": _intrinsics(cfg)[0],
                    "fy": _intrinsics(cfg)[1],
                    "fps": cfg.spiral_fps,
                    "c2w": spiral_c2w.tolist(),
                },
                fp,
            )
        print(
            f"Spiral: {cfg.spiral_frames} frames, {cfg.spiral_rots} rots, "
            f"r={spiral_meta['radius_start_m']:.2f}→{spiral_meta['radius_end_m']:.2f} m, "
            f"h={spiral_meta['height_start_m']:.2f}→{spiral_meta['height_end_m']:.2f} m"
        )
        if packed_base is not None:
            render_spiral_video(
                packed_base,
                spiral_c2w,
                cfg,
                baseline_spiral_mp4,
                desc="spiral baseline",
            )

    if packed_base is not None:
        del packed_base
        torch.cuda.empty_cache()

    names = discover_results(results_root, cfg.only)
    if not names:
        raise FileNotFoundError(f"No compression results under {results_root}")
    print(f"Found {len(names)} results: {names}")

    summary: Dict[str, Any] = {
        "baseline_ply": baseline_ply,
        "baseline_gaussians": n_base,
        "cameras": cameras_path,
        "num_views": int(c2w.shape[0]),
        "width": cfg.width,
        "height": cfg.height,
        "results": {},
    }
    if cfg.render_spiral_video:
        summary["spiral_video"] = {
            "cameras": spiral_cameras_path,
            "baseline_mp4": baseline_spiral_mp4,
            **(spiral_meta or {}),
        }

    for name in names:
        metrics_path = os.path.join(results_root, name, "eval_vs_baseline.json")
        render_dir = os.path.join(results_root, name, "renders", "eval_100views")
        result_spiral_mp4 = os.path.join(
            results_root, name, "renders", cfg.spiral_subdir, f"{name}_spiral.mp4"
        )
        pred_paths = [
            os.path.join(render_dir, f"view_{i:03d}.png") for i in range(c2w.shape[0])
        ]

        need_metrics = not cfg.spiral_video_only
        metrics_done = (
            cfg.skip_existing
            and os.path.isfile(metrics_path)
            and all(os.path.isfile(p) for p in pred_paths)
        )
        spiral_done = (
            cfg.render_spiral_video
            and cfg.skip_existing
            and os.path.isfile(result_spiral_mp4)
        )

        if cfg.metrics_only:
            if not all(os.path.isfile(p) for p in pred_paths):
                print(f"[skip] {name}: missing renders in {render_dir}")
                continue
            print(f"\n=== {name} (metrics only) ===")
        elif need_metrics and metrics_done and (not cfg.render_spiral_video or spiral_done):
            print(f"[skip] {name} already evaluated"
                  + (" + spiral" if cfg.render_spiral_video else ""))
            with open(metrics_path, "r") as fp:
                metrics = json.load(fp)
            summary["results"][name] = {
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "psnr_std": metrics.get("psnr_std"),
                "ssim_std": metrics.get("ssim_std"),
                "lpips_std": metrics.get("lpips_std"),
                "metrics_path": metrics_path,
                "render_dir": render_dir,
            }
            if cfg.render_spiral_video:
                summary["results"][name]["spiral_mp4"] = result_spiral_mp4
            continue
        elif cfg.spiral_video_only and spiral_done:
            print(f"[skip] {name} spiral video exists")
            summary["results"][name] = {"spiral_mp4": result_spiral_mp4}
            continue
        else:
            compress_dir = os.path.join(results_root, name, "compression")
            print(f"\n=== {name} ===")
            print(f"Decoding {compress_dir} ...")
            frames = decode_sequence(compress_dir, keep_padding=cfg.keep_padding)
            if cfg.frame_index >= len(frames):
                raise IndexError(
                    f"{name}: frame_index={cfg.frame_index} but only {len(frames)} frames"
                )
            splat = frames[cfg.frame_index]
            del frames
            print(f"Decoded Gaussians: {int(splat['means'].shape[0])}")

            packed = splats_to_device(splat, device)
            del splat

            if need_metrics and not metrics_done:
                pred_paths = render_views(
                    packed, c2w, cfg, render_dir, desc=f"render {name}"
                )

            if cfg.render_spiral_video and spiral_c2w is not None and not spiral_done:
                render_spiral_video(
                    packed,
                    spiral_c2w,
                    cfg,
                    result_spiral_mp4,
                    desc=f"spiral {name}",
                )

            del packed
            torch.cuda.empty_cache()

        entry: Dict[str, Any] = {"render_dir": render_dir}
        if cfg.render_spiral_video:
            entry["spiral_mp4"] = result_spiral_mp4

        if need_metrics:
            print(f"Evaluating {name} ...")
            metrics = evaluate_pair(ref_paths, pred_paths, device)
            metrics["name"] = name
            metrics["render_dir"] = render_dir
            if cfg.render_spiral_video:
                metrics["spiral_mp4"] = result_spiral_mp4
            with open(metrics_path, "w") as fp:
                json.dump(metrics, fp, indent=2)
            entry.update(
                {
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips": metrics["lpips"],
                    "psnr_std": metrics["psnr_std"],
                    "ssim_std": metrics["ssim_std"],
                    "lpips_std": metrics["lpips_std"],
                    "metrics_path": metrics_path,
                }
            )
            print(
                f"{name}: PSNR={metrics['psnr']:.3f}  "
                f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}"
            )

        summary["results"][name] = entry
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2)

    # Final sorted table
    rows = sorted(summary["results"].items(), key=lambda kv: kv[0])
    if not cfg.spiral_video_only:
        print("\n=== Summary (vs baseline) ===")
        print(f"{'result':<32} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        for name, m in rows:
            if "psnr" not in m:
                continue
            print(
                f"{name:<32} {m['psnr']:8.3f} {m['ssim']:8.4f} {m['lpips']:8.4f}"
            )
    if cfg.render_spiral_video:
        print("\n=== Spiral videos ===")
        print(f"baseline: {baseline_spiral_mp4}")
        for name, m in rows:
            if m.get("spiral_mp4"):
                print(f"{name}: {m['spiral_mp4']}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
