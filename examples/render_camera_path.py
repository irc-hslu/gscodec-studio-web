"""Render baseline + every compressed result along a user-defined camera path.

The camera JSON is the ``{"c2w": [...], "width", "height", "fx", "fy", "cx", "cy"}``
file saved from the viser viewer (``decode_and_view.py`` → View list → Save JSON).

For each model (baseline PLY and every ``<results_root>/*/compression``):
  * one PNG per camera pose  -> ``<output_dir>/<name>/view_XXX.png``
  * one MP4 following the path -> ``<output_dir>/<name>.mp4``

The video is either the raw poses (one frame per camera) or a smooth path
interpolated through them (centripetal Catmull–Rom positions at constant speed +
quaternion SLERP orientations), controlled by ``--video-frames``.

Optionally computes PSNR / SSIM / LPIPS of each result's stills vs the baseline stills.

Run from ``examples/``:

    python render_camera_path.py \\
      --cameras eva/AbbeyStreet_Stairs/cameras_100.json \\
      --baseline-ply ../AbbeyStreet_Stairs/ply-result/lod0/point_cloud.ply \\
      --results-root results/AbbeyStreet_Stairs \\
      --output-dir eva/AbbeyStreet_Stairs/camera_path

    # raw poses only (100 frames), no metrics
    python render_camera_path.py ... --video-frames 0 --no-compute-metrics
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from tqdm import tqdm

from decode_and_view import decode_sequence
from eval_encoded_results import (
    discover_results,
    evaluate_pair,
    maybe_merge_environment_baseline,
    splats_to_device,
)
from gsplat.rendering import rasterization
from jpeg_rf_trajectory import (
    quaternion_to_rotation,
    rotation_to_quaternion,
    sample_spline_arc_length,
    slerp,
)
from utils import load_ply


@dataclass
class Config:
    # Camera path JSON saved from the viser viewer.
    cameras: str = "eva/AbbeyStreet_Stairs/cameras_100.json"
    baseline_ply: str = "../AbbeyStreet_Stairs/ply-result/lod0/point_cloud.ply"
    # Also load environment.ply next to the baseline (see eval_encoded_results).
    merge_environment_baseline: bool = False
    results_root: str = "results/AbbeyStreet_Stairs"
    output_dir: str = "eva/AbbeyStreet_Stairs/camera_path"
    # Only process these result folder names (default: all under results_root).
    only: Optional[List[str]] = None
    # Skip the baseline model entirely.
    skip_baseline: bool = False

    # Override render size; intrinsics are scaled from the JSON values.
    width: Optional[int] = None
    height: Optional[int] = None

    render_stills: bool = True
    render_video: bool = True
    # Frames in the interpolated video. 0 = one frame per camera pose (no interpolation).
    video_frames: int = 600
    fps: int = 25
    # Close the loop (last camera back to the first) in the interpolated video.
    loop: bool = False
    arc_length_samples: int = 4096

    compute_metrics: bool = True
    frame_index: int = 0
    keep_padding: bool = False
    near_plane: float = 0.01
    far_plane: float = 1e10
    radius_clip: float = 0.0
    skip_existing: bool = True


def load_camera_path(cfg: Config) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(cfg.cameras, "r") as fp:
        payload = json.load(fp)
    if isinstance(payload, list):
        payload = {"c2w": payload}
    c2w = np.asarray(payload["c2w"], dtype=np.float32).reshape(-1, 4, 4)

    src_w = int(payload.get("width", 1920))
    src_h = int(payload.get("height", 1080))
    width = int(cfg.width or src_w)
    height = int(cfg.height or src_h)
    sx, sy = width / src_w, height / src_h
    fx = float(payload.get("fx", src_h)) * sx
    fy = float(payload.get("fy", src_h)) * sy
    cx = float(payload.get("cx", src_w * 0.5)) * sx
    cy = float(payload.get("cy", src_h * 0.5)) * sy
    intr = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return c2w, intr


def interpolate_path(
    keyposes: np.ndarray, n_frames: int, loop: bool, arc_length_samples: int
) -> np.ndarray:
    """Constant-speed Catmull–Rom positions + SLERP orientations through ``keyposes``."""
    keys = [np.asarray(k, dtype=np.float64) for k in keyposes]
    if loop and len(keys) > 2:
        keys.append(keys[0].copy())
    if len(keys) < 2 or n_frames <= 0:
        return np.asarray(keyposes, dtype=np.float32)

    pts = np.stack([k[:3, 3] for k in keys], axis=0)
    positions = sample_spline_arc_length(pts, n_frames, arc_length_samples, alpha=0.5)

    quats = [rotation_to_quaternion(k[:3, :3]) for k in keys]
    seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    seg_len = np.maximum(seg_len, 1e-9)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, cum[-1], n_frames)

    out = np.zeros((n_frames, 4, 4), dtype=np.float32)
    for i, s in enumerate(targets):
        seg = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(keys) - 2))
        t = float((s - cum[seg]) / seg_len[seg])
        q = slerp(quats[seg], quats[seg + 1], float(np.clip(t, 0.0, 1.0)))
        out[i, :3, :3] = quaternion_to_rotation(q)
        out[i, :3, 3] = positions[i]
        out[i, 3, 3] = 1.0
    return out


class Renderer:
    def __init__(self, packed: Dict[str, Any], intr: Dict[str, Any], cfg: Config):
        self.packed = packed
        self.cfg = cfg
        self.w, self.h = intr["width"], intr["height"]
        device = packed["means"].device
        self.K = torch.tensor(
            [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )[None]
        self.kwargs = {"radius_clip": cfg.radius_clip} if cfg.radius_clip > 0 else {}

    @torch.no_grad()
    def __call__(self, c2w: np.ndarray) -> np.ndarray:
        p = self.packed
        cam = torch.from_numpy(np.asarray(c2w, dtype=np.float32)).to(p["means"].device)
        colors, _, _ = rasterization(
            p["means"],
            p["quats"],
            p["scales"],
            p["opacities"],
            p["colors"],
            torch.linalg.inv(cam)[None],
            self.K,
            self.w,
            self.h,
            sh_degree=p["sh_degree"],
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB",
            packed=False,
            **self.kwargs,
        )
        rgb = torch.clamp(colors[0, ..., :3], 0.0, 1.0).cpu().numpy()
        return (rgb * 255.0 + 0.5).astype(np.uint8)


def render_stills(render: Renderer, c2w: np.ndarray, out_dir: str, desc: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in tqdm(range(c2w.shape[0]), desc=desc):
        path = os.path.join(out_dir, f"view_{i:03d}.png")
        paths.append(path)
        if render.cfg.skip_existing and os.path.isfile(path):
            continue
        imageio.imwrite(path, render(c2w[i]))
    return paths


def render_video(render: Renderer, c2w: np.ndarray, path: str, desc: str) -> None:
    if render.cfg.skip_existing and os.path.isfile(path):
        print(f"[skip] video exists: {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(
        path, fps=render.cfg.fps, codec="libx264", quality=8, pixelformat="yuv420p"
    )
    try:
        for i in tqdm(range(c2w.shape[0]), desc=desc):
            writer.append_data(render(c2w[i]))
    finally:
        writer.close()
    print(f"Wrote video: {path}")


def render_model(
    name: str,
    packed: Dict[str, Any],
    still_c2w: np.ndarray,
    video_c2w: np.ndarray,
    intr: Dict[str, Any],
    cfg: Config,
    output_dir: str,
) -> Dict[str, Any]:
    render = Renderer(packed, intr, cfg)
    entry: Dict[str, Any] = {}
    if cfg.render_stills:
        still_dir = os.path.join(output_dir, name)
        entry["stills_dir"] = still_dir
        entry["still_paths"] = render_stills(render, still_c2w, still_dir, f"stills {name}")
    if cfg.render_video:
        mp4 = os.path.join(output_dir, f"{name}.mp4")
        render_video(render, video_c2w, mp4, f"video {name}")
        entry["video"] = mp4
    return entry


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for gsplat rasterization.")

    output_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    still_c2w, intr = load_camera_path(cfg)
    print(f"Loaded {still_c2w.shape[0]} camera poses from {cfg.cameras}")
    print(f"Render size {intr['width']}x{intr['height']}, fx={intr['fx']:.1f} fy={intr['fy']:.1f}")

    video_c2w = interpolate_path(still_c2w, cfg.video_frames, cfg.loop, cfg.arc_length_samples)
    video_cams_path = os.path.join(output_dir, "cameras_video.json")
    with open(video_cams_path, "w") as fp:
        json.dump(
            {
                "source_cameras": os.path.abspath(cfg.cameras),
                "num_keyposes": int(still_c2w.shape[0]),
                "n_frames": int(video_c2w.shape[0]),
                "fps": cfg.fps,
                "interpolated": cfg.video_frames > 0,
                "loop": cfg.loop,
                **intr,
                "c2w": video_c2w.tolist(),
            },
            fp,
        )
    print(f"Video path: {video_c2w.shape[0]} frames @ {cfg.fps} fps -> {video_cams_path}")

    summary: Dict[str, Any] = {
        "cameras": os.path.abspath(cfg.cameras),
        "video_cameras": video_cams_path,
        **intr,
        "num_views": int(still_c2w.shape[0]),
        "video_frames": int(video_c2w.shape[0]),
        "fps": cfg.fps,
        "baseline": None,
        "results": {},
    }
    summary_path = os.path.join(output_dir, "summary.json")

    ref_paths: Optional[List[str]] = None
    if not cfg.skip_baseline:
        baseline_ply = os.path.abspath(cfg.baseline_ply)
        print(f"Loading baseline PLY: {baseline_ply}")
        baseline = load_ply(baseline_ply)
        baseline = maybe_merge_environment_baseline(
            baseline, baseline_ply, cfg.merge_environment_baseline
        )
        n_base = int(baseline["means"].shape[0])
        print(f"Baseline Gaussians: {n_base}")
        packed = splats_to_device(baseline, device)
        del baseline
        entry = render_model("baseline", packed, still_c2w, video_c2w, intr, cfg, output_dir)
        ref_paths = entry.pop("still_paths", None)
        summary["baseline"] = {"ply": baseline_ply, "gaussians": n_base, **entry}
        del packed
        torch.cuda.empty_cache()

    names = discover_results(os.path.abspath(cfg.results_root), cfg.only)
    print(f"Found {len(names)} compressed results: {names}")

    for name in names:
        compress_dir = os.path.join(os.path.abspath(cfg.results_root), name, "compression")
        print(f"\n=== {name} ===")
        frames = decode_sequence(compress_dir, keep_padding=cfg.keep_padding)
        if cfg.frame_index >= len(frames):
            raise IndexError(f"{name}: frame_index={cfg.frame_index}, only {len(frames)} frames")
        splat = frames[cfg.frame_index]
        del frames
        n = int(splat["means"].shape[0])
        print(f"Decoded Gaussians: {n}")
        packed = splats_to_device(splat, device)
        del splat
        entry = render_model(name, packed, still_c2w, video_c2w, intr, cfg, output_dir)
        del packed
        torch.cuda.empty_cache()

        pred_paths = entry.pop("still_paths", None)
        entry["gaussians"] = n
        if cfg.compute_metrics and ref_paths and pred_paths:
            metrics = evaluate_pair(ref_paths, pred_paths, device)
            with open(os.path.join(output_dir, name, "metrics_vs_baseline.json"), "w") as fp:
                json.dump(metrics, fp, indent=2)
            for k in ("psnr", "ssim", "lpips", "psnr_std", "ssim_std", "lpips_std"):
                entry[k] = metrics[k]
            print(
                f"{name}: PSNR={metrics['psnr']:.3f}  SSIM={metrics['ssim']:.4f}  "
                f"LPIPS={metrics['lpips']:.4f}"
            )
        summary["results"][name] = entry
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2)

    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    if cfg.compute_metrics and summary["results"]:
        print("\n=== Summary (vs baseline, camera path stills) ===")
        print(f"{'result':<24} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
        for name, m in sorted(summary["results"].items()):
            if "psnr" in m:
                print(f"{name:<24} {m['psnr']:8.3f} {m['ssim']:8.4f} {m['lpips']:8.4f}")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
