"""Evaluate SOGS (PNG+PLAS) compressed results against the Mellifont baseline."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_encoded_results import (
    Config as EvalConfig,
    evaluate_pair,
    load_cameras,
    maybe_merge_environment_baseline,
    render_spiral_video,
    render_views,
    splats_to_device,
)
from gsplat.compression import PngCompression
from utils import load_ply


@dataclass
class Config:
    results_root: str = "results"
    baseline_ply: str = (
        "../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply"
    )
    center_from_ply: Optional[str] = None
    merge_environment_baseline: bool = True
    output_dir: str = "eva/_eval_100views_sogs"
    only: Optional[List[str]] = None
    skip_existing: bool = True
    render_spiral_video: bool = False
    spiral_video_only: bool = False
    metrics_only: bool = False
    num_views: int = 104
    width: int = 1920
    height: int = 1080
    views_per_ring: int = 8
    radius_step_m: float = 1.2
    ring_height_step_m: float = 0.2
    pitch_down_deg: float = 0.0
    spiral_frames: int = 600
    spiral_fps: int = 25
    spiral_rots: float = 2.0


def discover_sogs_results(results_root: str, only: Optional[List[str]]) -> List[str]:
    names = []
    for name in sorted(os.listdir(results_root)):
        if name.startswith("_") or "SOGS" not in name:
            continue
        manifest = os.path.join(results_root, name, "compression", "sogs_manifest.json")
        if not os.path.isfile(manifest):
            continue
        if only is not None and name not in only:
            continue
        names.append(name)
    return names


def decode_sogs(compress_dir: str) -> Dict[str, torch.Tensor]:
    return PngCompression(verbose=False).decompress(compress_dir)


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for gsplat rasterization.")

    eval_cfg = EvalConfig(
        results_root=cfg.results_root,
        baseline_ply=cfg.baseline_ply,
        center_from_ply=cfg.center_from_ply,
        merge_environment_baseline=cfg.merge_environment_baseline,
        output_dir=cfg.output_dir,
        num_views=cfg.num_views,
        width=cfg.width,
        height=cfg.height,
        views_per_ring=cfg.views_per_ring,
        radius_step_m=cfg.radius_step_m,
        ring_height_step_m=cfg.ring_height_step_m,
        pitch_down_deg=cfg.pitch_down_deg,
        skip_existing=cfg.skip_existing,
        render_spiral_video=cfg.render_spiral_video or cfg.spiral_video_only,
        spiral_video_only=cfg.spiral_video_only,
        metrics_only=cfg.metrics_only,
        spiral_frames=cfg.spiral_frames,
        spiral_fps=cfg.spiral_fps,
        spiral_rots=cfg.spiral_rots,
    )

    results_root = os.path.abspath(cfg.results_root)
    output_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cameras_path = os.path.join(output_dir, "cameras_100.json")
    baseline_dir = os.path.join(output_dir, "baseline")
    summary_path = os.path.join(output_dir, "summary.json")
    spiral_dir = os.path.join(output_dir, "spiral_video")
    spiral_cameras_path = os.path.join(spiral_dir, "cameras_spiral.json")

    main_eval_dir = os.path.abspath("eva/_eval_100views")
    main_cameras = os.path.join(main_eval_dir, "cameras_100.json")
    main_baseline = os.path.join(main_eval_dir, "baseline")
    if os.path.isfile(main_cameras) and not os.path.isfile(cameras_path):
        shutil.copy2(main_cameras, cameras_path)
        if os.path.isdir(main_baseline):
            shutil.copytree(main_baseline, baseline_dir, dirs_exist_ok=True)
        main_spiral_cam = os.path.join(main_eval_dir, "spiral_video", "cameras_spiral.json")
        if os.path.isfile(main_spiral_cam):
            os.makedirs(spiral_dir, exist_ok=True)
            shutil.copy2(main_spiral_cam, spiral_cameras_path)

    if not os.path.isfile(cameras_path):
        raise FileNotFoundError(
            f"Missing cameras at {cameras_path}. Run eval_encoded_results.py first."
        )

    c2w, cam_meta = load_cameras(cameras_path)
    eval_cfg.width = int(cam_meta.get("width", eval_cfg.width))
    eval_cfg.height = int(cam_meta.get("height", eval_cfg.height))
    if cam_meta.get("fx") is not None:
        eval_cfg.fx = float(cam_meta["fx"])
    if cam_meta.get("fy") is not None:
        eval_cfg.fy = float(cam_meta["fy"])

    ref_paths = [
        os.path.join(baseline_dir, f"view_{i:03d}.png") for i in range(c2w.shape[0])
    ]
    if not all(os.path.isfile(p) for p in ref_paths):
        baseline = load_ply(os.path.abspath(cfg.baseline_ply))
        baseline = maybe_merge_environment_baseline(
            baseline, cfg.baseline_ply, cfg.merge_environment_baseline
        )
        packed = splats_to_device(dict(baseline), device)
        render_views(packed, c2w, eval_cfg, baseline_dir, desc="baseline")

    names = discover_sogs_results(results_root, cfg.only)
    if not names:
        raise FileNotFoundError(f"No SOGS results under {results_root}")

    summary: Dict[str, Any] = {"results": {}, "codec": "sogs_png"}
    spiral_c2w = None
    if eval_cfg.render_spiral_video and os.path.isfile(spiral_cameras_path):
        spiral_c2w, _ = load_cameras(spiral_cameras_path)

    for name in names:
        print(f"\n=== {name} ===")
        compress_dir = os.path.join(results_root, name, "compression")
        render_dir = os.path.join(results_root, name, "renders", "eval_100views")
        metrics_path = os.path.join(results_root, name, "eval_vs_baseline.json")
        spiral_mp4 = os.path.join(
            results_root, name, "renders", eval_cfg.spiral_subdir, f"{name}_spiral.mp4"
        )

        metrics: Dict[str, Any] = {}
        if cfg.skip_existing and os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            print(f"[skip] metrics exist: {metrics_path}")
        elif not cfg.spiral_video_only:
            splats = decode_sogs(compress_dir)
            packed = splats_to_device(splats, device)
            pred_paths = render_views(packed, c2w, eval_cfg, render_dir, desc=name)
            metrics = evaluate_pair(ref_paths, pred_paths, device)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.3f} "
                f"LPIPS={metrics['lpips']:.3f}"
            )
        else:
            splats = decode_sogs(compress_dir)
            packed = splats_to_device(splats, device)
            render_views(packed, c2w, eval_cfg, render_dir, desc=name)

        if eval_cfg.render_spiral_video and spiral_c2w is not None:
            splats = decode_sogs(compress_dir)
            packed = splats_to_device(splats, device)
            os.makedirs(os.path.dirname(spiral_mp4), exist_ok=True)
            render_spiral_video(packed, spiral_c2w, eval_cfg, spiral_mp4, desc=f"spiral {name}")

        with open(os.path.join(compress_dir, "sogs_manifest.json")) as f:
            manifest = json.load(f)
        entry = {
            "render_dir": render_dir,
            "spiral_mp4": spiral_mp4 if eval_cfg.render_spiral_video else None,
            "size_MB": manifest.get("size_MB"),
            "metrics_path": metrics_path,
            **metrics,
        }
        summary["results"][name] = entry

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    tyro.cli(main)
