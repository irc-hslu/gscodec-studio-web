"""Generate JPEG-RF (PCQ ES5) camera trajectories and optionally render evaluation videos."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import tyro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_encoded_results import (
    Config as EvalConfig,
    load_cameras,
    maybe_merge_environment_baseline,
    render_spiral_video,
    splats_to_device,
)
from jpeg_rf_trajectory import (
    build_trajectory,
    load_trajectory_config,
    save_trajectory_json,
)
from utils import load_ply


@dataclass
class Config:
    trajectory: str = "mellifont_recommended"
    registry: str = "configs/jpeg_rf_trajectories.json"
    training_cameras: str = "eva/_eval_100views/cameras_100.json"
    output_dir: str = "eva/jpeg_rf_trajectories"
    baseline_ply: str = (
        "../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply"
    )
    merge_environment_baseline: bool = True
    center_from_ply: Optional[str] = (
        "../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100/point_cloud.ply"
    )
    look_at: Optional[List[float]] = None
    n_frames: Optional[int] = None
    fps: Optional[int] = None
    seed: int = 0
    render_video: bool = False
    width: int = 1920
    height: int = 1080
    all_presets: bool = False


def main(cfg: Config) -> None:
    out_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_c2w, train_meta = load_cameras(os.path.abspath(cfg.training_cameras))
    traj_cfg = load_trajectory_config(os.path.abspath(cfg.registry), cfg.trajectory)
    if cfg.n_frames is not None:
        traj_cfg.n_frames = cfg.n_frames
    if cfg.fps is not None:
        traj_cfg.fps = cfg.fps

    look_at = None
    if cfg.look_at is not None:
        look_at = np.asarray(cfg.look_at, dtype=np.float64)
    elif train_meta.get("center") is not None:
        look_at = np.asarray(train_meta["center"], dtype=np.float64)
    elif cfg.center_from_ply:
        scene = load_ply(os.path.abspath(cfg.center_from_ply))
        look_at = scene["means"].detach().float().mean(dim=0).cpu().numpy()

    presets = [cfg.trajectory]
    if cfg.all_presets:
        with open(os.path.abspath(cfg.registry), encoding="utf-8") as f:
            presets = list(json.load(f)["trajectories"].keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.render_video and device.type != "cuda":
        raise RuntimeError("CUDA required for video rendering.")

    packed = None
    if cfg.render_video:
        baseline = load_ply(os.path.abspath(cfg.baseline_ply))
        baseline = maybe_merge_environment_baseline(
            baseline, cfg.baseline_ply, cfg.merge_environment_baseline
        )
        packed = splats_to_device(dict(baseline), device)

    eval_cfg = EvalConfig(
        width=int(train_meta.get("width", cfg.width)),
        height=int(train_meta.get("height", cfg.height)),
        spiral_fps=traj_cfg.fps,
        spiral_frames=traj_cfg.n_frames,
    )
    if train_meta.get("fx") is not None:
        eval_cfg.fx = float(train_meta["fx"])
    if train_meta.get("fy") is not None:
        eval_cfg.fy = float(train_meta["fy"])

    for name in presets:
        tcfg = load_trajectory_config(os.path.abspath(cfg.registry), name)
        if cfg.n_frames is not None:
            tcfg.n_frames = cfg.n_frames
        if cfg.fps is not None:
            tcfg.fps = cfg.fps

        c2w, meta = build_trajectory(train_c2w, tcfg, look_at=look_at, seed=cfg.seed)
        json_path = os.path.join(out_dir, f"cameras_{name}.json")
        save_trajectory_json(json_path, c2w, meta, train_meta)
        print(f"Wrote {json_path} ({tcfg.n_frames} frames, {tcfg.fps} fps)")

        if cfg.render_video and packed is not None:
            mp4 = os.path.join(out_dir, f"{name}.mp4")
            eval_cfg.spiral_fps = tcfg.fps
            eval_cfg.spiral_frames = tcfg.n_frames
            render_spiral_video(packed, c2w, eval_cfg, mp4, desc=f"jpeg-rf {name}")
            print(f"Wrote {mp4}")


if __name__ == "__main__":
    tyro.cli(main)
