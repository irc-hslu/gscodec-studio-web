"""Run the external HAC++ baseline and collect bitstream size / test metrics.

HAC++ lives in examples/third_party/HAC-plus (see _setup_hacpp.sh).
It uses a **separate** conda environment (HAC_env), not gscodec_studio.

Upstream: https://github.com/YihangChen-ee/HAC-plus
Paper: HAC++: Towards 100X Compression of 3D Gaussian Splatting (TPAMI 2025)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tyro

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_HAC_ROOT = EXAMPLES_DIR / "third_party" / "HAC-plus"


@dataclass
class Config:
    hac_root: str = str(DEFAULT_HAC_ROOT)
    dataset: str = "tandt"
    scene: str = "train"
    # HAC++ rate knob (see run_shell_tnt.py); try 0.004, 0.002, 0.001 for RD sweep.
    lmbda: float = 0.004
    cuda_device: int = 0
    iterations: int = 30_000
    # Only print paths / parse existing output without launching train.py.
    collect_only: bool = False
    dry_run: bool = False


def hac_env_python(hac_root: Path) -> str:
    """Prefer HAC_env if active; otherwise assume `python` is HAC_env."""
    return os.environ.get("HAC_PYTHON", sys.executable)


def bitstream_size_mb(bitstreams_dir: Path) -> float:
    if not bitstreams_dir.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in bitstreams_dir.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def parse_output_log(log_path: Path) -> dict:
    """Best-effort scrape of HAC++ output.log for PSNR/SSIM/LPIPS lines."""
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    out: dict = {}
    for line in text.splitlines():
        lower = line.lower()
        if "psnr" in lower and ":" in line:
            out.setdefault("log_lines", []).append(line.strip())
    return out


def run_hacpp(cfg: Config) -> None:
    hac_root = Path(cfg.hac_root).resolve()
    train_py = hac_root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            f"HAC++ not found at {hac_root}. Run: bash examples/_setup_hacpp.sh"
        )

    data_dir = hac_root / "data" / cfg.dataset / cfg.scene
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing COLMAP data at {data_dir}. "
            f"Download T&T and extract to {hac_root / 'data' / 'tandt'}."
        )

    mask_lr_final = 0.0001 * cfg.lmbda / 0.001
    out_dir = hac_root / "outputs" / cfg.dataset / cfg.scene / str(cfg.lmbda)
    bitstreams = out_dir / "bitstreams"
    log_path = out_dir / "output.log"
    renders = out_dir / "test" / "ours_30000" / "renders"

    summary = {
        "method": "HAC++",
        "hac_root": str(hac_root),
        "scene": f"{cfg.dataset}/{cfg.scene}",
        "lmbda": cfg.lmbda,
        "output_dir": str(out_dir),
        "bitstreams_dir": str(bitstreams),
        "renders_dir": str(renders),
        "size_MB": bitstream_size_mb(bitstreams),
        "log": str(log_path),
    }

    if cfg.collect_only or cfg.dry_run:
        summary.update(parse_output_log(log_path))
        print(json.dumps(summary, indent=2))
        if cfg.dry_run:
            return
        if cfg.collect_only:
            out_json = EXAMPLES_DIR / "eva" / f"hacpp_{cfg.dataset}_{cfg.scene}_l{cfg.lmbda}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(summary, indent=2))
            print(f"Wrote {out_json}")
            return

    cmd: List[str] = [
        hac_env_python(hac_root),
        str(train_py),
        "-s",
        str(data_dir),
        "--eval",
        "--lod",
        "0",
        "--voxel_size",
        "0.01",
        "--update_init_factor",
        "16",
        "--iterations",
        str(cfg.iterations),
        "-m",
        str(out_dir),
        "--lmbda",
        str(cfg.lmbda),
        "--mask_lr_final",
        str(mask_lr_final),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_device)

    print("Running HAC++ (train + encode + decode + test):")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(hac_root), env=env, check=True)

    summary["size_MB"] = bitstream_size_mb(bitstreams)
    summary.update(parse_output_log(log_path))
    out_json = EXAMPLES_DIR / "eva" / f"hacpp_{cfg.dataset}_{cfg.scene}_l{cfg.lmbda}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nBitstream size: {summary['size_MB']:.2f} MB (use bitstreams/, not point_cloud.ply)")
    print(f"Test renders: {renders}")


if __name__ == "__main__":
    tyro.cli(run_hacpp)
