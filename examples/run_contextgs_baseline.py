"""Run the external ContextGS baseline and collect bitstream size / test metrics.

ContextGS lives in examples/third_party/ContextGS (see _setup_contextgs.sh).
It uses a **separate** conda environment (contextgs), not gscodec_studio.

Upstream: https://github.com/wyf0912/ContextGS
Paper: ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Model (NeurIPS 2024)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import tyro

EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_CTX_ROOT = EXAMPLES_DIR / "third_party" / "ContextGS"


@dataclass
class Config:
    contextgs_root: str = str(DEFAULT_CTX_ROOT)
    dataset: str = "tandt"
    scene: str = "train"
    # Rate knob (see train_scripts/run_shell_tnt.py): 0.004, 0.002, 0.001, ...
    lmbda: float = 0.004
    cuda_device: int = 0
    iterations: int = 30_000
    lod: int = 0
    voxel_size: float = 0.01
    update_init_factor: int = 16
    use_wandb: bool = False
    # train = full pipeline; test = evaluate existing checkpoint at -m path
    mode: str = "train"
    collect_only: bool = False
    dry_run: bool = False


def ctx_python() -> str:
    return os.environ.get("CONTEXTGS_PYTHON", sys.executable)


def bitstream_size_mb(bitstreams_dir: Path) -> float:
    if not bitstreams_dir.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in bitstreams_dir.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def parse_output_log(log_path: Path) -> dict:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "log_lines": [ln.strip() for ln in text.splitlines() if "psnr" in ln.lower()][-20:]
    }


def output_dir(cfg: Config, ctx_root: Path) -> Path:
    # Matches upstream train_scripts/run_shell_tnt.py layout
    return ctx_root / "outputs" / "final_ours" / cfg.dataset / f"{cfg.scene}_{cfg.lmbda}"


def run_contextgs(cfg: Config) -> None:
    ctx_root = Path(cfg.contextgs_root).resolve()
    train_py = ctx_root / "train.py"
    test_py = ctx_root / "test.py"
    if not train_py.is_file():
        raise FileNotFoundError(
            f"ContextGS not found at {ctx_root}. Run: bash examples/_setup_contextgs.sh"
        )

    data_dir = ctx_root / "data" / cfg.dataset / cfg.scene
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing COLMAP data at {data_dir}. "
            f"Download T&T to {ctx_root / 'data' / 'tandt'}."
        )

    out_dir = output_dir(cfg, ctx_root)
    bitstreams = out_dir / "bitstreams"
    log_path = out_dir / "output.log"
    renders = out_dir / "test" / "ours_30000" / "renders"

    summary = {
        "method": "ContextGS",
        "contextgs_root": str(ctx_root),
        "scene": f"{cfg.dataset}/{cfg.scene}",
        "lmbda": cfg.lmbda,
        "mode": cfg.mode,
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
            out_json = EXAMPLES_DIR / "eva" / f"contextgs_{cfg.dataset}_{cfg.scene}_l{cfg.lmbda}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(summary, indent=2))
            print(f"Wrote {out_json}")
            return

    script = train_py if cfg.mode == "train" else test_py
    cmd: List[str] = [
        ctx_python(),
        str(script),
        "-s",
        str(data_dir),
        "--eval",
        "--lod",
        str(cfg.lod),
        "--voxel_size",
        str(cfg.voxel_size),
        "--iterations",
        str(cfg.iterations),
        "-m",
        str(out_dir),
    ]
    if cfg.mode == "train":
        cmd.extend(
            [
                "--update_init_factor",
                str(cfg.update_init_factor),
                "--lmbda",
                str(cfg.lmbda),
            ]
        )
        if cfg.use_wandb:
            cmd.append("--use_wandb")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_device)

    print(f"Running ContextGS ({cfg.mode}):")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ctx_root), env=env, check=True)

    summary["size_MB"] = bitstream_size_mb(bitstreams)
    summary.update(parse_output_log(log_path))
    out_json = EXAMPLES_DIR / "eva" / f"contextgs_{cfg.dataset}_{cfg.scene}_l{cfg.lmbda}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nBitstream size: {summary['size_MB']:.2f} MB (use bitstreams/, not point_cloud.ply)")
    print(f"Test renders: {renders}")
    print(
        "\nDecompress-only replay:\n"
        f"  python decompress.py -s {data_dir} --eval --lod {cfg.lod} "
        f"-m {out_dir} --voxel_size {cfg.voxel_size}"
    )


if __name__ == "__main__":
    tyro.cli(run_contextgs)
