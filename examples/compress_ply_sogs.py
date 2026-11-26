"""Compress static PLY Gaussians with SOGS-style PNG+PLAS (PngCompression baseline)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import tyro
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gsplat.compression import PngCompression
from utils import concat_splats, load_3dgs_iteration, load_ply


@dataclass
class SogsRatePoint:
    """One SOGS operating point (uniform k-bit attrs + K-means SH bands)."""

    name: str
    quantization: int = 8
    n_clusters: int = 16384
    include_shN: bool = True


SOGS_RATE_POINTS: Dict[str, SogsRatePoint] = {
    "rp0": SogsRatePoint("rp0", quantization=5, n_clusters=4096),
    "rp1": SogsRatePoint("rp1", quantization=6, n_clusters=8192),
    "rp2": SogsRatePoint("rp2", quantization=7, n_clusters=16384),
    "rp3": SogsRatePoint("rp3", quantization=8, n_clusters=32768),
}


@dataclass
class Config:
    iteration_dir: str = (
        "../MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100"
    )
    result_dir: str = "results"
    scene_tag: str = "MellifontAbbey"
    # None = point_cloud.ply (full); 1/2/3 = LOD files.
    ply_lod: Optional[int] = None
    merge_environment: bool = True
    rate_point: str = "rp2"
    use_sort: bool = True
    skip_existing: bool = False


def _param_dict_to_tensors(splats: torch.nn.ParameterDict) -> Dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in splats.items()}


class ConfigurablePngCompression(PngCompression):
    """PngCompression with per-run quantization and SH cluster count."""

    def __init__(
        self,
        quantization: int = 8,
        n_clusters: int = 16384,
        include_shN: bool = True,
        use_sort: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(use_sort=use_sort, verbose=verbose)
        self.quantization = int(quantization)
        self.n_clusters = int(n_clusters)
        self.include_shN = bool(include_shN)

    def compress(self, compress_dir: str, splats: Dict[str, Tensor], entropy_models=None) -> None:
        if entropy_models is not None:
            raise ValueError("PngCompression should not require entropy_models")

        from gsplat.compression.outlier_filter import filter_splats
        import torch.nn.functional as F
        from gsplat.utils import log_transform

        splats = {k: v for k, v in splats.items()}
        if not self.include_shN:
            splats["shN"] = torch.zeros(
                (splats["means"].shape[0], 0, 3),
                dtype=splats["sh0"].dtype,
                device=splats["means"].device,
            )

        vaild_mask, splats = filter_splats(splats)
        del vaild_mask

        splats["means"] = log_transform(splats["means"])
        splats["quats"] = F.normalize(splats["quats"], dim=-1)

        n_gs = len(splats["means"])
        n_sidelen = int(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2
        if n_crop != 0:
            from gsplat.compression.png_compression import _crop_n_splats as crop_fn

            splats = crop_fn(splats, n_crop)
            print(f"Warning: removed {n_crop} Gaussians to make a square grid.")

        if self.use_sort:
            from gsplat.compression.sort import sort_splats

            splats = sort_splats(splats)

        os.makedirs(compress_dir, exist_ok=True)
        meta = {}
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
            }
            if param_name in {"scales", "quats", "sh0"}:
                kwargs["quantization"] = self.quantization
            if param_name == "shN":
                kwargs["n_clusters"] = self.n_clusters
            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f)


def result_name(cfg: Config) -> str:
    lod = "full" if cfg.ply_lod is None else f"LOD{cfg.ply_lod}"
    variant = "SOGS" if SOGS_RATE_POINTS[cfg.rate_point].include_shN else "SOGS_noSH"
    return f"{cfg.scene_tag}_{variant}_{lod}_{cfg.rate_point}"


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def main(cfg: Config) -> None:
    if cfg.rate_point not in SOGS_RATE_POINTS:
        raise ValueError(f"Unknown rate point {cfg.rate_point}; choose from {list(SOGS_RATE_POINTS)}")

    rp = SOGS_RATE_POINTS[cfg.rate_point]
    iteration_dir = os.path.abspath(cfg.iteration_dir)
    out_name = result_name(cfg)
    result_root = os.path.join(os.path.abspath(cfg.result_dir), out_name)
    compress_dir = os.path.join(result_root, "compression")
    stats_dir = os.path.join(result_root, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    manifest_path = os.path.join(compress_dir, "sogs_manifest.json")
    if cfg.skip_existing and os.path.isfile(manifest_path):
        print(f"[skip] {out_name} already compressed")
        return

    scene_layer, env_layer = load_3dgs_iteration(
        iteration_dir,
        merge_environment=False,
        ply_lod=cfg.ply_lod,
    )
    if cfg.merge_environment and env_layer is not None:
        splats = concat_splats([env_layer, scene_layer])
        print(
            f"Merged env+scene: {int(splats['means'].shape[0])} Gaussians "
            f"(scene={int(scene_layer['means'].shape[0])}, env={int(env_layer['means'].shape[0])})"
        )
    else:
        splats = scene_layer
        print(f"Scene only: {int(splats['means'].shape[0])} Gaussians")

    if os.path.isdir(compress_dir):
        shutil.rmtree(compress_dir)
    os.makedirs(compress_dir, exist_ok=True)

    compressor = ConfigurablePngCompression(
        quantization=rp.quantization,
        n_clusters=rp.n_clusters,
        include_shN=rp.include_shN,
        use_sort=cfg.use_sort,
    )
    tensors = _param_dict_to_tensors(splats)
    compressor.compress(compress_dir, tensors)

    size_bytes = dir_size_bytes(compress_dir)
    manifest = {
        "codec": "sogs_png",
        "rate_point": cfg.rate_point,
        "quantization": rp.quantization,
        "n_clusters": rp.n_clusters,
        "include_shN": rp.include_shN,
        "use_sort": cfg.use_sort,
        "ply_lod": cfg.ply_lod,
        "merge_environment": cfg.merge_environment,
        "iteration_dir": iteration_dir,
        "n_gaussians": int(splats["means"].shape[0]),
        "size_bytes": size_bytes,
        "size_MB": size_bytes / (1024 * 1024),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    stats = {
        "result": out_name,
        "compress_dir": compress_dir,
        **manifest,
    }
    with open(os.path.join(stats_dir, "compress.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote SOGS bitstream: {compress_dir} ({manifest['size_MB']:.2f} MB)")


if __name__ == "__main__":
    tyro.cli(main)
