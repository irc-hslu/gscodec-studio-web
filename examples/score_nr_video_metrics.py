"""Score the spiral fly-through videos with no-reference video quality metrics.

Computes VSFA (Quality Assessment of In-the-Wild Videos, ACM MM 2019) and DOVER
(Disentangled Objective Video Quality Evaluator, ICCV 2023) for every result
listed in an ``eval_encoded_results.py`` summary, and writes a sidecar consumed
by ``plot_rd_curves.py``:

    eva/_eval_100views/nr_metrics.json
    {"MellifontAbbey_LOD1_rp0": {"vsfa": ..., "dover": ..., "dover_technical": ...}}

Both models are no-reference, so the scores describe perceived quality of each
decoded fly-through on its own rather than fidelity to the baseline. Absolute
values are only meaningful relative to each other, so keep the sampling flags
identical across a comparison set.

Verified against the reference implementations: DoverScorer reproduces the
distribution of ``evaluate_one_video.py -v demo/17734.mp4 -f``, and the VSFA
path matches ``test_demo.py`` apart from decoding with decord instead of the
numpy-incompatible skvideo (decord and cv2 decode identical pixels).

Setup (once):  bash _setup_nr_metrics.sh
Run:           python score_nr_video_metrics.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DOVER_ROOT = HERE / "third_party" / "DOVER"
VSFA_ROOT = HERE / "third_party" / "VSFA"

DOVER_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
DOVER_STD = torch.FloatTensor([58.395, 57.12, 57.375])
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def require_setup() -> None:
    missing = [p for p in (DOVER_ROOT, VSFA_ROOT) if not p.is_dir()]
    if missing:
        raise SystemExit(
            "Missing model repos: "
            + ", ".join(str(p) for p in missing)
            + "\nRun: bash _setup_nr_metrics.sh"
        )


@contextlib.contextmanager
def pushd(path: Path):
    """DOVER resolves its config's checkpoint path relative to the repo root."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@contextlib.contextmanager
def syspath(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(path))


# --------------------------------------------------------------------------- #
# VSFA
# --------------------------------------------------------------------------- #


class VSFAScorer:
    def __init__(self, device: torch.device, frame_stride: int, batch_size: int):
        from torchvision.models import ResNet50_Weights, resnet50

        with syspath(VSFA_ROOT):
            from VSFA import TP, VSFA  # noqa: N811

        self._TP = TP
        self.device = device
        self.frame_stride = max(1, frame_stride)
        self.batch_size = batch_size

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # VSFA features are mean+std pooled res5c activations.
        self.extractor = torch.nn.Sequential(*list(backbone.children())[:8])
        self.extractor.eval().to(device)

        ckpt = VSFA_ROOT / "models" / "VSFA.pt"
        if not ckpt.is_file():
            raise SystemExit(f"Missing VSFA checkpoint: {ckpt}")
        self.model = VSFA()
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval().to(device)

    @torch.no_grad()
    def _frame_features(self, video_path: Path) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        for batch in iter_frame_batches(video_path, self.frame_stride, self.batch_size):
            x = batch.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            y = self.extractor(x.to(self.device))
            mean = torch.nn.functional.adaptive_avg_pool2d(y, 1)
            std = torch.std(y.view(y.size(0), y.size(1), -1, 1), dim=2, keepdim=True)
            feats.append(torch.cat((mean, std), 1).squeeze(-1).squeeze(-1))
        return torch.cat(feats, 0)

    @torch.no_grad()
    def score(self, video_path: Path) -> float:
        features = self._frame_features(video_path).unsqueeze(0)
        # Reimplements VSFA.forward, whose upstream version calls the removed np.int.
        reduced = self.model.ann(features)
        outputs, _ = self.model.rnn(
            reduced, self.model._get_initial_state(reduced.size(0), reduced.device)
        )
        q = self.model.q(outputs)
        return float(torch.mean(self._TP(q[0])).item())


# --------------------------------------------------------------------------- #
# DOVER
# --------------------------------------------------------------------------- #


class DoverScorer:
    """DOVER's fragment sampler draws random spatial/temporal crops, so a single
    pass has ~0.02 of run-to-run spread on the fused score. Averaging repeats
    shrinks that below the differences we care about between rate points."""

    def __init__(self, device: torch.device, repeats: int = 1):
        import yaml

        with syspath(DOVER_ROOT):
            from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
            from dover.models import DOVER

        self._decompose = spatial_temporal_view_decomposition
        self.device = device
        self.repeats = max(1, repeats)

        with open(DOVER_ROOT / "dover.yml") as f:
            opt = yaml.safe_load(f)
        ckpt = DOVER_ROOT / "pretrained_weights" / "DOVER.pth"
        if not ckpt.is_file():
            raise SystemExit(f"Missing DOVER checkpoint: {ckpt}")

        self.model = DOVER(**opt["model"]["args"]).to(device)
        self.model.load_state_dict(torch.load(ckpt, map_location=device))
        self.model.eval()

        # The 1080p profile matches the resolution of our rendered fly-throughs.
        self.sample_types = opt["data"]["val-l1080p"]["args"]["sample_types"]
        self.samplers = {}
        for stype, sopt in self.sample_types.items():
            if "t_frag" not in sopt:
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                self.samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

    @staticmethod
    def fuse(technical: float, aesthetic: float) -> float:
        """DOVER's default score-level fusion, squashed to [0, 1]."""
        x = (technical - 0.1107) / 0.07355 * 0.6104
        x += (aesthetic + 0.08285) / 0.03774 * 0.3896
        return float(1.0 / (1.0 + np.exp(-x)))

    @torch.no_grad()
    def _score_once(self, video_path: Path) -> Tuple[float, float, float]:
        with pushd(DOVER_ROOT):
            views, _ = self._decompose(
                str(video_path), self.sample_types, self.samplers
            )
        for key, v in views.items():
            num_clips = self.sample_types[key].get("num_clips", 1)
            views[key] = (
                ((v.permute(1, 2, 3, 0) - DOVER_MEAN) / DOVER_STD)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )
        technical, aesthetic = (r.mean().item() for r in self.model(views))
        return self.fuse(technical, aesthetic), technical, aesthetic

    def score(self, video_path: Path) -> Dict[str, float]:
        draws = [self._score_once(video_path) for _ in range(self.repeats)]
        fused, technical, aesthetic = (np.array(c) for c in zip(*draws))
        return {
            "dover": float(fused.mean()),
            "dover_std": float(fused.std()),
            "dover_technical": float(technical.mean()),
            "dover_aesthetic": float(aesthetic.mean()),
            "dover_repeats": self.repeats,
        }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def iter_frame_batches(video_path: Path, stride: int, batch_size: int):
    """Yield uint8 NHWC batches; streaming keeps 1080p clips out of a single blob."""
    import decord

    reader = decord.VideoReader(str(video_path), num_threads=4)
    idx = list(range(0, len(reader), max(1, stride)))
    for start in range(0, len(idx), batch_size):
        batch = reader.get_batch(idx[start : start + batch_size])
        # DOVER sets decord's global bridge to torch, so handle either return type.
        yield batch if torch.is_tensor(batch) else torch.from_numpy(batch.asnumpy())


def to_local_path(path_str: str) -> Path:
    """Summary paths were written from WSL; fall back to the Windows drive form."""
    p = Path(path_str)
    if p.exists():
        return p
    if path_str.startswith("/mnt/") and len(path_str) > 6:
        drive = path_str[5]
        alt = Path(f"{drive.upper()}:/" + path_str[7:])
        if alt.exists():
            return alt
    return p


def collect_videos(summary_path: Path) -> Dict[str, Path]:
    with open(summary_path) as f:
        summary = json.load(f)
    videos: Dict[str, Path] = {}
    for name, entry in summary["results"].items():
        mp4 = entry.get("spiral_mp4")
        if mp4:
            videos[name] = to_local_path(mp4)
    baseline = summary.get("spiral_video", {}).get("baseline_mp4")
    if baseline:
        videos["baseline"] = to_local_path(baseline)
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="eva/_eval_100views/summary.json")
    parser.add_argument("--out", default="eva/_eval_100views/nr_metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--vsfa-frame-stride",
        type=int,
        default=2,
        help="Sample every Nth frame for VSFA (default: 2, i.e. 300 of 600 frames)",
    )
    parser.add_argument("--vsfa-batch-size", type=int, default=8)
    parser.add_argument(
        "--dover-repeats",
        type=int,
        default=8,
        help="Average this many random sampler draws per video (default: 8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rescore videos already present in the output file",
    )
    args = parser.parse_args()

    require_setup()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    videos = collect_videos(Path(args.summary))
    if not videos:
        raise SystemExit(f"No spiral videos referenced by {args.summary}")

    out_path = Path(args.out)
    scores: Dict[str, Dict[str, float]] = {}
    if out_path.is_file() and not args.overwrite:
        with open(out_path) as f:
            scores = json.load(f)

    print(f"Loading models on {device}...")
    vsfa = VSFAScorer(device, args.vsfa_frame_stride, args.vsfa_batch_size)
    dover = DoverScorer(device, repeats=args.dover_repeats)

    for i, (name, path) in enumerate(sorted(videos.items()), start=1):
        if name in scores and not args.overwrite:
            print(f"[{i}/{len(videos)}] {name}: cached, skipping")
            continue
        if not path.is_file():
            print(f"[{i}/{len(videos)}] {name}: MISSING {path}")
            continue

        vsfa_score = vsfa.score(path)
        entry = {"vsfa": vsfa_score, **dover.score(path)}
        scores[name] = entry
        print(
            f"[{i}/{len(videos)}] {name}: VSFA={vsfa_score:.4f} "
            f"DOVER={entry['dover']:.4f}±{entry['dover_std']:.4f}"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2, sort_keys=True)

    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
