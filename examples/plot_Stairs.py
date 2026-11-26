"""Plot rate-distortion curves: compressed file size vs quality metrics.

Full-reference metrics (PSNR / SSIM / LPIPS) come from ``summary.json`` written by
``eval_encoded_results.py``. No-reference video metrics (VSFA / DOVER) are read
from an optional sidecar produced by ``score_nr_video_metrics.py``:

    eva/_eval_100views/nr_metrics.json
    {"MellifontAbbey_LOD1_rp0": {"vsfa": 0.63, "dover": 0.71}, ...}

Panels without any data are skipped, so the same script works before and after
the no-reference scores exist.

Run from ``examples/``:

    python plot_rd_curves.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

SUMMARY = Path("eva/AbbeyStreet_Stairs/summary.json") 
NR_METRICS = Path("eva/AbbeyStreet_Stairs/nr_metrics.json")
RESULTS_ROOT = Path("results/AbbeyStreet_Stairs")
OUT_PNG = Path("eva/AbbeyStreet_Stairs/rd_curves.png")
OUT_PDF = Path("eva/AbbeyStreet_Stairs/rd_curves.pdf")
OUT_CSV = Path("eva/AbbeyStreet_Stairs/rd_points.csv")
TITLE = "Rate–distortion: AbbeyStreet_Stairs GSCodec"

NAME_RE = re.compile(r"^(?P<family>[^_]+)_(?P<rp>rp\d+)$")

# Draw order + styling; families not listed fall back to the palette below.
FAMILY_STYLE = {
    "LOD2": {"color": "#2ca02c", "marker": "s"},
    "LOD1": {"color": "#ff7f0e", "marker": "^"}
}
FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
FALLBACK_MARKERS = ["v", "P", "X", "*", "<", ">"]

# key, axis label, lower_is_better
METRICS = [
    ("psnr", "PSNR (dB)", False),
    ("ssim", "SSIM", False),
    ("lpips", "LPIPS", True),
    ("vsfa", "VSFA", False),
    ("dover", "DOVER", False),
]


def compression_size_bytes(name: str) -> int:
    comp = RESULTS_ROOT / name / "compression"
    if not comp.is_dir():
        raise FileNotFoundError(f"Missing compression folder: {comp}")
    return sum(p.stat().st_size for p in comp.rglob("*") if p.is_file())


def load_nr_metrics() -> Dict[str, Dict[str, float]]:
    if not NR_METRICS.is_file():
        return {}
    with open(NR_METRICS) as f:
        return json.load(f)


def style_for(family: str, index: int) -> Dict[str, str]:
    if family in FAMILY_STYLE:
        return FAMILY_STYLE[family]
    return {
        "color": FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        "marker": FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)],
    }


def collect_rows() -> List[Dict[str, Any]]:
    with open(SUMMARY) as f:
        summary = json.load(f)
    nr_scores = load_nr_metrics()

    rows: List[Dict[str, Any]] = []
    unknown_families: List[str] = []
    for name, m in summary["results"].items():
        match = NAME_RE.match(name)
        if not match:
            print(f"[skip] cannot parse family/rate point from {name!r}")
            continue
        family = match.group("family")
        if family not in FAMILY_STYLE and family not in unknown_families:
            unknown_families.append(family)

        row: Dict[str, Any] = {
            "name": name,
            "family": family,
            "rp": match.group("rp"),
            "size_bytes": compression_size_bytes(name),
        }
        row["size_mb"] = row["size_bytes"] / (1024**2)
        for key, _, _ in METRICS:
            value = m.get(key, nr_scores.get(name, {}).get(key))
            row[key] = float(value) if value is not None else None
        rows.append(row)

    for row in rows:
        idx = unknown_families.index(row["family"]) if row["family"] in unknown_families else 0
        row.update(style_for(row["family"], idx))
    return rows


def family_order(rows: List[Dict[str, Any]]) -> List[str]:
    seen = [f for f in FAMILY_STYLE if any(r["family"] == f for r in rows)]
    extra = sorted({r["family"] for r in rows} - set(seen))
    return seen + extra


def write_csv(rows: List[Dict[str, Any]]) -> None:
    metric_keys = [k for k, _, _ in METRICS]
    header = ["name", "family", "rp", "size_MB", "size_bytes", *metric_keys]
    with open(OUT_CSV, "w") as f:
        f.write(",".join(header) + "\n")
        for r in sorted(rows, key=lambda x: x["size_mb"]):
            values = [
                r["name"],
                r["family"],
                r["rp"],
                f"{r['size_mb']:.6f}",
                str(r["size_bytes"]),
            ]
            values += [
                "" if r[k] is None else f"{r[k]:.6f}" for k in metric_keys
            ]
            f.write(",".join(values) + "\n")


def parse_args() -> None:
    global SUMMARY, NR_METRICS, RESULTS_ROOT, OUT_PNG, OUT_PDF, OUT_CSV, TITLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write rd_curves.png/pdf and rd_points.csv (default: next to summary).",
    )
    parser.add_argument("--title", default=TITLE)
    args = parser.parse_args()
    SUMMARY = args.summary
    RESULTS_ROOT = args.results_root
    out_dir = args.out_dir or SUMMARY.parent
    NR_METRICS = out_dir / "nr_metrics.json"
    OUT_PNG = out_dir / "rd_curves.png"
    OUT_PDF = out_dir / "rd_curves.pdf"
    OUT_CSV = out_dir / "rd_points.csv"
    TITLE = args.title


def main() -> None:
    parse_args()
    rows = collect_rows()
    if not rows:
        raise SystemExit(f"No parsable results in {SUMMARY}")
    write_csv(rows)

    active = [m for m in METRICS if any(r[m[0]] is not None for r in rows)]
    missing = [m[1] for m in METRICS if m not in active]
    n_panels = len(active)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True
    )
    flat_axes = list(axes.flat) if n_panels > 1 else [axes]
    families = family_order(rows)

    for ax, (key, ylabel, lower_better) in zip(flat_axes, active):
        for family in families:
            pts = sorted(
                (r for r in rows if r["family"] == family and r[key] is not None),
                key=lambda r: r["size_mb"],
            )
            if not pts:
                continue
            style = pts[0]
            xs = [p["size_mb"] for p in pts]
            ys = [p[key] for p in pts]
            ax.plot(xs, ys, color=style["color"], linewidth=1.5, alpha=0.85, zorder=1)
            ax.scatter(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                s=55,
                label=family,
                edgecolors="white",
                linewidths=0.6,
                zorder=2,
            )
            for p in pts:
                ax.annotate(
                    p["rp"].replace("rp", ""),
                    (p["size_mb"], p[key]),
                    textcoords="offset points",
                    xytext=(4, 3),
                    fontsize=7,
                    color=style["color"],
                )
        ax.set_xlabel("Compressed file size (MB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' (')[0]} {'↓' if lower_better else '↑'}")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    flat_axes[0].legend(frameon=False, loc="best", title="Model")
    subtitle = f"{len(rows)} operating points"

    fig.suptitle(f"{TITLE} ({subtitle})", fontsize=12)
    fig.savefig(OUT_PNG, dpi=200)
    fig.savefig(OUT_PDF)

    print(f"Wrote {OUT_PNG.resolve()}")
    print(f"Wrote {OUT_PDF.resolve()}")
    print(f"Wrote {OUT_CSV.resolve()}")
    if missing:
        print(
            f"No data for: {', '.join(missing)} — run score_nr_video_metrics.py "
            f"to create {NR_METRICS}"
        )
    for r in sorted(rows, key=lambda x: x["size_mb"]):
        parts = [f"{r['name']:32s} {r['size_mb']:8.2f} MB"]
        for key, label, _ in METRICS:
            if r[key] is not None:
                parts.append(f"{label.split(' (')[0]}={r[key]:.4f}")
        print("  ".join(parts))


if __name__ == "__main__":
    main()
