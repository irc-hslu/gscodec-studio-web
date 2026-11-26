#!/usr/bin/env python3
"""Generate a WG1-style DOCX for Proposed Heritage Test Material (JPEG RF).

Mirrors the structure of M111041 CTC §2 Test Material. Reads coding results from
examples/eva/<scene>/{rd_points.csv,summary.json,nr_metrics.json}.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
EVA = ROOT / "examples" / "eva"
OUT = ROOT / "docs" / "jpeg" / "Proposed_Heritage_Test_Material_for_JPEG_RF.docx"
FIGDIR = ROOT / "docs" / "jpeg" / "figures"

QP_LADDER = {
    "rp0": ["16", "22", "22", "16", "28 / 34 / 40"],
    "rp1": ["10", "16", "16", "10", "22 / 28 / 34"],
    "rp2": ["4", "10", "10", "4", "16 / 22 / 28"],
    "rp3": ["4", "4", "4", "4", "4 / 4 / 4"],
}

FAMILY_STYLE = {
    "lod0": {"color": "#d62728", "marker": "D", "label": "LOD0"},
    "full": {"color": "#d62728", "marker": "D", "label": "LOD0 (full)"},
    "lod1": {"color": "#ff7f0e", "marker": "^", "label": "LOD1"},
    "lod2": {"color": "#2ca02c", "marker": "s", "label": "LOD2"},
    "lod3": {"color": "#1f77b4", "marker": "o", "label": "LOD3"},
}

SCENES = [
    {
        "name": "MellifontAbbey",
        "eva": "MellifontAbbey_1",
        "location": "Outdoors (heritage abbey)",
        "eval_stills": "104 concentric",
        "eval_views": 104,
        "gaussians_txt": "3 735 339 scene + 81 597 env ≈ 3 816 936",
        "lod": [
            ["point_cloud.ply (full / LOD0)", "3 735 339", "Highest detail"],
            ["point_cloud_1.ply (LOD1)", "1 844 208", "Medium"],
            ["point_cloud_2.ply (LOD2)", "903 520", "Low–medium"],
            ["point_cloud_3.ply (LOD3)", "439 488", "Lowest"],
            ["environment.ply", "81 597", "Environment shell"],
        ],
        "blurb": (
            "Outdoor-scale cultural-heritage abbey reconstruction. Scene Gaussians are separated from "
            "a surrounding environment shell so that coding experiments may apply split-layer "
            "normalization (scene and environment encoded independently, merged at decode)."
        ),
        "eval_note": (
            "Eval still set: 104 concentric views, resolution 1920×1080. Baseline Gaussians include "
            "the scene layer and the environment shell (3 816 936)."
        ),
        "table_lod": "2a",
        "table_rd": "2h",
        "figure_rd": 5,
    },
    {
        "name": "AbbeyStreet_UpperFloor",
        "eva": "AbbeyStreet_UpperFloor",
        "location": "Indoors (upper-floor interior)",
        "eval_stills": "100 (10×10)",
        "eval_views": 100,
        "gaussians_txt": "1 541 667 scene (+ 8 197 env optional)",
        "lod": [
            ["lod0/point_cloud.ply", "1 541 667", "Full"],
            ["lod1/point_cloud_1.ply", "771 839", "Medium"],
            ["lod2/point_cloud_2.ply", "385 893", "Low"],
            ["environment.ply", "8 197", "Optional environment"],
        ],
        "blurb": (
            "Indoor upper-floor heritage interior. Smaller environment shell; evaluation uses the "
            "full scene PLY as baseline (environment optional)."
        ),
        "eval_note": (
            "Eval still set: 100 views, 10 rings × 10 cameras, ring radii 0.5–5.0 m, ring height "
            "step 0.1 m, resolution 1920×1080."
        ),
        "table_lod": "2b",
        "table_rd": "2i",
        "figure_rd": 6,
    },
    {
        "name": "AbbeyStreet_LowerFloor",
        "eva": "AbbeyStreet_LowerFloor",
        "location": "Indoors (lower-floor interior)",
        "eval_stills": "100 (10×10)",
        "eval_views": 100,
        "gaussians_txt": "1 766 302 scene (+ 1 363 env optional)",
        "lod": [
            ["lod0/point_cloud.ply", "1 766 302", "Full"],
            ["lod1/point_cloud_1.ply", "882 927", "Medium"],
            ["lod2/point_cloud_2.ply", "441 429", "Low"],
            ["environment.ply", "1 363", "Optional environment"],
        ],
        "blurb": (
            "Indoor lower-floor heritage interior (same Abbey Street site as UpperFloor). Evaluation "
            "uses the full scene LOD0 PLY as baseline; a small environment shell is available optionally."
        ),
        "eval_note": (
            "Eval still set: 100 views, 10 rings × 10 cameras, ring radii 0.5–5.0 m, ring height "
            "step 0.1 m, resolution 1920×1080."
        ),
        "table_lod": "2c",
        "table_rd": "2j",
        "figure_rd": 7,
    },
    {
        "name": "AbbeyStreet_Stairs",
        "eva": "AbbeyStreet_Stairs",
        "location": "Indoors (stairwell)",
        "eval_stills": "100 (10×10)",
        "eval_views": 100,
        "gaussians_txt": "875 735 scene (+ 836 env optional)",
        "lod": [
            ["lod0/point_cloud.ply", "875 735", "Full"],
            ["lod1/point_cloud_1.ply", "437 794", "Medium"],
            ["environment.ply", "836", "Optional environment"],
        ],
        "blurb": (
            "Indoor stairwell of the same Abbey Street site. The reconstructed volume is narrower "
            "than the floor interiors; two LODs are provided. Evaluation uses the full scene LOD0 "
            "PLY as baseline."
        ),
        "eval_note": (
            "Eval still set: 100 views, 10 rings × 10 cameras, ring radii 0.45–4.5 m, ring height "
            "step 0.1 m, resolution 1920×1080."
        ),
        "table_lod": "2d",
        "table_rd": "2k",
        "figure_rd": 8,
    },
]


def set_run_font(run, size=11, bold=False, italic=False, name="Times New Roman"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = name
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)


def add_para(doc, text, size=11, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    return p


def add_heading_custom(doc, text, level=1):
    sizes = {1: 14, 2: 12, 3: 11}
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    set_run_font(run, size=sizes.get(level, 11), bold=True)
    return p


def add_meta_line(doc, label, value):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r1 = p.add_run(f"{label}: ")
    set_run_font(r1, bold=True, size=11)
    r2 = p.add_run(value)
    set_run_font(r2, size=11)


def add_table(doc, headers, rows, col_cm=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        run = p.add_run(h)
        set_run_font(run, bold=True, size=9)
    for r_i, row in enumerate(rows):
        cells = table.rows[r_i + 1].cells
        for c_i, val in enumerate(row):
            cells[c_i].text = ""
            p = cells[c_i].paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=9)
    if col_cm:
        for row in table.rows:
            for i, w in enumerate(col_cm):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


def _fmt(val, spec):
    if val is None or val == "":
        return "–"
    return f"{float(val):{spec}}"


def family_label(family: str) -> str:
    f = family.lower()
    if f == "full":
        return "LOD0"
    return family.upper() if family.lower().startswith("lod") else family.upper()


def load_points(scene):
    eva = EVA / scene["eva"]
    with open(eva / "rd_points.csv") as f:
        points = list(csv.DictReader(f))
    for p in points:
        p["family_key"] = p["family"].lower()
        p["lod_sort"] = {"full": 0, "lod0": 0, "lod1": 1, "lod2": 2, "lod3": 3}.get(p["family_key"], 9)
        p["rp_sort"] = p["rp"]
    points.sort(key=lambda r: (r["lod_sort"], r["rp_sort"]))
    nr = {}
    if (eva / "nr_metrics.json").is_file():
        with open(eva / "nr_metrics.json") as f:
            nr = json.load(f)
    summary = {}
    if (eva / "summary.json").is_file():
        with open(eva / "summary.json") as f:
            summary = json.load(f)
    return points, nr, summary


def generate_rd_png(scene, points, out_path: Path):
    metrics = [
        ("psnr", "PSNR (dB)", False),
        ("ssim", "SSIM", False),
        ("lpips", "LPIPS", True),
        ("vsfa", "VSFA", False),
        ("dover", "DOVER", False),
    ]
    active = [m for m in metrics if any((r.get(m[0]) or "") != "" for r in points)]
    n = len(active)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), constrained_layout=True)
    flat = list(axes.flat) if n > 1 else [axes]
    families = []
    for r in points:
        if r["family_key"] not in families:
            families.append(r["family_key"])
    for ax, (key, ylabel, lower) in zip(flat, active):
        for fam in families:
            pts = [p for p in points if p["family_key"] == fam and (p.get(key) or "") != ""]
            pts = sorted(pts, key=lambda p: float(p["size_MB"]))
            if not pts:
                continue
            style = FAMILY_STYLE.get(fam, {"color": "#9467bd", "marker": "v", "label": fam})
            xs = [float(p["size_MB"]) for p in pts]
            ys = [float(p[key]) for p in pts]
            ax.plot(xs, ys, color=style["color"], linewidth=1.5, alpha=0.85, zorder=1)
            ax.scatter(
                xs, ys, color=style["color"], marker=style["marker"], s=55,
                label=style["label"], edgecolors="white", linewidths=0.6, zorder=2,
            )
            for p in pts:
                ax.annotate(
                    p["rp"].replace("rp", ""), (float(p["size_MB"]), float(p[key])),
                    textcoords="offset points", xytext=(4, 3), fontsize=7, color=style["color"],
                )
        ax.set_xlabel("Compressed file size (MB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' (')[0]} {'↓' if lower else '↑'}")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    for ax in flat[n:]:
        ax.axis("off")
    flat[0].legend(frameon=False, loc="best", title="LOD")
    fig.suptitle(f"Rate–distortion: {scene['name']} (x265 video anchor)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def add_scene_results(doc, scene):
    points, nr, summary = load_points(scene)
    res = f"{summary.get('width', 1920)}×{summary.get('height', 1080)}"
    nviews = summary.get("num_views", scene["eval_views"])
    has_nr = any((r.get("vsfa") or "") != "" for r in points)

    add_heading_custom(doc, f"2.{7 + SCENES.index(scene)} Example coding results – {scene['name']}", 2)
    nr_sentence = (
        "VSFA and DOVER are no-reference scores of the 180-frame spiral video (higher is better)."
        if has_nr
        else (
            "No-reference video scores (VSFA, DOVER) were not computed for this scene; the table "
            "reports full-reference still metrics only."
        )
    )
    add_para(
        doc,
        f"The {scene['name']} LODs were coded with the same GSCodec Studio x265 / HEVC video "
        f"anchor and QP ladder (Table 2g). Objective quality is measured on the {nviews} evaluation "
        f"stills ({res}) against renders of the uncompressed full scene PLY, so reduced-LOD points "
        f"include the LOD reduction loss in addition to coding loss. {nr_sentence}",
    )
    add_para(
        doc,
        f"Table {scene['table_rd']} – {scene['name']} coding results ({res}, {nviews} views)",
        size=10,
        italic=True,
    )
    rows = []
    for r in points:
        rows.append([
            family_label(r["family"]),
            r["rp"],
            _fmt(r["size_MB"], ".2f"),
            _fmt(r["psnr"], ".2f"),
            _fmt(r["ssim"], ".4f"),
            _fmt(r["lpips"], ".4f"),
            _fmt(r.get("vsfa"), ".3f"),
            _fmt(r.get("dover"), ".3f"),
        ])
    if "baseline" in nr:
        rows.append([
            "Uncompressed", "–", "–", "∞", "1", "0",
            f"{nr['baseline']['vsfa']:.3f}", f"{nr['baseline']['dover']:.3f}",
        ])
    add_table(doc, ["LOD", "Rate point", "Size (MB)", "PSNR (dB)", "SSIM", "LPIPS", "VSFA", "DOVER"], rows)

    png = FIGDIR / f"{scene['name']}_rd_curves_1920x1080.png"
    generate_rd_png(scene, points, png)
    eva_png = EVA / scene["eva"] / "rd_curves.png"
    if eva_png.parent.is_dir():
        generate_rd_png(scene, points, eva_png)
    doc.add_picture(str(png), width=Cm(16.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_metrics = "PSNR, SSIM, LPIPS, VSFA and DOVER" if has_nr else "PSNR, SSIM and LPIPS"
    add_para(
        doc,
        f"Figure {scene['figure_rd']} – Rate–distortion curves for {scene['name']} ({res}): "
        f"compressed size vs {caption_metrics} per LOD; labels give the rate point.",
        size=10,
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
    )


def build() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    add_para(doc, "ISO/IEC JTC 1/SC 29/WG 1", size=12, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "(ITU-T SG21)  Coding of Still Pictures", size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "JBIG / JPEG", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=18)

    add_para(doc, "Mxxxxxx", size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(
        doc,
        "[Nth JPEG Meeting – Venue – Dates]",
        size=11,
        italic=True,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        space_after=18,
    )

    add_meta_line(doc, "TITLE", "Proposed Heritage Test Material for JPEG Radiance Fields (Coding-Only Evaluation)")
    add_meta_line(doc, "SOURCE", "[Author Name] ([Affiliation])")
    add_meta_line(doc, "PROJECT", "JPEG RF")
    add_meta_line(doc, "STATUS", "Draft / Informative")
    add_meta_line(doc, "REQUESTED ACTION", "Discussion")
    add_meta_line(doc, "DISTRIBUTION", "WG1")
    doc.add_paragraph()
    add_para(
        doc,
        "Contact: ISO/IEC JTC 1/SC 29/WG 1 Convener – Prof. Touradj Ebrahimi, "
        "EPFL/STI/IEL/GR-EB, Station 11, CH-1015 Lausanne, Switzerland",
        size=9,
        italic=True,
    )
    doc.add_page_break()

    add_heading_custom(doc, "1 Scope", 1)
    add_para(
        doc,
        "This document describes heritage radiance-field test material proposed for JPEG Radiance "
        "Fields (JPEG RF) experiments, with emphasis on the coding-only assessment pipeline "
        "(CTC Case A): an already instantiated 3D Gaussian Splatting (3DGS) model is the fixed "
        "target representation to be encoded. The material complements the public CTC sources "
        "(Tanks and Temples, Mip-NeRF 360, Den-SOFT) by providing large-scale cultural-heritage "
        "scenes with multi-LOD Gaussian assets, optional environment shells, fixed evaluation "
        "cameras, and passive visualization trajectories.",
    )

    add_heading_custom(doc, "2 Test Material", 1)
    add_para(
        doc,
        "This section describes the proposed test material. Scenes are selected from one source "
        "collection with shared reconstruction conventions. A short description of the source "
        "dataset is given in Table 1.",
    )

    add_para(doc, "Table 1 – Source dataset selected for heritage radiance-field evaluation", size=10, italic=True)
    add_table(
        doc,
        ["Dataset", "Acquisition / capture", "Representation", "Number of test scenes"],
        [
            [
                "Heritage Radiance Fields (HRF)",
                "Photogrammetry / multi-view capture of cultural-heritage sites",
                "Instantiated 3DGS (INRIA-compatible PLY), SH degree 3",
                "4",
            ]
        ],
    )

    add_para(doc, "For each test scene, the following material is provided:", size=11)
    bullets = [
        "Instantiated radiance-field model — one or more 3DGS PLY files (full resolution and LOD variants), optionally with a separate environment shell PLY.",
        "Reconstruction metadata — spherical-harmonics degree and background flag (sh_degree=3, white_background=True).",
        "Evaluation camera poses — camera-to-world matrices and intrinsics for objective still-image assessment (concentric multi-ring paths).",
        "Passive visualization trajectories — spiral/helix paths and JPEG RF Exploration Study 5–style key/ghost trajectories for subjective fly-through assessment.",
        "Reference renderings — views rendered from the uncompressed model at the evaluation poses (coding-only reference), where applicable.",
    ]
    for b in bullets:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(b)
        set_run_font(run, size=11)

    add_para(
        doc,
        "NOTE — Unlike CTC outdoor/indoor photo datasets that ship raw training images and COLMAP "
        "sparse models, this contribution targets Case A (coding-only). Training multi-view images "
        "and SfM reconstructions may be distributed under a separate license; the normative "
        "compressed/uncompressed comparison uses the fixed PLY models and provided poses.",
        size=10,
        italic=True,
    )

    add_para(doc, "Table 2 – Test scenes selected for evaluation", size=10, italic=True)
    add_table(
        doc,
        ["Test scene", "Source", "Location", "Eval resolution", "Eval stills", "Baseline Gaussians", "SH"],
        [
            [s["name"], "HRF", s["location"].split("(")[0].strip(), "1920×1080", s["eval_stills"], s["gaussians_txt"], "3"]
            for s in SCENES
        ],
    )

    add_para(
        doc,
        "Figure 1–4 — Example reference renderings for MellifontAbbey, AbbeyStreet_UpperFloor, "
        "AbbeyStreet_LowerFloor, and AbbeyStreet_Stairs (insert baseline still frames).",
        size=10,
        italic=True,
    )

    for i, s in enumerate(SCENES, 1):
        add_heading_custom(doc, f"2.{i} {s['name']}", 2)
        add_para(doc, s["blurb"] + " " + s["eval_note"])
        add_para(doc, f"Table {s['table_lod']} – {s['name']} LOD inventory", size=10, italic=True)
        add_table(doc, ["Asset", "# Gaussians", "Role"], s["lod"])

    add_heading_custom(doc, "2.5 Material to be employed (coding-only)", 2)
    add_para(doc, "Table 2e – Normative usage for JPEG RF coding-only experiments", size=10, italic=True)
    add_table(
        doc,
        ["Item", "Specification"],
        [
            ["Target format", "Instantiated 3DGS PLY (means, scales, quats, opacity, SH)"],
            ["Reference model", "Uncompressed PLY (± environment as declared)"],
            ["Objective still poses", "Fixed cameras_eval_*.json (c2w + intrinsics)"],
            ["Objective metrics", "PSNR, SSIM/MS-SSIM, LPIPS; CTC FR metrics where applicable"],
            ["Passive trajectories", "Fixed FPS/duration; spiral and/or JPEG RF ES5 paths"],
            ["Subjective protocol", "Double-stimulus vs. uncompressed reference (CTC §3.3.3)"],
            ["Rate points", "LOD × QP/λ ladder; report size (MB)"],
        ],
    )
    add_para(
        doc,
        "Orbit/trajectory centers shall be computed from the scene-layer PLY centroid when an "
        "environment shell is present.",
        size=11,
    )

    add_heading_custom(doc, "2.6 Compression method", 2)
    add_para(
        doc,
        "Every result in §2.7–§2.10 was produced with the same coder: the GSCodec Studio sequential "
        "HEVC anchor (compress_ply_sequence.py presets x265_compression_rp0 to x265_compression_rp3, "
        "compression=\"seq_hevc\"). Each LOD is a single static 3DGS PLY, passed with one frame and "
        "GOP size 16. The environment shell is not part of the coded bitstream; sizes below are the "
        "scene layer only.",
    )
    add_para(
        doc,
        "The coder follows the self-organizing Gaussian grid approach. Gaussians are reordered with "
        "PLAS so that nearby primitives become neighbours in a 2D grid, then reshaped into a square "
        "image. If the Gaussian count is not a square, the grid is padded and the padded primitives "
        "are dropped at decode. Each attribute is then min–max normalized over that grid and "
        "uniformly quantized: positions (means) to 16 bits, and opacity, scale, quaternion, and "
        "spherical-harmonic coefficients to 8 bits. The stored minima and maxima invert the "
        "normalization after decode. No colour-space conversion and no quaternion-to-Euler "
        "conversion are applied at these rate points.",
    )
    add_para(
        doc,
        "The quantized maps are coded with libx265 (HEVC) through FFmpeg and stored as MP4. With one "
        "frame per asset, each bitstream is a single intra picture; all-intra mode was not enabled. "
        "Rate is set only by the constant QP in Table 2g. rp0 is the lowest rate and rp3 the highest. "
        "The same ladder is used for every scene and every LOD.",
    )
    add_para(doc, "Table 2f – Attribute coding in the sequential HEVC anchor", size=10, italic=True)
    add_table(
        doc,
        ["Attribute", "Quantization", "HEVC representation"],
        [
            ["Means (x, y, z)", "16-bit", "Two byte planes, lossless (x265 lossless=1, preset veryslow; configured QP −1)"],
            ["Opacity", "8-bit", "One gray plane, constant QP"],
            ["Scale", "8-bit", "One 3-channel map, constant QP"],
            ["Quaternion (w, x, y, z)", "8-bit", "Gray plane for w and a 3-channel map for (x, y, z), same QP"],
            ["SH DC (degree 0)", "8-bit", "One 3-channel map, constant QP"],
            ["SH degrees 1–3", "8-bit", "One 3-channel map per coefficient (15 maps); QP set per degree"],
        ],
    )
    add_para(doc, "Table 2g – x265 QP ladder per rate point (means coded as 16-bit, lossless)", size=10, italic=True)
    add_table(
        doc,
        ["Rate point", "Opacity", "Quaternions", "Scales", "SH DC", "SH bands 1 / 2 / 3"],
        [[rp, *qps] for rp, qps in QP_LADDER.items()],
    )

    for s in SCENES:
        add_scene_results(doc, s)

    add_heading_custom(doc, "3 Camera path design", 1)
    add_para(
        doc,
        "Heritage evaluation uses three complementary path families, all rendered at 1920×1080 with "
        "fx = fy = 1080 px, cx = 960, cy = 540 (unless a study explicitly rescales). Camera-to-world "
        "poses follow the gsplat / COLMAP convention used in examples/eval_encoded_results.py.",
    )
    add_heading_custom(doc, "3.1 Objective still viewpoints (FR metrics)", 2)
    add_para(
        doc,
        "Full-reference PSNR / SSIM / LPIPS are computed on a fixed set of static poses generated by "
        "concentric horizontal rings around a scene look-at centre (generate_concentric_c2w). Ring r "
        "(0-based) has radius (r + 1)·radius_step_m and height height_offset_m + r·ring_height_step_m; "
        "views_per_ring azimuth samples are evenly spaced. Each camera looks at the centre, with an "
        "optional pitch-down rotation about the local X axis (0° for all heritage runs reported here). "
        "When scene and environment splats are merged for rendering, the orbit centre is taken from the "
        "scene-layer PLY centroid (--center-from-ply), not from the merged cloud, so rings stay inside "
        "the architectural volume.",
    )
    add_para(doc, "Table 3 – Per-scene objective ring parameters", size=10, italic=True)
    add_table(
        doc,
        [
            "Scene",
            "Views",
            "Rings × per ring",
            "radius_step (m)",
            "ring height step (m)",
            "Pitch (°)",
            "Saved JSON",
        ],
        [
            [
                "MellifontAbbey",
                "104",
                "13 × 8",
                "1.2",
                "0.2",
                "0",
                "eva/_eval_100views/cameras_100.json",
            ],
            [
                "AbbeyStreet UpperFloor",
                "100",
                "10 × 10",
                "0.5",
                "0.1",
                "0",
                "eva/AbbeyStreet_UpperFloor/cameras_100.json",
            ],
            [
                "AbbeyStreet LowerFloor",
                "100",
                "10 × 10",
                "0.45",
                "0.1",
                "0",
                "eva/AbbeyStreet_LowerFloor/cameras_100.json",
            ],
            [
                "AbbeyStreet Stairs",
                "100",
                "10 × 10",
                "0.45",
                "0.1",
                "0",
                "eva/AbbeyStreet_Stairs/cameras_100.json",
            ],
        ],
    )
    add_heading_custom(doc, "3.2 Spiral fly-through (NR video metrics)", 2)
    add_para(
        doc,
        "No-reference VSFA and DOVER scores in §2 use a smooth helix derived from the objective ring "
        "set (generate_spiral_c2w_from_cameras): radius and height interpolate linearly from the innermost "
        "ring (camera 0) to the outermost (last camera), while azimuth sweeps spiral_rots full turns "
        "(2.0) before landing on the last pose. Orientation matches the ring generator (look-at centre, "
        "same pitch-down). Abbey Street scenes use 180 frames at 25 fps (7.2 s); MellifontAbbey uses "
        "600 frames at 25 fps (24 s) for the reported spiral baseline.",
    )
    add_heading_custom(doc, "3.3 Viewer-authored path (AbbeyStreet Stairs)", 2)
    add_para(
        doc,
        "For longer subjective clips along a stair-focused route, optional renders use key poses saved "
        "from the viser viewer (render_camera_path.py): centripetal Catmull–Rom position interpolation "
        "with arc-length reparameterization and quaternion SLERP, 600 frames at 25 fps, output under "
        "eva/AbbeyStreet_Stairs/camera_path/. This path is independent of the concentric ring set used "
        "for Tables 2h–2k.",
    )
    add_heading_custom(doc, "3.4 JPEG RF ES5 subjective trajectories", 2)
    add_para(
        doc,
        "Passive-visualization trajectories for JPEG RF Exploration Study 5 and CTC §3.2.2 are built "
        "from training / evaluation key poses via examples/generate_jpeg_rf_trajectory.py and "
        "jpeg_rf_trajectory.py: key views balancing spatial coverage and directional diversity; "
        "optional ghost (virtual) viewpoints; centripetal Catmull–Rom with arc-length reparameterization; "
        "quaternion SLERP. Default clip length: 600 frames at 25 fps (24 s).",
    )
    add_para(doc, "Table 4 – Example ES5 trajectory configurations (MellifontAbbey)", size=10, italic=True)
    add_table(
        doc,
        ["Preset", "K", "G", "α", "Notes"],
        [
            ["fps_keys8", "8", "0", "1.0", "Spatial coverage only"],
            ["coverage_keys12", "12", "0", "0.65", "Wide spatial sweep"],
            ["coverage_orient_keys8", "8", "0", "0.4", "Orientation diversity"],
            ["ghost_gentle_orient (recommended)", "8", "4", "0.8", "Generalization"],
            ["ghost_strong_orient", "8", "5", "0.65", "View-dependent stress"],
            ["ghost_strong_span", "9", "6", "0.7", "Parallax / occlusion"],
        ],
    )
    add_para(
        doc,
        "Machine-readable presets: examples/configs/jpeg_rf_trajectories.json. Reproducible generation: "
        "python examples/generate_jpeg_rf_trajectory.py --trajectory ghost_gentle_orient.",
    )

    add_heading_custom(doc, "4 Relation to Common Test Conditions", 1)
    add_table(
        doc,
        ["CTC element", "Heritage material"],
        [
            ["Pipeline", "Case A – coding-only (fixed instantiated 3DGS)"],
            ["Public CTC scenes", "Complementary (not a replacement)"],
            ["FR reference", "Uncompressed PLY renderings at shared poses/trajectories"],
            ["Novel-view trajectories", "Spiral + ES5 paths; FR metrics only where references exist"],
        ],
    )

    add_heading_custom(doc, "5 Distribution and licensing", 1)
    add_para(
        doc,
        "[To be completed: download URL / license / contact email. Training photographs may be "
        "released separately. Reconstruction: 3DGS, sh_degree=3.]",
        italic=True,
    )

    add_heading_custom(doc, "6 References", 1)
    refs = [
        "ISO/IEC JTC 1/SC 29/WG 1, Proposed Common Test Conditions for JPEG Radiance Fields, M111041, v0.6, April 2026.",
        "S. Mahmoudpour, Report on Exploration Study 5 of JPEG RF, M109040, Oct 2025.",
        "WG1, Exploration Study 7 of JPEG Radiance Fields, N101414.",
        "WG1, Use Cases and Requirements for JPEG RF, N101398, v1.0.",
        "B. Kerbl et al., “3D Gaussian Splatting for Real-Time Radiance Field Rendering,” ACM Trans. Graph., 2023.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        run = p.add_run(f"[{i}] {r}")
        set_run_font(run, size=10)

    return doc


def md_table_rd(scene):
    points, nr, _ = load_points(scene)
    lines = [
        f"### Table {scene['table_rd']} – {scene['name']} coding results (1920 × 1080, {scene['eval_views']} views)",
        "",
        "| LOD | Rate point | Size (MB) | PSNR (dB) | SSIM | LPIPS | VSFA | DOVER |",
        "|-----|------------|-----------|-----------|------|-------|------|-------|",
    ]
    for r in points:
        lines.append(
            f"| {family_label(r['family'])} | {r['rp']} | {_fmt(r['size_MB'], '.2f')} | "
            f"{_fmt(r['psnr'], '.2f')} | {_fmt(r['ssim'], '.4f')} | {_fmt(r['lpips'], '.4f')} | "
            f"{_fmt(r.get('vsfa'), '.3f')} | {_fmt(r.get('dover'), '.3f')} |"
        )
    if "baseline" in nr:
        lines.append(
            f"| Uncompressed | – | – | ∞ | 1 | 0 | "
            f"{nr['baseline']['vsfa']:.3f} | {nr['baseline']['dover']:.3f} |"
        )
    has_nr = any((r.get("vsfa") or "") != "" for r in points)
    cap = "PSNR, SSIM, LPIPS, VSFA and DOVER" if has_nr else "PSNR, SSIM and LPIPS"
    lines += [
        "",
        f"![Figure {scene['figure_rd']} – Rate–distortion curves for {scene['name']}]"
        f"(figures/{scene['name']}_rd_curves_1920x1080.png)",
        "",
        f"**Figure {scene['figure_rd']} –** Rate–distortion curves for {scene['name']} (1920 × 1080): "
        f"compressed size vs {cap} per LOD; labels give the rate point.",
        "",
        "",
    ]
    return "\n".join(lines)


def write_markdown():
    md_path = ROOT / "docs" / "jpeg" / "Proposed_Heritage_Test_Material_for_JPEG_RF.md"
    scene_rows = "\n".join(
        f"| {s['name']} | HRF | {s['location']} | 1920 × 1080 | {s['eval_stills']} | {s['gaussians_txt']} | 3 |"
        for s in SCENES
    )
    lod_sections = []
    for i, s in enumerate(SCENES, 1):
        rows = "\n".join(f"| `{a}` | {n} | {r} |" for a, n, r in s["lod"])
        lod_sections.append(
            f"### 2.1.{i} {s['name']}\n\n{s['blurb']}\n\n**LOD inventory:**\n\n"
            f"| Asset | Approx. # Gaussians | Role |\n|-------|---------------------|------|\n{rows}\n\n"
            f"{s['eval_note']}\n"
        )
    rd_blocks = []
    for s in SCENES:
        points, _, _ = load_points(s)
        has_nr = any((r.get("vsfa") or "") != "" for r in points)
        nr_sentence = (
            "VSFA and DOVER are no-reference scores of the 180-frame spiral video (higher is better)."
            if has_nr
            else "No-reference video scores (VSFA, DOVER) were not computed for this scene."
        )
        rd_blocks.append(
            f"## 2.{7 + SCENES.index(s)} Example coding results – {s['name']}\n\n"
            f"The {s['name']} LODs were coded with the GSCodec Studio x265 / HEVC video anchor and "
            f"the QP ladder in Table 2g. Objective quality is measured on the {s['eval_views']} "
            f"evaluation stills (**1920 × 1080**) against renders of the uncompressed full scene PLY, "
            f"so reduced-LOD points include the LOD reduction loss in addition to coding loss. "
            f"{nr_sentence}\n\n{md_table_rd(s)}"
        )
    annex = {
        "dataset": "Heritage Radiance Fields (HRF)",
        "pipeline": "coding_only_case_A",
        "scenes": [
            {
                "name": "MellifontAbbey",
                "location": "outdoors",
                "sh_degree": 3,
                "eval_resolution": [1920, 1080],
                "eval_views": 104,
                "gaussians_scene": 3735339,
                "gaussians_environment": 81597,
                "lod_scene": [3735339, 1844208, 903520, 439488],
            },
            {
                "name": "AbbeyStreet_UpperFloor",
                "location": "indoors",
                "sh_degree": 3,
                "eval_resolution": [1920, 1080],
                "eval_views": 100,
                "gaussians_scene": 1541667,
                "gaussians_environment": 8197,
                "lod_scene": [1541667, 771839, 385893],
            },
            {
                "name": "AbbeyStreet_LowerFloor",
                "location": "indoors",
                "sh_degree": 3,
                "eval_resolution": [1920, 1080],
                "eval_views": 100,
                "gaussians_scene": 1766302,
                "gaussians_environment": 1363,
                "lod_scene": [1766302, 882927, 441429],
            },
            {
                "name": "AbbeyStreet_Stairs",
                "location": "indoors",
                "sh_degree": 3,
                "eval_resolution": [1920, 1080],
                "eval_views": 100,
                "gaussians_scene": 875735,
                "gaussians_environment": 836,
                "lod_scene": [875735, 437794],
            },
        ],
    }
    text = f"""# Proposed Heritage Test Material for JPEG Radiance Fields

**Document style:** aligned with ISO/IEC JTC 1/SC 29/WG 1 *Proposed Common Test Conditions for JPEG Radiance Fields* (M111041, §2 Test Material) and Exploration Study 5 trajectory guidelines.

| Field | Value |
|-------|--------|
| **TITLE** | Proposed Heritage Test Material for JPEG Radiance Fields (Coding-Only Evaluation) |
| **SOURCE** | [Author Name / Affiliation] |
| **PROJECT** | JPEG RF |
| **STATUS** | Draft / Informative contribution |
| **REQUESTED ACTION** | Discussion |
| **DISTRIBUTION** | WG1 |
| **DATE** | 2026-10-14 |

> Fill in **SOURCE**, document number (e.g. `Mxxxxx`), and meeting header before WG1 submission. Replace bracketed placeholders below.

---

## 1 Scope

This document describes **heritage radiance-field test material** proposed for JPEG Radiance Fields (JPEG RF) experiments, with emphasis on the **coding-only assessment pipeline** (CTC Case A): an already instantiated 3D Gaussian Splatting (3DGS) model is the fixed target representation to be encoded.

The material complements the public CTC sources (Tanks and Temples, Mip-NeRF 360, Den-SOFT) by providing **large-scale cultural-heritage scenes** with multi-LOD Gaussian assets, optional environment shells, fixed evaluation cameras, and passive visualization trajectories suitable for subjective and objective quality assessment.

---

## 2 Test Material

This section describes the proposed test material. Scenes are selected from one source collection with shared reconstruction conventions. A short description of the source is given in **Table 1**.

### Table 1 – Source dataset selected for heritage radiance-field evaluation

| Dataset | Acquisition / capture | Representation | Number of test scenes |
|---------|------------------------|----------------|------------------------|
| Heritage Radiance Fields (HRF) | Photogrammetry / multi-view capture of cultural-heritage sites (institution-specific; details TBD) | Instantiated 3DGS (INRIA-compatible PLY), SH degree 3 | 4 |

For each test scene, the following material is provided:

1. **Instantiated radiance-field model** — one or more 3DGS PLY files (full resolution and LOD variants), optionally with a separate **environment** shell PLY.
2. **Reconstruction metadata** — spherical-harmonics degree and background flag (`sh_degree=3`, `white_background=True`).
3. **Evaluation camera poses** — camera-to-world matrices and intrinsics for objective still-image assessment (concentric multi-ring paths).
4. **Passive visualization trajectories** — spiral / helix camera paths and JPEG RF Exploration Study 5–style key/ghost trajectories for subjective fly-through assessment.
5. **Reference renderings** — views rendered from the **uncompressed** model at the evaluation poses (coding-only reference), where applicable.

**NOTE.** Unlike CTC outdoor/indoor photo datasets that ship raw training images + COLMAP sparse models, this contribution targets **Case A (coding-only)**. Training multi-view images and SfM reconstructions may be distributed under a separate license agreement; the normative compressed/uncompressed comparison uses the fixed PLY models and provided poses.

### Table 2 – Test scenes selected for evaluation

| Test scene | Source dataset | Location | Eval resolution | Eval still views | Baseline Gaussians (scene [+ env]) | SH degree |
|------------|----------------|----------|-----------------|------------------|--------------------------------------|-----------|
{scene_rows}

**Figure 1–4 (to insert):** Example reference renderings for *MellifontAbbey*, *AbbeyStreet_UpperFloor*, *AbbeyStreet_LowerFloor*, and *AbbeyStreet_Stairs*.

---

## 2.1 Scene descriptions

{"".join(lod_sections)}
---

## 2.2 Material to be employed (coding-only)

For JPEG RF coding-only experiments on these scenes, the following shall be used consistently across proposals:

| Item | Specification |
|------|----------------|
| Target format | Instantiated 3DGS PLY (means, scales, quaternions, opacity, SH) |
| Reference model | Uncompressed PLY (± environment as declared per experiment) |
| Objective still poses | Fixed `cameras_eval_*.json` (`c2w` list + `width`/`height`/`fx`/`fy`) |
| Objective metrics (recommended) | PSNR, SSIM / MS-SSIM, LPIPS (and CTC metrics MS-SSIM, IW-SSIM, FSIM where applicable) |
| Passive trajectories | Fixed FPS and duration across conditions; spiral and/or JPEG RF ES5 key/ghost paths |
| Subjective protocol (Case A) | Double-stimulus vs. uncompressed-model reference trajectory (per CTC §3.3.3) |
| Rate operating points | Scene LOD × codec QP / λ ladder (proposal-specific; report size in MB and bits/Gaussian if relevant) |

Orbit / trajectory centers shall be computed from the **scene-layer** PLY centroid when an environment shell is present, so that paths remain focused on the heritage structure.

## 2.3 Compression method

Every result in §2.7–§2.10 was produced with the same coder: the GSCodec Studio sequential HEVC anchor (`compress_ply_sequence.py` presets `x265_compression_rp0` to `x265_compression_rp3`, `compression="seq_hevc"`). Each LOD is a single static 3DGS PLY, passed with one frame and GOP size 16. The environment shell is not part of the coded bitstream; sizes below are the scene layer only.

The coder follows the self-organizing Gaussian grid approach. Gaussians are reordered with PLAS so that nearby primitives become neighbours in a 2D grid, then reshaped into a square image. If the Gaussian count is not a square, the grid is padded and the padded primitives are dropped at decode. Each attribute is then min–max normalized over that grid and uniformly quantized: positions (means) to 16 bits, and opacity, scale, quaternion, and spherical-harmonic coefficients to 8 bits. The stored minima and maxima invert the normalization after decode. No colour-space conversion and no quaternion-to-Euler conversion are applied at these rate points.

The quantized maps are coded with libx265 (HEVC) through FFmpeg and stored as MP4. With one frame per asset, each bitstream is a single intra picture; all-intra mode was not enabled. Rate is set only by the constant QP in Table 2g. rp0 is the lowest rate and rp3 the highest. The same ladder is used for every scene and every LOD.

### Table 2f – Attribute coding in the sequential HEVC anchor

| Attribute | Quantization | HEVC representation |
|-----------|--------------|---------------------|
| Means (x, y, z) | 16-bit | Two byte planes, lossless (x265 `lossless=1`, preset `veryslow`; configured QP −1) |
| Opacity | 8-bit | One gray plane, constant QP |
| Scale | 8-bit | One 3-channel map, constant QP |
| Quaternion (w, x, y, z) | 8-bit | Gray plane for w and a 3-channel map for (x, y, z), same QP |
| SH DC (degree 0) | 8-bit | One 3-channel map, constant QP |
| SH degrees 1–3 | 8-bit | One 3-channel map per coefficient (15 maps); QP set per degree |

### Table 2g – x265 QP ladder per rate point (means coded as 16-bit, lossless)

| Rate point | Opacity | Quaternions | Scales | SH DC | SH bands 1 / 2 / 3 |
|------------|---------|-------------|--------|-------|--------------------|
| rp0 | 16 | 22 | 22 | 16 | 28 / 34 / 40 |
| rp1 | 10 | 16 | 16 | 10 | 22 / 28 / 34 |
| rp2 | 4 | 10 | 10 | 4 | 16 / 22 / 28 |
| rp3 | 4 | 4 | 4 | 4 | 4 / 4 / 4 |

{"".join(rd_blocks)}
---

## 3 Camera path design

Heritage evaluation uses three complementary path families, all rendered at **1920×1080** with **fx = fy = 1080** px, **cx = 960**, **cy = 540** (unless a study explicitly rescales). Camera-to-world poses follow the gsplat / COLMAP convention implemented in `examples/eval_encoded_results.py`.

### 3.1 Objective still viewpoints (FR metrics)

Full-reference PSNR / SSIM / LPIPS are computed on a fixed set of static poses generated by **concentric horizontal rings** around a scene look-at centre (`generate_concentric_c2w`). Ring *r* (0-based) has radius *(r + 1)·radius_step_m* and height *height_offset_m + r·ring_height_step_m*; `views_per_ring` azimuth samples are evenly spaced. Each camera looks at the centre, with an optional pitch-down rotation about the local **X** axis (**0°** for all heritage runs reported here). When scene and environment splats are merged for rendering, the orbit centre is taken from the **scene-layer PLY centroid** (`--center-from-ply`), not from the merged cloud, so rings stay inside the architectural volume.

### Table 3 – Per-scene objective ring parameters

| Scene | Views | Rings × per ring | radius_step (m) | ring height step (m) | Pitch (°) | Saved JSON |
|-------|------:|------------------|----------------:|---------------------:|----------:|------------|
| MellifontAbbey | 104 | 13 × 8 | 1.2 | 0.2 | 0 | `eva/_eval_100views/cameras_100.json` |
| AbbeyStreet UpperFloor | 100 | 10 × 10 | 0.5 | 0.1 | 0 | `eva/AbbeyStreet_UpperFloor/cameras_100.json` |
| AbbeyStreet LowerFloor | 100 | 10 × 10 | 0.45 | 0.1 | 0 | `eva/AbbeyStreet_LowerFloor/cameras_100.json` |
| AbbeyStreet Stairs | 100 | 10 × 10 | 0.45 | 0.1 | 0 | `eva/AbbeyStreet_Stairs/cameras_100.json` |

### 3.2 Spiral fly-through (NR video metrics)

No-reference VSFA and DOVER scores in §2 use a smooth **helix** derived from the objective ring set (`generate_spiral_c2w_from_cameras`): radius and height interpolate linearly from the innermost ring (camera 0) to the outermost (last camera), while azimuth sweeps **`spiral_rots`** full turns (**2.0**) before landing on the last pose. Orientation matches the ring generator (look-at centre, same pitch-down). **Abbey Street** scenes use **180 frames @ 25 fps** (7.2 s); **MellifontAbbey** uses **600 frames @ 25 fps** (24 s) for the reported spiral baseline.

### 3.3 Viewer-authored path (AbbeyStreet Stairs)

For longer subjective clips along a stair-focused route, optional renders use key poses saved from the **viser** viewer (`examples/render_camera_path.py`): centripetal **Catmull–Rom** position interpolation with **arc-length** reparameterization and quaternion **SLERP**, **600 frames @ 25 fps**, output under `eva/AbbeyStreet_Stairs/camera_path/`. This path is independent of the concentric ring set used for Tables 2h–2k.

### 3.4 JPEG RF ES5 subjective trajectories

Passive-visualization trajectories for JPEG RF Exploration Study 5 and CTC §3.2.2 are built from training / evaluation key poses via `examples/generate_jpeg_rf_trajectory.py` and `examples/jpeg_rf_trajectory.py`:

- Key views balancing **spatial coverage** and **directional diversity**
- Optional **ghost (virtual)** viewpoints in under-sampled regions
- Centripetal **Catmull–Rom** with **arc-length** reparameterization (approx. constant speed)
- Quaternion **SLERP** for orientation

### Table 4 – Example ES5 trajectory configurations (MellifontAbbey)

| Preset | K | G | α | Notes |
|--------|---|---|---|--------|
| fps_keys8 | 8 | 0 | 1.0 | Spatial coverage only |
| coverage_keys12 | 12 | 0 | 0.65 | Wide spatial sweep |
| coverage_orient_keys8 | 8 | 0 | 0.4 | Orientation diversity |
| ghost_gentle_orient (**recommended**) | 8 | 4 | 0.8 | Generalization without unstable extrapolation |
| ghost_strong_orient | 8 | 5 | 0.65 | View-dependent appearance stress |
| ghost_strong_span | 9 | 6 | 0.7 | Parallax / occlusion stress |

Default subjective clip: **600 frames @ 25 fps** (24 s). Machine-readable presets: `examples/configs/jpeg_rf_trajectories.json`. Reproducible generation: `python examples/generate_jpeg_rf_trajectory.py --trajectory ghost_gentle_orient`.

---

## 4 Relation to Common Test Conditions

| CTC element | Heritage material |
|-------------|-------------------|
| Pipeline | **Case A – coding-only** (fixed instantiated 3DGS) |
| Public CTC scenes | Complementary (not a replacement for Playground / Garden / Bonsai / …) |
| Reference for FR subjective / objective | Uncompressed PLY renderings at shared poses / trajectories |
| Novel-view trajectories | Supported via spiral + ES5 paths; FR metrics only where references exist |

Proposals that jointly instantiate and code (Case B) may still use these scenes if they adopt the same evaluation poses; FR comparison to an uncompressed PLY is then limited to coding-only arms.

---

## 5 Distribution and licensing

| Item | Status |
|------|--------|
| PLY models + camera JSON | [To be published / available under …] |
| Training photographs | [Optional separate release] |
| Contact | [Author email] |
| Software used for reconstruction | 3D Gaussian Splatting pipeline (`sh_degree=3`) |
| Evaluation harness (informative) | GSCodec Studio examples (`eval_encoded_results.py`, JPEG RF trajectory tools) |

---

## 6 References

1. ISO/IEC JTC 1/SC 29/WG 1, *Proposed Common Test Conditions for JPEG Radiance Fields*, M111041, v0.6, April 2026.
2. S. Mahmoudpour, *Report on Exploration Study 5 of JPEG RF*, M109040, Oct 2025.
3. WG1, *Exploration Study 7 of JPEG Radiance Fields*, N101414.
4. WG1, *Use Cases and Requirements for JPEG RF*, N101398, v1.0.
5. B. Kerbl et al., “3D Gaussian Splatting for Real-Time Radiance Field Rendering,” *ACM Trans. Graph.*, 2023.
6. [Additional citation for site capture / institution, if available.]

---

## Annex A – Machine-readable summary (informative)

```json
{json.dumps(annex, indent=2)}
```
"""
    md_path.write_text(text, encoding="utf-8")
    print(f"Wrote {md_path}")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    write_markdown()
    doc = build()
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
