#!/usr/bin/env python3
"""Generate PowerPoint report for JPEG RF Exploration Study 5 (trajectory focus)."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x0F, 0x2C, 0x4C)
TEAL = RGBColor(0x1F, 0x7A, 0x8C)
ACCENT = RGBColor(0xE8, 0xB8, 0x4A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1B, 0x1B, 0x1B)
MUTED = RGBColor(0x4A, 0x55, 0x63)
ROW_A = RGBColor(0xF4, 0xF7, 0xFA)


def set_run(run, text, size=18, bold=False, color=INK, font="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def add_bg(slide, color):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_bar(slide, top=0, height=0.12, color=ACCENT):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(top), Inches(13.333), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, NAVY)
    add_bar(slide)
    add_text = slide.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.2))
    p = add_text.text_frame.paragraphs[0]
    set_run(p.add_run(), title, size=36, bold=True, color=WHITE)
    sub = slide.shapes.add_textbox(Inches(0.8), Inches(3.5), Inches(11.5), Inches(2.5))
    tf = sub.text_frame
    tf.word_wrap = True
    for i, line in enumerate(subtitle.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        set_run(para.add_run(), line, size=18, color=WHITE)


def add_text_box(slide, l, t, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(p.add_run(), text, size=size, bold=bold, color=color)
    return box


def add_bullets(slide, l, t, w, h, items, size=17):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = item.get("level", 0)
        p.space_after = Pt(6)
        set_run(
            p.add_run(),
            item["text"],
            size=size - 2 * item.get("level", 0),
            bold=item.get("bold", False),
            color=item.get("color", INK),
        )


def add_table(slide, l, t, w, h, rows, col_widths=None):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(l), Inches(t), Inches(w), Inches(h))
    table = table_shape.table
    if col_widths:
        for i, width in enumerate(col_widths):
            table.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            for para in cell.text_frame.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(11 if r else 12)
                    run.font.bold = r == 0
                    run.font.color.rgb = WHITE if r == 0 else INK
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TEAL
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = ROW_A


def add_content_slide(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_bar(slide, color=TEAL)
    add_text_box(slide, 0.6, 0.35, 12, 0.6, title, size=28, bold=True, color=NAVY)
    add_bullets(slide, 0.7, 1.0, 12.0, 6.0, bullets)


def build_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "JPEG RF Exploration Study 5",
        "Report on Test Camera Trajectories for Subjective QA\n"
        "ISO/IEC JTC1/SC29/WG1 M109040 · Nürnberg, Oct 2025\n"
        "Source: PCQ — Saeed Mahmoudpour · GSCodec Studio implementation notes",
    )

    add_content_slide(
        prs,
        "1. Goal & Motivation",
        [
            {"text": "Explore how test camera trajectories affect human perception in radiance-field QA.", "bold": True},
            {"text": "NeRF and 3DGS need continuous fly-through paths — manual paths are slow and scene-dependent."},
            {"text": "SfM/COLMAP training cameras cluster in some regions; large gaps have no real views."},
            {"text": "Trajectories must reveal parallax, novel viewpoints, and stress the renderer."},
            {"text": "Deliverable: automated pipeline + guidelines for future JPEG RF subjective tests."},
        ],
    )

    add_content_slide(
        prs,
        "2. Pipeline Overview",
        [
            {"text": "Step 1 — Key view selection (K training cameras)", "bold": True},
            {"text": "Step 2 — Ghost view augmentation (G virtual cameras in sparse regions)"},
            {"text": "Step 3 — Centripetal Catmull–Rom spline through control points"},
            {"text": "Step 4 — Arc-length reparameterization → constant travel speed"},
            {"text": "Step 5 — Quaternion SLERP for smooth orientations"},
            {"text": "Integrates: submodular coverage [Krause 2014], Catmull–Rom [1974], SLERP [Shoemake 1985]."},
        ],
    )

    add_content_slide(
        prs,
        "3. Key View Selection (Section 2)",
        [
            {"text": "Density-weighted scene center X₀ from top-N% densest training cameras.", "bold": True},
            {"text": "ρᵢ = 1 / (ε + mean k-NN distance) — down-weights isolated outliers."},
            {"text": "First key: camera farthest from X₀ (scene periphery)."},
            {"text": "Greedy score: s(j) = α·D(j) + (1−α)·O(j)", "bold": True},
            {"text": "D(j): normalized distance to nearest selected key (spatial spread)."},
            {"text": "O(j): 1 − max |fⱼ·fₖ| — orientation diversity vs. existing keys."},
            {"text": "α ∈ [0,1]: high α → coverage/parallax; low α → varied headings."},
        ],
    )

    add_content_slide(
        prs,
        "4. Ghost (Virtual) Views (Section 3)",
        [
            {"text": "Insert G ghost viewpoints in under-sampled / unseen regions.", "bold": True},
            {"text": "Candidates: interpolate between key pairs + perturb toward sparse zones."},
            {"text": "Score: S(g) = w_empty·E + w_orient·O + w_span·‖Cₚ−Cᵧ‖", "bold": True},
            {"text": "E: distance to nearest real camera (emptiness)."},
            {"text": "O: orientation novelty vs. bridge segment."},
            {"text": "Span: favors long-baseline key pairs (strong parallax)."},
            {"text": "Keep ghosts ≥ δ_ghost from all other control points."},
        ],
    )

    add_content_slide(
        prs,
        "5. Trajectory Construction (Section 4)",
        [
            {"text": "Centripetal Catmull–Rom spline through M = K + G control points.", "bold": True},
            {"text": "Interpolates all keys/ghosts; avoids uniform-parameter oscillations."},
            {"text": "Arc-length reparameterization:", "bold": True},
            {"text": "Sample spline densely → cumulative arc length s(u)."},
            {"text": "Place frame n at s̃ₙ = n/(N−1)·s_end for constant camera speed."},
            {"text": "Critical for subjective tests — avoids acceleration bursts that bias attention."},
        ],
    )

    add_content_slide(
        prs,
        "6. Camera Orientation (Section 5)",
        [
            {"text": "Each key → rotation Rₖ → quaternion qₖ.", "bold": True},
            {"text": "Between keys: SLERP(q₀, q₁; t) with bounded angular velocity."},
            {"text": "Ghosts inherit rotation from bridge segment for consistency."},
            {"text": "Preserves training look directions while allowing gradual head movement."},
        ],
    )

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    add_bar(slide, color=TEAL)
    add_text_box(slide, 0.6, 0.35, 12, 0.6, "7. Trajectory Presets (Garden / Playground)", size=28, bold=True, color=NAVY)
    add_table(
        slide,
        0.4,
        1.0,
        12.5,
        5.8,
        [
            ["Preset", "K", "G", "α", "w_orient", "w_empty", "w_span", "Purpose"],
            ["FPS", "8", "0", "1.0", "0", "0", "0", "Spatial coverage only"],
            ["Coverage", "12", "0", "0.65", "0", "0", "0", "Wide spatial sweep"],
            ["Orient.", "8", "0", "0.4", "0", "0", "0", "Direction diversity"],
            ["Ghost gentle", "8", "4", "0.8", "0.45", "0.45", "0.10", "Generalization (recommended)"],
            ["Ghost strong-O", "8", "5", "0.65", "0.78", "0.17", "0.07", "View-dependent appearance"],
            ["Ghost strong-span", "9", "6", "0.7", "0.20", "0.25", "0.55", "Parallax / occlusion"],
        ],
        col_widths=[1.5, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 2.8],
    )

    add_content_slide(
        prs,
        "8. Subjective Test Recommendations (Section 7)",
        [
            {"text": "Fix FPS and duration across all trajectories in a study.", "bold": True},
            {"text": "Prefer multiple shorter arcs extracted from the full path."},
            {"text": "Three path families:", "bold": True},
            {"text": "(i) Coverage-focused — high α, few ghosts → parallax & occlusions."},
            {"text": "(ii) Orientation-diverse — low α → view-dependent SH / specular stress."},
            {"text": "(iii) Ghost-augmented — sparse regions → generalization / extrapolation."},
            {"text": "Tune α by scene: tight baselines → fewer keys; broad scenes → more keys."},
        ],
    )

    add_content_slide(
        prs,
        "9. GSCodec Studio Implementation",
        [
            {"text": "Config: examples/configs/jpeg_rf_trajectories.json", "bold": True},
            {"text": "Generator: examples/jpeg_rf_trajectory.py (PCQ ES5 algorithm)."},
            {"text": "CLI: python generate_jpeg_rf_trajectory.py --cfg.trajectory mellifont_recommended"},
            {"text": "Training poses: eva/_eval_100views/cameras_100.json (104 concentric views)."},
            {"text": "Output: eva/jpeg_rf_trajectories/cameras_<preset>.json + optional MP4."},
            {"text": "Render all presets: --cfg.all-presets --cfg.render-video --cfg.n-frames 600 --cfg.fps 25"},
            {"text": "Default recommended: ghost_gentle_orient (K=8, G=4, α=0.8)."},
        ],
    )

    add_content_slide(
        prs,
        "10. Comparison vs. Current Spiral Eval",
        [
            {"text": "Current spiral (eval_encoded_results): helix from cam0 → last, fixed center.", "bold": True},
            {"text": "JPEG RF ES5: data-driven keys from training distribution + ghosts in gaps."},
            {"text": "JPEG RF: constant speed via arc-length; SLERP orientations."},
            {"text": "Use JPEG RF trajectories for PCQ-aligned subjective / NR video metrics."},
            {"text": "Keep concentric 104-view set for objective PSNR/SSIM/LPIPS stills."},
        ],
    )

    return prs


def main():
    out = Path(__file__).resolve().parents[1] / "docs" / "JPEG_RF_ES5_Trajectory_Report.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs = build_presentation()
    prs.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
