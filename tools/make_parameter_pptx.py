#!/usr/bin/env python3
"""Generate a PowerPoint introducing compress_ply_sequence.py parameters."""

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
ROW_B = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1B, 0x7A, 0x4A)
RED = RGBColor(0xB4, 0x3A, 0x3A)


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
    return shape


def add_bar(slide, top=0, height=0.12, color=ACCENT):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(top), Inches(13.333), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_text_box(slide, l, t, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(p.add_run(), text, size=size, bold=bold, color=color)
    return box


def add_bullets(slide, l, t, w, h, items, size=18):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = item.get("level", 0)
        p.space_after = Pt(8)
        set_run(p.add_run(), item["text"], size=size - 2 * item.get("level", 0), bold=item.get("bold", False), color=item.get("color", INK))
    return box


def add_table(slide, l, t, w, h, rows, col_widths=None, header=True):
    n_rows = len(rows)
    n_cols = len(rows[0])
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(l), Inches(t), Inches(w), Inches(h))
    table = table_shape.table
    if col_widths:
        for i, width in enumerate(col_widths):
            table.columns[i].width = Inches(width)
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            is_header = header and r == 0
            set_run(
                p.add_run(),
                str(val),
                size=13 if not is_header else 13,
                bold=is_header or c == 0,
                color=WHITE if is_header else INK,
            )
            fill = NAVY if is_header else (ROW_A if r % 2 == 0 else ROW_B)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
    return table_shape


def content_slide(prs, title, subtitle=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.05)
    )
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()
    add_bar(slide, top=1.05, height=0.08, color=ACCENT)
    add_text_box(slide, 0.5, 0.22, 12.2, 0.5, title, size=28, bold=True, color=WHITE)
    if subtitle:
        add_text_box(slide, 0.5, 0.68, 12.2, 0.32, subtitle, size=14, color=RGBColor(0xC9, 0xD6, 0xE3))
    return slide


def build():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1. Title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, NAVY)
    add_bar(s, top=0, height=0.12, color=ACCENT)
    add_text_box(s, 0.7, 2.0, 12, 0.5, "GSCodec Studio", size=20, bold=True, color=ACCENT)
    add_text_box(s, 0.7, 2.5, 12, 1.2, "Parameter Guide", size=48, bold=True, color=WHITE)
    add_text_box(
        s, 0.7, 3.8, 11.5, 1.2,
        "How each compress_ply_sequence.py setting maps Gaussian splat attributes\nto HEVC videos — and what to turn for quality vs file size.",
        size=20, color=RGBColor(0xD7, 0xE3, 0xEE),
    )
    add_text_box(s, 0.7, 6.7, 11, 0.4, "examples/compress_ply_sequence.py  ·  seq_hevc / libx265", size=14, color=RGBColor(0x9A, 0xB0, 0xC4))

    # 2. Pipeline
    s = content_slide(prs, "What this script compresses", "One frame (or a GOP of frames) of 3D Gaussians → attribute videos in MP4")
    add_bullets(s, 0.55, 1.4, 12, 5.6, [
        {"text": "Input: PLY Gaussians (or a 3DGS folder with point_cloud.ply + environment.ply)"},
        {"text": "Pack: pad to a square H×H grid, optionally PLAS-sort so similar splats sit together"},
        {"text": "Normalize: per-attribute min/max over the GOP, then quantize to 8-bit (means: 16-bit lossless)"},
        {"text": "Encode: libx265 writes one or more MP4s per attribute into result_dir/compression/"},
        {"text": "Decode (same run): reconstruct splats; optionally render vs COLMAP cameras"},
        {"text": "Bitrate is dominated by SH videos. Geometry (means) is lossless by default."},
    ], size=20)

    # 3. CLI
    s = content_slide(prs, "CLI anatomy", "Preset first, then flags. Nested fields use dotted names.")
    add_text_box(
        s, 0.55, 1.4, 12.2, 1.4,
        "python compress_ply_sequence.py x265_compression_rp3 \\\n  --ply-dir PATH --frame-num 1 --gop-size 16 --skip-eval \\\n  --compression-cfg.qp.shN.sh3 34",
        size=16, color=NAVY,
    )
    add_table(s, 0.55, 3.1, 12.2, 3.5, [
        ["Token", "Meaning"],
        ["x265_compression_rp3", "Preset: codec + default QPs (tyro overridable_config_cli)"],
        ["--compression-cfg.qp.*", "Override HEVC quantizer per Gaussian attribute"],
        ["--ply-dir / --result-dir", "Input Gaussians and output folder"],
        ["--skip-eval", "Encode only; COLMAP cameras not required"],
    ], col_widths=[3.4, 8.8])

    # 4. I/O
    s = content_slide(prs, "Input / output paths", "These decide what is loaded, not how hard it is compressed")
    add_table(s, 0.4, 1.35, 12.5, 5.5, [
        ["Parameter", "Default", "Role"],
        ["ply_dir", '""', "Flat *.ply sequence, or a 3DGS result folder with point_cloud/iteration_*"],
        ["ply_iteration", "None", "Which training checkpoint; latest iteration_* if unset"],
        ["merge_environment", "True", "Concat environment.ply + point_cloud.ply as one splat cloud"],
        ["result_dir", "results/garden", "Writes compression/, stats/, renders/, unity_manifest.json"],
        ["frame_num", "1", "How many PLYs / time steps to encode"],
        ["anchor_type", "video", "video = seq_hevc MP4s; pcc = GeS-TM point-cloud codec"],
        ["compression", "seq_hevc", "seq_hevc (x265 MP4) or seq_yuv (raw YUV + reference codec)"],
    ], col_widths=[2.6, 2.2, 7.7])

    # 5. GOP
    s = content_slide(prs, "Sequence / GOP", "Temporal packing of splat frames into HEVC GOPs")
    add_table(s, 0.4, 1.35, 12.5, 2.6, [
        ["Parameter", "Default", "Quality / size effect"],
        ["gop_size", "16", "Larger GOP → more inter prediction → smaller files if frames are similar"],
        ["frame_num", "1", "Single snapshot: GOP size does not change bitrate"],
        ["use_all_intra", "False", "True = every video frame is an I-frame: larger files, random-access friendly"],
    ], col_widths=[2.4, 1.6, 8.5])
    add_bullets(s, 0.55, 4.2, 12, 2.6, [
        {"text": "num_gop = ceil(frame_num / gop_size). Each GOP writes gop{id}_*.mp4 + gop{id}_meta.json"},
        {"text": "Default sort uses frame 0 of the GOP for all T frames (use_all_intra=False)"},
        {"text": "For a static Mellifont export, --frame-num 1 is correct; GOP is a no-op"},
    ], size=18)

    # 6. Presets
    s = content_slide(prs, "Rate-point presets", "First CLI argument. rp3 = highest quality; rp0 = smallest of the four")
    add_table(s, 0.35, 1.35, 12.6, 5.5, [
        ["Preset", "opac / quat / scale / sh0", "shN sh1/2/3", "Intent"],
        ["x265_compression_rp3", "4 / 4 / 4 / 4", "4 / 4 / 4", "Best quality (your current default)"],
        ["x265_compression_rp2", "4 / 10 / 10 / 4", "16 / 22 / 28", "Balanced cut"],
        ["x265_compression_rp1", "10 / 16 / 16 / 10", "22 / 28 / 34", "Smaller, more artifacts"],
        ["x265_compression_rp0", "16 / 22 / 22 / 16", "28 / 34 / 40", "Smallest of the numbered set"],
        ["large_shN_qp_rp1", "4 / 10 / 10 / 4", "36 / 40 / 44", "Keep geometry; crush view-dep. color"],
        ["transform_x265_*", "same as rp*", "same as rp*", "RGB→YCbCr on SH + normalize quats"],
        ["pcc_compression_rp*", "n/a", "n/a", "Different codec (GeS-TM .bin), not MP4"],
    ], col_widths=[3.1, 3.5, 2.2, 3.8])

    # 7. QP law
    s = content_slide(prs, "The QP law", "This is the only family of knobs that strongly moves quality vs MP4 size")
    add_bullets(s, 0.55, 1.45, 12.2, 5.5, [
        {"text": "QP = HEVC quantization parameter, passed as x265-params qp=…"},
        {"text": "Higher QP → coarser quantization → smaller file, more error"},
        {"text": "Lower QP → finer quantization → larger file, closer to the original PLY"},
        {"text": "Typical useful range here: 4 (near-transparent) … 22–40 (aggressive)"},
        {"text": "means QP is ignored: positions are 16-bit lossless (qp=-1 / lossless=1)"},
        {"text": "Override example:  --compression-cfg.qp.opacities 10  --compression-cfg.qp.shN.sh3 34"},
        {"text": "shN is 15 RGB videos (degree 1–3). Raising those QPs is the biggest size win."},
    ], size=20)

    # 8. Per-attribute
    s = content_slide(prs, "Per-attribute QP", "What each Gaussian channel becomes in the bitstream")
    add_table(s, 0.3, 1.3, 12.7, 5.6, [
        ["Parameter", "Stored as", "If you raise QP…", "Files"],
        ["qp.means (-1)", "xyz, 16-bit split L/U", "Do not; already lossless", "gop*_means_l/u.mp4"],
        ["qp.opacities", "pre-sigmoid logit, gray 8-bit", "Holes, missing splats, floaters", "gop*_opacities.mp4"],
        ["qp.quats", "wxyz split w + xyz", "Stretching / wrong ellipsoid axes", "gop*_quats_w/xyz.mp4"],
        ["qp.scales", "pre-exp log-scale RGB", "Blobs too large/small, blurry edges", "gop*_scales.mp4"],
        ["qp.sh0", "SH DC / base RGB", "Wrong albedo / overall tint", "gop*_sh0.mp4"],
        ["qp.shN.sh1", "SH degree 1 (3 bands)", "Flatter lighting, less glints", "3 × shN_sh1_*.mp4"],
        ["qp.shN.sh2", "SH degree 2 (5 bands)", "Lost mid-frequency view color", "5 × shN_sh2_*.mp4"],
        ["qp.shN.sh3", "SH degree 3 (7 bands)", "Safest to crush for size", "7 × shN_sh3_*.mp4"],
    ], col_widths=[2.2, 2.8, 4.0, 3.7])

    # 9. Sort / transform
    s = content_slide(prs, "Sort, intra, and color transforms", "Secondary RD knobs around the HEVC encoder")
    add_table(s, 0.35, 1.35, 12.6, 5.5, [
        ["Parameter", "Default", "Better quality?", "Smaller file?"],
        ["use_sort (PLAS)", "True", "Usually yes (less ringing)", "Yes — keep ON"],
        ["use_all_intra", "False", "Slightly, per-frame independent", "No — files grow"],
        ["transform quats", "False on rp3", "Normalizes rotations", "Neutral / slight"],
        ["transform sh0/shN", "False on rp3", "RGB→YCbCr can help SH coding", "Often slightly smaller"],
        ["debug", "False", "Writes .265 + PNG dumps", "No — debug only"],
        ["verbose", "True", "Log only", "No"],
        ["n_clusters", "32768", "VQ for other codecs, not seq_hevc size", "No (x265 path)"],
    ], col_widths=[2.8, 2.2, 3.8, 3.8])

    # 10. Eval
    s = content_slide(prs, "Evaluation / cameras", "Needed only if you want PSNR/SSIM/LPIPS, not for encoding MP4s")
    add_table(s, 0.35, 1.35, 12.6, 5.5, [
        ["Parameter", "Default", "Notes"],
        ["skip_eval", "False", "True: no COLMAP, no LPIPS, no renders — use for PLY-only encode"],
        ["data_dir", '""', "Per-frame COLMAP folders (sparse/0 + images). Required unless skip_eval"],
        ["data_factor", "4", "Image downsample for eval cameras (1 = full res)"],
        ["normalize_world_space", "True", "Aligns COLMAP to GS world; use --no-normalize-world-space if already aligned"],
        ["test_view_id", "None", "Which camera indices to render, e.g. --test-view-id 0"],
        ["lpips_net", "alex", "alex or vgg; eval-only"],
        ["scene_type", "default", "GSC vs default dataset wrapping"],
        ["test_every", "8", "Legacy COLMAP train/val split; GSCDataset uses test_view_id"],
    ], col_widths=[3.2, 1.8, 7.6])

    # 11. Unused
    s = content_slide(prs, "Inherited trainer flags (do not change x265 size)", "Still on Config because this file was forked from simple_trainer.py")
    add_bullets(s, 0.55, 1.4, 12.2, 5.6, [
        {"text": "Training loop: max_steps, eval_steps, save_steps, batch_size, steps_scaler"},
        {"text": "Init: init_type, init_num_pts, init_extent, init_opa, init_scale, sh_degree_interval"},
        {"text": "Loss / densify: ssim_lambda, strategy, opacity_reg, scale_reg, depth_loss, depth_lambda"},
        {"text": "Raster extras: packed, antialiased, sparse_grad, visible_adam, random_bkgd, camera_model"},
        {"text": "Pose / appearance / bilateral grid: pose_opt*, app_opt*, use_bilateral_grid"},
        {"text": "RD training sim: compression_sim, quantization_sim, entropy_model_*, rd_lambda, entropy_steps"},
        {"text": "shN_ada_mask_*: training-time SH masking, not the x265 encoder"},
        {"text": "chroma_subsampling / use_chroma_qp_offset: YUV path only; stripped for seq_hevc"},
    ], size=18)

    # 12. Recipes
    s = content_slide(prs, "Recommended recipes", "Same Mellifont 3DGS export; only the preset/QPs change")
    add_table(s, 0.3, 1.3, 12.7, 4.4, [
        ["Goal", "Command idea"],
        ["Best quality", "x265_compression_rp3  (QP 4 everywhere, means lossless)"],
        ["Smaller, sharp geometry", "x265_compression_large_shN_qp_rp1  or raise only shN.sh1/2/3"],
        ["Much smaller", "x265_compression_rp0  — expect muddy view-dependent color"],
        ["Drop environment shell", "same preset +  --no-merge-environment"],
        ["Encode without COLMAP", "always pass  --skip-eval"],
    ], col_widths=[3.3, 9.4])
    add_text_box(
        s, 0.5, 5.9, 12.2, 1.1,
        "Rule of thumb: never raise means QP. Raise shN first, then opacities/sh0, then quats/scales last.",
        size=18, bold=True, color=NAVY,
    )

    # 13. Output files
    s = content_slide(prs, "What each output file is", "result_dir/compression after a seq_hevc run")
    add_table(s, 0.3, 1.3, 12.7, 5.6, [
        ["File", "Contents"],
        ["gop{id}_meta.json", "shape, dtype, mins/maxs, file_extension per attribute"],
        ["unity_manifest.json", "gop_size, frame_num, n_orig, transform flags for the Unity plugin"],
        ["gop*_means_l.mp4 / _u.mp4", "Low/high bytes of 16-bit xyz (lossless HEVC)"],
        ["gop*_opacities.mp4", "8-bit grayscale logits"],
        ["gop*_scales.mp4", "8-bit RGB log-scales"],
        ["gop*_quats_w.mp4 / _xyz.mp4", "Quaternion w (gray) and xyz (RGB)"],
        ["gop*_sh0.mp4", "Degree-0 SH RGB"],
        ["gop*_shN_sh{d}_{l}.mp4", "15 view-dependent SH band videos (absent if SH degree 0)"],
    ], col_widths=[4.0, 8.7])

    out = "/mnt/d/LiveStream/GSCodec_Studio/docs/GSCodec_parameter_guide.pptx"
    prs.save(out)
    print(out)


if __name__ == "__main__":
    build()
