"""Decode GSCodec seq_hevc MP4s to PLY and open nerfview. No COLMAP.

Run from the examples/ folder:

    python decode_and_view.py --result-dir results/001CL8K14100HK_movie
    python decode_and_view.py --compress-dir results/001CL8K14100HK_movie/compression --frame 0
    python decode_and_view.py --result-dir results/MellifontAbbey_1_SH_fast --no-view
    python decode_and_view.py --result-dir results/MellifontAbbey_SH_full_rp3 \\
        --views-json eva/_eval_100views/cameras_100.json

Path drawer (in the web UI):
  1. Enable *Draw mode* under "Path drawer"
  2. Click the 3D scene to drop movable keypoints (drag the gizmo to edit)
  3. Click *Generate path & save JSON* to write an interpolated camera path
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tyro
import viser
from torch import Tensor
from viser.transforms import SO3

from datasets.traj import viewmatrix
from gsplat.compression import SeqHevcCompression
from gsplat.compression.seq_hevc_compression import has_gop_layer
from gsplat.rendering import rasterization
from jpeg_rf_trajectory import (
    quaternion_to_rotation,
    rotation_to_quaternion,
    sample_spline_arc_length,
    slerp,
)
from utils import concat_splat_dicts, save_ply, trim_splat_padding

PAD_LOG_SCALE = -9.7
_IDENTITY_TRANSFORM = {
    "means": False,
    "scales": False,
    "quats": False,
    "opacities": False,
    "sh0": False,
    "shN": False,
}


@dataclass
class Config:
    # Result folder that contains compression/ (and optionally unity_manifest.json).
    result_dir: Optional[str] = None
    # Folder with gop*_meta.json and attribute MP4s. Overrides result_dir/compression.
    compress_dir: Optional[str] = None
    # Where to write reconstructed PLYs. Default: <result_dir>/decoded_ply
    ply_dir: Optional[str] = None
    # Frame index to show first (0-based).
    frame: int = 0
    # Decode and write PLYs only; do not start nerfview.
    no_view: bool = False
    # Keep square-grid / GOP-align pads (log-scale ≈ -10).
    keep_padding: bool = False
    # nerfview / viser port.
    port: int = 8080
    # Skip far-away Gaussians in the viewer (pixels).
    radius_clip: float = 3.0
    # Load / save path for the editable view-position list (cameras_*.json).
    views_json: Optional[str] = None
    # Output path for the Path drawer generated camera trajectory JSON.
    path_json: Optional[str] = None
    # Image size metadata written into the views JSON (used by eval).
    views_width: int = 960
    views_height: int = 540
    views_fx: Optional[float] = None
    views_fy: Optional[float] = None


def _resolve_dirs(cfg: Config) -> Tuple[str, str, str]:
    if cfg.compress_dir:
        compress_dir = os.path.abspath(cfg.compress_dir)
    elif cfg.result_dir:
        result_dir = os.path.abspath(cfg.result_dir)
        nested = os.path.join(result_dir, "compression")
        compress_dir = nested if os.path.isdir(nested) else result_dir
    else:
        raise SystemExit("Pass --result-dir or --compress-dir.")

    if not os.path.isdir(compress_dir):
        raise FileNotFoundError(f"Compression directory does not exist: {compress_dir}")

    if cfg.result_dir:
        result_dir = os.path.abspath(cfg.result_dir)
    elif os.path.basename(compress_dir) == "compression":
        result_dir = os.path.dirname(compress_dir)
    else:
        result_dir = compress_dir

    ply_dir = os.path.abspath(cfg.ply_dir) if cfg.ply_dir else os.path.join(result_dir, "decoded_ply")
    return result_dir, compress_dir, ply_dir


def _list_gop_ids(compress_dir: str) -> List[int]:
    ids = []
    for path in glob.glob(os.path.join(compress_dir, "gop*_meta.json")):
        match = re.match(r"gop(\d+)_meta\.json$", os.path.basename(path))
        if match:
            ids.append(int(match.group(1)))
    if not ids:
        raise FileNotFoundError(
            f"No gop*_meta.json in {compress_dir}. Point at a GSCodec compression folder."
        )
    return sorted(ids)


def _load_manifest(compress_dir: str) -> Dict[str, Any]:
    path = os.path.join(compress_dir, "unity_manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as fp:
        return json.load(fp)


def _trim_padding(splat: Dict[str, Tensor], n_orig: Optional[int]) -> Dict[str, Tensor]:
    return trim_splat_padding(splat, n_orig, PAD_LOG_SCALE)


def decode_sequence(compress_dir: str, keep_padding: bool) -> List[Dict[str, Tensor]]:
    manifest = _load_manifest(compress_dir)
    transform_attributes = dict(manifest.get("transform_attributes") or _IDENTITY_TRANSFORM)
    n_orig_per_frame = manifest.get("n_orig_per_frame")
    layer_encoding = manifest.get("layer_encoding")
    layer_order = manifest.get("layer_order") or ["env", "scene"]
    n_scene_per_frame = manifest.get("n_scene_per_frame")
    n_env_per_frame = manifest.get("n_env_per_frame")

    os.makedirs(os.path.join(os.path.dirname(compress_dir), "stats"), exist_ok=True)

    codec = SeqHevcCompression(verbose=True)
    splats_list: List[Dict[str, Tensor]] = []
    for gop_id in _list_gop_ids(compress_dir):
        print(f"Decoding GOP {gop_id} ...")
        scene_video = codec.decompress(compress_dir, gop_id, layer_prefix="")
        scene_frames = codec.deorganize(scene_video)
        scene_frames = codec.param_inverse_transform(scene_frames, transform_attributes)

        env_frames = None
        if layer_encoding == "split" or has_gop_layer(compress_dir, gop_id, "env_"):
            env_video = codec.decompress(compress_dir, gop_id, layer_prefix="env_")
            env_frames = codec.deorganize(env_video)
            env_frames = codec.param_inverse_transform(env_frames, transform_attributes)

        if env_frames is not None:
            for local_idx, scene in enumerate(scene_frames):
                frame_idx = len(splats_list)
                n_scene = None
                n_env = None
                if isinstance(n_scene_per_frame, list) and frame_idx < len(n_scene_per_frame):
                    n_scene = int(n_scene_per_frame[frame_idx])
                if isinstance(n_env_per_frame, list) and frame_idx < len(n_env_per_frame):
                    n_env = int(n_env_per_frame[frame_idx])
                scene = trim_splat_padding(scene, n_scene, PAD_LOG_SCALE)
                env = trim_splat_padding(env_frames[local_idx], n_env, PAD_LOG_SCALE)
                parts = []
                for layer_name in layer_order:
                    parts.append(env if layer_name == "env" else scene)
                splats_list.append(concat_splat_dicts(parts))
        else:
            splats_list.extend(scene_frames)

    if keep_padding:
        return splats_list

    trimmed = []
    for idx, splat in enumerate(splats_list):
        n_orig = None
        if isinstance(n_orig_per_frame, list) and idx < len(n_orig_per_frame):
            n_orig = int(n_orig_per_frame[idx])
        before = int(splat["means"].shape[0])
        splat = _trim_padding(splat, n_orig)
        after = int(splat["means"].shape[0])
        if after != before:
            print(f"Frame {idx}: dropped {before - after} pad Gaussians ({after} kept).")
        trimmed.append(splat)
    return trimmed


def save_plys(splats_list: List[Dict[str, Tensor]], ply_dir: str) -> List[str]:
    os.makedirs(ply_dir, exist_ok=True)
    paths = []
    for idx, splat in enumerate(splats_list):
        path = os.path.join(ply_dir, f"frame{idx:03d}.ply")
        save_ply(splat, path)
        paths.append(path)
        print(f"Wrote {path}  ({int(splat['means'].shape[0])} Gaussians)")
    return paths


def _activate_for_render(splat: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
    means = splat["means"].to(device)
    quats = F.normalize(splat["quats"].to(device), p=2, dim=-1)
    scales = torch.exp(splat["scales"].to(device))
    opacities = torch.sigmoid(splat["opacities"].to(device))
    sh0 = splat["sh0"].to(device)
    shN = splat["shN"].to(device)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "sh_degree": sh_degree,
    }


def _scene_center(splat: Dict[str, Tensor]) -> Tuple[np.ndarray, float]:
    means = splat["means"].detach().float()
    center = means.mean(dim=0).cpu().numpy()
    extent = float((means.max(dim=0).values - means.min(dim=0).values).max().item())
    return center, max(extent, 1.0)


def c2w_to_wxyz_position(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 camera-to-world into viser (wxyz, position)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    wxyz = np.asarray(SO3.from_matrix(c2w[:3, :3]).wxyz, dtype=np.float64)
    position = c2w[:3, 3].astype(np.float64)
    return wxyz, position


def wxyz_position_to_c2w(wxyz: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Build a 4x4 camera-to-world from viser orientation + position."""
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(SO3(wxyz).as_matrix(), dtype=np.float32)
    c2w[:3, 3] = np.asarray(position, dtype=np.float32)
    return c2w


def client_camera_to_c2w(camera: Any) -> np.ndarray:
    """Read the interactive viser client camera as a 4x4 c2w."""
    return wxyz_position_to_c2w(np.asarray(camera.wxyz), np.asarray(camera.position))


def apply_c2w_to_client(camera: Any, c2w: np.ndarray) -> None:
    """Move the interactive client camera to a saved c2w pose."""
    wxyz, position = c2w_to_wxyz_position(c2w)
    camera.wxyz = wxyz
    camera.position = position


def pose_look_at(
    position: np.ndarray,
    look_at: np.ndarray,
    up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a 4x4 c2w that looks from ``position`` toward ``look_at`` (nerfview axes)."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3)
    up = np.asarray([0.0, 0.0, 1.0] if up is None else up, dtype=np.float64)
    lookdir = look_at - position
    if np.linalg.norm(lookdir) < 1e-8:
        lookdir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    vm = viewmatrix(lookdir, up, position)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = vm.astype(np.float32)
    # Match datasets.traj / eval spiral convention used elsewhere in this repo.
    pose[:3, 0] *= -1.0
    pose[:3, 1] *= -1.0
    return pose


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    fallback_depth: float = 5.0,
) -> np.ndarray:
    """Intersect a ray with a plane; fall back to a point along the ray if parallel."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    plane_point = np.asarray(plane_point, dtype=np.float64).reshape(3)
    plane_normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    denom = float(np.dot(direction, plane_normal))
    if abs(denom) < 1e-6:
        return origin + direction * fallback_depth
    t = float(np.dot(plane_point - origin, plane_normal) / denom)
    if t < 1e-4:
        return origin + direction * fallback_depth
    return origin + direction * t


def interpolate_camera_path(
    keypoints: List[np.ndarray],
    n_frames: int,
    look_at: np.ndarray,
    arc_length_samples: int = 4096,
) -> List[np.ndarray]:
    """Dense constant-speed path from keypoint poses (Catmull–Rom + SLERP)."""
    if not keypoints:
        return []
    if len(keypoints) == 1:
        return [np.asarray(keypoints[0], dtype=np.float32).copy() for _ in range(n_frames)]

    pts = np.stack([np.asarray(k, dtype=np.float64)[:3, 3] for k in keypoints], axis=0)
    positions = sample_spline_arc_length(pts, n_frames, arc_length_samples, alpha=0.5)

    quats = [rotation_to_quaternion(np.asarray(k, dtype=np.float64)[:3, :3]) for k in keypoints]
    seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = float(seg_len.sum())
    if total < 1e-9:
        seg_len = np.ones(len(keypoints) - 1, dtype=np.float64)
        total = float(seg_len.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, total, n_frames)
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3)

    out: List[np.ndarray] = []
    for i, s in enumerate(targets):
        seg = int(np.searchsorted(cum, s, side="right") - 1)
        seg = min(max(seg, 0), len(quats) - 2)
        local = (s - cum[seg]) / (seg_len[seg] + 1e-12)
        q_key = slerp(quats[seg], quats[seg + 1], float(local))
        pose_look = pose_look_at(positions[i], look_at)
        q_look = rotation_to_quaternion(pose_look[:3, :3].astype(np.float64))
        q = slerp(q_key, q_look, 0.35)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = quaternion_to_rotation(q).astype(np.float32)
        pose[:3, 3] = positions[i].astype(np.float32)
        out.append(pose)
    return out


def load_views_json(path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Load a view list. Supports ``{\"c2w\": [...]}`` or a bare list of 4x4 matrices."""
    with open(path, "r") as fp:
        payload = json.load(fp)
    if isinstance(payload, list):
        c2w_list = payload
        meta: Dict[str, Any] = {}
    elif isinstance(payload, dict) and "c2w" in payload:
        c2w_list = payload["c2w"]
        meta = {k: v for k, v in payload.items() if k != "c2w"}
    else:
        raise ValueError(f"Unrecognized views JSON format: {path}")
    views = [np.asarray(m, dtype=np.float32).reshape(4, 4) for m in c2w_list]
    return views, meta


def save_views_json(
    path: str,
    views: List[np.ndarray],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save view positions in the same schema as eval_encoded_results cameras_*.json."""
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: Dict[str, Any] = dict(meta or {})
    payload["num_views"] = len(views)
    payload["c2w"] = [np.asarray(m, dtype=np.float32).tolist() for m in views]
    with open(abs_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved {len(views)} views -> {abs_path}")


@dataclass
class PathDrawer:
    """Click-to-place movable keypoints, then generate an interpolated camera path JSON."""

    server: viser.ViserServer
    json_path: str
    look_at: np.ndarray
    extent: float = 5.0
    width: int = 960
    height: int = 540
    fx: float = 540.0
    fy: float = 540.0
    n_frames: int = 600
    fps: int = 25
    marker_radius: float = 0.15
    keypoints: List[np.ndarray] = field(default_factory=list)
    selected: int = 0
    draw_mode: bool = False
    _markers: Dict[int, Any] = field(default_factory=dict, repr=False)
    _spline: Any = field(default=None, repr=False)
    _gizmo: Any = field(default=None, repr=False)
    _status: Any = field(default=None, repr=False)
    _path_text: Any = field(default=None, repr=False)
    _draw_cb: Any = field(default=None, repr=False)
    _frames_slider: Any = field(default=None, repr=False)
    _fps_slider: Any = field(default=None, repr=False)
    _index_slider: Any = field(default=None, repr=False)
    _suppress_gizmo_cb: bool = field(default=False, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def setup(self) -> None:
        self.look_at = np.asarray(self.look_at, dtype=np.float64).reshape(3)
        self.marker_radius = max(float(self.extent) * 0.02, 0.05)

        with self.server.gui.add_folder("Path drawer"):
            self._status = self.server.gui.add_markdown(self._status_text())
            self._path_text = self.server.gui.add_text(
                "Path JSON", initial_value=self.json_path
            )
            draw_cb = self.server.gui.add_checkbox("Draw mode (click scene)", False)
            self._draw_cb = draw_cb
            self._frames_slider = self.server.gui.add_slider(
                "Generated frames", min=30, max=1200, step=10, initial_value=self.n_frames
            )
            self._fps_slider = self.server.gui.add_slider(
                "FPS", min=1, max=60, step=1, initial_value=self.fps
            )
            self._index_slider = self.server.gui.add_slider(
                "Keypoint",
                min=0,
                max=0,
                step=1,
                initial_value=0,
                disabled=True,
            )
            btn_add_cam = self.server.gui.add_button("Add current camera as keypoint")
            btn_delete = self.server.gui.add_button("Delete selected keypoint")
            btn_clear = self.server.gui.add_button("Clear keypoints")
            btn_generate = self.server.gui.add_button("Generate path & save JSON")

        @draw_cb.on_update
        def _(_) -> None:
            self.set_draw_mode(bool(draw_cb.value))

        @self._frames_slider.on_update
        def _(_) -> None:
            self.n_frames = int(self._frames_slider.value)

        @self._fps_slider.on_update
        def _(_) -> None:
            self.fps = int(self._fps_slider.value)

        @self._index_slider.on_update
        def _(_) -> None:
            if self.keypoints:
                self.select(int(self._index_slider.value))

        @btn_add_cam.on_click
        def _(_) -> None:
            self.add_current_camera()

        @btn_delete.on_click
        def _(_) -> None:
            self.delete_selected()

        @btn_clear.on_click
        def _(_) -> None:
            self.clear()

        @btn_generate.on_click
        def _(_) -> None:
            self.generate_and_save()

        self.refresh_markers()
        self._attach_gizmo()
        self._refresh_status()

    def _status_text(self) -> str:
        mode = "ON — click scene to add keypoints" if self.draw_mode else "off"
        if not self.keypoints:
            return (
                f"**Path drawer** · Draw mode: {mode}<br/>"
                "No keypoints yet. Enable draw mode and click the scene, "
                "or *Add current camera as keypoint*."
            )
        return (
            f"**Path drawer** · Draw mode: {mode}<br/>"
            f"**Keypoints:** {len(self.keypoints)} &nbsp;|&nbsp; "
            f"**Selected:** {self.selected} &nbsp;|&nbsp; "
            f"Generate → `{os.path.basename(self.json_path)}` "
            f"({self.n_frames} frames @ {self.fps} fps)"
        )

    def _refresh_status(self) -> None:
        if self._status is not None:
            self._status.content = self._status_text()
        if self._index_slider is not None:
            n = len(self.keypoints)
            self._index_slider.disabled = n == 0
            self._index_slider.max = max(n - 1, 0)
            if n and self._index_slider.value != self.selected:
                self._index_slider.value = int(np.clip(self.selected, 0, n - 1))

    def _primary_client(self) -> Optional[viser.ClientHandle]:
        clients = list(self.server.get_clients().values())
        return clients[0] if clients else None

    def set_draw_mode(self, enabled: bool) -> None:
        self.draw_mode = bool(enabled)
        if self.draw_mode:
            @self.server.scene.on_pointer_event("click")
            def _on_click(event: viser.ScenePointerEvent) -> None:
                self._on_scene_click(event)

            print("Path drawer: draw mode ON — click the scene to place keypoints.")
        else:
            try:
                self.server.scene.remove_pointer_callback()
            except Exception:  # noqa: BLE001
                pass
            print("Path drawer: draw mode OFF.")
        self._refresh_status()

    def _on_scene_click(self, event: viser.ScenePointerEvent) -> None:
        if not self.draw_mode:
            return
        if event.ray_origin is None or event.ray_direction is None:
            return
        hit = ray_plane_intersection(
            np.asarray(event.ray_origin, dtype=np.float64),
            np.asarray(event.ray_direction, dtype=np.float64),
            self.look_at,
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            fallback_depth=max(self.extent * 0.5, 1.0),
        )
        # Prefer selecting a nearby keypoint over stacking a new one on top of it.
        if self.keypoints:
            dists = [
                float(np.linalg.norm(np.asarray(k[:3, 3], dtype=np.float64) - hit))
                for k in self.keypoints
            ]
            nearest = int(np.argmin(dists))
            if dists[nearest] < max(self.marker_radius * 3.0, 0.25):
                self.select(nearest)
                return
        pose = pose_look_at(hit, self.look_at)
        self._append_keypoint(pose)
        print(f"Path drawer: added keypoint {self.selected} at {hit.round(3).tolist()}")

    def _append_keypoint(self, pose: np.ndarray) -> None:
        self.keypoints.append(np.asarray(pose, dtype=np.float32))
        self.selected = len(self.keypoints) - 1
        self.refresh_markers()
        self._attach_gizmo()
        self._refresh_status()

    def add_current_camera(self) -> None:
        client = self._primary_client()
        if client is None:
            print("No viser client connected; open the web UI first.")
            return
        self._append_keypoint(client_camera_to_c2w(client.camera))
        print(f"Path drawer: added keypoint from client camera ({len(self.keypoints)} total)")

    def select(self, idx: int) -> None:
        if not self.keypoints:
            return
        idx = int(np.clip(idx, 0, len(self.keypoints) - 1))
        if idx == self.selected and idx in self._markers:
            self._move_gizmo()
            return
        self.selected = idx
        self.refresh_markers()
        self._attach_gizmo()
        self._refresh_status()

    def delete_selected(self) -> None:
        if not self.keypoints:
            return
        del self.keypoints[self.selected]
        self.selected = min(self.selected, max(len(self.keypoints) - 1, 0))
        self.refresh_markers()
        self._attach_gizmo()
        self._refresh_status()

    def clear(self) -> None:
        self.keypoints = []
        self.selected = 0
        self._remove_gizmo()
        self.refresh_markers()
        self._refresh_status()

    def refresh_markers(self) -> None:
        with self._lock, self.server.atomic():
            for idx in [i for i in self._markers if i >= len(self.keypoints)]:
                self._markers.pop(idx).remove()
            if self._spline is not None:
                self._spline.remove()
                self._spline = None

            for idx, pose in enumerate(self.keypoints):
                pos = np.asarray(pose[:3, 3], dtype=np.float64)
                color = (255, 180, 40) if idx == self.selected else (40, 200, 120)
                handle = self.server.scene.add_icosphere(
                    f"/path_drawer/key_{idx:04d}",
                    radius=self.marker_radius,
                    color=color,
                    position=pos,
                )

                @handle.on_click
                def _(_ev, i: int = idx) -> None:
                    self.select(i)

                self._markers[idx] = handle

            if len(self.keypoints) >= 2:
                positions = tuple(
                    tuple(float(x) for x in np.asarray(p[:3, 3], dtype=np.float64))
                    for p in self.keypoints
                )
                self._spline = self.server.scene.add_spline_catmull_rom(
                    "/path_drawer/spline",
                    positions=positions,
                    curve_type="centripetal",
                    line_width=3.0,
                    color=(255, 160, 40),
                    segments=max(64, 16 * len(self.keypoints)),
                )

    def _remove_gizmo(self) -> None:
        if self._gizmo is not None:
            self._gizmo.remove()
            self._gizmo = None

    def _move_gizmo(self) -> None:
        if self._gizmo is None or not self.keypoints:
            return
        wxyz, position = c2w_to_wxyz_position(self.keypoints[self.selected])
        self._suppress_gizmo_cb = True
        try:
            with self.server.atomic():
                self._gizmo.wxyz = wxyz
                self._gizmo.position = position
        finally:
            self._suppress_gizmo_cb = False

    def _attach_gizmo(self) -> None:
        if not self.keypoints:
            self._remove_gizmo()
            return
        self.selected = int(np.clip(self.selected, 0, len(self.keypoints) - 1))
        wxyz, position = c2w_to_wxyz_position(self.keypoints[self.selected])
        self._remove_gizmo()
        gizmo = self.server.scene.add_transform_controls(
            "/path_drawer/gizmo",
            scale=max(self.marker_radius * 6.0, 0.4),
            wxyz=wxyz,
            position=position,
        )

        @gizmo.on_update
        def _(_) -> None:
            if self._suppress_gizmo_cb or not self.keypoints:
                return
            idx = self.selected
            if idx < 0 or idx >= len(self.keypoints):
                return
            self.keypoints[idx] = wxyz_position_to_c2w(
                np.asarray(gizmo.wxyz), np.asarray(gizmo.position)
            )
            marker = self._markers.get(idx)
            if marker is not None:
                marker.position = np.asarray(gizmo.position, dtype=np.float64)
            self._debounce_spline_refresh()

        self._gizmo = gizmo

    def _debounce_spline_refresh(self) -> None:
        # Lightweight: rebuild spline only (markers already moved).
        if len(self.keypoints) < 2:
            return
        with self._lock, self.server.atomic():
            if self._spline is not None:
                self._spline.remove()
            positions = tuple(
                tuple(float(x) for x in np.asarray(p[:3, 3], dtype=np.float64))
                for p in self.keypoints
            )
            self._spline = self.server.scene.add_spline_catmull_rom(
                "/path_drawer/spline",
                positions=positions,
                curve_type="centripetal",
                line_width=3.0,
                color=(255, 160, 40),
                segments=max(64, 16 * len(self.keypoints)),
            )

    def generate_and_save(self) -> None:
        if len(self.keypoints) < 2:
            print("Path drawer: need at least 2 keypoints to generate a path.")
            return
        path = str(self._path_text.value).strip() if self._path_text else self.json_path
        if not path:
            print("Path drawer: JSON path is empty.")
            return
        self.n_frames = int(self._frames_slider.value) if self._frames_slider else self.n_frames
        self.fps = int(self._fps_slider.value) if self._fps_slider else self.fps
        frames = interpolate_camera_path(self.keypoints, self.n_frames, self.look_at)
        meta = {
            "trajectory_type": "path_drawer",
            "num_keypoints": len(self.keypoints),
            "keypoints_c2w": [np.asarray(k, dtype=np.float32).tolist() for k in self.keypoints],
            "look_at": self.look_at.astype(float).tolist(),
            "n_frames": self.n_frames,
            "fps": self.fps,
            "duration_s": self.n_frames / max(self.fps, 1),
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.width * 0.5,
            "cy": self.height * 0.5,
        }
        save_views_json(path, frames, meta=meta)
        self.json_path = path
        self._refresh_status()
        print(
            f"Path drawer: generated {len(frames)} frames from "
            f"{len(self.keypoints)} keypoints -> {os.path.abspath(path)}"
        )


@dataclass
class ViewListEditor:
    """Viser UI to browse / edit / save a list of camera poses in the web scene."""

    server: viser.ViserServer
    json_path: str
    width: int = 960
    height: int = 540
    fx: float = 540.0
    fy: float = 540.0
    frustum_scale: float = 0.35
    views: List[np.ndarray] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    selected: int = 0
    _frustums: Dict[int, Any] = field(default_factory=dict, repr=False)
    _gizmo: Any = field(default=None, repr=False)
    _gizmo_scale: float = field(default=0.0, repr=False)
    _status: Any = field(default=None, repr=False)
    _index_slider: Any = field(default=None, repr=False)
    _path_text: Any = field(default=None, repr=False)
    _show_frustums: Any = field(default=None, repr=False)
    _suppress_gizmo_cb: bool = field(default=False, repr=False)
    _timers: Dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def setup(self) -> None:
        if os.path.isfile(self.json_path):
            try:
                self.views, loaded_meta = load_views_json(self.json_path)
                self.meta.update(loaded_meta)
                self.width = int(self.meta.get("width", self.width))
                self.height = int(self.meta.get("height", self.height))
                self.fx = float(self.meta.get("fx", self.fx))
                self.fy = float(self.meta.get("fy", self.fy))
                print(f"Loaded {len(self.views)} views from {self.json_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not load views JSON ({exc}); starting empty.")
                self.views = []

        with self.server.gui.add_folder("View list"):
            self._status = self.server.gui.add_markdown(self._status_text())
            self._path_text = self.server.gui.add_text("JSON path", initial_value=self.json_path)
            self._show_frustums = self.server.gui.add_checkbox("Show frustums", True)
            scale = self.server.gui.add_slider(
                "Frustum scale", min=0.05, max=2.0, step=0.05, initial_value=self.frustum_scale
            )
            self._index_slider = self.server.gui.add_slider(
                "Selected",
                min=0,
                max=max(len(self.views) - 1, 0),
                step=1,
                initial_value=0,
                disabled=len(self.views) == 0,
            )

            btn_add = self.server.gui.add_button("Add current camera")
            btn_update = self.server.gui.add_button("Update selected from camera")
            btn_goto = self.server.gui.add_button("Go to selected")
            btn_delete = self.server.gui.add_button("Delete selected")
            btn_clear = self.server.gui.add_button("Clear all")
            btn_load = self.server.gui.add_button("Load JSON")
            btn_save = self.server.gui.add_button("Save JSON")

        @self._show_frustums.on_update
        def _(_) -> None:
            self.set_frustums_visible(bool(self._show_frustums.value))

        @scale.on_update
        def _(_) -> None:
            # Changing scale needs a rebuild, so coalesce the drag into one.
            def apply() -> None:
                self.frustum_scale = float(scale.value)
                self.refresh_frustums()
                self._attach_gizmo()

            self._debounce("scale", 0.15, apply)

        @self._index_slider.on_update
        def _(_) -> None:
            self.select(int(self._index_slider.value))

        @btn_add.on_click
        def _(_) -> None:
            self.add_current_camera()

        @btn_update.on_click
        def _(_) -> None:
            self.update_selected_from_camera()

        @btn_goto.on_click
        def _(_) -> None:
            self.goto_selected()

        @btn_delete.on_click
        def _(_) -> None:
            self.delete_selected()

        @btn_clear.on_click
        def _(_) -> None:
            self.clear_all()

        @btn_load.on_click
        def _(_) -> None:
            self.json_path = str(self._path_text.value).strip()
            self.load()

        @btn_save.on_click
        def _(_) -> None:
            self.json_path = str(self._path_text.value).strip()
            self.save()

        self.refresh_frustums()
        self._attach_gizmo()
        self._refresh_status()

    def _status_text(self) -> str:
        if not self.views:
            return "**Views:** 0 (empty). Orbit to a pose, then *Add current camera*."
        return (
            f"**Views:** {len(self.views)} &nbsp;|&nbsp; "
            f"**Selected:** {self.selected} &nbsp;|&nbsp; "
            f"`{os.path.basename(self.json_path)}`"
        )

    def _refresh_status(self) -> None:
        if self._status is not None:
            self._status.content = self._status_text()
        if self._index_slider is not None:
            n = len(self.views)
            self._index_slider.disabled = n == 0
            self._index_slider.max = max(n - 1, 0)
            clamped = int(np.clip(self.selected, 0, max(n - 1, 0)))
            # Assigning .value re-fires on_update, so only write a real change.
            if n and self._index_slider.value != clamped:
                self._index_slider.value = clamped

    def _primary_client(self) -> Optional[viser.ClientHandle]:
        clients = list(self.server.get_clients().values())
        return clients[0] if clients else None

    def _fov_aspect(self) -> Tuple[float, float]:
        fy = self.fy if self.fy > 0 else float(self.height)
        fov = 2.0 * math.atan(0.5 * self.height / fy)
        aspect = float(self.width) / float(max(self.height, 1))
        return float(fov), aspect

    def _debounce(self, key: str, delay: float, fn: Any) -> None:
        """Coalesce bursts of GUI callbacks (slider drags fire per pixel)."""
        pending = self._timers.pop(key, None)
        if pending is not None:
            pending.cancel()
        timer = threading.Timer(delay, fn)
        timer.daemon = True
        self._timers[key] = timer
        timer.start()

    def _add_frustum(self, idx: int, fov: float, aspect: float) -> Any:
        """(Re)create one frustum node. Re-adding the same name replaces it, which
        is the only way to change colour/scale: viser 0.2.1 exposes just
        position/wxyz/visible as mutable properties on the handle."""
        wxyz, position = c2w_to_wxyz_position(self.views[idx])
        handle = self.server.scene.add_camera_frustum(
            f"/views/frustum_{idx:04d}",
            fov=fov,
            aspect=aspect,
            scale=self.frustum_scale,
            color=(80, 180, 255) if idx == self.selected else (40, 40, 40),
            wxyz=wxyz,
            position=position,
            visible=self._show_frustums is None or bool(self._show_frustums.value),
        )

        @handle.on_click
        def _(_ev, i: int = idx) -> None:
            self.select(i)

        self._frustums[idx] = handle
        return handle

    def refresh_frustums(self) -> None:
        """Full rebuild. Only needed when the view list or frustum scale changes."""
        with self._lock:
            fov, aspect = self._fov_aspect()
            show = self._show_frustums is None or bool(self._show_frustums.value)
            # One websocket flush for the whole batch instead of one per node.
            with self.server.atomic():
                for idx in [i for i in self._frustums if i >= len(self.views)]:
                    self._frustums.pop(idx).remove()
                if not show:
                    for handle in self._frustums.values():
                        handle.visible = False
                    return
                for idx in range(len(self.views)):
                    self._add_frustum(idx, fov, aspect)

    def select(self, idx: int) -> None:
        """Move the selection, touching only the two frustums whose colour changes."""
        if not self.views:
            return
        idx = int(np.clip(idx, 0, len(self.views) - 1))
        previous = self.selected
        if idx == previous and previous in self._frustums:
            return
        with self._lock:
            self.selected = idx
            fov, aspect = self._fov_aspect()
            if self._show_frustums is None or bool(self._show_frustums.value):
                with self.server.atomic():
                    for i in {previous, idx}:
                        if 0 <= i < len(self.views):
                            self._add_frustum(i, fov, aspect)
        self._move_gizmo()
        self._refresh_status()

    def set_frustums_visible(self, visible: bool) -> None:
        """Toggling visibility does not require rebuilding the nodes."""
        if visible and len(self._frustums) != len(self.views):
            self.refresh_frustums()
            return
        with self._lock, self.server.atomic():
            for handle in self._frustums.values():
                handle.visible = visible

    def _remove_gizmo(self) -> None:
        if self._gizmo is not None:
            self._gizmo.remove()
            self._gizmo = None

    def _move_gizmo(self) -> None:
        """Point the existing gizmo at the selection instead of recreating it.

        Writing wxyz/position fires the gizmo's own on_update, so suppress the
        callback or it would immediately write the pose back into views[selected].
        """
        if self._gizmo is None or not self.views:
            return
        wxyz, position = c2w_to_wxyz_position(self.views[self.selected])
        self._suppress_gizmo_cb = True
        try:
            with self.server.atomic():
                self._gizmo.wxyz = wxyz
                self._gizmo.position = position
        finally:
            self._suppress_gizmo_cb = False

    def _attach_gizmo(self) -> None:
        if not self.views:
            self._remove_gizmo()
            return
        self.selected = int(np.clip(self.selected, 0, len(self.views) - 1))
        scale = max(self.frustum_scale * 2.5, 0.5)
        # add_transform_controls has no mutable scale, so only rebuild when it changes.
        if self._gizmo is not None and scale == self._gizmo_scale:
            self._move_gizmo()
            return

        self._remove_gizmo()
        wxyz, position = c2w_to_wxyz_position(self.views[self.selected])
        gizmo = self.server.scene.add_transform_controls(
            "/views/gizmo",
            scale=scale,
            wxyz=wxyz,
            position=position,
        )

        @gizmo.on_update
        def _(_) -> None:
            if self._suppress_gizmo_cb or not self.views:
                return
            idx = self.selected
            if idx < 0 or idx >= len(self.views):
                return
            self.views[idx] = wxyz_position_to_c2w(
                np.asarray(gizmo.wxyz), np.asarray(gizmo.position)
            )
            frustum = self._frustums.get(idx)
            if frustum is not None:
                frustum.wxyz = gizmo.wxyz
                frustum.position = gizmo.position

        self._gizmo = gizmo
        self._gizmo_scale = scale

    def add_current_camera(self) -> None:
        client = self._primary_client()
        if client is None:
            print("No viser client connected; open the web UI first.")
            return
        c2w = client_camera_to_c2w(client.camera)
        self.views.append(c2w)
        previous, self.selected = self.selected, len(self.views) - 1
        fov, aspect = self._fov_aspect()
        with self._lock, self.server.atomic():
            for i in {previous, self.selected}:
                if 0 <= i < len(self.views):
                    self._add_frustum(i, fov, aspect)
        self._attach_gizmo()
        self._refresh_status()
        print(f"Added view {self.selected} (total {len(self.views)})")

    def update_selected_from_camera(self) -> None:
        client = self._primary_client()
        if client is None or not self.views:
            return
        idx = self.selected
        self.views[idx] = client_camera_to_c2w(client.camera)
        handle = self._frustums.get(idx)
        if handle is not None:
            wxyz, position = c2w_to_wxyz_position(self.views[idx])
            with self.server.atomic():
                handle.wxyz = wxyz
                handle.position = position
        self._move_gizmo()
        self._refresh_status()
        print(f"Updated view {idx}")

    def goto_selected(self) -> None:
        client = self._primary_client()
        if client is None or not self.views:
            return
        apply_c2w_to_client(client.camera, self.views[self.selected])
        print(f"Moved client camera to view {self.selected}")

    def delete_selected(self) -> None:
        if not self.views:
            return
        del self.views[self.selected]
        self.selected = min(self.selected, max(len(self.views) - 1, 0))
        self.refresh_frustums()
        self._attach_gizmo()
        self._refresh_status()

    def clear_all(self) -> None:
        self.views = []
        self.selected = 0
        self._remove_gizmo()
        self.refresh_frustums()
        self._refresh_status()

    def load(self) -> None:
        path = str(self._path_text.value).strip() if self._path_text else self.json_path
        if not path:
            print("JSON path is empty.")
            return
        self.views, loaded_meta = load_views_json(path)
        self.meta.update(loaded_meta)
        self.json_path = path
        self.selected = 0
        self.width = int(self.meta.get("width", self.width))
        self.height = int(self.meta.get("height", self.height))
        self.fx = float(self.meta.get("fx", self.fx))
        self.fy = float(self.meta.get("fy", self.fy))
        self.refresh_frustums()
        self._attach_gizmo()
        self._refresh_status()
        print(f"Loaded {len(self.views)} views from {path}")

    def save(self) -> None:
        path = str(self._path_text.value).strip() if self._path_text else self.json_path
        if not path:
            print("JSON path is empty.")
            return
        meta = dict(self.meta)
        meta.update(
            {
                "width": self.width,
                "height": self.height,
                "fx": self.fx,
                "fy": self.fy,
                "cx": self.width * 0.5,
                "cy": self.height * 0.5,
            }
        )
        save_views_json(path, self.views, meta=meta)
        self.json_path = path
        self._refresh_status()


def setup_view_list_editor(
    server: viser.ViserServer,
    json_path: str,
    width: int = 960,
    height: int = 540,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
) -> ViewListEditor:
    """Attach the editable view-position list UI to an existing viser server."""
    editor = ViewListEditor(
        server=server,
        json_path=os.path.abspath(json_path),
        width=width,
        height=height,
        fx=float(fx if fx is not None else height),
        fy=float(fy if fy is not None else height),
    )
    editor.setup()
    return editor


def setup_path_drawer(
    server: viser.ViserServer,
    json_path: str,
    look_at: np.ndarray,
    extent: float,
    width: int = 960,
    height: int = 540,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    n_frames: int = 600,
    fps: int = 25,
) -> PathDrawer:
    """Attach the click-to-draw camera path editor to an existing viser server."""
    drawer = PathDrawer(
        server=server,
        json_path=os.path.abspath(json_path),
        look_at=np.asarray(look_at, dtype=np.float64),
        extent=float(extent),
        width=width,
        height=height,
        fx=float(fx if fx is not None else height),
        fy=float(fy if fy is not None else height),
        n_frames=n_frames,
        fps=fps,
    )
    drawer.setup()
    return drawer


def launch_viewer(
    splats_list: List[Dict[str, Tensor]],
    frame: int,
    port: int,
    radius_clip: float,
    views_json: Optional[str] = None,
    views_width: int = 960,
    views_height: int = 540,
    views_fx: Optional[float] = None,
    views_fy: Optional[float] = None,
    path_json: Optional[str] = None,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("nerfview rasterization needs a CUDA GPU.")
    device = torch.device("cuda")

    n_frames = len(splats_list)
    frame = int(np.clip(frame, 0, n_frames - 1))
    cache: Dict[str, Any] = {"idx": -1, "gpu": None}

    def gpu_frame(idx: int) -> Dict[str, Tensor]:
        if cache["idx"] != idx:
            cache["gpu"] = _activate_for_render(splats_list[idx], device)
            cache["idx"] = idx
            print(f"Viewer frame {idx}: {int(cache['gpu']['means'].shape[0])} Gaussians")
        return cache["gpu"]

    gpu_frame(frame)
    center, extent = _scene_center(splats_list[frame])
    state = {"frame": frame}

    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        packed = gpu_frame(int(state["frame"]))
        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K = torch.from_numpy(camera_state.get_K(img_wh)).float().to(device)
        render_colors, _, _ = rasterization(
            packed["means"],
            packed["quats"],
            packed["scales"],
            packed["opacities"],
            packed["colors"],
            c2w.inverse()[None],
            K[None],
            width,
            height,
            sh_degree=packed["sh_degree"],
            render_mode="RGB",
            radius_clip=radius_clip,
        )
        return render_colors[0, ..., 0:3].cpu().numpy()

    server = viser.ViserServer(port=port, verbose=False)
    viewer = nerfview.Viewer(server=server, render_fn=viewer_render_fn, mode="rendering")

    if n_frames > 1:
        gui_frame = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=frame
        )

        @gui_frame.on_update
        def _(_) -> None:
            state["frame"] = int(gui_frame.value)
            gpu_frame(state["frame"])
            rerender = getattr(viewer, "rerender", None)
            if callable(rerender):
                rerender(_)

    if views_json is None:
        views_json = os.path.join(os.getcwd(), "cameras_edited.json")
    setup_view_list_editor(
        server,
        json_path=views_json,
        width=views_width,
        height=views_height,
        fx=views_fx,
        fy=views_fy,
    )
    if path_json is None:
        path_json = os.path.join(os.getcwd(), "cameras_drawn_path.json")
    setup_path_drawer(
        server,
        json_path=path_json,
        look_at=center,
        extent=extent,
        width=views_width,
        height=views_height,
        fx=views_fx,
        fy=views_fy,
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = center + np.array([0.0, 0.0, extent], dtype=np.float32)
        look_at = getattr(client.camera, "look_at", None)
        if look_at is not None:
            client.camera.look_at = center

    print(f"nerfview at http://localhost:{port}  (Ctrl+C to exit)")
    print(f"View list JSON: {os.path.abspath(views_json)}")
    print(f"Path drawer JSON: {os.path.abspath(path_json)}")
    while True:
        time.sleep(100000)


def main(cfg: Config) -> None:
    _, compress_dir, ply_dir = _resolve_dirs(cfg)
    print(f"Compress dir: {compress_dir}")
    splats_list = decode_sequence(compress_dir, keep_padding=cfg.keep_padding)
    save_plys(splats_list, ply_dir)
    if cfg.no_view:
        print(f"Done. Decoded PLYs in {ply_dir}")
        return
    launch_viewer(
        splats_list,
        cfg.frame,
        cfg.port,
        cfg.radius_clip,
        views_json=cfg.views_json,
        views_width=cfg.views_width,
        views_height=cfg.views_height,
        views_fx=cfg.views_fx,
        views_fy=cfg.views_fy,
        path_json=cfg.path_json,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
