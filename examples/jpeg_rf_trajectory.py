"""JPEG RF Exploration Study 5 — automated test camera trajectories (PCQ M109040).

Implements key-view selection, ghost augmentation, centripetal Catmull–Rom splines,
arc-length reparameterization, and quaternion SLERP orientations as described in:
  docs/PCQ-Report on Exploration Study 5 of JPEG RF.pdf
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.traj import viewmatrix


@dataclass
class GhostWeights:
    w_orient: float = 0.45
    w_empty: float = 0.45
    w_span: float = 0.10


@dataclass
class TrajectoryConfig:
    name: str = "ghost_gentle_orient"
    label: str = ""
    family: str = "ghost"
    num_keys: int = 8
    num_ghosts: int = 4
    alpha: float = 0.8
    ghost_weights: GhostWeights = field(default_factory=GhostWeights)
    n_frames: int = 600
    fps: int = 25
    k_neighbors: int = 8
    dense_top_percent: float = 80.0
    delta_min_keys: float = 0.15
    delta_min_ghosts: float = 0.10
    ghost_candidates: int = 256
    arc_length_samples: int = 4096
    notes: str = ""

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any], defaults: Dict[str, Any]) -> "TrajectoryConfig":
        gw = data.get("ghost_weights", {})
        video = defaults.get("default_video", {})
        sel = defaults.get("key_selection", {})
        return cls(
            name=name,
            label=str(data.get("label", name)),
            family=str(data.get("family", "ghost")),
            num_keys=int(data.get("num_keys", 8)),
            num_ghosts=int(data.get("num_ghosts", 0)),
            alpha=float(data.get("alpha", 0.8)),
            ghost_weights=GhostWeights(
                w_orient=float(gw.get("w_orient", 0.45)),
                w_empty=float(gw.get("w_empty", 0.45)),
                w_span=float(gw.get("w_span", 0.10)),
            ),
            n_frames=int(data.get("n_frames", video.get("n_frames", 600))),
            fps=int(data.get("fps", video.get("fps", 25))),
            k_neighbors=int(data.get("k_neighbors", sel.get("k_neighbors", 8))),
            dense_top_percent=float(data.get("dense_top_percent", sel.get("dense_top_percent", 80.0))),
            delta_min_keys=float(data.get("delta_min_keys", sel.get("delta_min_keys", 0.15))),
            delta_min_ghosts=float(data.get("delta_min_ghosts", sel.get("delta_min_ghosts", 0.10))),
            ghost_candidates=int(data.get("ghost_candidates", sel.get("ghost_candidates", 256))),
            arc_length_samples=int(
                data.get("arc_length_samples", video.get("arc_length_samples", 4096))
            ),
            notes=str(data.get("notes", "")),
        )


def load_trajectory_registry(path: str | Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_trajectory_config(path: str | Path, name: str) -> TrajectoryConfig:
    reg = load_trajectory_registry(path)
    if name not in reg["trajectories"]:
        available = ", ".join(reg["trajectories"].keys())
        raise KeyError(f"Unknown trajectory '{name}'. Available: {available}")
    return TrajectoryConfig.from_dict(name, reg["trajectories"][name], reg)


def c2w_to_center_forward(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c2w = np.asarray(c2w, dtype=np.float64)
    center = c2w[:3, 3]
    forward = -c2w[:3, 2]
    forward /= np.linalg.norm(forward) + 1e-12
    return center, forward


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """w, x, y, z."""
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta = math.acos(dot)
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) / s) * q0 + (math.sin(t * theta) / s) * q1


def density_weighted_scene_center(
    centers: np.ndarray,
    k_neighbors: int,
    dense_top_percent: float,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    n = centers.shape[0]
    k = min(k_neighbors, max(1, n - 1))
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    knn = np.partition(dists, kth=k - 1, axis=1)[:, :k]
    mean_d = knn.mean(axis=1)
    rho = 1.0 / (eps + mean_d)
    keep_n = max(1, int(math.ceil(n * dense_top_percent / 100.0)))
    dense_idx = np.argsort(-rho)[:keep_n]
    x0 = (rho[dense_idx, None] * centers[dense_idx]).sum(axis=0) / rho[dense_idx].sum()
    return x0, dense_idx


def select_key_views(
    c2w: np.ndarray,
    num_keys: int,
    alpha: float,
    x0: np.ndarray,
    delta_min: float,
) -> List[int]:
    n = c2w.shape[0]
    centers = c2w[:, :3, 3]
    forwards = -c2w[:, :3, 2]
    forwards /= np.linalg.norm(forwards, axis=1, keepdims=True) + 1e-12

    dists0 = np.linalg.norm(centers - x0[None, :], axis=1)
    r95 = float(np.percentile(dists0, 95)) + 1e-9

    first = int(np.argmax(dists0))
    keys = [first]
    min_sep = delta_min * r95

    while len(keys) < min(num_keys, n):
        best_j = -1
        best_score = -1.0
        for j in range(n):
            if j in keys:
                continue
            d_to_keys = min(np.linalg.norm(centers[j] - centers[k]) for k in keys)
            if d_to_keys < min_sep:
                continue
            D = d_to_keys / r95
            O = 1.0 - max(abs(float(np.dot(forwards[j], forwards[k]))) for k in keys)
            score = alpha * D + (1.0 - alpha) * O
            if score > best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            break
        keys.append(best_j)
    return keys


def _order_control_points(
    centers: np.ndarray,
    indices: List[int],
    start_idx: int,
) -> List[int]:
    """Greedy nearest-neighbor tour through key/ghost indices."""
    remaining = [i for i in indices if i != start_idx]
    ordered = [start_idx]
    while remaining:
        last = ordered[-1]
        nxt = min(remaining, key=lambda j: np.linalg.norm(centers[j] - centers[last]))
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


def generate_ghost_views(
    c2w: np.ndarray,
    key_indices: List[int],
    num_ghosts: int,
    weights: GhostWeights,
    x0: np.ndarray,
    delta_min: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    if num_ghosts <= 0 or len(key_indices) < 2:
        return []

    centers = c2w[:, :3, 3]
    forwards = -c2w[:, :3, 2]
    forwards /= np.linalg.norm(forwards, axis=1, keepdims=True) + 1e-12
    r95 = float(np.percentile(np.linalg.norm(centers - x0[None, :], axis=1), 95)) + 1e-9
    min_sep = delta_min * r95

    existing = [centers[i] for i in key_indices]
    existing_f = [forwards[i] for i in key_indices]
    all_pts = list(centers)
    all_f = list(forwards)

    ghosts: List[Dict[str, Any]] = []
    pairs = [(key_indices[i], key_indices[i + 1]) for i in range(len(key_indices) - 1)]
    if len(key_indices) >= 2:
        pairs.append((key_indices[-1], key_indices[0]))

    candidates: List[Tuple[float, np.ndarray, np.ndarray, Tuple[int, int]]] = []
    for p, q in pairs:
        cp = centers[p]
        cq = centers[q]
        fp = forwards[p]
        span = float(np.linalg.norm(cq - cp))
        for _ in range(max(1, n_candidates // max(1, len(pairs)))):
            t = float(rng.uniform(0.15, 0.85))
            pos = (1 - t) * cp + t * cq
            # Perturb toward sparse zone: push away from nearest real camera.
            nearest = centers[np.argmin(np.linalg.norm(centers - pos[None, :], axis=1))]
            direction = pos - nearest
            if np.linalg.norm(direction) > 1e-6:
                pos = pos + 0.25 * direction / (np.linalg.norm(direction) + 1e-12) * r95 * 0.1
            f = (1 - t) * fp + t * forwards[q]
            f /= np.linalg.norm(f) + 1e-12
            E = float(np.min(np.linalg.norm(centers - pos[None, :], axis=1)))
            O = 1.0 - abs(float(np.dot(f, fp)))
            S = span
            score = weights.w_empty * (E / r95) + weights.w_orient * O + weights.w_span * (S / r95)
            candidates.append((score, pos, f, (p, q)))

    candidates.sort(key=lambda x: -x[0])
    for score, pos, f, bridge in candidates:
        if len(ghosts) >= num_ghosts:
            break
        if any(np.linalg.norm(pos - g["position"]) < min_sep for g in ghosts):
            continue
        if any(np.linalg.norm(pos - p) < min_sep for p in all_pts):
            continue
        ghosts.append(
            {
                "position": pos,
                "forward": f,
                "score": score,
                "bridge": bridge,
            }
        )
        all_pts.append(pos)
        all_f.append(f)
    return ghosts


def _centripetal_tj(pi: np.ndarray, pj: np.ndarray, alpha: float = 0.5) -> float:
    return float(np.linalg.norm(pj - pi) ** alpha)


def catmull_rom_point(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    u: float,
    alpha: float = 0.5,
) -> np.ndarray:
    t0 = 0.0
    t1 = _centripetal_tj(p0, p1, alpha)
    t2 = t1 + _centripetal_tj(p1, p2, alpha)
    t3 = t2 + _centripetal_tj(p2, p3, alpha)
    if t2 - t1 < 1e-12:
        return p1.copy()
    t = t1 + u * (t2 - t1)

    def lerp(a: np.ndarray, b: np.ndarray, ta: float, tb: float) -> np.ndarray:
        if abs(tb - ta) < 1e-12:
            return a.copy()
        w = (t - ta) / (tb - ta)
        return (1 - w) * a + w * b

    A1 = lerp(p0, p1, t0, t1)
    A2 = lerp(p1, p2, t1, t2)
    A3 = lerp(p2, p3, t2, t3)
    B1 = lerp(A1, A2, t0, t2)
    B2 = lerp(A2, A3, t1, t3)
    return lerp(B1, B2, t1, t2)


def sample_spline_arc_length(
    points: np.ndarray,
    n_frames: int,
    dense_samples: int,
    alpha: float = 0.5,
) -> np.ndarray:
    m = points.shape[0]
    if m < 2:
        return np.repeat(points[:1], n_frames, axis=0)
    if m == 2:
        t = np.linspace(0, 1, n_frames)
        return (1 - t)[:, None] * points[0] + t[:, None] * points[1]

    dense_pts = []
    segments = m - 1
    per_seg = max(2, dense_samples // segments)
    for i in range(segments):
        p0 = points[i - 1 if i > 0 else 0]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2 if i + 2 < m else m - 1]
        for j in range(per_seg):
            u = j / per_seg
            dense_pts.append(catmull_rom_point(p0, p1, p2, p3, u, alpha=alpha))
    dense_pts.append(points[-1])
    dense_pts = np.stack(dense_pts, axis=0)

    seg_len = np.linalg.norm(dense_pts[1:] - dense_pts[:-1], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(dense_pts[:1], n_frames, axis=0)

    targets = np.linspace(0.0, total, n_frames)
    out = np.zeros((n_frames, 3), dtype=np.float64)
    for i, s in enumerate(targets):
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = min(max(idx, 0), len(seg_len) - 1)
        local = (s - cum[idx]) / (seg_len[idx] + 1e-12)
        out[i] = (1 - local) * dense_pts[idx] + local * dense_pts[idx + 1]
    return out


def interpolate_orientations(
    control_c2w: List[np.ndarray],
    n_frames: int,
    segment_lengths: np.ndarray,
) -> List[np.ndarray]:
    quats = [rotation_to_quaternion(c[:3, :3]) for c in control_c2w]
    m = len(quats)
    if m == 1:
        return [control_c2w[0].copy() for _ in range(n_frames)]

    total = segment_lengths.sum()
    if total < 1e-9:
        return [control_c2w[0].copy() for _ in range(n_frames)]

    cum = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, total, n_frames)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    poses = []
    for s in targets:
        seg = int(np.searchsorted(cum, s, side="right") - 1)
        seg = min(max(seg, 0), m - 2)
        local = (s - cum[seg]) / (segment_lengths[seg] + 1e-12)
        q = slerp(quats[seg], quats[seg + 1], float(local))
        R = quaternion_to_rotation(q)
        pos = control_c2w[seg][:3, 3]  # replaced later by spline position
        c2w3x4 = np.eye(4, dtype=np.float64)
        c2w3x4[:3, :3] = R
        c2w3x4[:3, 3] = pos
        poses.append(c2w3x4)
    return poses


def build_trajectory(
    training_c2w: np.ndarray,
    cfg: TrajectoryConfig,
    look_at: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate JPEG-RF trajectory poses [N,4,4] and metadata."""
    c2w = np.asarray(training_c2w, dtype=np.float64)
    centers = c2w[:, :3, 3]
    x0, dense_idx = density_weighted_scene_center(
        centers, cfg.k_neighbors, cfg.dense_top_percent
    )
    key_idx = select_key_views(
        c2w, cfg.num_keys, cfg.alpha, x0, cfg.delta_min_keys
    )
    rng = np.random.default_rng(seed)
    ghosts = generate_ghost_views(
        c2w,
        key_idx,
        cfg.num_ghosts,
        cfg.ghost_weights,
        x0,
        cfg.delta_min_ghosts,
        cfg.ghost_candidates,
        rng,
    )

    control_positions: List[np.ndarray] = [centers[i] for i in key_idx]
    control_c2w: List[np.ndarray] = [c2w[i].copy() for i in key_idx]
    ghost_meta = []
    for g in ghosts:
        pos = g["position"]
        fwd = g["forward"]
        if look_at is not None:
            lookdir = look_at - pos
        else:
            lookdir = fwd
        vm = viewmatrix(lookdir, np.array([0.0, 0.0, 1.0]), pos)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :4] = vm
        pose[:3, 0] *= -1.0
        pose[:3, 1] *= -1.0
        control_c2w.append(pose)
        control_positions.append(pos)
        ghost_meta.append(
            {
                "position": pos.tolist(),
                "bridge": list(g["bridge"]),
                "score": float(g["score"]),
            }
        )

    start = key_idx[0]
    order = _order_control_points(
        np.stack(control_positions, axis=0),
        list(range(len(control_positions))),
        0 if start == key_idx[0] else 0,
    )
    control_positions = [control_positions[i] for i in order]
    control_c2w = [control_c2w[i] for i in order]

    pts = np.stack(control_positions, axis=0)
    positions = sample_spline_arc_length(
        pts, cfg.n_frames, cfg.arc_length_samples, alpha=0.5
    )

    seg_len = np.linalg.norm(pts[1:] - pts[:-1], axis=1) if len(pts) > 1 else np.array([1.0])
    orient_templates = interpolate_orientations(control_c2w, cfg.n_frames, seg_len)

    if look_at is None:
        look_at = x0
    look_at = np.asarray(look_at, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    out_poses = []
    for i in range(cfg.n_frames):
        pos = positions[i]
        lookdir = look_at - pos
        vm = viewmatrix(lookdir, up, pos)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :4] = vm.astype(np.float32)
        pose[:3, 0] *= -1.0
        pose[:3, 1] *= -1.0
        # Blend SLERP orientation with look-at for stability.
        R_look = pose[:3, :3].astype(np.float64)
        R_slerp = orient_templates[i][:3, :3]
        q_look = rotation_to_quaternion(R_look)
        q_slerp = rotation_to_quaternion(R_slerp)
        q = slerp(q_slerp, q_look, 0.35)
        pose[:3, :3] = quaternion_to_rotation(q).astype(np.float32)
        pose[:3, 3] = pos.astype(np.float32)
        out_poses.append(pose)

    meta = {
        "trajectory": cfg.name,
        "label": cfg.label,
        "family": cfg.family,
        "document": "JPEG RF Exploration Study 5 (PCQ M109040)",
        "scene_center_x0": x0.tolist(),
        "look_at": look_at.tolist(),
        "num_keys": len(key_idx),
        "num_ghosts": len(ghosts),
        "key_indices": key_idx,
        "ghosts": ghost_meta,
        "alpha": cfg.alpha,
        "ghost_weights": {
            "w_orient": cfg.ghost_weights.w_orient,
            "w_empty": cfg.ghost_weights.w_empty,
            "w_span": cfg.ghost_weights.w_span,
        },
        "n_frames": cfg.n_frames,
        "fps": cfg.fps,
        "duration_s": cfg.n_frames / max(cfg.fps, 1),
        "notes": cfg.notes,
    }
    return np.stack(out_poses, axis=0), meta


def save_trajectory_json(
    path: str | Path,
    c2w: np.ndarray,
    meta: Dict[str, Any],
    source_cameras: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "trajectory_type": "jpeg_rf_es5",
        **meta,
        "c2w": c2w.tolist(),
    }
    if source_cameras:
        for k in ("width", "height", "fx", "fy", "cx", "cy"):
            if k in source_cameras:
                payload[k] = source_cameras[k]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
