#!/usr/bin/env python3
"""Compare scene-only vs env+scene centroids for Mellifont."""
import os
import numpy as np
from plyfile import PlyData
def centroid(path: str):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    return xyz.mean(0), xyz.shape[0], xyz.min(0), xyz.max(0)
base = "/mnt/d/LiveStream/GSCodec_Studio/AbbeyStreet_UpperFloor/ply-result/lod0"
scene_c, n_s, mn_s, mx_s = centroid(os.path.join(base, "point_cloud.ply"))
env_c, n_e, mn_e, mx_e = centroid(os.path.join(base, "environment.ply"))
# merged mean = weighted by count
merged = (scene_c * n_s + env_c * n_e) / (n_s + n_e)
print(f"scene n={n_s} center={scene_c.tolist()}")
print(f"scene bbox min={mn_s.tolist()} max={mx_s.tolist()}")
print(f"env   n={n_e} center={env_c.tolist()}")
print(f"env   bbox min={mn_e.tolist()} max={mx_e.tolist()}")
print(f"merged n={n_s+n_e} center={merged.tolist()}")
print(f"delta (merged - scene)={ (merged - scene_c).tolist()}")
