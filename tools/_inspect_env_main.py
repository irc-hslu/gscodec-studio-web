#!/usr/bin/env python3
"""Compare environment.ply vs point_cloud.ply attribute ranges."""
import os
import sys
import numpy as np
from plyfile import PlyData


def stats(path: str) -> None:
    p = PlyData.read(path)
    v = p["vertex"]
    n = v.count
    props = [pr.name for pr in v.properties]
    sh0 = len([x for x in props if x.startswith("f_dc_")])
    shN = len([x for x in props if x.startswith("f_rest_")])
    x, y, z = v["x"], v["y"], v["z"]
    s0, s1, s2 = v["scale_0"], v["scale_1"], v["scale_2"]
    op = v["opacity"]
    print("===", os.path.basename(path), "n=", n, "sh0=", sh0, "shN=", shN)
    print("  pos min", np.min([x.min(), y.min(), z.min()]), "max", np.max([x.max(), y.max(), z.max()]))
    print("  scale log min", np.min([s0.min(), s1.min(), s2.min()]), "max", np.max([s0.max(), s1.max(), s2.max()]))
    print("  opacity min", float(op.min()), "max", float(op.max()))
    exp_s = np.exp(np.stack([s0, s1, s2], axis=1))
    print("  scale exp median", float(np.median(exp_s)), "p99", float(np.percentile(exp_s, 99)))


def main() -> None:
    bases = [
        "/mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH/ply-result/point_cloud/iteration_100",
        "/mnt/d/LiveStream/GSCodec_Studio/MellifontAbbey_1_SH_fast/ply-result/point_cloud/iteration_100",
        "/mnt/d/LiveStream/GSCodec_Studio/ply",
    ]
    if len(sys.argv) > 1:
        bases = sys.argv[1:]
    for base in bases:
        print("\n#", base)
        if os.path.isdir(base):
            names = sorted(
                f for f in os.listdir(base)
                if f.endswith(".ply")
            )
        else:
            names = ["environment.ply", "point_cloud.ply"]
        for name in names:
            p = os.path.join(base, name) if os.path.isdir(base) else base
            if os.path.isfile(p):
                stats(p)
            else:
                print("missing", p)


if __name__ == "__main__":
    main()
