"""Explore rank-based pruning rules on a blocking sample: recall kept vs pairs kept."""
import sys

import numpy as np
import pandas as pd

import config
from ctx import group_stats

tag = sys.argv[1] if len(sys.argv) > 1 else "_s100k"
gt = pd.read_parquet(config.work("gt_pairs.parquet"))
for src, name in ((2, "source2"), (3, "source3")):
    c = pd.read_parquet(config.work("cand", f"train_{name}{tag}.parquet"))
    s1 = c["s1"].to_numpy()
    c["r_score"] = group_stats(s1, c["score"].to_numpy(np.float32))[0]
    c["r_a"] = group_stats(s1, (c["cos_a"] + c["cos_x"]).to_numpy(np.float32))[0]
    c["r_n"] = group_stats(s1, (c["cos_n"] + c["cos_g"] + c["cos_k"] + c["cos_p"]).to_numpy(np.float32))[0]
    c["r_m"] = group_stats(s1, c["cos_m"].to_numpy(np.float32))[0]
    key = set(zip(gt.loc[gt["src"] == src, "s1"], gt.loc[gt["src"] == src, "t"]))
    c["y"] = [(a, b) in key for a, b in zip(c["s1"].to_numpy(), c["t"].to_numpy())]
    n_true = len(gt[(gt["src"] == src) & gt["s1"].isin(set(s1))])
    print(f"\n== {name}: blocked recall {c['y'].sum() / n_true:.4f}, pairs/S1 {len(c) / c['s1'].nunique():.1f}")
    rev = (c["hit"].to_numpy() & 8) > 0
    for R1, R2, R3, R4 in [(10, 3, 3, 0), (15, 5, 5, 0), (15, 5, 5, 3), (20, 5, 5, 3), (20, 8, 8, 3), (25, 10, 10, 5),
                           (30, 10, 10, 5), (40, 15, 15, 5)]:
        keep = (c["r_score"] < R1) | (c["r_a"] < R2) | (c["r_n"] < R3) | (c["r_m"] < R4)
        keep_r = keep | rev
        for nm, k in (("rank-rule", keep), ("+reverse", keep_r)):
            print(f"  score<{R1:2d} a<{R2:2d} n<{R3:2d} m<{R4} {nm:9s}: recall {c.loc[k, 'y'].sum() / n_true:.4f}  "
                  f"pairs/S1 {k.sum() / c['s1'].nunique():.1f}")
