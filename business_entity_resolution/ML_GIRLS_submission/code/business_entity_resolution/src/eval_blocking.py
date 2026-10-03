"""Recall ceiling of the candidate set vs. ground truth (train only).

usage: python eval_blocking.py [tag]
"""
import sys

import numpy as np
import pandas as pd

import config


def load_gt_pairs():
    """-> DataFrame(s1, src, t) with row indices into the normalised parquet files, and the S1 true-count."""
    gt = pd.read_csv(config.raw_path("train", "ground_truth"), sep="\t", dtype=str, keep_default_na=False)
    s1_ids = pd.read_parquet(config.norm_path("train", "source1"), columns=["entity_id"])["entity_id"]
    s1_pos = pd.Series(np.arange(len(s1_ids), dtype=np.int32), index=s1_ids.values)
    e = gt.assign(c=gt["matched_entity_ids"].str.split(",")).explode("c")
    e = e[e["c"].fillna("") != ""]
    out = []
    for src in ("source2", "source3"):
        ids = pd.read_parquet(config.norm_path("train", src), columns=["entity_id"])["entity_id"]
        pos = pd.Series(np.arange(len(ids), dtype=np.int32), index=ids.values)
        pre = "S2-" if src == "source2" else "S3-"
        sub = e[e["c"].str.startswith(pre)]
        out.append(pd.DataFrame({"s1": s1_pos.loc[sub["source1_entity_id"].values].values,
                                 "src": np.int8(2 if src == "source2" else 3),
                                 "t": pos.loc[sub["c"].values].values}))
    return pd.concat(out, ignore_index=True)


def main(tag=""):
    gt = load_gt_pairs()
    for src, name in ((2, "source2"), (3, "source3")):
        path = config.work("cand", f"train_{name}{tag}.parquet")
        if not __import__("os").path.exists(path):
            continue
        cand = pd.read_parquet(path, columns=["s1", "t", "hit", "rank", "score"])
        s1_in = np.unique(cand["s1"].values)
        g = gt[(gt["src"] == src) & np.isin(gt["s1"].values, s1_in)]
        m = g.merge(cand, on=["s1", "t"], how="left")
        print(f"\n== {name}: {len(s1_in)} S1 queried, {len(g)} true pairs, {len(cand)} candidates "
              f"({len(cand) / len(s1_in):.1f}/S1)")
        print(f"  union recall: {m['rank'].notna().mean():.4f}")
        for bit, nm in ((1, "name"), (2, "addr"), (4, "comb"), (8, "rev")):
            print(f"    found by {nm:5s}: {((m['hit'].fillna(0).astype(int) & bit) > 0).mean():.4f}")
        for k in (1, 3, 5, 10, 15, 20, 30, 40, 60):
            print(f"  recall@{k:<3d}: {(m['rank'] < k).mean():.4f}")
        found = m["rank"].notna()
        print("  score of true pairs (quantiles):", np.round(m.loc[found, "score"].quantile([.01, .05, .1, .5]).values, 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
