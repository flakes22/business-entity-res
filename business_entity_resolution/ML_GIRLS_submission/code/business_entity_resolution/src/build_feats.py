"""Step 3: pair features for the candidate set.

iter_feature_chunks() streams (ids, features) for every candidate pair of one target
source, optionally restricted to a subset of S1 rows. Context features are computed
over the WHOLE candidate set first (all S1 records), so they mean the same thing for
a subset as for the full test set.
"""
import gc
import time

import numpy as np
import pandas as pd

import config
from ctx import add_context
from features import Records, string_features

SRC_CODE = {"source2": 2, "source3": 3}
BLOCK_FEATS = ["cos_n", "cos_k", "cos_g", "cos_p", "cos_a", "cos_x", "cos_m", "score", "rank", "rrank",
               "hit_name", "hit_addr", "hit_comb", "hit_rev",
               "grp_rank", "grp_gap", "grp_n", "grp_gap2", "grp_rank_a", "grp_gap_a", "grp_rank_n", "grp_gap_n",
               "rev_rank", "rev_gap", "rev_n", "rev_gap2", "is_s3"]


def load_candidates(split, source):
    c = pd.read_parquet(config.work("cand", f"{split}_{source}.parquet"))
    c = add_context(c)
    hit = c["hit"].to_numpy()
    for bit, nm in enumerate(("name", "addr", "comb", "rev")):
        c["hit_" + nm] = ((hit >> bit) & 1).astype(np.float32)
    del c["hit"]
    c["rrank"] = c["rrank"].astype(np.float32)
    c["rank"] = c["rank"].astype(np.float32)
    c["is_s3"] = np.float32(source == "source3")
    c["src"] = np.int8(SRC_CODE[source])
    return c


def iter_feature_chunks(split, source, s1_rows=None, chunk=2_000_000, recs=None, row_mask=None):
    """yields DataFrames: s1, t, src + BLOCK_FEATS + string features.
    row_mask: optional boolean array aligned with the candidate file (applied after the context features)."""
    c = load_candidates(split, source)
    keep = np.ones(len(c), dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool).copy()
    if s1_rows is not None:
        keep &= np.isin(c["s1"].to_numpy(), s1_rows)
    pos = np.flatnonzero(keep)             # positional chunks: never copies the whole table
    del keep
    gc.collect()
    if recs is None:
        recs = (Records(split, "source1"), Records(split, source))
    a, b = recs
    t0 = time.time()
    for st in range(0, len(pos), chunk):
        part = c.iloc[pos[st:st + chunk]].reset_index(drop=True)
        f = string_features(a, b, part["s1"].to_numpy(), part["t"].to_numpy())
        out = pd.concat([part, pd.DataFrame(f)], axis=1)
        print(f"    {split}/{source}: {min(st + chunk, len(pos))}/{len(pos)} pairs ({time.time() - t0:.0f}s)",
              flush=True)
        yield out
        del part, f, out


def feature_columns(df):
    return [c for c in df.columns if c not in ("s1", "t", "src", "label")]


def gt_keys(split="train"):
    """set of (src*2^32 + s1)*... -> int64 keys for fast label lookup: key = src<<60 | s1<<30 | t."""
    gt = pd.read_parquet(config.work("gt_pairs.parquet"))
    return set(pair_key(gt["src"].to_numpy(), gt["s1"].to_numpy(), gt["t"].to_numpy()).tolist())


def pair_key(src, s1, t):
    return (src.astype(np.int64) << 60) | (s1.astype(np.int64) << 30) | t.astype(np.int64)


def label(df, keys_arr):
    k = pair_key(df["src"].to_numpy(), df["s1"].to_numpy(), df["t"].to_numpy())
    return np.isin(k, keys_arr).astype(np.int8)
