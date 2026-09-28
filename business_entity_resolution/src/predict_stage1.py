"""Step 5: stage-1 probabilities for EVERY candidate pair of a split (streamed; features are not kept).

Output work/p1/<split>_<source>.parquet: s1, t, src, pA, pB  (model A / model B probabilities)
usage: python predict_stage1.py train|test
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from build_feats import iter_feature_chunks
from features import Records


def load_stage1():
    with open(config.work("models", "stage1_features.json")) as f:
        cols = json.load(f)
    models = {k: lgb.Booster(model_file=config.work("models", f"stage1_{k}.txt")) for k in ("A", "B")}
    return models, cols


def run(split):
    models, cols = load_stage1()
    a = Records(split, "source1")
    for src in ("source2", "source3"):
        out = config.work("p1", f"{split}_{src}.parquet")
        if os.path.exists(out):
            print(f"  {out} exists, skipping")
            continue
        b = Records(split, src)
        mask = None
        if split == "train":
            # stage 2 only needs pairs of the model-selection S1 records plus every competing
            # claim on the targets those records retrieved
            from train_stage1 import splits
            rows = np.concatenate(list(splits().values()))
            c = pd.read_parquet(config.work("cand", f"{split}_{src}.parquet"), columns=["s1", "t"])
            in_sub = np.isin(c["s1"].to_numpy(), rows)
            mask = in_sub | np.isin(c["t"].to_numpy(), np.unique(c["t"].to_numpy()[in_sub]))
            print(f"  {src}: scoring {mask.sum()} of {len(mask)} train pairs", flush=True)
            del c
        parts = []
        for ch in iter_feature_chunks(split, src, recs=(a, b), row_mask=mask):
            X = ch[cols].to_numpy(np.float32)
            parts.append(pd.DataFrame({"s1": ch["s1"].to_numpy(), "t": ch["t"].to_numpy(),
                                       "src": ch["src"].to_numpy(),
                                       "pA": models["A"].predict(X, num_threads=config.N_JOBS).astype(np.float32),
                                       "pB": models["B"].predict(X, num_threads=config.N_JOBS).astype(np.float32)}))
            del X
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        print(f"  saved -> {out}", flush=True)
        del b, parts


if __name__ == "__main__":
    run(sys.argv[1])
