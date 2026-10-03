"""Step 4: pairwise matcher (stage 1, LightGBM).

Train S1 records are split (by S1 entity, so nothing leaks) into
  A, B      two disjoint training folds (each model predicts the other fold out-of-fold)
  VAL_ES    early stopping
  VAL_EVAL  final held-out evaluation (never used for fitting anything)
Features are computed for the candidate pairs of these S1 records only.
"""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from build_feats import iter_feature_chunks, feature_columns, gt_keys, label, pair_key
from features import Records

N_A = N_B = 125_000
N_VAL_ES = 30_000
N_VAL_EVAL = 100_000

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=config.N_JOBS, seed=config.SEED, verbose=-1)


def splits():
    path = config.work("splits.json")
    if os.path.exists(path):
        with open(path) as f:
            return {k: np.array(v, dtype=np.int32) for k, v in json.load(f).items()}
    n = len(pd.read_parquet(config.norm_path("train", "source1"), columns=["entity_id"]))
    perm = np.random.default_rng(config.SEED).permutation(n).astype(np.int32)
    cuts = np.cumsum([N_A, N_B, N_VAL_ES, N_VAL_EVAL])
    sp = {"A": np.sort(perm[:cuts[0]]), "B": np.sort(perm[cuts[0]:cuts[1]]),
          "VAL_ES": np.sort(perm[cuts[1]:cuts[2]]), "VAL_EVAL": np.sort(perm[cuts[2]:cuts[3]])}
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in sp.items()}, f)
    return sp


def build_subset_features():
    sp = splits()
    rows = np.sort(np.concatenate(list(sp.values())))
    keys = np.fromiter(gt_keys(), dtype=np.int64)
    a = Records("train", "source1")
    for src in ("source2", "source3"):
        out = config.work("feats", f"train_{src}.parquet")
        if os.path.exists(out):
            print(f"  {out} exists, skipping")
            continue
        b = Records("train", src)
        parts = []
        for ch in iter_feature_chunks("train", src, s1_rows=rows, recs=(a, b)):
            ch["label"] = label(ch, keys)
            parts.append(ch)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(out, index=False)
        print(f"  saved {len(df)} rows, {int(df['label'].sum())} positives -> {out}", flush=True)
        del b, parts, df


def load_subset(names):
    sp = splits()
    rows = np.sort(np.concatenate([sp[n] for n in names]))
    dfs = []
    for src in ("source2", "source3"):
        d = pd.read_parquet(config.work("feats", f"train_{src}.parquet"))
        dfs.append(d[np.isin(d["s1"].to_numpy(), rows)])
    return pd.concat(dfs, ignore_index=True)


def train_models():
    es = load_subset(["VAL_ES"])
    cols = feature_columns(es)
    models = {}
    for fold in ("A", "B"):
        path = config.work("models", f"stage1_{fold}.txt")
        if os.path.exists(path):
            models[fold] = lgb.Booster(model_file=path)
            continue
        tr = load_subset([fold])
        print(f"  stage1 fold {fold}: {len(tr)} rows, {int(tr['label'].sum())} positives, {len(cols)} features",
              flush=True)
        dtr = lgb.Dataset(tr[cols].to_numpy(np.float32), tr["label"].to_numpy(), feature_name=cols,
                          free_raw_data=True)
        dva = lgb.Dataset(es[cols].to_numpy(np.float32), es["label"].to_numpy(), reference=dtr)
        del tr
        m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
        m.save_model(path, num_iteration=m.best_iteration)
        models[fold] = m
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print(imp.head(25).round(0).to_string(), flush=True)
    with open(config.work("models", "stage1_features.json"), "w") as f:
        json.dump(cols, f)
    return models


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("feats", "all"):
        build_subset_features()
    if step in ("train", "all"):
        train_models()
