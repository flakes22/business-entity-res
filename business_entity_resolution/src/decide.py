"""Decision rules (probabilities -> matched sets) and the challenge metric.

macro F0.5: per S1 entity F = 1.25*TP / (0.25*n_true + n_pred); an entity with no
true matches scores 1 if nothing is predicted, else 0. Averaged over ALL entities.
"""
import numpy as np
import pandas as pd


def macro_f05(s1_eval, ntrue, sel_s1, sel_label):
    """s1_eval: array of all evaluated S1 rows; ntrue: array (aligned) of their true-match counts
    (including matches blocking missed); sel_s1/sel_label: selected pairs and their 0/1 labels."""
    idx = pd.Index(s1_eval)
    pos = idx.get_indexer(sel_s1)
    assert (pos >= 0).all()
    n_pred = np.bincount(pos, minlength=len(idx)).astype(np.float64)
    tp = np.bincount(pos, weights=sel_label, minlength=len(idx))
    denom = 0.25 * ntrue + n_pred
    f = np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-12), 1.0)
    return float(f.mean()), f


def select_threshold(df, thr, prob="p"):
    return df[df[prob].to_numpy() >= thr]


def select_expected_f(df, prob="p", miss_rate=0.03, min_p=0.0, alpha=1.0):
    """Per S1: sort candidates by p, pick the top-j that maximises the approximate expected F0.5
         E[F](j) ~= 1.25 * sum_{i<=j} p_i / (0.25 * E[n_true] + j),   E[n_true] = sum p / (1 - miss_rate)
       versus j = 0 whose expected score is P(no true match) ~= prod(1 - p_i) (times no-miss prob).
       alpha scales the j=0 option (tuned on validation)."""
    d = df[df[prob].to_numpy() > min_p][["s1", prob]].copy()
    d = d.sort_values(["s1", prob], ascending=[True, False], kind="mergesort")
    p = d[prob].to_numpy(np.float64)
    s1 = d["s1"].to_numpy()
    starts = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])
    sizes = np.diff(np.r_[starts, len(s1)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    j = np.arange(len(s1)) - starts[gid] + 1
    csum = np.cumsum(p)
    grp_cum = csum - np.r_[0, csum[starts[1:] - 1]][gid]
    tot = np.add.reduceat(p, starts)
    en = tot / (1 - miss_rate)
    ef = 1.25 * grp_cum / (0.25 * en[gid] + j)
    logq = np.add.reduceat(np.log1p(-np.minimum(p, 1 - 1e-9)), starts)
    f0 = alpha * np.exp(logq) * (1 - miss_rate) ** np.maximum(en * 0 + 1, 1)
    best_ef = np.maximum.reduceat(ef, starts)
    # best j per group = first index reaching the group max
    is_best = ef >= best_ef[gid] - 1e-12
    first_best = np.full(len(starts), np.iinfo(np.int64).max)
    np.minimum.at(first_best, gid[is_best], j[is_best])
    take_j = np.where(best_ef > f0, first_best, 0)
    keep = j <= take_j[gid]
    return df.loc[d.index[keep]]


def exclusive(df, prob="p"):
    """keep, for every (src, target), only the highest-probability S1 claimant."""
    d = df.sort_values(prob, ascending=False, kind="mergesort")
    return d.drop_duplicates(["src", "t"])
