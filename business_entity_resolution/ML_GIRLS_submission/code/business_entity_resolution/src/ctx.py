"""Memory-light group statistics on large pair tables (numpy, sort based).

Used for the candidate-context features: how a pair ranks among all candidates
of its S1 record (forward) and among all S1 records that retrieved the same
target record (reverse).
"""
import numpy as np


def group_stats(keys, vals):
    """For each element: (rank desc within its key group [0 = best], group max, group size, second best).
    keys: int array, vals: float array (same length)."""
    n = len(keys)
    order = np.lexsort((-vals, keys))
    k_sorted = keys[order]
    v_sorted = vals[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    grp_id = np.repeat(np.arange(len(starts)), sizes)
    rank_sorted = np.arange(n) - starts[grp_id]
    gmax = v_sorted[starts]
    second = np.where(sizes > 1, v_sorted[np.minimum(starts + 1, n - 1)], np.nan)
    rank = np.empty(n, dtype=np.float32)
    rank[order] = rank_sorted
    out_max = np.empty(n, dtype=np.float32)
    out_max[order] = gmax[grp_id]
    out_size = np.empty(n, dtype=np.float32)
    out_size[order] = sizes[grp_id]
    out_second = np.empty(n, dtype=np.float32)
    out_second[order] = second[grp_id]
    return rank, out_max, out_size, out_second


def add_context(df, key_s1="s1", key_t="t"):
    """Adds forward (per S1) and reverse (per target) context columns to a pair table of ONE target source.
    Needs columns score, cos_a, cos_n."""
    s1 = df[key_s1].to_numpy()
    t = df[key_t].to_numpy()
    score = df["score"].to_numpy(np.float32)
    r, m, n, sec = group_stats(s1, score)
    df["grp_rank"], df["grp_gap"], df["grp_n"] = r, score - m, n
    df["grp_gap2"] = np.where(r == 0, score - sec, score - m)          # margin over the runner-up
    for c in ("cos_a", "cos_n"):
        v = df[c].to_numpy(np.float32)
        r, m, _, _ = group_stats(s1, v)
        df[f"grp_rank_{c[-1]}"], df[f"grp_gap_{c[-1]}"] = r, v - m
    r, m, n, sec = group_stats(t, score)
    df["rev_rank"], df["rev_gap"], df["rev_n"] = r, score - m, n
    df["rev_gap2"] = np.where(r == 0, score - sec, score - m)
    return df
