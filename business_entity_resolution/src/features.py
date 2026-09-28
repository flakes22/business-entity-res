"""Pair features for the matching model.

Input pairs carry row indices into the normalised S1 / target parquet files plus
the blocking-stage columns (per-channel cosines, hit flags, ranks). Every
string similarity is computed with rapidfuzz.process.cpdist (aligned pairs,
multithreaded C++). Missing information is NaN ("unknown"), never 0.

No feature depends on the country label, so the model transfers to a country
unseen in training (France in the test set).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

import config

REC_COLS = ["name_full", "name_core", "name_dba", "name_skel", "is_domain", "name_indic",
            "addr", "state", "postal", "house", "city", "nums"]


class Records:
    """Per-source record attributes as numpy object arrays, indexed by parquet row."""

    def __init__(self, split, source):
        df = pd.read_parquet(config.norm_path(split, source), columns=REC_COLS)
        self.n = len(df)
        for c in REC_COLS:
            if df[c].dtype == object or str(df[c].dtype).startswith("str"):
                setattr(self, c, df[c].to_numpy(dtype=object))
            else:
                setattr(self, c, df[c].to_numpy())
        self.name_cat = np.array([s.replace(" ", "") for s in self.name_core], dtype=object)
        self.first_tok = np.array([s.split(" ", 1)[0] for s in self.name_core], dtype=object)
        self.n_name_tok = np.array([s.count(" ") + 1 if s else 0 for s in self.name_core], dtype=np.float32)
        self.n_addr_tok = np.array([s.count(" ") + 1 if s else 0 for s in self.addr], dtype=np.float32)
        self.name_len = np.array([len(s) for s in self.name_cat], dtype=np.float32)
        del df


def _sim(scorer, x, y, scale=0.01):
    out = cpdist(x, y, scorer=scorer, dtype=np.float32, workers=-1)
    if scale != 1:
        out *= np.float32(scale)
    return out


def _empty(x):
    return np.fromiter((not s for s in x), dtype=bool, count=len(x))


def _eq(x, y):
    out = (x == y).astype(np.float32)
    out[_empty(x) | _empty(y)] = np.nan
    return out


def _nan_if(v, mask):
    v[mask] = np.nan
    return v


def string_features(a, b, ia, ib):
    """a: Records of S1, b: Records of target, ia/ib aligned row arrays -> dict of feature arrays."""
    f = {}
    na, nb = a.name_core[ia], b.name_core[ib]
    f["name_ratio"] = _sim(fuzz.ratio, na, nb)
    f["name_partial"] = _sim(fuzz.partial_ratio, na, nb)
    f["name_tset"] = _sim(fuzz.token_set_ratio, na, nb)
    f["name_tsort"] = _sim(fuzz.token_sort_ratio, na, nb)
    f["name_jw"] = _sim(JaroWinkler.normalized_similarity, na, nb, 1)
    f["name_wratio"] = _sim(fuzz.WRatio, na, nb)
    ca, cb = a.name_cat[ia], b.name_cat[ib]
    f["namecat_ratio"] = _sim(fuzz.ratio, ca, cb)
    f["namecat_partial"] = _sim(fuzz.partial_ratio, ca, cb)
    ka, kb = a.name_skel[ia], b.name_skel[ib]
    f["skel_ratio"] = _sim(fuzz.ratio, ka, kb)
    f["skel_tset"] = _sim(fuzz.token_set_ratio, ka, kb)
    f["namefull_tset"] = _sim(fuzz.token_set_ratio, a.name_full[ia], b.name_full[ib])
    # dba / trade-name part vs the other side's core name (max over both directions)
    da, db = a.name_dba[ia], b.name_dba[ib]
    d1 = _nan_if(_sim(fuzz.token_set_ratio, da, nb), _empty(da))
    d2 = _nan_if(_sim(fuzz.token_set_ratio, na, db), _empty(db))
    f["dba_tset"] = np.fmax(d1, d2)
    f["first_tok_eq"] = _eq(a.first_tok[ia], b.first_tok[ib])
    f["first_tok_jw"] = _sim(JaroWinkler.normalized_similarity, a.first_tok[ia], b.first_tok[ib], 1)
    f["name_ntok_a"] = a.n_name_tok[ia]
    f["name_ntok_b"] = b.n_name_tok[ib]
    f["name_len_a"] = a.name_len[ia]
    f["name_len_diff"] = np.abs(a.name_len[ia] - b.name_len[ib])
    f["is_domain_b"] = b.is_domain[ib].astype(np.float32)
    f["indic_b"] = b.name_indic[ib].astype(np.float32)

    aa, ab = a.addr[ia], b.addr[ib]
    miss_b = _empty(ab)
    f["addr_empty_b"] = miss_b.astype(np.float32)
    f["addr_ratio"] = _nan_if(_sim(fuzz.ratio, aa, ab), miss_b)
    f["addr_tset"] = _nan_if(_sim(fuzz.token_set_ratio, aa, ab), miss_b)
    f["addr_tsort"] = _nan_if(_sim(fuzz.token_sort_ratio, aa, ab), miss_b)
    f["addr_partial"] = _nan_if(_sim(fuzz.partial_token_set_ratio, aa, ab), miss_b)
    f["addr_ntok_b"] = b.n_addr_tok[ib]
    f["addr_ntok_a"] = a.n_addr_tok[ia]
    ma, mb = a.nums[ia], b.nums[ib]
    nmiss = _empty(ma) | _empty(mb)
    f["nums_tset"] = _nan_if(_sim(fuzz.token_set_ratio, ma, mb), nmiss)
    f["nums_tsort"] = _nan_if(_sim(fuzz.token_sort_ratio, ma, mb), nmiss)
    f["nums_eq"] = _nan_if((ma == mb).astype(np.float32), nmiss)
    ha, hb = a.house[ia], b.house[ib]
    f["house_eq"] = _eq(ha, hb)
    f["house_lev"] = _nan_if(_sim(Levenshtein.normalized_similarity, ha, hb, 1), _empty(ha) | _empty(hb))
    f["postal_eq"] = _eq(a.postal[ia], b.postal[ib])
    f["state_eq"] = _eq(a.state[ia], b.state[ib])
    ya, yb = a.city[ia], b.city[ib]
    f["city_jw"] = _nan_if(_sim(JaroWinkler.normalized_similarity, ya, yb, 1), _empty(ya) | _empty(yb))
    # combined
    f["name_x_addr"] = f["name_tset"] * np.nan_to_num(f["addr_tset"], nan=0.5)
    return f


BLOCK_COLS = ["cos_n", "cos_k", "cos_g", "cos_a", "score", "hit", "rank"]


def context_features(p):
    """Group-context features from the blocking scores (cheap, computed over the whole candidate set).
    p: DataFrame with s1, src, t, score, cos_* ... Adds columns in place."""
    g = p.groupby(["s1", "src"], sort=False)["score"]
    p["grp_max"] = g.transform("max").astype(np.float32)
    p["grp_gap"] = (p["score"] - p["grp_max"]).astype(np.float32)
    p["grp_n"] = g.transform("size").astype(np.float32)
    p["grp_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    g2 = p.groupby(["s1", "src"], sort=False)["cos_a"]
    p["grp_rank_a"] = g2.rank(ascending=False, method="first").astype(np.float32)
    p["grp_gap_a"] = (p["cos_a"] - g2.transform("max")).astype(np.float32)
    g3 = p.groupby(["s1", "src"], sort=False)["cos_n"]
    p["grp_rank_n"] = g3.rank(ascending=False, method="first").astype(np.float32)
    p["grp_gap_n"] = (p["cos_n"] - g3.transform("max")).astype(np.float32)
    # reverse direction: how this S1 compares with every other S1 that retrieved the same target
    r = p.groupby(["src", "t"], sort=False)["score"]
    p["rev_max"] = r.transform("max").astype(np.float32)
    p["rev_gap"] = (p["score"] - p["rev_max"]).astype(np.float32)
    p["rev_n"] = r.transform("size").astype(np.float32)
    p["rev_rank"] = r.rank(ascending=False, method="first").astype(np.float32)
    # second best claimant margin (only meaningful for the best claimant)
    return p
