"""Stage-2 "difference" features: HOW two records differ, not only how much.

On held-out train entities the hard negatives are look-alike records built from a true one by
  * substituting a distinctive name word (Construction -> Distribution, Group -> Marlon) or the
    legal form (Private Limited -> LLP), and
  * replacing digits of the house number (22745 -> 22758, 2006 -> 2009),
whereas the noise on true matches mostly injects / drops generic words (Services, Center,
Partners) and drops or splits digits (1022 -> 022, 106 -> 1-06, 515 -> 515A). Edit-distance
features score both kinds of change alike, so we add:
  * name words of S1 missing from the target and extra words of the target (fuzzy token
    alignment: equal, same consonant skeleton, Jaro-Winkler >= 0.88, or prefix), their counts and
    a learned log-odds "how typical is this word as a missing / extra word in true matches"
    (cross-fitted: a pair never sees a table built from its own label);
  * legal-form agreement (canonical legal forms from the full name);
  * house-number change type: digits equal, substring (dropped digits), number of replaced digits
    at equal length, log numeric distance; extra / missing numbers in the whole address.
Computed only for the pairs stage 2 scores. Uses no country-specific rule.
"""
import math
import os
import re

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

import config
from textnorm import LEGAL, skeleton_token

_NONDIG = re.compile(r"\D")
_LEGAL_CANON = {"pvt": "private", "praivet": "private", "praaivet": "private", "praivaet": "private",
                "privet": "private", "praiveta": "private", "praivetaa": "private", "praiveet": "private",
                "praivaeta": "private", "praiveti": "private", "praivettt": "private",
                "ltd": "limited", "limitaed": "limited", "limiteda": "limited", "limitad": "limited",
                "limitd": "limited", "limitedaa": "limited", "incorporated": "inc", "incorporation": "inc",
                "corporation": "corp", "korporeshan": "corp", "company": "co", "kampanii": "co", "kampani": "co",
                "elelpi": "llp", "eleelpi": "llp", "elaelapii": "llp", "elelpii": "llp", "elaelapi": "llp",
                "elaelasii": "llc"}
NUM_COLS = ["n_miss", "n_extra", "miss_frac", "extra_frac", "legal_n_a", "legal_n_b", "legal_jacc",
            "h_digits_eq", "h_substr", "h_ndiff", "h_logdist", "nums_extra", "nums_miss"]
LO_COLS = ["miss_lo_min", "miss_lo_sum", "extra_lo_min", "extra_lo_sum"]
DIFF_COLS = NUM_COLS + LO_COLS


def _tok_match(x, ys, skx, skys):
    for y, sky in zip(ys, skys):
        if x == y or (skx and skx == sky):
            return True
        if len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)):
            return True
        if len(x) >= 4 and len(y) >= 4 and JaroWinkler.normalized_similarity(x, y) >= 0.88:
            return True
    return False


def _legal(full):
    return {_LEGAL_CANON.get(t, t) for t in full.split() if t in LEGAL}


def _digits_rel(x, y):
    """-> digits_eq, substring, n replaced digits (equal length), log10(1+|x-y|)"""
    if not x or not y:
        return (math.nan,) * 4
    dx, dy = _NONDIG.sub("", x), _NONDIG.sub("", y)
    if not dx or not dy:
        return (math.nan,) * 4
    eq = float(dx == dy)
    sub = float(dx != dy and (dx in dy or dy in dx))
    nd = float(sum(a != b for a, b in zip(dx, dy))) if len(dx) == len(dy) else math.nan
    dist = math.log10(1 + abs(int(dx[:15]) - int(dy[:15])))
    return eq, sub, nd, dist


def _nums_rel(na, nb):
    if not na or not nb:
        return math.nan, math.nan
    sa, sb = na.split(), nb.split()
    ok = lambda u, vs: any(u == v or (len(u) >= 2 and len(v) >= 2 and (u in v or v in u)) for v in vs)
    return float(sum(not ok(u, sa) for u in sb)), float(sum(not ok(u, sb) for u in sa))


def _one_chunk(args):
    ca, cb, fa, fb, ha, hb, na, nb = args
    n = len(ca)
    num = np.full((n, len(NUM_COLS)), np.nan, dtype=np.float32)
    miss_tok, extra_tok = [None] * n, [None] * n
    for i in range(n):
        ta, tb = ca[i].split(), cb[i].split()
        ka, kb = [skeleton_token(t) for t in ta], [skeleton_token(t) for t in tb]
        miss = [t for t, k in zip(ta, ka) if not _tok_match(t, tb, k, kb)]
        extra = [t for t, k in zip(tb, kb) if not _tok_match(t, ta, k, ka)]
        miss_tok[i], extra_tok[i] = " ".join(miss), " ".join(extra)
        la, lb = _legal(fa[i]), _legal(fb[i])
        jac = len(la & lb) / len(la | lb) if la and lb else math.nan
        num[i] = (len(miss), len(extra), len(miss) / max(len(ta), 1), len(extra) / max(len(tb), 1),
                  len(la), len(lb), jac, *_digits_rel(ha[i], hb[i]), *_nums_rel(na[i], nb[i]))
    return num, miss_tok, extra_tok


def compute(split, pairs, chunk=50_000):
    """pairs: DataFrame s1, src, t -> DataFrame (aligned) with NUM_COLS + 'miss_tok', 'extra_tok'."""
    cols = ["name_core", "name_full", "house", "nums"]
    rd = lambda src: {c: v.to_numpy(dtype=object) for c, v in
                      pd.read_parquet(config.norm_path(split, src), columns=cols).fillna("").items()}
    a = rd("source1")
    out_num = np.full((len(pairs), len(NUM_COLS)), np.nan, dtype=np.float32)
    miss_tok = np.empty(len(pairs), dtype=object)
    extra_tok = np.empty(len(pairs), dtype=object)
    for code, src in ((2, "source2"), (3, "source3")):
        idx = np.flatnonzero(pairs["src"].to_numpy() == code)
        if not len(idx):
            continue
        b = rd(src)
        ia, ib = pairs["s1"].to_numpy()[idx], pairs["t"].to_numpy()[idx]
        jobs = []
        for st in range(0, len(idx), chunk):
            x, y = ia[st:st + chunk], ib[st:st + chunk]
            jobs.append((a["name_core"][x], b["name_core"][y], a["name_full"][x], b["name_full"][y],
                         a["house"][x], b["house"][y], a["nums"][x], b["nums"][y]))
        with Pool(min(config.N_JOBS, 8)) as pool:
            res = pool.map(_one_chunk, jobs)
        pos = 0
        for num, mt, et in res:
            sl = idx[pos:pos + len(num)]
            out_num[sl], miss_tok[sl], extra_tok[sl] = num, mt, et
            pos += len(num)
        del b, jobs, res
        print(f"    diff features {split}/{src}: {len(idx)} pairs", flush=True)
    df = pd.DataFrame(out_num, columns=NUM_COLS)
    df["miss_tok"], df["extra_tok"] = miss_tok, extra_tok
    return df


# ---------------------------------------------------------------- learned word log-odds
def fit_logodds(tok_series, labels, min_n=3, shrink=10.0):
    """word -> log-odds of appearing (as a missing / extra word) in a TRUE vs a FALSE pair, shrunk to 0."""
    ct, cf = {}, {}
    for s, y in zip(tok_series, labels):
        if not s:
            continue
        d = ct if y else cf
        for w in set(s.split()):
            d[w] = d.get(w, 0) + 1
    nt, nf = max(sum(labels), 1), max(len(labels) - sum(labels), 1)
    base = math.log(nt / nf)
    table = {}
    for w in set(ct) | set(cf):
        a, b = ct.get(w, 0), cf.get(w, 0)
        if a + b < min_n:
            continue
        table[w] = (math.log((a + 0.5) / (b + 0.5)) - base) * (a + b) / (a + b + shrink)
    return table


def apply_logodds(tok_series, table):
    mn = np.full(len(tok_series), np.nan, dtype=np.float32)
    sm = np.zeros(len(tok_series), dtype=np.float32)
    for i, s in enumerate(tok_series):
        if s:
            v = [table.get(w, 0.0) for w in s.split()]
            mn[i], sm[i] = min(v), sum(v)
    return mn, sm


def add_logodds(df, tables):
    """tables: (miss_table, extra_table)."""
    df["miss_lo_min"], df["miss_lo_sum"] = apply_logodds(df["miss_tok"].to_numpy(), tables[0])
    df["extra_lo_min"], df["extra_lo_sum"] = apply_logodds(df["extra_tok"].to_numpy(), tables[1])
    return df
