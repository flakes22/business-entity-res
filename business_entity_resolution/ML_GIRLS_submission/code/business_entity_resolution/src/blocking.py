"""Step 2: candidate generation (blocking).

For every Source-1 record and each target source (S2, S3) separately, within the
same country label, we retrieve the top-K target records by an IDF-weighted
sparse similarity over three token channels:

  n_  name tokens (legal words / honorifics removed, Indic scripts romanised)
  k_  consonant-skeleton of the name tokens (robust to transliteration + vowel typos)
  a_  canonical address tokens (abbreviations unified, unit designators dropped,
      state names -> codes, numbers split out)
  g_  character 4-grams of the name core (typos, concatenated domain names)

Each channel is L2-normalised separately and weighted, so the score is a weighted
sum of per-channel cosines. Over-common tokens (df above a cap) are removed from the
target side of the product only (they would dominate cost and carry no signal), the
normalisation still accounts for them. The top-K search is sparse_dot_topn (MIT,
multithreaded C++).

Output: work/cand/<split>_<target>.parquet with columns
  s1 (row index into the S1 parquet), t (row index into the target parquet), score, rank
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

import config
from ctx import group_stats

CHANNEL_W = {"n": 0.45, "k": 0.25, "g": 0.3, "p": 0.3, "a": 0.8, "x": 0.6, "m": 0.5}
DF_CAP_FRAC = 0.002          # target-side token df cap (fraction of that country's target pool)
DF_CAP_MIN = 200
GRAM = 4
MAX_NUMS = 3                 # numbers used for number|word composite tokens
MAX_NAME_TOK = 6


def _tokens(df):
    """-> list of token lists per record, each token prefixed by its channel:
    n name token, k name skeleton token, g name char 4-gram, p sorted name-token pair,
    a address token, x number|address-word composite, m name-token|address-word composite.
    Composite tokens keep discriminative combinations of individually common tokens
    (house number + locality, common name + town) that the df cap would otherwise remove."""
    core = df["name_core"].values
    skel = df["name_skel"].values
    addr = df["addr"].values
    docs = []
    for c, k, a in zip(core, skel, addr):
        nt = [t for t in c.split() if len(t) >= 2 or t.isdigit()]
        toks = ["n_" + t for t in nt]
        toks += ["k_" + t for t in k.split() if len(t) >= 2]
        s = c.replace(" ", "")
        if len(s) >= GRAM:
            toks += ["g_" + s[i:i + GRAM] for i in range(len(s) - GRAM + 1)]
        nt6 = sorted(set(nt[:MAX_NAME_TOK]))
        toks += ["p_" + x + "|" + y for i, x in enumerate(nt6) for y in nt6[i + 1:]]
        at = a.split()
        toks += ["a_" + t for t in at]
        words = [t for t in dict.fromkeys(at) if len(t) >= 3 and t.isalpha()]
        nums = [t for t in dict.fromkeys(at) if t.isdigit()][:MAX_NUMS]
        toks += ["x_" + n + "|" + w for n in nums for w in words]
        toks += ["m_" + t + "|" + w for t in nt[:2] if len(t) >= 3 for w in words if len(w) >= 4]
        docs.append(toks)
    return docs


def _build_matrix(docs, vocab, grow):
    """binary doc-term CSR; unseen tokens are added to vocab when grow=True."""
    indptr = [0]
    indices = []
    for toks in docs:
        seen = set()
        for t in toks:
            j = vocab.get(t)
            if j is None:
                if not grow:
                    continue
                j = vocab[t] = len(vocab)
            if j not in seen:
                seen.add(j)
                indices.append(j)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    return indptr, np.asarray(indices, dtype=np.int32), data


def _weighted(X, idf, chan_of_col):
    """tf(binary)*idf, L2-normalised per channel, scaled by the channel weight."""
    X = X.tocsr().astype(np.float32)
    X.data = X.data * idf[X.indices]
    out = None
    for ci, w in enumerate(CHANNEL_W.values()):
        mask = (chan_of_col == ci).astype(np.float32)
        Xc = X @ sp.diags(mask)
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        Xc = sp.diags((w / norms).astype(np.float32)) @ Xc
        out = Xc if out is None else out + Xc
    out = out.tocsr()
    out.eliminate_zeros()
    return out


SEARCHES = {"name": ("n", "k", "g", "p"), "addr": ("a", "x"), "comb": ("n", "k", "g", "p", "a", "x", "m")}
K_SEARCH = {"name": 10, "addr": 12, "comb": 20}
K_REVERSE = 5
# pruning rule applied to the union of searches (ranks are within one S1 record and one target source):
# keep if combined-score rank < 15, or address rank < 5, or name rank < 5, or name|address rank < 3,
# or the S1 record is among the target record's 3 best S1 records (reverse search).
PRUNE = {"score": 15, "addr": 5, "name": 5, "m": 3, "rev": 3}
S1_BATCH = 200_000


def _pair_channel_dots(Sw, Tw, s_pos, t_pos, chan_of_col, step=1_000_000):
    """exact per-channel cosine contributions for aligned pairs -> (n_pairs, n_channels) float32"""
    nch = len(CHANNEL_W)
    out = np.zeros((len(s_pos), nch), dtype=np.float32)
    for st in range(0, len(s_pos), step):
        P = Sw[s_pos[st:st + step]].multiply(Tw[t_pos[st:st + step]]).tocsr()
        rows = np.repeat(np.arange(P.shape[0]), np.diff(P.indptr))
        acc = np.bincount(rows * nch + chan_of_col[P.indices], weights=P.data, minlength=P.shape[0] * nch)
        out[st:st + step] = acc.reshape(-1, nch)
        del P
    w = np.array(list(CHANNEL_W.values()), dtype=np.float32)
    return out / w            # undo the channel weight -> plain cosine per channel


def block_country(s1_c, t_c, n_threads=config.N_JOBS):
    """s1_c, t_c: normalised frames for ONE country.
    -> DataFrame(s, t, cos_n, cos_k, cos_g, cos_a, hit_name, hit_addr, hit_comb) with positions within the frames."""
    t0 = time.time()
    vocab = {}
    ti, tj, td = _build_matrix(_tokens(t_c), vocab, grow=True)
    si, sj, sd = _build_matrix(_tokens(s1_c), vocab, grow=True)
    V = len(vocab)
    chan_names = list(CHANNEL_W)
    chan_of_col = np.empty(V, dtype=np.int8)
    for tok, j in vocab.items():
        chan_of_col[j] = chan_names.index(tok[0])
    del vocab
    T = sp.csr_matrix((td, tj, ti), shape=(len(t_c), V))
    S = sp.csr_matrix((sd, sj, si), shape=(len(s1_c), V))
    del ti, tj, td, si, sj, sd
    df = np.bincount(T.indices, minlength=V).astype(np.float64) + np.bincount(S.indices, minlength=V)
    n_docs = T.shape[0] + S.shape[0]
    idf = (np.log((n_docs + 1) / (df + 1)) + 1).astype(np.float32)
    Tw = _weighted(T, idf, chan_of_col)
    Sw = _weighted(S, idf, chan_of_col)
    del T, S
    gc.collect()
    t_df = np.bincount(Tw.indices, minlength=V)
    cap = max(DF_CAP_MIN, int(DF_CAP_FRAC * Tw.shape[0]))
    keep_col = (t_df <= cap)
    TwT = (Tw @ sp.diags(keep_col.astype(np.float32))).tocsr()
    TwT.eliminate_zeros()
    TwT = TwT.T.tocsr()
    print(f"      vocab {V}, dropped {(~keep_col).sum()} common tokens (cap {cap}); build {time.time() - t0:.0f}s",
          flush=True)

    t1 = time.time()
    # reverse search first (needs every S1 record): for each target, its best S1 records.
    # Each S2/S3 record belongs to at most one S1 entity, so this recovers matches that are crowded
    # out of an S1's own top-K by look-alikes (same name / same building).
    s_df = np.bincount(Sw.indices, minlength=V)
    cap_s = max(DF_CAP_MIN, int(DF_CAP_FRAC * Sw.shape[0]))
    SwT = (Sw @ sp.diags((s_df <= cap_s).astype(np.float32))).tocsr()
    SwT.eliminate_zeros()
    SwT = SwT.T.tocsr()
    rev_s, rev_t, rev_r = [], [], []
    for start in range(0, Tw.shape[0], 500_000):
        C = sp_matmul_topn(Tw[start:start + 500_000], SwT, top_n=K_REVERSE, threshold=0.05, sort=True,
                           n_threads=n_threads).tocsr()
        cnt = np.diff(C.indptr)
        rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), cnt) + start
        rev_s.append(C.indices.astype(np.int32))
        rev_t.append(rows.astype(np.int32))
        rev_r.append((np.arange(len(C.indices)) - np.repeat(C.indptr[:-1], cnt)).astype(np.int8))
        del C
    del SwT
    rev = pd.DataFrame({"s": np.concatenate(rev_s), "t": np.concatenate(rev_t), "rrank": np.concatenate(rev_r)})
    del rev_s, rev_t, rev_r
    rev = rev[rev["rrank"] < PRUNE["rev"]].sort_values("s", kind="mergesort").reset_index(drop=True)
    print(f"      reverse search {time.time() - t1:.0f}s -> {len(rev)} pairs", flush=True)

    t1 = time.time()
    masks = {sname: sp.diags(np.isin(chan_of_col, [chan_names.index(c) for c in chans]).astype(np.float32))
             for sname, chans in SEARCHES.items()}
    rev_s_arr = rev["s"].to_numpy()
    out = []
    n_before = 0
    for start in range(0, Sw.shape[0], S1_BATCH):
        stop = min(start + S1_BATCH, Sw.shape[0])
        Sb = Sw[start:stop]
        found = []
        for bit, (sname, M) in enumerate(masks.items()):
            Q = (Sb @ M).tocsr()
            Q.eliminate_zeros()
            C = sp_matmul_topn(Q, TwT, top_n=K_SEARCH[sname], threshold=0.02, n_threads=n_threads).tocsr()
            rows = np.repeat(np.arange(C.shape[0], dtype=np.int64), np.diff(C.indptr)) + start
            found.append(pd.DataFrame({"s": rows.astype(np.int32), "t": C.indices.astype(np.int32),
                                       "hit": np.int8(1 << bit), "rrank": np.int8(99)}))
            del Q, C
        lo, hi = np.searchsorted(rev_s_arr, [start, stop])
        rb = rev.iloc[lo:hi]
        found.append(pd.DataFrame({"s": rb["s"].values, "t": rb["t"].values, "hit": np.int8(8),
                                   "rrank": rb["rrank"].values}))
        pairs = pd.concat(found, ignore_index=True)
        del found
        pairs = pairs.groupby(["s", "t"], sort=False).agg(hit=("hit", "sum"), rrank=("rrank", "min")).reset_index()
        n_before += len(pairs)
        dots = _pair_channel_dots(Sw, Tw, pairs["s"].values, pairs["t"].values, chan_of_col)
        for i, c in enumerate(chan_names):
            pairs["cos_" + c] = dots[:, i]
        del dots
        pairs["score"] = sum(pairs["cos_" + c] * w for c, w in CHANNEL_W.items()).astype(np.float32)
        sv = pairs["s"].to_numpy()
        r_score = group_stats(sv, pairs["score"].to_numpy(np.float32))[0]
        r_a = group_stats(sv, (pairs["cos_a"] + pairs["cos_x"]).to_numpy(np.float32))[0]
        r_n = group_stats(sv, (pairs["cos_n"] + pairs["cos_g"] + pairs["cos_k"] + pairs["cos_p"]).to_numpy(np.float32))[0]
        r_m = group_stats(sv, pairs["cos_m"].to_numpy(np.float32))[0]
        keep = ((r_score < PRUNE["score"]) | (r_a < PRUNE["addr"]) | (r_n < PRUNE["name"]) | (r_m < PRUNE["m"])
                | (pairs["rrank"].to_numpy() < PRUNE["rev"]))
        out.append(pairs[keep].reset_index(drop=True))
        del pairs
        gc.collect()
    res = pd.concat(out, ignore_index=True)
    print(f"      forward searches + cosines {time.time() - t1:.0f}s -> union {n_before} "
          f"({n_before / len(s1_c):.1f}/S1), kept {len(res)} ({len(res) / len(s1_c):.1f}/S1)", flush=True)
    return res


def run(split, s1_limit=None, tag=""):
    cols = ["country", "name_core", "name_skel", "addr"]
    s1 = pd.read_parquet(config.norm_path(split, "source1"), columns=cols)
    s1["row"] = np.arange(len(s1), dtype=np.int32)
    if s1_limit:
        s1 = s1.sample(n=s1_limit, random_state=config.SEED).sort_values("row")
    for tgt in ("source2", "source3"):
        out_path = config.work("cand", f"{split}_{tgt}{tag}.parquet")
        if os.path.exists(out_path):
            print(f"  {out_path} exists, skipping")
            continue
        t = pd.read_parquet(config.norm_path(split, tgt), columns=cols)
        t["row"] = np.arange(len(t), dtype=np.int32)
        frames = []
        # every country label present in S1 (open set: whatever labels the files contain)
        for country in sorted(set(s1["country"])):
            s1_c = s1[s1["country"] == country]
            t_c = t[t["country"] == country]
            if len(t_c) == 0:
                print(f"  [{split}] {country}: no {tgt} records with this country label")
                continue
            print(f"  [{split}] S1 -> {tgt} | {country}: {len(s1_c)} x {len(t_c)}", flush=True)
            pr = block_country(s1_c, t_c)
            pr["s"] = s1_c["row"].values[pr["s"].values]
            pr["t"] = t_c["row"].values[pr["t"].values]
            frames.append(pr.rename(columns={"s": "s1"}))
            del pr
            gc.collect()
        cand = pd.concat(frames, ignore_index=True)
        del frames
        cand = cand.sort_values(["s1", "score"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
        cand["rank"] = cand.groupby("s1").cumcount().astype(np.int16)
        cand.to_parquet(out_path, index=False)
        print(f"  saved {len(cand)} pairs -> {out_path}", flush=True)
        del t, cand
        gc.collect()


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    run(split, s1_limit=limit, tag=tag)
