"""Test-like simulation on train: drop S1 records so their S2/S3 matches become orphans.

Train has 1.22 unmatched ("orphan") S2/S3 records per S1 record, test about 2.30 (test pools
are ~24% larger per S1 in every country). On held-out train entities ~80% of stage-2 false
positives are orphans, so the test set is materially harder than train. Dropping a random
DROP_FRAC of the train S1 records that are not in any split (A, B, VAL_ES, VAL_EVAL) turns their
true matches into orphans and reproduces the test orphan rate. The candidate table loses the
dropped S1s' pairs and the reverse / competition context is recomputed. No test data is used.

    python orphan_sim.py build    # write the simulated stage-2 training tables (work/sim_table_*.parquet)
    python orphan_sim.py eval     # re-score VAL_EVAL under simulation with the current models
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from ctx import group_stats
from train_stage1 import splits

DROP_FRAC = 0.19
REV = ["rev_rank", "rev_gap", "rev_n", "rev_gap2"]


def dropped_s1(frac=DROP_FRAC, seed=7):
    sp = splits()
    n = len(pd.read_parquet(config.norm_path("train", "source1"), columns=["entity_id"]))
    used = np.zeros(n, dtype=bool)
    for v in sp.values():
        used[v] = True
    free = np.flatnonzero(~used)
    k = int(round(frac * n))
    return np.sort(np.random.default_rng(seed).choice(free, size=k, replace=False))


def sim_reverse_context(drop):
    """recomputed reverse context for every surviving candidate pair (both target sources)."""
    sp = splits()
    rows = np.concatenate([sp[k] for k in ("A", "B", "VAL_ES", "VAL_EVAL")])
    out = []
    for code, src in ((2, "source2"), (3, "source3")):
        c = pd.read_parquet(config.work("cand", f"train_{src}.parquet"), columns=["s1", "t", "score"])
        c = c[~np.isin(c["s1"].to_numpy(), drop)].reset_index(drop=True)
        t = c["t"].to_numpy()
        score = c["score"].to_numpy(np.float32)
        r, m, n, sec = group_stats(t, score)          # same as ctx.add_context, reverse part only
        keep = np.isin(c["s1"].to_numpy(), rows)      # features are only needed for the split rows
        out.append(pd.DataFrame({"s1": c["s1"].to_numpy()[keep], "t": t[keep], "src": np.int8(code),
                                 "rev_rank": r[keep], "rev_gap": (score - m)[keep], "rev_n": n[keep],
                                 "rev_gap2": np.where(r == 0, score - sec, score - m)[keep]}))
        del c, t, score, r, m, n, sec, keep
        gc.collect()
    return pd.concat(out, ignore_index=True)


def _key(df):
    from build_feats import pair_key
    return pair_key(df["src"].to_numpy(), df["s1"].to_numpy(), df["t"].to_numpy())


def _set_rev(sub, rev):
    """overwrite the reverse-context columns of sub in place (aligned by pair key)."""
    pos = pd.Index(_key(rev)).get_indexer(_key(sub))
    assert (pos >= 0).all()
    for k in REV:
        sub[k] = rev[k].to_numpy()[pos]


def _feats(src):
    return pd.read_parquet(config.work("feats", f"train_{src}.parquet"))


def sim_p1_subset(rev, chunk=1_000_000):
    """(key, p1) for the split rows, re-predicted by stage 1 with the recomputed reverse context."""
    with open(config.work("models", "stage1_features.json")) as f:
        cols = json.load(f)
    mA = lgb.Booster(model_file=config.work("models", "stage1_A.txt"))
    mB = lgb.Booster(model_file=config.work("models", "stage1_B.txt"))
    sp = splits()
    keys, ps = [], []
    for src in ("source2", "source3"):
        sub = _feats(src)
        _set_rev(sub, rev)
        p = np.empty(len(sub), np.float32)
        s1 = sub["s1"].to_numpy()
        inA, inB = np.isin(s1, sp["A"]), np.isin(s1, sp["B"])
        for st in range(0, len(sub), chunk):
            sl = slice(st, st + chunk)
            X = sub[cols].iloc[sl].to_numpy(np.float32)
            pA = mA.predict(X, num_threads=config.N_JOBS)
            pB = mB.predict(X, num_threads=config.N_JOBS)
            p[sl] = np.where(inA[sl], pB, np.where(inB[sl], pA, (pA + pB) / 2))
        keys.append(_key(sub))
        ps.append(p)
        del sub, X
        gc.collect()
    return np.concatenate(keys), np.concatenate(ps)


def lean_p1_context(d, mask):
    """stage2.p1_context computed over all pairs of d, returned only for rows where mask (memory-light)."""
    from stage2 import _group_sum
    p = d["p1"].to_numpy(np.float32)
    s1 = d["s1"].to_numpy().astype(np.int64)
    src = d["src"].to_numpy().astype(np.int64)
    t = d["t"].to_numpy().astype(np.int64)
    out = d[mask].reset_index(drop=True)
    del d
    gc.collect()
    pm = p[mask]
    k = s1 * 4 + src
    r, m, _, sec = group_stats(k, p)
    out["p1_rank_src"], out["p1_gap_src"] = r[mask], pm - m[mask]
    out["p1_gap2_src"] = np.where(r[mask] == 0, pm - np.nan_to_num(sec[mask], nan=0.0), pm - m[mask])
    del r, m, sec
    out["p1_sum_src"] = _group_sum(k, p)[mask]
    out["p1_n05_src"] = _group_sum(k, (p > 0.5).astype(np.float32))[mask]
    r, m, _, _ = group_stats(s1, p)
    out["p1_rank_all"], out["p1_gap_all"] = r[mask], pm - m[mask]
    del r, m
    out["p1_sum_all"] = _group_sum(s1, p)[mask]
    out["p1_n05_all"] = _group_sum(s1, (p > 0.5).astype(np.float32))[mask]
    del k, s1
    gc.collect()
    k = t * 4 + src
    del t, src
    r, m, n, sec = group_stats(k, p)
    rm, mm, secm = r[mask], m[mask], np.nan_to_num(sec[mask], nan=0.0)
    del r, m, sec
    out["p1_t_rank"] = rm
    out["p1_t_gap"] = pm - np.where(rm == 0, secm, mm)
    out["p1_t_gap2"] = pm - mm
    out["p1_t_n"] = n[mask]
    del n
    out["p1_t_sum"] = _group_sum(k, p)[mask]
    out["p1_t_n05"] = _group_sum(k, (p > 0.5).astype(np.float32))[mask]
    from stage2 import P1_CTX
    return out[["s1", "t", "src"] + P1_CTX]


def build_sim_table():
    """writes work/sim_table_{source}.parquet: stage-2 training table (layout of stage2.build_train_table)
    under the simulation."""
    from stage2 import p1_context, load_p1, P2_MIN
    drop = dropped_s1()
    print(f"  simulation: dropping {len(drop)} S1 records", flush=True)
    rev = sim_reverse_context(drop)
    key_sub, p_sub = sim_p1_subset(rev)
    rev.to_parquet(config.work("sim_rev.parquet"), index=False)
    del rev
    gc.collect()
    d = load_p1("train")
    d = d[~np.isin(d["s1"].to_numpy(), drop)].reset_index(drop=True)
    pos = pd.Index(key_sub).get_indexer(_key(d))
    d["p1"] = np.where(pos >= 0, p_sub[np.maximum(pos, 0)], d["p1"].to_numpy()).astype(np.float32)
    del pos, key_sub, p_sub
    gc.collect()
    sp = splits()
    rows = np.concatenate([sp[k] for k in ("A", "B", "VAL_ES", "VAL_EVAL")])
    d = lean_p1_context(d, np.isin(d["s1"].to_numpy(), rows))
    gc.collect()
    rev = pd.read_parquet(config.work("sim_rev.parquet"))
    for code, src in ((2, "source2"), (3, "source3")):
        sub = _feats(src)
        _set_rev(sub, rev)
        sub = sub.merge(d[d["src"].to_numpy() == code], on=["s1", "t", "src"], how="left")
        assert sub["p1"].notna().all()
        sub = sub[sub["p1"].to_numpy() >= P2_MIN]
        sub.to_parquet(config.work(f"sim_table_{src}.parquet"), index=False)
        print(f"  sim table {src}: {len(sub)} rows", flush=True)
        del sub
        gc.collect()


def load_sim_table(names):
    sp = splits()
    rows = np.concatenate([sp[k] for k in names])
    out = []
    for src in ("source2", "source3"):
        t = pd.read_parquet(config.work(f"sim_table_{src}.parquet"))
        out.append(t[np.isin(t["s1"].to_numpy(), rows)])
        del t
    return pd.concat(out, ignore_index=True)


def eval_current():
    from stage2 import load_stage2, evaluate_rules, ntrue_for, apply_rule
    from decide import macro_f05
    import os
    if not os.path.exists(config.work("sim_table_source3.parquet")):
        build_sim_table()
    model, cols = load_stage2()
    sp = splits()
    ev = load_sim_table(["VAL_EVAL"])
    ev["p2"] = model.predict(ev[cols].to_numpy(np.float32), num_threads=config.N_JOBS)
    ev = ev[["s1", "src", "t", "label", "p1", "p2", "p1_t_rank"]]
    ev.to_parquet(config.work("sim_val_eval_scored.parquet"), index=False)
    s1_rows = sp["VAL_EVAL"]
    ntrue = ntrue_for(s1_rows)
    print("  SIMULATED VAL_EVAL, current models:")
    evaluate_rules(ev, s1_rows, ntrue)
    with open(config.work("models", "decision.json")) as f:
        rule = json.load(f)["rule"]
    sel = apply_rule(ev, rule)
    print(f"  current rule {rule}: {macro_f05(s1_rows, ntrue, sel['s1'].to_numpy(), sel['label'].to_numpy())[0]:.5f}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if step == "build":
        build_sim_table()
    elif step == "eval":
        eval_current()
