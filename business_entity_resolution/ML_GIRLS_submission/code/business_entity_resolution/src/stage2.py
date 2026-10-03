"""Step 6: stage-2 re-scoring with probability context + decision rule.

Stage-1 scores every pair independently. Stage 2 adds, for each pair, how its
stage-1 probability p1 compares with
  * the other candidates of the same S1 record (per target source and overall), and
  * the other S1 records competing for the same S2/S3 record (each S2/S3 record belongs
    to at most one S1 entity in the ground truth),
and re-scores with a second LightGBM. The final per-S1 decision is then tuned for
macro F0.5 on held-out S1 records.
"""
import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from build_feats import feature_columns, gt_keys, label
from ctx import group_stats
from train_stage1 import splits, load_subset, PARAMS

P2_MIN = 0.02     # stage 2 only re-scores pairs with p1 >= P2_MIN (the rest are never selected)
P1_CTX = ["p1", "p1_rank_src", "p1_gap_src", "p1_gap2_src", "p1_sum_src", "p1_n05_src",
          "p1_rank_all", "p1_gap_all", "p1_sum_all", "p1_n05_all",
          "p1_t_rank", "p1_t_gap", "p1_t_gap2", "p1_t_sum", "p1_t_n05", "p1_t_n"]


def _group_sum(keys, vals):
    u, inv = np.unique(keys, return_inverse=True)
    return np.bincount(inv, weights=vals, minlength=len(u))[inv].astype(np.float32)


def load_p1(split):
    """p1 for every candidate pair of the split. Train: out-of-fold for folds A/B, mean elsewhere."""
    d = pd.concat([pd.read_parquet(config.work("p1", f"{split}_{s}.parquet")) for s in ("source2", "source3")],
                  ignore_index=True)
    p = (d["pA"].to_numpy() + d["pB"].to_numpy()) / 2
    if split == "train":
        sp = splits()
        inA = np.isin(d["s1"].to_numpy(), sp["A"])
        inB = np.isin(d["s1"].to_numpy(), sp["B"])
        p = np.where(inA, d["pB"].to_numpy(), np.where(inB, d["pA"].to_numpy(), p))
    d["p1"] = p.astype(np.float32)
    return d.drop(columns=["pA", "pB"])


def p1_context(d):
    """d: s1, t, src, p1 for ALL pairs of a split -> adds P1_CTX columns."""
    p = d["p1"].to_numpy(np.float32)
    s1 = d["s1"].to_numpy().astype(np.int64)
    src = d["src"].to_numpy().astype(np.int64)
    t = d["t"].to_numpy().astype(np.int64)
    k_src = s1 * 4 + src
    r, m, _, sec = group_stats(k_src, p)
    d["p1_rank_src"], d["p1_gap_src"] = r, p - m
    d["p1_gap2_src"] = np.where(r == 0, p - np.nan_to_num(sec, nan=0.0), p - m)
    d["p1_sum_src"] = _group_sum(k_src, p)
    d["p1_n05_src"] = _group_sum(k_src, (p > 0.5).astype(np.float32))
    r, m, _, _ = group_stats(s1, p)
    d["p1_rank_all"], d["p1_gap_all"] = r, p - m
    d["p1_sum_all"] = _group_sum(s1, p)
    d["p1_n05_all"] = _group_sum(s1, (p > 0.5).astype(np.float32))
    k_t = t * 4 + src
    r, m, n, sec = group_stats(k_t, p)
    d["p1_t_rank"] = r
    # margin over the strongest OTHER claimant of this target (0 if unopposed)
    other = np.where(r == 0, np.nan_to_num(sec, nan=0.0), m)
    d["p1_t_gap"] = p - other
    d["p1_t_gap2"] = p - m
    d["p1_t_sum"] = _group_sum(k_t, p)
    d["p1_t_n05"] = _group_sum(k_t, (p > 0.5).astype(np.float32))
    d["p1_t_n"] = n
    return d


def build_train_table():
    """stage-1 features of the subset rows + p1 context (computed over ALL train pairs)."""
    d = p1_context(load_p1("train"))
    sp = splits()
    rows = np.concatenate([sp[k] for k in ("A", "B", "VAL_ES", "VAL_EVAL")])
    d = d[np.isin(d["s1"].to_numpy(), rows)].reset_index(drop=True)
    gc.collect()
    sub = load_subset(["A", "B", "VAL_ES", "VAL_EVAL"])
    sub = sub.merge(d, on=["s1", "t", "src"], how="left")
    del d
    gc.collect()
    assert sub["p1"].notna().all()
    sub = sub[sub["p1"].to_numpy() >= P2_MIN].reset_index(drop=True)
    return sub


def train_stage2(tab):
    sp = splits()
    with open(config.work("models", "stage1_features.json")) as f:
        s1cols = json.load(f)
    cols = s1cols + P1_CTX
    tr = tab[np.isin(tab["s1"].to_numpy(), np.concatenate([sp["A"], sp["B"]]))]
    es = tab[np.isin(tab["s1"].to_numpy(), sp["VAL_ES"])]
    dtr = lgb.Dataset(tr[cols].to_numpy(np.float32), tr["label"].to_numpy(), feature_name=cols)
    dva = lgb.Dataset(es[cols].to_numpy(np.float32), es["label"].to_numpy(), reference=dtr)
    params = dict(PARAMS, learning_rate=0.05, num_leaves=63)
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    m.save_model(config.work("models", "stage2.txt"), num_iteration=m.best_iteration)
    with open(config.work("models", "stage2_features.json"), "w") as f:
        json.dump(cols, f)
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print(imp.head(25).round(0).to_string(), flush=True)
    return m, cols


def load_stage2():
    with open(config.work("models", "stage2_features.json")) as f:
        cols = json.load(f)
    return lgb.Booster(model_file=config.work("models", "stage2.txt")), cols


def ntrue_for(s1_rows):
    gt = pd.read_parquet(config.work("gt_pairs.parquet"))
    cnt = gt.groupby("s1").size()
    return cnt.reindex(s1_rows, fill_value=0).to_numpy().astype(np.float64)


def evaluate_rules(ev, s1_rows, ntrue, verbose=True):
    """ev: DataFrame(s1, src, t, label, p1, p2, p1_t_rank). Returns list of (score, rule dict)."""
    from decide import macro_f05, select_expected_f
    res = []
    for prob in ("p1", "p2"):
        for thr in np.round(np.arange(0.30, 0.96, 0.02), 2):
            sel = ev[ev[prob].to_numpy() >= thr]
            res.append((macro_f05(s1_rows, ntrue, sel["s1"].to_numpy(), sel["label"].to_numpy())[0],
                        {"rule": "threshold", "prob": prob, "thr": float(thr)}))
            sel = sel[sel["p1_t_rank"].to_numpy() == 0]
            res.append((macro_f05(s1_rows, ntrue, sel["s1"].to_numpy(), sel["label"].to_numpy())[0],
                        {"rule": "threshold+excl", "prob": prob, "thr": float(thr)}))
    for miss in (0.0, 0.02, 0.05):
        for alpha in (0.8, 1.0, 1.2, 1.5, 2.0):
            for min_p in (0.05, 0.2, 0.4):
                sel = select_expected_f(ev.rename(columns={"p2": "p"}), prob="p", miss_rate=miss, alpha=alpha,
                                        min_p=min_p)
                res.append((macro_f05(s1_rows, ntrue, sel["s1"].to_numpy(), sel["label"].to_numpy())[0],
                            {"rule": "expected_f", "prob": "p2", "miss_rate": miss, "alpha": alpha, "min_p": min_p}))
    res.sort(key=lambda x: -x[0])
    if verbose:
        best_by_rule = {}
        for sc, r in res:
            best_by_rule.setdefault((r["rule"], r["prob"]), (sc, r))
        for k, (sc, r) in best_by_rule.items():
            print(f"    {sc:.5f}  {r}")
    return res


def apply_rule(df, rule):
    from decide import select_expected_f
    if rule["rule"] == "expected_f":
        return select_expected_f(df.rename(columns={"p2": "p"}), prob="p", miss_rate=rule["miss_rate"],
                                 alpha=rule["alpha"], min_p=rule["min_p"])
    sel = df[df[rule["prob"]].to_numpy() >= rule["thr"]]
    if rule["rule"] == "threshold+excl":
        sel = sel[sel["p1_t_rank"].to_numpy() == 0]
    return sel


def fit_and_evaluate():
    from decide import macro_f05
    tab = build_train_table()
    model, cols = train_stage2(tab)
    sp = splits()
    ev = tab[np.isin(tab["s1"].to_numpy(), sp["VAL_EVAL"])].copy()
    ev["p2"] = model.predict(ev[cols].to_numpy(np.float32), num_threads=config.N_JOBS)
    ev = ev[["s1", "src", "t", "label", "p1", "p2", "p1_t_rank"]]
    ev.to_parquet(config.work("val_eval_scored.parquet"), index=False)
    s1_rows = sp["VAL_EVAL"]
    ntrue = ntrue_for(s1_rows)
    print(f"\nVAL_EVAL: {len(s1_rows)} S1, singleton share {np.mean(ntrue == 0):.4f}, "
          f"true pairs {int(ntrue.sum())}, reachable after blocking {int(ev['label'].sum())} "
          f"({ev['label'].sum() / ntrue.sum():.4f})")
    # tune on one half, report on the other (honest estimate), then tune on all for the final rule
    half = np.zeros(len(s1_rows), dtype=bool)
    half[np.random.default_rng(1).permutation(len(s1_rows))[: len(s1_rows) // 2]] = True
    h1, h2 = s1_rows[half], s1_rows[~half]
    ev1, ev2 = ev[np.isin(ev["s1"].to_numpy(), h1)], ev[np.isin(ev["s1"].to_numpy(), h2)]
    print("  rules tuned on half 1:")
    best1 = evaluate_rules(ev1, h1, ntrue[half])[0]
    sel = apply_rule(ev2, best1[1])
    honest = macro_f05(h2, ntrue[~half], sel["s1"].to_numpy(), sel["label"].to_numpy())[0]
    print(f"  -> best on half 1: {best1[0]:.5f} {best1[1]}\n  -> SAME rule on half 2 (unseen): {honest:.5f}")
    print("  rules on all of VAL_EVAL:")
    best = evaluate_rules(ev, s1_rows, ntrue)[0]
    print(f"FINAL rule: {best[1]}  (VAL_EVAL macro F0.5 {best[0]:.5f}, honest half-split estimate {honest:.5f})")
    with open(config.work("models", "decision.json"), "w") as f:
        json.dump({"rule": best[1], "val_macro_f05": best[0], "honest_estimate": honest}, f, indent=1)


def score_split(split):
    """stage-2 probabilities for every candidate pair of a split (streams stage-1 features again)."""
    from build_feats import iter_feature_chunks
    from features import Records
    model, cols = load_stage2()
    d = p1_context(load_p1(split))
    src_codes = d["src"].to_numpy()
    ctx_all = {code: (d["s1"].to_numpy()[src_codes == code], d["t"].to_numpy()[src_codes == code],
                      d.loc[src_codes == code, P1_CTX].to_numpy(np.float32)) for code in (2, 3)}
    del d, src_codes
    gc.collect()
    a = Records(split, "source1")
    out = []
    for src in ("source2", "source3"):
        b = Records(split, src)
        cs1, ct, cmat = ctx_all.pop(2 if src == "source2" else 3)
        keep = cmat[:, P1_CTX.index("p1")] >= P2_MIN
        cs1, ct, cmat = cs1[keep], ct[keep], cmat[keep]
        n_src = len(cs1)
        print(f"  {src}: stage 2 on {n_src} of {len(keep)} pairs (p1 >= {P2_MIN})", flush=True)
        pos = 0
        for ch in iter_feature_chunks(split, src, recs=(a, b), row_mask=keep):
            sl = slice(pos, pos + len(ch))
            assert (cs1[sl] == ch["s1"].to_numpy()).all() and (ct[sl] == ch["t"].to_numpy()).all()
            for i, k in enumerate(P1_CTX):
                ch[k] = cmat[sl, i]
            p2 = model.predict(ch[cols].to_numpy(np.float32), num_threads=config.N_JOBS).astype(np.float32)
            out.append(pd.DataFrame({"s1": ch["s1"].to_numpy(), "src": ch["src"].to_numpy(), "t": ch["t"].to_numpy(),
                                     "p1": ch["p1"].to_numpy(), "p2": p2, "p1_t_rank": ch["p1_t_rank"].to_numpy()}))
            pos += len(ch)
        assert pos == n_src
        del b, cs1, ct, cmat
        gc.collect()
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(config.work(f"scored_{split}.parquet"), index=False)
    return res


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if step == "fit":
        fit_and_evaluate()
    elif step == "score":
        score_split(sys.argv[2])
