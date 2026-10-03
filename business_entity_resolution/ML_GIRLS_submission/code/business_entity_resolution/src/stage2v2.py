"""Stage 2, version 2: trained on the test-like orphan simulation (orphan_sim.py) with the
difference features of difffeats.py.

    python stage2v2.py fit          # train on sim folds A+B, early-stop on sim VAL_ES, tune rule on sim VAL_EVAL
    python stage2v2.py score test   # stage-2 v2 probabilities for every test pair with p1 >= P2_MIN
"""
import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
import difffeats as dfx
from orphan_sim import load_sim_table
from stage2 import P1_CTX, P2_MIN, evaluate_rules, apply_rule, ntrue_for, load_p1, p1_context
from train_stage1 import splits, PARAMS

TAG = os.environ.get("BER_S2_TAG", "v2")


def _path(name):
    return config.work("models", f"stage2{TAG}_{name}")


def _tables(df, mask):
    y = df["label"].to_numpy()[mask].tolist()
    return (dfx.fit_logodds(df["miss_tok"].to_numpy()[mask], y),
            dfx.fit_logodds(df["extra_tok"].to_numpy()[mask], y))


def fit():
    from decide import macro_f05
    sp = splits()
    tab = load_sim_table(["A", "B", "VAL_ES", "VAL_EVAL"])
    diff_path = config.work("sim_diff.parquet")
    if os.path.exists(diff_path):
        diff = pd.read_parquet(diff_path)
    else:
        diff = dfx.compute("train", tab[["s1", "src", "t"]])
        diff.to_parquet(diff_path, index=False)
    for c in diff.columns:
        tab[c] = diff[c].to_numpy()
    del diff
    s1 = tab["s1"].to_numpy()
    inA, inB = np.isin(s1, sp["A"]), np.isin(s1, sp["B"])
    # cross-fitted word log-odds: A rows use the table from B, B rows from A, all others from A+B
    for mask_apply, mask_fit in ((inA, inB), (inB, inA), (~(inA | inB), inA | inB)):
        tb = _tables(tab, mask_fit)
        part = dfx.add_logodds(tab.loc[mask_apply, ["miss_tok", "extra_tok"]].copy(), tb)
        for c in dfx.LO_COLS:
            tab.loc[mask_apply, c] = part[c].to_numpy()
    tables = _tables(tab, inA | inB)
    with open(_path("logodds.json"), "w", encoding="utf-8") as f:
        json.dump({"miss": tables[0], "extra": tables[1]}, f)
    with open(config.work("models", "stage1_features.json")) as f:
        s1cols = json.load(f)
    variants = {"ctx_only": s1cols + P1_CTX, "diff": s1cols + P1_CTX + dfx.DIFF_COLS}
    es = np.isin(s1, sp["VAL_ES"])
    ev_mask = np.isin(s1, sp["VAL_EVAL"])
    s1_rows = sp["VAL_EVAL"]
    ntrue = ntrue_for(s1_rows)
    results = {}
    for name, cols in variants.items():
        dtr = lgb.Dataset(tab.loc[inA | inB, cols].to_numpy(np.float32), tab.loc[inA | inB, "label"].to_numpy(),
                          feature_name=cols)
        dva = lgb.Dataset(tab.loc[es, cols].to_numpy(np.float32), tab.loc[es, "label"].to_numpy(), reference=dtr)
        params = dict(PARAMS, learning_rate=0.03, num_leaves=63, min_data_in_leaf=50)
        m = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        ev = tab.loc[ev_mask, ["s1", "src", "t", "label", "p1", "p1_t_rank"]].copy()
        ev["p2"] = m.predict(tab.loc[ev_mask, cols].to_numpy(np.float32), num_threads=config.N_JOBS)
        print(f"\n=== variant {name}: {len(cols)} features, {m.best_iteration} trees")
        half = np.zeros(len(s1_rows), dtype=bool)
        half[np.random.default_rng(1).permutation(len(s1_rows))[: len(s1_rows) // 2]] = True
        h1, h2 = s1_rows[half], s1_rows[~half]
        ev1, ev2 = ev[np.isin(ev["s1"].to_numpy(), h1)], ev[np.isin(ev["s1"].to_numpy(), h2)]
        best1 = evaluate_rules(ev1, h1, ntrue[half], verbose=False)[0]
        sel = apply_rule(ev2, best1[1])
        honest = macro_f05(h2, ntrue[~half], sel["s1"].to_numpy(), sel["label"].to_numpy())[0]
        best = evaluate_rules(ev, s1_rows, ntrue)[0]
        print(f"  {name}: sim VAL_EVAL best {best[0]:.5f} {best[1]} | honest half-split {honest:.5f}")
        results[name] = (best, honest)
        if name == "diff":
            imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
            print(imp.head(30).round(0).to_string())
            m.save_model(_path("model.txt"), num_iteration=m.best_iteration)
            with open(_path("features.json"), "w") as f:
                json.dump(cols, f)
            with open(_path("decision.json"), "w") as f:
                json.dump({"rule": best[1], "val_macro_f05_sim": best[0], "honest_sim": honest}, f, indent=1)
            ev.to_parquet(config.work(f"sim_val_eval_scored_{TAG}.parquet"), index=False)
        del dtr, dva, m
        gc.collect()
    print({k: (round(v[0][0], 5), round(v[1], 5)) for k, v in results.items()})


def score(split):
    """like stage2.score_split, plus difference features, with the v2 model."""
    from build_feats import load_candidates
    from features import Records, string_features
    with open(_path("features.json")) as f:
        cols = json.load(f)
    model = lgb.Booster(model_file=_path("model.txt"))
    with open(_path("logodds.json"), encoding="utf-8") as f:
        t = json.load(f)
    tables = (t["miss"], t["extra"])
    from orphan_sim import lean_p1_context
    d = load_p1(split)
    src_all = d["src"].to_numpy()
    keep_all = d["p1"].to_numpy() >= P2_MIN
    keeps = {code: keep_all[src_all == code] for code in (2, 3)}      # aligned with each candidate file
    d = lean_p1_context(d, keep_all)
    del src_all, keep_all
    gc.collect()
    src_codes = d["src"].to_numpy()
    ctx_all = {code: (d["s1"].to_numpy()[src_codes == code], d["t"].to_numpy()[src_codes == code],
                      d.loc[src_codes == code, P1_CTX].to_numpy(np.float32)) for code in (2, 3)}
    del d, src_codes
    gc.collect()
    out = []
    for src in ("source2", "source3"):
        cs1, ct, cmat = ctx_all.pop(2 if src == "source2" else 3)
        keep = keeps[2 if src == "source2" else 3]
        code = 2 if src == "source2" else 3
        diff = dfx.compute(split, pd.DataFrame({"s1": cs1, "src": np.int8(code), "t": ct}))
        diff = dfx.add_logodds(diff, tables)[dfx.DIFF_COLS].to_numpy(np.float32)
        gc.collect()
        # candidate context over the full table, then keep only the stage-2 rows before loading the records
        c = load_candidates(split, src)
        c = c[keep].reset_index(drop=True)
        gc.collect()
        assert (cs1 == c["s1"].to_numpy()).all() and (ct == c["t"].to_numpy()).all()
        ra, rb = Records(split, "source1"), Records(split, src)
        pos = 0
        for st in range(0, len(c), 1_000_000):
            ch = c.iloc[st:st + 1_000_000].reset_index(drop=True)
            f = string_features(ra, rb, ch["s1"].to_numpy(), ch["t"].to_numpy())
            ch = pd.concat([ch, pd.DataFrame(f)], axis=1)
            sl = slice(pos, pos + len(ch))
            for i, k in enumerate(P1_CTX):
                ch[k] = cmat[sl, i]
            for i, k in enumerate(dfx.DIFF_COLS):
                ch[k] = diff[sl, i]
            p2 = model.predict(ch[cols].to_numpy(np.float32), num_threads=config.N_JOBS).astype(np.float32)
            out.append(pd.DataFrame({"s1": ch["s1"].to_numpy(), "src": ch["src"].to_numpy(), "t": ch["t"].to_numpy(),
                                     "p1": ch["p1"].to_numpy(), "p2": p2, "p1_t_rank": ch["p1_t_rank"].to_numpy()}))
            pos += len(ch)
            print(f"    {split}/{src}: {pos}/{len(c)} stage-2 pairs", flush=True)
        del c, ra, rb
        assert pos == len(cs1)
        del cs1, ct, cmat, diff
        gc.collect()
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(config.work(f"scored_{split}_{TAG}.parquet"), index=False)
    return res


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if step == "fit":
        fit()
    elif step == "score":
        score(sys.argv[2])
