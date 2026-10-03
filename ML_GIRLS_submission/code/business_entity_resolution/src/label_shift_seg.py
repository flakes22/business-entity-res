"""Step 6b (v3): label-shift correction per (country label, address segment).

Segments: target address empty / house numbers equal / house numbers differ / house number
unknown. Stage 2 is calibrated within each segment on held-out train entities; on test the
share of true matches differs strongly by segment (the extra test records are near-duplicate
decoys at neighbouring house numbers). EM (Saerens et al., 2002) estimates each test
segment's prior from the model's own probabilities; odds are rescaled accordingly.
"""
import json
import numpy as np
import pandas as pd
import config
from label_shift import em_prior, adjust
from stage2 import P2_MIN


def segment(d):
    return np.where(d["addr_empty_b"] == 1, "noaddr", np.where(d["house_eq"] == 1, "house_eq",
                    np.where(d["house_eq"] == 0, "house_ne", "house_na")))


def main():
    ev = pd.read_parquet(config.work("val_eval_scored.parquet"))
    ev = ev[ev["p1"] >= P2_MIN]
    f = pd.concat([pd.read_parquet(config.work("feats", f"train_{s}.parquet"),
                                   columns=["s1", "t", "src", "house_eq", "addr_empty_b"]) for s in ("source2", "source3")])
    ev = ev.merge(f, on=["s1", "t", "src"], how="left")
    ev["seg"] = segment(ev)
    pi_train = ev.groupby("seg")["label"].mean().to_dict()
    te = pd.read_parquet(config.work("scored_test_seg.parquet"))
    te["seg"] = segment(te)
    country = pd.read_parquet(config.norm_path("test", "source1"), columns=["country"])["country"].to_numpy()
    te["country"] = country[te["s1"].to_numpy()]
    p = te["p2_raw"].to_numpy(np.float64)
    p_adj = p.copy()
    priors = {}
    for (c, s), idx in te.groupby(["country", "seg"]).indices.items():
        pi_s = pi_train[s]
        pi_t = em_prior(p[idx], pi_s)
        p_adj[idx] = adjust(p[idx], pi_s, pi_t)
        priors[f"{c}|{s}"] = {"train_prior": round(pi_s, 4), "test_prior": round(pi_t, 4), "n": int(len(idx))}
        print(f"  {c:8s} {s:9s}: {pi_s:.4f} -> {pi_t:.4f}  ({len(idx)} pairs)")
    te["p2"] = p_adj.astype(np.float32)
    te[["s1", "src", "t", "p1", "p2", "p1_t_rank", "p2_raw"]].to_parquet(config.work("scored_test_adj.parquet"), index=False)
    with open(config.work("models", "label_shift.json"), "w") as fh:
        json.dump({"pi_train": pi_train, "priors": priors}, fh, indent=1)
    return pi_train, priors


if __name__ == "__main__":
    main()
