"""Step 6b: label-shift (prior) correction of the stage-2 probabilities on the test set.

The test pools contain ~23% more S2/S3 records per S1 entity than train, mostly near-duplicate
decoys (same name, neighbouring house number). Stage 2 is well calibrated on held-out train
entities, but on test the share of true matches among plausible pairs is lower. We estimate
the test prior with the EM procedure of Saerens, Latinne & Decaestecker (2002), separately
for every (country label, target source) group -- labels come from the data, open set --
and rescale each probability's odds by (pi_test / pi_train) / ((1 - pi_test) / (1 - pi_train)).
The unchanged, validation-tuned decision rule is then applied to the corrected probabilities.
No test labels are used.
"""
import json

import numpy as np
import pandas as pd

import config
from stage2 import P2_MIN


def em_prior(p, pi_s, iters=500, tol=1e-8):
    pi = pi_s
    for _ in range(iters):
        a, b = pi / pi_s, (1 - pi) / (1 - pi_s)
        q = a * p / (a * p + b * (1 - p))
        new = q.mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def adjust(p, pi_s, pi_t):
    a, b = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    return a * p / (a * p + b * (1 - p))


def main():
    ev = pd.read_parquet(config.work("val_eval_scored.parquet"))
    ev = ev[ev["p1"] >= P2_MIN]
    pi_src = ev.groupby("src")["label"].mean().to_dict()          # train prior per target source
    sc = pd.read_parquet(config.work("scored_test.parquet"))
    country = pd.read_parquet(config.norm_path("test", "source1"), columns=["country"])["country"].to_numpy()
    sc["country"] = country[sc["s1"].to_numpy()]
    p = sc["p2"].to_numpy(np.float64)
    p_adj = p.copy()
    priors = {}
    for (c, s), idx in sc.groupby(["country", "src"]).indices.items():
        pi_s = pi_src[s]
        pi_t = em_prior(p[idx], pi_s)
        p_adj[idx] = adjust(p[idx], pi_s, pi_t)
        priors[f"{c}|S{s}"] = {"train_prior": round(pi_s, 4), "test_prior": round(pi_t, 4), "n": int(len(idx))}
        print(f"  {c:8s} S{s}: train prior {pi_s:.4f} -> test EM prior {pi_t:.4f}  ({len(idx)} pairs)")
    sc["p2_raw"] = sc["p2"]
    sc["p2"] = p_adj.astype(np.float32)
    sc.drop(columns=["country"]).to_parquet(config.work("scored_test_adj.parquet"), index=False)
    with open(config.work("models", "label_shift.json"), "w") as f:
        json.dump(priors, f, indent=1)


if __name__ == "__main__":
    main()
