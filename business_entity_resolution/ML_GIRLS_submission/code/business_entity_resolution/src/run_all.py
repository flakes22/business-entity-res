"""End-to-end pipeline for the FINAL submission (stage 2 v2, public leaderboard 0.973):
data -> normalisation -> blocking -> stage-1 matcher -> orphan simulation -> stage-2 v2 -> output TSVs.

    python run_all.py            # everything (each step skips work whose output already exists)

  1. prepare.py            normalise the 6 source files                               -> work/norm/
  2. blocking.py train|test candidate generation S1 -> S2, S1 -> S3                     -> work/cand/
  3. train_stage1.py       stage-1 features for the train subsets + LightGBM folds A/B -> work/feats/, work/models/
  4. predict_stage1.py     stage-1 probabilities for the train pairs stage 2 needs     -> work/p1/
  5. stage2.py fit         stage-2 v1 (baseline, provides the shared context code / validation split)
  6. orphan_sim.py build   test-like simulation on train (drop 19% of unused S1 records -> orphans)
  7. stage2v2.py fit       stage-2 v2 with difference features, trained + rule-tuned on the simulation
  8. predict_stage1.py test, stage2v2.py score test   probabilities for every test candidate
  9. make_submission.py    output/matching_results.tsv + output/candidate_pairs.tsv (+ official validator)
"""
import os
import subprocess
import sys

import config

HERE = os.path.dirname(os.path.abspath(__file__))


def step(*args, env=None):
    print(f"\n########## {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, args[0])] + list(args[1:]), check=True, cwd=HERE,
                   env=dict(os.environ, **(env or {})))


if __name__ == "__main__":
    step("prepare.py", "all")
    step("blocking.py", "train")
    step("blocking.py", "test")
    step("train_stage1.py", "all")
    step("predict_stage1.py", "train")
    step("stage2.py", "fit")
    step("orphan_sim.py", "build")
    step("stage2v2.py", "fit")
    step("predict_stage1.py", "test")
    if not os.path.exists(config.work("scored_test_v2.parquet")):
        step("stage2v2.py", "score", "test")
    step("make_submission.py", env={"BER_SCORED": config.work("scored_test_v2.parquet"),
                                    "BER_DECISION": config.work("models", "stage2v2_decision.json")})
