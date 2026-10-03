"""Step 7: write output/matching_results.tsv and output/candidate_pairs.tsv for the test set.

candidate_pairs.tsv = every pair the matching model scored (the pruned blocking output).
matching_results.tsv = the pairs selected by the tuned decision rule (a subset of the candidates).
One row per test S1 entity, in the order of test_source1.tsv; empty list when nothing is selected.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import config
from stage2 import apply_rule


def _ids(split, source):
    return pd.read_parquet(config.norm_path(split, source), columns=["entity_id"])["entity_id"].to_numpy(dtype=object)


def _write(path, header, s1_ids, s1_pos, tgt_ids):
    """s1_pos/tgt_ids: aligned arrays of selected (S1 row, target entity id)."""
    order = np.lexsort((tgt_ids, s1_pos))
    s1_pos, tgt_ids = s1_pos[order], tgt_ids[order]
    lists = [""] * len(s1_ids)
    if len(s1_pos):
        starts = np.flatnonzero(np.r_[True, s1_pos[1:] != s1_pos[:-1]])
        ends = np.r_[starts[1:], len(s1_pos)]
        for a, b in zip(starts, ends):
            ids = list(dict.fromkeys(tgt_ids[a:b]))           # de-duplicate, keep order
            lists[s1_pos[a]] = ",".join(ids)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for sid, lst in zip(s1_ids, lists):
            f.write(f"{sid}\t{lst}\n")
    os.replace(tmp, path)


def main(write_candidates=True):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # BER_SCORED / BER_DECISION select another scored file / decision rule (e.g. stage2v2.py outputs)
    with open(os.environ.get("BER_DECISION", config.work("models", "decision.json"))) as f:
        rule = json.load(f)["rule"]
    # label-shift-corrected stage-2 probabilities (label_shift.py) when available
    adj = config.work("scored_test_adj.parquet")
    scored = pd.read_parquet(os.environ.get("BER_SCORED") or
                             (adj if os.path.exists(adj) else config.work("scored_test.parquet")))
    s1_ids = _ids("test", "source1")
    tgt = {2: _ids("test", "source2"), 3: _ids("test", "source3")}

    def target_ids(df):
        out = np.empty(len(df), dtype=object)
        for code, ids in tgt.items():
            m = df["src"].to_numpy() == code
            out[m] = ids[df["t"].to_numpy()[m]]
        return out

    # candidate set = every blocking pair the stage-1 matcher scored (stage 2 re-scores a subset of them)
    if write_candidates:
      cand = pd.concat([pd.read_parquet(config.work("cand", f"test_{s}.parquet"), columns=["s1", "t"])
                      .assign(src=np.int8(2 if s == "source2" else 3)) for s in ("source2", "source3")],
                     ignore_index=True)
      _write(os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"), ["source1_entity_id", "candidate_entity_ids"],
             s1_ids, cand["s1"].to_numpy(), target_ids(cand))
      print(f"candidates: {len(cand)} pairs for {cand['s1'].nunique()} / {len(s1_ids)} S1 entities")
      del cand
    sel = apply_rule(scored, rule)
    # each S2/S3 record belongs to at most one S1 entity: keep only its highest-probability claimant
    pcol = "p" if "p" in sel.columns else "p2"
    n0 = len(sel)
    sel = sel.sort_values(pcol, ascending=False, kind="mergesort").drop_duplicates(["src", "t"])
    print(f"exclusivity removed {n0 - len(sel)} doubly-claimed pairs")
    _write(os.path.join(config.OUTPUT_DIR, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"],
           s1_ids, sel["s1"].to_numpy(), target_ids(sel))
    n_s1 = sel["s1"].nunique()
    print(f"rule {rule}")
    print(f"matches   : {len(sel)} pairs; {n_s1} S1 entities with >= 1 match "
          f"({n_s1 / len(s1_ids):.3f}), {len(sel) / max(n_s1, 1):.2f} per matched entity")
    validator = os.path.join(config.DATA_DIR, "..", "utils", "validate_submission.py")
    if os.path.exists(validator):
        r = subprocess.run([sys.executable, validator,
                            "--matching", os.path.join(config.OUTPUT_DIR, "matching_results.tsv"),
                            "--candidate", os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"),
                            "--test-dir", os.path.join(config.DATA_DIR, "test"), "--check-ids"],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr)


if __name__ == "__main__":
    main(write_candidates="--matching-only" not in sys.argv)
