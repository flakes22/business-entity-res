"""Write output/matching_results.tsv from the stage-2 v2 scores (scored_test_v2.parquet) with v2's tuned rule."""
import json, os, shutil, subprocess, sys
import numpy as np, pandas as pd
import config
from stage2 import apply_rule
from make_submission import _ids, _write

tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
with open(config.work("models", f"stage2{tag}_decision.json")) as f:
    rule = json.load(f)["rule"]
scored = pd.read_parquet(config.work(f"scored_test_{tag}.parquet"))
s1_ids = _ids("test", "source1")
tgt = {2: _ids("test", "source2"), 3: _ids("test", "source3")}
sel = apply_rule(scored, rule)
pcol = "p" if "p" in sel.columns else "p2"
n0 = len(sel)
sel = sel.sort_values(pcol, ascending=False, kind="mergesort").drop_duplicates(["src", "t"])
out_t = np.empty(len(sel), dtype=object)
for code, ids in tgt.items():
    m = sel["src"].to_numpy() == code
    out_t[m] = ids[sel["t"].to_numpy()[m]]
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
_write(os.path.join(config.OUTPUT_DIR, "matching_results.tsv"), ["source1_entity_id", "matched_entity_ids"],
       s1_ids, sel["s1"].to_numpy(), out_t)
n_s1 = sel["s1"].nunique()
print(f"rule {rule}; exclusivity removed {n0 - len(sel)}")
print(f"matches: {len(sel)} pairs; {n_s1} S1 with >=1 match ({n_s1 / len(s1_ids):.3f})")
validator = os.path.join(config.DATA_DIR, "..", "utils", "validate_submission.py")
r = subprocess.run([sys.executable, validator, "--matching", os.path.join(config.OUTPUT_DIR, "matching_results.tsv"),
                    "--test-dir", os.path.join(config.DATA_DIR, "test"), "--check-ids"], capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-600:])
