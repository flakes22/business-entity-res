"""House-number / name / address agreement for the stage-2-scored test pairs (segment label-shift)."""
import numpy as np, pandas as pd, config, gc
from features import Records, _eq, _sim
from rapidfuzz import fuzz
sc = pd.read_parquet(config.work("scored_test_adj.parquet"))
a = Records("test", "source1")
out = []
for code, src in ((2, "source2"), (3, "source3")):
    d = sc[sc["src"] == code].copy()
    b = Records("test", src)
    ia, ib = d["s1"].to_numpy(), d["t"].to_numpy()
    d["house_eq"] = _eq(a.house[ia], b.house[ib])
    d["addr_empty_b"] = np.fromiter((not s for s in b.addr[ib]), dtype=bool, count=len(ib)).astype(np.float32)
    d["name_tset"] = _sim(fuzz.token_set_ratio, a.name_core[ia], b.name_core[ib])
    out.append(d); del b; gc.collect()
pd.concat(out, ignore_index=True).to_parquet(config.work("scored_test_seg.parquet"), index=False)
print("done")
