"""Flag pairs whose house numbers differ only by leading zeros ("25" vs "025") -- equal after stripping."""
import numpy as np, pandas as pd, config, gc
def strip0(x):
    return np.array([s.lstrip("0") or ("0" if s else "") for s in x], dtype=object)
def pad_flag(split, df):
    a = pd.read_parquet(config.norm_path(split, "source1"), columns=["house"])["house"].to_numpy(dtype=object)
    out = np.zeros(len(df), dtype=np.int8)
    for code, src in ((2, "source2"), (3, "source3")):
        b = pd.read_parquet(config.norm_path(split, src), columns=["house"])["house"].to_numpy(dtype=object)
        m = (df["src"].to_numpy() == code)
        ha, hb = a[df["s1"].to_numpy()[m]], b[df["t"].to_numpy()[m]]
        out[m] = ((ha != hb) & (strip0(ha) == strip0(hb)) & (ha != "") & (hb != "")).astype(np.int8)
        del b; gc.collect()
    return out
if __name__ == "__main__":
    ev = pd.read_parquet(config.work("val_eval_scored.parquet"))
    ev["house_pad"] = pad_flag("train", ev); ev.to_parquet(config.work("val_eval_scored_pad.parquet"), index=False)
    te = pd.read_parquet(config.work("scored_test_seg.parquet"))
    te["house_pad"] = pad_flag("test", te); te.to_parquet(config.work("scored_test_seg.parquet"), index=False)
    print("done")
