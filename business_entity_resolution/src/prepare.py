"""Step 1: normalise every source file once -> parquet (work/norm/<split>_<source>.parquet).

usage: python prepare.py [train|test|all]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
from textnorm import normalize_name, normalize_address, skeleton, has_indic

COLS = ["name_full", "name_core", "is_domain", "name_dba", "name_indic", "name_skel",
        "addr", "state", "postal", "house", "city", "nums"]


def _normalize_chunk(args):
    names, addrs, countries = args
    out = []
    for n, a, c in zip(names, addrs, countries):
        full, core, is_dom, dba = normalize_name(n)
        addr, state, postal, house, city, nums = normalize_address(a, c)
        out.append((full, core, is_dom, dba, int(has_indic(n)), skeleton(core),
                    addr, state, postal, house, city, nums))
    return out


def prepare_file(split, source, pool):
    out_path = config.norm_path(split, source)
    if os.path.exists(out_path):
        print(f"  {out_path} exists, skipping")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(config.raw_path(split, source), sep="\t", dtype=str, keep_default_na=False,
                     quoting=3)
    df["country"] = df["country"].str.strip()
    n = len(df)
    step = 20000
    jobs = [(df["business_name"].values[i:i + step].tolist(), df["business_address"].values[i:i + step].tolist(),
             df["country"].values[i:i + step].tolist()) for i in range(0, n, step)]
    rows = []
    for part in pool.imap(_normalize_chunk, jobs, chunksize=1):
        rows.extend(part)
    norm = pd.DataFrame(rows, columns=COLS)
    norm["is_domain"] = norm["is_domain"].astype(np.int8)
    norm["name_indic"] = norm["name_indic"].astype(np.int8)
    out = pd.concat([df[["entity_id", "country", "business_name", "business_address"]].reset_index(drop=True),
                     norm], axis=1)
    out.to_parquet(out_path, index=False)
    print(f"  {split}_{source}: {n} rows in {time.time() - t0:.0f}s")


def main(which="all"):
    splits = ["train", "test"] if which == "all" else [which]
    with Pool(config.N_JOBS) as pool:
        for split in splits:
            for src in config.SOURCES:
                prepare_file(split, src, pool)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
