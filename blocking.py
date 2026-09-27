import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import gc
import os
import sys
import time

TRAIN_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/normalized/"
GT_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/"
OUTPUT_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/candidates/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only the columns blocking actually needs -- cuts memory roughly in half on the
# multi-GB normalized files (skips the raw business_name/business_address and
# the suffix-bearing norm_name column, which nothing here reads).
BLOCKING_COLUMNS = ["entity_id", "country", "norm_name_no_suffix", "norm_address"]

# ------------------------------------------------------------------
# Tunables. Frequency caps are expressed as a FRACTION of the target
# source's size, not a fixed count -- a fixed count (e.g. 300) means a
# vastly looser filter once you go from a 50k-row smoke sample to the
# full 5M-row source, which is exactly what blew up the candidate sets
# in the original script (median ~464 candidates/entity on a 50k-row
# sample against a 50k-row target -- i.e. ~1% of the entire target pool
# per entity, not a "block").
# ------------------------------------------------------------------
FREQ_CAP_FRACTION = 0.0008   # a blocking key/token used by more than this
                             # fraction of the target pool is too generic
MIN_FREQ_CAP = 30            # floor so small samples/countries aren't gutted

MIN_TOKEN_LEN_NAME = 4
MIN_TOKEN_LEN_ADDR_ALPHA = 5
MIN_TOKEN_LEN_ADDR_NUM = 3
MIN_OVERLAP_ADDR = 2         # address words (street/road/city) are individually
                             # common; require >=2 shared tokens to count as a
                             # block match. Names are short, so 1 shared rare
                             # token is kept for name-token blocking.

MAX_CANDIDATES_PER_ENTITY = 75   # hard cap on what candidate_pairs.tsv can hold
                                 # per Source-1 entity -- this is what actually
                                 # gets fed to the matching model, so an
                                 # unbounded union here just moves the O(N*M)
                                 # blow-up one stage downstream instead of
                                 # fixing it.

MIN_CANDIDATES_BEFORE_TFIDF = 3  # TF-IDF blocks only run for S1 entities that
                                  # the cheap blocks (prefix/token) didn't
                                  # already cover well -- this is the single
                                  # biggest lever for making this runnable on
                                  # a laptop: TF-IDF cost scales with the
                                  # number of queries, and most entities don't
                                  # need it once names/addresses roughly match.

TFIDF_TOP_K = 8
TFIDF_DIST_THRESHOLD = 0.6


def _freq_cap(target_len):
    return max(MIN_FREQ_CAP, int(target_len * FREQ_CAP_FRACTION))


def _load_source(prefix, name, sample_size):
    path = os.path.join(TRAIN_DIR, f"{prefix}_{name}.tsv")
    df = pd.read_csv(path, sep="\t", usecols=BLOCKING_COLUMNS, nrows=sample_size)
    df["norm_name_no_suffix"] = df["norm_name_no_suffix"].fillna("")
    df["norm_address"] = df["norm_address"].fillna("")
    df["country"] = df["country"].astype(str)
    return df


# ============================================================
# BLOCK 1: Name Prefix Blocking (Country + first N chars)
# ============================================================
def block_name_prefix(s1, s_target, prefix_lengths=(3, 4, 5)):
    """Multiple name prefix blocks to catch variations."""
    all_pairs = []
    for n in prefix_lengths:
        # Filter on the actual name length (not the combined "country_prefix"
        # key length -- that conflated the country label's length with the
        # prefix length and filtered inconsistently across countries).
        s1_ok = s1["norm_name_no_suffix"].str.len() >= n
        t_ok = s_target["norm_name_no_suffix"].str.len() >= n

        s1_key = s1.loc[s1_ok, "country"] + "_" + s1.loc[s1_ok, "norm_name_no_suffix"].str[:n]
        t_key = s_target.loc[t_ok, "country"] + "_" + s_target.loc[t_ok, "norm_name_no_suffix"].str[:n]

        s1_temp = pd.DataFrame({"entity_id_s1": s1.loc[s1_ok, "entity_id"], "bk": s1_key})
        t_temp = pd.DataFrame({"entity_id_cand": s_target.loc[t_ok, "entity_id"], "bk": t_key})

        cap = _freq_cap(len(s_target))
        t_counts = t_temp["bk"].value_counts()
        valid_keys = t_counts[t_counts <= cap].index
        t_temp = t_temp[t_temp["bk"].isin(valid_keys)]

        merged = pd.merge(s1_temp, t_temp, on="bk")[["entity_id_s1", "entity_id_cand"]]
        all_pairs.append(merged)

    return pd.concat(all_pairs, ignore_index=True).drop_duplicates() if all_pairs else pd.DataFrame(columns=["entity_id_s1", "entity_id_cand"])


# ============================================================
# Shared vectorized token-overlap blocking (names or addresses)
# ============================================================
def _token_overlap_pairs(s1, s_target, id_col_s1, id_col_t, text_col, extract_tokens, min_overlap):
    """Explode text into blocking tokens, cap over-common ones, join on shared
    tokens, and require >= min_overlap distinct shared tokens per pair. This
    replaces the original per-row Python loops (iterrows over millions of rows
    with a hand-rolled dict index) with vectorized pandas operations, and adds
    the overlap requirement that turns a "any shared word" match (extremely
    weak at this data scale) into a real signal.
    """
    s1_tok = pd.DataFrame({
        "entity_id_s1": s1[id_col_s1],
        "country": s1["country"],
        "token": s1[text_col].map(extract_tokens),
    }).explode("token").dropna(subset=["token"])
    s1_tok["bk"] = s1_tok["country"] + "_" + s1_tok["token"]

    t_tok = pd.DataFrame({
        "entity_id_cand": s_target[id_col_t],
        "country": s_target["country"],
        "token": s_target[text_col].map(extract_tokens),
    }).explode("token").dropna(subset=["token"])
    t_tok["bk"] = t_tok["country"] + "_" + t_tok["token"]

    cap = _freq_cap(len(s_target))
    t_counts = t_tok["bk"].value_counts()
    valid_keys = t_counts[t_counts <= cap].index
    t_tok = t_tok[t_tok["bk"].isin(valid_keys)]

    merged = pd.merge(s1_tok[["entity_id_s1", "bk"]], t_tok[["entity_id_cand", "bk"]], on="bk")
    del s1_tok, t_tok
    gc.collect()

    if merged.empty:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_cand"])

    overlap = merged.groupby(["entity_id_s1", "entity_id_cand"]).size()
    pairs = overlap[overlap >= min_overlap].index.to_frame(index=False)
    return pairs


def _name_tokens(text):
    toks = text.split()
    return [t for t in toks if len(t) >= MIN_TOKEN_LEN_NAME] or None


def _addr_tokens(text):
    toks = text.split()
    out = [t for t in toks if (t.isdigit() and len(t) >= MIN_TOKEN_LEN_ADDR_NUM) or (len(t) >= MIN_TOKEN_LEN_ADDR_ALPHA and t.isalpha())]
    return out or None


# ============================================================
# BLOCK 2: Name Token Overlap Blocking
# ============================================================
def block_name_tokens(s1, s_target):
    return _token_overlap_pairs(s1, s_target, "entity_id", "entity_id", "norm_name_no_suffix", _name_tokens, min_overlap=1)


# ============================================================
# BLOCK 3: Address Token Overlap Blocking
# ============================================================
def block_address_tokens(s1, s_target):
    return _token_overlap_pairs(s1, s_target, "entity_id", "entity_id", "norm_address", _addr_tokens, min_overlap=MIN_OVERLAP_ADDR)


# ============================================================
# BLOCK 4/5: TF-IDF Blocking (char n-grams, per country)
# ============================================================
def block_tfidf(s1, s_target, text_column, top_k=TFIDF_TOP_K, batch_size=10000, distance_threshold=TFIDF_DIST_THRESHOLD):
    """TF-IDF char n-gram blocking, done PER COUNTRY to enforce country match.
    `s1` is expected to already be filtered down to the entities that need
    this fallback -- see generate_all_candidates."""
    if len(s1) == 0:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_cand"])

    countries = set(s1["country"].unique()) & set(s_target["country"].unique())

    all_pairs = []
    for country in countries:
        s1_c = s1[s1["country"] == country].reset_index(drop=True)
        t_c = s_target[s_target["country"] == country].reset_index(drop=True)

        if len(s1_c) == 0 or len(t_c) == 0:
            continue

        print(f"    TF-IDF [{text_column}] country={country}: {len(s1_c)} queries vs {len(t_c)} targets")
        s1_texts = s1_c[text_column].tolist()
        t_texts = t_c[text_column].tolist()

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=0.9,
                                     max_features=50000, dtype=np.float32)
        t_matrix = vectorizer.fit_transform(t_texts)
        s1_matrix = vectorizer.transform(s1_texts)

        actual_k = min(top_k, t_matrix.shape[0])
        nn = NearestNeighbors(n_neighbors=actual_k, metric="cosine", n_jobs=-1)
        nn.fit(t_matrix)

        s1_ids = s1_c["entity_id"].values
        t_ids = t_c["entity_id"].values

        for start in range(0, s1_matrix.shape[0], batch_size):
            end = min(start + batch_size, s1_matrix.shape[0])
            dists, idxs = nn.kneighbors(s1_matrix[start:end])

            for i in range(end - start):
                for j in range(actual_k):
                    if dists[i, j] < distance_threshold:
                        all_pairs.append({
                            "entity_id_s1": s1_ids[start + i],
                            "entity_id_cand": t_ids[idxs[i, j]],
                        })

        del s1_matrix, t_matrix, vectorizer, nn
        gc.collect()

    return pd.DataFrame(all_pairs).drop_duplicates() if all_pairs else pd.DataFrame(columns=["entity_id_s1", "entity_id_cand"])


def block_tfidf_address(s1, s_target, **kwargs):
    """TF-IDF on addresses PER COUNTRY. Catches transliteration cases where
    names share zero character overlap but addresses overlap significantly."""
    return block_tfidf(s1, s_target, text_column="norm_address", **kwargs)


# ============================================================
# MAIN PIPELINE
# ============================================================
def _cap_per_entity(tagged_blocks, max_candidates=MAX_CANDIDATES_PER_ENTITY):
    """Union candidates across blocks, scored by how many distinct blocking
    strategies agreed on each pair ("support"), and keep only the top
    `max_candidates` per Source-1 entity. Without this, a single noisy block
    (or several together) can hand hundreds of candidates per entity to the
    matching stage -- this is the hard ceiling on candidate_pairs.tsv size."""
    frames = [df.assign(block=name) for name, df in tagged_blocks if len(df) > 0]
    if not frames:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_cand"])

    combined = pd.concat(frames, ignore_index=True)
    del frames
    support = combined.groupby(["entity_id_s1", "entity_id_cand"])["block"].nunique().reset_index(name="support")
    del combined
    gc.collect()

    support = support.sort_values(["entity_id_s1", "support"], ascending=[True, False])
    capped = support.groupby("entity_id_s1", sort=False).head(max_candidates)
    return capped[["entity_id_s1", "entity_id_cand"]]


def generate_all_candidates(s1, s_target, source_label, tfidf_mode="fallback"):
    """Union of all blocking strategies for S1 -> one target source.
    tfidf_mode: "fallback" (default) runs TF-IDF only for S1 entities the
    cheap blocks didn't already cover -- much cheaper at this data scale.
    "full" runs TF-IDF for every S1 entity (only feasible with real compute
    budget, e.g. a cloud box, not a laptop)."""
    print(f"\n  === Blocking S1 -> {source_label} ({len(s1)} x {len(s_target)}) ===")

    t0 = time.time()
    print("  [1/5] Name Prefix Blocking...")
    b1 = block_name_prefix(s1, s_target)
    print(f"         -> {len(b1)} pairs ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  [2/5] Name Token Blocking...")
    b2 = block_name_tokens(s1, s_target)
    print(f"         -> {len(b2)} pairs ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  [3/5] Address Token Blocking...")
    b3 = block_address_tokens(s1, s_target)
    print(f"         -> {len(b3)} pairs ({time.time()-t0:.1f}s)")

    if tfidf_mode == "full":
        s1_for_tfidf = s1
    else:
        cheap = pd.concat([b1, b2, b3], ignore_index=True).drop_duplicates()
        cheap_counts = cheap.groupby("entity_id_s1").size()
        del cheap
        well_covered = set(cheap_counts[cheap_counts >= MIN_CANDIDATES_BEFORE_TFIDF].index)
        s1_for_tfidf = s1[~s1["entity_id"].isin(well_covered)]
        print(f"  TF-IDF fallback: {len(s1_for_tfidf)}/{len(s1)} entities need it "
              f"({len(s1) - len(s1_for_tfidf)} already had >= {MIN_CANDIDATES_BEFORE_TFIDF} cheap candidates)")

    t0 = time.time()
    print("  [4/5] TF-IDF Name Blocking (per country)...")
    b4 = block_tfidf(s1_for_tfidf, s_target, text_column="norm_name_no_suffix")
    print(f"         -> {len(b4)} pairs ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  [5/5] TF-IDF Address Blocking (per country)...")
    b5 = block_tfidf_address(s1_for_tfidf, s_target)
    print(f"         -> {len(b5)} pairs ({time.time()-t0:.1f}s)")

    capped = _cap_per_entity([
        ("name_prefix", b1), ("name_tokens", b2), ("address_tokens", b3),
        ("tfidf_name", b4), ("tfidf_address", b5),
    ])
    print(f"  Capped union (<= {MAX_CANDIDATES_PER_ENTITY}/entity): {len(capped)} pairs")

    del b1, b2, b3, b4, b5
    gc.collect()

    return capped


def _candidate_file(prefix, sample_size):
    suffix = "_TEST" if sample_size else ""
    return os.path.join(OUTPUT_DIR, f"{prefix}_candidate_pairs{suffix}.tsv")


def print_candidate_stats(df):
    counts = df["candidate_entity_ids"].fillna("").apply(lambda x: 0 if x == "" else len(x.split(",")))
    print(f"  entities: {len(df)} | zero-candidate: {(counts==0).sum()} | "
          f"mean: {counts.mean():.1f} | median: {counts.median():.0f} | max: {counts.max()}")


def run_blocking_pipeline(is_test=False, sample_size=None, tfidf_mode="fallback"):
    prefix = "test" if is_test else "train"
    print(f"\n{'='*60}\n  BLOCKING PIPELINE — {prefix.upper()} DATA\n{'='*60}")
    if sample_size:
        print(f"  SMOKE-TEST MODE: only first {sample_size} rows of each source")

    print("Loading normalized data...")
    s1 = _load_source(prefix, "source1", sample_size)
    s2 = _load_source(prefix, "source2", sample_size)
    s3 = _load_source(prefix, "source3", sample_size)

    s2_candidates = generate_all_candidates(s1, s2, "Source 2", tfidf_mode=tfidf_mode)
    del s2
    gc.collect()

    s3_candidates = generate_all_candidates(s1, s3, "Source 3", tfidf_mode=tfidf_mode)
    del s3
    gc.collect()

    print("\nUnioning S2 + S3 candidates and formatting output...")
    final_candidates = pd.concat([s2_candidates, s3_candidates], ignore_index=True).drop_duplicates()
    del s2_candidates, s3_candidates
    gc.collect()

    grouped = final_candidates.groupby("entity_id_s1")["entity_id_cand"].apply(
        lambda x: ",".join(sorted(set(x)))
    ).reset_index()
    grouped.rename(columns={"entity_id_s1": "source1_entity_id", "entity_id_cand": "candidate_entity_ids"}, inplace=True)

    all_s1_ids = pd.DataFrame({"source1_entity_id": s1["entity_id"]})
    final_output = pd.merge(all_s1_ids, grouped, on="source1_entity_id", how="left")
    final_output["candidate_entity_ids"] = final_output["candidate_entity_ids"].fillna("")

    out_file = _candidate_file(prefix, sample_size)
    final_output.to_csv(out_file, sep="\t", index=False)
    print(f"\nSUCCESS! Candidate pairs saved to {out_file}")
    print_candidate_stats(final_output)


# ============================================================
# RECALL CEILING EVALUATION
# ============================================================
def evaluate_recall(sample_size=None):
    """Measure recall ceiling against ground truth (train only -- test has no labels)."""
    print("\n" + "=" * 60)
    print("  RECALL CEILING EVALUATION")
    print("=" * 60)

    gt = pd.read_csv(os.path.join(GT_DIR, "train_ground_truth.tsv"), sep="\t")
    cand_file = _candidate_file("train", sample_size)
    cands = pd.read_csv(cand_file, sep="\t")
    cands["candidate_entity_ids"] = cands["candidate_entity_ids"].fillna("")
    cand_map = dict(zip(cands["source1_entity_id"], cands["candidate_entity_ids"]))

    s1 = _load_source("train", "source1", sample_size)
    s2 = _load_source("train", "source2", sample_size)
    s3 = _load_source("train", "source3", sample_size)

    s1_ids = set(s1["entity_id"].values)
    valid_targets = set(s2["entity_id"].values) | set(s3["entity_id"].values)
    del s1, s2, s3
    gc.collect()

    gt_sample = gt[gt["source1_entity_id"].isin(s1_ids)]

    total = 0
    found = 0
    for s1_id, matched in zip(gt_sample["source1_entity_id"], gt_sample["matched_entity_ids"]):
        if pd.isna(matched) or str(matched).strip() == "":
            continue
        true_in_sample = set(str(matched).split(",")) & valid_targets
        if not true_in_sample:
            continue
        our_cands = set(cand_map.get(s1_id, "").split(",")) if cand_map.get(s1_id, "") else set()
        total += len(true_in_sample)
        found += len(true_in_sample & our_cands)

    print(f"Total true matches (within sample): {total}")
    print(f"Found by blocking: {found}")
    print(f"Missed by blocking: {total - found}")
    if total > 0:
        print(f"RECALL CEILING: {found / total * 100:.2f}%")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        run_blocking_pipeline(is_test=False, sample_size=5000)
        evaluate_recall(sample_size=5000)
    elif mode == "full":
        run_blocking_pipeline(is_test=False, sample_size=None)
        evaluate_recall(sample_size=None)
        run_blocking_pipeline(is_test=True, sample_size=None)
    else:
        print("Usage: python blocking.py [smoke|full]")
