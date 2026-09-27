import pandas as pd
import numpy as np
import os
import gc

NORM_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/normalized/"
GT_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/"
CAND_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/candidates/"
OUTPUT_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/pairs/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Every candidate that survives blocking but isn't a true match is, by
# construction, something blocking thought looked plausible -- i.e. already a
# hard negative (similar name/address/tokens/country). We keep all positives
# and cap hard negatives per entity so a handful of entities with huge
# candidate lists don't dominate the training set. Easy/random negatives are
# added on top so the model also sees the "obviously different business"
# case, not just the hard boundary.
HARD_NEG_CAP_PER_ENTITY = 20
EASY_NEG_PER_ENTITY = 2
RANDOM_SEED = 42


def _candidate_file(prefix, sample_size):
    suffix = "_TEST" if sample_size else ""
    return os.path.join(CAND_DIR, f"{prefix}_candidate_pairs{suffix}.tsv")


def _explode_id_list(df, id_col, list_col, out_col):
    rows = df[df[list_col].notna() & (df[list_col].astype(str).str.strip() != "")].copy()
    if rows.empty:
        return pd.DataFrame(columns=[id_col, out_col])
    rows[list_col] = rows[list_col].astype(str).str.split(",")
    exploded = rows.explode(list_col)[[id_col, list_col]].rename(columns={list_col: out_col})
    return exploded


def build_training_pairs(sample_size=None, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    print("Loading candidate pairs and ground truth...")
    cand_file = _candidate_file("train", sample_size)
    cand_df = pd.read_csv(cand_file, sep="\t")
    gt = pd.read_csv(os.path.join(GT_DIR, "train_ground_truth.tsv"), sep="\t")

    cand_pairs = _explode_id_list(cand_df, "source1_entity_id", "candidate_entity_ids", "candidate_entity_id")
    true_pairs = _explode_id_list(gt, "source1_entity_id", "matched_entity_ids", "candidate_entity_id")

    cand_pairs["key"] = cand_pairs["source1_entity_id"] + "||" + cand_pairs["candidate_entity_id"]
    true_pairs["key"] = true_pairs["source1_entity_id"] + "||" + true_pairs["candidate_entity_id"]
    true_keys = set(true_pairs["key"])

    # Restrict to S1 entities actually present in this candidate file (matters
    # when sample_size subsets the source files for a smoke test).
    s1_ids_in_scope = set(cand_df["source1_entity_id"])
    true_pairs_in_scope = true_pairs[true_pairs["source1_entity_id"].isin(s1_ids_in_scope)]

    missed = true_pairs_in_scope[~true_pairs_in_scope["key"].isin(set(cand_pairs["key"]))]
    if len(missed):
        print(f"WARNING: {len(missed)}/{len(true_pairs_in_scope)} true matches were not produced by "
              f"blocking and cannot be used as positive training examples (this is the same "
              f"gap as blocking's recall-ceiling miss rate).")

    cand_pairs["label"] = cand_pairs["key"].isin(true_keys).astype(np.int8)

    positives = cand_pairs[cand_pairs["label"] == 1][["source1_entity_id", "candidate_entity_id", "label"]]
    hard_neg_pool = cand_pairs[cand_pairs["label"] == 0]

    print(f"Positives (true matches surviving blocking): {len(positives)}")
    print(f"Hard-negative pool (blocking candidates, wrong match): {len(hard_neg_pool)}")

    hard_neg_pool = hard_neg_pool.sample(frac=1.0, random_state=seed)
    hard_negatives = (
        hard_neg_pool.groupby("source1_entity_id", sort=False)
        .head(HARD_NEG_CAP_PER_ENTITY)[["source1_entity_id", "candidate_entity_id", "label"]]
    )
    print(f"Hard negatives kept (<= {HARD_NEG_CAP_PER_ENTITY}/entity): {len(hard_negatives)}")
    del cand_pairs, hard_neg_pool
    gc.collect()

    print("Sampling easy (random) negatives...")
    easy_negatives = _sample_easy_negatives(true_keys, sample_size, EASY_NEG_PER_ENTITY, rng)
    print(f"Easy negatives sampled: {len(easy_negatives)}")

    training_pairs = pd.concat([positives, hard_negatives, easy_negatives], ignore_index=True)
    training_pairs = training_pairs.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    training_pairs = training_pairs.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    out_file = os.path.join(OUTPUT_DIR, f"train_pairs{'_TEST' if sample_size else ''}.tsv")
    training_pairs.to_csv(out_file, sep="\t", index=False)

    print(f"\nSaved {len(training_pairs)} labeled pairs to {out_file}")
    print(f"  positives: {(training_pairs['label']==1).sum()}")
    print(f"  negatives: {(training_pairs['label']==0).sum()} "
          f"(hard: {len(hard_negatives)}, easy: {len(easy_negatives)})")
    return training_pairs


def _sample_easy_negatives(true_keys, sample_size, k, rng):
    """For each S1 entity, sample k random Source-2/3 records from the SAME
    country that never showed up as a candidate -- the "obviously different
    business" case (e.g. Acme Robotics vs. a random Delhi bakery)."""
    s1 = pd.read_csv(os.path.join(NORM_DIR, "train_source1.tsv"), sep="\t",
                      usecols=["entity_id", "country"], nrows=sample_size)
    s2 = pd.read_csv(os.path.join(NORM_DIR, "train_source2.tsv"), sep="\t",
                      usecols=["entity_id", "country"], nrows=sample_size)
    s3 = pd.read_csv(os.path.join(NORM_DIR, "train_source3.tsv"), sep="\t",
                      usecols=["entity_id", "country"], nrows=sample_size)
    target_pool = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()

    rows_s1, rows_cand = [], []
    for country, group in s1.groupby("country"):
        pool = target_pool.loc[target_pool["country"] == country, "entity_id"].values
        if len(pool) == 0:
            continue
        n = len(group)
        idx = rng.integers(0, len(pool), size=(n, k))
        sampled = pool[idx]
        s1_ids = group["entity_id"].values
        rows_s1.append(np.repeat(s1_ids, k))
        rows_cand.append(sampled.reshape(-1))

    if not rows_s1:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "label"])

    easy = pd.DataFrame({
        "source1_entity_id": np.concatenate(rows_s1),
        "candidate_entity_id": np.concatenate(rows_cand),
    })
    easy["key"] = easy["source1_entity_id"] + "||" + easy["candidate_entity_id"]

    # Drop any accidental collision with a true match (astronomically rare
    # given pool sizes, but cheap to guarantee); collisions with an existing
    # hard-negative candidate are harmless -- build_training_pairs() dedupes.
    collision = easy["key"].isin(true_keys)
    if collision.any():
        easy = easy[~collision]

    easy["label"] = 0
    return easy[["source1_entity_id", "candidate_entity_id", "label"]]


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        build_training_pairs(sample_size=5000)
    elif mode == "full":
        build_training_pairs(sample_size=None)
    else:
        print("Usage: python build_training_pairs.py [smoke|full]")
