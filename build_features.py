import os
import re
import gc
import sys
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

# --- paths ---
DATASET_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset/"
NORM_DIR = DATASET_DIR + "normalized/"
CAND_DIR = DATASET_DIR + "candidates/"
PAIRS_DIR = DATASET_DIR + "pairs/"
FEATURES_DIR = DATASET_DIR + "features/"
os.makedirs(FEATURES_DIR, exist_ok=True)
# --- end paths ---

REC_COLS = ["entity_id", "business_address", "country", "norm_name", "norm_name_no_suffix", "norm_address"]
TFIDF_FIT_SAMPLE = 300_000    # IDF is estimated on a sample of records, not every record
CHUNK_SIZE = 1_000_000        # pairs per feature chunk (one parquet part each) -- bounds RAM

# State gazetteers. The SAME state is written differently per source (S1 US: "NC", S3 US:
# "North Carolina"; S1 India: "Madhya Pradesh", S3 India: "MP" or Devanagari), so every alias
# is mapped to one canonical lowercase name -- otherwise state_match is NaN for most S1<->S3
# pairs. Tables are per country because abbreviations collide ("OR" = Oregon vs Odisha).
# Countries not listed (e.g. France) get an empty state -> state_match = NaN ("unknown"),
# never a false mismatch.
_US = """al alabama|ak alaska|az arizona|ar arkansas|ca california|co colorado|ct connecticut|
de delaware|dc district of columbia|fl florida|ga georgia|hi hawaii|id idaho|il illinois|
in indiana|ia iowa|ks kansas|ky kentucky|la louisiana|me maine|md maryland|ma massachusetts|
mi michigan|mn minnesota|ms mississippi|mo missouri|mt montana|ne nebraska|nv nevada|
nh new hampshire|nj new jersey|nm new mexico|ny new york|nc north carolina|nd north dakota|
oh ohio|ok oklahoma|or oregon|pa pennsylvania|ri rhode island|sc south carolina|
sd south dakota|tn tennessee|tx texas|ut utah|vt vermont|va virginia|wa washington|
wv west virginia|wi wisconsin|wy wyoming"""

# canonical name -> "|"-separated aliases (abbreviations + Devanagari/Tamil/Telugu/Kannada/Bengali/... names)
_INDIA = {
    "andhra pradesh": "ap|आंध्र प्रदेश|ఆంధ్రప్రదేశ్", "arunachal pradesh": "ar|अरुणाचल प्रदेश", "assam": "as|असम",
    "bihar": "br|बिहार", "chhattisgarh": "cg|ct|छत्तीसगढ़", "goa": "ga|गोवा", "gujarat": "gj|गुजरात|ગુજરાત",
    "haryana": "hr|hy|हरियाणा", "himachal pradesh": "hp|हिमाचल प्रदेश", "jharkhand": "jh|झारखंड",
    "karnataka": "ka|कर्नाटक|ಕರ್ನಾಟಕ", "kerala": "kl|kr|केरल|കേരളം", "madhya pradesh": "mp|मध्य प्रदेश",
    "maharashtra": "mh|महाराष्ट्र", "manipur": "mn|मणिपुर", "meghalaya": "ml|मेघालय", "mizoram": "mz|मिज़ोरम",
    "nagaland": "nl|नागालैंड", "odisha": "od|or|orissa|ओडिशा|उड़ीसा|ଓଡ଼ିଶା", "punjab": "pb|पंजाब|ਪੰਜਾਬ",
    "rajasthan": "rj|राजस्थान", "sikkim": "sk|सिक्किम", "tamil nadu": "tn|तमिलनाडु|तमिल नाडु|தமிழ்நாடு",
    "telangana": "ts|tg|तेलंगाना|తెలంగాణ", "tripura": "tr|त्रिपुरा", "uttar pradesh": "up|उत्तर प्रदेश",
    "uttarakhand": "uk|ut|उत्तराखंड", "west bengal": "wb|पश्चिम बंगाल|পশ্চিমবঙ্গ", "delhi": "dl|दिल्ली",
    "jammu and kashmir": "jk|जम्मू और कश्मीर", "ladakh": "ld|लद्दाख", "chandigarh": "ch|चंडीगढ़",
    "puducherry": "py|pondicherry|पुडुचेरी", "andaman and nicobar islands": "an",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "lk",
}

STATE_TABLES = {"US": {}, "India": {}}
for _entry in _US.replace("\n", "").split("|"):
    _abbr, _name = _entry.split(" ", 1)
    STATE_TABLES["US"][_abbr] = STATE_TABLES["US"][_name] = _name
for _name, _aliases in _INDIA.items():
    for _alias in [_name] + _aliases.split("|"):
        STATE_TABLES["India"][_alias] = _name

# fallback (address has no comma structure): scan the normalized text for full English names
_STATE_RES = {
    c: re.compile(r"\b(" + "|".join(sorted((k for k in t if len(k) > 2 and k.isascii()), key=len, reverse=True)) + r")\b")
    for c, t in STATE_TABLES.items()
}
_PUNCT = re.compile(r"[!-/:-@\[-`{-~।]")   # ASCII punctuation + danda; keeps Devanagari vowel signs intact


# ============================================================
# PER-RECORD ATTRIBUTES (computed once per unique entity, not per pair)
# ============================================================
def _clean_part(p):
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", p.lower().replace("&", " and "))).strip()


def _state_of_part(part, table):
    """part = one cleaned comma-separated chunk of the raw address -> canonical state or ""."""
    letters = " ".join(w for w in part.split() if not w.isdigit())   # "nc 27260" -> "nc"
    return table.get(letters, "")


def extract_geo(raw, norm, country):
    """-> (state, city, postal, house). "" means 'not found'. Heuristic, order-robust:
    the state is the rightmost comma-chunk that IS a state; the city is the chunk next
    to it (before it, else after it) that has no digits."""
    table = STATE_TABLES.get(country, {})
    parts = [_clean_part(p) for p in raw.split(",")]

    state, si = "", -1
    if table:
        for i in range(len(parts) - 1, -1, -1):
            state = _state_of_part(parts[i], table)
            if state:
                si = i
                break
        if not state:                                   # no comma structure: scan the normalized text
            m = _STATE_RES[country].findall(norm)
            if m:
                state = table[m[-1]]
            else:
                toks = norm.split()
                state = next((table[t] for t in toks[-1:] + toks[:1] if len(t) == 2 and t in table), "")

    city = ""
    if si >= 0:
        for j in (si - 1, si + 1):
            if 0 <= j < len(parts) and parts[j] and not any(ch.isdigit() for ch in parts[j]):
                city = parts[j]
                break

    nums = re.findall(r"\b\d+\b", norm)
    # PIN (India, 6 digits) / postal code (5 digits, only if a house number precedes it --
    # a lone leading 5-digit number is a US house number like "17560 Ellis Road").
    postal = next((n for n in nums if len(n) == 6), "") or next((n for n in nums[1:] if len(n) == 5), "")
    house = next((n for n in nums if n != postal and len(n) <= 5), "")
    return state, city, postal, house


def add_record_attributes(r):
    r["name"] = r["norm_name_no_suffix"].where(r["norm_name_no_suffix"] != "", r["norm_name"])
    r["name_first_tok"] = r["name"].str.split(n=1).str[0].fillna("")
    r["name_nonlatin"] = r["name"].str.contains(r"[^\x00-ɏ]", regex=True).astype(np.float32)

    geo = [extract_geo(a, b, c) for a, b, c in zip(r["business_address"].values, r["norm_address"].values, r["country"].values)]
    geo = pd.DataFrame(geo, columns=["state", "city", "postal", "house"], index=r.index)
    return r.join(geo)


# ============================================================
# PAIR-LEVEL PRIMITIVES
# ============================================================
def _obj(s):
    return s.to_numpy(dtype=object)


def _eq(x, y):
    """1.0 / 0.0 where both sides are non-empty, NaN ("unknown") otherwise."""
    out = (x == y).astype(np.float32)
    out[(x == "") | (y == "")] = np.nan
    return out


def _sim(scorer, x, y, scale=1.0):
    """Aligned pairwise similarity x[i] vs y[i], multithreaded in C++. NaN if either side empty."""
    out = cpdist(x.tolist(), y.tolist(), scorer=scorer, dtype=np.float32, workers=-1) * np.float32(scale)
    out[(x == "") | (y == "")] = np.nan
    return out


def _rowdot(X, a, b):
    return np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel().astype(np.float32)


def _cosine(X, a, b, x, y):
    out = _rowdot(X, a, b)
    out[(x == "") | (y == "")] = np.nan
    return out


def _jaccard(H, a, b):
    inter = _rowdot(H, a, b)
    size = np.asarray(H.sum(axis=1)).ravel().astype(np.float32)
    union = size[a] + size[b] - inter
    out = np.full(len(a), np.nan, dtype=np.float32)
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


def _abs_len_diff(x, y):
    lx = np.fromiter((len(s) for s in x), dtype=np.float32, count=len(x))
    ly = np.fromiter((len(s) for s in y), dtype=np.float32, count=len(y))
    return np.abs(lx - ly)


# ============================================================
# VECTORIZERS
# ============================================================
def fit_vectorizers(rec, seed=42):
    s = rec.sample(min(TFIDF_FIT_SAMPLE, len(rec)), random_state=seed)
    names = s["norm_name_no_suffix"].where(s["norm_name_no_suffix"] != "", s["norm_name"])
    print(f"Fitting TF-IDF vocabularies on {len(s)} records...")
    return {
        # names: char n-grams survive typos / word-order changes
        "name_tfidf": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True,
                                      dtype=np.float32).fit(names),
        # addresses: word-level, so IDF down-weights "street", "road", "nagar", ...
        "addr_tfidf": TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, sublinear_tf=True,
                                      dtype=np.float32).fit(s["norm_address"]),
        # Jaccard needs no vocabulary: hash tokens -> binary sets (no OOV tokens dropped)
        "tok": HashingVectorizer(token_pattern=r"\S+", binary=True, norm=None, alternate_sign=False,
                                 n_features=2 ** 20, dtype=np.float32),
        "num": HashingVectorizer(token_pattern=r"\b\d+\b", binary=True, norm=None, alternate_sign=False,
                                 n_features=2 ** 20, dtype=np.float32),
    }


# ============================================================
# FEATURES FOR A CHUNK OF PAIRS
# ============================================================
def compute_features(pairs, rec, vecs):
    ids = pd.unique(np.concatenate([pairs["source1_entity_id"].values, pairs["candidate_entity_id"].values]))
    r = add_record_attributes(rec.loc[ids].copy())
    pos = pd.Series(np.arange(len(r)), index=r.index)
    a = pos.loc[pairs["source1_entity_id"]].to_numpy()
    b = pos.loc[pairs["candidate_entity_id"]].to_numpy()

    name, addr = _obj(r["name"]), _obj(r["norm_address"])
    na, nb = name[a], name[b]
    aa, ab = addr[a], addr[b]

    Xn = vecs["name_tfidf"].transform(r["name"])
    Xa = vecs["addr_tfidf"].transform(r["norm_address"])
    Hn = vecs["tok"].transform(r["name"])
    Ha = vecs["tok"].transform(r["norm_address"])
    Hnum = vecs["num"].transform(r["norm_address"])

    f = {
        "source1_entity_id": pairs["source1_entity_id"].values,
        "candidate_entity_id": pairs["candidate_entity_id"].values,
        "cand_is_s3": pairs["candidate_entity_id"].str.startswith("S3-").to_numpy().astype(np.float32),
    }

    # ---- country (open-set: plain string equality, no hard-coded country list) ----
    country = _obj(r["country"])
    f["country_match"] = (country[a] == country[b]).astype(np.float32)

    # ---- name ----
    f["name_exact"] = _eq(na, nb)
    f["name_exact_with_suffix"] = _eq(_obj(r["norm_name"])[a], _obj(r["norm_name"])[b])
    f["name_lev"] = _sim(Levenshtein.normalized_similarity, na, nb)
    f["name_jaro_winkler"] = _sim(JaroWinkler.similarity, na, nb)
    f["name_jaccard"] = _jaccard(Hn, a, b)
    f["name_token_set"] = _sim(fuzz.token_set_ratio, na, nb, 0.01)     # "abc tech india pvt" ~ "abc tech pvt india"
    f["name_token_sort"] = _sim(fuzz.token_sort_ratio, na, nb, 0.01)
    f["name_tfidf"] = _cosine(Xn, a, b, na, nb)
    f["name_first_token_match"] = _eq(_obj(r["name_first_tok"])[a], _obj(r["name_first_tok"])[b])
    f["name_len_diff"] = _abs_len_diff(na, nb)
    # If exactly one side is non-Latin script (e.g. Devanagari vs English), the name features are
    # unreliable -> lets the model lean on address instead.
    f["name_script_mismatch"] = np.abs(r["name_nonlatin"].to_numpy()[a] - r["name_nonlatin"].to_numpy()[b])

    # ---- address ----
    f["address_exact"] = _eq(aa, ab)
    f["address_lev"] = _sim(Levenshtein.normalized_similarity, aa, ab)
    f["address_jaro_winkler"] = _sim(JaroWinkler.similarity, aa, ab)
    f["address_jaccard"] = _jaccard(Ha, a, b)
    f["address_token_set"] = _sim(fuzz.token_set_ratio, aa, ab, 0.01)  # robust to component reordering
    f["address_tfidf"] = _cosine(Xa, a, b, aa, ab)
    f["address_num_jaccard"] = _jaccard(Hnum, a, b)
    f["address_len_diff"] = _abs_len_diff(aa, ab)

    # ---- address components (NaN = missing on at least one side, i.e. unknown, not a mismatch) ----
    f["house_number_match"] = _eq(_obj(r["house"])[a], _obj(r["house"])[b])
    f["pin_match"] = _eq(_obj(r["postal"])[a], _obj(r["postal"])[b])
    f["state_match"] = _eq(_obj(r["state"])[a], _obj(r["state"])[b])
    city = _obj(r["city"])
    f["city_match"] = _eq(city[a], city[b])
    f["city_jaro_winkler"] = _sim(JaroWinkler.similarity, city[a], city[b])

    # ---- cross-field ----
    name_score = pd.DataFrame({k: f[k] for k in ("name_jaro_winkler", "name_tfidf", "name_token_set")}).mean(axis=1).to_numpy(np.float32)
    addr_score = pd.DataFrame({k: f[k] for k in ("address_jaro_winkler", "address_tfidf", "address_token_set")}).mean(axis=1).to_numpy(np.float32)
    f["name_score"], f["address_score"] = name_score, addr_score
    f["name_x_address"] = name_score * addr_score
    f["name_plus_address"] = name_score + addr_score
    f["name_plus_address_plus_country"] = name_score + addr_score + f["country_match"]
    f["both_strong"] = ((name_score > 0.8) & (addr_score > 0.8)).astype(np.float32)
    f["name_strong_addr_weak"] = ((name_score > 0.8) & (addr_score < 0.5)).astype(np.float32)
    f["name_weak_addr_strong"] = ((name_score < 0.5) & (addr_score > 0.8)).astype(np.float32)

    out = pd.DataFrame(f)
    if "label" in pairs.columns:
        out["label"] = pairs["label"].to_numpy(np.int8)
    return out


# ============================================================
# I/O + DRIVER
# ============================================================
def load_pairs(prefix, sample_size):
    suffix = "_TEST" if sample_size else ""
    if prefix == "train":
        pairs = pd.read_csv(os.path.join(PAIRS_DIR, f"train_pairs{suffix}.tsv"), sep="\t", dtype=str)
        pairs["label"] = pairs["label"].astype(np.int8)
        return pairs
    # test: no labels -- explode the blocking candidate lists
    cand = pd.read_csv(os.path.join(CAND_DIR, f"test_candidate_pairs{suffix}.tsv"), sep="\t", dtype=str)
    cand = cand[cand["candidate_entity_ids"].fillna("") != ""]
    cand = cand.assign(candidate_entity_id=cand["candidate_entity_ids"].str.split(",")).explode("candidate_entity_id")
    return cand[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)


def load_records(prefix, sample_size, needed_ids):
    frames = []
    for src in ("source1", "source2", "source3"):
        path = os.path.join(NORM_DIR, f"{prefix}_{src}.tsv")
        # keep_default_na=False: empty cells stay "" (never NaN), so every string op below is safe
        kw = dict(sep="\t", usecols=REC_COLS, dtype=str, keep_default_na=False)
        if sample_size:
            df = pd.read_csv(path, nrows=sample_size, **kw)
            frames.append(df[df["entity_id"].isin(needed_ids)])
        else:
            for chunk in pd.read_csv(path, chunksize=1_000_000, **kw):
                frames.append(chunk[chunk["entity_id"].isin(needed_ids)])
    rec = pd.concat(frames, ignore_index=True).drop_duplicates("entity_id").set_index("entity_id")
    print(f"Loaded {len(rec)} records")
    return rec


def build_features(prefix="train", sample_size=None, chunk_size=CHUNK_SIZE):
    t0 = time.time()
    tag = f"{prefix}{'_TEST' if sample_size else ''}"
    print(f"\n{'=' * 60}\n  FEATURE ENGINEERING — {tag}\n{'=' * 60}")

    pairs = load_pairs(prefix, sample_size)
    print(f"Pairs: {len(pairs)}")
    needed = set(pairs["source1_entity_id"]) | set(pairs["candidate_entity_id"])
    rec = load_records(prefix, sample_size, needed)
    del needed
    gc.collect()

    known = pairs["source1_entity_id"].isin(rec.index) & pairs["candidate_entity_id"].isin(rec.index)
    if not known.all():
        print(f"WARNING: dropping {(~known).sum()} pairs whose records were not found")
        pairs = pairs[known].reset_index(drop=True)

    vecs = fit_vectorizers(rec)

    out_dir = os.path.join(FEATURES_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):               # this script's own previous parts
        if old.startswith("part-"):
            os.remove(os.path.join(out_dir, old))

    n_parts = 0
    for start in range(0, len(pairs), chunk_size):
        feats = compute_features(pairs.iloc[start:start + chunk_size], rec, vecs)
        feats.to_parquet(os.path.join(out_dir, f"part-{n_parts:04d}.parquet"), index=False)
        n_parts += 1
        print(f"  chunk {n_parts}: {start + len(feats)}/{len(pairs)} pairs ({time.time() - t0:.0f}s)")
        del feats
        gc.collect()

    print(f"\nSaved {n_parts} part(s) to {out_dir}  (read back with pd.read_parquet('{out_dir}'))")
    return out_dir


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        build_features("train", sample_size=5000)
    elif mode == "full":
        build_features("train", sample_size=None)
        build_features("test", sample_size=None)
    else:
        print("Usage: python build_features.py [smoke|full]")
