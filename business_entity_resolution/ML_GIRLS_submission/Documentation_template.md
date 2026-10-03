# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** ML GIRLS 
**Team Members:** Anagha Prajapati, Sarah Roomi, Mahek Desai
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We use a two-stage pipeline: retrieval, then learned matching. Candidates come from
IDF-weighted sparse retrieval over seven token channels: name words, consonant
skeletons, character 4-grams, name-word pairs, address tokens, number×locality
composites and name×locality composites. The retrieval runs in both directions, S1→target
and target→S1. It keeps about 20 candidates per S1 record per source and reaches a 97.1%
recall ceiling on the full training set. Two LightGBM models then score the pairs. Stage 1
uses about 60 string and blocking features. Stage 2 re-scores each pair using how its
probability compares with the other candidates of the same S1 record and with the other S1
records competing for the same S2/S3 record (each S2/S3 record belongs to at most one S1
entity). The final stage 2 is trained on a test-like "orphan" simulation of the training
data, which reproduces the test set's larger pool of unmatched look-alike records. It adds
"difference" features that describe how two records differ: a distinctive word substituted
vs a generic word added, house-number digits replaced vs dropped. The decision rule is tuned
directly for macro-F0.5 on held-out S1 entities. **Public leaderboard: 0.973.**
**Held-out macro-F0.5: 0.9719 for stage 2 v1 on 100,000 held-out train S1 entities; 0.9801 for the final stage 2 v2 on the simulated held-out set. Public leaderboard (final): 0.973.**

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the full data (train: 2.21M S1, 5.03M S2 and 5.29M S3 records; test: 1.73M S1,
4.89M S2 and 5.08M S3 records, including 259k French S1 records) showed:

* **Match structure.** 5.6% of S1 entities are singletons. The others have 1–11 matches
  (mode 3), split across S2 and S3. In the training ground truth, every S2/S3 record is
  matched to at most one S1 entity. This is the key structural fact we exploit.
* **Heavy look-alikes.** S1 is deduplicated, but 30% of S1 names are exact duplicates of
  another S1 name ("First Church", "Classic Industries Pvt Ltd") and 3.5% of addresses are
  shared. Name similarity alone is therefore not enough: most hard negatives share the name
  or the building.
* **Name noise.** Typos and letter swaps, accents added ("Fóundation"), legal-suffix
  changes (Pvt/Private/Ltd/Limited/LLC/SARL…), injected words ("Services", "Center",
  "Partners"), honorifics (Mr/Shri/Sri), duplicated tokens, shuffled word order, bracketed
  IDs ("(ID: 43446)"), and names written as web domains ("leehahneiron.com", "#physicaltherapy").
  Some records use a trade name ("Zetalum t/a Super 2 Hair Studio") or a completely
  unrelated brand name ("Avinovi"), in which case only the address links them.
* **Scripts.** 23% of Indian S2 names and 13% of Indian S3 names are written in Indic
  scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Gurmukhi, Oriya). S1 names
  are always Latin.
* **Address noise.** Abbreviations (Rd/Road, St/Street, R./Rue, AV/Avenue), state names vs
  codes vs native-script names (TN / Tamil Nadu / தமிழ்நாடு), unit designators ("Shop No 9",
  "#9", "Door No 9", "Unit 9"), perturbed house numbers (9184→9186, 3153→153), comma-chunk
  reordering, missing components. About 3% of S2/S3 addresses are empty.
* **France (test only).** French legal forms (SARL, SAS, EURL…), street types (Rue, Avenue,
  Allée, Impasse, and abbreviations R./AV/BD), and accented text. No training labels exist for
  France, so every feature is country-agnostic.

### 2.2 Solution Strategy

**Approach Type:** Blocking (multi-channel sparse retrieval, forward + reverse) → two-stage
gradient-boosted classifier → decision rule tuned for macro-F0.5.

**Core Innovation:**
1. **Composite blocking tokens** (number×locality, name×locality, name-word pairs). These
   keep combinations of individually common tokens that frequency caps would otherwise
   discard. Together with rebalanced channel weights, this raised retrieval recall from
   about 90% to 97%.
2. **Reverse retrieval** (target→S1). This finds matches that look-alikes push out of an S1
   record's own top-K list.
3. **Competition-aware re-scoring** (stage 2), which exploits the one-owner-per-record
   structure.

---

## 3. Candidate Generation (Blocking)

**Normalisation** (`textnorm.py`, rule-based, no external data):
* Indic scripts are romanised with one table shared across the ISCII-parallel Unicode blocks
  (handles inherent vowels, virama, vowel signs and schwa deletion). For example
  "जैन हॉस्पिटैलिटी" becomes "jain hospitailitii".
* Latin diacritics are stripped, "&" becomes "and", and dotted acronyms are joined
  ("E.U.R.L." becomes "eurl").
* Legal suffixes and honorifics are removed to form the "core" name. Duplicate tokens are
  removed. Web domains and hashtags are detected and reduced to their stem. DBA / "t/a" parts
  are extracted.
* **Consonant skeleton**: vowels removed, voicing and aspiration merged. Romanised Hindi and
  English then agree, for example hospitality → *hsptlt* and hospitailitii → *hsptlt*.
* Address abbreviations (US, Indian, French) are mapped to one canonical token, and unit
  designators are dropped. State names, codes and native-script names are mapped to one code,
  using the US / India gazetteers only when the country label matches; other labels such as
  France simply skip this. Digit/letter runs are split, and house number, postal code, state
  and city are extracted.

**Blocking keys / channels** (per country label, per target source). Each record is a bag of
prefixed tokens:

| channel | tokens | weight |
| --- | --- | --- |
| n | core-name words | 0.45 |
| k | consonant skeletons of name words | 0.25 |
| g | character 4-grams of the space-free core name (typos, domains) | 0.30 |
| p | sorted pairs of name words (word order) | 0.30 |
| a | canonical address tokens | 0.80 |
| x | number × address-word composites ("36\|tirupur") | 0.60 |
| m | name-word × address-word composites ("classic\|tirupur") | 0.50 |

Token weight is IDF. Each channel is L2-normalised separately and then weighted, so the score
is a weighted sum of per-channel cosines. Tokens with target-side document frequency above
0.2% of the pool are left out of the sparse product (but kept in the normalisation). Top-K
search uses `sparse_dot_topn` (multithreaded C++):

* forward searches per S1 record: name channels (top 10), address channels (top 12), all
  channels (top 20);
* a reverse search per target record: all channels, top 5 S1 records.

The union (about 30 pairs per S1 per source) is pruned by a rank rule: keep a pair if its
combined-score rank is below 15, its address rank below 5, its name rank below 5, its
name×address rank below 3, or it is among the target's 3 best S1 records. Exact per-channel
cosines of every kept pair become model features.

* **Candidate pairs generated:** 85.1M on train (19.2 per S1 per source) and **69.3M on test**
  (about 40 per S1 in total).
* **Recall ceiling (full train, all 2.2M S1):** 97.1% of the 7.64M true pairs (S2 97.09%,
  S3 97.07%). The pruned forward searches alone give 95.4%, and the reverse search alone 96.1%.
* **How we ensured true matches were not lost:** we analysed the missed pairs directly. They
  were not dissimilar (median name token-set similarity 0.90, address 0.97). Instead, they
  were crowded out by look-alikes, and their shared tokens were individually too common to
  survive the frequency cap. The composite channels, balanced name and address weights, and
  reverse retrieval target exactly this failure mode (recall 90.0% → 94.1% → 97.1% as each
  was added). The pruning rule was chosen on a 100k-S1 sample to lose under 0.3 points of
  recall while cutting the candidate count by about 35%.

---

## 4. Matching Model

**Features used (63 in stage 1):**
* *Name:* rapidfuzz ratio, partial ratio, token-set, token-sort, WRatio, Jaro-Winkler on the
  core name; ratio and partial ratio on the space-free name (domains); ratio and token-set on
  the consonant skeleton (transliteration); token-set on the full name incl. legal words;
  DBA-part similarity; first-token equality and Jaro-Winkler; token counts and length
  difference; target is a domain name; target name is in an Indic script.
* *Address:* ratio, token-set, token-sort and partial token-set on canonical addresses;
  number-set token-set/sort and equality; house-number equality and Levenshtein; postal
  code, state and city agreement (NaN when a side is missing, so "unknown" is never read as
  a mismatch); empty-address flag; token counts.
* *Blocking and context:* the 7 channel cosines, combined score, search hits, rank within
  the S1 record's list (overall, by address, by name) and the gap to the best candidate and
  runner-up; reverse rank, gap and number of S1 records competing for the same target; target
  source.
* *Stage 2 adds 16 probability-context features:* p1; its rank, gap to best and runner-up,
  sum and count above 0.5 within the S1 record (per source and overall); and for the
  target record, this S1's rank among all claimants, its margin over the strongest other
  claimant, the claimant sum/count and the number of claimants.

The country label is never a feature (plain string equality is used only to partition
blocking), so the model applies unchanged to France.

**Model type:** LightGBM binary classifiers (MIT licence, well under the 8B-parameter limit).
* Train S1 records are split by entity: A (125k) / B (125k) / VAL_ES (30k, early stopping) /
  VAL_EVAL (100k, never used for fitting).
* Stage 1: two models (lr 0.1, 127 leaves, early stopping at about 700 trees; validation
  log-loss 0.0099). Each predicts the other fold out-of-fold. Held-out and test pairs get the
  average of the two.
* Stage 1 scores every candidate pair (all 69.3M test pairs). The probability-context
  features are computed over the whole candidate set.
* Stage 2: LightGBM on stage-1 features plus context, trained on A∪B pairs (with out-of-fold
  p1) that have p1 ≥ 0.02. Pairs below 0.02 are never selected.

**Threshold selection method:** direct macro-F0.5 optimisation on VAL_EVAL. The per-S1
F-score denominator uses the full ground truth, so pairs lost at blocking still count as
misses, and singletons are included. We compared a global threshold on p1 or p2, a
threshold plus exclusivity (keep a pair only if its S1 is the target's top claimant), and a
per-entity expected-F0.5 subset rule. Results on VAL_EVAL (100k S1):

| decision rule | macro F0.5 |
| --- | --- |
| stage 1, best global threshold (p1 >= 0.72) | 0.9681 |
| stage 1 + exclusivity (p1 >= 0.70) | 0.9682 |
| stage 2, best global threshold (p2 >= 0.64) | 0.9713 |
| stage 2 + exclusivity (p2 >= 0.64) | 0.9714 |
| **stage 2 + per-entity expected-F0.5 subset (chosen)** | **0.9719** |

The chosen rule sorts one S1 record's candidates by p2 and picks the top-j set that
maximises the approximate expected F0.5, 1.25·Σ_{i<=j} p_i / (0.25·E[n_true] + j), where
E[n_true] = Σp / (1 − 0.02). It compares that against predicting nothing, whose expected
score is P(no match) = Π(1 − p_i), scaled by 1.2. Only pairs with p2 >= 0.4 are considered.
This lets the effective threshold adapt per entity, and it handles singletons explicitly.
To estimate the effect of tuning honestly, the rule was selected on one half of VAL_EVAL
and scored on the other half (half-1 tuned rule scored 0.97187 on the unseen half 2, the same as on half 1, so the tuning is not overfitted).

**Final model: stage 2 v2, trained on a test-like simulation (public leaderboard 0.973).**
The label-shift corrections only rescaled probabilities. They could not teach the model to
recognise the test decoys, and they reached 0.960. The final submission instead changes the
training data and the features of stage 2:

* **Orphan simulation (`orphan_sim.py`).** Train has 1.22 unmatched ("orphan") S2/S3 records
  per S1 record; test has about 2.30 (test pools are ~24% larger per S1 in every country). On
  held-out train entities, about 80% of stage-2 false positives are orphans, so test is
  materially harder. We drop a random 19% of the train S1 records that are in no model or
  validation split. Their true matches become orphans, which reproduces the test orphan rate.
  The candidate table loses the dropped records' pairs, and the reverse / competition context
  is recomputed. No test data is used.
* **Difference features (`difffeats.py`, 17 features).** Hard negatives are look-alikes made
  from a true record by substituting a *distinctive* name word (Construction → Distribution)
  or the legal form (Private Limited → LLP), and by *replacing* house-number digits
  (22745 → 22758). Noise on true matches instead *adds or drops generic* words (Services,
  Center, Partners) and *drops or splits* digits (1022 → 022, 106 → 1-06). Edit distances
  score both kinds of change alike. We therefore add:
  - name words missing from / extra in the target (fuzzy token alignment), with counts and a
    cross-fitted log-odds of how typical each word is as a missing / extra word in true
    matches;
  - legal-form agreement;
  - house-number change type: equal digits, substring (dropped digits), number of replaced
    digits, log numeric distance;
  - extra / missing numbers in the address.

  No rule is country-specific.
* **Stage 2 v2** is a LightGBM with 96 features (79 stage-1 + context features, plus the 17
  difference features), 3,446 trees. It is trained on simulated folds A+B, early-stopped on
  simulated VAL_ES, and its decision rule is tuned on simulated VAL_EVAL: expected-F0.5 subset,
  miss rate 0.02, α 1.5, p ≥ 0.4, followed by exclusivity (a record claimed by two S1
  entities keeps only the higher-probability claimant).

| simulated held-out VAL_EVAL (100k S1) | macro F0.5 |
| --- | --- |
| stage 2 v1 context features, retrained on the simulation | 0.9710 |
| **+ difference features (final)** | **0.9801** (honest half-split 0.9802) |

**Public leaderboard progression:** stage 2 v1 0.958 → + global prior correction 0.959 →
+ per-(country, segment) prior correction 0.960 → **stage 2 v2 (simulation + difference
features) 0.973 (final submission)**.

**Earlier iteration: test-time label-shift correction** (superseded by stage 2 v2). The test pools hold about 23% more
S2/S3 records per S1 entity than train (5.75 vs 4.68). Inspection shows that the extra records
are mostly near-duplicate decoys: same name, neighbouring house or unit number. Stage 2 is well
calibrated on held-out train entities, both overall (mean p2 0.8674 vs true rate 0.8677) and
within every address segment (within ±0.002). On test, however, the share of true matches among
plausible pairs is lower, and the shift is very uneven. We estimate each test group's prior with
the EM procedure of Saerens et al. (2002), without using any test labels, separately for each
(country label × address segment). The segments are: house numbers equal, house numbers
differ, house number unknown, and target address empty. Each probability's odds are then
rescaled by (π_test/π_train)/((1−π_test)/(1−π_train)):

| segment (train prior) | US | India | France |
| --- | --- | --- | --- |
| house numbers equal (0.974) | 0.983 | 0.935 | 0.927 |
| house numbers differ (0.705) | 0.488 | 0.696 | **0.181** |
| house number unknown (0.906) | 0.965 | 0.741 | 0.726 |
| target address empty (0.393) | 0.501 | 0.381 | 0.438 |

The unchanged, validation-tuned decision rule is applied to the corrected probabilities. A
record claimed by two S1 entities is then kept only for the higher-probability claimant
(exclusivity). If there were no real shift, this correction would cost at most 0.0013 on
validation.


**Known limitation found late.** House numbers are compared as strings, so zero-padded numbers
("25" vs "025") count as "different". This puts some genuine matches into the "house numbers
differ" segment. The fix (strip leading zeros in `textnorm.py`) was identified but not applied
to the final submission; it is the next improvement.

---

## 5. Results & Error Analysis

- **F_0.5 score (macro, held-out train S1 entities):** 0.9719 (100,000 held-out train S1 entities, singletons included)
- **Breakdown (VAL_EVAL, chosen rule):** 346,288 true pairs. 10,617 (3.1%) are lost at
blocking, 10,767 (3.1%) are candidates the model rejects, and 324,904 are found. 2,020
predictions are false. Pair-level precision is **99.38%** and recall **93.82%**.
Singletons (5.6% of entities) score F 0.962 and entities with matches score 0.973.
By country: US 0.977, India 0.964.
- **Model diagnostics:** stage-1 validation log-loss 0.0099 (about 700 trees per fold). Stage 2
  stopped at 1,777 trees (lr 0.05) on the harder p1 >= 0.02 population. Its top feature by
  gain is `p1_t_gap`, the margin over the strongest competing S1 claimant, which confirms the
  value of the one-owner structure. In stage 1 the reverse-competition features `rev_gap2`
  and `rev_rank` rank first and second, ahead of every string similarity.
- **Test output (final, stage 2 v2):** 69.3M candidate pairs for all 1,732,544 test S1
  entities and 5,741,139 matches. 1,630,365 entities (94.1%) have at least one match, with
  3.52 matches per matched entity. Public leaderboard: **0.973**. In the first version, the
  share of entities with no match (France 4.9%, India 6.1%, US 5.7%) and the matches per entity
  (3.3–3.5) were consistent across countries and with train (5.6% singletons). France was
  never seen in training and behaves like the other countries. The official validator
  (`--check-ids`) returns PASS.
- **Common false positives (wrong merges):** Near-duplicate businesses that differ only in a unit, plot or house number at the same
building ("Tirumalagiri Producer, Plot No S1" vs "Plot No S2"; "2201 Romig Pl" vs "220 Romig Pl").
Also target records with an empty address whose generic name ("Sois LLC Enterprises") fits
two S1 entities, where the ground-truth owner is the other one.
- **Common false negatives (missed matches):** Target records with an empty or truncated address plus a noisy or partial name ("Action &
Limited 5ons", no address). These are cases where predicting the link would cost more
precision than it gains. Some are lost at blocking: Indic-script names with only a city in
the address, or brand-only names ("Avinovi") at addresses shared by many businesses.

---

## 6. Conclusion

Most of the score comes from retrieving the right candidates and resolving competition.
The misses were caused by look-alike crowding, not dissimilarity, and composite tokens plus
reverse retrieval raised the recall ceiling from 90% to 97%. The one-owner-per-record
structure, used as features rather than as a hard rule, is the strongest signal in both
stages. Optimising the decision per entity for F0.5 yields a held-out macro-F0.5 of 0.972
with 99.4% pair precision, from two small LightGBM models and no external data.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: all source is in `src/`, and `README.md` gives exact
commands. `python src/run_all.py` reproduces both output files from the raw TSVs: it
normalises, blocks, builds features, trains and scores both stages, tunes the decision rule,
and writes and validates the files. See the README for the file-by-file map and the
per-step commands.

### B. Additional Results

Blocking recall progression (100k S1 sample → full train):

| blocking version | recall ceiling | candidates per S1 per source |
| --- | --- | --- |
| combined name+address TF-IDF, top 40 | 90.0% | 40 |
| + separate name / address / combined searches | 94.1% | 53 |
| + balanced weights, composite tokens (n,k,g,p,a,x,m) | 96.2% (combined search alone) | 68 |
| + reverse search and rank-rule pruning (full train, 2.2M S1) | **97.1%** | **19.2** |

Runtime on a 14-core / 16 GB laptop: normalisation 10 min; blocking about 60 min for
train and 55 min for test; stage-1 features and training 25 min; scoring all train and
test pairs about 2 h; stage 2 and submission 25 min.
