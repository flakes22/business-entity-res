# Business Entity Resolution: reproducible pipeline

Matches every Source-1 business record to its Source-2 / Source-3 records
(ML Challenge 2026). The pipeline uses only the provided data: no external
databases, APIs, geocoders or pretrained language models. The only models are
two small LightGBM gradient-boosted tree ensembles (MIT licence).

## 1. Environment

* Python 3.11 (64-bit), tested on Windows 11. Pure Python plus the wheels in `requirements.txt`.
* Hardware used: 14 CPU cores, 16 GB RAM, no GPU. Peak RAM is about 11 GB, and the full
  run takes about 3.5 h. About 12 GB of free disk is needed for intermediate files.

```bash
pip install -r requirements.txt
```

## 2. Data layout

By default the code expects the challenge folder next to this one:

```
<root>/
├── business_entity_resolution/          # this folder
│   ├── src/ ...
└── 6ab10eb3b23ba_student_resource/student_resource/
    ├── dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
    ├── dataset/test/test_source{1,2,3}.tsv
    └── utils/validate_submission.py
```

Other locations can be set with environment variables:

| variable | meaning | default |
| --- | --- | --- |
| `BER_DATA_DIR` | folder containing `train/` and `test/` | `../6ab10eb3b23ba_student_resource/student_resource/dataset` |
| `BER_WORK_DIR` | intermediate files | `../work` |
| `BER_OUTPUT_DIR` | the two submission TSVs | `../output` |
| `BER_N_JOBS` | worker threads | CPU count - 2 |

## 3. Run end to end

```bash
cd src
python run_all.py
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs
the official validator on them. Every step skips work whose output already exists, so
an interrupted run can simply be restarted. You can also run the steps one by one:

| # | command | what it does | output |
| --- | --- | --- | --- |
| 1 | `python prepare.py all` | normalise names/addresses of the 6 source files (multiprocess) | `work/norm/*.parquet` |
| 2 | `python blocking.py train` then `python blocking.py test` | candidate generation S1→S2 and S1→S3 | `work/cand/*.parquet` |
| 3 | `python train_stage1.py all` | pair features for the training subsets, two stage-1 LightGBMs (folds A/B) | `work/feats/`, `work/models/stage1_*` |
| 4 | `python predict_stage1.py train` | stage-1 probabilities for the train pairs stage 2 needs | `work/p1/train_*` |
| 5 | `python stage2.py fit` | stage-2 v1 (baseline; also provides the shared context code) | `work/models/stage2*` |
| 6 | `python orphan_sim.py build` | test-like simulation on train: drop 19% of the unused S1 records so their matches become orphans | `work/sim_table_*.parquet` |
| 7 | `python stage2v2.py fit` | **final stage 2**: LightGBM with difference features, trained and decision-rule-tuned on the simulation | `work/models/stage2v2_*` |
| 8 | `python predict_stage1.py test` then `python stage2v2.py score test` | stage-1 and stage-2 v2 probabilities for every test candidate pair | `work/scored_test_v2.parquet` |
| 9 | `BER_SCORED=../work/scored_test_v2.parquet BER_DECISION=../work/models/stage2v2_decision.json python make_submission.py` | writes both TSVs and runs the validator (`run_all.py` sets these variables) | `output/` |

Earlier leaderboard iterations are kept for reference: `label_shift.py` (0.959), and `segfeats.py` + `label_shift_seg.py` (0.960) for EM prior correction of stage-2 v1. `make_submission_v2.py` writes only the matching file from the v2 scores. `pad_seg.py` is a diagnostic for zero-padded house numbers.

Helper scripts: `eval_blocking.py` reports blocking recall against the train ground truth,
and `analyze_prune.py` was used to choose the candidate-pruning rule.

## 4. Source files

| file | role |
| --- | --- |
| `config.py` | paths and settings |
| `textnorm.py` | normalisation: Indic-script romanisation, accent stripping, legal-suffix / honorific removal, domain-name handling, address abbreviation canonicalisation, state gazetteers (used only for matching country labels), house / postal / state / city extraction, consonant skeletons |
| `prepare.py` | step 1 |
| `blocking.py` | step 2: multi-channel IDF-weighted sparse top-K retrieval (forward + reverse) and pruning |
| `ctx.py` | numpy group statistics (rank / gap within an S1 record and within a target record) |
| `features.py` | rapidfuzz string-similarity pair features |
| `build_feats.py` | streams candidate pairs through context and string features |
| `train_stage1.py`, `predict_stage1.py` | stage-1 matcher |
| `stage2.py` | stage-2 re-scorer, macro-F0.5 evaluation, decision rule, test scoring |
| `label_shift.py` | test-time prior correction (Saerens et al., 2002) |
| `orphan_sim.py` | test-like orphan simulation on train (final stage-2 training data) |
| `difffeats.py` | stage-2 v2 "difference" features: missing/extra name words (+ cross-fitted log-odds), legal-form agreement, house-number change type |
| `stage2v2.py` | final stage-2 model: fit on the simulation, score test |
| `decide.py` | challenge metric and decision rules |
| `make_submission.py` | TSV writer and validator call |
| `run_all.py` | orchestration |

Randomness (the data split and LightGBM bagging) is seeded with `SEED = 42`, so reruns give
the same output.
