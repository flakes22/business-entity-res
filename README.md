# Business Entity Resolution (Amazon ML Challenge 2026)

Matches each Source-1 business record to its records in Source 2 and Source 3 (noisy names and
addresses; US, India, and France unseen in training), scored by macro F0.5.

**Result:** public leaderboard macro F0.5 **0.973**.

## Approach (short)

1. **Normalisation**: Indic-script romanisation, accent stripping, legal-suffix / honorific
   removal, address abbreviation canonicalisation, consonant skeletons.
2. **Blocking**: IDF-weighted sparse top-K retrieval over 7 token channels (name words,
   skeletons, char 4-grams, name-word pairs, address tokens, number×locality and
   name×locality composites), forward (S1 → target) and reverse (target → S1). About 20
   candidates per record per source; recall ceiling 97.1%.
3. **Stage 1**: LightGBM on ~60 string-similarity and blocking-context features.
4. **Stage 2 (final)**: LightGBM re-scoring with probability-context features (competition
   between S1 records for the same target) and "difference" features. It is trained on a
   test-like simulation in which the extra unmatched look-alike records that test contains
   are reproduced on train.
5. **Decision**: per-entity expected-F0.5-optimal match set, plus one owner per target record.

Full write-up: [Documentation.md](Documentation.md). EDA: [eda.ipynb](eda.ipynb).

## Reproduce

See [business_entity_resolution/README.md](business_entity_resolution/README.md).

```bash
cd business_entity_resolution
pip install -r requirements.txt
cd src && python run_all.py
```

The challenge dataset is not included. Place it as described in the pipeline README, or set
`BER_DATA_DIR`.

## Team

Anagha Prajapati, Sarah Roomie, Mahek Desai
